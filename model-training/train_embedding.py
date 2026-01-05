import json
import numpy as np
import os
import random
import warnings
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    AutoConfig
)

"""
Dataset format:
Synonym: ["neşeli", "sevinçli"]
Antonym: ["üzgün", "mutsuz"]
Co-hyponym: ["huzurlu", "keyifli"] 

{
  "query": "mutlu",
  "positives": [
    "neşeli",
    "sevinçli"
  ],
  "hard_negatives": [
    "üzgün",
    "mutsuz",
    "huzurlu",
    "keyifli"
  ]
}
"""

# ======================
# Model Architecture
# ======================

class ContrastiveModel(PreTrainedModel):
    _supports_sdpa = False
    """
    Custom model that produces semantic vectors for contrastive learning.
    It is based on AutoModel and adds a mean pooling layer on top.
    To be compatible with sentence-transformers losses, the forward method
    returns a dict in the format: {'sentence_embedding': embeddings}.
    """
    def __init__(self, config, model_name: str):
        super().__init__(config)
        self.transformer = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager"
        )

    def mean_pooling(self, model_output, attention_mask):
        """
        Creates a sentence embedding by mean pooling token embeddings.
        Padding tokens are excluded from the average.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, features: Dict[str, torch.Tensor]):
        """
        Forward pass of the model. Takes a features dictionary as input,
        runs it through the transformer, applies pooling, and returns the
        sentence embedding. The output format matches what sentence-transformers
        losses expect.
        """
        model_output = self.transformer(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"]
        )
        embeddings = self.mean_pooling(model_output, features["attention_mask"])
        return {'sentence_embedding': embeddings}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Forwards the gradient checkpointing enable call to the underlying
        transformer model.
        """
        self.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )


# ======================
# Trainer and Loss
# ======================

class CachedLossTrainer(Trainer):
    """
    Custom Trainer for contrastive learning using `CachedMultipleNegativesRankingLoss`.
    This loss expands the negative pool significantly by using cached embeddings
    from previous batches in addition to in-batch negatives.
    """
    def __init__(self, *args, temperature=0.07, **kwargs):
        super().__init__(*args, **kwargs)
        st_model = SentenceTransformer(modules=[self.model])

        self.loss_fct = CachedMultipleNegativesRankingLoss(
            model=st_model,
            scale=1.0 / temperature  # scale corresponds to 1/temperature
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes loss using CachedMultipleNegativesRankingLoss.
        Passes query and positive embeddings (prepared by the data collator)
        as two separate inputs to the loss function.
        """
        # Inputs are prepared by the data collator in two groups using
        # the 'query_' and 'positive_' prefixes.
        query_inputs = {
            "input_ids": inputs["query_input_ids"],
            "attention_mask": inputs["query_attention_mask"]
        }
        positive_inputs = {
            "input_ids": inputs["positive_input_ids"],
            "attention_mask": inputs["positive_attention_mask"]
        }

        # sentence_features is the format expected by the loss function:
        # [dict(anchor_features), dict(positive_features)]
        sentence_features = [query_inputs, positive_inputs]

        # Compute loss by calling the loss function
        loss = self.loss_fct(sentence_features, labels=None)

        # Return outputs in the format expected by Hugging Face Trainer.
        # If return_outputs=True, return the model outputs alongside the loss.
        return (loss, {"loss": loss}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Defines how predictions are produced during evaluation.
        Returns two embedding sets as a tuple.
        """
        # Split inputs (hard negatives are not used during prediction)
        inputs1 = {"input_ids": inputs["query_input_ids"], "attention_mask": inputs["query_attention_mask"]}
        inputs2 = {"input_ids": inputs["positive_input_ids"], "attention_mask": inputs["positive_attention_mask"]}

        # Run the model separately for each input set
        with torch.no_grad():
            # The model forward expects a single 'features' dictionary,
            # so we pass inputs via that keyword argument.
            embeddings_a = model(features=inputs1)['sentence_embedding']
            embeddings_b = model(features=inputs2)['sentence_embedding']

        # Create dummy labels so Trainer will call compute_metrics.
        # This prevents Trainer from skipping metrics when labels are absent.
        dummy_labels = torch.arange(embeddings_a.size(0), device=embeddings_a.device)

        # Return predictions in the format expected by compute_metrics
        return (None, (embeddings_a, embeddings_b), dummy_labels)


# ======================
# Data Processing
# ======================

def load_contrastive_data(path: str) -> List[Dict[str, Any]]:
    """
    Loads JSONL data in the specified format.
    Cached loss does not require hard negatives, so we only create
    (query, positive) pairs.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            # Create one sample per positive example
            for positive in data["positives"]:
                samples.append({
                    "query": query,
                    "positive": positive,
                })
    return samples


def save_samples_to_jsonl(samples, path):
    """Save the provided list of samples in JSONL format."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def shuffle_and_split_data(samples, split_ratio=0.99, seed=42):
    """Shuffle data and split into train/validation sets."""
    random.seed(seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * split_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    return train_samples, val_samples

def prepare_datasets(train_samples, val_samples):
    """Convert to Hugging Face Dataset objects."""
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    print(f"✅ Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"✅ Example train sample: {train_samples[0] if train_samples else 'None'}")
    print(f"✅ Example validation sample: {val_samples[0] if val_samples else 'None'}")

    return train_dataset, val_dataset

class ContrastiveDataCollator:
    """
    Takes (query, positive) pairs, tokenizes them, and builds batches
    in the format expected by the Trainer.
    Compatible with CachedLossTrainer.
    """
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [f["query"] for f in features]
        positives = [f["positive"] for f in features]

        tokenized_queries = self.tokenizer(
            queries, padding="longest", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        tokenized_positives = self.tokenizer(
            positives, padding="longest", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "query_input_ids": tokenized_queries["input_ids"],
            "query_attention_mask": tokenized_queries["attention_mask"],
            "positive_input_ids": tokenized_positives["input_ids"],
            "positive_attention_mask": tokenized_positives["attention_mask"],
        }

# ======================
# Environment and Helper Functions
# ======================

def check_environment():
    """Checks environment variables and logs in to wandb."""
    warnings.filterwarnings("ignore")
    wandb_key = os.getenv("WANDB_API_KEY")
    try:
        wandb.login(key=wandb_key)
        print("✅ WANDB login successful.")
    except Exception as e:
        print(f"⚠️ WANDB login error: {e}")
        print("⚠️ Continuing without WANDB...")
        os.environ["WANDB_MODE"] = "disabled"


def create_training_arguments(output_dir):
    """Create training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=50,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="retrieval_accuracy",
        greater_is_better=True,
        report_to="wandb",
        bf16=True,
        seed=42,
        gradient_checkpointing=True,
        remove_unused_columns=False
    )

def compute_metrics(eval_pred):
    """
    Metrics to measure synonym embedding quality:

    1. positive_similarity: Mean cosine similarity between synonym pairs.
       Should be high (>0.7 ideal, >0.85 excellent).

    2. negative_similarity: Mean similarity between different synonym pairs within the
       same batch. Should be low (<0.5 ideal).

    3. synonym_margin: Difference between positive and negative similarities.
       Should be high (>0.3 ideal, >0.5 excellent).

    4. retrieval_accuracy: Given a query, checks whether the correct synonym
       achieves the highest score among all other words in the batch.
       Should be close to 100%.

    5. mrr (Mean Reciprocal Rank): Average rank of the correct synonym.
       Should be close to 1.0 (1.0 = always ranked first).
    """
    embeddings_a, embeddings_b = eval_pred.predictions

    # Convert numpy arrays to torch tensors and normalize
    embeddings_a = F.normalize(torch.from_numpy(embeddings_a), p=2, dim=1)
    embeddings_b = F.normalize(torch.from_numpy(embeddings_b), p=2, dim=1)

    batch_size = embeddings_a.shape[0]

    # ============================================================
    # METRIC 1: Positive Similarity (Synonym Pair Similarity)
    # ============================================================
    # Similarity between each query and its own synonym (diagonal)
    positive_similarities = (embeddings_a * embeddings_b).sum(dim=1)
    positive_sim_mean = positive_similarities.mean().item()
    positive_sim_std = positive_similarities.std().item()

    # ============================================================
    # METRIC 2: Negative Similarity (Similarity Between Different Pairs)
    # ============================================================
    # Compare all queries with all positives
    all_similarities = torch.mm(embeddings_a, embeddings_b.t())  # [batch_size, batch_size]

    # Off-diagonal elements = similarities between different synonym pairs
    # These should be low (negatives)
    mask = torch.eye(batch_size, dtype=torch.bool, device=all_similarities.device)
    negative_similarities = all_similarities[~mask]
    negative_sim_mean = negative_similarities.mean().item()
    negative_sim_std = negative_similarities.std().item()

    # ============================================================
    # METRIC 3: Synonym Margin (Separation Power)
    # ============================================================
    # Difference between positive and negative similarities
    # Higher margin = better separation of synonyms from non-synonyms
    synonym_margin = positive_sim_mean - negative_sim_mean

    # ============================================================
    # METRIC 4: Retrieval Accuracy (Top-1 Accuracy)
    # ============================================================
    # Find the most similar positive for each query
    # The correct match should be on the diagonal (its own synonym)
    top1_values, top1_indices = all_similarities.max(dim=1)
    correct_labels = torch.arange(batch_size, device=all_similarities.device)
    retrieval_accuracy = (top1_indices == correct_labels).float().mean().item()

    # Debug: compare diagonal values vs max values
    diagonal_values = all_similarities.diag()
    # How close is the diagonal value to the maximum value?
    accuracy_gap = (top1_values - diagonal_values).mean().item()

    # ============================================================
    # METRIC 5: Mean Reciprocal Rank (MRR)
    # ============================================================
    # Reciprocal of the rank position of the correct synonym
    # Ranking: descending (most similar = rank 1)
    sorted_indices = torch.argsort(all_similarities, dim=1, descending=True)

    # For each row, find the rank position of the correct answer (diagonal)
    ranks = []
    for i in range(batch_size):
        # For the i-th query, the correct answer is the i-th positive (diagonal)
        rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1  # +1 because ranks start at 1
        ranks.append(1.0 / rank)

    mrr = np.mean(ranks)

    # ============================================================
    # Additional Statistics: Distribution Analysis
    # ============================================================
    # Fraction of positives that are very low (< 0.5)
    weak_positives_ratio = (positive_similarities < 0.5).float().mean().item()

    # Fraction of negatives that are very high (> 0.7) - undesired
    false_positives_ratio = (negative_similarities > 0.7).float().mean().item()

    return {
        # Core metrics
        "positive_similarity": positive_sim_mean,
        "negative_similarity": negative_sim_mean,
        "synonym_margin": synonym_margin,
        "retrieval_accuracy": retrieval_accuracy,
        "mrr": mrr,

        # Distribution statistics
        "positive_sim_std": positive_sim_std,
        "negative_sim_std": negative_sim_std,
        "weak_positives_ratio": weak_positives_ratio,
        "false_positives_ratio": false_positives_ratio,

        # Debug metric
        "accuracy_gap": accuracy_gap,  # How far the diagonal is from the max
    }



def train_model(model, training_args, train_dataset, val_dataset, data_collator):
    """Train the model."""
    trainer = CachedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer


def save_model(trainer, tokenizer, save_path):
    """Save the model and tokenizer."""
    # This should work since it inherits from PreTrainedModel
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Training completed. Model saved to {save_path}.")

def initialize_wandb(project_name, run_name, model_name, dataset_path, training_args=None):
    """Initialize WANDB and set the run configuration."""
    wandb_config = {
        "model_name": model_name,
        "data_file": dataset_path,
        "approach": "contrastive_cached_negatives",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if training_args:
        wandb_config.update({
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "learning_rate": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_ratio": training_args.warmup_ratio,
        })

    try:
        wandb.init(project=project_name, name=run_name, config=wandb_config)
        print("✅ WANDB initialized.")
        return True
    except Exception as e:
        print(f"⚠️ WANDB initialization error: {e}")
        return False

# ======================
# Main Pipeline
# ======================

def main():
    """Main training pipeline."""
    # --- Configuration ---
    dataset_path = "dataset/contrastive-synonym-dataset_55k.jsonl"
    model_name = "intfloat/multilingual-e5-large"
    model_name_safe = model_name.replace("/", "-")
    output_dir = f"models/tr-synonym-{model_name_safe}-cached-v1"
    save_path = f"./tr-synonym-{model_name_safe}-cached-v1"
    wandb_project = "turkish-synonym-model"
    wandb_run_name = f"tr-synonym-{model_name_safe}-cached-v1"

    # --- Pipeline Steps ---
    check_environment()

    print("📂 Loading and preparing data...")
    samples = load_contrastive_data(dataset_path)
    print(f"✅ Loaded {len(samples)} original samples.")

    print("🔀 Shuffling and splitting data...")
    train_samples, val_samples = shuffle_and_split_data(samples)
    print(f"✅ Train set: {len(train_samples)}, Validation set: {len(val_samples)}")

    print("📊 Building datasets...")
    train_dataset, val_dataset = prepare_datasets(train_samples, val_samples)

    print("🤖 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # ContrastiveModel inherits from PreTrainedModel, so it requires a config.
    # We can use the transformer's own config; we only load the config here.
    base_model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = ContrastiveModel(config=base_model_config, model_name=model_name)

    data_collator = ContrastiveDataCollator(tokenizer=tokenizer)

    print("⚙️ Setting training parameters...")
    training_args = create_training_arguments(output_dir)

    print("📊 Initializing WANDB...")
    wandb_enabled = initialize_wandb(wandb_project, wandb_run_name, model_name, dataset_path, training_args)

    print("🚀 Training started...")
    trainer = train_model(
        model,
        training_args,
        train_dataset,
        val_dataset,
        data_collator
    )

    print("💾 Saving model...")
    save_model(trainer, tokenizer, save_path)

    if wandb_enabled:
        wandb.finish()

    print("🎉 All steps completed!")

if __name__ == "__main__":
    main()
