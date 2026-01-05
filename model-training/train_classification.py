import json
import numpy as np
import os
import random
import warnings
import torch
import wandb
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss

# Add parent directory to path for config import
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# Configuration
MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"
DATASET_PATH = "synonym_dataset_ce.jsonl"
OUTPUT_DIR = "models/classification"
WANDB_PROJECT = "synonym-framework"
MAX_LENGTH = 64

# Label definitions
LABEL2ID = {
    "antonym": 0,
    "co-hyponym": 1,
    "synonym": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that inherits from Hugging Face Trainer and supports class weights.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Redefines the loss function (CrossEntropyLoss) to use class weights.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss



def check_environment():
    """Check environment and login to wandb."""
    warnings.filterwarnings("ignore")
    
    wandb_key = config.get("wandb_api_key")
    if not wandb_key or wandb_key == "YOUR_WANDB_API_KEY_HERE":
        print("⚠️ WANDB API key not configured. Running without wandb...")
        os.environ["WANDB_MODE"] = "disabled"
        return
    
    try:
        wandb.login(key=wandb_key)
        print("✅ WANDB login successful.")
    except Exception as e:
        print(f"⚠️ WANDB login error: {e}")
        print("⚠️ Continuing without wandb...")
        os.environ["WANDB_MODE"] = "disabled"


def load_jsonl(path):
    """Load JSONL format data and convert labels."""
    with open(path, "r") as f:
        samples = [
            {
                "sentence1": data["sentence1"],
                "sentence2": data["sentence2"],
                "label": LABEL2ID[data["label"]]
            }
            for line in f
            for data in [json.loads(line)]
        ]
    return samples


def save_samples_to_jsonl(samples, path, id2label=None):
    """Save sample list to JSONL format."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            sample_to_write = sample.copy()
            if id2label and 'label' in sample_to_write:
                sample_to_write['label'] = id2label[sample_to_write['label']]
            f.write(json.dumps(sample_to_write, ensure_ascii=False) + "\n")


def shuffle_and_split_data(samples, split_ratio=0.999, seed=42):
    """Shuffle data and split into train/validation."""
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
    
    train_label_dist = Counter([sample['label'] for sample in train_samples])
    val_label_dist = Counter([sample['label'] for sample in val_samples])
    
    print(f"✅ Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"✅ Train samples: {train_samples[:5]}, Validation samples: {val_samples[:5]}")
    print(f"✅ Train label distribution: {dict(train_label_dist)}")
    print(f"   - Antonym (0): {train_label_dist.get(0, 0)} ({train_label_dist.get(0, 0)/len(train_samples)*100:.2f}%)")
    print(f"   - Co-hyponym (1): {train_label_dist.get(1, 0)} ({train_label_dist.get(1, 0)/len(train_samples)*100:.2f}%)")
    print(f"   - Synonym (2): {train_label_dist.get(2, 0)} ({train_label_dist.get(2, 0)/len(train_samples)*100:.2f}%)")
    print(f"✅ Validation label distribution: {dict(val_label_dist)}")
    print(f"   - Antonym (0): {val_label_dist.get(0, 0)} ({val_label_dist.get(0, 0)/len(val_samples)*100:.2f}%)")
    print(f"   - Co-hyponym (1): {val_label_dist.get(1, 0)} ({val_label_dist.get(1, 0)/len(val_samples)*100:.2f}%)")
    print(f"   - Synonym (2): {val_label_dist.get(2, 0)} ({val_label_dist.get(2, 0)/len(val_samples)*100:.2f}%)")
    
    return train_dataset, val_dataset


def tokenize_datasets(train_dataset, val_dataset, model_name, max_length=MAX_LENGTH):
    """Tokenize datasets."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    def preprocess(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    train_dataset = train_dataset.map(preprocess, batched=True)
    val_dataset = val_dataset.map(preprocess, batched=True)
    
    # Dataset için format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    return train_dataset, val_dataset, tokenizer


def load_model(model_name, num_labels=3):
    """Load model for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True
    )
    return model


def create_training_arguments(output_dir):
    """Create training arguments."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=50,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="wandb",
        bf16=True,
        seed=42,
        gradient_checkpointing=True
    )
    return training_args


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    This function calculates both general (macro) metrics and per-class
    precision, recall and f1 scores.
    """
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    metrics = {}

    # Macro-averaged metrics (evaluates all classes with equal weight)
    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["f1_macro"] = f1_score(labels, preds, average="macro")
    metrics["precision_macro"] = precision_score(labels, preds, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(labels, preds, average="macro", zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    
    for i, label_name in ID2LABEL.items():
        metrics[f"{label_name}_f1"] = per_class_f1[i]
        metrics[f"{label_name}_precision"] = per_class_precision[i]
        metrics[f"{label_name}_recall"] = per_class_recall[i]

    return metrics


def train_model(model, training_args, train_dataset, val_dataset, tokenizer, class_weights=None):
    """Train the model."""
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )
    
    trainer.train()
    return trainer


def save_model(trainer, tokenizer, save_path):
    """Save model and tokenizer."""
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Training completed. Model saved to {save_path}")


def initialize_wandb(project_name, run_name, model_name, dataset_path, training_args=None):
    """Initialize wandb and set configuration."""
    wandb_config = {
        "model_name": model_name,
        "data_file": dataset_path,
        "approach": "classification_with_class_weights",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_labels": 3,
        "label_types": ["antonym", "co-hyponym", "synonym"]
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
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "max_grad_norm": training_args.max_grad_norm,
            "bf16": training_args.bf16,
            "gradient_checkpointing": training_args.gradient_checkpointing,
        })
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config
        )
        print("✅ WANDB initialized.")
        return True
    except Exception as e:
        print(f"⚠️ WANDB initialization error: {e}")
        return False


def main():
    """Main training pipeline."""
    # Setup paths
    model_name_safe = MODEL_NAME.replace("/", "-")
    output_dir = os.path.join(OUTPUT_DIR, model_name_safe)
    save_path = os.path.join(OUTPUT_DIR, model_name_safe)
    wandb_run_name = f"classification-{model_name_safe}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Environment check and WANDB login
    check_environment()
    
    # Load data
    print("📂 Loading data...")
    samples = load_jsonl(DATASET_PATH)
    
    # Prepare data
    print("🔀 Shuffling and splitting data...")
    train_samples, val_samples = shuffle_and_split_data(samples)
    
    # Create datasets
    print("📊 Preparing datasets...")
    train_dataset, val_dataset = prepare_datasets(train_samples, val_samples)
    
    print("⚖️ Computing class weights...")
    train_labels = [sample['label'] for sample in train_samples]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"✅ Class weights: {class_weights_tensor.tolist()}")
    
    # Tokenization
    print("🔤 Tokenizing...")
    train_dataset, val_dataset, tokenizer = tokenize_datasets(
        train_dataset, val_dataset, MODEL_NAME
    )
    
    # Load model
    print("🤖 Loading model...")
    model = load_model(MODEL_NAME)
    
    # Training arguments
    print("⚙️ Setting training parameters...")
    training_args = create_training_arguments(output_dir)
    
    # Initialize WANDB
    print("📊 Initializing WANDB...")
    wandb_enabled = initialize_wandb(WANDB_PROJECT, wandb_run_name, MODEL_NAME, DATASET_PATH, training_args)
    
    # Training
    print("🚀 Starting training...")
    trainer = train_model(
        model, 
        training_args, 
        train_dataset, 
        val_dataset, 
        tokenizer,
        class_weights=class_weights_tensor
    )
    
    # Save model
    print("💾 Saving model...")
    save_model(trainer, tokenizer, save_path)
    
    # Finish WANDB
    if wandb_enabled:
        wandb.finish()
    
    print("🎉 All done!")


if __name__ == "__main__":
    main()
