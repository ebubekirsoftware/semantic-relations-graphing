import json
import os
import subprocess

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Keep ID2LABEL immutable/global for multiprocessing pickling
ID2LABEL = {0: "antonym", 1: "co-hyponym", 2: "synonym"}


def count_lines(filepath: str) -> int:
    """Efficiently count total lines using 'wc -l'."""
    print(f"Counting lines in {filepath} via 'wc -l' to calculate total chunks...")
    try:
        result = subprocess.check_output(["wc", "-l", filepath])
        lines = int(result.decode("utf-8").strip().split()[0])
        print(f"✓ Found {lines:,} total pairs.")
        return lines
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
        print(f"⚠️ 'wc -l' command failed ({e}). Ensure standard Unix utilities are available.")
        raise SystemExit("Exiting: Cannot determine total line count.")


class CollateWrapper:
    """
    Global callable for multiprocessing-safe collation.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch_list):
        s1 = [x["word1"] for x in batch_list]
        s2 = [x["word2"] for x in batch_list]
        return self.tokenizer(
            s1,
            s2,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )


class PairingDataset(Dataset):
    def __init__(self, data_chunk):
        self.data = data_chunk

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SynonymClassifierRunner:
    """
    Chunked inference for synonym/antonym/co-hyponym classification.
    """

    def __init__(
        self,
        model_path: str,
        input_jsonl: str,
        output_dir: str,
        batch_size: int,
        chunk_size: int,
        confidence_threshold: float,
        num_workers: int,
        device: str,
    ):
        self.model_path = model_path
        self.input_jsonl = input_jsonl
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.confidence_threshold = confidence_threshold
        self.num_workers = num_workers
        self.device = torch.device(device)

    @staticmethod
    def _setup_environment(output_dir: str):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _get_completed_chunk_count(output_dir: str) -> int:
        if not os.path.exists(output_dir):
            return 0
        chunk_files = [
            f
            for f in os.listdir(output_dir)
            if f.startswith("chunk_") and f.endswith(".parquet")
        ]
        if not chunk_files:
            return 0
        try:
            indices = [int(f.split("_")[1].split(".")[0]) for f in chunk_files if "_" in f]
            return max(indices) + 1 if indices else 0
        except (ValueError, IndexError):
            return 0

    def _inference_loop(self, model, dataloader, original_data_chunk):
        model.eval()
        all_results = []
        current_item_idx = 0

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in tqdm(
                dataloader, desc="Inference Batch", leave=False, mininterval=1.0
            ):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=-1)
                max_probs, predicted_indices = torch.max(probs, dim=-1)

                batch_preds = predicted_indices.cpu().numpy()
                batch_conf = max_probs.cpu().numpy()
                batch_probs_cpu = probs.cpu().numpy()

                for i in range(len(batch_preds)):
                    pair_data = original_data_chunk[current_item_idx]
                    pred_idx = int(batch_preds[i])
                    confidence = float(batch_conf[i])
                    predicted_label = ID2LABEL[pred_idx]

                    if predicted_label in ["synonym", "antonym"] and confidence < self.confidence_threshold:
                        final_class = "co-hyponym"
                    else:
                        final_class = predicted_label

                    all_results.append(
                        {
                            "word1": pair_data["word1"],
                            "word2": pair_data["word2"],
                            "predicted_class": final_class,
                            "confidence": round(confidence, 4),
                            "probabilities": {
                                "antonym": round(float(batch_probs_cpu[i][0]), 4),
                                "co-hyponym": round(float(batch_probs_cpu[i][1]), 4),
                                "synonym": round(float(batch_probs_cpu[i][2]), 4),
                            },
                        }
                    )
                    current_item_idx += 1
        return all_results

    def run(self):
        self._setup_environment(self.output_dir)

        total_lines = count_lines(self.input_jsonl)
        total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size

        print(
            f"Model: {self.model_path} | Batch: {self.batch_size} | Workers: {self.num_workers}"
        )
        print(f"Total Chunks to Process: {total_chunks}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        collate_fn = CollateWrapper(tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        model = model.to(self.device, dtype=torch.bfloat16)
        model.eval()

        model_id2label = {int(k): v for k, v in model.config.id2label.items()}
        if model_id2label != ID2LABEL:
            print("=" * 80)
            print("⚠️  WARNING: Label mapping mismatch detected!")
            print(f"   Script ID2LABEL:  {ID2LABEL}")
            print(f"   Model id2label:   {model_id2label}")
            print("=" * 80)
            raise SystemExit("Exiting: Label mapping inconsistency. Please fix ID2LABEL constant.")
        else:
            print(f"✓ Label mapping validated: {ID2LABEL}")

        processed_chunks = self._get_completed_chunk_count(self.output_dir)
        lines_to_skip = processed_chunks * self.chunk_size

        current_chunk_data = []
        chunk_idx = processed_chunks

        with open(self.input_jsonl, "r", encoding="utf-8") as f:
            if lines_to_skip > 0:
                for _ in tqdm(range(lines_to_skip), desc="Skipping"):
                    next(f, None)

            for line in tqdm(f, desc="Reading", mininterval=1.0, unit="lines"):
                try:
                    current_chunk_data.append(json.loads(line))

                    if len(current_chunk_data) >= self.chunk_size:
                        current_chunk_data.sort(
                            key=lambda x: len(x["word1"]) + len(x["word2"])
                        )

                        dataset = PairingDataset(current_chunk_data)
                        dataloader = DataLoader(
                            dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True,
                        )

                        tqdm.write(
                            f"--- Processing Chunk {chunk_idx + 1}/{total_chunks} ({len(current_chunk_data):,} pairs) ---"
                        )
                        results = self._inference_loop(model, dataloader, current_chunk_data)

                        df = pd.DataFrame(results)
                        df.to_parquet(
                            f"{self.output_dir}/chunk_{chunk_idx:05d}.parquet",
                            index=False,
                            compression="snappy",
                            engine="pyarrow",
                        )
                        tqdm.write(f"✓ Chunk {chunk_idx + 1}/{total_chunks} done.")

                        del current_chunk_data, results, df, dataset, dataloader
                        current_chunk_data = []
                        chunk_idx += 1

                except json.JSONDecodeError:
                    continue

            if current_chunk_data:
                current_chunk_data.sort(
                    key=lambda x: len(x["word1"]) + len(x["word2"])
                )

                tqdm.write(
                    f"--- Processing Final Chunk {chunk_idx + 1}/{total_chunks} ({len(current_chunk_data):,} pairs) ---"
                )
                dataset = PairingDataset(current_chunk_data)
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                )
                results = self._inference_loop(model, dataloader, current_chunk_data)
                pd.DataFrame(results).to_parquet(
                    f"{self.output_dir}/chunk_{chunk_idx:05d}.parquet",
                    index=False,
                    compression="snappy",
                    engine="pyarrow",
                )
                print(f"✓ Final Chunk {chunk_idx + 1}/{total_chunks} done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify word pairs into synonym/antonym/co-hyponym classes with chunked inference."
    )
    parser.add_argument("--model-path", required=True, help="Path to sequence classification model.")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL of word pairs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for parquet chunks.")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for inference.")
    parser.add_argument("--chunk-size", type=int, required=True, help="Number of pairs per chunk.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        required=True,
        help="Confidence threshold to downgrade synonym/antonym to co-hyponym.",
    )
    parser.add_argument("--num-workers", type=int, required=True, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        required=True,
        help='Device string, e.g., "cuda:0".',
    )
    args = parser.parse_args()

    runner = SynonymClassifierRunner(
        model_path=args.model_path,
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        confidence_threshold=args.confidence_threshold,
        num_workers=args.num_workers,
        device=args.device,
    )
    runner.run()
