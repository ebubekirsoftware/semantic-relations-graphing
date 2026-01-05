"""
Generate embeddings for text data using trained embedding model.

Supports multi-GPU processing for efficient embedding generation from large
Parquet datasets. All paths and parameters are injected externally for
framework use.
"""

import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Performance optimizations
torch.set_float32_matmul_precision("high")


class EmbeddingGeneratorRunner:
    """
    Generate embeddings from a Parquet file using a SentenceTransformer model.
    """

    def __init__(
        self,
        model_path: str,
        input_parquet: str,
        output_parquet: str,
        num_gpus: int,
        batch_size: int,
    ):
        self.model_path = model_path
        self.input_parquet = input_parquet
        self.output_parquet = output_parquet
        self.num_gpus = num_gpus
        self.batch_size = batch_size

        self._progress_lock = Lock()
        self._progress_bar = None

    def _process_worker(self, worker_id, device_id, data_chunk):
        """
        Process a chunk of data on a specific GPU device using SentenceTransformer.
        """
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)

        model = SentenceTransformer(self.model_path, device=device)
        print(f"Worker {worker_id}: Model loaded from {self.model_path}")

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        results = []

        for i in range(0, len(data_chunk), self.batch_size):
            batch = data_chunk[i : i + self.batch_size]
            batch_indices = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]

            embeddings = model.encode(
                batch_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for j, idx in enumerate(batch_indices):
                results.append((idx, embeddings[j]))

            if i % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()

            with self._progress_lock:
                self._progress_bar.update(len(batch))

        return results

    def run(self):
        """Main pipeline for generating embeddings from Parquet data."""
        print(f"📂 Loading data from {self.input_parquet}")
        df = pd.read_parquet(self.input_parquet)
        total_records = len(df)
        print(f"✅ Loaded {total_records:,} records")

        topics = [(idx, str(row["konu"])) for idx, row in df.iterrows()]

        chunk_size = (len(topics) + self.num_gpus - 1) // self.num_gpus
        partitions = [
            topics[i * chunk_size : (i + 1) * chunk_size] for i in range(self.num_gpus)
        ]
        partitions = [p for p in partitions if p]

        print(f"\n⚙️ Processing with {self.num_gpus} GPUs, batch size {self.batch_size}")
        print("🚀 Starting embedding generation...\n")

        self._progress_bar = tqdm(
            total=total_records,
            desc="Embeddings",
            unit="records",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:.1f}%] [{elapsed}<{remaining}]",
        )

        all_results = {}
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = [
                executor.submit(self._process_worker, i, i, partitions[i])
                for i in range(len(partitions))
            ]

            for future in futures:
                for idx, embedding in future.result():
                    all_results[idx] = embedding

        self._progress_bar.close()

        print(f"\n✅ Generated {len(all_results):,} embeddings")
        print(f"💾 Saving results to {self.output_parquet}...")

        sorted_indices = sorted(all_results.keys())
        embeddings_array = np.array(
            [all_results[idx] for idx in sorted_indices], dtype=np.float32
        )

        output_df = df.loc[sorted_indices].copy()
        output_df["embedding"] = list(embeddings_array)

        output_df.to_parquet(
            self.output_parquet, index=True, compression="snappy", engine="pyarrow"
        )

        del all_results, embeddings_array
        gc.collect()

        print(f"✅ Saved to {self.output_parquet}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings from a Parquet file using a SentenceTransformer model."
    )
    parser.add_argument("--model-path", required=True, help="Path to the embedding model.")
    parser.add_argument("--input-parquet", required=True, help="Input Parquet file.")
    parser.add_argument("--output-parquet", required=True, help="Output Parquet file.")
    parser.add_argument(
        "--num-gpus", type=int, required=True, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--batch-size", type=int, required=True, help="Batch size for embedding."
    )
    args = parser.parse_args()

    runner = EmbeddingGeneratorRunner(
        model_path=args.model_path,
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
    )
    runner.run()

