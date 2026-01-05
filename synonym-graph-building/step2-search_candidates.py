"""
Search for synonym candidates using FAISS GPU index with scalar quantization.

Builds an optimized FAISS index (IVF + SQ8) and searches nearest neighbors.
All yollar/parametreler dışarıdan verilir.
"""

import gc
import os
from glob import glob

import faiss
import numpy as np
import polars as pl
from tqdm import tqdm


class SynonymCandidateSearchRunner:
    """
    Run FAISS-based neighbor search with IVF + scalar quantization (SQ8).
    """

    def __init__(
        self,
        input_path: str,
        work_dir: str,
        output_dir: str,
        id_col: str,
        emb_col: str,
        default_dim: int,
        top_k: int,
        threshold: float,
        batch_size_search: int,
        ivf_nlist: int,
        ivf_nprobe: int,
        vectors_file: str | None = None,
        ids_file: str | None = None,
    ):
        self.input_path = input_path
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.id_col = id_col
        self.emb_col = emb_col
        self.default_dim = default_dim
        self.top_k = top_k
        self.threshold = threshold
        self.batch_size_search = batch_size_search
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.vectors_file = vectors_file or os.path.join(work_dir, "vectors_fp32.dat")
        self.ids_file = ids_file or os.path.join(work_dir, "ids.npy")

        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _print_step(msg: str) -> None:
        print(f"\n[STEP] {msg}")

    def _load_binary_data(self):
        """
        Load and validate binary vector data and IDs.
        """
        self._print_step("1. Data Validation")

        if os.path.exists(self.vectors_file) and os.path.exists(self.ids_file):
            print("   -> Binary files found, skipping conversion (Resume Mode).")

            ids = np.load(self.ids_file, allow_pickle=True)
            total_rows = len(ids)

            file_size = os.path.getsize(self.vectors_file)
            dim = int(file_size / (total_rows * 4))

            print(f"   -> Detected: {total_rows:,} rows, {dim} dimensions.")
            return ids, total_rows, dim

        print("   -> Binary files not found! Please complete step 1 first.")
        print("   -> Exiting to prevent errors.")
        raise SystemExit(1)

    def _build_gpu_index(self, data_mmap, total_rows, dim):
        """
        Build and populate a GPU-based FAISS index with scalar quantization.
        """
        self._print_step("2. GPU Index Preparation (SQ8 - Low Memory)")

        res = faiss.StandardGpuResources()
        res.setTempMemory(2 * 1024 * 1024 * 1024)

        gpu_options = faiss.GpuClonerOptions()
        gpu_options.useFloat16 = True
        gpu_options.verbose = True

        print("   -> Building index architecture (IVF + ScalarQuantizer 8-bit)...")
        quantizer = faiss.IndexFlatIP(dim)
        index_cpu = faiss.IndexIVFScalarQuantizer(
            quantizer,
            dim,
            self.ivf_nlist,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )

        print("   -> Moving index to GPU...")
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, gpu_options)

        print("   -> Training index (using first 500k vectors)...")
        train_size = min(500_000, total_rows)
        train_data = data_mmap[:train_size].copy()
        faiss.normalize_L2(train_data)
        index_gpu.train(train_data)
        del train_data
        gc.collect()

        print("   -> Loading all data to GPU VRAM (with SQ8 compression)...")
        chunk_size_add = 500_000

        for i in tqdm(range(0, total_rows, chunk_size_add), desc="   Loading to GPU"):
            end = min(i + chunk_size_add, total_rows)

            batch = data_mmap[i:end].copy()
            if batch.shape[0] == 0:
                break

            faiss.normalize_L2(batch)
            index_gpu.add(batch)

            del batch
            gc.collect()

        index_gpu.nprobe = self.ivf_nprobe
        print(f"   -> GPU Index Ready. Total Vectors: {index_gpu.ntotal}")

        return index_gpu

    def _search_and_write_neighbors(self, index_gpu, data_mmap, ids, total_rows):
        """
        Search for nearest neighbors and write results in batches.
        """
        self._print_step("3. Search and Write Results")

        num_batches = (total_rows + self.batch_size_search - 1) // self.batch_size_search
        input_scan = pl.scan_parquet(self.input_path)

        for i in tqdm(range(num_batches), desc="   Searching & Writing"):
            start_idx = i * self.batch_size_search
            end_idx = min((i + 1) * self.batch_size_search, total_rows)

            output_file = os.path.join(self.output_dir, f"augmented_{i:05d}.parquet")
            if os.path.exists(output_file):
                continue

            query_vectors = data_mmap[start_idx:end_idx].copy()
            query_ids = ids[start_idx:end_idx]

            batch_len = query_vectors.shape[0]
            if batch_len == 0:
                continue

            faiss.normalize_L2(query_vectors)

            distances, indices = index_gpu.search(query_vectors, self.top_k)

            neighbor_ids_2d = ids[indices]
            query_ids_np = np.asarray(query_ids)

            valid_mask = (distances >= self.threshold) & (
                neighbor_ids_2d != query_ids_np[:, None]
            )

            neighbors_lists = [
                neighbor_ids_2d[j, valid_mask[j]].tolist() for j in range(batch_len)
            ]

            df_input_batch = input_scan.slice(start_idx, batch_len).collect()
            df_out = df_input_batch.with_columns(
                pl.Series("neighbors_ml_ids", neighbors_lists)
            )

            df_out.write_parquet(output_file)

            del query_vectors, df_input_batch, df_out, neighbors_lists
            gc.collect()

    def _merge_results(self):
        """
        Merge all augmented parquet files into a single output file.
        """
        self._print_step("4. Merge Results")

        pattern = os.path.join(self.output_dir, "augmented_*.parquet")
        aug_files = sorted(glob(pattern))

        if not aug_files:
            print("   -> No files found to merge.")
            return None

        final_path = os.path.join(self.output_dir, "merged_with_neighbors.parquet")
        print(f"   -> Found {len(aug_files)} parts. Merging...")

        if os.path.exists(final_path):
            os.remove(final_path)

        lf = pl.scan_parquet(aug_files)
        lf.sink_parquet(final_path)
        print(f"   -> Completed: {final_path}")
        return final_path

    def run(self):
        """
        Main pipeline for searching synonym candidates using FAISS GPU index.
        """
        ids, total_rows, dim = self._load_binary_data()

        data_mmap = np.memmap(
            self.vectors_file, dtype="float32", mode="r", shape=(total_rows, dim)
        )
        index_gpu = self._build_gpu_index(data_mmap, total_rows, dim)

        self._search_and_write_neighbors(index_gpu, data_mmap, ids, total_rows)
        self._merge_results()

        self._print_step("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search synonym candidates using FAISS GPU index with SQ8."
    )
    parser.add_argument("--input-path", required=True, help="Input Parquet file path.")
    parser.add_argument("--work-dir", required=True, help="Working directory for binaries.")
    parser.add_argument("--output-dir", required=True, help="Output directory for results.")
    parser.add_argument("--id-col", required=True, help="ID column name.")
    parser.add_argument("--emb-col", required=True, help="Embedding column name.")
    parser.add_argument("--default-dim", type=int, required=True, help="Embedding dimension.")
    parser.add_argument("--top-k", type=int, required=True, help="Top K neighbors to retrieve.")
    parser.add_argument("--threshold", type=float, required=True, help="Similarity threshold.")
    parser.add_argument(
        "--batch-size-search", type=int, required=True, help="Batch size for search."
    )
    parser.add_argument("--ivf-nlist", type=int, required=True, help="IVF nlist.")
    parser.add_argument("--ivf-nprobe", type=int, required=True, help="IVF nprobe.")
    parser.add_argument("--vectors-file", help="Path to vectors binary file.")
    parser.add_argument("--ids-file", help="Path to ids numpy file.")
    args = parser.parse_args()

    runner = SynonymCandidateSearchRunner(
        input_path=args.input_path,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        id_col=args.id_col,
        emb_col=args.emb_col,
        default_dim=args.default_dim,
        top_k=args.top_k,
        threshold=args.threshold,
        batch_size_search=args.batch_size_search,
        ivf_nlist=args.ivf_nlist,
        ivf_nprobe=args.ivf_nprobe,
        vectors_file=args.vectors_file,
        ids_file=args.ids_file,
    )
    runner.run()