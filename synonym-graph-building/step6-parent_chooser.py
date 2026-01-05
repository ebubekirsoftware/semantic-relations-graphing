import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


class ParentChooserRunner:
    """
    Assign parent terms to clusters based on centroid similarity (with heuristic fallback).
    All paths/params are provided externally for framework use.
    """

    def __init__(
        self,
        embeddings_path: str,
        clusters_file: str,
        output_file: str,
        num_calc_workers: int,
        num_read_workers: int,
        chunk_size: int,
    ):
        self.embeddings_path = embeddings_path
        self.clusters_file = clusters_file
        self.output_file = output_file
        self.num_calc_workers = num_calc_workers
        self.num_read_workers = num_read_workers
        self.chunk_size = chunk_size

        self._needed_terms: Set[str] = set()
        self._all_lines: List[str] = []
        self._embedding_lookup: Dict[str, np.ndarray] = {}

    @staticmethod
    def _collect_needed_terms(clusters_file: str) -> Tuple[Set[str], List[str]]:
        print("⏳ [1/4] Collecting required terms...")
        needed_terms: Set[str] = set()

        with open(clusters_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        for line in tqdm(all_lines, desc="Term analysis"):
            try:
                cluster = json.loads(line)
                if not cluster.get("parent"):
                    needed_terms.update(cluster.get("synonyms", []))
            except json.JSONDecodeError:
                continue

        print(f"✅ Total {len(needed_terms)} terms needed.")
        return needed_terms, all_lines

    @staticmethod
    def _load_row_group_process(args):
        path, row_group_idx, needed_terms = args

        parquet_file = pq.ParquetFile(path)
        table = parquet_file.read_row_group(
            row_group_idx, columns=["konu", "normalized_embeddings"]
        )

        konu_col = table.column("konu").to_pylist()
        embeddings_col = table.column("normalized_embeddings").to_pylist()

        local_dict = {}
        for term, vec in zip(konu_col, embeddings_col):
            if term in needed_terms and vec is not None:
                local_dict[term] = np.array(vec, dtype=np.float32)
        return local_dict

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        print(f"⏳ [2/4] Loading embeddings with processes ({self.num_read_workers} workers)...")

        parquet_file = pq.ParquetFile(self.embeddings_path)
        total_row_groups = parquet_file.num_row_groups

        args_list = [(self.embeddings_path, i, self._needed_terms) for i in range(total_row_groups)]

        data_dict: Dict[str, np.ndarray] = {}

        with ProcessPoolExecutor(max_workers=self.num_read_workers) as executor:
            futures = {
                executor.submit(self._load_row_group_process, arg): arg for arg in args_list
            }

            for future in tqdm(as_completed(futures), total=total_row_groups, desc="Parquet Read"):
                try:
                    result = future.result()
                    data_dict.update(result)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️ Read error (RowGroup): {e}")

        print(f"✅ Embedding map ready in memory: {len(data_dict)} terms.")
        return data_dict

    def _get_centroid_parent(self, synonyms: Iterable[str]) -> Optional[str]:
        valid_vectors = []
        valid_terms = []

        for term in synonyms:
            vec = self._embedding_lookup.get(term)
            if vec is not None:
                valid_vectors.append(vec)
                valid_terms.append(term)

        if not valid_vectors:
            return None

        vectors_matrix = np.array(valid_vectors)
        centroid = np.mean(vectors_matrix, axis=0)

        vec_norms = np.linalg.norm(vectors_matrix, axis=1)
        centroid_norm = np.linalg.norm(centroid)

        if centroid_norm == 0:
            return valid_terms[0]

        dot_products = np.dot(vectors_matrix, centroid)
        eps = 1e-9
        similarities = dot_products / ((vec_norms * centroid_norm) + eps)

        best_idx = int(np.argmax(similarities))
        return valid_terms[best_idx]

    @staticmethod
    def _get_heuristic_parent(synonyms: Sequence[str]) -> str:
        if not synonyms:
            return ""
        sorted_syns = sorted(synonyms, key=lambda x: (x[0].islower(), len(x), x))
        return sorted_syns[0]

    def _process_chunk(self, lines_chunk: List[str]) -> List[str]:
        results = []
        for line in lines_chunk:
            try:
                cluster = json.loads(line)
                existing_parent = cluster.get("parent")

                if existing_parent:
                    results.append(line.strip())
                    continue

                synonyms = cluster.get("synonyms", [])
                parent = self._get_centroid_parent(synonyms)

                if not parent:
                    parent = self._get_heuristic_parent(synonyms)

                cluster["parent"] = parent
                results.append(json.dumps(cluster, ensure_ascii=False))

            except json.JSONDecodeError:
                continue
        return results

    def run(self):
        try:
            multiprocessing.set_start_method("fork")
        except RuntimeError:
            pass

        if not os.path.exists(self.embeddings_path) or not os.path.exists(self.clusters_file):
            print("❌ Missing files.")
            return

        self._needed_terms, self._all_lines = self._collect_needed_terms(self.clusters_file)
        if not self._needed_terms:
            return

        self._embedding_lookup = self._load_embeddings()

        print(f"🚀 [3/4] Computing with {self.num_calc_workers} CPUs...")

        chunks = [
            self._all_lines[i : i + self.chunk_size]
            for i in range(0, len(self._all_lines), self.chunk_size)
        ]
        print(f"   {len(chunks)} chunks to process.")

        print(f"💾 [4/4] Processing and writing: {self.output_file}")

        processed_count = 0

        with open(self.output_file, "w", encoding="utf-8") as f_out:
            with ProcessPoolExecutor(max_workers=self.num_calc_workers) as executor:
                for chunk_result in tqdm(
                    executor.map(self._process_chunk, chunks),
                    total=len(chunks),
                    desc="Processing",
                ):
                    for line in chunk_result:
                        f_out.write(line + "\n")
                        processed_count += 1

        print(f"🎉 Completed! Total {processed_count} lines written.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Assign cluster parents using centroid similarity with a heuristic fallback."
    )
    parser.add_argument("--embeddings-path", required=True, help="Parquet file with embeddings.")
    parser.add_argument("--clusters-file", required=True, help="Clusters JSONL needing parents.")
    parser.add_argument("--output-file", required=True, help="Output JSONL with parents assigned.")
    parser.add_argument("--num-calc-workers", type=int, required=True, help="CPU workers for compute.")
    parser.add_argument("--num-read-workers", type=int, required=True, help="Workers for parquet read.")
    parser.add_argument("--chunk-size", type=int, required=True, help="Lines per processing chunk.")
    args = parser.parse_args()

    runner = ParentChooserRunner(
        embeddings_path=args.embeddings_path,
        clusters_file=args.clusters_file,
        output_file=args.output_file,
        num_calc_workers=args.num_calc_workers,
        num_read_workers=args.num_read_workers,
        chunk_size=args.chunk_size,
    )
    runner.run()