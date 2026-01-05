import gc
import os
import pickle
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, Set

import polars as pl
from tqdm import tqdm


class SynonymClustererStep1Runner:
    adjacency: DefaultDict[int, Set[int]]
    clusters: Dict[int, Set[int]]
    word_membership: DefaultDict[int, Set[int]]
    id_to_term: Dict[int, str]
    """
    Build adjacency and soft clusters from filtered synonym pairs, then save a checkpoint.
    All paths/thresholds are provided externally.
    """

    def __init__(self, input_file: str, checkpoint_file: str, threshold_ratio: float):
        self.input_file = input_file
        self.checkpoint_file = checkpoint_file
        self.threshold_ratio = threshold_ratio

        self.adjacency: DefaultDict[int, Set[int]] = defaultdict(set)
        self.clusters: Dict[int, Set[int]] = {}
        self.word_membership: DefaultDict[int, Set[int]] = defaultdict(set)
        self.id_to_term: Dict[int, str] = {}
        self.next_cluster_id = 0

    def _load_sort_and_build(self):
        print(f"[*] [Step 1] Loading with Polars: {self.input_file}")

        df = pl.read_parquet(self.input_file).select(["word1", "word2", "confidence"])
        df = df.drop_nulls()

        print("[*] [Step 1] Building deterministic ID mapping...")
        unique_terms = pl.concat(
            [df.select(pl.col("word1").alias("term")), df.select(pl.col("word2").alias("term"))]
        ).unique(subset=["term"]).sort("term")

        unique_terms = unique_terms.with_row_index("id")
        self.id_to_term = dict(unique_terms.select(["id", "term"]).iter_rows())
        print(f"[*] [Step 1] ID mapping completed: {len(self.id_to_term):,} unique terms")

        df = df.join(unique_terms, left_on="word1", right_on="term", how="left").rename({"id": "u_id"})
        df = df.join(unique_terms, left_on="word2", right_on="term", how="left").rename({"id": "v_id"})

        df = df.select(["u_id", "v_id", "confidence"])

        print("[*] [Step 1] Sorting data for determinism...")
        df = df.sort(["confidence", "u_id", "v_id"], descending=[True, False, False])

        print("[*] [Step 1] Building adjacency map...")
        edges = []
        for row in tqdm(df.iter_rows(), total=len(df), desc="Indexing"):
            u, v, _ = row
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
            edges.append((u, v))

        del df, unique_terms
        gc.collect()

        print(f"[*] [Step 1] Adjacency map built: {len(self.adjacency):,} nodes")
        return edges

    def _create_new_cluster(self, members):
        c_id = self.next_cluster_id
        self.clusters[c_id] = set(members)
        for m in members:
            self.word_membership[m].add(c_id)
        self.next_cluster_id += 1
        return c_id

    def _add_to_cluster(self, word_id, cluster_id):
        self.clusters[cluster_id].add(word_id)
        self.word_membership[word_id].add(cluster_id)

    def _run_soft_clustering(self, edges):
        print("[*] [Step 1] Soft clustering (expansion) starting...")

        for u, v in tqdm(edges, desc="Clustering"):
            u_clusters = self.word_membership.get(u, set())
            v_clusters = self.word_membership.get(v, set())

            if not u_clusters and not v_clusters:
                self._create_new_cluster([u, v])
                continue

            candidates = []
            for c_id in v_clusters:
                candidates.append((u, c_id))
            for c_id in u_clusters:
                candidates.append((v, c_id))

            for word, c_id in candidates:
                if c_id in self.word_membership[word]:
                    continue

                members = self.clusters[c_id]
                current_size = len(members)

                intersection_count = len(self.adjacency[word].intersection(members))
                ratio = intersection_count / current_size if current_size else 0.0

                if ratio > self.threshold_ratio:
                    self._add_to_cluster(word, c_id)

        print(f"[*] [Step 1] Completed. Total candidate clusters: {len(self.clusters)}")

    def _save_checkpoint(self):
        print(f"💾 [Step 1] Saving checkpoint: {self.checkpoint_file}")
        print(f"    - Clusters: {len(self.clusters):,}")
        print(f"    - Word Membership: {len(self.word_membership):,}")
        print(f"    - Adjacency: {len(self.adjacency):,}")
        print(f"    - ID to Term: {len(self.id_to_term):,}")

        state = {
            "clusters": self.clusters,
            "word_membership": self.word_membership,
            "adjacency": self.adjacency,
            "id_to_term": self.id_to_term,
        }

        print("[*] [Step 1] Pickle dump starting (may take ~20-30 minutes)...")
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_gb = os.path.getsize(self.checkpoint_file) / (1024**3)
        print(f"✅ [Step 1] Checkpoint saved: {file_size_gb:.2f} GB")
        print("✅ [Step 1] COMPLETED. You can run step 2 directly.")

    def run(self):
        sys.setrecursionlimit(20000)
        edges = self._load_sort_and_build()
        self._run_soft_clustering(edges)
        self._save_checkpoint()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Soft clustering of synonym pairs and checkpoint saving."
    )
    parser.add_argument("--input-file", required=True, help="Filtered synonyms parquet.")
    parser.add_argument("--checkpoint-file", required=True, help="Output checkpoint pickle.")
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        required=True,
        help="Neighbor overlap ratio to join a cluster.",
    )
    args = parser.parse_args()

    runner = SynonymClustererStep1Runner(
        input_file=args.input_file,
        checkpoint_file=args.checkpoint_file,
        threshold_ratio=args.threshold_ratio,
    )
    runner.run()