import gc
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, Set

from joblib import Parallel, delayed

# Global shared structures for joblib multiprocessing (fork-based)
GLOBAL_ADJ: DefaultDict[int, Set[int]] = defaultdict(set)
GLOBAL_CLUSTERS: Dict[int, Set[int]] = {}
GLOBAL_WORD_MEMBERSHIP: DefaultDict[int, Set[int]] = defaultdict(set)


class SynonymClustererStep2Runner:
    """
    Prune ambiguous memberships and produce final synonym clusters.
    Uses joblib multiprocessing with shared globals (fork).
    """

    def __init__(
        self,
        checkpoint_file: str,
        output_file: str,
        n_jobs: int,
        chunk_size: int,
        batch_size: int,
    ):
        self.checkpoint_file = checkpoint_file
        self.output_file = output_file
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.id_to_term = {}

    def _load_checkpoint(self):
        global GLOBAL_CLUSTERS, GLOBAL_WORD_MEMBERSHIP, GLOBAL_ADJ

        if not os.path.exists(self.checkpoint_file):
            print(f"❌ ERROR: {self.checkpoint_file} not found!")
            sys.exit(1)

        print(f"♻️ [Step 2] Loading checkpoint: {self.checkpoint_file}")
        with open(self.checkpoint_file, "rb") as f:
            state = pickle.load(f)

        required_keys = {"clusters", "word_membership", "adjacency", "id_to_term"}
        missing = required_keys.difference(state.keys())
        if missing:
            print("❌ ERROR: Checkpoint format missing required fields!")
            print(f"   Missing: {', '.join(sorted(missing))}")
            sys.exit(1)

        GLOBAL_CLUSTERS.update(state["clusters"])
        GLOBAL_WORD_MEMBERSHIP.update(state["word_membership"])

        GLOBAL_ADJ.clear()
        for node, neighbors in state["adjacency"].items():
            GLOBAL_ADJ[node] = set(neighbors)

        self.id_to_term = state["id_to_term"]

        del state
        gc.collect()

        print("✅ [Step 2] Checkpoint loaded. Globals ready.")

    def _run_pruning(self):
        print("[*] [Step 2] Pruning (Disambiguation) starting... (Multi-Core)")

        ambiguous_words = sorted(
            [w for w, c_ids in GLOBAL_WORD_MEMBERSHIP.items() if len(c_ids) > 1]
        )
        print(f"[*] Ambiguous words: {len(ambiguous_words):,}")

        if not ambiguous_words:
            return

        total_words = len(ambiguous_words)
        total_chunks = (total_words + self.chunk_size - 1) // self.chunk_size

        print(f"[*] RAM optimization: split into {total_chunks} chunks")
        print(
            f"[*] Backend: multiprocessing, Workers: {self.n_jobs}, Batch: {self.batch_size}"
        )
        print(f"[*] Each chunk: ~{self.chunk_size:,} words\n")

        all_results = []

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, total_words)
            chunk = ambiguous_words[start_idx:end_idx]

            print(
                f"[Chunk {chunk_idx + 1}/{total_chunks}] Processing: {len(chunk):,} words ({start_idx:,} - {end_idx:,})"
            )

            results = Parallel(
                n_jobs=self.n_jobs,
                batch_size=self.batch_size,
                backend="multiprocessing",
                verbose=5,
            )([delayed(resolve_conflict)(w) for w in chunk])

            all_results.extend(results)

            del results
            gc.collect()

            print(
                f"[Chunk {chunk_idx + 1}/{total_chunks}] Done. Total processed: {len(all_results):,}/{total_words:,}\n"
            )

        print("[*] [Step 2] Applying results...")
        removed_count = 0
        for w, winner_c_id in all_results:
            old_clusters = GLOBAL_WORD_MEMBERSHIP[w]
            for c_id in old_clusters:
                if c_id != winner_c_id:
                    GLOBAL_CLUSTERS[c_id].discard(w)
                    removed_count += 1
            GLOBAL_WORD_MEMBERSHIP[w] = {winner_c_id}

        del all_results
        gc.collect()

        print(f"[*] Pruning completed. Removed {removed_count:,} word-cluster relations.")

    def _save_final_results(self):
        print(f"[*] [Step 2] Saving final results: {self.output_file}")
        count = 0
        with open(self.output_file, "w", encoding="utf-8") as f:
            for c_id in sorted(GLOBAL_CLUSTERS.keys()):
                members = GLOBAL_CLUSTERS[c_id]
                if len(members) >= 2:
                    terms = sorted([self.id_to_term[m] for m in members])
                    f.write(
                        json.dumps({"id": c_id, "synonyms": terms}, ensure_ascii=False) + "\n"
                    )
                    count += 1
        print(f"✅ COMPLETED! {count} clean clusters written.")

    def run(self):
        sys.setrecursionlimit(20000)
        self._load_checkpoint()
        self._run_pruning()
        self._save_final_results()


def resolve_conflict(word_id):
    """
    Decide deterministically which cluster a word belongs to.

    Priority:
    1. Max shared neighbors (intersection size)
    2. If tie: smaller cluster size
    3. If tie: smaller cluster id
    """
    candidate_ids = GLOBAL_WORD_MEMBERSHIP[word_id]
    my_neighbors = GLOBAL_ADJ[word_id]

    best_cid = -1
    max_intersect = -1
    best_cluster_len = float("inf")

    for c_id in sorted(candidate_ids):
        members = GLOBAL_CLUSTERS[c_id]
        intersect = len(my_neighbors.intersection(members))

        if intersect > max_intersect:
            max_intersect = intersect
            best_cid = c_id
            best_cluster_len = len(members)
        elif intersect == max_intersect:
            if len(members) < best_cluster_len:
                best_cid = c_id
                best_cluster_len = len(members)
            elif len(members) == best_cluster_len and c_id < best_cid:
                best_cid = c_id

    return word_id, best_cid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prune ambiguous cluster memberships and save final clusters."
    )
    parser.add_argument("--checkpoint-file", required=True, help="Input checkpoint pickle.")
    parser.add_argument("--output-file", required=True, help="Output JSONL for final clusters.")
    parser.add_argument("--n-jobs", type=int, required=True, help="Worker count for joblib.")
    parser.add_argument("--chunk-size", type=int, required=True, help="Words per chunk.")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for joblib.")
    args = parser.parse_args()

    runner = SynonymClustererStep2Runner(
        checkpoint_file=args.checkpoint_file,
        output_file=args.output_file,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )
    runner.run()