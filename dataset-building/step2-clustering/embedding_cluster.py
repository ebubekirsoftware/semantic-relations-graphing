import ast
import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from unicode_tr import unicode_tr


class EmbeddingClusterRunner:
    """
    Run agglomerative clustering on FastText embeddings.
    All I/O paths and clustering parameters are provided externally.
    """

    def __init__(self, input_file: str, output_file: str, distance_threshold: float):
        self.input_file = input_file
        self.output_file = output_file
        self.distance_threshold = distance_threshold

    @staticmethod
    def _load_and_prepare_dataset(path: str):
    """Load embeddings, normalize text, remove duplicates and zero vectors."""
    df = pd.read_csv(path)
    text_col = df.columns[0]
    
    df[text_col] = df[text_col].apply(lambda x: unicode_tr(x).lower())
        df = df.drop_duplicates(subset=[text_col], keep="first")
    
    texts = df[text_col].values
        embeddings = np.array([ast.literal_eval(emb) for emb in df["embedding"]])

    non_zero_mask = np.any(embeddings != 0, axis=1)
    num_zero = (~non_zero_mask).sum()
    
    if num_zero > 0:
        print(f"Warning: {num_zero} zero vectors filtered from '{path}'")

    return texts[non_zero_mask], embeddings[non_zero_mask]

    @staticmethod
    def _l2_normalize(embeddings):
    """Apply L2 normalization to embeddings."""
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    return embeddings / norm

    def _cluster_agglomerative(self, embeddings):
    """Cluster using Agglomerative Clustering with cosine distance."""
        embeddings = self._l2_normalize(embeddings)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="complete",
    )
    return clusterer.fit_predict(embeddings)

    @staticmethod
    def _save_clusters(texts, labels, output_file):
    """Save clusters to JSON file."""
    clusters = {}
    for text, label in zip(texts, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text)
    
        with open(output_file, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in clusters.items()}, f, ensure_ascii=False, indent=2)
    
        print(f"Saved {len(clusters)} clusters -> {output_file}")

    def run(self):
        texts, embeddings = self._load_and_prepare_dataset(self.input_file)
    print(f"Loaded {len(texts)} terms with {embeddings.shape[1]}D embeddings")
    
        labels = self._cluster_agglomerative(embeddings)
    print(f"Found {len(set(labels))} clusters")
    
        self._save_clusters(texts, labels, self.output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run agglomerative clustering on FastText embeddings."
    )
    parser.add_argument("--input-file", required=True, help="CSV file containing embeddings.")
    parser.add_argument("--output-file", required=True, help="JSON path to write clusters.")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        required=True,
        help="Distance threshold for agglomerative clustering.",
    )
    args = parser.parse_args()

    runner = EmbeddingClusterRunner(
        input_file=args.input_file,
        output_file=args.output_file,
        distance_threshold=args.distance_threshold,
    )
    runner.run()
