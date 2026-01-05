import json


class ClusterPostProcessorRunner:
    """
    Post-process clustering output: optional substring filtering, move singles to
    a noise cluster, and reindex cluster IDs sequentially.
    """

    def __init__(self, input_file: str, output_file: str, use_substring_filter: bool):
        self.input_file = input_file
        self.output_file = output_file
        self.use_substring_filter = use_substring_filter

    @staticmethod
    def _filter_substring_clusters(clusters):
    """
    Remove terms that are substrings of other terms within the same cluster.
    Example: ["app", "app store", "app market"] -> ["app store", "app market"]
    """
    filtered_clusters = {}
    removed_words = []
    
    for cluster_id, words in clusters.items():
        if len(words) < 2:
            continue
            
        to_remove = set()
        
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i != j and word1 in word2 and word1 != word2:
                    to_remove.add(word1)
        
        removed_words.extend(list(to_remove))
        filtered_words = [word for word in words if word not in to_remove]
        
        if len(filtered_words) >= 2:
            filtered_clusters[cluster_id] = filtered_words
            
    return filtered_clusters, removed_words

    @staticmethod
    def _create_noise_cluster(clusters, removed_words=None):
    """Move single-term clusters and removed words to noise cluster."""
    filtered_clusters = {}
    noise_words = []
    
    for cluster_id, words in clusters.items():
        if len(words) == 1:
            noise_words.append(words[0])
        else:
            filtered_clusters[cluster_id] = words
    
    if removed_words:
        noise_words.extend(removed_words)
    
    if noise_words:
        filtered_clusters["noise_cluster"] = noise_words
    
    return filtered_clusters

    def _process_clusters(self, clusters):
    """
    Post-process clusters by removing single-term clusters to noise.
    Optionally apply substring filtering.
    """
        if self.use_substring_filter:
            filtered_clusters, removed_words = self._filter_substring_clusters(clusters)
            return self._create_noise_cluster(filtered_clusters, removed_words)
        return self._create_noise_cluster(clusters, None)

    @staticmethod
    def _reindex_clusters(clusters):
    """Reindex clusters sequentially, with noise cluster as -1."""
    reindexed = {}
    index = 0
    
    for cluster_id, words in clusters.items():
        if cluster_id == "noise_cluster":
            reindexed["-1"] = words
        else:
            reindexed[str(index)] = words
            index += 1
    
    return reindexed, index

    def run(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    
    print(f"Loaded {len(clusters)} clusters")

        processed_clusters = self._process_clusters(clusters)
        reindexed_clusters, num_clusters = self._reindex_clusters(processed_clusters)
    
        print(f"Processed {num_clusters} clusters -> {self.output_file}")
    
        with open(self.output_file, "w", encoding="utf-8") as f:
        json.dump(reindexed_clusters, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process clustering output (substring filter, noise cluster, reindex)."
    )
    parser.add_argument("--input-file", required=True, help="JSON file of raw clusters.")
    parser.add_argument("--output-file", required=True, help="JSON path to write processed clusters.")
    parser.add_argument(
        "--use-substring-filter",
        action="store_true",
        help="Enable substring filtering within clusters.",
    )
    args = parser.parse_args()

    runner = ClusterPostProcessorRunner(
        input_file=args.input_file,
        output_file=args.output_file,
        use_substring_filter=args.use_substring_filter,
    )
    runner.run()
