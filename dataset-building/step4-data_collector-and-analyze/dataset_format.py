import json
from collections import Counter


class DatasetFormatRunner:
    """
    Build CE and contrastive datasets from augmented clusters.
    All paths are provided externally for framework use.
    """

    def __init__(self, input_file: str, output_ce: str, output_contrastive: str):
        self.input_file = input_file
        self.output_ce = output_ce
        self.output_contrastive = output_contrastive

    @staticmethod
    def _load_augmented_data(filepath):
        """Load augmented semantic data from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _create_ce_pairs(data):
        """
        Create classification pairs from augmented data.
        Format: {"sentence1": "term", "sentence2": "related_term", "label": "synonym|antonym|co-hyponym"}
        """
        pairs = []
        stats = Counter()

        for item in data:
            for term, relations in item.items():
                for syn in relations.get("synonyms", []):
                    pairs.append(
                        {
                            "sentence1": term,
                            "sentence2": syn,
                            "label": "synonym",
                        }
                    )
                    stats["synonym"] += 1

                for ant in relations.get("antonyms", []):
                    pairs.append(
                        {
                            "sentence1": term,
                            "sentence2": ant,
                            "label": "antonym",
                        }
                    )
                    stats["antonym"] += 1

                for cohyp in relations.get("co-hyponyms", []):
                    pairs.append(
                        {
                            "sentence1": term,
                            "sentence2": cohyp,
                            "label": "co-hyponym",
                        }
                    )
                    stats["co-hyponym"] += 1

        return pairs, stats

    @staticmethod
    def _create_contrastive_samples(data):
        """
        Create contrastive learning samples from augmented data.
        Format: {"query": "term", "positives": ["syn1", "syn2"], "hard_negatives": ["ant1", "co1"]}
        """
        samples = []
        stats = {"total_queries": 0, "total_positives": 0, "total_hard_negatives": 0}

        for item in data:
            for term, relations in item.items():
                positives = relations.get("synonyms", [])
                hard_negatives = relations.get("antonyms", []) + relations.get(
                    "co-hyponyms", []
                )

                if positives:
                    samples.append(
                        {
                            "query": term,
                            "positives": positives,
                            "hard_negatives": hard_negatives,
                        }
                    )
                    stats["total_queries"] += 1
                    stats["total_positives"] += len(positives)
                    stats["total_hard_negatives"] += len(hard_negatives)

        return samples, stats

    @staticmethod
    def _save_jsonl(data, filepath):
        """Save data to JSONL format."""
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _print_stats(ce_stats, contrastive_stats):
        """Print dataset statistics."""
        print("=" * 60)
        print("Dataset Creation Statistics")
        print("=" * 60)

        print("\nClassification (CE) Dataset:")
        total_pairs = sum(ce_stats.values())
        print(f"  Total pairs: {total_pairs}")
        for label, count in ce_stats.items():
            percentage = (count / total_pairs) * 100 if total_pairs else 0
            print(f"  - {label:<12}: {count:>6} ({percentage:>5.2f}%)")

        print("\nContrastive Learning Dataset:")
        print(f"  Total queries: {contrastive_stats['total_queries']}")
        print(f"  Total positives: {contrastive_stats['total_positives']}")
        if contrastive_stats["total_queries"] > 0:
            avg_pos = contrastive_stats["total_positives"] / contrastive_stats[
                "total_queries"
            ]
            avg_neg = contrastive_stats["total_hard_negatives"] / contrastive_stats[
                "total_queries"
            ]
            print(f"  Avg positives per query: {avg_pos:.2f}")
            print(f"  Avg hard negatives per query: {avg_neg:.2f}")

        print("=" * 60)

    def run(self):
        print("Loading augmented data...")
        data = self._load_augmented_data(self.input_file)
        print(f"Loaded {len(data)} augmented terms")

        print("\nCreating classification pairs...")
        ce_pairs, ce_stats = self._create_ce_pairs(data)
        self._save_jsonl(ce_pairs, self.output_ce)
        print(f"Saved {len(ce_pairs)} pairs -> {self.output_ce}")

        print("\nCreating contrastive samples...")
        contrastive_samples, contrastive_stats = self._create_contrastive_samples(data)
        self._save_jsonl(contrastive_samples, self.output_contrastive)
        print(f"Saved {len(contrastive_samples)} samples -> {self.output_contrastive}")

        self._print_stats(ce_stats, contrastive_stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Format augmented clusters into CE and contrastive datasets."
    )
    parser.add_argument("--input-file", required=True, help="Input augmented JSON file.")
    parser.add_argument("--output-ce", required=True, help="Output path for CE JSONL.")
    parser.add_argument(
        "--output-contrastive",
        required=True,
        help="Output path for contrastive JSONL.",
    )
    args = parser.parse_args()

    runner = DatasetFormatRunner(
        input_file=args.input_file,
        output_ce=args.output_ce,
        output_contrastive=args.output_contrastive,
    )
    runner.run()
