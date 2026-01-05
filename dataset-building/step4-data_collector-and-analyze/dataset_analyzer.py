import json
from collections import Counter
from typing import Iterable, Set


class DatasetAnalyzerRunner:
    """
    Analyze JSONL dataset for duplicates, conflicts, and label distribution.
    All paths and settings are provided externally.
    """

    def __init__(
        self,
        input_file: str,
        expected_labels: Iterable[str],
        directional_dup_check: bool,
    ):
        self.input_file = input_file
        self.expected_labels: Set[str] = set(expected_labels)
        self.directional_dup_check = directional_dup_check

    def _analyze_jsonl(self, path):
        """Analyze JSONL dataset for duplicates, conflicts, and label distribution."""
        data = []
        label_counts = Counter()
        seen_pairs = {}
        duplicates = []
        conflicts = []
        invalid_rows = []

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_rows.append((i, "JSON decode error"))
                    continue

                if not all(k in row for k in ("sentence1", "sentence2", "label")):
                    invalid_rows.append((i, "missing field"))
                    continue

                s1 = row["sentence1"].strip()
                s2 = row["sentence2"].strip()
                label = row["label"].strip()

                data.append((s1, s2, label))
                label_counts[label] += 1

                key = (s1, s2) if self.directional_dup_check else tuple(
                    sorted([s1, s2])
                )

                if key in seen_pairs:
                    prev_label = seen_pairs[key]
                    if prev_label == label:
                        duplicates.append((s1, s2, label))
                    else:
                        conflicts.append((s1, s2, prev_label, label))
                else:
                    seen_pairs[key] = label

        total = len(data)
        print("=" * 60)
        print(f"Total samples: {total}")
        print(f"Invalid rows: {len(invalid_rows)}")

        print("\nLabel distribution:")
        for lbl, cnt in label_counts.items():
            print(f"  {lbl:<12}: {cnt}")

        missing = self.expected_labels - set(label_counts)
        extra = set(label_counts) - self.expected_labels
        if not missing and not extra:
            print("\n✅ Label set correct.")
        else:
            if missing:
                print(f"⚠️ Missing label(s): {missing}")
            if extra:
                print(f"⚠️ Unexpected label(s): {extra}")

        print("\nDuplicate / Conflict analysis:")
        print(f"  Direction: {'Sensitive' if self.directional_dup_check else 'Insensitive'}")
        print(f"  Unique pairs: {len(seen_pairs)}")
        print(f"  🔁 Duplicates with same label: {len(duplicates)}")
        print(f"  ⚠️ Conflicts with different label: {len(conflicts)}")

        if duplicates:
            print("\n🔁 First 3 duplicate examples:")
            for d in duplicates[:3]:
                print(f"  ({d[0]} <> {d[1]}) -> {d[2]}")

        if conflicts:
            print("\n⚠️ First 3 conflict examples:")
            for c in conflicts[:3]:
                print(f"  ({c[0]} <> {c[1]}) -> {c[2]} vs {c[3]}")

        print("=" * 60)

    def run(self):
        self._analyze_jsonl(self.input_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze CE dataset for duplicates, conflicts, and label distribution."
    )
    parser.add_argument("--input-file", required=True, help="Input CE JSONL file.")
    parser.add_argument(
        "--expected-labels",
        required=True,
        help="Comma-separated list of expected labels.",
    )
    parser.add_argument(
        "--directional-dup-check",
        action="store_true",
        help="Treat (s1, s2) as ordered when checking duplicates.",
    )
    args = parser.parse_args()

    expected = [lbl.strip() for lbl in args.expected_labels.split(",") if lbl.strip()]

    runner = DatasetAnalyzerRunner(
        input_file=args.input_file,
        expected_labels=expected,
        directional_dup_check=args.directional_dup_check,
    )
    runner.run()
