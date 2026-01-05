import gc
import json
import os
from typing import Dict

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

os.environ["POLARS_MAX_THREADS"] = "64"


class ParentChildEnricherRunner:
    """
    Enrich parquet with parent-child mapping by adding synonym_id column.
    """

    def __init__(
        self,
        parquet_path: str,
        clusters_path: str,
        output_path: str,
        batch_size: int,
    ):
        self.parquet_path = parquet_path
        self.clusters_path = clusters_path
        self.output_path = output_path
        self.batch_size = batch_size

    def _build_mapping(self) -> Dict[str, str]:
        """
        Build {child_konu: parent_ml_id} mapping.
        Parents remain without synonym_id (None).
        """
        print("1. Mapping Preparation: Loading lookup (konu -> ml_id)...")
        df_lookup = pl.read_parquet(self.parquet_path, columns=["ml_id", "konu"])
        term_to_uuid = dict(zip(df_lookup["konu"].to_list(), df_lookup["ml_id"].to_list()))
        print(f"   - Loaded {len(term_to_uuid):,} terms.")

        del df_lookup
        gc.collect()

        print(f"2. Mapping Preparation: Scanning clusters file ({self.clusters_path})...")
        child_to_parent_uuid: Dict[str, str] = {}
        parent_count = 0
        skipped_count = 0

        with open(self.clusters_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    parent_term = data.get("parent")
                    synonyms = data.get("synonyms", [])

                    if not parent_term or not synonyms:
                        skipped_count += 1
                        continue

                    if parent_term in term_to_uuid:
                        p_uuid = term_to_uuid[parent_term]
                        parent_count += 1

                        for child in synonyms:
                            if child != parent_term and child in term_to_uuid:
                                child_to_parent_uuid[child] = p_uuid
                    else:
                        skipped_count += 1

                except Exception:
                    skipped_count += 1
                    continue

        print(f"   - {parent_count:,} parent clusters processed.")
        print(f"   - {len(child_to_parent_uuid):,} child mappings prepared.")
        print(f"   - {skipped_count:,} lines skipped (invalid or missing data).")

        return child_to_parent_uuid

    def run(self):
        print("=" * 70)
        print("STARTING SYNONYM_ID ENRICHMENT")
        print("=" * 70)

        mapping_dict = self._build_mapping()

        print("\n3. Batch Processing (RAM-safe)...")

        parquet_file = pq.ParquetFile(self.parquet_path)

        original_schema = parquet_file.schema_arrow
        new_field = pa.field("synonym_id", pa.string())
        new_schema = original_schema.append(new_field)

        writer = None
        total_rows = parquet_file.metadata.num_rows

        print(f"   - Total rows: {total_rows:,}")
        print(f"   - Batch size: {self.batch_size:,}")

        pbar = tqdm(total=total_rows, unit="row", desc="Processing")

        parent_count = 0
        child_count = 0

        for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            df_chunk = pl.from_arrow(batch)

            df_chunk = df_chunk.with_columns(
                pl.col("konu")
                .replace_strict(mapping_dict, default=None, return_dtype=pl.String)
                .alias("synonym_id")
            )

            parent_count += df_chunk.filter(pl.col("synonym_id").is_null()).height
            child_count += df_chunk.filter(pl.col("synonym_id").is_not_null()).height

            table = df_chunk.to_arrow()
            table = table.cast(new_schema)

            if writer is None:
                writer = pq.ParquetWriter(self.output_path, new_schema)

            writer.write_table(table)

            pbar.update(df_chunk.height)

            del df_chunk
            del table
            gc.collect()

        if writer:
            writer.close()

        pbar.close()

        print("\n" + "=" * 70)
        print("✅ ENRICHMENT COMPLETED")
        print("=" * 70)
        print(f"Output file: {self.output_path}")
        print(f"Total rows: {total_rows:,}")
        print(f"  - Parent rows (synonym_id=None): {parent_count:,}")
        print(f"  - Child rows (synonym_id=parent_ml_id): {child_count:,}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add synonym_id to parquet based on cluster parent-child mapping."
    )
    parser.add_argument("--parquet-path", required=True, help="Input parquet path.")
    parser.add_argument("--clusters-path", required=True, help="Clusters JSONL with parents.")
    parser.add_argument("--output-path", required=True, help="Output enriched parquet path.")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size.")
    args = parser.parse_args()

    runner = ParentChildEnricherRunner(
        parquet_path=args.parquet_path,
        clusters_path=args.clusters_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    runner.run()