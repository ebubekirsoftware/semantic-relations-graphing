import glob
import json
import logging
import math
import multiprocessing
import os

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompts import PromptBuilder  # type: ignore[attr-defined]


class MultiprocessorAugmentRunner:
    """
    Augment clusters with LLM in parallel. All I/O paths and model settings are
    provided externally to keep the runner framework-friendly.
    """

    def __init__(
        self,
        model_name: str,
        num_processes: int,
        input_file: str,
        output_file: str,
        batch_output_dir: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        load_dotenv()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.model_name = model_name
        self.num_processes = num_processes
        self.input_file = input_file
        self.output_file = output_file
        self.batch_output_dir = batch_output_dir
        self.base_url = base_url or os.environ.get("GEMINI_BASE_URL")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.base_url or not self.api_key:
            raise ValueError("GEMINI_BASE_URL and GEMINI_API_KEY must be provided.")

        self._client = None

    # Utilities
    @staticmethod
    def _load_json(filepath, default):
        """Load JSON file or return default."""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logging.warning(
                        "Could not decode JSON from %s. Returning default.", filepath
                    )
                    return default
        return default

    @staticmethod
    def _save_json(filepath, data):
        """Save data to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _clean_results(results):
        """Clean results to remove self-synonyms and duplicates."""
        cleaned = []

        for result in results:
            for word, data in result.items():
                for key in ["synonyms", "antonyms", "co-hyponyms"]:
                    if key in data and isinstance(data[key], list):
                        data[key] = list({s for s in data[key] if s != word})

                cleaned.append({word: data})

        return cleaned

    @staticmethod
    def _parse_response(content):
        """Extract JSON objects from model response."""
        results = []
        content = content.replace("```json", "").replace("```", "").strip()

        depth = 0
        current_json = ""

        for char in content:
            if char == "{":
                if depth == 0:
                    current_json = "{"
                else:
                    current_json += char
                depth += 1
            elif char == "}":
                depth -= 1
                current_json += char
                if depth == 0:
                    try:
                        results.append(json.loads(current_json))
                    except json.JSONDecodeError:
                        logging.warning("Could not parse JSON snippet: %s", current_json)
                    current_json = ""
            elif depth > 0:
                current_json += char

        return results

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _process_cluster(self, cluster_id, words):
        """Process a single cluster with LLM."""
        if cluster_id == "-1" or len(words) < 2 or len(words) > 50:
            return None

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": PromptBuilder.create_user_prompt(words),
                    },
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content
            usage = response.usage

            results = self._clean_results(self._parse_response(content))

            if not results:
                logging.error("Cluster %s: No results generated!", cluster_id)
                return None

            return {
                "results": results,
                "tokens": {
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens,
                },
            }

        except Exception as e:  # noqa: BLE001
            logging.error("Cluster %s: Error - %s", cluster_id, e)
            return None

    def _process_batch(self, batch_id, cluster_ids, all_clusters):
        """Worker function to process a batch of clusters."""
        batch_output_file = os.path.join(
            self.batch_output_dir, f"output_batch_{batch_id}.json"
        )
        batch_process_file = os.path.join(
            self.batch_output_dir, f"process_batch_{batch_id}.json"
        )

        output_data = self._load_json(batch_output_file, [])
        process_data = self._load_json(
            batch_process_file,
            {
                "last_cluster": None,
                "total_words": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
            },
        )

        if process_data["last_cluster"]:
            try:
                last_idx = cluster_ids.index(process_data["last_cluster"])
                cluster_ids = cluster_ids[last_idx + 1 :]
            except ValueError:
                pass

        pbar = tqdm(cluster_ids, desc=f"Batch {batch_id}", position=batch_id)
        for cluster_id in pbar:
            words = all_clusters[cluster_id]
            result = self._process_cluster(cluster_id, words)

            if result:
                output_data.extend(result["results"])
                process_data["last_cluster"] = cluster_id
                process_data["total_words"] += len(result["results"])
                process_data["total_input_tokens"] += result["tokens"]["input"]
                process_data["total_output_tokens"] += result["tokens"]["output"]
                process_data["total_tokens"] += result["tokens"]["total"]

                self._save_json(batch_output_file, output_data)
                self._save_json(batch_process_file, process_data)

                pbar.set_postfix(
                    {
                        "words": process_data["total_words"],
                        "tokens": process_data["total_tokens"],
                    }
                )

    def _merge_results(self):
        """Merge results from all batches."""
        logging.info("=== Merging Batch Results ===")

        all_output_data = []
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_words = 0

        output_files = glob.glob(
            os.path.join(self.batch_output_dir, "output_batch_*.json")
        )
        for f in output_files:
            data = self._load_json(f, [])
            all_output_data.extend(data)

        process_files = glob.glob(
            os.path.join(self.batch_output_dir, "process_batch_*.json")
        )
        for f in process_files:
            data = self._load_json(f, {})
            if data:
                total_input += data.get("total_input_tokens", 0)
                total_output += data.get("total_output_tokens", 0)
                total_tokens += data.get("total_tokens", 0)
                total_words += data.get("total_words", 0)

        self._save_json(self.output_file, all_output_data)

        logging.info("=== Aggregation Completed ===")
        logging.info("Total words generated: %s", total_words)
        logging.info(
            "Final Token Usage - Input: %s, Output: %s, Total: %s",
            total_input,
            total_output,
            total_tokens,
        )

    def run(self):
        """Main processing function."""
        logging.info("=== Starting Cluster Augmentation (Multiprocess) ===")

        os.makedirs(self.batch_output_dir, exist_ok=True)

        clusters = self._load_json(self.input_file, {})
        if not clusters:
            logging.error("Could not load or parse cluster file: %s", self.input_file)
            return

        cluster_ids = sorted([k for k in clusters.keys() if k != "-1"], key=int)
        logging.info("Loaded %s clusters to process.", len(cluster_ids))

        batch_size = math.ceil(len(cluster_ids) / self.num_processes)
        batches = [
            cluster_ids[i : i + batch_size]
            for i in range(0, len(cluster_ids), batch_size)
        ]

        logging.info(
            "Splitting into %s batches for %s processes.",
            len(batches),
            self.num_processes,
        )

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            pool.starmap(
                self._process_batch,
                [(i, batch, clusters) for i, batch in enumerate(batches)],
            )

        self._merge_results()

        logging.info("=== All Batches Processed ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Augment clusters with LLM using multiprocessing."
    )
    parser.add_argument("--model-name", required=True, help="LLM model name.")
    parser.add_argument(
        "--num-processes", required=True, type=int, help="Number of worker processes."
    )
    parser.add_argument("--input-file", required=True, help="Input clusters JSON file.")
    parser.add_argument(
        "--output-file", required=True, help="Output JSON for augmented clusters."
    )
    parser.add_argument(
        "--batch-output-dir",
        required=True,
        help="Directory to store per-batch intermediate results.",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for the LLM API (defaults to GEMINI_BASE_URL env).",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the LLM (defaults to GEMINI_API_KEY env).",
    )
    args = parser.parse_args()

    runner = MultiprocessorAugmentRunner(
        model_name=args.model_name,
        num_processes=args.num_processes,
        input_file=args.input_file,
        output_file=args.output_file,
        batch_output_dir=args.batch_output_dir,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    runner.run()
