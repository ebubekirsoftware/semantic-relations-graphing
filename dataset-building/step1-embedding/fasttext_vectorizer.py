import json

import fasttext
import pandas as pd
from unicode_tr import unicode_tr


class FastTextVectorizerRunner:
    """
    Wraps FastText term vectorization into a reusable, importable runner.
    Paths and parameters are provided at construction time to keep the class
    free of hard-coded I/O defaults.
    """

    def __init__(self, model_path: str, input_file: str, output_file: str):
        self.model_path = model_path
        self.input_file = input_file
        self.output_file = output_file
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = fasttext.load_model(self.model_path)
        return self._model

    @staticmethod
    def _normalize_term(term: str) -> str:
        """Lowercase and strip simple punctuation for more consistent vectors."""
    normalized = unicode_tr(term).lower()
        for char in [".", ",", ";", "\n"]:
            normalized = normalized.replace(char, " ")
    return normalized.strip()

    def _get_term_vector(self, term: str):
        model = self._load_model()
        return model.get_sentence_vector(self._normalize_term(term))

    def _load_terms(self):
    """Load terms from JSON file (supports both list and dict formats)."""
        with open(self.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
        if isinstance(data, dict):
            return data.get("terms", data.get("remaining", []))
    return []

    def run(self):
        terms = self._load_terms()
        embeddings = [self._get_term_vector(str(term)) for term in terms]
    
        result = pd.DataFrame(
            {
                "words": terms,
                "embedding": [emb.tolist() for emb in embeddings],
            }
        )
    
        result.to_csv(self.output_file, index=False)
        print(f"Processed {len(embeddings)} terms -> {self.output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate FastText embeddings for provided terms."
    )
    parser.add_argument("--model-path", required=True, help="Path to FastText model.")
    parser.add_argument("--input-file", required=True, help="JSON file of terms.")
    parser.add_argument(
        "--output-file",
        required=True,
        help="CSV path to write embeddings.",
    )
    args = parser.parse_args()

    runner = FastTextVectorizerRunner(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=args.output_file,
    )
    runner.run()