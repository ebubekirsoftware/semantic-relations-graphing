# Semantic Relations Graphing Framework

An end-to-end framework for building large-scale semantic relation datasets and training synonym detection models. This repository contains all code used in our research on Turkish semantic relations.

## Papers

This framework supports two research papers:

1. **Beyond Cosine Similarity: Taming Semantic Drift and Antonym Intrusion in a 15-Million Node Turkish Synonym Graph**  
   *Describes the synonym graph construction pipeline and model training.*

2. **A Hybrid Protocol for Large-Scale Semantic Dataset Generation in Low-Resource Languages: The Turkish Semantic Relations Corpus**  
   *Details the dataset generation methodology combining clustering and LLM augmentation.*

## Overview

The framework consists of three main components:

| Component | Description |
|-----------|-------------|
| **Dataset Building** | FastText embedding → Clustering → LLM augmentation → Dataset formatting |
| **Model Training** | Contrastive embedding model + Classification model |
| **Synonym Graph Building** | Embedding generation → Candidate search → Classification → Clustering → Parent assignment |

## Project Structure

```
semantic-relations-graphing/
├── dataset-building/
│   ├── step1-embedding/          # FastText vectorization
│   ├── step2-clustering/         # Agglomerative clustering
│   ├── step3-llm-augmention/     # LLM-based semantic augmentation
│   └── step4-data_collector-and-analyze/
├── model-training/
│   ├── train_embedding.py        # Contrastive learning model
│   └── train_classification.py   # Sequence classification model
├── synonym-graph-building/
│   ├── step1-get_embeddings.py   # Generate embeddings
│   ├── step2-search_candidates.py # FAISS neighbor search
│   ├── step3-classification.py   # Pair classification
│   ├── step4-clustering.py       # Soft clustering
│   ├── step5-prunining.py        # Ambiguity resolution
│   ├── step6-parent_chooser.py   # Centroid-based parent selection
│   └── step7-parent_child.py     # Final enrichment
├── end2end_run.py                # Pipeline orchestrator
├── config.json                   # Configuration
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/ebubekirsoftware/semantic-relations-graphing.git
cd semantic-relations-graphing
pip install -r requirements.txt
```

## Configuration

Edit `config.json` with your paths and API keys:

```json
{
  "wandb_api_key": "YOUR_WANDB_API_KEY_HERE",
  "gemini_base_url": "YOUR_GEMINI_BASE_URL_HERE",
  "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE",
  "steps": { ... }
}
```

## Usage

### Full Pipeline

```bash
python end2end_run.py
```

### Individual Steps

Each module can be run standalone via CLI:

```bash
# Dataset building
python dataset-building/step1-embedding/fasttext_vectorizer.py \
    --model-path cc_tr_300.bin \ #Facebook Turkish Fasttext Model
    --input-file terms.json \
    --output-file embeddings.csv

# Model training
python model-training/train_embedding.py
python model-training/train_classification.py

# Graph building
python synonym-graph-building/step1-get_embeddings.py --help
```

### Programmatic Usage

```python
from dataset-building.step1-embedding.fasttext_vectorizer import FastTextVectorizerRunner

runner = FastTextVectorizerRunner(
    model_path="cc_tr_300.bin",
    input_file="terms.json",
    output_file="embeddings.csv"
)
runner.run()
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- See `requirements.txt` for full dependencies

## Citation

If you use this framework, please cite our papers:

```bibtex
@article{
  title={Beyond Cosine Similarity: Taming Semantic Drift and Antonym Intrusion in a 15-Million Node Turkish Synonym Graph},
  author={Tosun, Ezerceli,...},
  year={2025}
}

@article{
  title={A Hybrid Protocol for Large-Scale Semantic Dataset Generation in Low-Resource Languages: The Turkish Semantic Relations Corpus},
  author={Tosun, Ezerceli,...},
  year={2025}
}
```
