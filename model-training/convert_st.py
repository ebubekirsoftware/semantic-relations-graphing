#!/usr/bin/env python3
"""
Convert HuggingFace model to Sentence Transformers format.

This script adds necessary Sentence Transformers config files to a trained model,
enabling direct loading with SentenceTransformer without warnings.
"""

import os
import json
import argparse


# Configuration
DEFAULT_MODEL_DIR = "models/embedding"
MAX_SEQ_LENGTH = 512
WORD_EMBEDDING_DIM = 1024


def create_sentence_bert_config(max_seq_length=MAX_SEQ_LENGTH):
    """Create sentence_bert_config.json configuration."""
    return {
        "max_seq_length": max_seq_length,
        "do_lower_case": False
    }


def create_modules_config():
    """Create modules.json configuration."""
    return [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer"
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling"
        },
        {
            "idx": 2,
            "name": "2",
            "path": "2_Normalize",
            "type": "sentence_transformers.models.Normalize"
        }
    ]


def create_pooling_config(embedding_dim=WORD_EMBEDDING_DIM):
    """Create pooling configuration."""
    return {
        "word_embedding_dimension": embedding_dim,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False
    }


def create_normalize_config():
    """Create normalization configuration."""
    return {}


def validate_model_path(model_path):
    """Validate that model path exists and contains necessary files."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    required_files = ["config.json", "pytorch_model.bin"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_files:
        print(f"⚠️ Warning: Missing files: {', '.join(missing_files)}")
        print("   Model might not be complete. Continuing anyway...")


def add_sentence_transformers_config(model_path, max_seq_length=MAX_SEQ_LENGTH, 
                                     embedding_dim=WORD_EMBEDDING_DIM, verify=True):
    """
    Add Sentence Transformers configuration to existing HuggingFace model.
    
    Args:
        model_path: Path to the model directory
        max_seq_length: Maximum sequence length for the model
        embedding_dim: Word embedding dimension
        verify: Whether to verify the conversion by loading the model
    """
    print("=" * 80)
    print("Converting HuggingFace Model to Sentence Transformers Format")
    print("=" * 80)
    print(f"\nModel Path: {model_path}")
    
    # Validate model
    validate_model_path(model_path)
    
    # Create sentence_bert_config.json
    print("\n[1/4] Creating sentence_bert_config.json...")
    config_path = os.path.join(model_path, "sentence_bert_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(create_sentence_bert_config(max_seq_length), f, indent=2)
    print(f"✓ Created: {config_path}")
    
    # Create modules.json
    print("\n[2/4] Creating modules.json...")
    modules_path = os.path.join(model_path, "modules.json")
    with open(modules_path, 'w', encoding='utf-8') as f:
        json.dump(create_modules_config(), f, indent=2)
    print(f"✓ Created: {modules_path}")
    
    # Create pooling config
    print("\n[3/4] Creating pooling configuration...")
    pooling_dir = os.path.join(model_path, "1_Pooling")
    os.makedirs(pooling_dir, exist_ok=True)
    
    pooling_config_path = os.path.join(pooling_dir, "config.json")
    with open(pooling_config_path, 'w', encoding='utf-8') as f:
        json.dump(create_pooling_config(embedding_dim), f, indent=2)
    print(f"✓ Created: {pooling_config_path}")
    
    # Create normalization config
    print("\n[4/4] Creating normalization configuration...")
    normalize_dir = os.path.join(model_path, "2_Normalize")
    os.makedirs(normalize_dir, exist_ok=True)
    
    normalize_config_path = os.path.join(normalize_dir, "config.json")
    with open(normalize_config_path, 'w', encoding='utf-8') as f:
        json.dump(create_normalize_config(), f, indent=2)
    print(f"✓ Created: {normalize_config_path}")
    
    # Verify conversion
    if verify:
        verify_conversion(model_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"\nModel is now ready at: {model_path}")
    print("\nAdded files:")
    print("  - sentence_bert_config.json")
    print("  - modules.json")
    print("  - 1_Pooling/config.json")
    print("  - 2_Normalize/config.json")
    print("\nUsage:")
    print("```python")
    print("from sentence_transformers import SentenceTransformer")
    print(f"model = SentenceTransformer('{model_path}')")
    print("embeddings = model.encode(['text1', 'text2'])")
    print("```")


def verify_conversion(model_path):
    """Verify the conversion by loading and testing the model."""
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        print("\nLoading model...")
        model = SentenceTransformer(model_path)
        print("✓ Model loaded successfully!")
        
        # Test encoding
        print("\nTesting encoding...")
        test_texts = ["test sentence", "another test"]
        embeddings = model.encode(test_texts, convert_to_tensor=True, normalize_embeddings=True)
        print(f"✓ Encoding successful! Shape: {embeddings.shape}")
        
        # Test similarity
        print("\nTesting similarity computation...")
        similarity = torch.mm(embeddings, embeddings.t())
        print("✓ Similarity calculation successful!")
        print(f"\nSimilarity matrix:\n{similarity}")
        
    except ImportError as e:
        print(f"⚠️ Verification skipped: {e}")
        print("   Install sentence-transformers to verify: pip install sentence-transformers")
    except Exception as e:
        print(f"❌ Verification error: {e}")
        print("   The model files were created but verification failed.")


def find_model_directories(base_dir=DEFAULT_MODEL_DIR):
    """Find all model directories in the base directory."""
    if not os.path.exists(base_dir):
        return []
    
    model_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a valid model directory
            if os.path.exists(os.path.join(item_path, "config.json")):
                model_dirs.append(item_path)
    
    return model_dirs


def main():
    """Main conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to Sentence Transformers format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory (if not provided, will list available models)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help=f"Maximum sequence length (default: {MAX_SEQ_LENGTH})"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=WORD_EMBEDDING_DIM,
        help=f"Word embedding dimension (default: {WORD_EMBEDDING_DIM})"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step"
    )
    
    args = parser.parse_args()
    
    # If no model path provided, list available models
    if not args.model_path:
        print("No model path provided. Searching for models...\n")
        model_dirs = find_model_directories()
        
        if not model_dirs:
            print(f"❌ No models found in {DEFAULT_MODEL_DIR}")
            print("\nPlease specify a model path with --model-path")
            return
        
        print("Available models:")
        for i, model_dir in enumerate(model_dirs, 1):
            print(f"  {i}. {model_dir}")
        
        print("\nPlease specify a model path with --model-path")
        return
    
    # Convert model
    try:
        add_sentence_transformers_config(
            model_path=args.model_path,
            max_seq_length=args.max_seq_length,
            embedding_dim=args.embedding_dim,
            verify=not args.no_verify
        )
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

