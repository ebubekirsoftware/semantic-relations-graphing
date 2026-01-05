"""
Central entry point to run the full synonym framework pipeline.

Each step runner is instantiated here with explicit paths/parameters so
individual modules remain importable and free of hard-coded I/O.
"""

import importlib.util
from pathlib import Path

from config import config

PROJECT_ROOT = Path(__file__).parent
STEP1_EMBEDDING_DIR = PROJECT_ROOT / "dataset-building" / "step1-embedding"
STEP2_CLUSTERING_DIR = PROJECT_ROOT / "dataset-building" / "step2-clustering"
STEP3_AUGMENT_DIR = PROJECT_ROOT / "dataset-building" / "step3-llm-augmention"
STEP4_DATA_DIR = PROJECT_ROOT / "dataset-building" / "step4-data_collector-and-analyze"
STEP_SYNG_BUILD_DIR = PROJECT_ROOT / "synonym-graph-building"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {module_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fasttext_vectorizer = _load_module(
    "fasttext_vectorizer", STEP1_EMBEDDING_DIR / "fasttext_vectorizer.py"
)
FastTextVectorizerRunner = fasttext_vectorizer.FastTextVectorizerRunner

embedding_cluster = _load_module(
    "embedding_cluster", STEP2_CLUSTERING_DIR / "embedding_cluster.py"
)
EmbeddingClusterRunner = embedding_cluster.EmbeddingClusterRunner

post_processing_cluster = _load_module(
    "post_processing_cluster", STEP2_CLUSTERING_DIR / "post_processing_cluster.py"
)
ClusterPostProcessorRunner = post_processing_cluster.ClusterPostProcessorRunner

multiprocessor_augment = _load_module(
    "multiprocessor_augment", STEP3_AUGMENT_DIR / "multiprocessor_augment.py"
)
MultiprocessorAugmentRunner = multiprocessor_augment.MultiprocessorAugmentRunner

dataset_format = _load_module(
    "dataset_format", STEP4_DATA_DIR / "dataset_format.py"
)
DatasetFormatRunner = dataset_format.DatasetFormatRunner

dataset_analyzer = _load_module(
    "dataset_analyzer", STEP4_DATA_DIR / "dataset_analyzer.py"
)
DatasetAnalyzerRunner = dataset_analyzer.DatasetAnalyzerRunner

step1_get_embeddings = _load_module(
    "step1_get_embeddings", STEP_SYNG_BUILD_DIR / "step1-get_embeddings.py"
)
EmbeddingGeneratorRunner = step1_get_embeddings.EmbeddingGeneratorRunner

step2_search_candidates = _load_module(
    "step2_search_candidates", STEP_SYNG_BUILD_DIR / "step2-search_candidates.py"
)
SynonymCandidateSearchRunner = step2_search_candidates.SynonymCandidateSearchRunner

step3_classification = _load_module(
    "step3_classification", STEP_SYNG_BUILD_DIR / "step3-classification.py"
)
SynonymClassifierRunner = step3_classification.SynonymClassifierRunner

step4_clustering = _load_module(
    "step4_clustering", STEP_SYNG_BUILD_DIR / "step4-clustering.py"
)
SynonymClustererStep1Runner = step4_clustering.SynonymClustererStep1Runner

step5_pruning = _load_module(
    "step5_prunining", STEP_SYNG_BUILD_DIR / "step5-prunining.py"
)
SynonymClustererStep2Runner = step5_pruning.SynonymClustererStep2Runner

step6_parent_chooser = _load_module(
    "step6_parent_chooser", STEP_SYNG_BUILD_DIR / "step6-parent_chooser.py"
)
ParentChooserRunner = step6_parent_chooser.ParentChooserRunner

step7_parent_child = _load_module(
    "step7_parent_child", STEP_SYNG_BUILD_DIR / "step7-parent_child.py"
)
ParentChildEnricherRunner = step7_parent_child.ParentChildEnricherRunner


def run_fasttext_vectorization():
    step_cfg = config["steps"]["fasttext_vectorization"]
    model_path = step_cfg["model_path"]
    input_file = step_cfg["input_file"]
    output_file = step_cfg["output_file"]

    runner = FastTextVectorizerRunner(
        model_path=model_path,
        input_file=input_file,
        output_file=output_file,
    )
    runner.run()


def run_embedding_clustering():
    step_cfg = config["steps"]["embedding_clustering"]
    input_file = step_cfg["input_file"]
    output_file = step_cfg["output_file"]
    distance_threshold = step_cfg["distance_threshold"]

    runner = EmbeddingClusterRunner(
        input_file=input_file,
        output_file=output_file,
        distance_threshold=distance_threshold,
    )
    runner.run()


def run_cluster_postprocessing():
    step_cfg = config["steps"]["cluster_postprocessing"]
    input_file = step_cfg["input_file"]
    output_file = step_cfg["output_file"]
    use_substring_filter = step_cfg["use_substring_filter"]

    runner = ClusterPostProcessorRunner(
        input_file=input_file,
        output_file=output_file,
        use_substring_filter=use_substring_filter,
    )
    runner.run()


def run_llm_augmentation():
    step_cfg = config["steps"]["llm_augmentation"]
    model_name = step_cfg["model_name"]
    num_processes = step_cfg["num_processes"]
    input_file = step_cfg["input_file"]
    output_file = step_cfg["output_file"]
    batch_output_dir = step_cfg["batch_output_dir"]

    runner = MultiprocessorAugmentRunner(
        model_name=model_name,
        num_processes=num_processes,
        input_file=input_file,
        output_file=output_file,
        batch_output_dir=batch_output_dir,
        base_url=config.get("gemini_base_url"),
        api_key=config.get("gemini_api_key"),
    )
    runner.run()


def run_dataset_formatting():
    step_cfg = config["steps"]["dataset_formatting"]
    input_file = step_cfg["input_file"]
    output_ce = step_cfg["output_ce"]
    output_contrastive = step_cfg["output_contrastive"]

    runner = DatasetFormatRunner(
        input_file=input_file,
        output_ce=output_ce,
        output_contrastive=output_contrastive,
    )
    runner.run()


def run_dataset_analysis():
    step_cfg = config["steps"]["dataset_analysis"]
    input_file = step_cfg["input_file"]
    expected_labels = step_cfg["expected_labels"]
    directional_dup_check = step_cfg["directional_dup_check"]

    runner = DatasetAnalyzerRunner(
        input_file=input_file,
        expected_labels=expected_labels,
        directional_dup_check=directional_dup_check,
    )
    runner.run()


def run_synonym_graph_embeddings():
    step_cfg = config["steps"]["synonym_graph"]["embeddings"]
    model_path = step_cfg["model_path"]
    input_parquet = step_cfg["input_parquet"]
    output_parquet = step_cfg["output_parquet"]
    num_gpus = step_cfg["num_gpus"]
    batch_size = step_cfg["batch_size"]

    runner = EmbeddingGeneratorRunner(
        model_path=model_path,
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    runner.run()


def run_synonym_graph_search_candidates():
    step_cfg = config["steps"]["synonym_graph"]["search_candidates"]
    input_path = step_cfg["input_path"]
    work_dir = step_cfg["work_dir"]
    output_dir = step_cfg["output_dir"]
    id_col = step_cfg["id_col"]
    emb_col = step_cfg["emb_col"]
    default_dim = step_cfg["default_dim"]
    top_k = step_cfg["top_k"]
    threshold = step_cfg["threshold"]
    batch_size_search = step_cfg["batch_size_search"]
    ivf_nlist = step_cfg["ivf_nlist"]
    ivf_nprobe = step_cfg["ivf_nprobe"]

    runner = SynonymCandidateSearchRunner(
        input_path=input_path,
        work_dir=work_dir,
        output_dir=output_dir,
        id_col=id_col,
        emb_col=emb_col,
        default_dim=default_dim,
        top_k=top_k,
        threshold=threshold,
        batch_size_search=batch_size_search,
        ivf_nlist=ivf_nlist,
        ivf_nprobe=ivf_nprobe,
    )
    runner.run()


def run_synonym_graph_classification():
    step_cfg = config["steps"]["synonym_graph"]["classification"]
    model_path = step_cfg["model_path"]
    input_jsonl = step_cfg["input_jsonl"]
    output_dir = step_cfg["output_dir"]
    batch_size = step_cfg["batch_size"]
    chunk_size = step_cfg["chunk_size"]
    confidence_threshold = step_cfg["confidence_threshold"]
    num_workers = step_cfg["num_workers"]
    device = step_cfg["device"]

    runner = SynonymClassifierRunner(
        model_path=model_path,
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        batch_size=batch_size,
        chunk_size=chunk_size,
        confidence_threshold=confidence_threshold,
        num_workers=num_workers,
        device=device,
    )
    runner.run()


def run_synonym_graph_cluster_step1():
    step_cfg = config["steps"]["synonym_graph"]["cluster_step1"]
    input_file = step_cfg["input_file"]
    checkpoint_file = step_cfg["checkpoint_file"]
    threshold_ratio = step_cfg["threshold_ratio"]

    runner = SynonymClustererStep1Runner(
        input_file=input_file,
        checkpoint_file=checkpoint_file,
        threshold_ratio=threshold_ratio,
    )
    runner.run()


def run_synonym_graph_cluster_step2():
    step_cfg = config["steps"]["synonym_graph"]["cluster_step2"]
    checkpoint_file = step_cfg["checkpoint_file"]
    output_file = step_cfg["output_file"]
    n_jobs = step_cfg["n_jobs"]
    chunk_size = step_cfg["chunk_size"]
    batch_size = step_cfg["batch_size"]

    runner = SynonymClustererStep2Runner(
        checkpoint_file=checkpoint_file,
        output_file=output_file,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )
    runner.run()


def run_synonym_graph_parent_chooser():
    step_cfg = config["steps"]["synonym_graph"]["parent_chooser"]
    embeddings_path = step_cfg["embeddings_path"]
    clusters_file = step_cfg["clusters_file"]
    output_file = step_cfg["output_file"]
    num_calc_workers = step_cfg["num_calc_workers"]
    num_read_workers = step_cfg["num_read_workers"]
    chunk_size = step_cfg["chunk_size"]

    runner = ParentChooserRunner(
        embeddings_path=embeddings_path,
        clusters_file=clusters_file,
        output_file=output_file,
        num_calc_workers=num_calc_workers,
        num_read_workers=num_read_workers,
        chunk_size=chunk_size,
    )
    runner.run()


def run_synonym_graph_parent_child():
    step_cfg = config["steps"]["synonym_graph"]["parent_child"]
    parquet_path = step_cfg["parquet_path"]
    clusters_path = step_cfg["clusters_path"]
    output_path = step_cfg["output_path"]
    batch_size = step_cfg["batch_size"]

    runner = ParentChildEnricherRunner(
        parquet_path=parquet_path,
        clusters_path=clusters_path,
        output_path=output_path,
        batch_size=batch_size,
    )
    runner.run()


if __name__ == "__main__":
    run_fasttext_vectorization()
    run_embedding_clustering()
    run_cluster_postprocessing()
    run_llm_augmentation()
    run_dataset_formatting()
    run_dataset_analysis()
    run_synonym_graph_embeddings()
    run_synonym_graph_search_candidates()
    run_synonym_graph_classification()
    run_synonym_graph_cluster_step1()
    run_synonym_graph_cluster_step2()
    run_synonym_graph_parent_chooser()
    run_synonym_graph_parent_child()

