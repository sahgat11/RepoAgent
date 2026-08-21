import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

import argparse
from pathlib import Path

from repoagent.loader import load_repository
from repoagent.chunker import chunk_repository
from repoagent.embeddings import EmbeddingModel
from repoagent.index import (
    load_index,
    index_exists,
    save_index,
)
from repoagent.repo_source import (
    RepositorySource,
    get_repository_id,
)
from repoagent.tools import RepoTools


BENCHMARKS = [
    {
        "name": "Click",
        "repository": "https://github.com/pallets/click",
        "questions": [
            {
                "question": "Where is command parsing implemented?",
                "expected_paths": [
                    "src/click/parser.py",
                    "src/click/core.py",
                ],
            },
            {
                "question": "Where are command decorators implemented?",
                "expected_paths": [
                    "src/click/decorators.py",
                ],
            },
            {
                "question": "Where is option parsing handled?",
                "expected_paths": [
                    "src/click/parser.py",
                    "src/click/core.py",
                ],
            },
            {
                "question": "Where is terminal output handled?",
                "expected_paths": [
                    "src/click/_termui_impl.py",
                    "src/click/termui.py",
                ],
            },
            {
                "question": "Where are command line exceptions defined?",
                "expected_paths": [
                    "src/click/exceptions.py",
                ],
            },
            {
                "question": "Where is type conversion implemented?",
                "expected_paths": [
                    "src/click/types.py",
                ],
            },
        ],
    },
    {
        "name": "Requests",
        "repository": "https://github.com/psf/requests",
        "questions": [
            {
                "question": "Where are HTTP sessions implemented?",
                "expected_paths": [
                    "src/requests/sessions.py",
                ],
            },
            {
                "question": "Where are HTTP request and response objects defined?",
                "expected_paths": [
                    "src/requests/models.py",
                ],
            },
            {
                "question": "Where is HTTP authentication implemented?",
                "expected_paths": [
                    "src/requests/auth.py",
                ],
            },
            {
                "question": "Where are HTTP adapters implemented?",
                "expected_paths": [
                    "src/requests/adapters.py",
                ],
            },
            {
                "question": "Where are request exceptions defined?",
                "expected_paths": [
                    "src/requests/exceptions.py",
                ],
            },
            {
                "question": "Where are the high level request API functions implemented?",
                "expected_paths": [
                    "src/requests/api.py",
                ],
            },
        ],
    },
]


def build_tools(
    repo_path: Path,
    source: str,
    embedding_model: EmbeddingModel,
) -> RepoTools:
    repository_id = get_repository_id(source)

    if not index_exists(repository_id):
        print("No cached index found.")
        print("Building repository index...")

        files = load_repository(
            str(repo_path)
        )

        print(
            f"Found {len(files)} source files."
        )

        chunks = chunk_repository(
            files=files,
            repo_root=repo_path,
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        embeddings = embedding_model.embed_chunks(
            chunks
        )

        save_index(
            repository_id=repository_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        print(
            "Repository index saved."
        )

    else:
        chunks, embeddings = load_index(
            repository_id
        )

        print(
            f"Loaded cached index with "
            f"{len(chunks)} chunks."
        )

    return RepoTools(
        repo_path=repo_path,
        embedding_model=embedding_model,
        chunks=chunks,
        embeddings=embeddings,
    )


def unique_results(
    results: list[dict],
    top_k: int,
) -> list[dict]:
    unique = []
    seen_paths = set()

    for result in results:
        path = result["path"]

        if path in seen_paths:
            continue

        seen_paths.add(path)
        unique.append(result)

        if len(unique) == top_k:
            break

    return unique


def evaluate_question(
    tools: RepoTools,
    question: str,
    expected_paths: list[str],
    top_k: int,
) -> dict:
    # Retrieve extra chunks because multiple chunks
    # may come from the same source file.
    raw_results = tools.search_code(
        query=question,
        top_k=top_k * 4,
    )

    results = unique_results(
        raw_results,
        top_k,
    )

    retrieved_paths = [
        result["path"]
        for result in results
    ]

    matched_rank = None

    for rank, path in enumerate(
        retrieved_paths,
        start=1,
    ):
        if path in expected_paths:
            matched_rank = rank
            break

    return {
        "question": question,
        "expected_paths": expected_paths,
        "retrieved_paths": retrieved_paths,
        "matched_rank": matched_rank,
        "hit": matched_rank is not None,
    }


def evaluate_repository(
    benchmark: dict,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[dict]:
    repository_name = benchmark["name"]
    source = benchmark["repository"]

    print()
    print("=" * 70)
    print(
        f"Repository: {repository_name}"
    )
    print(
        f"Source: {source}"
    )
    print("=" * 70)

    results = []

    with RepositorySource(source) as repo_path:
        tools = build_tools(
            repo_path=repo_path,
            source=source,
            embedding_model=embedding_model,
        )

        print()
        print(
            f"Running "
            f"{len(benchmark['questions'])} "
            f"retrieval questions..."
        )
        print()

        for item in benchmark["questions"]:
            result = evaluate_question(
                tools=tools,
                question=item["question"],
                expected_paths=item["expected_paths"],
                top_k=top_k,
            )

            result["repository"] = (
                repository_name
            )

            results.append(result)

            print(
                f"Question: "
                f"{result['question']}"
            )

            print(
                "Expected: "
                + ", ".join(
                    result["expected_paths"]
                )
            )

            print("Retrieved:")

            for rank, path in enumerate(
                result["retrieved_paths"],
                start=1,
            ):
                marker = ""

                if path in result["expected_paths"]:
                    marker = "  <-- HIT"

                print(
                    f"  {rank}. "
                    f"{path}"
                    f"{marker}"
                )

            if result["hit"]:
                print(
                    f"Result: PASS "
                    f"(rank "
                    f"{result['matched_rank']})"
                )
            else:
                print(
                    "Result: FAIL"
                )

            print(
                "-" * 70
            )

    return results


def calculate_metrics(
    results: list[dict],
    top_k: int,
) -> dict:
    total = len(results)

    hits = sum(
        result["hit"]
        for result in results
    )

    recall_at_k = (
        hits / total
        if total
        else 0.0
    )

    reciprocal_ranks = []

    rank_one_hits = 0

    for result in results:
        rank = result["matched_rank"]

        if rank is None:
            reciprocal_ranks.append(
                0.0
            )
        else:
            reciprocal_ranks.append(
                1.0 / rank
            )

            if rank == 1:
                rank_one_hits += 1

    mrr = (
        sum(reciprocal_ranks) / total
        if total
        else 0.0
    )

    accuracy_at_1 = (
        rank_one_hits / total
        if total
        else 0.0
    )

    return {
        "questions": total,
        "hits": hits,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "accuracy_at_1": accuracy_at_1,
        "top_k": top_k,
    }


def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    print()
    print(title)
    print("=" * 70)

    print(
        f"Questions: "
        f"{metrics['questions']}"
    )

    print(
        f"Hits: "
        f"{metrics['hits']}/"
        f"{metrics['questions']}"
    )

    print(
        f"Accuracy@1: "
        f"{metrics['accuracy_at_1']:.2%}"
    )

    print(
        f"Recall@{metrics['top_k']}: "
        f"{metrics['recall_at_k']:.2%}"
    )

    print(
        f"MRR: "
        f"{metrics['mrr']:.3f}"
    )


def run_evaluation(
    top_k: int,
) -> None:
    print(
        "Loading embedding model..."
    )

    embedding_model = EmbeddingModel()

    all_results = []

    for benchmark in BENCHMARKS:
        repository_results = (
            evaluate_repository(
                benchmark=benchmark,
                embedding_model=embedding_model,
                top_k=top_k,
            )
        )

        all_results.extend(
            repository_results
        )

        repository_metrics = (
            calculate_metrics(
                repository_results,
                top_k,
            )
        )

        print_metrics(
            title=(
                f"{benchmark['name']} Summary"
            ),
            metrics=repository_metrics,
        )

    overall_metrics = calculate_metrics(
        all_results,
        top_k,
    )

    print()
    print()
    print_metrics(
        title="Overall Benchmark",
        metrics=overall_metrics,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RepoAgent retrieval "
            "across multiple repositories."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of unique files to "
            "evaluate. Default: 5"
        ),
    )

    args = parser.parse_args()

    run_evaluation(
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()