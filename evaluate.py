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
from repoagent.index import load_index, index_exists, save_index
from repoagent.repo_source import RepositorySource, get_repository_id
from repoagent.tools import RepoTools


EVALUATION_QUESTIONS = [
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
]


def build_tools(
    repo_path: Path,
    source: str,
) -> RepoTools:
    repository_id = get_repository_id(source)
    embedding_model = EmbeddingModel()

    if not index_exists(repository_id):
        print("No cached index found.")
        print("Building repository index...")

        files = load_repository(str(repo_path))

        chunks = chunk_repository(
            files=files,
            repo_root=repo_path,
        )

        embeddings = embedding_model.embed_chunks(
            chunks
        )

        save_index(
            repository_id=repository_id,
            chunks=chunks,
            embeddings=embeddings,
        )

    chunks, embeddings = load_index(
        repository_id
    )

    return RepoTools(
        repo_path=repo_path,
        embedding_model=embedding_model,
        chunks=chunks,
        embeddings=embeddings,
    )


def evaluate_question(
    tools: RepoTools,
    question: str,
    expected_paths: list[str],
    top_k: int,
) -> dict:
    results = tools.search_code(
        query=question,
        top_k=top_k,
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


def run_evaluation(
    source: str,
    top_k: int,
) -> None:
    with RepositorySource(source) as repo_path:
        tools = build_tools(
            repo_path=repo_path,
            source=source,
        )

        results = []

        print()
        print(
            f"Running {len(EVALUATION_QUESTIONS)} "
            f"retrieval questions..."
        )
        print()

        for item in EVALUATION_QUESTIONS:
            result = evaluate_question(
                tools=tools,
                question=item["question"],
                expected_paths=item["expected_paths"],
                top_k=top_k,
            )

            results.append(result)

            print(
                f"Question: {result['question']}"
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
                    f"  {rank}. {path}{marker}"
                )

            if result["hit"]:
                print(
                    f"Result: PASS "
                    f"(rank {result['matched_rank']})"
                )
            else:
                print("Result: FAIL")

            print("-" * 60)

        hits = sum(
            result["hit"]
            for result in results
        )

        total = len(results)

        recall_at_k = (
            hits / total
            if total
            else 0.0
        )

        reciprocal_ranks = []

        for result in results:
            rank = result["matched_rank"]

            if rank is None:
                reciprocal_ranks.append(0.0)
            else:
                reciprocal_ranks.append(
                    1.0 / rank
                )

        mrr = (
            sum(reciprocal_ranks)
            / total
            if total
            else 0.0
        )

        print()
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Questions: {total}")
        print(f"Hits: {hits}/{total}")
        print(
            f"Recall@{top_k}: "
            f"{recall_at_k:.2%}"
        )
        print(f"MRR: {mrr:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RepoAgent retrieval quality."
        )
    )

    parser.add_argument(
        "repository",
        help=(
            "Local repository path or "
            "public GitHub URL."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of retrieval results "
            "to evaluate. Default: 5"
        ),
    )

    args = parser.parse_args()

    run_evaluation(
        source=args.repository,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()