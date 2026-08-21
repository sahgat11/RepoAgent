import sys
from pathlib import Path

from repoagent.loader import load_repository
from repoagent.chunker import chunk_repository
from repoagent.embeddings import EmbeddingModel
from repoagent.index import save_index, load_index, index_exists
from repoagent.tools import RepoTools
from repoagent.agent import RepoAgent


def build_index(
    repo_path: Path,
    embedding_model: EmbeddingModel,
):
    files = load_repository(str(repo_path))
    print(f"Found {len(files)} source files.")

    chunks = chunk_repository(
        files=files,
        repo_root=repo_path,
    )

    print(f"Created {len(chunks)} chunks.")

    embeddings = embedding_model.embed_chunks(chunks)

    print(f"Generated {len(embeddings)} embeddings.")

    save_index(chunks, embeddings)

    print("Saved index to data/index/")

    return chunks, embeddings


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <repo-path>")
        return

    repo_path = Path(sys.argv[1]).resolve()

    embedding_model = EmbeddingModel()

    if index_exists():
        print("Loading existing index...")

        chunks, embeddings = load_index()

        print(f"Loaded {len(chunks)} chunks.")
    else:
        print("No index found. Building one...")

        chunks, embeddings = build_index(
            repo_path=repo_path,
            embedding_model=embedding_model,
        )

    tools = RepoTools(
        repo_path=repo_path,
        embedding_model=embedding_model,
        chunks=chunks,
        embeddings=embeddings,
    )

    agent = RepoAgent(tools)

    print("\nRepoAgent ready.")

    while True:
        query = input(
            "\nAsk about the repository "
            "(or type 'exit'): "
        )

        if query.lower() in {"exit", "quit"}:
            break

        print("\nThinking...")

        answer = agent.run(query)

        print("\nRepoAgent:\n")
        print(answer)


if __name__ == "__main__":
    main()