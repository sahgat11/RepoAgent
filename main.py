import sys
from pathlib import Path

from repoagent.loader import load_repository
from repoagent.chunker import chunk_repository


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <repo-path>")
        return

    repo_path = Path(sys.argv[1]).resolve()

    files = load_repository(str(repo_path))

    print(f"Found {len(files)} source files.")

    chunks = chunk_repository(
        files=files,
        repo_root=repo_path,
    )

    print(f"Created {len(chunks)} chunks.\n")

    for chunk in chunks[:5]:
        print(
            f"{chunk['path']} "
            f"[lines {chunk['start_line']}-{chunk['end_line']}]"
        )
        print(chunk["content"])
        print("-" * 60)


if __name__ == "__main__":
    main()