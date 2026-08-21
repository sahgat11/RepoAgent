import json
from pathlib import Path

import numpy as np


BASE_INDEX_DIR = Path("data/index")


def get_index_dir(repository_id: str) -> Path:
    return BASE_INDEX_DIR / repository_id


def save_index(
    repository_id: str,
    chunks: list[dict],
    embeddings: np.ndarray,
) -> None:
    index_dir = get_index_dir(repository_id)

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_file = (
        index_dir / "embeddings.npy"
    )

    chunks_file = (
        index_dir / "chunks.json"
    )

    np.save(
        embeddings_file,
        embeddings,
    )

    with chunks_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            indent=2,
        )


def load_index(
    repository_id: str,
) -> tuple[list[dict], np.ndarray]:
    index_dir = get_index_dir(repository_id)

    embeddings_file = (
        index_dir / "embeddings.npy"
    )

    chunks_file = (
        index_dir / "chunks.json"
    )

    if (
        not embeddings_file.exists()
        or not chunks_file.exists()
    ):
        raise FileNotFoundError(
            "No saved index found."
        )

    embeddings = np.load(
        embeddings_file
    )

    with chunks_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    return chunks, embeddings


def index_exists(
    repository_id: str,
) -> bool:
    index_dir = get_index_dir(repository_id)

    return (
        (index_dir / "embeddings.npy").exists()
        and
        (index_dir / "chunks.json").exists()
    )