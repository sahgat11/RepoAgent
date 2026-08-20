import json
from pathlib import Path

import numpy as np


INDEX_DIR = Path("data/index")
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"


def save_index(chunks: list[dict], embeddings: np.ndarray) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, embeddings)

    with CHUNKS_FILE.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2)


def load_index() -> tuple[list[dict], np.ndarray]:
    if not EMBEDDINGS_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError("No saved index found.")

    embeddings = np.load(EMBEDDINGS_FILE)

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks, embeddings


def index_exists() -> bool:
    return EMBEDDINGS_FILE.exists() and CHUNKS_FILE.exists()