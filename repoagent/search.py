import re

import numpy as np


CORE_PATH_HINTS = {
    "src": 0.08,
    "lib": 0.06,
    "app": 0.05,
    "core": 0.05,
}

LOW_PRIORITY_PATH_HINTS = {
    "examples": -0.10,
    "example": -0.10,
    "tests": -0.08,
    "test": -0.08,
    "docs": -0.06,
    "documentation": -0.06,
    "benchmarks": -0.04,
}


def _path_adjustment(path: str) -> float:
    parts = path.lower().split("/")

    adjustment = 0.0

    for part in parts:
        adjustment += CORE_PATH_HINTS.get(part, 0.0)
        adjustment += LOW_PRIORITY_PATH_HINTS.get(part, 0.0)

    return adjustment


def _tokenize(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*",
            text.lower(),
        )
    )


def _keyword_score(
    query: str,
    chunk: dict,
) -> float:
    query_tokens = _tokenize(query)

    if not query_tokens:
        return 0.0

    path_tokens = _tokenize(
        chunk["path"]
    )

    symbol_tokens = _tokenize(
        chunk.get("symbol") or ""
    )

    content_tokens = _tokenize(
        chunk["content"]
    )

    score = 0.0

    for token in query_tokens:
        if token in symbol_tokens:
            score += 0.10

        if token in path_tokens:
            score += 0.08

        if token in content_tokens:
            score += 0.02

    return min(score, 0.20)


def semantic_search(
    query: str,
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    if len(chunks) == 0:
        return []

    semantic_scores = (
        embeddings @ query_embedding
    )

    ranked_results = []

    for index, chunk in enumerate(chunks):
        semantic_score = float(
            semantic_scores[index]
        )

        path_score = _path_adjustment(
            chunk["path"]
        )

        keyword_score = _keyword_score(
            query,
            chunk,
        )

        final_score = (
            semantic_score
            + path_score
            + keyword_score
        )

        result = chunk.copy()

        result["semantic_score"] = semantic_score
        result["path_adjustment"] = path_score
        result["keyword_score"] = keyword_score
        result["score"] = final_score

        ranked_results.append(result)

    ranked_results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return ranked_results[:top_k]