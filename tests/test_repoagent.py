from pathlib import Path

import numpy as np

from repoagent.chunker import chunk_file, chunk_repository
from repoagent.index import save_index, load_index
from repoagent.loader import load_repository
from repoagent.search import semantic_search
from repoagent.tools import RepoTools


class DummyEmbeddingModel:
    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0])


def test_loader_finds_supported_files(tmp_path):
    (tmp_path / "main.py").write_text(
        "print('hello')",
        encoding="utf-8",
    )

    (tmp_path / "script.js").write_text(
        "console.log('hello')",
        encoding="utf-8",
    )

    (tmp_path / "notes.txt").write_text(
        "ignore me",
        encoding="utf-8",
    )

    files = load_repository(
        str(tmp_path)
    )

    relative_paths = {
        file.relative_to(tmp_path).as_posix()
        for file in files
    }

    assert "main.py" in relative_paths
    assert "script.js" in relative_paths
    assert "notes.txt" not in relative_paths


def test_loader_ignores_ignored_directories(
    tmp_path,
):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()

    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (git_dir / "hidden.py").write_text(
        "print('ignore')",
        encoding="utf-8",
    )

    (venv_dir / "dependency.py").write_text(
        "print('ignore')",
        encoding="utf-8",
    )

    (src_dir / "app.py").write_text(
        "print('keep')",
        encoding="utf-8",
    )

    files = load_repository(
        str(tmp_path)
    )

    relative_paths = {
        file.relative_to(tmp_path).as_posix()
        for file in files
    }

    assert relative_paths == {
        "src/app.py"
    }


def test_python_chunking_uses_ast(
    tmp_path,
):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        """
def hello():
    return "hello"


class Greeter:
    def greet(self):
        return "hi"
""".strip(),
        encoding="utf-8",
    )

    chunks = chunk_file(
        file_path=file_path,
        repo_root=tmp_path,
    )

    symbols = {
        chunk["symbol"]
        for chunk in chunks
    }

    assert "hello" in symbols
    assert "Greeter" in symbols

    chunk_types = {
        chunk["type"]
        for chunk in chunks
    }

    assert "function" in chunk_types
    assert "class" in chunk_types


def test_non_python_file_uses_line_chunking(
    tmp_path,
):
    file_path = tmp_path / "example.js"

    file_path.write_text(
        "\n".join(
            f"const value{i} = {i};"
            for i in range(100)
        ),
        encoding="utf-8",
    )

    chunks = chunk_file(
        file_path=file_path,
        repo_root=tmp_path,
        chunk_size=40,
        overlap=10,
    )

    assert len(chunks) > 1

    assert all(
        chunk["type"] == "line_chunk"
        for chunk in chunks
    )


def test_chunk_repository_combines_files(
    tmp_path,
):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"

    first.write_text(
        "def first():\n    return 1",
        encoding="utf-8",
    )

    second.write_text(
        "def second():\n    return 2",
        encoding="utf-8",
    )

    chunks = chunk_repository(
        files=[first, second],
        repo_root=tmp_path,
    )

    symbols = {
        chunk["symbol"]
        for chunk in chunks
    }

    assert symbols == {
        "first",
        "second",
    }


def test_index_save_and_load(
    tmp_path,
    monkeypatch,
):
    import repoagent.index as index_module

    monkeypatch.setattr(
        index_module,
        "BASE_INDEX_DIR",
        tmp_path,
    )

    chunks = [
        {
            "path": "main.py",
            "start_line": 1,
            "end_line": 2,
            "content": "def main(): pass",
            "symbol": "main",
            "type": "function",
        }
    ]

    embeddings = np.array(
        [
            [0.5, 0.5],
        ],
        dtype=float,
    )

    save_index(
        repository_id="test-repo",
        chunks=chunks,
        embeddings=embeddings,
    )

    loaded_chunks, loaded_embeddings = (
        load_index("test-repo")
    )

    assert loaded_chunks == chunks

    assert np.array_equal(
        loaded_embeddings,
        embeddings,
    )


def test_semantic_search_returns_best_match():
    chunks = [
        {
            "path": "src/parser.py",
            "start_line": 1,
            "end_line": 10,
            "content": (
                "parse command line arguments"
            ),
            "symbol": "parse",
            "type": "function",
        },
        {
            "path": "examples/demo.py",
            "start_line": 1,
            "end_line": 10,
            "content": (
                "example application"
            ),
            "symbol": "demo",
            "type": "function",
        },
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    query_embedding = np.array(
        [1.0, 0.0],
        dtype=float,
    )

    results = semantic_search(
        query="command parsing",
        query_embedding=query_embedding,
        embeddings=embeddings,
        chunks=chunks,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["path"] == (
        "src/parser.py"
    )


def test_read_file_blocks_path_traversal(
    tmp_path,
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    inside_file = repo_dir / "main.py"

    inside_file.write_text(
        "print('inside')",
        encoding="utf-8",
    )

    outside_file = tmp_path / "secret.py"

    outside_file.write_text(
        "print('outside')",
        encoding="utf-8",
    )

    chunks = []

    embeddings = np.empty(
        (0, 2)
    )

    tools = RepoTools(
        repo_path=repo_dir,
        embedding_model=DummyEmbeddingModel(),
        chunks=chunks,
        embeddings=embeddings,
    )

    inside_result = tools.read_file(
        "main.py"
    )

    outside_result = tools.read_file(
        "../secret.py"
    )

    assert "inside" in inside_result

    assert outside_result.startswith(
        "Invalid path outside repository"
    )