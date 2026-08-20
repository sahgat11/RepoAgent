import ast
from pathlib import Path


def chunk_file(
    file_path: Path,
    repo_root: Path,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[dict]:
    if file_path.suffix == ".py":
        python_chunks = chunk_python_file(file_path, repo_root)

        if python_chunks:
            return python_chunks

    return chunk_file_by_lines(
        file_path=file_path,
        repo_root=repo_root,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def chunk_python_file(
    file_path: Path,
    repo_root: Path,
) -> list[dict]:
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    lines = content.splitlines()

    chunks = []

    for node in tree.body:
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue

        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)

        chunk_content = "\n".join(
            lines[start_line - 1:end_line]
        )

        if isinstance(node, ast.ClassDef):
            chunk_type = "class"
        else:
            chunk_type = "function"

        chunks.append(
            {
                "path": str(file_path.relative_to(repo_root)),
                "start_line": start_line,
                "end_line": end_line,
                "content": chunk_content,
                "symbol": node.name,
                "type": chunk_type,
            }
        )

    return chunks


def chunk_file_by_lines(
    file_path: Path,
    repo_root: Path,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[dict]:
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    lines = content.splitlines()

    if not lines:
        return []

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(lines), step):
        end = min(start + chunk_size, len(lines))

        chunk_lines = lines[start:end]

        chunk = {
            "path": str(file_path.relative_to(repo_root)),
            "start_line": start + 1,
            "end_line": end,
            "content": "\n".join(chunk_lines),
            "symbol": None,
            "type": "line_chunk",
        }

        chunks.append(chunk)

        if end == len(lines):
            break

    return chunks


def chunk_repository(
    files: list[Path],
    repo_root: Path,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[dict]:
    all_chunks = []

    for file_path in files:
        file_chunks = chunk_file(
            file_path=file_path,
            repo_root=repo_root,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_chunks.extend(file_chunks)

    return all_chunks