from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
}

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
}


def load_repository(repo_path: str) -> list[Path]:
    root = Path(repo_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository not found: {root}")

    if not root.is_dir():
        raise ValueError(f"Expected directory, got: {root}")

    source_files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        source_files.append(path)

    return source_files