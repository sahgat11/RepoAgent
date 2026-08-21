import hashlib
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def is_github_url(source: str) -> bool:
    try:
        parsed = urlparse(source)

        return (
            parsed.scheme in {"http", "https"}
            and parsed.netloc.lower() == "github.com"
        )

    except ValueError:
        return False


class RepositorySource:
    def __init__(self, source: str):
        self.source = source
        self.repo_path = None
        self._temp_dir = None

    def __enter__(self) -> Path:
        if is_github_url(self.source):
            self._temp_dir = tempfile.TemporaryDirectory(
                prefix="repoagent_"
            )

            temp_root = Path(
                self._temp_dir.name
            ).resolve()

            repo_path = temp_root / "repository"

            print(
                f"Cloning GitHub repository: {self.source}"
            )

            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        self.source,
                        str(repo_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            except subprocess.CalledProcessError as error:
                self._temp_dir.cleanup()
                self._temp_dir = None

                message = error.stderr.strip()

                raise RuntimeError(
                    f"Unable to clone repository: {message}"
                ) from error

            self.repo_path = repo_path.resolve()

            print(
                "Repository cloned successfully."
            )

            return self.repo_path

        repo_path = (
            Path(self.source)
            .expanduser()
            .resolve()
        )

        if not repo_path.exists():
            raise FileNotFoundError(
                f"Repository not found: {repo_path}"
            )

        if not repo_path.is_dir():
            raise ValueError(
                f"Expected repository directory: {repo_path}"
            )

        self.repo_path = repo_path

        return self.repo_path

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

            self._temp_dir = None


def get_repository_id(source: str) -> str:
    normalized = source.strip()

    if not is_github_url(normalized):
        normalized = str(
            Path(normalized)
            .expanduser()
            .resolve()
        )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]