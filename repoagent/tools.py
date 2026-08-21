from pathlib import Path

from repoagent.loader import load_repository
from repoagent.embeddings import EmbeddingModel
from repoagent.search import semantic_search


class RepoTools:
    def __init__(
        self,
        repo_path: Path,
        embedding_model: EmbeddingModel,
        chunks: list[dict],
        embeddings,
    ):
        self.repo_path = repo_path
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.embeddings = embeddings

    def search_code(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        query_embedding = self.embedding_model.embed_query(query)

        return semantic_search(
            query=query,
            query_embedding=query_embedding,
            embeddings=self.embeddings,
            chunks=self.chunks,
            top_k=top_k,
        )

    def read_file(self, path: str) -> str:
        file_path = (self.repo_path / path).resolve()

        # Prevent reading outside the repo
        try:
            file_path.relative_to(self.repo_path)
        except ValueError:
            return f"Invalid path outside repository: {path}"

        if not file_path.exists():
            return f"File not found: {path}"

        if not file_path.is_file():
            return f"Not a file: {path}"

        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Unable to read file as text: {path}"

    def find_symbol(self, symbol: str) -> list[dict]:
        matches = []

        for chunk in self.chunks:
            chunk_symbol = chunk.get("symbol")

            if chunk_symbol and chunk_symbol.lower() == symbol.lower():
                matches.append(chunk)

        return matches

    def list_files(self) -> list[str]:
        files = load_repository(str(self.repo_path))

        return [
            str(file.relative_to(self.repo_path))
            for file in files
        ]