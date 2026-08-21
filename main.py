import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from repoagent.loader import load_repository
from repoagent.chunker import chunk_repository
from repoagent.embeddings import EmbeddingModel
from repoagent.index import save_index, load_index, index_exists
from repoagent.tools import RepoTools
from repoagent.agent import RepoAgent
from repoagent.repo_source import RepositorySource, get_repository_id


console = Console()


def build_index(
    repo_path: Path,
    repository_id: str,
    embedding_model: EmbeddingModel,
):
    console.print(
        "[bold]Scanning repository...[/bold]"
    )

    files = load_repository(
        str(repo_path)
    )

    console.print(
        f"Found [cyan]{len(files)}[/cyan] source files."
    )

    chunks = chunk_repository(
        files=files,
        repo_root=repo_path,
    )

    console.print(
        f"Created [cyan]{len(chunks)}[/cyan] code chunks."
    )

    console.print(
        "[bold]Generating embeddings...[/bold]"
    )

    embeddings = embedding_model.embed_chunks(
        chunks
    )

    console.print(
        f"Generated [cyan]{len(embeddings)}[/cyan] embeddings."
    )

    save_index(
        repository_id=repository_id,
        chunks=chunks,
        embeddings=embeddings,
    )

    console.print(
        "[green]Repository index saved.[/green]"
    )

    return chunks, embeddings


def run_repoagent(
    repo_path: Path,
    source: str,
    rebuild: bool,
):
    repository_id = get_repository_id(
        source
    )

    console.print(
        "\n[bold]Loading embedding model...[/bold]"
    )

    embedding_model = EmbeddingModel()

    if rebuild:
        console.print(
            "[yellow]Rebuilding repository index...[/yellow]"
        )

        chunks, embeddings = build_index(
            repo_path=repo_path,
            repository_id=repository_id,
            embedding_model=embedding_model,
        )

    elif index_exists(repository_id):
        console.print(
            "[green]Cached repository index found.[/green]"
        )

        chunks, embeddings = load_index(
            repository_id
        )

        console.print(
            f"Loaded [cyan]{len(chunks)}[/cyan] chunks."
        )

    else:
        console.print(
            "[yellow]No cached index found.[/yellow]"
        )

        chunks, embeddings = build_index(
            repo_path=repo_path,
            repository_id=repository_id,
            embedding_model=embedding_model,
        )

    tools = RepoTools(
        repo_path=repo_path,
        embedding_model=embedding_model,
        chunks=chunks,
        embeddings=embeddings,
    )

    agent = RepoAgent(tools)

    console.print()

    console.print(
        Panel.fit(
            "[bold green]RepoAgent is ready[/bold green]\n"
            "[dim]Ask questions about the repository. "
            "Type 'exit' to quit.[/dim]",
            title="RepoAgent",
            border_style="green",
        )
    )

    while True:
        console.print()

        query = Prompt.ask(
            "[bold cyan]You[/bold cyan]"
        )

        if query.lower().strip() in {
            "exit",
            "quit",
        }:
            console.print(
                "\n[dim]RepoAgent stopped.[/dim]"
            )
            break

        if not query.strip():
            continue

        console.print()

        with console.status(
            "[bold cyan]Investigating repository...[/bold cyan]",
            spinner="dots",
        ):
            answer = agent.run(query)

        console.print()

        console.print(
            Panel(
                answer,
                title="[bold]RepoAgent[/bold]",
                border_style="blue",
                padding=(1, 2),
            )
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "RepoAgent - local AI codebase assistant."
        )
    )

    parser.add_argument(
        "repository",
        help=(
            "Local repository path or "
            "public GitHub URL."
        ),
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Force RepoAgent to rebuild the "
            "repository index instead of using "
            "a cached index."
        ),
    )

    return parser.parse_args()


def main():
    console.print(
        Panel.fit(
            "[bold blue]RepoAgent[/bold blue]\n"
            "[dim]Local AI Codebase Assistant[/dim]",
            border_style="blue",
        )
    )

    args = parse_arguments()

    source = args.repository

    console.print(
        f"\n[bold]Repository:[/bold] {source}"
    )

    try:
        if source.startswith(
            "https://github.com/"
        ):
            console.print(
                "[dim]Cloning repository temporarily...[/dim]"
            )

        with RepositorySource(
            source
        ) as repo_path:
            run_repoagent(
                repo_path=repo_path,
                source=source,
                rebuild=args.rebuild,
            )

    except KeyboardInterrupt:
        console.print(
            "\n\n[yellow]RepoAgent stopped.[/yellow]"
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:
        console.print(
            f"\n[bold red]Error:[/bold red] {error}"
        )


if __name__ == "__main__":
    main()