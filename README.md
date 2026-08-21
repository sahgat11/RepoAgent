# RepoAgent

RepoAgent is a fully local AI codebase assistant that indexes Git repositories, semantically searches source code, and uses an agentic LLM to answer questions about a repository.

It can analyze either a local repository or a public GitHub URL and answer questions about implementation, architecture, functions, and project structure.

## Features

* Analyze local Git repositories
* Clone and analyze public GitHub repositories
* Automatically remove temporary GitHub clones after use
* AST-aware Python code chunking
* Semantic code search using Sentence Transformers
* Hybrid semantic, lexical, and path-aware retrieval
* Persistent per-repository embedding indexes
* Agentic repository tools:

  * `search_code`
  * `find_symbol`
  * `read_file`
  * `list_files`
* Local LLM inference through Ollama
* No paid API required
* Concise answers focused on repository implementation

## Architecture

```text
                     RepoAgent

 GitHub URL / Local Repository
              |
              v
       Repository Loader
              |
              v
       Source File Filter
              |
              v
      AST / Line Chunking
              |
              v
      Embedding Generation
              |
              v
      Persistent Vector Index
              |
              |
        User Question
              |
              v
          RepoAgent
              |
     ---------------------
     |        |          |
     v        v          v
 Search    Find       Read File
 Code      Symbol
     \        |          /
      \       |         /
       -----------------
              |
              v
        Local Ollama LLM
              |
              v
      Concise Code Answer
```

## Tech Stack

* Python
* Ollama
* Qwen2.5
* Sentence Transformers
* NumPy
* Python AST
* Git
* OpenAI-compatible local API

## Installation

Clone RepoAgent:

```bash
git clone https://github.com/sahgat11/RepoAgent.git
cd RepoAgent
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and download the local model:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before starting RepoAgent.

## Usage

### Analyze a local repository

```bash
python main.py /path/to/repository
```

For example:

```bash
python main.py .
```

### Analyze a GitHub repository

```bash
python main.py https://github.com/user/repository
```

For example:

```bash
python main.py https://github.com/pallets/click
```

RepoAgent temporarily clones the GitHub repository, builds or loads its semantic index, and removes the cloned source files when the session ends.

## Example

```text
Cloning GitHub repository: https://github.com/pallets/click
Repository cloned successfully.
Loading existing index...
Loaded 910 chunks.

RepoAgent ready.

Ask about the repository (or type 'exit'):
Where is command parsing implemented?

Thinking...

RepoAgent:

Command parsing is primarily handled inside Click's core source modules,
including src/click/core.py and src/click/parser.py.
```

## How It Works

### 1. Repository Loading

RepoAgent recursively scans the repository and keeps supported source-code files while ignoring directories such as:

```text
.git
venv
node_modules
build
dist
__pycache__
```

### 2. Code Chunking

Python source files are parsed using Python's Abstract Syntax Tree (AST).

Functions and classes become individual searchable chunks containing:

```text
file path
symbol name
chunk type
starting line
ending line
source code
```

Other supported languages fall back to overlapping line-based chunking.

### 3. Embeddings

Each code chunk is converted into a semantic embedding using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized and stored as a NumPy matrix.

### 4. Hybrid Retrieval

RepoAgent combines multiple retrieval signals:

* Semantic embedding similarity
* Keyword relevance
* Symbol relevance
* Filename relevance
* Repository path weighting

Core implementation directories such as `src/` receive preference over directories such as `examples/`, `tests/`, and `docs/`.

### 5. Persistent Indexing

Repository embeddings and chunk metadata are cached under:

```text
data/index/
```

Each repository receives its own identifier, allowing RepoAgent to reuse previously generated embeddings.

Generated indexes are excluded from Git.

### 6. Agent Tools

The local model can investigate the repository using several tools.

#### `search_code`

Semantically searches for code related to a concept or behavior.

#### `find_symbol`

Finds functions or classes by exact symbol name.

#### `read_file`

Reads a specific repository file when additional context is required.

#### `list_files`

Returns the source-file structure of the repository.

The agent can use multiple tools before producing its final answer.

## Local AI

RepoAgent runs its language model locally using Ollama.

The default model is:

```text
qwen2.5:3b
```

This means repository source code does not need to be sent to a paid cloud LLM API.

Embedding generation also runs locally.

## Supported Languages

RepoAgent currently indexes:

* Python
* JavaScript
* TypeScript
* JSX
* TSX
* Java
* C
* C++
* Go
* Rust

Python receives AST-aware chunking, while other languages currently use overlapping line-based chunking.

## Current Limitations

RepoAgent is an experimental developer tool.

Current limitations include:

* Local model reasoning quality depends on the selected Ollama model
* Very large repositories may require longer initial indexing times
* Non-Python languages currently use line-based rather than syntax-aware chunking
* Cached GitHub indexes do not yet automatically detect upstream repository updates
* Retrieval may occasionally return conceptually related code instead of the exact implementation

## Planned Improvements

* Retrieval evaluation benchmark
* Automatic GitHub index freshness detection
* CLI `--rebuild` option
* Improved terminal interface
* Automated tests
* Additional language-aware parsers

## Why RepoAgent?

General-purpose LLMs often require developers to manually paste files or explain repository structure.

RepoAgent instead builds a searchable representation of the entire codebase and gives the model tools to retrieve relevant implementation details when needed.

The goal is a faster workflow for questions such as:

```text
Where is authentication implemented?

What calls this function?

How is the repository indexed?

Where is request validation handled?

What files are responsible for database access?

What does load_repository do?
```

## License

This project is intended for educational and portfolio use.
