# 🤖 RepoAgent

**RepoAgent is a fully local AI assistant for understanding unfamiliar codebases.**

Give it a local repository or a public GitHub URL, ask questions in plain English, and RepoAgent searches the codebase, inspects relevant files and symbols, and returns concise answers grounded in the repository.

No paid LLM API is required.

---

## 🌟 Highlights

- 🔍 **Ask natural-language questions about an entire repository**
- 🧠 **Hybrid semantic + lexical code retrieval**
- 🛠️ **Agentic tool calling** for code search, symbol lookup, file reading, and repository inspection
- 🌳 **AST-aware Python chunking** instead of arbitrary text splitting
- ⚡ **Persistent embedding cache** avoids recomputing repository indexes
- 🌐 **Analyze public GitHub repositories directly from a URL**
- 💻 **Runs locally with Ollama + Sentence Transformers**
- 🔒 **Repository source stays local during inference**
- 🧪 **8 automated tests covering the core indexing and retrieval pipeline**
- 📊 **100% Recall@5, 75% Accuracy@1, and 0.833 MRR** across the current multi-repository retrieval benchmark

---

## 📊 Performance

RepoAgent includes a retrieval benchmark built around implementation-level questions from two real open-source repositories:

- [Pallets Click](https://github.com/pallets/click)
- [Requests](https://github.com/psf/requests)

### Current Results

| Metric | Result |
| --- | ---: |
| Benchmark Questions | 12 |
| Accuracy@1 | **75.0%** |
| Recall@5 | **100.0%** |
| Mean Reciprocal Rank | **0.833** |

### Per-Repository Results

| Repository | Questions | Accuracy@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Click | 6 | 66.67% | 100.0% | 0.750 |
| Requests | 6 | 83.33% | 100.0% | 0.917 |
| **Overall** | **12** | **75.0%** | **100.0%** | **0.833** |

The benchmark measures whether RepoAgent retrieves the correct implementation file for a codebase question.

Results are deduplicated by source file before ranking so multiple chunks from the same file cannot artificially occupy the top results.

Run the benchmark yourself:

```bash
python evaluate.py
```

---

## ⚡ Why RepoAgent?

Large codebases are difficult to explore with a normal LLM.

You often have to:

- manually paste files into the prompt
- explain the project structure yourself
- repeatedly search through directories
- send large amounts of irrelevant code to the model
- rely on a model that may hallucinate implementation details

RepoAgent instead builds a searchable representation of the repository and lets the model retrieve only the code it needs.

```text
Repository
    ↓
Source discovery
    ↓
AST / line-based chunking
    ↓
Local embeddings
    ↓
Persistent repository index
    ↓
Hybrid retrieval
    ↓
Agent tool calls
    ↓
Grounded answer
```

---

## 🚀 Usage

Analyze a public GitHub repository:

```bash
python main.py https://github.com/pallets/click
```

Or analyze a local repository:

```bash
python main.py /path/to/repository
```

Then ask questions such as:

```text
Where is command parsing implemented?

What does load_repository do?

Where are HTTP sessions implemented?

How is the repository index built?

Where are exceptions defined?

What files handle authentication?
```

Example:

```text
╭─────────────────────────────╮
│ RepoAgent                   │
│ Local AI Codebase Assistant │
╰─────────────────────────────╯

Repository: https://github.com/pallets/click
Cloning repository temporarily...

Loading embedding model...
Cached repository index found.
Loaded 910 chunks.

╭─────────────────────── RepoAgent ────────────────────────╮
│ RepoAgent is ready                                       │
│ Ask questions about the repository. Type 'exit' to quit. │
╰──────────────────────────────────────────────────────────╯

You: Where is option parsing handled?

RepoAgent:

Option parsing is primarily implemented in src/click/parser.py,
with higher-level command processing in src/click/core.py.
```

---

## ⬇️ Installation

### Requirements

- Python 3.9+
- Git
- Ollama

Clone the repository:

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

Install the local LLM:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running, then start RepoAgent:

```bash
python main.py https://github.com/pallets/click
```

---

## 🧠 How It Works

RepoAgent combines retrieval, repository-specific tooling, and a local LLM.

### 1. Repository Loading

RepoAgent recursively scans supported source files while ignoring directories such as:

```text
.git
venv
.venv
node_modules
__pycache__
build
dist
```

Public GitHub repositories are shallow-cloned using:

```bash
git clone --depth 1
```

The temporary clone is automatically deleted when the RepoAgent session ends.

---

### 2. Code Chunking

Python files use **AST-aware chunking**.

Functions, async functions, and classes are extracted as logical code units containing metadata such as:

```text
file path
symbol
chunk type
start line
end line
source code
```

For example:

```python
{
    "path": "repoagent/loader.py",
    "symbol": "load_repository",
    "type": "function",
    "start_line": 15,
    "end_line": 32,
    "content": "..."
}
```

Other supported languages use overlapping line-based chunks.

---

### 3. Local Embeddings

Chunks are embedded locally using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model generates normalized **384-dimensional embeddings**.

Repository embeddings are stored as a NumPy matrix for efficient similarity search.

---

### 4. Hybrid Retrieval

RepoAgent combines several ranking signals:

- semantic embedding similarity
- keyword overlap
- symbol relevance
- filename relevance
- source-path relevance

Core source directories are favored over lower-priority locations such as examples, documentation, and tests.

This improves retrieval compared with relying only on vector similarity.

---

### 5. Persistent Indexing

Generated indexes are cached locally under:

```text
data/index/
```

Each repository stores:

```text
embeddings.npy
chunks.json
```

When RepoAgent sees the same repository again, it loads the cached index instead of regenerating every embedding.

For example, the Click repository currently produces:

```text
79 source files
910 indexed chunks
```

After the initial indexing pass, those 910 chunks can be loaded directly from the cache.

Force a fresh index with:

```bash
python main.py https://github.com/pallets/click --rebuild
```

---

## 🛠️ Agent Tools

The local LLM does not receive the entire repository at once.

Instead, RepoAgent exposes repository-specific tools that the model can call when needed.

### `search_code`

Finds source code related to a concept or behavior.

```text
Where is authentication implemented?
```

### `find_symbol`

Finds an exact function or class.

```text
What does load_repository do?
```

### `read_file`

Reads a specific repository file when additional context is needed.

### `list_files`

Returns the supported source files in the repository.

The model can make multiple tool calls before producing its final answer.

---

## 💻 Local-First AI

RepoAgent runs both major AI components locally.

### Embedding Model

```text
all-MiniLM-L6-v2
```

### Language Model

```text
qwen2.5:3b
```

The language model runs through Ollama's OpenAI-compatible endpoint:

```text
http://localhost:11434/v1
```

No paid OpenAI API key is required.

This architecture also means repository source code does not need to be sent to a paid cloud LLM during normal use.

---

## ⚙️ Efficiency

RepoAgent is designed to avoid unnecessary computation.

### Persistent Embedding Cache

Repositories are embedded once and reused across future sessions.

```text
First run:
clone → scan → chunk → embed → save index

Later runs:
clone → load cached index
```

This avoids recomputing hundreds or thousands of embeddings every time RepoAgent starts.

### Shallow Git Cloning

Remote repositories use:

```bash
git clone --depth 1
```

This avoids downloading unnecessary Git history.

### Targeted Context Retrieval

Instead of placing an entire codebase into the LLM context window, RepoAgent retrieves only the most relevant code chunks.

This reduces the amount of code processed by the language model and makes local inference practical with a relatively small model.

### Vectorized Similarity Search

Normalized repository embeddings are stored in a NumPy matrix.

Query similarity is computed using vectorized matrix operations rather than comparing chunks individually in Python.

---

## 🧪 Testing

Run the automated test suite:

```bash
python -m pytest -v
```

Current result:

```text
8 passed
```

The tests cover:

- source-file discovery
- ignored repository directories
- AST-aware Python chunking
- line-based fallback chunking
- multi-file chunk generation
- persistent index save/load
- semantic retrieval ranking
- repository path-traversal protection

---

## 🔒 File Safety

The `read_file` tool validates requested paths before reading files.

Attempts to escape the repository root, such as:

```text
../secret.py
```

are rejected.

This prevents an agent tool call from reading arbitrary files outside the repository being analyzed.

---

## 🌐 Supported Languages

RepoAgent currently indexes:

- Python
- JavaScript
- JSX
- TypeScript
- TSX
- Java
- C
- C++
- Go
- Rust

Python receives syntax-aware AST chunking.

Other languages currently use overlapping line-based chunks.

---

## 📁 Project Structure

```text
RepoAgent/
├── main.py
├── evaluate.py
├── requirements.txt
├── pytest.ini
├── README.md
│
├── repoagent/
│   ├── agent.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── index.py
│   ├── loader.py
│   ├── repo_source.py
│   ├── search.py
│   └── tools.py
│
└── tests/
    └── test_repoagent.py
```

---

## 🧰 Tech Stack

| Component | Technology |
| --- | --- |
| Language | Python |
| Local LLM | Qwen2.5 3B |
| LLM Runtime | Ollama |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Operations | NumPy |
| Python Parsing | AST |
| CLI | Rich |
| Repository Operations | Git |
| Tests | Pytest |

---

## 🚧 Current Limitations

RepoAgent is currently a v1 developer tool.

Known limitations include:

- local answer quality depends on the reasoning ability of the selected Ollama model
- non-Python languages currently use line-based rather than syntax-aware chunking
- cached GitHub indexes do not automatically detect new upstream commits
- retrieval may occasionally rank a related implementation above the most specific file
- the current benchmark contains 12 questions across two repositories
- large repositories require more time during their initial indexing pass

---

## 🔮 Possible Future Improvements

Potential extensions include:

- Git commit SHA-based cache invalidation
- syntax-aware parsing for more languages
- automatic source citations in answers
- larger multi-language retrieval benchmarks
- cross-encoder reranking
- configurable embedding models
- configurable Ollama models
- function call-graph analysis
- repository dependency graph analysis
- recursive AST extraction for class methods

---

## 🤝 Feedback & Contributions

RepoAgent is an experimental developer tool, and feedback is welcome.

If you find a bug, have an idea for better retrieval, or want support for another language, open an issue on GitHub.

Contributions are also welcome, particularly around:

- retrieval quality
- additional language parsers
- evaluation questions
- local model support
- developer experience

---

## 👤 Author

Created by [Sahil Gattu](https://github.com/sahgat11).

RepoAgent was built as an exploration of local LLM agents, retrieval-augmented generation, semantic code search, and practical AI tooling for software development.

---

## 📄 License

This project is currently intended for educational and portfolio use.