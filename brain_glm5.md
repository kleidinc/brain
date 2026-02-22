# Brain - Local RAG System

A Rust-based local RAG (Retrieval-Augmented Generation) system using Candle for GPU embeddings, LanceDB for vector storage, and mistral.rs for generation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                                │
├─────────────────────────────────────────────────────────────────────┤
│  GitHub Repos ──┐                                                    │
│  Local Docs ────┼──→ [Loader/Chunker] ──→ [Candle-embed (GPU)]     │
│  Codebases ────┘              │                    │                │
│                               ↓                    ↓                │
│                         Text Chunks          Embeddings             │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        STORAGE                                       │
├─────────────────────────────────────────────────────────────────────┤
│                    [LanceDB Vector Store]                           │
│                    - Embeddings + Metadata                          │
│                    - Source tracking                                │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│  Query ──→ [Candle-embed] ──→ [LanceDB Search] ──→ [Rig RAG]       │
│                                                            │         │
│                                                            ↓         │
│                                                   [mistral.rs]       │
│                                                   (port 1234)        │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        INTERFACES                                    │
├─────────────────────────────────────────────────────────────────────┤
│  [CLI Tool]                      [HTTP Server]                       │
│  - Index sources                 - POST /index                      │
│  - Query brain                   - POST /query                      │
│  - List sources                  - GET  /sources                     │
│  - Update/refresh                - DELETE /sources/:id              │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Embedding Module (`src/embedding/mod.rs`)

**Purpose**: Generate vector embeddings from text using Candle with CUDA support.

**Model**: BAAI/bge-small-en-v1.5 (384 dimensions)

**Key Functions**:
- `new()`: Load model from HuggingFace, initialize tokenizer, load weights to GPU
- `embed_one()`: Generate embedding for single text with truncation and normalization
- `embed()`: Batch embedding for multiple texts

### 2. Storage Module (`src/storage/mod.rs`)

**Purpose**: Vector database operations using LanceDB.

**Data Structure**:
- `DocumentWithEmbedding`: id, content, source, source_type, file_path, chunk_index, created_at, embedding
- `SearchResult`: id, content, source, file_path

**Key Functions**:
- `new()`: Initialize LanceDB connection, create table if needed
- `insert()`: Add documents with embeddings
- `search()`: Vector similarity search
- `delete_by_source()`: Remove documents by source
- `list_sources()`: List all indexed sources
- `count()`: Get total document count

### 3. Loaders Module (`src/loaders/`)

**Purpose**: Load and chunk documents from various sources.

**Components**:

#### Chunker (`chunker.rs`)
- `chunk()`: Word-based chunking
- `chunk_by_paragraphs()`: Paragraph-aware chunking
- `chunk_code()`: Code-aware chunking with overlap

#### GitHub Loader (`github.rs`)
- `clone_or_update()`: Clone or update repository
- `load_repo()`: Load all relevant files from repository

#### Local Loader (`local.rs`)
- `load_directory()`: Load files from local directory (respects .gitignore)
- `load_file()`: Load single file

### 4. RAG Module (`src/rag/`)

**Purpose**: RAG pipeline for context retrieval and generation.

**Components**:

#### Client (`client.rs`)
- `MistralRsClient`: HTTP client for mistral.rs server
- `chat()`: Send chat completion request
- `complete()`: Simple completion request

#### Pipeline (`pipeline.rs`)
- `index_documents()`: Index documents with embeddings
- `search()`: Vector search
- `query()`: RAG query with LLM response
- `query_with_sources()`: RAG query with source citations

### 5. Server Module (`src/server/mod.rs`)

**Purpose**: HTTP API server using Axum.

**Endpoints**:
| Method | Path | Description |
|-------|------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | RAG query with LLM |
| POST | `/search` | Vector search |
| GET | `/sources` | List sources |
| DELETE | `/sources/:source` | Delete source |
| GET | `/status` | System status |

### 6. CLI Module (`src/cli/mod.rs`)

**Purpose**: Command-line interface using Clap.

**Commands**:
| Command | Description |
|---------|-------------|
| `index github <owner>/<repo>` | Index GitHub repository |
| `index local <path>` | Index local directory |
| `index defaults` | Index default sources |
| `query <query>` | Query the brain |
| `search <query>` | Search documents |
| `serve` | Start HTTP server |
| `sources` | List indexed sources |
| `delete <source>` | Delete source |
| `status` | Show system status |

## File Structure

```
~/brain/
├── Cargo.toml              # Dependencies
├── config.toml             # Configuration
├── src/
│   ├── main.rs             # CLI entry point
│   ├── lib.rs              # Library exports
│   ├── config.rs           # Configuration parsing
│   ├── embedding/
│   │   └── mod.rs           # Candle GPU embeddings
│   ├── storage/
│   │   └── mod.rs           # LanceDB operations
│   ├── loaders/
│   │   ├── mod.rs
│   │   ├── chunker.rs       # Text chunking
│   │   ├── github.rs         # GitHub repo loader
│   │   └── local.rs          # Local file loader
│   ├── rag/
│   │   ├── mod.rs
│   │   ├── client.rs         # mistral.rs HTTP client
│   │   └── pipeline.rs       # RAG pipeline
│   ├── server/
│   │   └── mod.rs            # HTTP API (Axum)
│   └── cli/
│       └── mod.rs            # CLI commands
├── data/
│   └── lancedb/             # Vector database storage
└── repos/                   # Cloned GitHub repos
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| candle-core/nn/transformers | 0.9 | GPU embeddings with CUDA |
| lancedb | 0.26 | Embedded vector database |
| arrow-array/schema | 57 | Arrow data structures |
| rig-core | 0.7 | RAG framework |
| axum | 0.8 | HTTP server |
| clap | 4 | CLI framework |
| git2 | 0.19 | Git operations |
| tokenizers | 0.21 | HuggingFace tokenizers |
| hf-hub | 0.4 | HuggingFace model downloads |
| tokio | 1 | Async runtime |
| serde/serde_json | 1/1 | Serialization |
| anyhow/thiserror | 1/2 | Error handling |

## Configuration (`config.toml`)

```toml
[brain]
name = "brain"
chunk_size = 512          # Tokens per chunk
chunk_overlap = 50        # Overlap between chunks

[embedding]
model = "BAAI/bge-small-en-v1.5"  # HuggingFace model ID
dimensions = 384                   # Embedding dimensions
max_length = 512                   # Max sequence length
cuda_device = 1                    # GPU device (1 or 2, GPU 0 reserved)

[storage]
lancedb_path = "data/lancedb"
table_name = "documents"

[llm]
provider = "mistralrs"
base_url = "http://localhost:1234"
model = "default"
max_tokens = 2048
temperature = 0.7

[server]
host = "127.0.0.1"
port = 8080

[sources]
repos_path = "repos"

[[sources.defaults]]
owner = "rust-lang"
repo = "rust"
branch = "master"
```

## How to Use

### Prerequisites

1. **Environment Variables** (add to `~/.bashrc`):
```bash
export PROTOC="$HOME/.local/bin/protoc"
export PROTOC_INCLUDE="$HOME/.local/share/protobuf"
```

2. **Build**:
```bash
cd ~/brain
cargo build --release
```

### Starting mistral.rs Server

The RAG query endpoint requires mistral.rs running:

```bash
# Start mistral.rs on GPUs 1 and 2
CUDA_VISIBLE_DEVICES=1,2 MISTRALRS_MN_LOCAL_WORLD_SIZE=2 mistralrs serve --ui -m Qwen/Qwen3-4B
```

### Indexing Documents

```bash
# Index a GitHub repository
brain index github rust-lang/rust

# Index a local directory
brain index local ~/projects/my-app

# Index all default sources
brain index defaults
```

### Querying the Brain

```bash
# CLI query
brain query "How do I use async in Rust?"

# JSON output
brain query "How do I use async in Rust?" --json

# Search (no LLM, just retrieval)
brain search "tokio runtime"
```

### Starting the HTTP Server

```bash
# Start server on default port (8080) or custom port
brain serve --port 9090
```

### HTTP API Examples

```bash
# Health check
curl http://127.0.0.1:9090/health

# Status
curl http://127.0.0.1:9090/status

# Search
curl -X POST http://127.0.0.1:9090/search \
  -H "Content-Type: application/json" \
  -d '{"query": "embedding model", "limit": 5}'

# Query (requires mistral.rs running)
curl -X POST http://127.0.0.1:9090/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does async work in Rust?", "limit": 3}'

# List sources
curl http://127.0.0.1:9090/sources

# Delete a source
curl -X DELETE http://127.0.0.1:9090/sources/local:/home/alex/brain/src
```

## Technical Notes

### Embedding Model

- Uses **BAAI/bge-small-en-v1.5** (384 dimensions, 512 max tokens)
- Runs on **CUDA device 1** (GPU 0 reserved for system)
- Text is truncated to 8000 chars before tokenization
- Tokens are truncated to 512 max
- Embeddings are L2-normalized

### Vector Search

- LanceDB stores documents with embeddings
- Similarity search uses cosine distance
- Returns: id, content, source, file_path

### Chunking Strategy

- **Code files**: Line-aware chunking with 5-line overlap
- **Documentation**: Paragraph-aware chunking
- **Default chunk size**: 512 words with 50 word overlap

### File Types Supported

**Code**: `.rs`, `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.go`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.lua`, `.r`, `.zig`, `.toml`, `.yaml`, `.yml`, `.json`, `.sql`, `.sh`, `.bash`

**Documentation**: `.md`, `.txt`, `.rst`, `.adoc`, `.org`

## Known Limitations

1. **Query endpoint requires mistral.rs**: The `/query` endpoint needs a running mistral.rs server on port 1234
2. **No incremental updates**: Re-indexing a source adds duplicates; delete source first
3. **Single embedding model**: Model is loaded once at startup; changing requires restart

## Troubleshooting

### CUDA Errors

If you see `CUDA_ERROR_ASSERT`:
- Ensure GPU has enough memory
- Check CUDA device is correct (use 1 or 2, not 0 if reserved)

### Port Conflicts

If port 8080 is in use:
```bash
brain serve --port 9090
```

### Missing protoc

If build fails with `protoc not found`:
```bash
source ~/.bashrc  # Or restart shell
```
