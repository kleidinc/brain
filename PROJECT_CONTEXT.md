# Brain Project Context

**Last Updated:** 2026-02-22
**Status:** Working

## Quick Start

```bash
# Environment (already in ~/.bashrc)
export PROTOC="$HOME/.local/bin/protoc"
export PROTOC_INCLUDE="$HOME/.local/share/protobuf"

# Build
cd ~/brain && cargo build --release

# Index a GitHub repo
./target/release/brain index github <owner> <repo> --branch <branch>

# Index local directory
./target/release/brain index local <path>

# Start server
./target/release/brain serve --port 9090

# Endpoints
curl http://127.0.0.1:9090/status
curl -X POST http://127.0.0.1:9090/search -H 'Content-Type: application/json' -d '{"query":"test","limit":5}'
```

## Current State

- **Documents indexed:** 56,964
- **Sources:**
  - `github:rust-lang/rust` (master branch)
  - `local:/home/alex/brain/src`
- **Embedding model:** BAAI/bge-small-en-v1.5 (384 dimensions)
- **GPU:** CUDA device 1
- **Database:** `data/lancedb/documents/`

## Architecture

```
src/
├── main.rs           # CLI entry, command handlers
├── lib.rs            # Module exports
├── config.rs         # Config loading (config.toml)
├── embedding/
│   └── mod.rs        # Candle GPU embeddings (BGE-small)
├── storage/
│   └── mod.rs        # LanceDB vector store
├── loaders/
│   ├── mod.rs
│   ├── chunker.rs    # Text chunking (512 words, 50 overlap)
│   ├── github.rs     # Clone/update GitHub repos
│   └── local.rs      # Local file scanning
├── rag/
│   ├── mod.rs
│   ├── client.rs     # mistral.rs HTTP client
│   └── pipeline.rs   # RAG pipeline (search + query)
├── server/
│   └── mod.rs        # Axum HTTP API
└── cli/
    └── mod.rs        # Clap CLI definitions
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /status | Document count, status |
| GET | /sources | List indexed sources |
| POST | /search | Vector similarity search |
| POST | /query | RAG query (needs mistral.rs) |
| DELETE | /sources/:source | Delete source |

## Configuration (config.toml)

```toml
[brain]
chunk_size = 512
chunk_overlap = 50

[embedding]
model = "BAAI/bge-small-en-v1.5"
dimensions = 384
cuda_device = 1        # GPU 1 (GPU 0 reserved for system)

[llm]
base_url = "http://localhost:1234"
```

## Key Fixes Applied

1. **Model changed:** nomic-embed-text → bge-small-en-v1.5 (standard BERT architecture)
2. **Token truncation:** Max 512 tokens, text truncated to 8000 chars (fixes CUDA assertion error)
3. **Tensor shape:** Added `squeeze(0)` after `mean(1)` for 1D embedding vector
4. **GPU device:** Changed from 0 to 1 (GPU 0 reserved for system)
5. **Protobuf:** Installed protoc to ~/.local/bin/ and protobuf includes to ~/.local/share/protobuf/

## Dependencies (key ones)

- candle-core/nn/transformers 0.9 (CUDA)
- lancedb 0.26
- arrow-array/schema 57
- axum 0.8
- git2 0.19
- tokenizers 0.21
- hf-hub 0.4

## Known Limitations

1. **Query endpoint needs mistral.rs:** Start with `mistralrs-2gpu serve --ui -m Qwen/Qwen3-4B`
2. **No incremental updates:** Re-indexing adds duplicates; delete source first
3. **Single embedding model:** Changing model requires code change

## Tested File Types

**Code:** .rs, .py, .js, .ts, .go, .java, .c, .cpp, .h, .toml, .yaml, .json, .sql, .sh
**Docs:** .md, .txt, .rst

## Next Steps / TODO

- [ ] Add more repos: dioxuslabs/dioxus, launchbadge/sqlx, iced-rs/iced
- [ ] Test query endpoint with mistral.rs running
- [ ] Consider adding incremental update support
- [ ] Add embedding model configuration at runtime

## Build Commands

```bash
# Check
PROTOC=~/.local/bin/protoc PROTOC_INCLUDE=~/.local/share/protobuf cargo check

# Build release
PROTOC=~/.local/bin/protoc PROTOC_INCLUDE=~/.local/share/protobuf cargo build --release
```

## Data Locations

- **Repo clones:** `~/brain/data/repos/`
- **Vector DB:** `~/brain/data/lancedb/`
- **Model cache:** `~/.cache/huggingface/hub/`

## Related Files

- `~/remember/brain_glm5.md` - Technical documentation
- `~/remember/mistral.md` - mistral.rs usage guide
- `~/brain/config.toml` - Configuration file
