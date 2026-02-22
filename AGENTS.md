# Brain Project - Agent Instructions

When working on this project, read `PROJECT_CONTEXT.md` first for current state and architecture.

## Project Overview

Brain is a local RAG (Retrieval-Augmented Generation) system written in Rust:
- **Embeddings:** Candle with CUDA (BAAI/bge-small-en-v1.5, 384 dims)
- **Vector DB:** LanceDB (embedded)
- **LLM:** mistral.rs server (OpenAI-compatible API)
- **HTTP Server:** Axum

## Build Requirements

```bash
export PROTOC="$HOME/.local/bin/protoc"
export PROTOC_INCLUDE="$HOME/.local/share/protobuf"
```

These are already in `~/.bashrc`.

## Build Commands

```bash
cd ~/brain
cargo build --release
```

## Test Commands

```bash
# Status
./target/release/brain status

# Search (no LLM needed)
./target/release/brain search "async future"

# Start server
./target/release/brain serve --port 9090

# API test
curl http://127.0.0.1:9090/status
curl -X POST http://127.0.0.1:9090/search -H 'Content-Type: application/json' -d '{"query":"test","limit":5}'
```

## Key Files

| File | Purpose |
|------|---------|
| `PROJECT_CONTEXT.md` | Full project context, current state, architecture |
| `config.toml` | Runtime configuration |
| `src/embedding/mod.rs` | Candle GPU embeddings |
| `src/storage/mod.rs` | LanceDB operations |
| `src/rag/pipeline.rs` | RAG pipeline |
| `src/server/mod.rs` | HTTP API |

## Current State

- 56,964 documents indexed
- Sources: rust-lang/rust, brain/src
- GPU 1 for embeddings

## Common Tasks

### Index a new repo
```bash
./target/release/brain index github <owner> <repo> --branch <branch>
```

### Index local directory
```bash
./target/release/brain index local <path>
```

### Query (needs mistral.rs running on port 1234)
```bash
./target/release/brain query "How do futures work?"
```

## Rust Best Practices

### Error Handling
- Use `anyhow::Result` for application code (simple, flexible)
- Use `thiserror` for library crates (custom error types)
- Always propagate errors with `?`, never panic in production code
- Use `.context()` from anyhow to add context to errors

### Async Patterns
- Use `tokio` as the async runtime
- Prefer `async fn` over returning `impl Future`
- Use `tokio::sync` primitives (Mutex, RwLock, channels) not `std::sync`
- Avoid holding async locks across `.await` points

### Memory Management
- Use `Arc<T>` for shared ownership across threads
- Use `&str` for string slices, `String` for owned strings
- Prefer `Box<dyn Trait>` for trait objects when needed
- Use `Cow<str>` for functions that may or may not allocate

### Code Organization
- One module per file, use `mod.rs` for module re-exports
- Group related functionality in submodules
- Keep `main.rs` thin - delegate to lib.rs
- Use `pub(crate)` for internal visibility

### Testing
- Unit tests in same file with `#[cfg(test)]` module
- Integration tests in `tests/` directory
- Use `#[tokio::test]` for async tests
- Mock external dependencies with trait objects

### Dependencies
- Pin exact versions in Cargo.toml for reproducibility
- Use `cargo audit` to check for security vulnerabilities
- Prefer `rustls` over native TLS for portability
- Use `parking_lot` for faster mutex/rwlock implementations

### Performance
- Use `&[T]` instead of `&Vec<T>` in function signatures
- Prefer iterators over manual loops
- Use `Cow` to avoid unnecessary allocations
- Profile with `cargo flamegraph` before optimizing

### Documentation
- Document public APIs with `///` doc comments
- Include examples in doc comments with `# Example`
- Use `#![deny(missing_docs)]` for library crates
- Keep module-level docs in `mod.rs`

### Safety
- Prefer safe Rust; avoid `unsafe` unless necessary
- When using `unsafe`, document safety invariants
- Use `#[deny(unsafe_op_in_unsafe_fn)]` for unsafe code
- Run `cargo clippy` and fix all warnings

### Common Patterns in This Project

```rust
// Error handling
fn process() -> anyhow::Result<()> { ... }

// Async function
pub async fn process(&self) -> Result<()> { ... }

// Shared state
Arc<tokio::sync::Mutex<State>>

// Module structure
// src/storage/mod.rs - re-exports
pub mod document;
pub use document::Document;

// Builder pattern for configuration
Config::builder()
    .setting(value)
    .build()?
```

## Issue Tracking with Beads

This project uses [beads](https://github.com/steveyegge/beads) for git-backed issue tracking designed for AI agents.

### Installation
```bash
curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash
```

### Commands
```bash
bd init --quiet              # Initialize beads in project
bd create "Issue title" -p 1 -t task   # Create issue (priority 1-5, type: task/bug/feature)
bd list --json               # List all issues (JSON for agents)
bd ready --json              # Show unblocked, ready work
bd show <id> --json          # Show issue details
bd sync                      # Sync JSONL with SQLite
```

### Beads Workflow
- Issues stored in `.beads/issues.jsonl` (git-tracked) and `.beads/beads.db` (local SQLite)
- Use `bd ready --json` to find unblocked work
- Use `--json` flag for all programmatic access
- Run `bd sync` at end of session to ensure sync

### Dependencies
```bash
bd create "Fix bug" --deps blocks:bd-42    # New issue blocks bd-42
bd create "Follow-up" --deps discovered-from:bd-10  # Discovered during work
```

## Lint and Check Commands

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo test
```

## Related Documentation

- `~/remember/brain_glm5.md` - Technical paper
- `~/remember/mistral.md` - mistral.rs usage

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
