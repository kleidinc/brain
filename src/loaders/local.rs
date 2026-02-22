use crate::loaders::chunker::{Chunk, TextChunker};
use std::path::Path;

pub struct LocalLoader {
    chunker: TextChunker,
}

impl LocalLoader {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunker: TextChunker::new(chunk_size, chunk_overlap),
        }
    }

    pub fn load_directory(&self, dir: &Path) -> anyhow::Result<Vec<(String, String, Vec<Chunk>)>> {
        let mut results = Vec::new();

        let code_extensions = [
            "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "cpp", "h", "hpp", "rb",
            "php", "swift", "kt", "scala", "lua", "r", "zig", "toml", "yaml", "yml", "json", "sql",
            "sh", "bash",
        ];

        let doc_extensions = ["md", "txt", "rst", "adoc", "org"];

        for result in ignore::WalkBuilder::new(dir)
            .hidden(false)
            .git_ignore(true)
            .git_global(false)
            .git_exclude(false)
            .build()
        {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let extension = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            let is_code = code_extensions.contains(&extension.as_str());
            let is_doc = doc_extensions.contains(&extension.as_str());

            if !is_code && !is_doc {
                continue;
            }

            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if content.trim().is_empty() {
                continue;
            }

            let relative_path = path.strip_prefix(dir)?.to_string_lossy().to_string();

            let chunks = if is_code {
                self.chunker.chunk_code(&content)
            } else {
                self.chunker.chunk_by_paragraphs(&content)
            };

            if !chunks.is_empty() {
                let source_name = dir.file_name().unwrap().to_string_lossy().to_string();

                results.push((source_name, relative_path, chunks));
            }
        }

        tracing::info!("Loaded {} files from directory", results.len());
        Ok(results)
    }

    pub fn load_file(&self, file_path: &Path) -> anyhow::Result<Vec<(String, String, Vec<Chunk>)>> {
        let content = std::fs::read_to_string(file_path)?;

        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        let extension = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let code_extensions = [
            "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "cpp", "h", "hpp", "rb",
            "php", "swift", "kt", "scala", "lua", "r", "zig", "toml", "yaml", "yml", "json", "sql",
            "sh", "bash",
        ];

        let is_code = code_extensions.contains(&extension.as_str());

        let chunks = if is_code {
            self.chunker.chunk_code(&content)
        } else {
            self.chunker.chunk_by_paragraphs(&content)
        };

        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();

        Ok(vec![(file_name.clone(), file_name, chunks)])
    }
}
