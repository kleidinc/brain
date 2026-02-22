use crate::loaders::chunker::{Chunk, TextChunker};
use anyhow::Result;
use git2::Repository;
use std::path::Path;

pub struct GitHubLoader {
    repos_path: std::path::PathBuf,
    chunker: TextChunker,
}

impl GitHubLoader {
    pub fn new(repos_path: std::path::PathBuf, chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            repos_path,
            chunker: TextChunker::new(chunk_size, chunk_overlap),
        }
    }

    pub fn clone_or_update(
        &self,
        owner: &str,
        repo: &str,
        branch: &str,
    ) -> Result<std::path::PathBuf> {
        let repo_dir = self.repos_path.join(format!("{}-{}", owner, repo));
        let repo_url = format!("https://github.com/{}/{}.git", owner, repo);

        if repo_dir.exists() {
            tracing::info!("Updating existing repository: {}", repo_url);
            let repository = Repository::open(&repo_dir)?;

            let mut remote = repository.find_remote("origin")?;
            remote.fetch(&[branch], None, None)?;

            let fetch_head = repository.find_reference("FETCH_HEAD")?;
            let fetch_commit = repository.reference_to_annotated_commit(&fetch_head)?;
            let commit = repository.find_commit(fetch_commit.id())?;
            let obj = commit.into_object();

            repository.reset(&obj, git2::ResetType::Hard, None)?;
            tracing::info!("Repository updated: {}", repo_dir.display());
        } else {
            tracing::info!("Cloning repository: {}", repo_url);
            Repository::clone(&repo_url, &repo_dir)?;
            tracing::info!("Repository cloned to: {}", repo_dir.display());
        }

        Ok(repo_dir)
    }

    pub fn load_repo(&self, repo_path: &Path) -> Result<Vec<(String, String, Vec<Chunk>)>> {
        let mut results = Vec::new();

        let code_extensions = [
            "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "cpp", "h", "hpp", "rb",
            "php", "swift", "kt", "scala", "lua", "r", "zig",
        ];

        let doc_extensions = ["md", "txt", "rst", "adoc", "org"];

        for entry in walkdir::WalkDir::new(repo_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if path.is_dir() {
                continue;
            }

            let path_str = path.to_string_lossy();

            if path_str.contains("/.git/")
                || path_str.contains("/target/")
                || path_str.contains("/node_modules/")
                || path_str.contains("/__pycache__/")
            {
                continue;
            }

            let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

            let is_code = code_extensions.contains(&extension);
            let is_doc = doc_extensions.contains(&extension);

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

            let relative_path = path.strip_prefix(repo_path)?.to_string_lossy().to_string();

            let chunks = if is_code {
                self.chunker.chunk_code(&content)
            } else {
                self.chunker.chunk_by_paragraphs(&content)
            };

            if !chunks.is_empty() {
                let source_name = repo_path.file_name().unwrap().to_string_lossy().to_string();

                results.push((source_name, relative_path, chunks));
            }
        }

        tracing::info!("Loaded {} files from repository", results.len());
        Ok(results)
    }
}
