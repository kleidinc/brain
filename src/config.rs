use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub brain: BrainConfig,
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub llm: LlmConfig,
    pub server: ServerConfig,
    pub sources: SourcesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    pub name: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub dimensions: usize,
    pub max_length: usize,
    pub cuda_device: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub lancedb_path: PathBuf,
    pub table_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcesConfig {
    pub repos_path: PathBuf,
    pub defaults: Vec<DefaultSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultSource {
    pub owner: String,
    pub repo: String,
    pub branch: String,
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let config_path = directories::ProjectDirs::from("com", "local", "brain")
            .map(|dirs| dirs.config_dir().join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("config.toml"));

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            let content = std::fs::read_to_string("config.toml")?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        }
    }

    pub fn data_dir(&self) -> PathBuf {
        PathBuf::from(&self.storage.lancedb_path)
            .parent()
            .unwrap()
            .to_path_buf()
    }
}
