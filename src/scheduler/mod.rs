use anyhow::Result;
use chrono::{DateTime, Timelike, Utc};
use chrono_tz::Tz;
use git2::Repository;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub source: String,
    pub source_type: String,
    pub last_commit_hash: Option<String>,
    pub last_check: Option<String>,
    pub last_update: Option<String>,
    pub owner: Option<String>,
    pub repo: Option<String>,
    pub branch: Option<String>,
    pub local_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataStore {
    pub sources: HashMap<String, SourceMetadata>,
}

impl MetadataStore {
    pub fn load(path: &PathBuf) -> Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let store: MetadataStore = serde_json::from_str(&content)?;
            Ok(store)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    pub fn get(&self, source: &str) -> Option<&SourceMetadata> {
        self.sources.get(source)
    }

    pub fn upsert(&mut self, metadata: SourceMetadata) {
        self.sources.insert(metadata.source.clone(), metadata);
    }
}

pub struct Scheduler {
    metadata_path: PathBuf,
    check_interval_hours: u64,
    window_start: u8,
    window_end: u8,
    timezone: Tz,
}

impl Scheduler {
    pub fn new(
        metadata_path: PathBuf,
        check_interval_hours: u64,
        window_start: u8,
        window_end: u8,
        timezone: &str,
    ) -> Result<Self> {
        let tz: Tz = timezone
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid timezone: {}", e))?;
        Ok(Self {
            metadata_path,
            check_interval_hours,
            window_start,
            window_end,
            timezone: tz,
        })
    }

    pub fn load_metadata(&self) -> Result<MetadataStore> {
        MetadataStore::load(&self.metadata_path)
    }

    pub fn save_metadata(&self, store: &MetadataStore) -> Result<()> {
        store.save(&self.metadata_path)
    }

    pub fn is_in_download_window(&self) -> bool {
        let now_utc: DateTime<Utc> = Utc::now();
        let now_tz = now_utc.with_timezone(&self.timezone);
        let hour = now_tz.hour() as u8;

        if self.window_start > self.window_end {
            hour >= self.window_start || hour < self.window_end
        } else {
            hour >= self.window_start && hour < self.window_end
        }
    }

    pub fn time_until_window(&self) -> std::time::Duration {
        let now_utc: DateTime<Utc> = Utc::now();
        let now_tz = now_utc.with_timezone(&self.timezone);
        let current_hour = now_tz.hour() as u8;

        let hours_until_start = if self.window_start > self.window_end {
            if current_hour >= self.window_start || current_hour < self.window_end {
                0
            } else if current_hour < self.window_start {
                (self.window_start - current_hour) as i64
            } else {
                (24 - current_hour + self.window_start) as i64
            }
        } else if current_hour >= self.window_start && current_hour < self.window_end {
            0
        } else if current_hour < self.window_start {
            (self.window_start - current_hour) as i64
        } else {
            (24 - current_hour + self.window_start) as i64
        };

        std::time::Duration::from_secs((hours_until_start * 3600) as u64)
    }

    pub fn needs_check(&self, metadata: &SourceMetadata) -> bool {
        if let Some(last_check) = &metadata.last_check
            && let Ok(last_check_time) = DateTime::parse_from_rfc3339(last_check)
        {
            let elapsed = Utc::now().signed_duration_since(last_check_time.with_timezone(&Utc));
            return elapsed.num_hours() >= self.check_interval_hours as i64;
        }
        true
    }

    pub fn get_current_commit_hash(repo_path: &std::path::Path) -> Result<String> {
        let repo = Repository::open(repo_path)?;
        let head = repo.head()?;
        let commit = head.peel_to_commit()?;
        Ok(commit.id().to_string())
    }

    pub fn has_updates(
        &self,
        metadata: &SourceMetadata,
        repo_path: &std::path::Path,
    ) -> Result<bool> {
        if let Some(stored_hash) = &metadata.last_commit_hash {
            let current_hash = Self::get_current_commit_hash(repo_path)?;
            return Ok(stored_hash != &current_hash);
        }
        Ok(true)
    }

    pub fn create_github_metadata(
        &self,
        owner: &str,
        repo: &str,
        branch: &str,
        repo_path: &std::path::Path,
    ) -> Result<SourceMetadata> {
        let commit_hash = Self::get_current_commit_hash(repo_path)?;
        let now = Utc::now().to_rfc3339();

        Ok(SourceMetadata {
            source: format!("github:{}/{}", owner, repo),
            source_type: "github".to_string(),
            last_commit_hash: Some(commit_hash),
            last_check: Some(now.clone()),
            last_update: Some(now),
            owner: Some(owner.to_string()),
            repo: Some(repo.to_string()),
            branch: Some(branch.to_string()),
            local_path: Some(repo_path.to_string_lossy().to_string()),
        })
    }

    pub fn create_local_metadata(&self, path: &std::path::Path) -> Result<SourceMetadata> {
        let now = Utc::now().to_rfc3339();

        Ok(SourceMetadata {
            source: format!("local:{}", path.display()),
            source_type: "local".to_string(),
            last_commit_hash: None,
            last_check: Some(now.clone()),
            last_update: Some(now),
            owner: None,
            repo: None,
            branch: None,
            local_path: Some(path.to_string_lossy().to_string()),
        })
    }

    pub fn update_check_time(&self, metadata: &mut SourceMetadata) {
        metadata.last_check = Some(Utc::now().to_rfc3339());
    }

    pub fn update_after_refresh(
        &self,
        metadata: &mut SourceMetadata,
        repo_path: &std::path::Path,
    ) -> Result<()> {
        let commit_hash = Self::get_current_commit_hash(repo_path)?;
        let now = Utc::now().to_rfc3339();
        metadata.last_commit_hash = Some(commit_hash);
        metadata.last_check = Some(now.clone());
        metadata.last_update = Some(now);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateCheckResult {
    pub source: String,
    pub needs_update: bool,
    pub reason: String,
    pub in_window: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateReport {
    pub checked_at: String,
    pub timezone: String,
    pub in_download_window: bool,
    pub time_until_window_seconds: u64,
    pub results: Vec<UpdateCheckResult>,
}
