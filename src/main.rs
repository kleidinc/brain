use anyhow::Result;
use brain::cli::{Cli, Commands, IndexCommands, UpdateCommands};
use brain::config::Config;
use brain::embedding::EmbeddingModel;
use brain::loaders::{GitHubLoader, LocalLoader};
use brain::rag::{MistralRsClient, RagPipeline};
use brain::scheduler::{Scheduler, UpdateCheckResult, UpdateReport};
use brain::server::{self, AppState};
use brain::storage::{DocumentWithEmbedding, SourceType, VectorStore};
use clap::Parser;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let config = Config::load()?;

    match cli.command {
        Commands::Index { source } => handle_index(&config, source).await?,
        Commands::Update { action } => handle_update(&config, action).await?,
        Commands::Query { query, limit, json } => {
            handle_query(&config, &query, limit, json).await?
        }
        Commands::Search { query, limit } => handle_search(&config, &query, limit).await?,
        Commands::Serve { host, port } => handle_serve(&config, &host, port).await?,
        Commands::Sources { json } => handle_sources(&config, json).await?,
        Commands::Delete { source } => handle_delete(&config, &source).await?,
        Commands::Status => handle_status(&config).await?,
    }

    Ok(())
}

async fn init_pipeline(config: &Config) -> Result<RagPipeline> {
    tracing::info!("Loading embedding model: {}", config.embedding.model);
    let embedding_model =
        EmbeddingModel::new(&config.embedding.model, config.embedding.cuda_device)?;

    let dimensions = embedding_model.dimensions();
    tracing::info!("Embedding dimensions: {}", dimensions);

    let db_path = config.data_dir().join(&config.storage.lancedb_path);
    std::fs::create_dir_all(&db_path)?;

    let vector_store = VectorStore::new(
        &db_path.join(&config.storage.table_name),
        &config.storage.table_name,
        dimensions,
    )
    .await?;

    let llm_client = MistralRsClient::new(
        &config.llm.base_url,
        &config.llm.model,
        config.llm.max_tokens,
        config.llm.temperature,
    );

    Ok(RagPipeline::new(embedding_model, vector_store, llm_client))
}

async fn handle_index(config: &Config, source: IndexCommands) -> Result<()> {
    let pipeline = init_pipeline(config).await?;

    let scheduler = Scheduler::new(
        config.data_dir().join(&config.scheduler.metadata_file),
        config.scheduler.check_interval_hours,
        config.scheduler.download_window_start,
        config.scheduler.download_window_end,
        &config.scheduler.timezone,
    )?;
    let mut metadata_store = scheduler.load_metadata()?;

    match source {
        IndexCommands::Github {
            owner,
            repo,
            branch,
        } => {
            tracing::info!(
                "Indexing GitHub repository: {}/{} ({})",
                owner,
                repo,
                branch
            );

            let repos_path = config.data_dir().join(&config.sources.repos_path);
            std::fs::create_dir_all(&repos_path)?;

            let loader = GitHubLoader::new(
                repos_path.clone(),
                config.brain.chunk_size,
                config.brain.chunk_overlap,
            );

            let repo_path = loader.clone_or_update(&owner, &repo, &branch)?;
            let documents = loader.load_repo(&repo_path)?;

            let source_name = format!("github:{}/{}", owner, repo);
            let count =
                index_documents_simple(&pipeline, &source_name, SourceType::GitHub, documents)
                    .await?;

            let metadata = scheduler.create_github_metadata(&owner, &repo, &branch, &repo_path)?;
            metadata_store.upsert(metadata);
            scheduler.save_metadata(&metadata_store)?;

            println!("Indexed {} chunks from {}/{}", count, owner, repo);
        }
        IndexCommands::Local { path } => {
            tracing::info!("Indexing local directory: {:?}", path);

            let loader = LocalLoader::new(config.brain.chunk_size, config.brain.chunk_overlap);

            let documents = loader.load_directory(&path)?;

            let source_name = format!("local:{}", path.display());
            let count =
                index_documents_simple(&pipeline, &source_name, SourceType::Local, documents)
                    .await?;

            let metadata = scheduler.create_local_metadata(&path)?;
            metadata_store.upsert(metadata);
            scheduler.save_metadata(&metadata_store)?;

            println!("Indexed {} chunks from {}", count, path.display());
        }
        IndexCommands::Defaults => {
            tracing::info!("Indexing default repositories...");

            let repos_path = config.data_dir().join(&config.sources.repos_path);
            std::fs::create_dir_all(&repos_path)?;

            let loader = GitHubLoader::new(
                repos_path.clone(),
                config.brain.chunk_size,
                config.brain.chunk_overlap,
            );

            for default in &config.sources.defaults {
                tracing::info!("Indexing: {}/{}", default.owner, default.repo);

                let repo_path =
                    loader.clone_or_update(&default.owner, &default.repo, &default.branch)?;
                let documents = loader.load_repo(&repo_path)?;

                let source_name = format!("github:{}/{}", default.owner, default.repo);
                let count =
                    index_documents_simple(&pipeline, &source_name, SourceType::GitHub, documents)
                        .await?;

                let metadata = scheduler.create_github_metadata(
                    &default.owner,
                    &default.repo,
                    &default.branch,
                    &repo_path,
                )?;
                metadata_store.upsert(metadata);

                println!(
                    "Indexed {} chunks from {}/{}",
                    count, default.owner, default.repo
                );
            }

            scheduler.save_metadata(&metadata_store)?;
        }
    }

    Ok(())
}

async fn index_documents_simple(
    pipeline: &brain::rag::RagPipeline,
    source: &str,
    source_type: SourceType,
    documents: Vec<(String, String, Vec<brain::loaders::chunker::Chunk>)>,
) -> Result<usize> {
    let mut total_indexed = 0;
    let mut batch = Vec::new();

    for (_source_name, file_path, chunks) in documents {
        for chunk in chunks {
            let id = uuid::Uuid::new_v4().to_string();
            let embedding = pipeline.embed(&chunk.content)?;

            batch.push(DocumentWithEmbedding {
                id,
                content: chunk.content.clone(),
                source: source.to_string(),
                source_type: source_type.to_string(),
                file_path: file_path.clone(),
                chunk_index: chunk.index as i64,
                created_at: chrono::Utc::now().to_rfc3339(),
                embedding,
            });

            total_indexed += 1;

            if batch.len() >= 100 {
                pipeline.insert_batch(batch.clone()).await?;
                batch.clear();
                tracing::info!("Indexed {} chunks...", total_indexed);
            }
        }
    }

    if !batch.is_empty() {
        pipeline.insert_batch(batch).await?;
    }

    tracing::info!("Total indexed: {} chunks", total_indexed);
    Ok(total_indexed)
}

async fn handle_query(config: &Config, query: &str, limit: usize, json: bool) -> Result<()> {
    let pipeline = init_pipeline(config).await?;
    let response = pipeline.query_with_sources(query, limit).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&response)?);
    } else {
        println!("Answer:\n{}\n", response.answer);
        if !response.sources.is_empty() {
            println!("Sources:");
            for source in response.sources {
                println!("  - {} ({})", source.file_path, source.source);
            }
        }
    }

    Ok(())
}

async fn handle_search(config: &Config, query: &str, limit: usize) -> Result<()> {
    let pipeline = init_pipeline(config).await?;
    let results = pipeline.search(query, limit).await?;

    println!("Found {} results:\n", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("--- Result {} ---", i + 1);
        println!("Source: {}", result.source);
        println!("File: {}", result.file_path);
        println!(
            "Content:\n{}\n",
            result.content.chars().take(500).collect::<String>()
        );
    }

    Ok(())
}

async fn handle_update(config: &Config, action: UpdateCommands) -> Result<()> {
    let scheduler = Scheduler::new(
        config.data_dir().join(&config.scheduler.metadata_file),
        config.scheduler.check_interval_hours,
        config.scheduler.download_window_start,
        config.scheduler.download_window_end,
        &config.scheduler.timezone,
    )?;

    match action {
        UpdateCommands::Check { json } => {
            handle_update_check(config, &scheduler, json).await?;
        }
        UpdateCommands::Run { force, json } => {
            handle_update_run(config, &scheduler, force, json).await?;
        }
        UpdateCommands::Status => {
            handle_update_status(config, &scheduler).await?;
        }
    }

    Ok(())
}

async fn handle_update_check(config: &Config, scheduler: &Scheduler, json: bool) -> Result<()> {
    let mut metadata_store = scheduler.load_metadata()?;
    let pipeline = init_pipeline(config).await?;
    let indexed_sources = pipeline.vector_store().list_sources().await?;

    let in_window = scheduler.is_in_download_window();
    let mut results = Vec::new();

    for source in &indexed_sources {
        let metadata = metadata_store.get(source).cloned();

        let (needs_update, reason) = if let Some(meta) = metadata {
            let needs_check = scheduler.needs_check(&meta);

            if !needs_check {
                (false, "Not due for check yet".to_string())
            } else if let (Some(local_path), Some(_)) = (&meta.local_path, &meta.owner) {
                let repo_path = std::path::Path::new(local_path);
                if repo_path.exists() {
                    let loader = GitHubLoader::new(
                        config.data_dir().join(&config.sources.repos_path),
                        config.brain.chunk_size,
                        config.brain.chunk_overlap,
                    );

                    if loader
                        .clone_or_update(
                            meta.owner.as_ref().unwrap(),
                            meta.repo.as_ref().unwrap(),
                            meta.branch.as_ref().unwrap(),
                        )
                        .is_ok()
                    {
                        match scheduler.has_updates(&meta, repo_path) {
                            Ok(has_updates) => {
                                if has_updates {
                                    (true, "New commits available".to_string())
                                } else {
                                    let mut meta = meta.clone();
                                    scheduler.update_check_time(&mut meta);
                                    metadata_store.upsert(meta);
                                    (false, "Already up to date".to_string())
                                }
                            }
                            Err(e) => (false, format!("Error checking: {}", e)),
                        }
                    } else {
                        (false, "Failed to fetch updates".to_string())
                    }
                } else {
                    (false, "Local path not found".to_string())
                }
            } else {
                (false, "Local source - no remote updates".to_string())
            }
        } else {
            (true, "Not tracked in metadata".to_string())
        };

        results.push(UpdateCheckResult {
            source: source.clone(),
            needs_update,
            reason,
            in_window,
        });
    }

    scheduler.save_metadata(&metadata_store)?;

    let report = UpdateReport {
        checked_at: chrono::Utc::now().to_rfc3339(),
        timezone: config.scheduler.timezone.clone(),
        in_download_window: in_window,
        time_until_window_seconds: scheduler.time_until_window().as_secs(),
        results,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Update Check Report");
        println!("==================");
        println!("Time: {} ({})", report.checked_at, report.timezone);
        println!(
            "In download window: {}",
            if in_window { "YES" } else { "NO" }
        );
        if !in_window {
            let hours = report.time_until_window_seconds / 3600;
            let mins = (report.time_until_window_seconds % 3600) / 60;
            println!("Time until window: {}h {}m", hours, mins);
        }
        println!();

        for result in &report.results {
            let status = if result.needs_update {
                "UPDATE NEEDED"
            } else {
                "OK"
            };
            println!("  [{}] {}", status, result.source);
            println!("         {}", result.reason);
        }
    }

    Ok(())
}

async fn handle_update_run(
    config: &Config,
    scheduler: &Scheduler,
    force: bool,
    json: bool,
) -> Result<()> {
    let in_window = scheduler.is_in_download_window();

    if !in_window && !force {
        let time_until = scheduler.time_until_window();
        let hours = time_until.as_secs() / 3600;
        let mins = (time_until.as_secs() % 3600) / 60;

        if json {
            let report = serde_json::json!({
                "error": "Outside download window",
                "in_download_window": false,
                "time_until_window_seconds": time_until.as_secs(),
                "use_force": "Use --force to override"
            });
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            println!("Outside download window (22:00-8:00 Moscow time)");
            println!("Time until window: {}h {}m", hours, mins);
            println!("Use --force to override");
        }
        return Ok(());
    }

    let mut metadata_store = scheduler.load_metadata()?;
    let pipeline = init_pipeline(config).await?;
    let indexed_sources = pipeline.vector_store().list_sources().await?;

    let mut updated = Vec::new();
    let mut skipped = Vec::new();

    for source in &indexed_sources {
        let metadata = metadata_store.get(source).cloned();

        if let Some(meta) = metadata {
            if let (Some(local_path), Some(owner), Some(repo), Some(branch)) =
                (&meta.local_path, &meta.owner, &meta.repo, &meta.branch)
            {
                let repo_path = std::path::Path::new(local_path);
                if repo_path.exists() {
                    let loader = GitHubLoader::new(
                        config.data_dir().join(&config.sources.repos_path),
                        config.brain.chunk_size,
                        config.brain.chunk_overlap,
                    );

                    if loader.clone_or_update(owner, repo, branch).is_ok() {
                        match scheduler.has_updates(&meta, repo_path) {
                            Ok(true) => {
                                tracing::info!("Re-indexing updated source: {}", source);
                                pipeline.vector_store().delete_by_source(source).await?;

                                let documents = loader.load_repo(repo_path)?;
                                let count = index_documents_simple(
                                    &pipeline,
                                    source,
                                    SourceType::GitHub,
                                    documents,
                                )
                                .await?;

                                let mut meta = meta.clone();
                                scheduler.update_after_refresh(&mut meta, repo_path)?;
                                metadata_store.upsert(meta);

                                updated.push((source.clone(), count));
                            }
                            Ok(false) => {
                                let mut meta = meta.clone();
                                scheduler.update_check_time(&mut meta);
                                metadata_store.upsert(meta);
                                skipped.push((source.clone(), "Already up to date".to_string()));
                            }
                            Err(e) => {
                                skipped.push((source.clone(), format!("Error: {}", e)));
                            }
                        }
                    } else {
                        skipped.push((source.clone(), "Failed to fetch".to_string()));
                    }
                } else {
                    skipped.push((source.clone(), "Local path not found".to_string()));
                }
            } else {
                skipped.push((source.clone(), "Not a GitHub source".to_string()));
            }
        } else {
            skipped.push((source.clone(), "Not in metadata".to_string()));
        }
    }

    scheduler.save_metadata(&metadata_store)?;

    if json {
        let report = serde_json::json!({
            "updated": updated,
            "skipped": skipped,
            "forced": force && !in_window
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Update Run Complete");
        println!("===================");
        if force && !in_window {
            println!("(Forced outside window)");
        }

        if !updated.is_empty() {
            println!("\nUpdated:");
            for (source, count) in &updated {
                println!("  {} ({} chunks)", source, count);
            }
        }

        if !skipped.is_empty() {
            println!("\nSkipped:");
            for (source, reason) in &skipped {
                println!("  {} - {}", source, reason);
            }
        }
    }

    Ok(())
}

async fn handle_update_status(config: &Config, scheduler: &Scheduler) -> Result<()> {
    let metadata_store = scheduler.load_metadata()?;
    let in_window = scheduler.is_in_download_window();
    let time_until = scheduler.time_until_window();

    println!("Scheduler Status");
    println!("================");
    println!("Timezone: {}", config.scheduler.timezone);
    println!(
        "Check interval: {} hours",
        config.scheduler.check_interval_hours
    );
    println!(
        "Download window: {}:00 - {}:00",
        config.scheduler.download_window_start, config.scheduler.download_window_end
    );
    println!(
        "Currently in window: {}",
        if in_window { "YES" } else { "NO" }
    );

    if !in_window {
        let hours = time_until.as_secs() / 3600;
        let mins = (time_until.as_secs() % 3600) / 60;
        println!("Time until window: {}h {}m", hours, mins);
    }

    println!("\nTracked Sources:");

    if metadata_store.sources.is_empty() {
        println!("  (none)");
    } else {
        for (source, meta) in &metadata_store.sources {
            println!("  {}", source);
            if let Some(last_check) = &meta.last_check {
                println!("    Last check: {}", last_check);
            }
            if let Some(last_update) = &meta.last_update {
                println!("    Last update: {}", last_update);
            }
            if let Some(hash) = &meta.last_commit_hash {
                println!("    Commit: {}...", &hash[..8]);
            }
        }
    }

    Ok(())
}

async fn handle_serve(config: &Config, host: &str, port: u16) -> Result<()> {
    let pipeline = init_pipeline(config).await?;
    let state = Arc::new(AppState {
        pipeline,
        config: config.clone(),
    });

    println!("Starting brain server on {}:{}", host, port);
    println!("Endpoints:");
    println!("  POST /query          - Query the brain with RAG");
    println!("  POST /search         - Search for similar documents");
    println!("  GET  /sources        - List indexed sources");
    println!("  POST /sources/github - Add GitHub source");
    println!("  POST /sources/local  - Add local source");
    println!("  GET  /status         - Get system status");

    server::run_server(state, host, port).await
}

async fn handle_sources(config: &Config, json: bool) -> Result<()> {
    let pipeline = init_pipeline(config).await?;
    let sources = pipeline.vector_store().list_sources().await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&sources)?);
    } else {
        println!("Indexed sources:");
        for source in sources {
            println!("  - {}", source);
        }
    }

    Ok(())
}

async fn handle_delete(config: &Config, source: &str) -> Result<()> {
    let pipeline = init_pipeline(config).await?;
    pipeline.vector_store().delete_by_source(source).await?;
    println!("Deleted source: {}", source);
    Ok(())
}

async fn handle_status(config: &Config) -> Result<()> {
    let pipeline = init_pipeline(config).await?;
    let count = pipeline.vector_store().count().await?;
    let sources = pipeline.vector_store().list_sources().await?;

    println!("Brain Status:");
    println!("  Documents: {}", count);
    println!("  Sources: {}", sources.len());
    println!("  Embedding model: {}", config.embedding.model);
    println!("  LLM endpoint: {}", config.llm.base_url);

    Ok(())
}
