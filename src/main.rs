use anyhow::Result;
use brain::cli::{Cli, Commands, IndexCommands};
use brain::config::Config;
use brain::embedding::EmbeddingModel;
use brain::loaders::{GitHubLoader, LocalLoader};
use brain::rag::{MistralRsClient, RagPipeline};
use brain::server::{self, AppState};
use brain::storage::{SourceType, VectorStore, DocumentWithEmbedding};
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
        Commands::Query { query, limit, json } => handle_query(&config, &query, limit, json).await?,
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
    let embedding_model = EmbeddingModel::new(
        &config.embedding.model,
        config.embedding.cuda_device,
    )?;

    let dimensions = embedding_model.dimensions();
    tracing::info!("Embedding dimensions: {}", dimensions);

    let db_path = config.data_dir().join(&config.storage.lancedb_path);
    std::fs::create_dir_all(&db_path)?;

    let vector_store = VectorStore::new(
        &db_path.join(&config.storage.table_name),
        &config.storage.table_name,
        dimensions,
    ).await?;

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

    match source {
        IndexCommands::Github { owner, repo, branch } => {
            tracing::info!("Indexing GitHub repository: {}/{} ({})", owner, repo, branch);
            
            let repos_path = config.data_dir().join(&config.sources.repos_path);
            std::fs::create_dir_all(&repos_path)?;

            let loader = GitHubLoader::new(
                repos_path,
                config.brain.chunk_size,
                config.brain.chunk_overlap,
            );

            let repo_path = loader.clone_or_update(&owner, &repo, &branch)?;
            let documents = loader.load_repo(&repo_path)?;

            let source_name = format!("github:{}/{}", owner, repo);
            let count = index_documents_simple(&pipeline, &source_name, SourceType::GitHub, documents).await?;

            println!("Indexed {} chunks from {}/{}", count, owner, repo);
        }
        IndexCommands::Local { path } => {
            tracing::info!("Indexing local directory: {:?}", path);

            let loader = LocalLoader::new(
                config.brain.chunk_size,
                config.brain.chunk_overlap,
            );

            let documents = loader.load_directory(&path)?;

            let source_name = format!("local:{}", path.display());
            let count = index_documents_simple(&pipeline, &source_name, SourceType::Local, documents).await?;

            println!("Indexed {} chunks from {}", count, path.display());
        }
        IndexCommands::Defaults => {
            tracing::info!("Indexing default repositories...");

            let repos_path = config.data_dir().join(&config.sources.repos_path);
            std::fs::create_dir_all(&repos_path)?;

            let loader = GitHubLoader::new(
                repos_path,
                config.brain.chunk_size,
                config.brain.chunk_overlap,
            );

            for default in &config.sources.defaults {
                tracing::info!("Indexing: {}/{}", default.owner, default.repo);

                let repo_path = loader.clone_or_update(&default.owner, &default.repo, &default.branch)?;
                let documents = loader.load_repo(&repo_path)?;

                let source_name = format!("github:{}/{}", default.owner, default.repo);
                let count = index_documents_simple(&pipeline, &source_name, SourceType::GitHub, documents).await?;

                println!("Indexed {} chunks from {}/{}", count, default.owner, default.repo);
            }
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

    for (source_name, file_path, chunks) in documents {
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
        println!("Content:\n{}\n", result.content.chars().take(500).collect::<String>());
    }

    Ok(())
}

async fn handle_serve(config: &Config, host: &str, port: u16) -> Result<()> {
    let pipeline = init_pipeline(config).await?;
    let state = Arc::new(AppState { pipeline });

    println!("Starting brain server on {}:{}", host, port);
    println!("Endpoints:");
    println!("  POST /query   - Query the brain with RAG");
    println!("  POST /search  - Search for similar documents");
    println!("  GET  /sources - List indexed sources");
    println!("  GET  /status  - Get system status");

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
