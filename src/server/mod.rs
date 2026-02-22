use axum::{
    Router,
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

pub struct AppState {
    pub pipeline: crate::rag::RagPipeline,
    pub config: crate::config::Config,
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct AddGitHubSourceRequest {
    pub owner: String,
    pub repo: String,
    #[serde(default = "default_branch")]
    pub branch: String,
}

#[derive(Debug, Deserialize)]
pub struct AddLocalSourceRequest {
    pub path: String,
}

fn default_limit() -> usize {
    5
}

fn default_branch() -> String {
    "main".to_string()
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct AddSourceResponse {
    pub source: String,
    pub chunks_indexed: usize,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/query", post(query))
        .route("/search", post(search))
        .route("/sources", get(list_sources))
        .route("/sources/github", post(add_github_source))
        .route("/sources/local", post(add_local_source))
        .route("/sources/{source}", delete(delete_source))
        .route("/status", get(status))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn query(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    match state
        .pipeline
        .query_with_sources(&req.query, req.limit)
        .await
    {
        Ok(response) => Json(response).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn search(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    match state.pipeline.search(&req.query, req.limit).await {
        Ok(results) => Json(results).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn list_sources(state: axum::extract::State<Arc<AppState>>) -> impl IntoResponse {
    match state.pipeline.vector_store().list_sources().await {
        Ok(sources) => Json(sources).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn add_github_source(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<AddGitHubSourceRequest>,
) -> impl IntoResponse {
    let repos_path = state
        .config
        .data_dir()
        .join(&state.config.sources.repos_path);

    if let Err(e) = std::fs::create_dir_all(&repos_path) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to create repos directory: {}", e),
            }),
        )
            .into_response();
    }

    let loader = crate::loaders::GitHubLoader::new(
        repos_path,
        state.config.brain.chunk_size,
        state.config.brain.chunk_overlap,
    );

    let repo_path = match loader.clone_or_update(&req.owner, &req.repo, &req.branch) {
        Ok(path) => path,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to clone/update repository: {}", e),
                }),
            )
                .into_response();
        }
    };

    let documents = match loader.load_repo(&repo_path) {
        Ok(docs) => docs,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to load repository: {}", e),
                }),
            )
                .into_response();
        }
    };

    let source_name = format!("github:{}/{}", req.owner, req.repo);

    let count = match index_documents(
        &state.pipeline,
        &source_name,
        crate::storage::SourceType::GitHub,
        documents,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to index documents: {}", e),
                }),
            )
                .into_response();
        }
    };

    if let Err(e) = save_source_metadata(
        &state.config,
        &source_name,
        &req.owner,
        &req.repo,
        &req.branch,
        &repo_path,
    ) {
        tracing::warn!("Failed to save metadata: {}", e);
    }

    Json(AddSourceResponse {
        source: source_name,
        chunks_indexed: count,
    })
    .into_response()
}

async fn add_local_source(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<AddLocalSourceRequest>,
) -> impl IntoResponse {
    let path = std::path::Path::new(&req.path);

    if !path.exists() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Path does not exist: {}", req.path),
            }),
        )
            .into_response();
    }

    let loader = crate::loaders::LocalLoader::new(
        state.config.brain.chunk_size,
        state.config.brain.chunk_overlap,
    );

    let documents = match loader.load_directory(path) {
        Ok(docs) => docs,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to load directory: {}", e),
                }),
            )
                .into_response();
        }
    };

    let source_name = format!("local:{}", path.display());

    let count = match index_documents(
        &state.pipeline,
        &source_name,
        crate::storage::SourceType::Local,
        documents,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to index documents: {}", e),
                }),
            )
                .into_response();
        }
    };

    Json(AddSourceResponse {
        source: source_name,
        chunks_indexed: count,
    })
    .into_response()
}

async fn index_documents(
    pipeline: &crate::rag::RagPipeline,
    source: &str,
    source_type: crate::storage::SourceType,
    documents: Vec<(String, String, Vec<crate::loaders::chunker::Chunk>)>,
) -> anyhow::Result<usize> {
    let mut total_indexed = 0;
    let mut batch = Vec::new();

    for (_source_name, file_path, chunks) in documents {
        for chunk in chunks {
            let id = uuid::Uuid::new_v4().to_string();
            let embedding = pipeline.embed(&chunk.content)?;

            batch.push(crate::storage::DocumentWithEmbedding {
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
            }
        }
    }

    if !batch.is_empty() {
        pipeline.insert_batch(batch).await?;
    }

    Ok(total_indexed)
}

fn save_source_metadata(
    config: &crate::config::Config,
    _source: &str,
    owner: &str,
    repo: &str,
    branch: &str,
    repo_path: &std::path::Path,
) -> anyhow::Result<()> {
    let scheduler = crate::scheduler::Scheduler::new(
        config.data_dir().join(&config.scheduler.metadata_file),
        config.scheduler.check_interval_hours,
        config.scheduler.download_window_start,
        config.scheduler.download_window_end,
        &config.scheduler.timezone,
    )?;

    let mut metadata_store = scheduler.load_metadata()?;
    let metadata = scheduler.create_github_metadata(owner, repo, branch, repo_path)?;
    metadata_store.upsert(metadata);
    scheduler.save_metadata(&metadata_store)?;

    Ok(())
}

async fn delete_source(
    state: axum::extract::State<Arc<AppState>>,
    axum::extract::Path(source): axum::extract::Path<String>,
) -> impl IntoResponse {
    match state
        .pipeline
        .vector_store()
        .delete_by_source(&source)
        .await
    {
        Ok(_) => Json(serde_json::json!({ "deleted": source })).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn status(state: axum::extract::State<Arc<AppState>>) -> impl IntoResponse {
    match state.pipeline.vector_store().count().await {
        Ok(count) => Json(serde_json::json!({
            "status": "ok",
            "document_count": count
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

pub async fn run_server(state: Arc<AppState>, host: &str, port: u16) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    let app = create_router(state);

    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
