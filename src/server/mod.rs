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
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    5
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/query", post(query))
        .route("/search", post(search))
        .route("/sources", get(list_sources))
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
