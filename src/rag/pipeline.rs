use anyhow::Result;
use crate::embedding::EmbeddingModel;
use crate::storage::{VectorStore, SearchResult, DocumentWithEmbedding};
use crate::rag::MistralRsClient;
use crate::rag::client::Message;
use serde::Serialize;

pub struct RagPipeline {
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    llm_client: MistralRsClient,
}

impl RagPipeline {
    pub fn new(
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        llm_client: MistralRsClient,
    ) -> Self {
        Self {
            embedding_model,
            vector_store,
            llm_client,
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embedding_model.embed_one(text)
    }

    pub async fn insert_batch(&self, documents: Vec<DocumentWithEmbedding>) -> Result<()> {
        self.vector_store.insert(documents).await
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embedding_model.embed_one(query)?;
        let results = self.vector_store.search(&query_embedding, limit).await?;
        Ok(results)
    }

    pub async fn query(&self, query: &str, context_limit: usize) -> Result<String> {
        let results = self.search(query, context_limit).await?;

        if results.is_empty() {
            return self.llm_client.complete(query).await;
        }

        let context = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "[Context {} - {}]:\n{}\n",
                    i + 1,
                    r.file_path,
                    r.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n");

        let system_prompt = r#"You are a helpful assistant that answers questions based on the provided context.
Use the context to provide accurate and relevant answers.
If the context doesn't contain enough information to answer the question, say so.
Always cite which context(s) you used in your answer."#;

        let user_prompt = format!(
            "Context:\n{}\n\nQuestion: {}\n\nPlease answer the question using the context provided.",
            context, query
        );

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: user_prompt,
            },
        ];

        self.llm_client.chat(messages).await
    }

    pub async fn query_with_sources(
        &self,
        query: &str,
        context_limit: usize,
    ) -> Result<QueryResponse> {
        let results = self.search(query, context_limit).await?;

        let sources: Vec<SourceInfo> = results
            .iter()
            .map(|r| SourceInfo {
                source: r.source.clone(),
                file_path: r.file_path.clone(),
                content_preview: r.content.chars().take(200).collect(),
            })
            .collect();

        let answer = if results.is_empty() {
            self.llm_client.complete(query).await?
        } else {
            let context = results
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    format!(
                        "[Context {} - {}]:\n{}\n",
                        i + 1,
                        r.file_path,
                        r.content
                    )
                })
                .collect::<Vec<_>>()
                .join("\n---\n");

            let system_prompt = r#"You are a helpful assistant that answers questions based on the provided context.
Use the context to provide accurate and relevant answers.
If the context doesn't contain enough information to answer the question, say so.
Always cite which context(s) you used in your answer."#;

            let user_prompt = format!(
                "Context:\n{}\n\nQuestion: {}\n\nPlease answer the question using the context provided.",
                context, query
            );

            let messages = vec![
                Message {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ];

            self.llm_client.chat(messages).await?
        };

        Ok(QueryResponse {
            answer,
            sources,
        })
    }

    pub fn vector_store(&self) -> &VectorStore {
        &self.vector_store
    }
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub answer: String,
    pub sources: Vec<SourceInfo>,
}

#[derive(Debug, Serialize)]
pub struct SourceInfo {
    pub source: String,
    pub file_path: String,
    pub content_preview: String,
}
