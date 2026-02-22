use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use arrow_array::{StringArray, Int64Array, Float32Array, FixedSizeListArray, RecordBatch, Array, types::Float32Type, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lancedb::connection::connect;
use lancedb::query::{QueryBase, ExecutableQuery};
use futures::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub source: String,
    pub source_type: SourceType,
    pub file_path: Option<String>,
    pub chunk_index: usize,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceType {
    GitHub,
    Local,
    Manual,
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceType::GitHub => write!(f, "github"),
            SourceType::Local => write!(f, "local"),
            SourceType::Manual => write!(f, "manual"),
        }
    }
}

impl std::str::FromStr for SourceType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "github" => Ok(SourceType::GitHub),
            "local" => Ok(SourceType::Local),
            "manual" => Ok(SourceType::Manual),
            _ => Err(format!("Unknown source type: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentWithEmbedding {
    pub id: String,
    pub content: String,
    pub source: String,
    pub source_type: String,
    pub file_path: String,
    pub chunk_index: i64,
    pub created_at: String,
    pub embedding: Vec<f32>,
}

pub struct VectorStore {
    db: lancedb::connection::Connection,
    table_name: String,
    dimensions: usize,
}

impl VectorStore {
    pub async fn new(db_path: &Path, table_name: &str, dimensions: usize) -> Result<Self> {
        let db = connect(db_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?).execute().await?;
        
        let store = Self {
            db,
            table_name: table_name.to_string(),
            dimensions,
        };

        store.ensure_table().await?;
        Ok(store)
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("source", DataType::Utf8, false),
            Field::new("source_type", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, true),
            Field::new("chunk_index", DataType::Int64, false),
            Field::new("created_at", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dimensions as i32,
                ),
                false,
            ),
        ]))
    }

    async fn ensure_table(&self) -> Result<()> {
        let tables = self.db.table_names().execute().await?;
        if !tables.contains(&self.table_name) {
            let schema = self.schema();
            let batch = RecordBatch::new_empty(schema.clone());
            let batches: Vec<Result<RecordBatch, arrow_schema::ArrowError>> = vec![Ok(batch)];
            let reader = RecordBatchIterator::new(batches.into_iter(), schema);
            
            self.db
                .create_table(&self.table_name, reader)
                .execute()
                .await?;
            
            tracing::info!("Created table: {}", self.table_name);
        }

        Ok(())
    }

    pub async fn insert(&self, documents: Vec<DocumentWithEmbedding>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        let schema = self.schema();

        let ids: StringArray = StringArray::from_iter_values(documents.iter().map(|d| d.id.as_str()));
        let contents: StringArray = StringArray::from_iter_values(documents.iter().map(|d| d.content.as_str()));
        let sources: StringArray = StringArray::from_iter_values(documents.iter().map(|d| d.source.as_str()));
        let source_types: StringArray = StringArray::from_iter_values(documents.iter().map(|d| d.source_type.as_str()));
        let file_paths: StringArray = StringArray::from_iter_values(documents.iter().map(|d| d.file_path.as_str()));
        let chunk_indices: Int64Array = documents.iter().map(|d| d.chunk_index).collect();
        let created_ats: StringArray = StringArray::from_iter_values(documents.iter().map(|d| d.created_at.as_str()));

        let embeddings = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            documents.iter().map(|d| Some(d.embedding.iter().map(|&v| Some(v)).collect::<Vec<_>>())),
            self.dimensions as i32,
        );

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(contents),
                Arc::new(sources),
                Arc::new(source_types),
                Arc::new(file_paths),
                Arc::new(chunk_indices),
                Arc::new(created_ats),
                Arc::new(embeddings),
            ],
        )?;

        let table = self.db.open_table(&self.table_name).execute().await?;
        let batches: Vec<Result<RecordBatch, arrow_schema::ArrowError>> = vec![Ok(batch)];
        let reader = RecordBatchIterator::new(batches.into_iter(), self.schema());
        table.add(reader).execute().await?;
        
        tracing::info!("Inserted {} documents", documents.len());
        Ok(())
    }

    pub async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let table = self.db.open_table(&self.table_name).execute().await?;

        let query_vec = query_embedding.to_vec();
        
        let mut stream = table
            .query()
            .nearest_to(query_vec)?
            .limit(limit)
            .execute()
            .await?;

        let mut results = Vec::new();
        
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| anyhow::anyhow!("Missing/invalid id column"))?;

            let contents = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| anyhow::anyhow!("Missing/invalid content column"))?;

            let sources = batch
                .column_by_name("source")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| anyhow::anyhow!("Missing/invalid source column"))?;

            let file_paths = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| anyhow::anyhow!("Missing/invalid file_path column"))?;

            for i in 0..batch.num_rows() {
                results.push(SearchResult {
                    id: ids.value(i).to_string(),
                    content: contents.value(i).to_string(),
                    source: sources.value(i).to_string(),
                    file_path: file_paths.value(i).to_string(),
                });
            }
        }

        Ok(results)
    }

    pub async fn delete_by_source(&self, source: &str) -> Result<()> {
        let table = self.db.open_table(&self.table_name).execute().await?;
        let predicate = format!("source = '{}'", source);
        table.delete(&predicate).await?;
        tracing::info!("Deleted documents from source: {}", source);
        Ok(())
    }

    pub async fn list_sources(&self) -> Result<Vec<String>> {
        let table = self.db.open_table(&self.table_name).execute().await?;

        let mut stream = table
            .query()
            .select(lancedb::query::Select::columns(&["source"]))
            .execute()
            .await?;

        let mut sources = std::collections::HashSet::new();

        while let Some(batch) = stream.next().await {
            let batch = batch?;
            if let Some(col) = batch.column_by_name("source") {
                if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                    for i in 0..arr.len() {
                        sources.insert(arr.value(i).to_string());
                    }
                }
            }
        }

        Ok(sources.into_iter().collect())
    }

    pub async fn count(&self) -> Result<usize> {
        let table = self.db.open_table(&self.table_name).execute().await?;
        Ok(table.count_rows(None).await?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub source: String,
    pub file_path: String,
}
