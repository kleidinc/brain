use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{Repo, api::sync::Api};
use tokenizers::Tokenizer;

pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimensions: usize,
}

impl EmbeddingModel {
    pub fn new(model_id: &str, cuda_device: usize) -> Result<Self> {
        let device = Device::cuda_if_available(cuda_device)?;
        tracing::info!("Using device: {:?}", device);

        let api = Api::new()?;
        let repo = Repo::model(model_id.to_string());
        let api_repo = api.repo(repo);

        tracing::info!("Downloading model files for {}...", model_id);

        let config_path = api_repo.get("config.json")?;
        let tokenizer_path = api_repo.get("tokenizer.json")?;
        let weights_path = api_repo.get("model.safetensors")?;

        let config: BertConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let dimensions = config.hidden_size;

        tracing::info!("Model config loaded: {} dimensions", dimensions);

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        tracing::info!("Loading model weights...");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
        };

        let model = BertModel::load(vb, &config)?;
        tracing::info!("Model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
            dimensions,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.embed_one(text)?;
            all_embeddings.push(embedding);
        }

        Ok(all_embeddings)
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let truncated_text = if text.len() > 8000 {
            &text[..8000]
        } else {
            text
        };

        let encoded = self
            .tokenizer
            .encode(truncated_text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

        let tokens = encoded.get_ids();
        let attention_mask = encoded.get_attention_mask();

        let max_tokens = 512;
        let tokens: Vec<u32> = if tokens.len() > max_tokens {
            tokens[..max_tokens].to_vec()
        } else {
            tokens.to_vec()
        };
        let attention_mask: Vec<u32> = if attention_mask.len() > max_tokens {
            attention_mask[..max_tokens].to_vec()
        } else {
            attention_mask.to_vec()
        };

        let input_ids = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor =
            Tensor::new(attention_mask.as_slice(), &self.device)?.unsqueeze(0)?;

        let embeddings = self
            .model
            .forward(&input_ids, &attention_mask_tensor, None)?;

        let mean_embedding = embeddings.mean(1)?;
        let mean_embedding = mean_embedding.squeeze(0)?;

        let embedding_vec = mean_embedding.to_vec1::<f32>()?;

        let norm: f32 = embedding_vec.iter().map(|x| x * x).sum();
        let norm = norm.sqrt();
        let normalized_embedding: Vec<f32> = embedding_vec.iter().map(|x| x / norm).collect();

        Ok(normalized_embedding)
    }

    pub fn token_count(&self, text: &str) -> Result<usize> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        Ok(encoded.get_ids().len())
    }
}
