use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct MistralRsClient {
    base_url: String,
    model: String,
    client: Client,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

impl MistralRsClient {
    pub fn new(base_url: &str, model: &str, max_tokens: usize, temperature: f32) -> Self {
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client: Client::new(),
            max_tokens,
            temperature,
        }
    }

    pub async fn chat(&self, messages: Vec<Message>) -> Result<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("mistral.rs error: {}", error);
        }

        let chat_response: ChatResponse = response.json().await?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("No response from mistral.rs"))
    }

    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        self.chat(messages).await
    }

    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/v1/models", self.base_url);

        let response = self.client.get(&url).send().await?;

        Ok(response.status().is_success())
    }
}
