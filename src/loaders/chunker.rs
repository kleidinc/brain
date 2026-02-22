pub struct TextChunker {
    chunk_size: usize,
    chunk_overlap: usize,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub index: usize,
}

impl TextChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < words.len() {
            let end = (start + self.chunk_size).min(words.len());
            let chunk_words = &words[start..end];
            let content = chunk_words.join(" ");

            if !content.trim().is_empty() {
                chunks.push(Chunk { content, index });
                index += 1;
            }

            if end >= words.len() {
                break;
            }

            start = end.saturating_sub(self.chunk_overlap);
        }

        chunks
    }

    pub fn chunk_by_paragraphs(&self, text: &str) -> Vec<Chunk> {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_size = 0;
        let mut index = 0;

        for para in paragraphs {
            let para_words = para.split_whitespace().count();

            if current_size + para_words > self.chunk_size && !current_chunk.is_empty() {
                if !current_chunk.trim().is_empty() {
                    chunks.push(Chunk {
                        content: current_chunk.trim().to_string(),
                        index,
                    });
                    index += 1;
                }
                current_chunk = String::new();
                current_size = 0;
            }

            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(para);
            current_size += para_words;
        }

        if !current_chunk.trim().is_empty() {
            chunks.push(Chunk {
                content: current_chunk.trim().to_string(),
                index,
            });
        }

        chunks
    }

    pub fn chunk_code(&self, code: &str) -> Vec<Chunk> {
        let lines: Vec<&str> = code.lines().collect();
        let mut chunks = Vec::new();
        let mut current_lines: Vec<&str> = Vec::new();
        let mut current_size = 0;
        let mut index = 0;

        for line in lines {
            let line_words = line.split_whitespace().count();

            if current_size + line_words > self.chunk_size && !current_lines.is_empty() {
                let content = current_lines.join("\n");
                if !content.trim().is_empty() {
                    chunks.push(Chunk { content, index });
                    index += 1;
                }

                let overlap_start = current_lines.len().saturating_sub(5);
                current_lines = current_lines[overlap_start..].to_vec();
                current_size = current_lines
                    .iter()
                    .map(|l: &&str| l.split_whitespace().count())
                    .sum();
            }

            current_lines.push(line);
            current_size += line_words;
        }

        if !current_lines.is_empty() {
            let content = current_lines.join("\n");
            if !content.trim().is_empty() {
                chunks.push(Chunk { content, index });
            }
        }

        chunks
    }
}
