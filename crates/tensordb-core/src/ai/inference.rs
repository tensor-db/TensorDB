use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::{Mutex, RwLock};

use crate::util::time::unix_millis;

// ---------------------------------------------------------------------------
// 1. InferenceEngine
// ---------------------------------------------------------------------------

/// Activation function applied element-wise after the linear transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Linear: output = Wx + b (identity).
    None,
    /// Sigmoid: 1 / (1 + exp(-x)).
    Sigmoid,
    /// ReLU: max(0, x).
    ReLU,
    /// Softmax: exp(x_i) / sum(exp(x_j)) over all outputs.
    Softmax,
}

/// A simple linear scoring model: output = activation(W * input + bias).
pub struct ScoringFunction {
    pub name: String,
    pub input_dims: usize,
    pub output_dims: usize,
    /// Row-major weight matrix of shape [output_dims x input_dims].
    pub weights: Vec<f64>,
    /// Bias vector of length output_dims.
    pub bias: Vec<f64>,
    pub activation: Activation,
    pub created_at: u64,
}

/// Record of a single inference invocation.
#[derive(Debug, Clone)]
pub struct InferenceRecord {
    pub scorer_name: String,
    pub input_hash: u64,
    pub output: Vec<f64>,
    pub latency_ns: u64,
    pub timestamp: u64,
}

/// Summary info for a registered scorer (returned by `list_scorers`).
#[derive(Debug, Clone)]
pub struct ScorerInfo {
    pub name: String,
    pub input_dims: usize,
    pub output_dims: usize,
    pub activation: Activation,
    pub created_at: u64,
}

/// Aggregate inference statistics.
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub total_inferences: u64,
    pub total_errors: u64,
    pub registered_scorers: usize,
    pub history_size: usize,
}

/// A lightweight in-process inference engine for running simple linear models.
pub struct InferenceEngine {
    /// Registered scoring functions (name -> function).
    scorers: RwLock<HashMap<String, ScoringFunction>>,
    /// Inference history for monitoring.
    history: Mutex<VecDeque<InferenceRecord>>,
    /// Max history size.
    max_history: usize,
    /// Stats
    total_inferences: AtomicU64,
    total_errors: AtomicU64,
}

impl InferenceEngine {
    pub fn new(max_history: usize) -> Self {
        Self {
            scorers: RwLock::new(HashMap::new()),
            history: Mutex::new(VecDeque::new()),
            max_history,
            total_inferences: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
        }
    }

    /// Register a linear scoring model.
    pub fn register_scorer(
        &self,
        name: String,
        input_dims: usize,
        output_dims: usize,
        weights: Vec<f64>,
        bias: Vec<f64>,
        activation: Activation,
    ) -> Result<(), String> {
        if weights.len() != output_dims * input_dims {
            return Err(format!(
                "weights length {} != output_dims({}) * input_dims({})",
                weights.len(),
                output_dims,
                input_dims
            ));
        }
        if bias.len() != output_dims {
            return Err(format!(
                "bias length {} != output_dims({})",
                bias.len(),
                output_dims
            ));
        }
        let scorer = ScoringFunction {
            name: name.clone(),
            input_dims,
            output_dims,
            weights,
            bias,
            activation,
            created_at: unix_millis(),
        };
        self.scorers.write().insert(name, scorer);
        Ok(())
    }

    /// Run inference on a single input vector.
    pub fn infer(&self, scorer_name: &str, input: &[f64]) -> Result<Vec<f64>, String> {
        let start = Instant::now();
        let scorers = self.scorers.read();
        let scorer = scorers
            .get(scorer_name)
            .ok_or_else(|| format!("scorer '{}' not found", scorer_name))?;

        if input.len() != scorer.input_dims {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
            return Err(format!(
                "input dimension mismatch: expected {}, got {}",
                scorer.input_dims,
                input.len()
            ));
        }

        let output = compute_linear(
            &scorer.weights,
            &scorer.bias,
            input,
            scorer.input_dims,
            scorer.output_dims,
            scorer.activation,
        );

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.total_inferences.fetch_add(1, Ordering::Relaxed);

        let record = InferenceRecord {
            scorer_name: scorer_name.to_string(),
            input_hash: simple_hash(input),
            output: output.clone(),
            latency_ns,
            timestamp: unix_millis(),
        };
        let mut hist = self.history.lock();
        if hist.len() >= self.max_history {
            hist.pop_front();
        }
        hist.push_back(record);

        Ok(output)
    }

    /// Batch inference: run the same scorer on multiple inputs.
    pub fn batch_infer(
        &self,
        scorer_name: &str,
        inputs: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.infer(scorer_name, input)?);
        }
        Ok(results)
    }

    /// List all registered scorers.
    pub fn list_scorers(&self) -> Vec<ScorerInfo> {
        let scorers = self.scorers.read();
        let mut infos: Vec<ScorerInfo> = scorers
            .values()
            .map(|s| ScorerInfo {
                name: s.name.clone(),
                input_dims: s.input_dims,
                output_dims: s.output_dims,
                activation: s.activation,
                created_at: s.created_at,
            })
            .collect();
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        infos
    }

    /// Remove a scorer by name. Returns true if it existed.
    pub fn remove_scorer(&self, name: &str) -> bool {
        self.scorers.write().remove(name).is_some()
    }

    /// Return the most recent inference records (up to `limit`).
    pub fn recent_inferences(&self, limit: usize) -> Vec<InferenceRecord> {
        let hist = self.history.lock();
        hist.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            total_inferences: self.total_inferences.load(Ordering::Relaxed),
            total_errors: self.total_errors.load(Ordering::Relaxed),
            registered_scorers: self.scorers.read().len(),
            history_size: self.history.lock().len(),
        }
    }
}

/// Compute output = activation(W * input + bias).
fn compute_linear(
    weights: &[f64],
    bias: &[f64],
    input: &[f64],
    input_dims: usize,
    output_dims: usize,
    activation: Activation,
) -> Vec<f64> {
    let mut output = Vec::with_capacity(output_dims);
    for (j, &b) in bias.iter().enumerate().take(output_dims) {
        let mut val = b;
        let row_start = j * input_dims;
        for i in 0..input_dims {
            val += weights[row_start + i] * input[i];
        }
        output.push(val);
    }
    apply_activation(&mut output, activation);
    output
}

/// Apply activation function element-wise (in-place).
fn apply_activation(output: &mut [f64], activation: Activation) {
    match activation {
        Activation::None => {}
        Activation::Sigmoid => {
            for v in output.iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        }
        Activation::ReLU => {
            for v in output.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        Activation::Softmax => {
            if output.is_empty() {
                return;
            }
            let max_val = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for v in output.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in output.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }
}

/// Simple hash for input vectors (used for tracking, not security).
fn simple_hash(input: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in input {
        let bits = v.to_bits();
        h ^= bits;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// 2. RAG Pipeline
// ---------------------------------------------------------------------------

/// A single chunk of a document.
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub chunk_id: String,
    pub document_id: String,
    pub text: String,
    pub chunk_index: usize,
    pub embedding: Option<Vec<f32>>,
    pub metadata: serde_json::Value,
}

/// Result of a retrieval query.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub chunk_id: String,
    pub document_id: String,
    pub text: String,
    pub score: f64,
    pub method: RetrievalMethod,
}

/// Which retrieval strategy produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMethod {
    Vector,
    Keyword,
    Hybrid,
}

/// Aggregate statistics for the RAG pipeline.
#[derive(Debug, Clone)]
pub struct RagStats {
    pub total_documents: u64,
    pub total_chunks: u64,
    pub total_retrievals: u64,
    pub term_index_size: usize,
}

/// Retrieval-Augmented Generation pipeline.
///
/// Combines document chunking, TF-IDF keyword search, and vector similarity
/// search for flexible document retrieval.
pub struct RagPipeline {
    /// Chunk store: document_id -> chunks.
    chunks: RwLock<HashMap<String, Vec<DocumentChunk>>>,
    /// Simple TF-IDF-like term index for keyword matching.
    /// term -> [(chunk_id, frequency)]
    term_index: RwLock<HashMap<String, Vec<(String, usize)>>>,
    /// Stats
    total_documents: AtomicU64,
    total_chunks: AtomicU64,
    total_retrievals: AtomicU64,
}

impl RagPipeline {
    pub fn new() -> Self {
        Self {
            chunks: RwLock::new(HashMap::new()),
            term_index: RwLock::new(HashMap::new()),
            total_documents: AtomicU64::new(0),
            total_chunks: AtomicU64::new(0),
            total_retrievals: AtomicU64::new(0),
        }
    }

    /// Ingest a document by splitting it into chunks and indexing terms.
    /// Returns the number of chunks created.
    pub fn ingest_document(
        &self,
        document_id: &str,
        text: &str,
        chunk_size: usize,
    ) -> Result<usize, String> {
        if text.is_empty() {
            return Err("empty document text".to_string());
        }
        let chunk_size = chunk_size.max(1);
        let raw_chunks = split_into_chunks(text, chunk_size);
        let mut doc_chunks = Vec::with_capacity(raw_chunks.len());
        for (i, chunk_text) in raw_chunks.into_iter().enumerate() {
            let chunk_id = format!("{document_id}_chunk_{i}");
            doc_chunks.push(DocumentChunk {
                chunk_id,
                document_id: document_id.to_string(),
                text: chunk_text,
                chunk_index: i,
                embedding: None,
                metadata: serde_json::Value::Null,
            });
        }
        let chunk_count = doc_chunks.len();

        // Build term index for these chunks.
        {
            let mut idx = self.term_index.write();
            for chunk in &doc_chunks {
                let terms = tokenize(&chunk.text);
                let mut freq_map: HashMap<String, usize> = HashMap::new();
                for term in &terms {
                    *freq_map.entry(term.clone()).or_default() += 1;
                }
                for (term, freq) in freq_map {
                    idx.entry(term)
                        .or_default()
                        .push((chunk.chunk_id.clone(), freq));
                }
            }
        }

        self.chunks
            .write()
            .insert(document_id.to_string(), doc_chunks);
        self.total_documents.fetch_add(1, Ordering::Relaxed);
        self.total_chunks
            .fetch_add(chunk_count as u64, Ordering::Relaxed);
        Ok(chunk_count)
    }

    /// Ingest a pre-chunked document fragment.
    pub fn ingest_chunk(
        &self,
        document_id: &str,
        chunk_id: &str,
        text: &str,
        embedding: Option<Vec<f32>>,
        metadata: serde_json::Value,
    ) -> Result<(), String> {
        let chunk = DocumentChunk {
            chunk_id: chunk_id.to_string(),
            document_id: document_id.to_string(),
            text: text.to_string(),
            chunk_index: 0,
            embedding,
            metadata,
        };

        // Index terms.
        {
            let terms = tokenize(text);
            let mut freq_map: HashMap<String, usize> = HashMap::new();
            for term in &terms {
                *freq_map.entry(term.clone()).or_default() += 1;
            }
            let mut idx = self.term_index.write();
            for (term, freq) in freq_map {
                idx.entry(term)
                    .or_default()
                    .push((chunk_id.to_string(), freq));
            }
        }

        let mut chunks = self.chunks.write();
        let doc_entry = chunks.entry(document_id.to_string()).or_default();
        let next_index = doc_entry.len();
        let mut chunk = chunk;
        chunk.chunk_index = next_index;
        doc_entry.push(chunk);

        // Only increment document count if this is a new document.
        if next_index == 0 {
            self.total_documents.fetch_add(1, Ordering::Relaxed);
        }
        self.total_chunks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// TF-IDF keyword search. Returns top_k results sorted by score descending.
    pub fn retrieve_by_keyword(&self, query: &str, top_k: usize) -> Vec<RetrievalResult> {
        self.total_retrievals.fetch_add(1, Ordering::Relaxed);
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        // Accumulate scores per chunk_id.
        let mut scores: HashMap<String, f64> = HashMap::new();
        let idx = self.term_index.read();
        let total_chunks_count = self.total_chunks.load(Ordering::Relaxed).max(1) as f64;

        for term in &query_terms {
            if let Some(postings) = idx.get(term) {
                // IDF = log(N / df) where df = number of chunks containing the term.
                let df = postings.len() as f64;
                let idf = (total_chunks_count / df).ln().max(0.0);
                for (chunk_id, freq) in postings {
                    let tf = *freq as f64;
                    *scores.entry(chunk_id.clone()).or_default() += tf * idf;
                }
            }
        }

        // Sort by score descending, take top_k.
        let mut results: Vec<(String, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        // Resolve chunk text.
        let chunks = self.chunks.read();
        results
            .into_iter()
            .filter_map(|(chunk_id, score)| {
                self.find_chunk_by_id(&chunks, &chunk_id)
                    .map(|c| RetrievalResult {
                        chunk_id: c.chunk_id.clone(),
                        document_id: c.document_id.clone(),
                        text: c.text.clone(),
                        score,
                        method: RetrievalMethod::Keyword,
                    })
            })
            .collect()
    }

    /// Vector similarity search over stored chunks that have embeddings.
    /// Uses cosine similarity.
    pub fn retrieve_by_embedding(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<RetrievalResult> {
        self.total_retrievals.fetch_add(1, Ordering::Relaxed);
        let chunks = self.chunks.read();
        let mut scored: Vec<RetrievalResult> = Vec::new();

        for doc_chunks in chunks.values() {
            for chunk in doc_chunks {
                if let Some(ref emb) = chunk.embedding {
                    let sim = cosine_similarity(query_embedding, emb);
                    scored.push(RetrievalResult {
                        chunk_id: chunk.chunk_id.clone(),
                        document_id: chunk.document_id.clone(),
                        text: chunk.text.clone(),
                        score: sim,
                        method: RetrievalMethod::Vector,
                    });
                }
            }
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }

    /// Hybrid retrieval: weighted combination of keyword + vector scores.
    /// `keyword_weight` in [0.0, 1.0]; vector weight = 1.0 - keyword_weight.
    pub fn hybrid_retrieve(
        &self,
        keyword_query: &str,
        query_embedding: &[f32],
        top_k: usize,
        keyword_weight: f64,
    ) -> Vec<RetrievalResult> {
        let keyword_weight = keyword_weight.clamp(0.0, 1.0);
        let vector_weight = 1.0 - keyword_weight;

        let keyword_results = self.retrieve_by_keyword(keyword_query, top_k * 2);
        let vector_results = self.retrieve_by_embedding(query_embedding, top_k * 2);

        // Merge scores by chunk_id.
        let mut combined: HashMap<String, (f64, Option<RetrievalResult>)> = HashMap::new();

        // Normalize keyword scores.
        let kw_max = keyword_results
            .iter()
            .map(|r| r.score)
            .fold(0.0f64, f64::max)
            .max(1e-9);
        for r in &keyword_results {
            let norm_score = (r.score / kw_max) * keyword_weight;
            let entry = combined.entry(r.chunk_id.clone()).or_insert((0.0, None));
            entry.0 += norm_score;
            if entry.1.is_none() {
                entry.1 = Some(r.clone());
            }
        }

        // Normalize vector scores (cosine similarity is already in [-1, 1]).
        let vec_max = vector_results
            .iter()
            .map(|r| r.score)
            .fold(0.0f64, f64::max)
            .max(1e-9);
        for r in &vector_results {
            let norm_score = (r.score / vec_max) * vector_weight;
            let entry = combined.entry(r.chunk_id.clone()).or_insert((0.0, None));
            entry.0 += norm_score;
            if entry.1.is_none() {
                entry.1 = Some(r.clone());
            }
        }

        let mut results: Vec<RetrievalResult> = combined
            .into_iter()
            .filter_map(|(_chunk_id, (score, result))| {
                result.map(|mut r| {
                    r.score = score;
                    r.method = RetrievalMethod::Hybrid;
                    r
                })
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Get all chunks for a document.
    pub fn get_document_chunks(&self, document_id: &str) -> Vec<DocumentChunk> {
        let chunks = self.chunks.read();
        chunks.get(document_id).cloned().unwrap_or_default()
    }

    /// Delete a document and all its chunks. Returns the number of chunks removed.
    pub fn delete_document(&self, document_id: &str) -> usize {
        let removed_chunks = {
            let mut chunks = self.chunks.write();
            chunks.remove(document_id).unwrap_or_default()
        };
        let count = removed_chunks.len();

        // Remove term index entries for deleted chunks.
        if count > 0 {
            let chunk_ids: std::collections::HashSet<String> =
                removed_chunks.iter().map(|c| c.chunk_id.clone()).collect();
            let mut idx = self.term_index.write();
            idx.retain(|_term, postings| {
                postings.retain(|(cid, _)| !chunk_ids.contains(cid));
                !postings.is_empty()
            });
        }

        count
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> RagStats {
        RagStats {
            total_documents: self.total_documents.load(Ordering::Relaxed),
            total_chunks: self.total_chunks.load(Ordering::Relaxed),
            total_retrievals: self.total_retrievals.load(Ordering::Relaxed),
            term_index_size: self.term_index.read().len(),
        }
    }

    /// Look up a chunk by ID across all documents.
    fn find_chunk_by_id<'a>(
        &self,
        chunks: &'a HashMap<String, Vec<DocumentChunk>>,
        chunk_id: &str,
    ) -> Option<&'a DocumentChunk> {
        for doc_chunks in chunks.values() {
            for chunk in doc_chunks {
                if chunk.chunk_id == chunk_id {
                    return Some(chunk);
                }
            }
        }
        None
    }
}

impl Default for RagPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Split text into chunks on sentence boundaries (`. ` or `\n\n`).
/// Falls back to word boundaries if no sentence boundary is found within chunk_size.
fn split_into_chunks(text: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= chunk_size {
            let trimmed = remaining.trim();
            if !trimmed.is_empty() {
                chunks.push(trimmed.to_string());
            }
            break;
        }

        let window = &remaining[..chunk_size];

        // Try to find a sentence boundary (`. ` or `\n\n`) within the window.
        let boundary = find_last_sentence_boundary(window);

        let split_at = if let Some(pos) = boundary {
            pos
        } else {
            // Fall back to word boundary.
            find_last_word_boundary(window).unwrap_or(chunk_size)
        };

        let split_at = split_at.max(1); // always advance at least 1 char
        let chunk_text = remaining[..split_at].trim();
        if !chunk_text.is_empty() {
            chunks.push(chunk_text.to_string());
        }
        remaining = &remaining[split_at..];
        // Skip leading whitespace for next chunk.
        remaining = remaining.trim_start();
    }

    chunks
}

/// Find the last occurrence of `. ` or `\n\n` in the window, returning the
/// position just after the boundary (so the next chunk starts after it).
fn find_last_sentence_boundary(window: &str) -> Option<usize> {
    let bytes = window.as_bytes();
    let mut best = None;

    // Look for `. `
    for i in (0..bytes.len().saturating_sub(1)).rev() {
        if bytes[i] == b'.' && bytes[i + 1] == b' ' {
            best = Some(i + 2); // include the period + space
            break;
        }
    }

    // Look for `\n\n`
    for i in (0..bytes.len().saturating_sub(1)).rev() {
        if bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            let pos = i + 2;
            if best.is_none() || pos > best.unwrap() {
                best = Some(pos);
            }
            break;
        }
    }

    best
}

/// Find the last whitespace position in the window for word-boundary splitting.
fn find_last_word_boundary(window: &str) -> Option<usize> {
    let bytes = window.as_bytes();
    for i in (0..bytes.len()).rev() {
        if bytes[i].is_ascii_whitespace() {
            return Some(i + 1);
        }
    }
    None
}

/// Tokenize text into lowercase terms, splitting on whitespace and punctuation.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1) // skip single-char tokens
        .map(|s| s.to_string())
        .collect()
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    dot / denom
}

// ---------------------------------------------------------------------------
// 3. RAG Advisor
// ---------------------------------------------------------------------------

/// Statistics for the RAG advisor.
#[derive(Debug, Clone)]
pub struct RagAdvisorStats {
    pub total_retrievals: u64,
    pub method_counts: [u64; 3],
    pub avg_recent_score: f64,
}

/// AI-driven advisor for RAG retrieval strategy selection.
pub struct RagAdvisor {
    /// Recent retrieval quality scores.
    retrieval_scores: VecDeque<f64>,
    /// Method usage counts: [vector, keyword, hybrid].
    method_counts: [u64; 3],
    /// Maximum tracked recent scores.
    max_recent: usize,
}

impl RagAdvisor {
    pub fn new(max_recent: usize) -> Self {
        Self {
            retrieval_scores: VecDeque::new(),
            method_counts: [0; 3],
            max_recent,
        }
    }

    /// Record a retrieval event with its quality score and method.
    pub fn record_retrieval(&mut self, score: f64, method: RetrievalMethod) {
        if self.retrieval_scores.len() >= self.max_recent {
            self.retrieval_scores.pop_front();
        }
        self.retrieval_scores.push_back(score);
        let idx = match method {
            RetrievalMethod::Vector => 0,
            RetrievalMethod::Keyword => 1,
            RetrievalMethod::Hybrid => 2,
        };
        self.method_counts[idx] += 1;
    }

    /// AI decision: recommend the best retrieval method based on recent quality.
    ///
    /// Logic:
    /// - If hybrid has been used most and average scores are good (>= 0.5), prefer hybrid.
    /// - If vector is dominant and average scores are decent (>= 0.4), prefer vector.
    /// - If keyword is dominant and scores are decent, prefer keyword.
    /// - Default to hybrid as the safest general strategy.
    pub fn recommend_method(&self) -> RetrievalMethod {
        let avg = self.avg_recent_score();
        let [vector_count, keyword_count, hybrid_count] = self.method_counts;
        let total = vector_count + keyword_count + hybrid_count;

        if total == 0 {
            return RetrievalMethod::Hybrid;
        }

        // Find the dominant method.
        let max_count = vector_count.max(keyword_count).max(hybrid_count);

        if max_count == hybrid_count && avg >= 0.5 {
            return RetrievalMethod::Hybrid;
        }
        if max_count == vector_count && avg >= 0.4 {
            return RetrievalMethod::Vector;
        }
        if max_count == keyword_count && avg >= 0.4 {
            return RetrievalMethod::Keyword;
        }

        // If scores are low, switch to hybrid as a corrective measure.
        RetrievalMethod::Hybrid
    }

    /// AI-recommended chunk size based on average document length.
    ///
    /// Heuristic: target ~10 chunks per document, with a minimum of 100 chars
    /// and a maximum of 2000 chars per chunk.
    pub fn recommend_chunk_size(&self, avg_doc_length: usize) -> usize {
        let target_chunks = 10;
        let raw = avg_doc_length / target_chunks;
        raw.clamp(100, 2000)
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> RagAdvisorStats {
        RagAdvisorStats {
            total_retrievals: self.method_counts.iter().sum(),
            method_counts: self.method_counts,
            avg_recent_score: self.avg_recent_score(),
        }
    }

    fn avg_recent_score(&self) -> f64 {
        if self.retrieval_scores.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.retrieval_scores.iter().sum();
        sum / self.retrieval_scores.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Inference tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_scorer() {
        let engine = InferenceEngine::new(100);
        let result = engine.register_scorer(
            "linear_2x1".to_string(),
            2,
            1,
            vec![0.5, -0.3],
            vec![0.1],
            Activation::None,
        );
        assert!(result.is_ok());
        let scorers = engine.list_scorers();
        assert_eq!(scorers.len(), 1);
        assert_eq!(scorers[0].name, "linear_2x1");
        assert_eq!(scorers[0].input_dims, 2);
        assert_eq!(scorers[0].output_dims, 1);
    }

    #[test]
    fn test_infer_linear() {
        let engine = InferenceEngine::new(100);
        // y = 0.5*x0 + (-0.3)*x1 + 0.1
        engine
            .register_scorer(
                "lin".to_string(),
                2,
                1,
                vec![0.5, -0.3],
                vec![0.1],
                Activation::None,
            )
            .unwrap();
        let out = engine.infer("lin", &[1.0, 2.0]).unwrap();
        assert_eq!(out.len(), 1);
        // 0.5*1.0 + (-0.3)*2.0 + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
        assert!((out[0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_infer_sigmoid() {
        let engine = InferenceEngine::new(100);
        engine
            .register_scorer(
                "sig".to_string(),
                1,
                1,
                vec![1.0],
                vec![0.0],
                Activation::Sigmoid,
            )
            .unwrap();
        let out = engine.infer("sig", &[0.0]).unwrap();
        // sigmoid(0) = 0.5
        assert!((out[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_infer_relu() {
        let engine = InferenceEngine::new(100);
        engine
            .register_scorer(
                "relu".to_string(),
                2,
                2,
                vec![1.0, 0.0, 0.0, 1.0], // identity matrix
                vec![0.0, 0.0],
                Activation::ReLU,
            )
            .unwrap();
        let out = engine.infer("relu", &[3.0, -2.0]).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.0).abs() < 1e-9);
        assert!((out[1] - 0.0).abs() < 1e-9); // ReLU clips negative
    }

    #[test]
    fn test_infer_softmax() {
        let engine = InferenceEngine::new(100);
        engine
            .register_scorer(
                "sm".to_string(),
                2,
                2,
                vec![1.0, 0.0, 0.0, 1.0], // identity
                vec![0.0, 0.0],
                Activation::Softmax,
            )
            .unwrap();
        let out = engine.infer("sm", &[1.0, 1.0]).unwrap();
        // softmax([1,1]) = [0.5, 0.5]
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        // Sum should be 1.0
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_batch_infer() {
        let engine = InferenceEngine::new(100);
        engine
            .register_scorer(
                "batch".to_string(),
                2,
                1,
                vec![1.0, 1.0],
                vec![0.0],
                Activation::None,
            )
            .unwrap();
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![0.0, 0.0]];
        let results = engine.batch_infer("batch", &inputs).unwrap();
        assert_eq!(results.len(), 3);
        assert!((results[0][0] - 3.0).abs() < 1e-9);
        assert!((results[1][0] - 7.0).abs() < 1e-9);
        assert!((results[2][0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dimension_mismatch() {
        let engine = InferenceEngine::new(100);
        engine
            .register_scorer(
                "dim".to_string(),
                3,
                1,
                vec![1.0, 1.0, 1.0],
                vec![0.0],
                Activation::None,
            )
            .unwrap();
        let result = engine.infer("dim", &[1.0, 2.0]); // 2 dims, expects 3
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn test_remove_scorer() {
        let engine = InferenceEngine::new(100);
        engine
            .register_scorer(
                "remove_me".to_string(),
                1,
                1,
                vec![1.0],
                vec![0.0],
                Activation::None,
            )
            .unwrap();
        assert!(engine.remove_scorer("remove_me"));
        assert!(!engine.remove_scorer("remove_me")); // already removed
        assert!(engine.list_scorers().is_empty());
    }

    #[test]
    fn test_inference_stats() {
        let engine = InferenceEngine::new(100);
        engine
            .register_scorer(
                "s".to_string(),
                1,
                1,
                vec![1.0],
                vec![0.0],
                Activation::None,
            )
            .unwrap();
        let _ = engine.infer("s", &[1.0]);
        let _ = engine.infer("s", &[2.0]);
        let _ = engine.infer("nonexistent", &[1.0]); // error
        let stats = engine.stats();
        assert_eq!(stats.total_inferences, 2);
        assert_eq!(stats.total_errors, 0); // "not found" doesn't increment error counter
        assert_eq!(stats.registered_scorers, 1);
        assert_eq!(stats.history_size, 2);
    }

    // -----------------------------------------------------------------------
    // RAG tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ingest_document() {
        let pipeline = RagPipeline::new();
        let text = "First sentence. Second sentence. Third sentence.";
        let count = pipeline.ingest_document("doc1", text, 50).unwrap();
        assert!(count >= 1);
        let chunks = pipeline.get_document_chunks("doc1");
        assert_eq!(chunks.len(), count);
        assert_eq!(chunks[0].document_id, "doc1");
    }

    #[test]
    fn test_chunk_splitting() {
        let text = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta.";
        let chunks = split_into_chunks(text, 20);
        // Each chunk should be at most ~20 chars, split on sentence boundaries
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_keyword_retrieval() {
        let pipeline = RagPipeline::new();
        pipeline
            .ingest_document("doc1", "Rust programming language is fast and safe", 200)
            .unwrap();
        pipeline
            .ingest_document("doc2", "Python programming language is easy to learn", 200)
            .unwrap();

        let results = pipeline.retrieve_by_keyword("programming language", 5);
        assert!(!results.is_empty());
        // Both documents should appear since both mention "programming language"
        assert!(!results.is_empty());
        assert_eq!(results[0].method, RetrievalMethod::Keyword);
    }

    #[test]
    fn test_retrieve_by_embedding() {
        let pipeline = RagPipeline::new();
        pipeline
            .ingest_chunk(
                "doc1",
                "c1",
                "hello world",
                Some(vec![1.0, 0.0, 0.0]),
                serde_json::Value::Null,
            )
            .unwrap();
        pipeline
            .ingest_chunk(
                "doc1",
                "c2",
                "goodbye world",
                Some(vec![0.0, 1.0, 0.0]),
                serde_json::Value::Null,
            )
            .unwrap();

        let query_emb = vec![1.0, 0.0, 0.0]; // identical to c1's embedding
        let results = pipeline.retrieve_by_embedding(&query_emb, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_id, "c1"); // highest similarity
        assert!((results[0].score - 1.0).abs() < 1e-6); // perfect match
        assert_eq!(results[0].method, RetrievalMethod::Vector);
    }

    #[test]
    fn test_hybrid_retrieve() {
        let pipeline = RagPipeline::new();
        pipeline
            .ingest_chunk(
                "doc1",
                "c1",
                "machine learning algorithms",
                Some(vec![1.0, 0.0]),
                serde_json::Value::Null,
            )
            .unwrap();
        pipeline
            .ingest_chunk(
                "doc1",
                "c2",
                "database indexing structures",
                Some(vec![0.0, 1.0]),
                serde_json::Value::Null,
            )
            .unwrap();

        let results = pipeline.hybrid_retrieve("machine learning", &[0.9, 0.1], 2, 0.5);
        assert!(!results.is_empty());
        assert_eq!(results[0].method, RetrievalMethod::Hybrid);
    }

    #[test]
    fn test_delete_document() {
        let pipeline = RagPipeline::new();
        pipeline
            .ingest_document("doc1", "Some text here. More text here.", 200)
            .unwrap();
        let chunks_before = pipeline.get_document_chunks("doc1");
        assert!(!chunks_before.is_empty());

        let removed = pipeline.delete_document("doc1");
        assert_eq!(removed, chunks_before.len());

        let chunks_after = pipeline.get_document_chunks("doc1");
        assert!(chunks_after.is_empty());
    }

    #[test]
    fn test_term_index_accuracy() {
        let pipeline = RagPipeline::new();
        pipeline
            .ingest_document("doc1", "rust rust rust programming", 200)
            .unwrap();
        let idx = pipeline.term_index.read();
        let rust_postings = idx.get("rust").expect("term 'rust' should be indexed");
        assert_eq!(rust_postings.len(), 1);
        // Frequency of "rust" should be 3.
        assert_eq!(rust_postings[0].1, 3);
    }

    #[test]
    fn test_empty_query() {
        let pipeline = RagPipeline::new();
        pipeline
            .ingest_document("doc1", "some document text", 200)
            .unwrap();
        let results = pipeline.retrieve_by_keyword("", 10);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // RAG Advisor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_recommend_method_after_vector_heavy_usage() {
        let mut advisor = RagAdvisor::new(100);
        // Record many vector retrievals with decent scores.
        for _ in 0..20 {
            advisor.record_retrieval(0.7, RetrievalMethod::Vector);
        }
        // A few keyword retrievals.
        for _ in 0..3 {
            advisor.record_retrieval(0.5, RetrievalMethod::Keyword);
        }
        let rec = advisor.recommend_method();
        assert_eq!(rec, RetrievalMethod::Vector);
    }

    #[test]
    fn test_recommend_chunk_size() {
        let advisor = RagAdvisor::new(100);
        // For a 5000-char average document, expect ~500 chunk size.
        let size = advisor.recommend_chunk_size(5000);
        assert_eq!(size, 500);

        // Small docs should clamp to minimum 100.
        let size = advisor.recommend_chunk_size(50);
        assert_eq!(size, 100);

        // Very large docs should clamp to maximum 2000.
        let size = advisor.recommend_chunk_size(100_000);
        assert_eq!(size, 2000);
    }

    #[test]
    fn test_rag_advisor_stats() {
        let mut advisor = RagAdvisor::new(100);
        advisor.record_retrieval(0.8, RetrievalMethod::Vector);
        advisor.record_retrieval(0.6, RetrievalMethod::Keyword);
        advisor.record_retrieval(0.9, RetrievalMethod::Hybrid);
        let stats = advisor.stats();
        assert_eq!(stats.total_retrievals, 3);
        assert_eq!(stats.method_counts, [1, 1, 1]);
        let expected_avg = (0.8 + 0.6 + 0.9) / 3.0;
        assert!((stats.avg_recent_score - expected_avg).abs() < 1e-9);
    }
}
