//! BPE tokenizer — loads vocabulary and merge rules from GGUF metadata.
//!
//! Implements byte-pair encoding (BPE) for Qwen-family models, supporting
//! the ~150k+ token vocabulary, special tokens (<|im_start|>, <|im_end|>, etc.),
//! and efficient encode/decode.

use std::collections::HashMap;

use crate::error::{Result, TensorError};

use super::gguf::{GgufFile, GgufValue};

/// A BPE tokenizer loaded from GGUF metadata.
pub struct BpeTokenizer {
    /// token_id → token string
    vocab: Vec<Vec<u8>>,
    /// token string → token_id
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Sorted merge rules: (pair_a, pair_b) with implicit priority by index
    #[allow(dead_code)]
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Merge priority: (token_a, token_b) → rank (lower = higher priority)
    merge_rank: HashMap<(Vec<u8>, Vec<u8>), usize>,
    /// Special token strings → token_id
    special_tokens: HashMap<String, u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    /// Token type array from GGUF (1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte)
    token_types: Vec<u32>,
}

impl BpeTokenizer {
    /// Load tokenizer from GGUF metadata.
    ///
    /// Extracts vocabulary from `tokenizer.ggml.tokens`, merge rules from
    /// `tokenizer.ggml.merges`, and special token IDs from metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Extract vocabulary tokens
        let tokens_val = gguf.get_metadata("tokenizer.ggml.tokens").ok_or_else(|| {
            TensorError::LlmError("missing tokenizer.ggml.tokens in GGUF metadata".into())
        })?;
        let tokens_arr = tokens_val
            .as_array()
            .ok_or_else(|| TensorError::LlmError("tokenizer.ggml.tokens is not an array".into()))?;

        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(tokens_arr.len());
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(tokens_arr.len());

        for (i, val) in tokens_arr.iter().enumerate() {
            let s = match val {
                GgufValue::String(s) => s.as_bytes().to_vec(),
                _ => return Err(TensorError::LlmError(format!("token {i} is not a string"))),
            };
            token_to_id.insert(s.clone(), i as u32);
            vocab.push(s);
        }

        // Extract token types (optional)
        let token_types = if let Some(types_val) = gguf.get_metadata("tokenizer.ggml.token_type") {
            if let Some(types_arr) = types_val.as_array() {
                types_arr.iter().map(|v| v.as_u32().unwrap_or(1)).collect()
            } else {
                vec![1u32; vocab.len()]
            }
        } else {
            vec![1u32; vocab.len()]
        };

        // Extract merge rules
        let mut merges = Vec::new();
        let mut merge_rank = HashMap::new();

        if let Some(merges_val) = gguf.get_metadata("tokenizer.ggml.merges") {
            if let Some(merges_arr) = merges_val.as_array() {
                merges.reserve(merges_arr.len());
                for (rank, val) in merges_arr.iter().enumerate() {
                    if let GgufValue::String(s) = val {
                        if let Some((a, b)) = s.split_once(' ') {
                            let a = a.as_bytes().to_vec();
                            let b = b.as_bytes().to_vec();
                            merge_rank.insert((a.clone(), b.clone()), rank);
                            merges.push((a, b));
                        }
                    }
                }
            }
        }

        // Extract special tokens
        let mut special_tokens = HashMap::new();

        // BOS token
        let bos_token_id = gguf
            .get_metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);

        // EOS token
        let eos_token_id = gguf
            .get_metadata("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        // Register known special tokens by scanning vocab for ChatML tokens
        let chatml_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_sep|>"];
        for token_str in &chatml_tokens {
            if let Some(&id) = token_to_id.get(token_str.as_bytes()) {
                special_tokens.insert(token_str.to_string(), id);
            }
        }

        // Also register BOS/EOS by ID
        if bos_token_id < vocab.len() as u32 {
            let bos_str = String::from_utf8_lossy(&vocab[bos_token_id as usize]).to_string();
            special_tokens.insert(bos_str, bos_token_id);
        }
        if eos_token_id < vocab.len() as u32 {
            let eos_str = String::from_utf8_lossy(&vocab[eos_token_id as usize]).to_string();
            special_tokens.insert(eos_str, eos_token_id);
        }

        Ok(Self {
            vocab,
            token_to_id,
            merges,
            merge_rank,
            special_tokens,
            bos_token_id,
            eos_token_id,
            token_types,
        })
    }

    /// Encode text to token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();

        // Split on special tokens first
        let segments = self.split_on_special_tokens(text);

        for segment in segments {
            match segment {
                Segment::Special(id) => result.push(id),
                Segment::Text(s) => {
                    // Encode each text segment with BPE
                    let tokens = self.bpe_encode(s.as_bytes());
                    result.extend(tokens);
                }
            }
        }

        result
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            if (id as usize) < self.vocab.len() {
                let token_bytes = &self.vocab[id as usize];
                // Check if this is a byte-level token (token_type == 6)
                if (id as usize) < self.token_types.len() && self.token_types[id as usize] == 6 {
                    // Byte-level fallback token, e.g. "<0x41>" → 0x41
                    if let Some(byte_val) = parse_byte_token(token_bytes) {
                        bytes.push(byte_val);
                        continue;
                    }
                }
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Check if a token ID is an end-of-sequence token.
    pub fn is_eos(&self, token_id: u32) -> bool {
        if token_id == self.eos_token_id {
            return true;
        }
        // Also check for <|im_end|> and <|endoftext|>
        if let Some(&im_end_id) = self.special_tokens.get("<|im_end|>") {
            if token_id == im_end_id {
                return true;
            }
        }
        if let Some(&eot_id) = self.special_tokens.get("<|endoftext|>") {
            if token_id == eot_id {
                return true;
            }
        }
        false
    }

    /// Get the special token ID for a given string, if it exists.
    pub fn special_token_id(&self, s: &str) -> Option<u32> {
        self.special_tokens.get(s).copied()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the token string for a given ID.
    pub fn token_str(&self, id: u32) -> Option<&[u8]> {
        self.vocab.get(id as usize).map(|v| v.as_slice())
    }

    /// Get token ID for a string.
    pub fn token_id(&self, s: &[u8]) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    // ── Internal BPE implementation ──────────────────────────────────

    fn split_on_special_tokens<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        if self.special_tokens.is_empty() {
            return vec![Segment::Text(text)];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find the earliest occurring special token
            let mut earliest: Option<(usize, &str, u32)> = None;
            for (token_str, &token_id) in &self.special_tokens {
                if let Some(pos) = remaining.find(token_str.as_str()) {
                    match earliest {
                        None => earliest = Some((pos, token_str.as_str(), token_id)),
                        Some((prev_pos, _, _)) if pos < prev_pos => {
                            earliest = Some((pos, token_str.as_str(), token_id))
                        }
                        _ => {}
                    }
                }
            }

            match earliest {
                Some((pos, token_str, token_id)) => {
                    if pos > 0 {
                        segments.push(Segment::Text(&remaining[..pos]));
                    }
                    segments.push(Segment::Special(token_id));
                    remaining = &remaining[pos + token_str.len()..];
                }
                None => {
                    segments.push(Segment::Text(remaining));
                    break;
                }
            }
        }

        segments
    }

    fn bpe_encode(&self, text: &[u8]) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with each byte as a separate token
        let mut pieces: Vec<Vec<u8>> = text.iter().map(|&b| vec![b]).collect();

        // If individual bytes aren't in vocab, try UTF-8 character boundaries
        if !pieces.is_empty() && !self.token_to_id.contains_key(&pieces[0]) {
            // Try character-level tokenization
            let text_str = String::from_utf8_lossy(text);
            let char_pieces: Vec<Vec<u8>> = text_str
                .chars()
                .map(|c| {
                    let mut buf = [0u8; 4];
                    c.encode_utf8(&mut buf);
                    buf[..c.len_utf8()].to_vec()
                })
                .collect();

            // Check if char-level pieces are in vocab
            if char_pieces.iter().all(|p| self.token_to_id.contains_key(p)) {
                pieces = char_pieces;
            }
        }

        // Iteratively apply the highest-priority merge
        loop {
            if pieces.len() < 2 {
                break;
            }

            // Find the merge pair with the lowest rank (highest priority)
            let mut best_rank = usize::MAX;
            let mut best_pos = usize::MAX;

            for i in 0..pieces.len() - 1 {
                if let Some(&rank) = self
                    .merge_rank
                    .get(&(pieces[i].clone(), pieces[i + 1].clone()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                    }
                }
            }

            if best_pos == usize::MAX {
                break; // No more merges applicable
            }

            // Apply the merge
            let merged = {
                let mut m = pieces[best_pos].clone();
                m.extend_from_slice(&pieces[best_pos + 1]);
                m
            };
            pieces[best_pos] = merged;
            pieces.remove(best_pos + 1);
        }

        // Convert pieces to token IDs
        let mut ids = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            if let Some(&id) = self.token_to_id.get(piece) {
                ids.push(id);
            } else {
                // Fallback: encode individual bytes using byte-level tokens
                for &b in piece {
                    let byte_token = format!("<0x{b:02X}>").into_bytes();
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        ids.push(id);
                    }
                    // If even byte tokens aren't found, skip (shouldn't happen with Qwen vocab)
                }
            }
        }

        ids
    }
}

enum Segment<'a> {
    Text(&'a str),
    Special(u32),
}

/// Parse a byte-level fallback token like "<0x41>" → Some(0x41).
fn parse_byte_token(token: &[u8]) -> Option<u8> {
    // Format: <0xHH> where HH is a hex byte
    if token.len() == 6
        && token[0] == b'<'
        && token[1] == b'0'
        && token[2] == b'x'
        && token[5] == b'>'
    {
        let hi = hex_nibble(token[3])?;
        let lo = hex_nibble(token[4])?;
        Some((hi << 4) | lo)
    } else {
        None
    }
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal tokenizer for testing (no GGUF file needed).
    fn test_tokenizer() -> BpeTokenizer {
        let vocab_strings = vec![
            "a",
            "b",
            "c",
            "d",
            " ",
            "ab",
            "cd",
            "abcd",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ];
        let vocab: Vec<Vec<u8>> = vocab_strings
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let mut token_to_id = HashMap::new();
        for (i, v) in vocab.iter().enumerate() {
            token_to_id.insert(v.clone(), i as u32);
        }

        let merges = vec![
            (b"a".to_vec(), b"b".to_vec()),   // a+b → ab (rank 0)
            (b"c".to_vec(), b"d".to_vec()),   // c+d → cd (rank 1)
            (b"ab".to_vec(), b"cd".to_vec()), // ab+cd → abcd (rank 2)
        ];
        let mut merge_rank = HashMap::new();
        for (rank, (a, b)) in merges.iter().enumerate() {
            merge_rank.insert((a.clone(), b.clone()), rank);
        }

        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|im_start|>".to_string(), 8);
        special_tokens.insert("<|im_end|>".to_string(), 9);
        special_tokens.insert("<|endoftext|>".to_string(), 10);

        BpeTokenizer {
            vocab,
            token_to_id,
            merges,
            merge_rank,
            special_tokens,
            bos_token_id: 0,
            eos_token_id: 10,
            token_types: vec![1; 11],
        }
    }

    #[test]
    fn encode_simple() {
        let tok = test_tokenizer();
        // "ab" should merge a+b → ab (token 5)
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![5]); // "ab" = token 5
    }

    #[test]
    fn encode_with_merges() {
        let tok = test_tokenizer();
        // "abcd" → a+b=ab, c+d=cd, ab+cd=abcd
        let ids = tok.encode("abcd");
        assert_eq!(ids, vec![7]); // "abcd" = token 7
    }

    #[test]
    fn encode_no_merge() {
        let tok = test_tokenizer();
        // "a" → just token 0
        let ids = tok.encode("a");
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn decode_roundtrip() {
        let tok = test_tokenizer();
        let ids = tok.encode("abcd");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "abcd");
    }

    #[test]
    fn special_token_handling() {
        let tok = test_tokenizer();
        let ids = tok.encode("<|im_start|>ab<|im_end|>");
        assert_eq!(ids, vec![8, 5, 9]);
    }

    #[test]
    fn is_eos_checks() {
        let tok = test_tokenizer();
        assert!(tok.is_eos(10)); // <|endoftext|>
        assert!(tok.is_eos(9)); // <|im_end|>
        assert!(!tok.is_eos(0)); // "a"
    }

    #[test]
    fn encode_empty() {
        let tok = test_tokenizer();
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn parse_byte_token_valid() {
        assert_eq!(parse_byte_token(b"<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token(b"<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token(b"<0x00>"), Some(0x00));
    }

    #[test]
    fn parse_byte_token_invalid() {
        assert_eq!(parse_byte_token(b"hello"), None);
        assert_eq!(parse_byte_token(b"<0xGG>"), None);
    }
}
