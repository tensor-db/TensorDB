//! Full-text search facet.
//!
//! Provides an inverted index stored as:
//!   `__fts/{table}/{token} → serialized posting list [pk1, pk2, ...]`
//!
//! Tokenizer: splits on whitespace, lowercases, strips punctuation.
//! Supports optional Porter-style stemming (simplified).

use crate::error::{Result, TensorError};

const FTS_PREFIX: &str = "__fts";

/// Generate the storage key for an FTS token posting list.
pub fn fts_token_key(table: &str, token: &str) -> Vec<u8> {
    format!("{FTS_PREFIX}/{table}/{token}").into_bytes()
}

/// Tokenize text for indexing: lowercase, strip punctuation, split on whitespace.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|t| !t.is_empty() && t.len() >= 2) // Skip single-char tokens
        .collect()
}

/// Simplified Porter-style stemming (handles common English suffixes).
pub fn stem(word: &str) -> String {
    let w = word.to_lowercase();
    // Very basic suffix removal
    if w.len() > 5 {
        if let Some(base) = w.strip_suffix("ing") {
            return base.to_string();
        }
        if let Some(base) = w.strip_suffix("tion") {
            return base.to_string();
        }
        if let Some(base) = w.strip_suffix("ness") {
            return base.to_string();
        }
        if let Some(base) = w.strip_suffix("ment") {
            return base.to_string();
        }
    }
    if w.len() > 4 {
        if let Some(base) = w.strip_suffix("ed") {
            if !base.is_empty() {
                return base.to_string();
            }
        }
        if let Some(base) = w.strip_suffix("ly") {
            if !base.is_empty() {
                return base.to_string();
            }
        }
        if let Some(base) = w.strip_suffix("es") {
            if !base.is_empty() {
                return base.to_string();
            }
        }
    }
    if w.len() > 3 {
        if let Some(base) = w.strip_suffix('s') {
            if !base.ends_with('s') {
                return base.to_string();
            }
        }
    }
    w
}

/// Encode a posting list (set of PKs) to bytes.
pub fn encode_posting_list(pks: &[String]) -> Vec<u8> {
    serde_json::to_vec(pks).unwrap_or_default()
}

/// Decode a posting list from bytes.
pub fn decode_posting_list(data: &[u8]) -> Result<Vec<String>> {
    serde_json::from_slice(data).map_err(|e| TensorError::SqlExec(e.to_string()))
}

/// Merge a new PK into a posting list (deduplicating).
pub fn merge_posting(existing: &[u8], pk: &str) -> Vec<u8> {
    let mut list = decode_posting_list(existing).unwrap_or_default();
    if !list.iter().any(|p| p == pk) {
        list.push(pk.to_string());
    }
    encode_posting_list(&list)
}

/// Match query: tokenize the query, intersect posting lists.
/// Returns the set of PKs that match ALL tokens.
pub fn match_query_pks(
    query: &str,
    table: &str,
    lookup: impl Fn(&[u8]) -> Result<Option<Vec<u8>>>,
) -> Result<Vec<String>> {
    let tokens = tokenize(query);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut result_set: Option<Vec<String>> = None;

    for token in &tokens {
        let stemmed = stem(token);
        let key = fts_token_key(table, &stemmed);
        let postings = match lookup(&key)? {
            Some(data) => decode_posting_list(&data)?,
            None => Vec::new(),
        };

        result_set = Some(match result_set {
            None => postings,
            Some(existing) => {
                // Intersect
                existing
                    .into_iter()
                    .filter(|pk| postings.contains(pk))
                    .collect()
            }
        });
    }

    Ok(result_set.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(!tokens.contains(&"a".to_string())); // Too short
    }

    #[test]
    fn stem_basic() {
        assert_eq!(stem("running"), "runn");
        assert_eq!(stem("tested"), "test");
        assert_eq!(stem("cats"), "cat");
        assert_eq!(stem("boxes"), "box");
    }

    #[test]
    fn posting_list_roundtrip() {
        let pks = vec!["pk1".to_string(), "pk2".to_string()];
        let encoded = encode_posting_list(&pks);
        let decoded = decode_posting_list(&encoded).unwrap();
        assert_eq!(decoded, pks);
    }

    #[test]
    fn merge_posting_deduplicates() {
        let initial = encode_posting_list(&["pk1".to_string()]);
        let merged = merge_posting(&initial, "pk2");
        let list = decode_posting_list(&merged).unwrap();
        assert_eq!(list, vec!["pk1", "pk2"]);

        // Merging same pk again should not duplicate
        let merged2 = merge_posting(&merged, "pk1");
        let list2 = decode_posting_list(&merged2).unwrap();
        assert_eq!(list2, vec!["pk1", "pk2"]);
    }
}
