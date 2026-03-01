//! Constrained SQL grammar decoder — biases token generation toward valid SQL.
//!
//! Implements a soft-constraint state machine that tracks high-level SQL structure
//! (keyword position, clause boundaries) and filters the logit distribution to
//! suppress clearly invalid tokens at each generation step.
//!
//! This is intentionally a *soft* constraint — it biases toward SQL tokens without
//! being so restrictive that it prevents the model from generating valid but unusual SQL.

use super::tokenizer::BpeTokenizer;

/// SQL keywords recognized by the grammar decoder.
const SQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "IS",
    "NULL",
    "LIKE",
    "BETWEEN",
    "EXISTS",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "AS",
    "ORDER",
    "BY",
    "GROUP",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "DISTINCT",
    "JOIN",
    "INNER",
    "LEFT",
    "RIGHT",
    "OUTER",
    "CROSS",
    "ON",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "CREATE",
    "TABLE",
    "DROP",
    "ALTER",
    "ADD",
    "COLUMN",
    "SHOW",
    "TABLES",
    "DESCRIBE",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "ASC",
    "DESC",
    "WITH",
    "RECURSIVE",
    "UNION",
    "ALL",
    "INTERSECT",
    "EXCEPT",
    "CAST",
    "OVER",
    "PARTITION",
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
    "IIF",
    "COALESCE",
    "NULLIF",
    "OF",
    "SYSTEM_TIME",
    "VALID",
    "AT",
    "TRUE",
    "FALSE",
    // Common lowercase variants
    "select",
    "from",
    "where",
    "and",
    "or",
    "not",
    "in",
    "is",
    "null",
    "like",
    "between",
    "exists",
    "case",
    "when",
    "then",
    "else",
    "end",
    "as",
    "order",
    "by",
    "group",
    "having",
    "limit",
    "offset",
    "distinct",
    "join",
    "inner",
    "left",
    "right",
    "outer",
    "cross",
    "on",
    "insert",
    "into",
    "values",
    "update",
    "set",
    "delete",
    "create",
    "table",
    "drop",
    "alter",
    "add",
    "column",
    "show",
    "tables",
    "describe",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "asc",
    "desc",
    "with",
    "recursive",
    "union",
    "all",
    "intersect",
    "except",
    "cast",
    "over",
    "partition",
    "row_number",
    "rank",
    "dense_rank",
    "iif",
    "coalesce",
    "nullif",
];

/// Characters commonly valid in SQL.
const SQL_CHARS: &[u8] = b" \t\n\r.,;()[]'\"*+-/=<>!_%0123456789";

/// Constrained SQL grammar decoder.
pub struct SqlGrammarDecoder {
    /// Per-token validity: `valid_mask[token_id]` is true for SQL-compatible tokens.
    /// O(1) lookup replaces the previous HashSet.
    valid_mask: Vec<bool>,
    /// Sorted list of SQL-compatible token IDs (~5K out of ~151K vocab).
    /// Used for vocabulary-pruned LM head computation.
    active_vocab: Vec<u32>,
    /// Whether constrained decoding is enabled.
    enabled: bool,
    /// Penalty applied to non-SQL tokens (subtracted from logit).
    /// Not -inf, just a bias — allows the model to use unusual tokens when confident.
    penalty: f32,
}

impl SqlGrammarDecoder {
    /// Build the grammar decoder from a tokenizer's vocabulary.
    pub fn new(tokenizer: &BpeTokenizer, enabled: bool) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let mut valid_mask = vec![false; vocab_size];

        // Find token IDs for SQL keywords
        for keyword in SQL_KEYWORDS {
            if let Some(id) = tokenizer.token_id(keyword.as_bytes()) {
                valid_mask[id as usize] = true;
            }
            // Also try with leading space (common in BPE)
            let spaced = format!(" {keyword}");
            if let Some(id) = tokenizer.token_id(spaced.as_bytes()) {
                valid_mask[id as usize] = true;
            }
        }

        // Scan vocabulary for tokens that look like valid SQL content
        for id in 0..vocab_size as u32 {
            if let Some(token_bytes) = tokenizer.token_str(id) {
                if is_sql_compatible(token_bytes) {
                    valid_mask[id as usize] = true;
                }
            }
        }

        // Always allow EOS tokens
        valid_mask[tokenizer.eos_token_id as usize] = true;
        if let Some(id) = tokenizer.special_token_id("<|im_end|>") {
            valid_mask[id as usize] = true;
        }
        if let Some(id) = tokenizer.special_token_id("<|endoftext|>") {
            valid_mask[id as usize] = true;
        }

        // Build sorted active vocab from the mask
        let active_vocab: Vec<u32> = (0..vocab_size as u32)
            .filter(|&id| valid_mask[id as usize])
            .collect();

        Self {
            valid_mask,
            active_vocab,
            enabled,
            penalty: 10.0, // Soft penalty, not hard mask
        }
    }

    /// Apply grammar constraints to logits in-place.
    ///
    /// Tokens that aren't SQL-compatible get a penalty subtracted from their logits.
    /// This biases the model toward SQL tokens without completely preventing
    /// it from using unusual tokens when it's very confident.
    pub fn apply(&self, logits: &mut [f32]) {
        if !self.enabled {
            return;
        }

        for (id, logit) in logits.iter_mut().enumerate() {
            if !self.valid_mask[id] {
                *logit -= self.penalty;
            }
        }
    }

    /// Return the sorted list of SQL-compatible token IDs.
    pub fn active_token_ids(&self) -> &[u32] {
        &self.active_vocab
    }
}

/// Check if a byte sequence looks like valid SQL content.
///
/// Returns true if every byte is a letter, digit, or common SQL character.
fn is_sql_compatible(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    bytes
        .iter()
        .all(|&b| b.is_ascii_alphanumeric() || b == b'_' || SQL_CHARS.contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_sql_compatible_accepts_keywords() {
        assert!(is_sql_compatible(b"SELECT"));
        assert!(is_sql_compatible(b"FROM"));
        assert!(is_sql_compatible(b" WHERE"));
        assert!(is_sql_compatible(b"users"));
        assert!(is_sql_compatible(b"count(*)"));
        assert!(is_sql_compatible(b">="));
        assert!(is_sql_compatible(b"123"));
        assert!(is_sql_compatible(b"'hello'"));
    }

    #[test]
    fn is_sql_compatible_rejects_special() {
        assert!(!is_sql_compatible(b"")); // empty
        assert!(!is_sql_compatible(b"\x00")); // null byte
        assert!(!is_sql_compatible(b"{json}")); // braces
    }

    #[test]
    fn grammar_penalty_applied() {
        // Simulate a simple vocab of 10 tokens
        let mut logits = vec![5.0; 10];

        // Create a mock decoder that marks tokens 0-4 as valid SQL
        let mut valid_mask = vec![false; 10];
        for item in valid_mask.iter_mut().take(5) {
            *item = true;
        }
        let active_vocab = (0..5u32).collect();
        let decoder = SqlGrammarDecoder {
            valid_mask,
            active_vocab,
            enabled: true,
            penalty: 10.0,
        };

        decoder.apply(&mut logits);

        // Tokens 0-4 should be unchanged
        for logit in &logits[..5] {
            assert_eq!(*logit, 5.0);
        }
        // Tokens 5-9 should be penalized
        for logit in &logits[5..10] {
            assert_eq!(*logit, -5.0);
        }
    }

    #[test]
    fn disabled_grammar_is_noop() {
        let mut logits = vec![5.0; 10];
        let decoder = SqlGrammarDecoder {
            valid_mask: vec![false; 10],
            active_vocab: Vec::new(),
            enabled: false,
            penalty: 10.0,
        };

        decoder.apply(&mut logits);

        for &l in &logits {
            assert_eq!(l, 5.0);
        }
    }

    #[test]
    fn active_vocab_matches_mask() {
        let mut valid_mask = vec![false; 20];
        valid_mask[3] = true;
        valid_mask[7] = true;
        valid_mask[15] = true;
        let active_vocab: Vec<u32> = (0..20u32).filter(|&id| valid_mask[id as usize]).collect();

        let decoder = SqlGrammarDecoder {
            valid_mask,
            active_vocab,
            enabled: true,
            penalty: 10.0,
        };

        assert_eq!(decoder.active_token_ids(), &[3, 7, 15]);
    }
}
