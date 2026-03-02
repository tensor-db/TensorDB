//! SQL template speculative decoding — zero-cost drafting using SQL grammar knowledge.
//!
//! Instead of a draft model, we use pre-tokenized SQL templates filled with
//! schema-specific table/column names to propose likely continuations.
//! Verification uses the same model (batched forward pass), so correctness is guaranteed.

use super::tokenizer::BpeTokenizer;

/// A pre-tokenized SQL template for speculation.
#[derive(Debug)]
struct SqlTemplate {
    tokens: Vec<u32>,
    #[allow(dead_code)]
    text: String,
}

/// SQL speculation engine.
pub struct SqlSpeculator {
    templates: Vec<SqlTemplate>,
    max_draft_len: usize,
}

/// Result of a speculation attempt.
pub struct SpeculationDraft {
    pub draft_tokens: Vec<u32>,
    pub match_confidence: f32,
}

/// SQL parse context for adaptive speculation window sizing.
#[derive(Debug, Clone, Copy)]
pub enum SqlContext {
    AfterKeyword,  // FROM, WHERE, JOIN — speculate aggressively (8 tokens)
    AfterOperator, // SELECT, comma — moderate (4 tokens)
    AfterValue,    // literals — conservative (2 tokens)
    Unknown,       // default (3 tokens)
}

impl SqlSpeculator {
    /// Build speculator from schema information.
    pub fn new(
        tokenizer: &BpeTokenizer,
        table_names: &[String],
        column_names: &[Vec<String>],
    ) -> Self {
        let mut templates = Vec::new();

        for (i, table) in table_names.iter().enumerate() {
            let cols = if i < column_names.len() {
                &column_names[i]
            } else {
                continue;
            };
            let cols_str = cols.join(", ");
            let first_col = cols.first().map(|s| s.as_str()).unwrap_or("id");

            let patterns = vec![
                format!("SELECT COUNT(*) FROM {table};"),
                format!("SELECT * FROM {table}"),
                format!("SELECT {cols_str} FROM {table}"),
                format!("SELECT * FROM {table} WHERE "),
                format!("SELECT * FROM {table} ORDER BY {first_col}"),
                format!("SELECT * FROM {table} LIMIT "),
                format!("SELECT COUNT(*) FROM {table} WHERE "),
                format!("SELECT DISTINCT "),
            ];

            for pat in patterns {
                let tokens = tokenizer.encode(&pat);
                templates.push(SqlTemplate { tokens, text: pat });
            }
        }

        // Cross-table JOIN patterns
        if table_names.len() >= 2 {
            for i in 0..table_names.len() {
                for j in 0..table_names.len() {
                    if i != j {
                        let t1 = &table_names[i];
                        let t2 = &table_names[j];
                        let text = format!("SELECT * FROM {t1} JOIN {t2} ON ");
                        let tokens = tokenizer.encode(&text);
                        templates.push(SqlTemplate { tokens, text });
                    }
                }
            }
        }

        // Common SQL fragments
        for frag in &[
            "GROUP BY ",
            "ORDER BY ",
            "HAVING ",
            " ASC",
            " DESC",
            " LIMIT ",
            " AND ",
            " OR ",
            " IN (",
            " NOT IN (",
            " IS NULL",
            " IS NOT NULL",
            "SHOW TABLES;",
            "DESCRIBE ",
            " AS OF EPOCH ",
        ] {
            let tokens = tokenizer.encode(frag);
            templates.push(SqlTemplate {
                tokens,
                text: frag.to_string(),
            });
        }

        Self {
            templates,
            max_draft_len: 8,
        }
    }

    /// Attempt to draft continuation tokens based on current output.
    pub fn draft(
        &self,
        output_tokens: &[u32],
        sql_context: SqlContext,
    ) -> Option<SpeculationDraft> {
        if output_tokens.is_empty() {
            return None;
        }

        let draft_len = match sql_context {
            SqlContext::AfterKeyword => self.max_draft_len,
            SqlContext::AfterOperator => 4,
            SqlContext::AfterValue => 2,
            SqlContext::Unknown => 3,
        };

        let mut best_match: Option<(usize, &SqlTemplate)> = None;
        for template in &self.templates {
            let match_len = Self::suffix_prefix_match(output_tokens, &template.tokens);
            if match_len > 0
                && (best_match.is_none() || match_len > best_match.unwrap().0)
            {
                best_match = Some((match_len, template));
            }
        }

        let (match_len, template) = best_match?;
        let remaining = &template.tokens[match_len..];
        if remaining.is_empty() {
            return None;
        }

        let draft_tokens: Vec<u32> = remaining.iter().copied().take(draft_len).collect();
        let confidence = match_len as f32 / template.tokens.len() as f32;

        Some(SpeculationDraft {
            draft_tokens,
            match_confidence: confidence,
        })
    }

    /// Find the longest suffix of `output` that matches a prefix of `template`.
    fn suffix_prefix_match(output: &[u32], template: &[u32]) -> usize {
        let max_check = output.len().min(template.len());
        let mut best = 0;
        for len in 1..=max_check {
            let output_suffix = &output[output.len() - len..];
            let template_prefix = &template[..len];
            if output_suffix == template_prefix {
                best = len;
            }
        }
        best
    }
}

/// Detect SQL context from the last few tokens.
pub fn detect_sql_context(tokenizer: &BpeTokenizer, recent_tokens: &[u32]) -> SqlContext {
    if recent_tokens.is_empty() {
        return SqlContext::Unknown;
    }

    let last_few = if recent_tokens.len() > 3 {
        &recent_tokens[recent_tokens.len() - 3..]
    } else {
        recent_tokens
    };

    let text = tokenizer.decode(last_few).to_uppercase();

    if text.ends_with("FROM ")
        || text.ends_with("WHERE ")
        || text.ends_with("JOIN ")
        || text.ends_with("ON ")
        || text.ends_with("SET ")
        || text.ends_with("INTO ")
    {
        SqlContext::AfterKeyword
    } else if text.ends_with("SELECT ")
        || text.ends_with(", ")
        || text.ends_with("= ")
        || text.ends_with("> ")
        || text.ends_with("< ")
        || text.ends_with("BY ")
    {
        SqlContext::AfterOperator
    } else if text.ends_with('\'') || text.ends_with(';') {
        SqlContext::AfterValue
    } else {
        SqlContext::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_prefix_match_finds_overlap() {
        // output ends with [1, 2, 3], template starts with [2, 3, 4, 5]
        assert_eq!(
            SqlSpeculator::suffix_prefix_match(&[1, 2, 3], &[2, 3, 4, 5]),
            2
        );
        // No overlap
        assert_eq!(
            SqlSpeculator::suffix_prefix_match(&[1, 2, 3], &[4, 5, 6]),
            0
        );
        // Full match
        assert_eq!(
            SqlSpeculator::suffix_prefix_match(&[1, 2, 3], &[1, 2, 3, 4]),
            3
        );
        // Empty
        assert_eq!(SqlSpeculator::suffix_prefix_match(&[], &[1, 2, 3]), 0);
    }

    #[test]
    fn draft_returns_none_for_empty_output() {
        // We can't easily construct a BpeTokenizer without a model, so test
        // the suffix_prefix_match logic directly and the draft edge cases.
        let speculator = SqlSpeculator {
            templates: vec![SqlTemplate {
                tokens: vec![10, 20, 30, 40],
                text: "test".to_string(),
            }],
            max_draft_len: 8,
        };

        assert!(speculator.draft(&[], SqlContext::Unknown).is_none());
    }

    #[test]
    fn draft_returns_continuation() {
        let speculator = SqlSpeculator {
            templates: vec![SqlTemplate {
                tokens: vec![10, 20, 30, 40, 50],
                text: "test".to_string(),
            }],
            max_draft_len: 8,
        };

        // Output ends with [10, 20] which matches template prefix [10, 20]
        let draft = speculator.draft(&[5, 10, 20], SqlContext::AfterKeyword);
        assert!(draft.is_some());
        let d = draft.unwrap();
        assert_eq!(d.draft_tokens, vec![30, 40, 50]); // continuation after match
    }

    #[test]
    fn draft_respects_context_window() {
        let speculator = SqlSpeculator {
            templates: vec![SqlTemplate {
                tokens: vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
                text: "test".to_string(),
            }],
            max_draft_len: 8,
        };

        // AfterValue context limits to 2 tokens
        let draft = speculator
            .draft(&[5, 10, 20], SqlContext::AfterValue)
            .unwrap();
        assert_eq!(draft.draft_tokens.len(), 2);
        assert_eq!(draft.draft_tokens, vec![30, 40]);
    }
}
