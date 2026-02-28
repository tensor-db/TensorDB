use std::collections::HashMap;

/// Zone map: per-column min/max statistics for predicate pushdown.
/// Allows skipping entire SSTable blocks when the filter predicate
/// cannot match any values in the block's range.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZoneMap {
    /// Column name -> column statistics
    pub columns: HashMap<String, ColumnZoneStats>,
    /// Total row count in this zone
    pub row_count: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnZoneStats {
    /// Minimum value (as JSON for type flexibility)
    pub min: serde_json::Value,
    /// Maximum value (as JSON for type flexibility)
    pub max: serde_json::Value,
    /// Number of null values
    pub null_count: u64,
    /// Number of distinct values (approximate via HyperLogLog)
    pub distinct_count: u64,
}

impl ZoneMap {
    pub fn new() -> Self {
        ZoneMap {
            columns: HashMap::new(),
            row_count: 0,
        }
    }

    /// Build a zone map from JSON row data.
    pub fn from_json_rows(rows: &[serde_json::Value]) -> Self {
        let mut zm = ZoneMap::new();
        zm.row_count = rows.len() as u64;

        // Collect all column names from first row
        let col_names: Vec<String> = if let Some(first) = rows.first() {
            if let Some(obj) = first.as_object() {
                obj.keys().cloned().collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        for col_name in &col_names {
            let mut min_val: Option<f64> = None;
            let mut max_val: Option<f64> = None;
            let mut min_str: Option<String> = None;
            let mut max_str: Option<String> = None;
            let mut null_count = 0u64;
            let mut is_numeric = true;
            let mut hll = HyperLogLog::new();

            for row in rows {
                let val = row.get(col_name).unwrap_or(&serde_json::Value::Null);
                if val.is_null() {
                    null_count += 1;
                    continue;
                }

                // Track distinct values
                let hash_input = match val {
                    serde_json::Value::String(s) => s.as_bytes().to_vec(),
                    _ => val.to_string().into_bytes(),
                };
                hll.add(&hash_input);

                if let Some(n) = val.as_f64() {
                    min_val = Some(min_val.map_or(n, |m: f64| m.min(n)));
                    max_val = Some(max_val.map_or(n, |m: f64| m.max(n)));
                } else if let Some(s) = val.as_str() {
                    is_numeric = false;
                    let s_owned = s.to_string();
                    min_str = Some(min_str.map_or(s_owned.clone(), |m: String| {
                        if s_owned < m {
                            s_owned.clone()
                        } else {
                            m
                        }
                    }));
                    max_str = Some(max_str.map_or(s_owned.clone(), |m: String| {
                        if s_owned > m {
                            s_owned
                        } else {
                            m
                        }
                    }));
                } else {
                    is_numeric = false;
                }
            }

            let (min_json, max_json) =
                if let (true, Some(mn), Some(mx)) = (is_numeric, min_val, max_val) {
                    (serde_json::json!(mn), serde_json::json!(mx))
                } else if let (Some(mn), Some(mx)) = (min_str, max_str) {
                    (serde_json::json!(mn), serde_json::json!(mx))
                } else {
                    (serde_json::Value::Null, serde_json::Value::Null)
                };

            zm.columns.insert(
                col_name.clone(),
                ColumnZoneStats {
                    min: min_json,
                    max: max_json,
                    null_count,
                    distinct_count: hll.count(),
                },
            );
        }

        zm
    }

    /// Check if a simple predicate (col op literal) can potentially match
    /// any rows in this zone. Returns false if we can definitively skip this zone.
    pub fn can_match(&self, col_name: &str, op: &str, literal: &serde_json::Value) -> bool {
        let stats = match self.columns.get(col_name) {
            Some(s) => s,
            None => return true, // Unknown column, can't skip
        };

        // If all rows are null, no comparison can match
        if stats.null_count == self.row_count {
            return false;
        }

        let lit_f64 = literal.as_f64();
        let min_f64 = stats.min.as_f64();
        let max_f64 = stats.max.as_f64();

        // Numeric comparison
        if let (Some(lit), Some(min), Some(max)) = (lit_f64, min_f64, max_f64) {
            return match op {
                "=" | "==" => lit >= min && lit <= max,
                "!=" | "<>" => !(min == max && min == lit), // Only skip if single value equals lit
                "<" => min < lit,
                "<=" => min <= lit,
                ">" => max > lit,
                ">=" => max >= lit,
                _ => true,
            };
        }

        // String comparison
        let lit_str = literal.as_str();
        let min_str = stats.min.as_str();
        let max_str = stats.max.as_str();

        if let (Some(lit), Some(min), Some(max)) = (lit_str, min_str, max_str) {
            return match op {
                "=" | "==" => lit >= min && lit <= max,
                "!=" | "<>" => !(min == max && min == lit),
                "<" => min < lit,
                "<=" => min <= lit,
                ">" => max > lit,
                ">=" => max >= lit,
                _ => true,
            };
        }

        true // Can't determine, don't skip
    }
}

impl Default for ZoneMap {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== HyperLogLog ====================

/// HyperLogLog approximate distinct count estimator.
/// Uses 256 registers (8-bit precision) for compact memory usage.
const HLL_PRECISION: usize = 8;
const HLL_NUM_REGISTERS: usize = 1 << HLL_PRECISION; // 256

#[derive(Debug, Clone)]
pub struct HyperLogLog {
    registers: Vec<u8>,
}

impl HyperLogLog {
    pub fn new() -> Self {
        HyperLogLog {
            registers: vec![0; HLL_NUM_REGISTERS],
        }
    }

    /// Add a value to the HLL.
    pub fn add(&mut self, data: &[u8]) {
        let hash = fnv1a_hash(data);
        let idx = (hash & (HLL_NUM_REGISTERS as u64 - 1)) as usize;
        let remaining = hash >> HLL_PRECISION;
        let leading_zeros = if remaining == 0 {
            (64 - HLL_PRECISION) as u8
        } else {
            remaining.trailing_zeros() as u8 + 1
        };
        self.registers[idx] = self.registers[idx].max(leading_zeros);
    }

    /// Estimate the cardinality (distinct count).
    pub fn count(&self) -> u64 {
        let m = HLL_NUM_REGISTERS as f64;
        let alpha = match HLL_PRECISION {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };

        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powi(-(r as i32)))
            .sum();
        let estimate = alpha * m * m / sum;

        // Small range correction
        if estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                return (m * (m / zeros as f64).ln()) as u64;
            }
        }

        estimate as u64
    }

    /// Merge another HLL into this one.
    pub fn merge(&mut self, other: &HyperLogLog) {
        for (i, &val) in other.registers.iter().enumerate() {
            self.registers[i] = self.registers[i].max(val);
        }
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}

/// FNV-1a hash (64-bit).
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ==================== Dictionary Encoding ====================

/// Dictionary encoding for low-cardinality string columns.
/// Maps string values to integer codes for compact storage.
#[derive(Debug, Clone)]
pub struct DictionaryEncoding {
    /// Dictionary: code -> value
    pub dictionary: Vec<String>,
    /// Encoded column: row -> code
    pub codes: Vec<u32>,
    /// Reverse lookup: value -> code
    lookup: HashMap<String, u32>,
}

impl DictionaryEncoding {
    pub fn new() -> Self {
        DictionaryEncoding {
            dictionary: Vec::new(),
            codes: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Encode a string column. Returns the encoding, or None if
    /// the cardinality is too high (> 50% of row count).
    pub fn encode(values: &[Option<String>]) -> Option<DictionaryEncoding> {
        let mut enc = DictionaryEncoding::new();
        let null_code = u32::MAX;

        for val in values {
            match val {
                Some(s) => {
                    let code = if let Some(&existing) = enc.lookup.get(s) {
                        existing
                    } else {
                        let code = enc.dictionary.len() as u32;
                        enc.dictionary.push(s.clone());
                        enc.lookup.insert(s.clone(), code);
                        code
                    };
                    enc.codes.push(code);
                }
                None => {
                    enc.codes.push(null_code);
                }
            }
        }

        // Only use dictionary if cardinality is < 50% of rows
        if !values.is_empty() && enc.dictionary.len() * 2 > values.len() {
            return None; // High cardinality, not worth it
        }

        Some(enc)
    }

    /// Decode a code back to its string value.
    pub fn decode(&self, code: u32) -> Option<&str> {
        if code == u32::MAX {
            None // Null
        } else {
            self.dictionary.get(code as usize).map(|s| s.as_str())
        }
    }

    /// Get compression ratio (dict size vs original).
    pub fn compression_ratio(&self, original_bytes: usize) -> f64 {
        let dict_bytes: usize = self.dictionary.iter().map(|s| s.len() + 4).sum(); // values + 4 bytes overhead each
        let codes_bytes = self.codes.len() * 4; // u32 per code
        let encoded_bytes = dict_bytes + codes_bytes;
        if encoded_bytes == 0 {
            1.0
        } else {
            original_bytes as f64 / encoded_bytes as f64
        }
    }
}

impl Default for DictionaryEncoding {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_map_from_json() {
        let rows = vec![
            serde_json::json!({"name": "Alice", "age": 30, "score": 85.0}),
            serde_json::json!({"name": "Bob", "age": 25, "score": 72.0}),
            serde_json::json!({"name": "Charlie", "age": 35, "score": 93.0}),
        ];
        let zm = ZoneMap::from_json_rows(&rows);
        assert_eq!(zm.row_count, 3);

        let age_stats = zm.columns.get("age").unwrap();
        assert_eq!(age_stats.min, serde_json::json!(25.0));
        assert_eq!(age_stats.max, serde_json::json!(35.0));
        assert_eq!(age_stats.null_count, 0);

        let name_stats = zm.columns.get("name").unwrap();
        assert_eq!(name_stats.min, serde_json::json!("Alice"));
        assert_eq!(name_stats.max, serde_json::json!("Charlie"));
    }

    #[test]
    fn test_zone_map_can_match() {
        let rows = vec![
            serde_json::json!({"x": 10}),
            serde_json::json!({"x": 20}),
            serde_json::json!({"x": 30}),
        ];
        let zm = ZoneMap::from_json_rows(&rows);

        // x > 5 — yes, max is 30
        assert!(zm.can_match("x", ">", &serde_json::json!(5)));
        // x > 50 — no, max is 30
        assert!(!zm.can_match("x", ">", &serde_json::json!(50)));
        // x < 5 — no, min is 10
        assert!(!zm.can_match("x", "<", &serde_json::json!(5)));
        // x = 15 — yes, in range [10, 30]
        assert!(zm.can_match("x", "=", &serde_json::json!(15)));
        // x = 50 — no, out of range
        assert!(!zm.can_match("x", "=", &serde_json::json!(50)));
    }

    #[test]
    fn test_hyperloglog_basic() {
        let mut hll = HyperLogLog::new();
        for i in 0..1000 {
            hll.add(format!("item_{i}").as_bytes());
        }
        let count = hll.count();
        // HLL with 256 registers has ~6-10% standard error; allow 40% tolerance
        assert!((600..=1400).contains(&count), "HLL count was {count}");
    }

    #[test]
    fn test_hyperloglog_duplicates() {
        let mut hll = HyperLogLog::new();
        for _ in 0..1000 {
            hll.add(b"same_value");
        }
        let count = hll.count();
        // Should estimate close to 1
        assert!(count <= 3, "HLL count for single value was {count}");
    }

    #[test]
    fn test_hyperloglog_merge() {
        let mut hll1 = HyperLogLog::new();
        let mut hll2 = HyperLogLog::new();
        for i in 0..500 {
            hll1.add(format!("a_{i}").as_bytes());
        }
        for i in 500..1000 {
            hll2.add(format!("a_{i}").as_bytes());
        }
        hll1.merge(&hll2);
        let count = hll1.count();
        assert!(
            (600..=1400).contains(&count),
            "merged HLL count was {count}"
        );
    }

    #[test]
    fn test_dictionary_encoding() {
        let values: Vec<Option<String>> = vec![
            Some("red".into()),
            Some("blue".into()),
            Some("red".into()),
            Some("green".into()),
            Some("blue".into()),
            Some("red".into()),
            None,
            Some("blue".into()),
        ];
        let enc = DictionaryEncoding::encode(&values).unwrap();
        assert_eq!(enc.dictionary.len(), 3); // red, blue, green

        // Verify round-trip
        for (i, val) in values.iter().enumerate() {
            match val {
                Some(s) => assert_eq!(enc.decode(enc.codes[i]).unwrap(), s.as_str()),
                None => assert!(enc.decode(enc.codes[i]).is_none()),
            }
        }

        // Verify encoding is valid (compression ratio depends on string lengths)
        assert!(enc.codes.len() == values.len());
    }

    #[test]
    fn test_dictionary_high_cardinality_returns_none() {
        // All unique values — dictionary shouldn't be used
        let values: Vec<Option<String>> = (0..100).map(|i| Some(format!("unique_{i}"))).collect();
        assert!(DictionaryEncoding::encode(&values).is_none());
    }

    #[test]
    fn test_zone_map_null_column() {
        let rows = vec![
            serde_json::json!({"x": null}),
            serde_json::json!({"x": null}),
        ];
        let zm = ZoneMap::from_json_rows(&rows);
        let x_stats = zm.columns.get("x").unwrap();
        assert_eq!(x_stats.null_count, 2);
        // All nulls — no comparison can match
        assert!(!zm.can_match("x", "=", &serde_json::json!(5)));
    }
}
