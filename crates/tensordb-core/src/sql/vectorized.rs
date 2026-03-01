use std::collections::HashMap;

use crate::error::{Result, TensorError};

/// Batch size for vectorized processing.
pub const DEFAULT_BATCH_SIZE: usize = 1024;

/// Schema for a record batch: column names and types.
#[derive(Debug, Clone)]
pub struct BatchSchema {
    pub columns: Vec<ColumnDef>,
}

#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub dtype: ColumnType,
}

/// Column data types for vectorized processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Int64,
    Float64,
    Boolean,
    Utf8,
    /// Fixed-dimension vector of f32 values.
    Vector,
}

/// A column vector holding typed data with null tracking.
#[derive(Debug, Clone)]
pub enum ColumnVector {
    Int64 {
        values: Vec<i64>,
        nulls: Vec<bool>, // true = null
    },
    Float64 {
        values: Vec<f64>,
        nulls: Vec<bool>,
    },
    Boolean {
        values: Vec<bool>,
        nulls: Vec<bool>,
    },
    Utf8 {
        values: Vec<String>,
        nulls: Vec<bool>,
    },
    /// Vector column: each value is a Vec<f32>.
    Vector {
        values: Vec<Vec<f32>>,
        nulls: Vec<bool>,
    },
}

impl ColumnVector {
    pub fn len(&self) -> usize {
        match self {
            ColumnVector::Int64 { values, .. } => values.len(),
            ColumnVector::Float64 { values, .. } => values.len(),
            ColumnVector::Boolean { values, .. } => values.len(),
            ColumnVector::Utf8 { values, .. } => values.len(),
            ColumnVector::Vector { values, .. } => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn nulls(&self) -> &[bool] {
        match self {
            ColumnVector::Int64 { nulls, .. } => nulls,
            ColumnVector::Float64 { nulls, .. } => nulls,
            ColumnVector::Boolean { nulls, .. } => nulls,
            ColumnVector::Utf8 { nulls, .. } => nulls,
            ColumnVector::Vector { nulls, .. } => nulls,
        }
    }

    pub fn is_null(&self, idx: usize) -> bool {
        self.nulls()[idx]
    }

    pub fn column_type(&self) -> ColumnType {
        match self {
            ColumnVector::Int64 { .. } => ColumnType::Int64,
            ColumnVector::Float64 { .. } => ColumnType::Float64,
            ColumnVector::Boolean { .. } => ColumnType::Boolean,
            ColumnVector::Utf8 { .. } => ColumnType::Utf8,
            ColumnVector::Vector { .. } => ColumnType::Vector,
        }
    }

    /// Get value as f64 (for numeric operations). Returns None if null.
    pub fn get_f64(&self, idx: usize) -> Option<f64> {
        if self.is_null(idx) {
            return None;
        }
        match self {
            ColumnVector::Int64 { values, .. } => Some(values[idx] as f64),
            ColumnVector::Float64 { values, .. } => Some(values[idx]),
            ColumnVector::Boolean { values, .. } => Some(if values[idx] { 1.0 } else { 0.0 }),
            ColumnVector::Utf8 { values, .. } => values[idx].parse::<f64>().ok(),
            ColumnVector::Vector { .. } => None,
        }
    }

    /// Get value as string (for display/comparison).
    pub fn get_string(&self, idx: usize) -> Option<String> {
        if self.is_null(idx) {
            return None;
        }
        match self {
            ColumnVector::Int64 { values, .. } => Some(values[idx].to_string()),
            ColumnVector::Float64 { values, .. } => Some(values[idx].to_string()),
            ColumnVector::Boolean { values, .. } => Some(values[idx].to_string()),
            ColumnVector::Utf8 { values, .. } => Some(values[idx].clone()),
            ColumnVector::Vector { values, .. } => Some(
                crate::facet::vector_persistence::format_vector(&values[idx]),
            ),
        }
    }

    /// Get value as JSON.
    pub fn get_json(&self, idx: usize) -> serde_json::Value {
        if self.is_null(idx) {
            return serde_json::Value::Null;
        }
        match self {
            ColumnVector::Int64 { values, .. } => serde_json::json!(values[idx]),
            ColumnVector::Float64 { values, .. } => serde_json::Number::from_f64(values[idx])
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            ColumnVector::Boolean { values, .. } => serde_json::json!(values[idx]),
            ColumnVector::Utf8 { values, .. } => serde_json::json!(values[idx]),
            ColumnVector::Vector { values, .. } => {
                serde_json::json!(crate::facet::vector_persistence::format_vector(
                    &values[idx]
                ))
            }
        }
    }

    /// Create an empty column of the given type.
    pub fn empty(dtype: ColumnType) -> Self {
        match dtype {
            ColumnType::Int64 => ColumnVector::Int64 {
                values: Vec::new(),
                nulls: Vec::new(),
            },
            ColumnType::Float64 => ColumnVector::Float64 {
                values: Vec::new(),
                nulls: Vec::new(),
            },
            ColumnType::Boolean => ColumnVector::Boolean {
                values: Vec::new(),
                nulls: Vec::new(),
            },
            ColumnType::Utf8 => ColumnVector::Utf8 {
                values: Vec::new(),
                nulls: Vec::new(),
            },
            ColumnType::Vector => ColumnVector::Vector {
                values: Vec::new(),
                nulls: Vec::new(),
            },
        }
    }

    /// Push a JSON value onto this column.
    pub fn push_json(&mut self, val: &serde_json::Value) {
        match self {
            ColumnVector::Int64 { values, nulls } => {
                if val.is_null() {
                    values.push(0);
                    nulls.push(true);
                } else if let Some(n) = val.as_i64() {
                    values.push(n);
                    nulls.push(false);
                } else if let Some(f) = val.as_f64() {
                    values.push(f as i64);
                    nulls.push(false);
                } else {
                    values.push(0);
                    nulls.push(true);
                }
            }
            ColumnVector::Float64 { values, nulls } => {
                if let Some(f) = val.as_f64() {
                    values.push(f);
                    nulls.push(false);
                } else {
                    values.push(0.0);
                    nulls.push(true);
                }
            }
            ColumnVector::Boolean { values, nulls } => {
                if let Some(b) = val.as_bool() {
                    values.push(b);
                    nulls.push(false);
                } else {
                    values.push(false);
                    nulls.push(true);
                }
            }
            ColumnVector::Utf8 { values, nulls } => {
                if val.is_null() {
                    values.push(String::new());
                    nulls.push(true);
                } else if let Some(s) = val.as_str() {
                    values.push(s.to_string());
                    nulls.push(false);
                } else {
                    values.push(val.to_string());
                    nulls.push(false);
                }
            }
            ColumnVector::Vector { values, nulls } => {
                if val.is_null() {
                    values.push(Vec::new());
                    nulls.push(true);
                } else if let Some(s) = val.as_str() {
                    match crate::facet::vector_persistence::parse_vector_literal(s) {
                        Ok(v) => {
                            values.push(v);
                            nulls.push(false);
                        }
                        Err(_) => {
                            values.push(Vec::new());
                            nulls.push(true);
                        }
                    }
                } else if let Some(arr) = val.as_array() {
                    let v: Vec<f32> = arr
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    values.push(v);
                    nulls.push(false);
                } else {
                    values.push(Vec::new());
                    nulls.push(true);
                }
            }
        }
    }

    /// Select rows by indices (gather operation).
    pub fn gather(&self, indices: &[usize]) -> Self {
        match self {
            ColumnVector::Int64 { values, nulls } => ColumnVector::Int64 {
                values: indices.iter().map(|&i| values[i]).collect(),
                nulls: indices.iter().map(|&i| nulls[i]).collect(),
            },
            ColumnVector::Float64 { values, nulls } => ColumnVector::Float64 {
                values: indices.iter().map(|&i| values[i]).collect(),
                nulls: indices.iter().map(|&i| nulls[i]).collect(),
            },
            ColumnVector::Boolean { values, nulls } => ColumnVector::Boolean {
                values: indices.iter().map(|&i| values[i]).collect(),
                nulls: indices.iter().map(|&i| nulls[i]).collect(),
            },
            ColumnVector::Utf8 { values, nulls } => ColumnVector::Utf8 {
                values: indices.iter().map(|&i| values[i].clone()).collect(),
                nulls: indices.iter().map(|&i| nulls[i]).collect(),
            },
            ColumnVector::Vector { values, nulls } => ColumnVector::Vector {
                values: indices.iter().map(|&i| values[i].clone()).collect(),
                nulls: indices.iter().map(|&i| nulls[i]).collect(),
            },
        }
    }
}

/// A batch of columnar data with schema.
#[derive(Debug, Clone)]
pub struct RecordBatch {
    pub schema: BatchSchema,
    pub columns: Vec<ColumnVector>,
    num_rows: usize,
}

impl RecordBatch {
    pub fn new(schema: BatchSchema, columns: Vec<ColumnVector>) -> Result<Self> {
        let num_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        for (i, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(TensorError::SqlExec(format!(
                    "column {} has {} rows, expected {}",
                    i,
                    col.len(),
                    num_rows
                )));
            }
        }
        Ok(RecordBatch {
            schema,
            columns,
            num_rows,
        })
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, idx: usize) -> &ColumnVector {
        &self.columns[idx]
    }

    pub fn column_by_name(&self, name: &str) -> Option<(usize, &ColumnVector)> {
        self.schema
            .columns
            .iter()
            .position(|c| c.name == name)
            .map(|i| (i, &self.columns[i]))
    }

    /// Convert JSON rows to a RecordBatch using the given schema.
    pub fn from_json_rows(rows: &[serde_json::Value], schema: &BatchSchema) -> Result<RecordBatch> {
        let mut columns: Vec<ColumnVector> = schema
            .columns
            .iter()
            .map(|c| ColumnVector::empty(c.dtype))
            .collect();

        for row in rows {
            for (i, col_def) in schema.columns.iter().enumerate() {
                let val = row.get(&col_def.name).unwrap_or(&serde_json::Value::Null);
                columns[i].push_json(val);
            }
        }

        RecordBatch::new(schema.clone(), columns)
    }

    /// Convert batch back to JSON rows.
    pub fn to_json_rows(&self) -> Vec<serde_json::Value> {
        let mut rows = Vec::with_capacity(self.num_rows);
        for i in 0..self.num_rows {
            let mut obj = serde_json::Map::new();
            for (col_idx, col_def) in self.schema.columns.iter().enumerate() {
                obj.insert(col_def.name.clone(), self.columns[col_idx].get_json(i));
            }
            rows.push(serde_json::Value::Object(obj));
        }
        rows
    }

    /// Convert batch to serialized JSON row bytes (for SqlResult).
    pub fn to_row_bytes(&self) -> Vec<Vec<u8>> {
        let mut result = Vec::with_capacity(self.num_rows);
        for i in 0..self.num_rows {
            let mut obj = serde_json::Map::new();
            for (col_idx, col_def) in self.schema.columns.iter().enumerate() {
                obj.insert(col_def.name.clone(), self.columns[col_idx].get_json(i));
            }
            result.push(serde_json::to_vec(&serde_json::Value::Object(obj)).unwrap_or_default());
        }
        result
    }
}

// ==================== Vectorized Operators ====================

/// Vectorized filter: apply a boolean selection vector to a batch.
pub fn vectorized_filter(batch: &RecordBatch, selection: &[bool]) -> Result<RecordBatch> {
    if selection.len() != batch.num_rows() {
        return Err(TensorError::SqlExec(
            "selection vector length mismatch".to_string(),
        ));
    }

    let indices: Vec<usize> = selection
        .iter()
        .enumerate()
        .filter(|(_, &selected)| selected)
        .map(|(i, _)| i)
        .collect();

    let columns = batch.columns.iter().map(|c| c.gather(&indices)).collect();
    RecordBatch::new(batch.schema.clone(), columns)
}

/// Evaluate a simple comparison on a column: col op literal.
/// Returns a boolean selection vector.
pub fn eval_column_comparison(
    col: &ColumnVector,
    op: CompareOp,
    literal: &serde_json::Value,
) -> Vec<bool> {
    let n = col.len();
    let mut result = vec![false; n];

    match col {
        ColumnVector::Int64 { values, nulls } => {
            if let Some(rhs) = literal.as_i64() {
                for i in 0..n {
                    if !nulls[i] {
                        result[i] = compare_i64(values[i], op, rhs);
                    }
                }
            } else if let Some(rhs) = literal.as_f64() {
                for i in 0..n {
                    if !nulls[i] {
                        result[i] = compare_f64(values[i] as f64, op, rhs);
                    }
                }
            }
        }
        ColumnVector::Float64 { values, nulls } => {
            if let Some(rhs) = literal.as_f64() {
                for i in 0..n {
                    if !nulls[i] {
                        result[i] = compare_f64(values[i], op, rhs);
                    }
                }
            }
        }
        ColumnVector::Utf8 { values, nulls } => {
            if let Some(rhs) = literal.as_str() {
                for i in 0..n {
                    if !nulls[i] {
                        result[i] = compare_str(&values[i], op, rhs);
                    }
                }
            }
        }
        ColumnVector::Boolean { values, nulls } => {
            if let Some(rhs) = literal.as_bool() {
                for i in 0..n {
                    if !nulls[i] {
                        result[i] = match op {
                            CompareOp::Eq => values[i] == rhs,
                            CompareOp::Neq => values[i] != rhs,
                            _ => false,
                        };
                    }
                }
            }
        }
        ColumnVector::Vector { .. } => {
            // Vector comparison not meaningful for scalar ops
        }
    }

    result
}

/// Vectorized sort: sort a batch by one column.
pub fn vectorized_sort(
    batch: &RecordBatch,
    sort_col: usize,
    ascending: bool,
) -> Result<RecordBatch> {
    let col = &batch.columns[sort_col];
    let n = batch.num_rows();
    let mut indices: Vec<usize> = (0..n).collect();

    match col {
        ColumnVector::Int64 { values, nulls } => {
            indices.sort_by(|&a, &b| {
                let null_a = nulls[a];
                let null_b = nulls[b];
                match (null_a, null_b) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => {
                        if ascending {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (false, true) => {
                        if ascending {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (false, false) => {
                        let ord = values[a].cmp(&values[b]);
                        if ascending {
                            ord
                        } else {
                            ord.reverse()
                        }
                    }
                }
            });
        }
        ColumnVector::Float64 { values, nulls } => {
            indices.sort_by(|&a, &b| {
                let null_a = nulls[a];
                let null_b = nulls[b];
                match (null_a, null_b) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => {
                        if ascending {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (false, true) => {
                        if ascending {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (false, false) => {
                        let ord = values[a]
                            .partial_cmp(&values[b])
                            .unwrap_or(std::cmp::Ordering::Equal);
                        if ascending {
                            ord
                        } else {
                            ord.reverse()
                        }
                    }
                }
            });
        }
        ColumnVector::Utf8 { values, nulls } => {
            indices.sort_by(|&a, &b| {
                let null_a = nulls[a];
                let null_b = nulls[b];
                match (null_a, null_b) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => {
                        if ascending {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (false, true) => {
                        if ascending {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (false, false) => {
                        let ord = values[a].cmp(&values[b]);
                        if ascending {
                            ord
                        } else {
                            ord.reverse()
                        }
                    }
                }
            });
        }
        ColumnVector::Boolean { values, nulls } => {
            indices.sort_by(|&a, &b| {
                let null_a = nulls[a];
                let null_b = nulls[b];
                match (null_a, null_b) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => {
                        let ord = values[a].cmp(&values[b]);
                        if ascending {
                            ord
                        } else {
                            ord.reverse()
                        }
                    }
                }
            });
        }
        ColumnVector::Vector { .. } => {
            // Vectors cannot be meaningfully sorted; leave in original order
        }
    }

    let columns = batch.columns.iter().map(|c| c.gather(&indices)).collect();
    RecordBatch::new(batch.schema.clone(), columns)
}

/// Vectorized LIMIT: take first n rows.
pub fn vectorized_limit(batch: &RecordBatch, limit: usize) -> Result<RecordBatch> {
    if limit >= batch.num_rows() {
        return Ok(batch.clone());
    }
    let indices: Vec<usize> = (0..limit).collect();
    let columns = batch.columns.iter().map(|c| c.gather(&indices)).collect();
    RecordBatch::new(batch.schema.clone(), columns)
}

/// Vectorized projection: select specific columns.
pub fn vectorized_project(batch: &RecordBatch, column_indices: &[usize]) -> Result<RecordBatch> {
    let schema = BatchSchema {
        columns: column_indices
            .iter()
            .map(|&i| batch.schema.columns[i].clone())
            .collect(),
    };
    let columns = column_indices
        .iter()
        .map(|&i| batch.columns[i].clone())
        .collect();
    RecordBatch::new(schema, columns)
}

/// Vectorized hash aggregate: GROUP BY + aggregation.
/// Supports SUM, COUNT, AVG, MIN, MAX on numeric columns.
pub fn vectorized_hash_aggregate(
    batch: &RecordBatch,
    group_col_indices: &[usize],
    agg_specs: &[AggSpec],
) -> Result<RecordBatch> {
    // Build group keys
    let n = batch.num_rows();
    let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

    for i in 0..n {
        let key: Vec<String> = group_col_indices
            .iter()
            .map(|&col_idx| {
                batch.columns[col_idx]
                    .get_string(i)
                    .unwrap_or_else(|| "NULL".to_string())
            })
            .collect();
        groups.entry(key).or_default().push(i);
    }

    // Build output schema
    let mut out_cols: Vec<ColumnDef> = group_col_indices
        .iter()
        .map(|&i| batch.schema.columns[i].clone())
        .collect();
    for spec in agg_specs {
        out_cols.push(ColumnDef {
            name: spec.output_name.clone(),
            dtype: ColumnType::Float64, // aggregates produce floats
        });
    }
    let out_schema = BatchSchema { columns: out_cols };

    // Build output columns
    let num_groups = groups.len();
    let mut group_key_cols: Vec<ColumnVector> = group_col_indices
        .iter()
        .map(|&i| ColumnVector::empty(batch.schema.columns[i].dtype))
        .collect();
    let mut agg_cols: Vec<Vec<f64>> = vec![Vec::with_capacity(num_groups); agg_specs.len()];

    // Sort groups for deterministic output
    let mut sorted_groups: Vec<(Vec<String>, Vec<usize>)> = groups.into_iter().collect();
    sorted_groups.sort_by(|a, b| a.0.cmp(&b.0));

    for (_key, row_indices) in &sorted_groups {
        // Push group key values
        for (ki, &col_idx) in group_col_indices.iter().enumerate() {
            if let Some(first_row) = row_indices.first() {
                let val = batch.columns[col_idx].get_json(*first_row);
                group_key_cols[ki].push_json(&val);
            }
        }

        // Compute aggregates
        for (ai, spec) in agg_specs.iter().enumerate() {
            let col = &batch.columns[spec.input_col];
            let result = compute_aggregate(col, row_indices, spec.agg_fn);
            agg_cols[ai].push(result);
        }
    }

    let mut all_columns = group_key_cols;
    for agg_col_data in agg_cols {
        all_columns.push(ColumnVector::Float64 {
            nulls: vec![false; agg_col_data.len()],
            values: agg_col_data,
        });
    }

    RecordBatch::new(out_schema, all_columns)
}

/// Vectorized hash join (inner join).
pub fn vectorized_hash_join(
    left: &RecordBatch,
    right: &RecordBatch,
    left_key_col: usize,
    right_key_col: usize,
) -> Result<RecordBatch> {
    // Build hash table on right side
    let mut hash_table: HashMap<String, Vec<usize>> = HashMap::new();
    for i in 0..right.num_rows() {
        if let Some(key) = right.columns[right_key_col].get_string(i) {
            hash_table.entry(key).or_default().push(i);
        }
    }

    // Probe with left side
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for i in 0..left.num_rows() {
        if let Some(key) = left.columns[left_key_col].get_string(i) {
            if let Some(matches) = hash_table.get(&key) {
                for &j in matches {
                    left_indices.push(i);
                    right_indices.push(j);
                }
            }
        }
    }

    // Build output columns
    let mut out_col_defs = Vec::new();
    let mut out_columns = Vec::new();

    for (i, col_def) in left.schema.columns.iter().enumerate() {
        out_col_defs.push(col_def.clone());
        out_columns.push(left.columns[i].gather(&left_indices));
    }

    for (i, col_def) in right.schema.columns.iter().enumerate() {
        if i == right_key_col {
            continue; // Skip duplicate join key
        }
        let mut renamed = col_def.clone();
        // Avoid name collision
        if out_col_defs.iter().any(|c| c.name == renamed.name) {
            renamed.name = format!("{}__right", renamed.name);
        }
        out_col_defs.push(renamed);
        out_columns.push(right.columns[i].gather(&right_indices));
    }

    let out_schema = BatchSchema {
        columns: out_col_defs,
    };
    RecordBatch::new(out_schema, out_columns)
}

// ==================== Aggregate Support ====================

#[derive(Debug, Clone)]
pub struct AggSpec {
    pub input_col: usize,
    pub agg_fn: AggFn,
    pub output_name: String,
}

#[derive(Debug, Clone, Copy)]
pub enum AggFn {
    Sum,
    Count,
    Avg,
    Min,
    Max,
}

fn compute_aggregate(col: &ColumnVector, indices: &[usize], agg_fn: AggFn) -> f64 {
    match agg_fn {
        AggFn::Count => indices.iter().filter(|&&i| !col.is_null(i)).count() as f64,
        AggFn::Sum => indices.iter().filter_map(|&i| col.get_f64(i)).sum(),
        AggFn::Avg => {
            let mut sum = 0.0;
            let mut count = 0;
            for &i in indices {
                if let Some(v) = col.get_f64(i) {
                    sum += v;
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        }
        AggFn::Min => indices
            .iter()
            .filter_map(|&i| col.get_f64(i))
            .fold(f64::INFINITY, f64::min),
        AggFn::Max => indices
            .iter()
            .filter_map(|&i| col.get_f64(i))
            .fold(f64::NEG_INFINITY, f64::max),
    }
}

// ==================== Comparison Helpers ====================

#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
}

fn compare_i64(lhs: i64, op: CompareOp, rhs: i64) -> bool {
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::Neq => lhs != rhs,
        CompareOp::Lt => lhs < rhs,
        CompareOp::Lte => lhs <= rhs,
        CompareOp::Gt => lhs > rhs,
        CompareOp::Gte => lhs >= rhs,
    }
}

fn compare_f64(lhs: f64, op: CompareOp, rhs: f64) -> bool {
    match op {
        CompareOp::Eq => (lhs - rhs).abs() < f64::EPSILON,
        CompareOp::Neq => (lhs - rhs).abs() >= f64::EPSILON,
        CompareOp::Lt => lhs < rhs,
        CompareOp::Lte => lhs <= rhs,
        CompareOp::Gt => lhs > rhs,
        CompareOp::Gte => lhs >= rhs,
    }
}

fn compare_str(lhs: &str, op: CompareOp, rhs: &str) -> bool {
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::Neq => lhs != rhs,
        CompareOp::Lt => lhs < rhs,
        CompareOp::Lte => lhs <= rhs,
        CompareOp::Gt => lhs > rhs,
        CompareOp::Gte => lhs >= rhs,
    }
}

// ==================== Boolean Selection Vector Ops ====================

/// AND two selection vectors.
pub fn selection_and(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect()
}

/// OR two selection vectors.
pub fn selection_or(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect()
}

/// NOT a selection vector.
pub fn selection_not(a: &[bool]) -> Vec<bool> {
    a.iter().map(|&x| !x).collect()
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_batch() -> RecordBatch {
        let schema = BatchSchema {
            columns: vec![
                ColumnDef {
                    name: "name".to_string(),
                    dtype: ColumnType::Utf8,
                },
                ColumnDef {
                    name: "age".to_string(),
                    dtype: ColumnType::Int64,
                },
                ColumnDef {
                    name: "score".to_string(),
                    dtype: ColumnType::Float64,
                },
            ],
        };
        let columns = vec![
            ColumnVector::Utf8 {
                values: vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                    "Diana".to_string(),
                ],
                nulls: vec![false, false, false, false],
            },
            ColumnVector::Int64 {
                values: vec![30, 25, 35, 28],
                nulls: vec![false, false, false, false],
            },
            ColumnVector::Float64 {
                values: vec![85.0, 72.0, 93.0, 68.0],
                nulls: vec![false, false, false, false],
            },
        ];
        RecordBatch::new(schema, columns).unwrap()
    }

    #[test]
    fn test_record_batch_creation() {
        let batch = sample_batch();
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 3);
    }

    #[test]
    fn test_vectorized_filter() {
        let batch = sample_batch();
        let selection = eval_column_comparison(
            batch.column(2), // score
            CompareOp::Gt,
            &serde_json::json!(80.0),
        );
        let filtered = vectorized_filter(&batch, &selection).unwrap();
        assert_eq!(filtered.num_rows(), 2); // Alice (85) and Charlie (93)
    }

    #[test]
    fn test_vectorized_sort_ascending() {
        let batch = sample_batch();
        let sorted = vectorized_sort(&batch, 2, true).unwrap(); // sort by score asc
        let scores = &sorted.columns[2];
        let vals: Vec<f64> = (0..sorted.num_rows())
            .map(|i| scores.get_f64(i).unwrap())
            .collect();
        assert_eq!(vals, vec![68.0, 72.0, 85.0, 93.0]);
    }

    #[test]
    fn test_vectorized_sort_descending() {
        let batch = sample_batch();
        let sorted = vectorized_sort(&batch, 1, false).unwrap(); // sort by age desc
        let ages: Vec<i64> = (0..sorted.num_rows())
            .filter_map(|i| sorted.columns[1].get_f64(i).map(|v| v as i64))
            .collect();
        assert_eq!(ages, vec![35, 30, 28, 25]);
    }

    #[test]
    fn test_vectorized_limit() {
        let batch = sample_batch();
        let limited = vectorized_limit(&batch, 2).unwrap();
        assert_eq!(limited.num_rows(), 2);
    }

    #[test]
    fn test_vectorized_project() {
        let batch = sample_batch();
        let projected = vectorized_project(&batch, &[0, 2]).unwrap(); // name, score
        assert_eq!(projected.num_columns(), 2);
        assert_eq!(projected.schema.columns[0].name, "name");
        assert_eq!(projected.schema.columns[1].name, "score");
    }

    #[test]
    fn test_vectorized_hash_aggregate() {
        let schema = BatchSchema {
            columns: vec![
                ColumnDef {
                    name: "region".to_string(),
                    dtype: ColumnType::Utf8,
                },
                ColumnDef {
                    name: "amount".to_string(),
                    dtype: ColumnType::Float64,
                },
            ],
        };
        let columns = vec![
            ColumnVector::Utf8 {
                values: vec![
                    "East".to_string(),
                    "West".to_string(),
                    "East".to_string(),
                    "West".to_string(),
                ],
                nulls: vec![false, false, false, false],
            },
            ColumnVector::Float64 {
                values: vec![100.0, 200.0, 150.0, 300.0],
                nulls: vec![false, false, false, false],
            },
        ];
        let batch = RecordBatch::new(schema, columns).unwrap();

        let result = vectorized_hash_aggregate(
            &batch,
            &[0], // group by region
            &[AggSpec {
                input_col: 1,
                agg_fn: AggFn::Sum,
                output_name: "total".to_string(),
            }],
        )
        .unwrap();

        assert_eq!(result.num_rows(), 2);
        let json_rows = result.to_json_rows();
        // Sorted by group key
        assert_eq!(json_rows[0]["region"], "East");
        assert_eq!(json_rows[0]["total"], 250.0);
        assert_eq!(json_rows[1]["region"], "West");
        assert_eq!(json_rows[1]["total"], 500.0);
    }

    #[test]
    fn test_vectorized_hash_join() {
        let left_schema = BatchSchema {
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    dtype: ColumnType::Int64,
                },
                ColumnDef {
                    name: "name".to_string(),
                    dtype: ColumnType::Utf8,
                },
            ],
        };
        let left = RecordBatch::new(
            left_schema,
            vec![
                ColumnVector::Int64 {
                    values: vec![1, 2, 3],
                    nulls: vec![false, false, false],
                },
                ColumnVector::Utf8 {
                    values: vec![
                        "Alice".to_string(),
                        "Bob".to_string(),
                        "Charlie".to_string(),
                    ],
                    nulls: vec![false, false, false],
                },
            ],
        )
        .unwrap();

        let right_schema = BatchSchema {
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    dtype: ColumnType::Int64,
                },
                ColumnDef {
                    name: "score".to_string(),
                    dtype: ColumnType::Float64,
                },
            ],
        };
        let right = RecordBatch::new(
            right_schema,
            vec![
                ColumnVector::Int64 {
                    values: vec![1, 3],
                    nulls: vec![false, false],
                },
                ColumnVector::Float64 {
                    values: vec![85.0, 93.0],
                    nulls: vec![false, false],
                },
            ],
        )
        .unwrap();

        let joined = vectorized_hash_join(&left, &right, 0, 0).unwrap();
        assert_eq!(joined.num_rows(), 2); // Alice+85, Charlie+93
        assert_eq!(joined.num_columns(), 3); // id, name, score (right.id skipped)
    }

    #[test]
    fn test_from_json_rows() {
        let schema = BatchSchema {
            columns: vec![
                ColumnDef {
                    name: "x".to_string(),
                    dtype: ColumnType::Int64,
                },
                ColumnDef {
                    name: "y".to_string(),
                    dtype: ColumnType::Utf8,
                },
            ],
        };
        let rows = vec![
            serde_json::json!({"x": 1, "y": "a"}),
            serde_json::json!({"x": 2, "y": "b"}),
        ];
        let batch = RecordBatch::from_json_rows(&rows, &schema).unwrap();
        assert_eq!(batch.num_rows(), 2);
        let back = batch.to_json_rows();
        assert_eq!(back[0]["x"], 1);
        assert_eq!(back[1]["y"], "b");
    }

    #[test]
    fn test_selection_ops() {
        let a = vec![true, false, true, false];
        let b = vec![true, true, false, false];
        assert_eq!(selection_and(&a, &b), vec![true, false, false, false]);
        assert_eq!(selection_or(&a, &b), vec![true, true, true, false]);
        assert_eq!(selection_not(&a), vec![false, true, false, true]);
    }

    #[test]
    fn test_null_handling() {
        let col = ColumnVector::Int64 {
            values: vec![10, 0, 30],
            nulls: vec![false, true, false],
        };
        assert_eq!(col.get_f64(0), Some(10.0));
        assert_eq!(col.get_f64(1), None); // null
        assert_eq!(col.get_f64(2), Some(30.0));

        let selection = eval_column_comparison(&col, CompareOp::Gt, &serde_json::json!(5));
        // null comparisons always false
        assert_eq!(selection, vec![true, false, true]);
    }

    #[test]
    fn test_to_row_bytes() {
        let batch = sample_batch();
        let bytes = batch.to_row_bytes();
        assert_eq!(bytes.len(), 4);
        let row0: serde_json::Value = serde_json::from_slice(&bytes[0]).unwrap();
        assert_eq!(row0["name"], "Alice");
        assert_eq!(row0["age"], 30);
    }
}
