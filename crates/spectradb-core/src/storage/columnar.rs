use crate::error::{Result, SpectraError};
use crate::sql::parser::SqlType;
use crate::util::varint;

/// Typed value for columnar storage.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Boolean(bool),
    Blob(Vec<u8>),
}

impl TypedValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            TypedValue::Null => "NULL",
            TypedValue::Integer(_) => "INTEGER",
            TypedValue::Real(_) => "REAL",
            TypedValue::Text(_) => "TEXT",
            TypedValue::Boolean(_) => "BOOLEAN",
            TypedValue::Blob(_) => "BLOB",
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            TypedValue::Null => serde_json::Value::Null,
            TypedValue::Integer(n) => serde_json::json!(*n),
            TypedValue::Real(f) => serde_json::json!(*f),
            TypedValue::Text(s) => serde_json::json!(s),
            TypedValue::Boolean(b) => serde_json::json!(*b),
            TypedValue::Blob(b) => serde_json::json!(base64_encode(b)),
        }
    }

    pub fn from_json(val: &serde_json::Value, target_type: SqlType) -> Result<TypedValue> {
        if val.is_null() {
            return Ok(TypedValue::Null);
        }
        match target_type {
            SqlType::Integer => match val {
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(TypedValue::Integer(i))
                    } else if let Some(f) = n.as_f64() {
                        Ok(TypedValue::Integer(f as i64))
                    } else {
                        Err(SpectraError::SqlExec("cannot convert to INTEGER".into()))
                    }
                }
                serde_json::Value::String(s) => s
                    .parse::<i64>()
                    .map(TypedValue::Integer)
                    .map_err(|_| SpectraError::SqlExec(format!("cannot parse '{s}' as INTEGER"))),
                _ => Err(SpectraError::SqlExec(format!(
                    "cannot convert {val} to INTEGER"
                ))),
            },
            SqlType::Real => match val {
                serde_json::Value::Number(n) => Ok(TypedValue::Real(n.as_f64().unwrap_or(0.0))),
                serde_json::Value::String(s) => s
                    .parse::<f64>()
                    .map(TypedValue::Real)
                    .map_err(|_| SpectraError::SqlExec(format!("cannot parse '{s}' as REAL"))),
                _ => Err(SpectraError::SqlExec(format!(
                    "cannot convert {val} to REAL"
                ))),
            },
            SqlType::Text => match val {
                serde_json::Value::String(s) => Ok(TypedValue::Text(s.clone())),
                other => Ok(TypedValue::Text(other.to_string())),
            },
            SqlType::Boolean => match val {
                serde_json::Value::Bool(b) => Ok(TypedValue::Boolean(*b)),
                serde_json::Value::Number(n) => Ok(TypedValue::Boolean(n.as_i64() != Some(0))),
                serde_json::Value::String(s) => match s.to_lowercase().as_str() {
                    "true" | "1" | "yes" => Ok(TypedValue::Boolean(true)),
                    "false" | "0" | "no" => Ok(TypedValue::Boolean(false)),
                    _ => Err(SpectraError::SqlExec(format!(
                        "cannot parse '{s}' as BOOLEAN"
                    ))),
                },
                _ => Err(SpectraError::SqlExec(format!(
                    "cannot convert {val} to BOOLEAN"
                ))),
            },
            SqlType::Blob => match val {
                serde_json::Value::String(s) => Ok(TypedValue::Blob(s.as_bytes().to_vec())),
                _ => Err(SpectraError::SqlExec("BLOB requires string value".into())),
            },
            SqlType::Json => Ok(TypedValue::Text(val.to_string())),
        }
    }
}

/// Encode a row of typed values into a compact binary format.
///
/// Format: [null_bitmap] [col0_data] [col1_data] ...
/// - null_bitmap: ceil(n_cols / 8) bytes, bit i set means column i is null
/// - Integer: 8 bytes LE i64
/// - Real: 8 bytes LE f64
/// - Text: varint(len) + utf8_bytes
/// - Boolean: 1 byte (0 or 1)
/// - Blob: varint(len) + raw_bytes
pub fn encode_row(values: &[TypedValue]) -> Vec<u8> {
    let n_cols = values.len();
    let bitmap_bytes = (n_cols + 7) / 8;
    let mut buf = vec![0u8; bitmap_bytes];

    // Set null bits
    for (i, val) in values.iter().enumerate() {
        if matches!(val, TypedValue::Null) {
            buf[i / 8] |= 1 << (i % 8);
        }
    }

    // Encode each non-null column
    for val in values {
        match val {
            TypedValue::Null => {} // Already in bitmap
            TypedValue::Integer(n) => buf.extend_from_slice(&n.to_le_bytes()),
            TypedValue::Real(f) => buf.extend_from_slice(&f.to_le_bytes()),
            TypedValue::Text(s) => {
                let bytes = s.as_bytes();
                varint::encode_u64(bytes.len() as u64, &mut buf);
                buf.extend_from_slice(bytes);
            }
            TypedValue::Boolean(b) => buf.push(if *b { 1 } else { 0 }),
            TypedValue::Blob(b) => {
                varint::encode_u64(b.len() as u64, &mut buf);
                buf.extend_from_slice(b);
            }
        }
    }

    buf
}

/// Decode a row of typed values from binary format.
pub fn decode_row(data: &[u8], types: &[SqlType]) -> Result<Vec<TypedValue>> {
    let n_cols = types.len();
    let bitmap_bytes = (n_cols + 7) / 8;
    if data.len() < bitmap_bytes {
        return Err(SpectraError::SqlExec("row data too short".into()));
    }

    let bitmap = &data[..bitmap_bytes];
    let mut offset = bitmap_bytes;
    let mut values = Vec::with_capacity(n_cols);

    for (i, col_type) in types.iter().enumerate() {
        let is_null = (bitmap[i / 8] >> (i % 8)) & 1 == 1;
        if is_null {
            values.push(TypedValue::Null);
            continue;
        }

        match col_type {
            SqlType::Integer => {
                if offset + 8 > data.len() {
                    return Err(SpectraError::SqlExec("truncated INTEGER".into()));
                }
                let n = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                offset += 8;
                values.push(TypedValue::Integer(n));
            }
            SqlType::Real => {
                if offset + 8 > data.len() {
                    return Err(SpectraError::SqlExec("truncated REAL".into()));
                }
                let f = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                offset += 8;
                values.push(TypedValue::Real(f));
            }
            SqlType::Text | SqlType::Json => {
                let len = varint::decode_u64(data, &mut offset)? as usize;
                if offset + len > data.len() {
                    return Err(SpectraError::SqlExec("truncated TEXT".into()));
                }
                let s = String::from_utf8_lossy(&data[offset..offset + len]).into_owned();
                offset += len;
                values.push(TypedValue::Text(s));
            }
            SqlType::Boolean => {
                if offset >= data.len() {
                    return Err(SpectraError::SqlExec("truncated BOOLEAN".into()));
                }
                values.push(TypedValue::Boolean(data[offset] != 0));
                offset += 1;
            }
            SqlType::Blob => {
                let len = varint::decode_u64(data, &mut offset)? as usize;
                if offset + len > data.len() {
                    return Err(SpectraError::SqlExec("truncated BLOB".into()));
                }
                values.push(TypedValue::Blob(data[offset..offset + len].to_vec()));
                offset += len;
            }
        }
    }

    Ok(values)
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_typed_row() {
        let values = vec![
            TypedValue::Integer(42),
            TypedValue::Real(3.14),
            TypedValue::Text("hello".into()),
            TypedValue::Boolean(true),
            TypedValue::Null,
        ];
        let types = vec![
            SqlType::Integer,
            SqlType::Real,
            SqlType::Text,
            SqlType::Boolean,
            SqlType::Integer,
        ];
        let encoded = encode_row(&values);
        let decoded = decode_row(&encoded, &types).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn roundtrip_all_null() {
        let values = vec![TypedValue::Null, TypedValue::Null];
        let types = vec![SqlType::Integer, SqlType::Text];
        let encoded = encode_row(&values);
        let decoded = decode_row(&encoded, &types).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn roundtrip_blob() {
        let values = vec![TypedValue::Blob(vec![0xDE, 0xAD, 0xBE, 0xEF])];
        let types = vec![SqlType::Blob];
        let encoded = encode_row(&values);
        let decoded = decode_row(&encoded, &types).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn typed_value_to_json() {
        assert_eq!(TypedValue::Integer(42).to_json(), serde_json::json!(42));
        assert_eq!(
            TypedValue::Text("hi".into()).to_json(),
            serde_json::json!("hi")
        );
        assert_eq!(TypedValue::Null.to_json(), serde_json::Value::Null);
    }
}
