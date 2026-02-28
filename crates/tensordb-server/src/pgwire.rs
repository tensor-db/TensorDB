//! PostgreSQL v3 wire protocol implementation.
//!
//! Handles the binary protocol used by all Postgres clients (psql, JDBC, libpq, etc).
//! Reference: <https://www.postgresql.org/docs/current/protocol-message-formats.html>

#![allow(dead_code)]

use bytes::{BufMut, BytesMut};
use std::collections::HashMap;

// ---------- Frontend (client → server) message types ----------

/// Parsed frontend message from a Postgres client.
#[derive(Debug)]
pub enum FrontendMessage {
    /// Startup message (no type byte — identified by protocol version).
    Startup { params: HashMap<String, String> },
    /// SSL negotiation request (8-byte special startup packet).
    SslRequest,
    /// Simple query: 'Q' + query string.
    Query(String),
    /// Parse (extended query): 'P' + name + query + param types.
    Parse {
        name: String,
        query: String,
        param_types: Vec<u32>,
    },
    /// Bind (extended query): 'B'.
    Bind {
        portal: String,
        statement: String,
        params: Vec<Option<Vec<u8>>>,
        result_formats: Vec<i16>,
    },
    /// Describe: 'D'.
    Describe {
        kind: u8, // 'S' for statement, 'P' for portal
        name: String,
    },
    /// Execute: 'E'.
    Execute { portal: String, max_rows: i32 },
    /// Sync: 'S' — marks end of extended query pipeline.
    Sync,
    /// Terminate: 'X'.
    Terminate,
    /// Password response.
    PasswordMessage(String),
}

/// Parse a frontend message from a byte buffer.
/// Returns None if the buffer doesn't contain a complete message.
pub fn parse_frontend_message(buf: &mut BytesMut) -> Option<FrontendMessage> {
    if buf.len() < 4 {
        return None;
    }

    // Check for startup messages (no type byte — first 4 bytes are length)
    // Peek at the potential type byte
    let first_byte = buf[0];

    // Type byte is an ASCII letter for regular messages
    if first_byte.is_ascii_alphabetic() {
        // Regular message: type(1) + length(4) + payload
        if buf.len() < 5 {
            return None;
        }
        let len = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
        if buf.len() < 1 + len {
            return None;
        }
        let msg_type = buf[0];
        let payload = buf[5..1 + len].to_vec();
        buf.advance(1 + len);
        return parse_typed_message(msg_type, &payload);
    }

    // Startup message: length(4) + payload
    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if buf.len() < len {
        return None;
    }
    let payload = buf[4..len].to_vec();
    buf.advance(len);
    parse_startup_message(&payload)
}

fn parse_startup_message(payload: &[u8]) -> Option<FrontendMessage> {
    if payload.len() < 4 {
        return None;
    }
    let version = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);

    // SSL request: version = 80877103
    if version == 80877103 {
        return Some(FrontendMessage::SslRequest);
    }

    // Cancel request: version = 80877102
    if version == 80877102 {
        return None; // Ignore cancel requests for now
    }

    // Normal startup: version = 196608 (3.0)
    if version != 196608 {
        return None;
    }

    let mut params = HashMap::new();
    let mut i = 4;
    while i < payload.len() {
        let key_end = payload[i..].iter().position(|&b| b == 0)?;
        let key = String::from_utf8_lossy(&payload[i..i + key_end]).to_string();
        i += key_end + 1;
        if key.is_empty() {
            break;
        }
        let val_end = payload[i..].iter().position(|&b| b == 0)?;
        let val = String::from_utf8_lossy(&payload[i..i + val_end]).to_string();
        i += val_end + 1;
        params.insert(key, val);
    }

    Some(FrontendMessage::Startup { params })
}

fn parse_typed_message(msg_type: u8, payload: &[u8]) -> Option<FrontendMessage> {
    match msg_type {
        b'Q' => {
            // Simple query
            let query = read_cstring(payload, 0)?;
            Some(FrontendMessage::Query(query))
        }
        b'P' => {
            // Parse
            let (name, offset) = read_cstring_offset(payload, 0)?;
            let (query, offset) = read_cstring_offset(payload, offset)?;
            let num_params = read_i16(payload, offset)?;
            let mut param_types = Vec::new();
            let mut pos = offset + 2;
            for _ in 0..num_params {
                let oid = read_u32(payload, pos)?;
                param_types.push(oid);
                pos += 4;
            }
            Some(FrontendMessage::Parse {
                name,
                query,
                param_types,
            })
        }
        b'B' => {
            // Bind
            let (portal, offset) = read_cstring_offset(payload, 0)?;
            let (statement, offset) = read_cstring_offset(payload, offset)?;
            let num_format_codes = read_i16(payload, offset)? as usize;
            let mut pos = offset + 2;
            // Skip format codes
            pos += num_format_codes * 2;
            let num_params = read_i16(payload, pos)? as usize;
            pos += 2;
            let mut params = Vec::new();
            for _ in 0..num_params {
                let param_len = read_i32(payload, pos)?;
                pos += 4;
                if param_len == -1 {
                    params.push(None);
                } else {
                    let len = param_len as usize;
                    if pos + len > payload.len() {
                        return None;
                    }
                    params.push(Some(payload[pos..pos + len].to_vec()));
                    pos += len;
                }
            }
            let num_result_formats = read_i16(payload, pos).unwrap_or(0) as usize;
            pos += 2;
            let mut result_formats = Vec::new();
            for _ in 0..num_result_formats {
                result_formats.push(read_i16(payload, pos)?);
                pos += 2;
            }
            Some(FrontendMessage::Bind {
                portal,
                statement,
                params,
                result_formats,
            })
        }
        b'D' => {
            // Describe
            let kind = *payload.first()?;
            let name = read_cstring(payload, 1)?;
            Some(FrontendMessage::Describe { kind, name })
        }
        b'E' => {
            // Execute
            let (portal, offset) = read_cstring_offset(payload, 0)?;
            let max_rows = read_i32(payload, offset)?;
            Some(FrontendMessage::Execute { portal, max_rows })
        }
        b'S' => Some(FrontendMessage::Sync),
        b'X' => Some(FrontendMessage::Terminate),
        b'p' => {
            // Password message
            let password = read_cstring(payload, 0)?;
            Some(FrontendMessage::PasswordMessage(password))
        }
        _ => None,
    }
}

// ---------- Backend (server → client) message builders ----------

/// Write an AuthenticationOk message.
pub fn auth_ok(buf: &mut BytesMut) {
    buf.put_u8(b'R');
    buf.put_i32(8);
    buf.put_i32(0); // AuthenticationOk
}

/// Write AuthenticationCleartextPassword.
pub fn auth_cleartext(buf: &mut BytesMut) {
    buf.put_u8(b'R');
    buf.put_i32(8);
    buf.put_i32(3); // CleartextPassword
}

/// Write a ParameterStatus message.
pub fn parameter_status(buf: &mut BytesMut, key: &str, value: &str) {
    let len = 4 + key.len() + 1 + value.len() + 1;
    buf.put_u8(b'S');
    buf.put_i32(len as i32);
    put_cstring(buf, key);
    put_cstring(buf, value);
}

/// Write BackendKeyData (process ID and secret key).
pub fn backend_key_data(buf: &mut BytesMut, pid: i32, secret: i32) {
    buf.put_u8(b'K');
    buf.put_i32(12);
    buf.put_i32(pid);
    buf.put_i32(secret);
}

/// Write a ReadyForQuery message.
pub fn ready_for_query(buf: &mut BytesMut, status: u8) {
    buf.put_u8(b'Z');
    buf.put_i32(5);
    buf.put_u8(status); // 'I' = idle, 'T' = in transaction, 'E' = failed transaction
}

/// Write a RowDescription message.
pub fn row_description(buf: &mut BytesMut, columns: &[ColumnDesc]) {
    let mut body = BytesMut::new();
    body.put_i16(columns.len() as i16);
    for col in columns {
        put_cstring(&mut body, &col.name);
        body.put_i32(0); // table OID
        body.put_i16(0); // column attr number
        body.put_i32(col.type_oid as i32);
        body.put_i16(col.type_size);
        body.put_i32(-1); // type modifier
        body.put_i16(0); // format: 0 = text
    }
    buf.put_u8(b'T');
    buf.put_i32(4 + body.len() as i32);
    buf.extend_from_slice(&body);
}

/// Write a DataRow message.
pub fn data_row(buf: &mut BytesMut, values: &[Option<&str>]) {
    let mut body = BytesMut::new();
    body.put_i16(values.len() as i16);
    for val in values {
        match val {
            Some(s) => {
                body.put_i32(s.len() as i32);
                body.extend_from_slice(s.as_bytes());
            }
            None => {
                body.put_i32(-1); // NULL
            }
        }
    }
    buf.put_u8(b'D');
    buf.put_i32(4 + body.len() as i32);
    buf.extend_from_slice(&body);
}

/// Write a CommandComplete message.
pub fn command_complete(buf: &mut BytesMut, tag: &str) {
    let len = 4 + tag.len() + 1;
    buf.put_u8(b'C');
    buf.put_i32(len as i32);
    put_cstring(buf, tag);
}

/// Write an ErrorResponse message.
pub fn error_response(buf: &mut BytesMut, severity: &str, code: &str, message: &str) {
    let mut body = BytesMut::new();
    body.put_u8(b'S');
    put_cstring(&mut body, severity);
    body.put_u8(b'C');
    put_cstring(&mut body, code);
    body.put_u8(b'M');
    put_cstring(&mut body, message);
    body.put_u8(0); // terminator
    buf.put_u8(b'E');
    buf.put_i32(4 + body.len() as i32);
    buf.extend_from_slice(&body);
}

/// Write a NoticeResponse message.
pub fn notice_response(buf: &mut BytesMut, message: &str) {
    let mut body = BytesMut::new();
    body.put_u8(b'S');
    put_cstring(&mut body, "NOTICE");
    body.put_u8(b'M');
    put_cstring(&mut body, message);
    body.put_u8(0);
    buf.put_u8(b'N');
    buf.put_i32(4 + body.len() as i32);
    buf.extend_from_slice(&body);
}

/// Write an EmptyQueryResponse.
pub fn empty_query_response(buf: &mut BytesMut) {
    buf.put_u8(b'I');
    buf.put_i32(4);
}

/// Write ParseComplete.
pub fn parse_complete(buf: &mut BytesMut) {
    buf.put_u8(b'1');
    buf.put_i32(4);
}

/// Write BindComplete.
pub fn bind_complete(buf: &mut BytesMut) {
    buf.put_u8(b'2');
    buf.put_i32(4);
}

/// Write NoData (response to Describe when there are no result columns).
pub fn no_data(buf: &mut BytesMut) {
    buf.put_u8(b'n');
    buf.put_i32(4);
}

/// Write ParameterDescription.
pub fn parameter_description(buf: &mut BytesMut, param_types: &[u32]) {
    let mut body = BytesMut::new();
    body.put_i16(param_types.len() as i16);
    for &oid in param_types {
        body.put_i32(oid as i32);
    }
    buf.put_u8(b't');
    buf.put_i32(4 + body.len() as i32);
    buf.extend_from_slice(&body);
}

/// Write an SSL rejection (single byte 'N').
pub fn ssl_reject(buf: &mut BytesMut) {
    buf.put_u8(b'N');
}

/// Write an SSL acceptance (single byte 'S').
pub fn ssl_accept(buf: &mut BytesMut) {
    buf.put_u8(b'S');
}

// ---------- Column descriptor ----------

/// Describes a column in a result set.
#[derive(Debug, Clone)]
pub struct ColumnDesc {
    pub name: String,
    pub type_oid: u32,
    pub type_size: i16,
}

/// Postgres type OIDs.
pub mod oid {
    pub const BOOL: u32 = 16;
    pub const INT4: u32 = 23;
    pub const INT8: u32 = 20;
    pub const FLOAT8: u32 = 701;
    pub const TEXT: u32 = 25;
    pub const JSON: u32 = 114;
    pub const JSONB: u32 = 3802;
    pub const TIMESTAMP: u32 = 1114;
    pub const BYTEA: u32 = 17;
}

// ---------- Helpers ----------

fn put_cstring(buf: &mut BytesMut, s: &str) {
    buf.extend_from_slice(s.as_bytes());
    buf.put_u8(0);
}

fn read_cstring(data: &[u8], offset: usize) -> Option<String> {
    let nul_pos = data[offset..].iter().position(|&b| b == 0)?;
    Some(String::from_utf8_lossy(&data[offset..offset + nul_pos]).to_string())
}

fn read_cstring_offset(data: &[u8], offset: usize) -> Option<(String, usize)> {
    let nul_pos = data[offset..].iter().position(|&b| b == 0)?;
    let s = String::from_utf8_lossy(&data[offset..offset + nul_pos]).to_string();
    Some((s, offset + nul_pos + 1))
}

fn read_i16(data: &[u8], offset: usize) -> Option<i16> {
    if offset + 2 > data.len() {
        return None;
    }
    Some(i16::from_be_bytes([data[offset], data[offset + 1]]))
}

fn read_i32(data: &[u8], offset: usize) -> Option<i32> {
    if offset + 4 > data.len() {
        return None;
    }
    Some(i32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    if offset + 4 > data.len() {
        return None;
    }
    Some(u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Advance a BytesMut buffer (consume bytes from the front).
trait AdvanceBuf {
    fn advance(&mut self, cnt: usize);
}

impl AdvanceBuf for BytesMut {
    fn advance(&mut self, cnt: usize) {
        *self = self.split_off(cnt);
    }
}
