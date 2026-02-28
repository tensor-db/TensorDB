//! Connection handler: bridges pgwire protocol to TensorDB SQL engine.

use std::sync::Arc;

use bytes::BytesMut;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{debug, error, info};

use tensordb_core::sql::exec::SqlResult;
use tensordb_core::Database;

use crate::pgwire::{self, ColumnDesc, FrontendMessage};

/// Handle a single Postgres client connection, optionally upgrading to TLS.
pub async fn handle_connection(
    stream: TcpStream,
    db: Arc<Database>,
    conn_id: u32,
    tls_acceptor: Option<Arc<tokio_rustls::TlsAcceptor>>,
) {
    let peer = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    info!("new connection from {peer} (id={conn_id})");

    // We need to handle the initial startup which may include SSL negotiation.
    // Read the first message to see if it's an SSL request.
    let (mut reader, mut writer) = stream.into_split();
    let mut read_buf = BytesMut::with_capacity(8192);
    let mut write_buf = BytesMut::with_capacity(8192);

    // Read initial data
    match reader.read_buf(&mut read_buf).await {
        Ok(0) => {
            debug!("connection closed immediately (id={conn_id})");
            return;
        }
        Ok(_) => {}
        Err(e) => {
            error!("read error during startup: {e}");
            return;
        }
    }

    // Check if this is an SSL request
    if let Some(FrontendMessage::SslRequest) = pgwire::parse_frontend_message(&mut read_buf) {
        if let Some(ref acceptor) = tls_acceptor {
            // Accept SSL: send 'S' then upgrade
            write_buf.clear();
            pgwire::ssl_accept(&mut write_buf);
            if writer.write_all(&write_buf).await.is_err() {
                return;
            }
            write_buf.clear();

            // Reunite the stream for TLS upgrade
            let stream = reader.reunite(writer).expect("reunite failed");
            match acceptor.accept(stream).await {
                Ok(tls_stream) => {
                    info!("TLS handshake complete (id={conn_id})");
                    let (tls_reader, tls_writer) = tokio::io::split(tls_stream);
                    run_connection(tls_reader, tls_writer, db, conn_id, read_buf).await;
                }
                Err(e) => {
                    error!("TLS handshake failed: {e} (id={conn_id})");
                }
            }
        } else {
            // No TLS configured: reject SSL and continue plaintext
            write_buf.clear();
            pgwire::ssl_reject(&mut write_buf);
            if writer.write_all(&write_buf).await.is_err() {
                return;
            }
            write_buf.clear();
            run_connection(reader, writer, db, conn_id, read_buf).await;
        }
    } else {
        // Not an SSL request — continue plaintext
        run_connection(reader, writer, db, conn_id, read_buf).await;
    }
}

/// Run the pgwire connection protocol on any async reader/writer (plain TCP or TLS).
async fn run_connection<R, W>(
    mut reader: R,
    mut writer: W,
    db: Arc<Database>,
    conn_id: u32,
    mut read_buf: BytesMut,
) where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut write_buf = BytesMut::with_capacity(8192);

    // Phase 1: Startup
    loop {
        // Try to parse a startup message from existing buffer
        if let Some(msg) = pgwire::parse_frontend_message(&mut read_buf) {
            match msg {
                FrontendMessage::SslRequest => {
                    // Second SSL request after rejection — reject again
                    pgwire::ssl_reject(&mut write_buf);
                    if writer.write_all(&write_buf).await.is_err() {
                        return;
                    }
                    write_buf.clear();
                    continue;
                }
                FrontendMessage::Startup { params } => {
                    let user = params.get("user").cloned().unwrap_or_default();
                    let database = params.get("database").cloned().unwrap_or_default();
                    info!("startup: user={user} database={database} (id={conn_id})");

                    // Send AuthenticationOk (trust mode)
                    pgwire::auth_ok(&mut write_buf);

                    // Send parameter status messages
                    pgwire::parameter_status(&mut write_buf, "server_version", "15.0.0");
                    pgwire::parameter_status(&mut write_buf, "server_encoding", "UTF8");
                    pgwire::parameter_status(&mut write_buf, "client_encoding", "UTF8");
                    pgwire::parameter_status(&mut write_buf, "DateStyle", "ISO, MDY");
                    pgwire::parameter_status(&mut write_buf, "TimeZone", "UTC");
                    pgwire::parameter_status(&mut write_buf, "integer_datetimes", "on");
                    pgwire::parameter_status(&mut write_buf, "standard_conforming_strings", "on");

                    // Send BackendKeyData
                    pgwire::backend_key_data(&mut write_buf, conn_id as i32, 0);

                    // ReadyForQuery
                    pgwire::ready_for_query(&mut write_buf, b'I');

                    if writer.write_all(&write_buf).await.is_err() {
                        return;
                    }
                    write_buf.clear();
                    break; // Move to query phase
                }
                _ => {
                    error!("unexpected message during startup");
                    return;
                }
            }
        }

        // Read more data
        match reader.read_buf(&mut read_buf).await {
            Ok(0) => {
                debug!("connection closed during startup (id={conn_id})");
                return;
            }
            Ok(_) => {}
            Err(e) => {
                error!("read error during startup: {e}");
                return;
            }
        }
    }

    // Phase 2: Query loop
    loop {
        // Check if we already have a complete message in the buffer
        while let Some(msg) = pgwire::parse_frontend_message(&mut read_buf) {
            match msg {
                FrontendMessage::Query(query) => {
                    handle_simple_query(&db, &query, &mut write_buf);
                    pgwire::ready_for_query(&mut write_buf, b'I');
                    if writer.write_all(&write_buf).await.is_err() {
                        return;
                    }
                    write_buf.clear();
                }
                FrontendMessage::Parse {
                    name: _,
                    query: _,
                    param_types: _,
                } => {
                    pgwire::parse_complete(&mut write_buf);
                }
                FrontendMessage::Bind { .. } => {
                    pgwire::bind_complete(&mut write_buf);
                }
                FrontendMessage::Describe { .. } => {
                    pgwire::no_data(&mut write_buf);
                }
                FrontendMessage::Execute { .. } => {
                    pgwire::command_complete(&mut write_buf, "SELECT 0");
                }
                FrontendMessage::Sync => {
                    pgwire::ready_for_query(&mut write_buf, b'I');
                    if writer.write_all(&write_buf).await.is_err() {
                        return;
                    }
                    write_buf.clear();
                }
                FrontendMessage::Terminate => {
                    info!("client disconnected (id={conn_id})");
                    return;
                }
                _ => {}
            }
        }

        // Read more data
        match reader.read_buf(&mut read_buf).await {
            Ok(0) => {
                info!("connection closed (id={conn_id})");
                return;
            }
            Ok(_) => {}
            Err(e) => {
                error!("read error: {e}");
                return;
            }
        }
    }
}

/// Handle a simple query (Q message) — execute SQL and write result messages.
fn handle_simple_query(db: &Database, query: &str, buf: &mut BytesMut) {
    let query = query.trim();
    if query.is_empty() {
        pgwire::empty_query_response(buf);
        return;
    }

    // Handle some Postgres-specific queries that clients send
    let lower = query.to_lowercase();
    if lower.starts_with("set ") || lower == "begin" || lower == "commit" || lower == "rollback" {
        pgwire::command_complete(buf, "SET");
        return;
    }

    // Handle psql meta-commands via pg_catalog
    if lower.starts_with("select") && lower.contains("pg_catalog") {
        handle_pg_catalog_query(db, query, buf);
        return;
    }

    // Execute the query against TensorDB
    match db.sql(query) {
        Ok(result) => match result {
            SqlResult::Rows(rows) => {
                let columns = if let Some(first) = rows.first() {
                    extract_columns_from_json(first)
                } else {
                    vec![ColumnDesc {
                        name: "result".to_string(),
                        type_oid: pgwire::oid::TEXT,
                        type_size: -1,
                    }]
                };

                pgwire::row_description(buf, &columns);

                for row_bytes in &rows {
                    let values = extract_values_from_json(row_bytes, &columns);
                    let refs: Vec<Option<&str>> = values.iter().map(|v| v.as_deref()).collect();
                    pgwire::data_row(buf, &refs);
                }

                pgwire::command_complete(buf, &format!("SELECT {}", rows.len()));
            }
            SqlResult::Affected { rows, message, .. } => {
                let tag = if lower.starts_with("insert") {
                    format!("INSERT 0 {rows}")
                } else if lower.starts_with("update") {
                    format!("UPDATE {rows}")
                } else if lower.starts_with("delete") {
                    format!("DELETE {rows}")
                } else {
                    message.clone()
                };
                pgwire::command_complete(buf, &tag);
            }
            SqlResult::Explain(text) => {
                let columns = vec![ColumnDesc {
                    name: "QUERY PLAN".to_string(),
                    type_oid: pgwire::oid::TEXT,
                    type_size: -1,
                }];
                pgwire::row_description(buf, &columns);
                for line in text.lines() {
                    pgwire::data_row(buf, &[Some(line)]);
                }
                pgwire::command_complete(buf, &format!("EXPLAIN {}", text.lines().count()));
            }
        },
        Err(e) => {
            pgwire::error_response(buf, "ERROR", "42000", &e.to_string());
        }
    }
}

/// Handle queries against pg_catalog system tables.
fn handle_pg_catalog_query(db: &Database, _query: &str, buf: &mut BytesMut) {
    let columns = vec![ColumnDesc {
        name: "name".to_string(),
        type_oid: pgwire::oid::TEXT,
        type_size: -1,
    }];

    if let Ok(SqlResult::Rows(tables)) = db.sql("SHOW TABLES;") {
        pgwire::row_description(buf, &columns);
        for table in &tables {
            let name = String::from_utf8_lossy(table).to_string();
            pgwire::data_row(buf, &[Some(&name)]);
        }
        pgwire::command_complete(buf, &format!("SELECT {}", tables.len()));
    } else {
        pgwire::row_description(buf, &columns);
        pgwire::command_complete(buf, "SELECT 0");
    }
}

/// Extract column descriptors from a JSON row.
fn extract_columns_from_json(row: &[u8]) -> Vec<ColumnDesc> {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_slice::<serde_json::Value>(row) {
        map.keys()
            .map(|k| {
                let type_oid = match map.get(k) {
                    Some(serde_json::Value::Number(_)) => pgwire::oid::FLOAT8,
                    Some(serde_json::Value::Bool(_)) => pgwire::oid::BOOL,
                    Some(serde_json::Value::Object(_)) | Some(serde_json::Value::Array(_)) => {
                        pgwire::oid::JSONB
                    }
                    _ => pgwire::oid::TEXT,
                };
                ColumnDesc {
                    name: k.clone(),
                    type_oid,
                    type_size: -1,
                }
            })
            .collect()
    } else {
        vec![ColumnDesc {
            name: "result".to_string(),
            type_oid: pgwire::oid::TEXT,
            type_size: -1,
        }]
    }
}

/// Extract column values from a JSON row.
fn extract_values_from_json(row: &[u8], columns: &[ColumnDesc]) -> Vec<Option<String>> {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_slice::<serde_json::Value>(row) {
        columns
            .iter()
            .map(|col| {
                map.get(&col.name).map(|v| match v {
                    serde_json::Value::Null => None,
                    serde_json::Value::String(s) => Some(s.clone()),
                    serde_json::Value::Number(n) => Some(n.to_string()),
                    serde_json::Value::Bool(b) => Some(if *b { "t" } else { "f" }.to_string()),
                    other => Some(other.to_string()),
                })?
            })
            .collect()
    } else {
        vec![Some(String::from_utf8_lossy(row).to_string())]
    }
}
