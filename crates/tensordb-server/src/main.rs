//! TensorDB PostgreSQL Wire Protocol Server
//!
//! Accepts Postgres client connections (psql, JDBC, libpq, etc.) and routes
//! SQL queries to the embedded TensorDB engine. Supports optional TLS.
//!
//! Usage:
//!   tensordb-server --data-dir ./data --port 5433
//!   tensordb-server --data-dir ./data --port 5433 --tls-cert cert.pem --tls-key key.pem

mod handler;
mod pgwire;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use tokio::net::TcpListener;
use tracing::{error, info};

use tensordb_core::config::Config;
use tensordb_core::Database;

fn load_tls_config(cert_path: &str, key_path: &str) -> Arc<rustls::ServerConfig> {
    use std::io::BufReader;

    let cert_file = std::fs::File::open(cert_path).unwrap_or_else(|e| {
        eprintln!("failed to open TLS cert file {cert_path}: {e}");
        std::process::exit(1);
    });
    let key_file = std::fs::File::open(key_path).unwrap_or_else(|e| {
        eprintln!("failed to open TLS key file {key_path}: {e}");
        std::process::exit(1);
    });

    let certs: Vec<_> = rustls_pemfile::certs(&mut BufReader::new(cert_file))
        .filter_map(|r| r.ok())
        .collect();
    if certs.is_empty() {
        eprintln!("no valid certificates found in {cert_path}");
        std::process::exit(1);
    }

    let key = rustls_pemfile::private_key(&mut BufReader::new(key_file))
        .unwrap_or_else(|e| {
            eprintln!("failed to read TLS key: {e}");
            std::process::exit(1);
        })
        .unwrap_or_else(|| {
            eprintln!("no private key found in {key_path}");
            std::process::exit(1);
        });

    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .unwrap_or_else(|e| {
            eprintln!("invalid TLS configuration: {e}");
            std::process::exit(1);
        });

    Arc::new(config)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let mut data_dir = PathBuf::from("./tensordb_data");
    let mut port: u16 = 5433;
    let mut tls_cert: Option<String> = None;
    let mut tls_key: Option<String> = None;

    // Simple arg parsing
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" | "-d" => {
                if i + 1 < args.len() {
                    data_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("--data-dir requires a path argument");
                    std::process::exit(1);
                }
            }
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("invalid port number: {}", args[i + 1]);
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("--port requires a number argument");
                    std::process::exit(1);
                }
            }
            "--tls-cert" => {
                if i + 1 < args.len() {
                    tls_cert = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--tls-cert requires a path argument");
                    std::process::exit(1);
                }
            }
            "--tls-key" => {
                if i + 1 < args.len() {
                    tls_key = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--tls-key requires a path argument");
                    std::process::exit(1);
                }
            }
            "--help" | "-h" => {
                println!("tensordb-server - PostgreSQL wire protocol server for TensorDB");
                println!();
                println!("Usage: tensordb-server [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -d, --data-dir <PATH>   Data directory (default: ./tensordb_data)");
                println!("  -p, --port <PORT>       Listen port (default: 5433)");
                println!("      --tls-cert <PATH>   TLS certificate PEM file");
                println!("      --tls-key <PATH>    TLS private key PEM file");
                println!("  -h, --help              Show this help");
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // Load TLS config if both cert and key are provided
    let tls_acceptor = match (&tls_cert, &tls_key) {
        (Some(cert), Some(key)) => {
            let config = load_tls_config(cert, key);
            info!("TLS enabled (cert={cert}, key={key})");
            Some(tokio_rustls::TlsAcceptor::from(config))
        }
        (Some(_), None) | (None, Some(_)) => {
            eprintln!("both --tls-cert and --tls-key are required for TLS");
            std::process::exit(1);
        }
        (None, None) => None,
    };

    // Open the database
    info!("opening database at {}", data_dir.display());
    let db = match Database::open(&data_dir, Config::default()) {
        Ok(db) => Arc::new(db),
        Err(e) => {
            error!("failed to open database: {e}");
            std::process::exit(1);
        }
    };

    // Start TCP listener
    let addr = format!("0.0.0.0:{port}");
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("failed to bind to {addr}: {e}");
            std::process::exit(1);
        }
    };

    let tls_mode = if tls_acceptor.is_some() {
        "TLS"
    } else {
        "plaintext"
    };
    info!("TensorDB pgwire server listening on {addr} ({tls_mode})");
    info!("Connect with: psql -h localhost -p {port} -U tensordb");

    let conn_counter = AtomicU32::new(1);
    let tls_acceptor = tls_acceptor.map(Arc::new);

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let db = Arc::clone(&db);
                let conn_id = conn_counter.fetch_add(1, Ordering::Relaxed);
                let tls = tls_acceptor.clone();
                tokio::spawn(async move {
                    handler::handle_connection(stream, db, conn_id, tls).await;
                });
            }
            Err(e) => {
                error!("accept error: {e}");
            }
        }
    }
}
