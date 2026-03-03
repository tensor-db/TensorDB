use std::convert::Infallible;
use std::sync::Arc;

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tracing::{error, info};

use tensordb_core::Database;

fn health_response(db: &Database) -> Response<Full<Bytes>> {
    let uptime_ms = db.uptime_ms();
    let shard_count = db.shard_count();
    let cache_hit_rate = db.block_cache().hit_rate();

    let (total_puts, total_gets) = match db.stats() {
        Ok(stats) => (stats.puts, stats.gets),
        Err(_) => (0, 0),
    };

    let storage = db.storage_info();
    let mut sstable_bytes: u64 = 0;
    let mut memtable_bytes: usize = 0;
    for (_, info) in &storage {
        sstable_bytes += info.level_sizes.iter().sum::<u64>();
        memtable_bytes += info.memtable_bytes + info.immutable_memtable_bytes;
    }

    let body = serde_json::json!({
        "status": "healthy",
        "uptime_ms": uptime_ms,
        "shard_count": shard_count,
        "total_puts": total_puts,
        "total_gets": total_gets,
        "cache_hit_rate": cache_hit_rate,
        "sstable_bytes": sstable_bytes,
        "memtable_bytes": memtable_bytes,
        "ready": true,
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap()
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    db: Arc<Database>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    if req.uri().path() == "/health" {
        Ok(health_response(&db))
    } else {
        let body = serde_json::json!({"error": "not found"});
        Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body.to_string())))
            .unwrap())
    }
}

pub async fn spawn_health_server(db: Arc<Database>, port: u16) {
    let addr = format!("0.0.0.0:{port}");
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("failed to bind health endpoint to {addr}: {e}");
            return;
        }
    };

    info!("Health endpoint listening on http://{addr}/health");

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let db = Arc::clone(&db);
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let svc = service_fn(move |req| {
                        let db = Arc::clone(&db);
                        async move { handle_request(req, db).await }
                    });
                    if let Err(e) = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, svc)
                        .await
                    {
                        error!("health connection error: {e}");
                    }
                });
            }
            Err(e) => {
                error!("health accept error: {e}");
            }
        }
    }
}
