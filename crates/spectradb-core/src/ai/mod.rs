pub mod access_stats;
pub mod cache_advisor;
pub mod compaction_advisor;
pub mod inference;
pub mod ml_pipeline;
pub mod query_advisor;

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, select, unbounded, Receiver, Sender};
use parking_lot::Mutex;

use crate::config::Config;
use crate::engine::shard::{ChangeEvent, ShardCommand, WriteBatchItem};
use crate::native_bridge::Hasher;
use crate::util::time::unix_millis;

pub const AI_INTERNAL_KEY_PREFIX: &[u8] = b"__ai/";
const AI_INSIGHT_KEY_PREFIX: &str = "__ai/insight";
const AI_CORRELATION_KEY_PREFIX: &str = "__ai/correlation";
const AI_MODEL_ID: &str = "core-ai";
const AI_CORRELATION_WINDOW_MS: u64 = 60_000;

#[derive(Debug, Clone, Default)]
pub struct AiRuntimeStats {
    pub enabled: bool,
    pub events_received: u64,
    pub insights_written: u64,
    pub skipped_internal_keys: u64,
    pub skipped_empty_docs: u64,
    pub write_failures: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AiInsight {
    #[serde(default)]
    pub insight_id: String,
    pub source_key_hex: String,
    pub source_commit_ts: u64,
    #[serde(default)]
    pub cluster_id: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub risk_score: f64,
    pub troubleshooting_hint: String,
    #[serde(default)]
    pub provenance: AiInsightProvenance,
    pub model_id: String,
    pub generated_at_ms: u64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AiInsightProvenance {
    #[serde(default)]
    pub matched_signals: Vec<String>,
    #[serde(default)]
    pub risk_factors: Vec<String>,
    #[serde(default)]
    pub hint_basis: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AiCorrelationRef {
    pub cluster_id: String,
    pub insight_id: String,
    pub source_key_hex: String,
    pub source_commit_ts: u64,
    pub risk_score: f64,
    pub tags: Vec<String>,
    pub summary: String,
    pub generated_at_ms: u64,
}

pub struct AiRuntimeHandle {
    stats: Arc<Mutex<AiRuntimeStats>>,
    shutdown_tx: Sender<()>,
    join: Option<JoinHandle<()>>,
}

impl AiRuntimeHandle {
    pub fn spawn(
        events_rx: Receiver<ChangeEvent>,
        shard_senders: Vec<Sender<ShardCommand>>,
        hasher: Arc<dyn Hasher + Send + Sync>,
        config: Config,
    ) -> std::io::Result<Self> {
        let stats = Arc::new(Mutex::new(AiRuntimeStats {
            enabled: true,
            ..AiRuntimeStats::default()
        }));
        let (shutdown_tx, shutdown_rx) = unbounded();
        let stats_for_thread = stats.clone();

        let join = thread::Builder::new()
            .name("spectradb-ai-runtime".to_string())
            .spawn(move || {
                run_ai_runtime(
                    events_rx,
                    shutdown_rx,
                    shard_senders,
                    hasher,
                    config,
                    stats_for_thread,
                )
            })?;

        Ok(Self {
            stats,
            shutdown_tx,
            join: Some(join),
        })
    }

    pub fn stats(&self) -> AiRuntimeStats {
        self.stats.lock().clone()
    }

    pub fn shutdown(&mut self) {
        let _ = self.shutdown_tx.send(());
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for AiRuntimeHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn is_internal_ai_key(user_key: &[u8]) -> bool {
    user_key.starts_with(AI_INTERNAL_KEY_PREFIX)
}

pub fn insight_prefix_for_source(user_key: &[u8]) -> Vec<u8> {
    format!("{AI_INSIGHT_KEY_PREFIX}/{}/", hex_encode(user_key)).into_bytes()
}

pub fn insight_storage_key(user_key: &[u8], commit_ts: u64) -> Vec<u8> {
    format!(
        "{AI_INSIGHT_KEY_PREFIX}/{}/{commit_ts:020}",
        hex_encode(user_key)
    )
    .into_bytes()
}

pub fn correlation_prefix_for_cluster(cluster_id: &str) -> Vec<u8> {
    format!("{AI_CORRELATION_KEY_PREFIX}/{cluster_id}/").into_bytes()
}

pub fn correlation_storage_key(cluster_id: &str, user_key: &[u8], commit_ts: u64) -> Vec<u8> {
    format!(
        "{AI_CORRELATION_KEY_PREFIX}/{cluster_id}/{commit_ts:020}/{}",
        hex_encode(user_key)
    )
    .into_bytes()
}

pub fn insight_id_for_source(user_key: &[u8], commit_ts: u64) -> String {
    format!("{}/{}", hex_encode(user_key), commit_ts)
}

pub fn parse_insight_id(id: &str) -> Option<(Vec<u8>, u64)> {
    let (source_key_hex, commit_ts) = id.split_once('/')?;
    let source_key = hex_decode(source_key_hex)?;
    let commit_ts = commit_ts.parse::<u64>().ok()?;
    Some((source_key, commit_ts))
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = from_hex_nibble(bytes[i])?;
        let lo = from_hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

fn from_hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn run_ai_runtime(
    events_rx: Receiver<ChangeEvent>,
    shutdown_rx: Receiver<()>,
    shard_senders: Vec<Sender<ShardCommand>>,
    hasher: Arc<dyn Hasher + Send + Sync>,
    config: Config,
    stats: Arc<Mutex<AiRuntimeStats>>,
) {
    if shard_senders.is_empty() {
        return;
    }

    let batch_window = Duration::from_millis(config.ai_batch_window_ms.max(1));
    let batch_max_events = config.ai_batch_max_events.max(1);

    loop {
        let first_evt = select! {
            recv(shutdown_rx) -> _ => break,
            recv(events_rx) -> evt => {
                let Ok(evt) = evt else { break; };
                evt
            }
        };

        let mut batch = Vec::with_capacity(batch_max_events);
        batch.push(first_evt);
        let keep_running = collect_event_batch(
            &mut batch,
            batch_max_events,
            batch_window,
            &events_rx,
            &shutdown_rx,
        );
        process_event_batch(batch, &shard_senders, hasher.as_ref(), &stats);
        if !keep_running {
            break;
        }
    }
}

fn collect_event_batch(
    batch: &mut Vec<ChangeEvent>,
    max_events: usize,
    batch_window: Duration,
    events_rx: &Receiver<ChangeEvent>,
    shutdown_rx: &Receiver<()>,
) -> bool {
    let deadline = Instant::now() + batch_window;
    while batch.len() < max_events {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let wait_for = deadline.saturating_duration_since(now);
        select! {
            recv(shutdown_rx) -> _ => return false,
            recv(events_rx) -> evt => {
                match evt {
                    Ok(evt) => batch.push(evt),
                    Err(_) => return false,
                }
            }
            default(wait_for) => break,
        }
    }
    true
}

fn process_event_batch(
    batch: Vec<ChangeEvent>,
    shard_senders: &[Sender<ShardCommand>],
    hasher: &dyn Hasher,
    stats: &Arc<Mutex<AiRuntimeStats>>,
) {
    let mut writes_by_shard: Vec<Vec<WriteBatchItem>> = vec![Vec::new(); shard_senders.len()];
    let mut insight_counts_by_shard: Vec<u64> = vec![0; shard_senders.len()];

    for evt in batch {
        stats.lock().events_received += 1;

        if is_internal_ai_key(&evt.user_key) {
            stats.lock().skipped_internal_keys += 1;
            continue;
        }
        if evt.doc.is_empty() {
            stats.lock().skipped_empty_docs += 1;
            continue;
        }

        let insight = match synthesize_insight(&evt, hasher, stats) {
            Ok(i) => i,
            Err(_) => {
                stats.lock().write_failures += 1;
                continue;
            }
        };

        let insight_key = insight_storage_key(&evt.user_key, evt.commit_ts);
        let insight_payload = match serde_json::to_vec(&insight) {
            Ok(v) => v,
            Err(_) => {
                stats.lock().write_failures += 1;
                continue;
            }
        };
        let shard_id = (hasher.hash64(&insight_key) as usize) % shard_senders.len();
        writes_by_shard[shard_id].push(WriteBatchItem {
            user_key: insight_key,
            doc: insight_payload,
            valid_from: evt.valid_from,
            valid_to: evt.valid_to,
            schema_version: Some(1),
        });
        insight_counts_by_shard[shard_id] += 1;

        let correlation_key =
            correlation_storage_key(&insight.cluster_id, &evt.user_key, evt.commit_ts);
        let correlation = AiCorrelationRef {
            cluster_id: insight.cluster_id.clone(),
            insight_id: insight.insight_id.clone(),
            source_key_hex: insight.source_key_hex.clone(),
            source_commit_ts: insight.source_commit_ts,
            risk_score: insight.risk_score,
            tags: insight.tags.clone(),
            summary: insight.summary.clone(),
            generated_at_ms: insight.generated_at_ms,
        };
        let correlation_payload = match serde_json::to_vec(&correlation) {
            Ok(v) => v,
            Err(_) => {
                stats.lock().write_failures += 1;
                continue;
            }
        };
        let correlation_shard_id = (hasher.hash64(&correlation_key) as usize) % shard_senders.len();
        writes_by_shard[correlation_shard_id].push(WriteBatchItem {
            user_key: correlation_key,
            doc: correlation_payload,
            valid_from: evt.valid_from,
            valid_to: evt.valid_to,
            schema_version: Some(1),
        });
    }

    for (shard_id, entries) in writes_by_shard.into_iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let entry_count = entries.len() as u64;
        let (resp_tx, resp_rx) = bounded(1);
        if shard_senders[shard_id]
            .send(ShardCommand::WriteBatch {
                entries,
                resp: resp_tx,
            })
            .is_err()
        {
            stats.lock().write_failures += entry_count;
            continue;
        }
        match resp_rx.recv() {
            Ok(Ok(commit_ts)) if commit_ts.len() as u64 == entry_count => {
                stats.lock().insights_written += insight_counts_by_shard[shard_id];
            }
            _ => {
                stats.lock().write_failures += entry_count;
            }
        }
    }
}

fn synthesize_insight(
    evt: &ChangeEvent,
    hasher: &dyn Hasher,
    stats: &Arc<Mutex<AiRuntimeStats>>,
) -> std::result::Result<AiInsight, String> {
    let text = extract_semantic_text(&evt.doc);
    let (payload, model_id) = run_model_backend(&text, stats)?;
    let generated_at_ms = unix_millis();
    let cluster_id = derive_cluster_id(hasher, &payload.tags, payload.risk_score, generated_at_ms);

    Ok(AiInsight {
        insight_id: insight_id_for_source(&evt.user_key, evt.commit_ts),
        source_key_hex: hex_encode(&evt.user_key),
        source_commit_ts: evt.commit_ts,
        cluster_id,
        summary: payload.summary,
        tags: payload.tags,
        risk_score: payload.risk_score,
        troubleshooting_hint: payload.troubleshooting_hint,
        provenance: payload.provenance,
        model_id,
        generated_at_ms,
    })
}

fn extract_semantic_text(doc: &[u8]) -> String {
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(doc) {
        let mut chunks = Vec::new();
        flatten_json("", &v, &mut chunks);
        if !chunks.is_empty() {
            return chunks.join(" ");
        }
    }
    String::from_utf8_lossy(doc).to_string()
}

fn flatten_json(prefix: &str, v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Null => {}
        serde_json::Value::Bool(b) => out.push(format!("{prefix}{b}")),
        serde_json::Value::Number(n) => out.push(format!("{prefix}{n}")),
        serde_json::Value::String(s) => out.push(format!("{prefix}{s}")),
        serde_json::Value::Array(arr) => {
            for item in arr {
                flatten_json(prefix, item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for (k, val) in map {
                let p = if prefix.is_empty() {
                    format!("{k}=")
                } else {
                    format!("{prefix}{k}=")
                };
                flatten_json(&p, val, out);
            }
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ModelSynthesis {
    summary: String,
    tags: Vec<String>,
    risk_score: f64,
    troubleshooting_hint: String,
    provenance: AiInsightProvenance,
}

fn native_core_synthesis(text: &str) -> ModelSynthesis {
    let base_tags = derive_tags(text);
    let summary = summarize(text, 220);
    let lower = text.to_ascii_lowercase();
    let mut matched_signals = Vec::new();

    let mut tags = base_tags.clone();
    if lower.contains("deploy") || lower.contains("release") {
        tags.push("release-signal".to_string());
        matched_signals.push("release keyword".to_string());
    }
    if lower.contains("connection") || lower.contains("reset") || lower.contains("upstream") {
        tags.push("connectivity".to_string());
        matched_signals.push("connectivity keyword".to_string());
    }
    if lower.contains("queue") || lower.contains("backlog") || lower.contains("throttle") {
        tags.push("backpressure".to_string());
        matched_signals.push("backpressure keyword".to_string());
    }
    tags.sort_unstable();
    tags.dedup();

    let (mut risk_score, mut risk_factors) = derive_risk_score(text, &tags);
    if lower.contains("degraded") || lower.contains("outage") {
        risk_score += 0.15;
        risk_factors.push("degraded/outage keyword".to_string());
    }
    if lower.contains("retry") && lower.contains("failed") {
        risk_score += 0.10;
        risk_factors.push("retry+failed pair".to_string());
    }
    risk_score = risk_score.clamp(0.0, 1.0);

    let (troubleshooting_hint, hint_basis) = if tags.iter().any(|t| t == "connectivity") {
        (
            "Inspect upstream connectivity, retry bursts, and shard health before escalating."
                .to_string(),
            "connectivity tag present".to_string(),
        )
    } else if tags.iter().any(|t| t == "release-signal") {
        (
            "Compare pre/post-release timelines and isolate key regressions via AS OF snapshots."
                .to_string(),
            "release-signal tag present".to_string(),
        )
    } else {
        (
            derive_hint(&tags, risk_score),
            "default hint policy".to_string(),
        )
    };

    normalize_synthesis(ModelSynthesis {
        summary,
        tags,
        risk_score,
        troubleshooting_hint,
        provenance: AiInsightProvenance {
            matched_signals,
            risk_factors,
            hint_basis,
        },
    })
}

fn run_model_backend(
    text: &str,
    _stats: &Arc<Mutex<AiRuntimeStats>>,
) -> std::result::Result<(ModelSynthesis, String), String> {
    Ok((native_core_synthesis(text), AI_MODEL_ID.to_string()))
}

fn derive_cluster_id(
    hasher: &dyn Hasher,
    tags: &[String],
    risk_score: f64,
    generated_at_ms: u64,
) -> String {
    let mut stable_tags = tags.to_vec();
    stable_tags.sort_unstable();
    let severity = if risk_score >= 0.75 {
        "critical"
    } else if risk_score >= 0.45 {
        "elevated"
    } else {
        "normal"
    };
    let window = generated_at_ms / AI_CORRELATION_WINDOW_MS;
    let signature = format!("{window}|{severity}|{}", stable_tags.join(","));
    let signature_hash = hasher.hash64(signature.as_bytes());
    format!("{window:010}-{severity}-{:016x}", signature_hash)
}

fn normalize_synthesis(mut s: ModelSynthesis) -> ModelSynthesis {
    if s.summary.trim().is_empty() {
        s.summary = "no summary".to_string();
    }
    if s.tags.is_empty() {
        s.tags.push("general-observation".to_string());
    }
    s.risk_score = s.risk_score.clamp(0.0, 1.0);
    if s.troubleshooting_hint.trim().is_empty() {
        s.troubleshooting_hint =
            "Inspect event history and recent writes for this key.".to_string();
    }
    if s.provenance.hint_basis.trim().is_empty() {
        s.provenance.hint_basis = "fallback hint policy".to_string();
    }
    s
}

fn summarize(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let mut out = String::new();
    for ch in text.chars() {
        if out.len() + ch.len_utf8() > max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn derive_tags(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    let mut tags = Vec::new();

    if lower.contains("error") || lower.contains("failed") || lower.contains("exception") {
        tags.push("operational-error".to_string());
    }
    if lower.contains("refund") || lower.contains("chargeback") || lower.contains("dispute") {
        tags.push("payments-risk".to_string());
    }
    if lower.contains("latency") || lower.contains("timeout") || lower.contains("slow") {
        tags.push("performance".to_string());
    }
    if lower.contains("cpu") || lower.contains("memory") || lower.contains("disk") {
        tags.push("capacity-signal".to_string());
    }
    if lower.contains("login") || lower.contains("password") || lower.contains("token") {
        tags.push("security".to_string());
    }
    if tags.is_empty() {
        tags.push("general-observation".to_string());
    }
    tags
}

fn derive_risk_score(text: &str, tags: &[String]) -> (f64, Vec<String>) {
    let mut score = 0.15f64;
    let mut factors = vec!["base-risk".to_string()];
    if text.len() > 240 {
        score += 0.10;
        factors.push("long-payload".to_string());
    }
    if tags.iter().any(|t| t == "operational-error") {
        score += 0.25;
        factors.push("operational-error tag".to_string());
    }
    if tags.iter().any(|t| t == "payments-risk") {
        score += 0.25;
        factors.push("payments-risk tag".to_string());
    }
    if tags.iter().any(|t| t == "security") {
        score += 0.30;
        factors.push("security tag".to_string());
    }
    (score.clamp(0.0, 1.0), factors)
}

fn derive_hint(tags: &[String], risk_score: f64) -> String {
    if tags.iter().any(|t| t == "security") {
        return "Correlate with auth/token events and inspect temporal history via AS OF snapshots."
            .to_string();
    }
    if tags.iter().any(|t| t == "performance") {
        return "Run .stats and benchmark mode, then inspect compaction and bloom miss rates."
            .to_string();
    }
    if tags.iter().any(|t| t == "payments-risk") {
        return "Use VALID AT and change-feed replay to audit transaction state transitions."
            .to_string();
    }
    if risk_score >= 0.5 {
        return "Inspect recent writes for this key and compare with AS OF timeline deltas."
            .to_string();
    }
    "No urgent action; monitor trend and aggregate related keys.".to_string()
}

/// Ultra-fast risk score for inline write-path assessment (< 500ns budget).
/// Scans raw bytes for high-risk keyword patterns without allocating or lowercasing.
pub fn quick_risk_score(doc: &[u8]) -> f64 {
    let mut score = 0.0f64;

    if contains_bytes(doc, b"error")
        || contains_bytes(doc, b"Error")
        || contains_bytes(doc, b"ERROR")
    {
        score += 0.30;
    }
    if contains_bytes(doc, b"failed")
        || contains_bytes(doc, b"Failed")
        || contains_bytes(doc, b"FAILED")
    {
        score += 0.25;
    }
    if contains_bytes(doc, b"critical")
        || contains_bytes(doc, b"Critical")
        || contains_bytes(doc, b"CRITICAL")
    {
        score += 0.35;
    }
    if contains_bytes(doc, b"unauthorized") || contains_bytes(doc, b"Unauthorized") {
        score += 0.40;
    }
    if contains_bytes(doc, b"timeout") || contains_bytes(doc, b"Timeout") {
        score += 0.20;
    }
    if contains_bytes(doc, b"panic") || contains_bytes(doc, b"PANIC") {
        score += 0.40;
    }

    score.min(1.0)
}

/// Fast byte-level substring search (no allocation, no case conversion).
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_bridge::build_hasher;

    #[test]
    fn hex_roundtrip_and_insight_id_parse() {
        let key = b"table/events/evt-1";
        let encoded = hex_encode(key);
        assert_eq!(hex_decode(&encoded).unwrap(), key);

        let id = insight_id_for_source(key, 42);
        let (parsed_key, parsed_ts) = parse_insight_id(&id).unwrap();
        assert_eq!(parsed_key, key);
        assert_eq!(parsed_ts, 42);
    }

    #[test]
    fn cluster_id_is_stable_for_same_signature() {
        let hasher = build_hasher();
        let tags = vec!["connectivity".to_string(), "performance".to_string()];
        let a = derive_cluster_id(hasher.as_ref(), &tags, 0.51, 1700000000000);
        let b = derive_cluster_id(hasher.as_ref(), &tags, 0.51, 1700000000000);
        assert_eq!(a, b);
    }

    #[test]
    fn quick_risk_score_zero_for_benign() {
        assert_eq!(quick_risk_score(b"hello world"), 0.0);
        assert_eq!(quick_risk_score(b"{\"status\":\"ok\"}"), 0.0);
    }

    #[test]
    fn quick_risk_score_high_for_critical() {
        let score = quick_risk_score(b"CRITICAL error: system failed");
        assert!(score >= 0.75, "expected >= 0.75, got {score}");
    }

    #[test]
    fn quick_risk_score_capped_at_one() {
        let score = quick_risk_score(b"CRITICAL error failed unauthorized panic timeout");
        assert!(score <= 1.0);
    }

    #[test]
    fn quick_risk_score_moderate() {
        let score = quick_risk_score(b"{\"status\":\"timeout\"}");
        assert!(score > 0.0 && score < 0.75);
    }
}
