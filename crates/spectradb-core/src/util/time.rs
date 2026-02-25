use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn unix_millis() -> u64 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0));
    d.as_millis() as u64
}
