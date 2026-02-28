use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;

pub const MANIFEST_FILE: &str = "MANIFEST.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestLevelFile {
    pub file_name: String,
    pub level: usize,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub file_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestShardState {
    pub shard_id: usize,
    pub wal_file: String,
    pub l0_files: Vec<String>,
    #[serde(default)]
    pub level_files: Option<Vec<Vec<ManifestLevelFile>>>,
    pub commit_ts_high_watermark: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestState {
    pub version: u32,
    pub next_file_id: u64,
    pub shards: Vec<ManifestShardState>,
}

impl ManifestState {
    pub fn default_for(shard_count: usize) -> Self {
        let mut shards = Vec::with_capacity(shard_count);
        for shard_id in 0..shard_count {
            shards.push(ManifestShardState {
                shard_id,
                wal_file: format!("shard-{shard_id}.wal"),
                l0_files: Vec::new(),
                level_files: None,
                commit_ts_high_watermark: 0,
            });
        }
        Self {
            version: 1,
            next_file_id: 1,
            shards,
        }
    }
}

pub struct Manifest {
    path: PathBuf,
    pub state: ManifestState,
}

impl Manifest {
    pub fn load_or_create(root: impl AsRef<Path>, shard_count: usize) -> Result<Self> {
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        let path = root.join(MANIFEST_FILE);

        if path.exists() {
            let bytes = fs::read(&path)?;
            let state: ManifestState = serde_json::from_slice(&bytes)?;
            Ok(Self { path, state })
        } else {
            let mut m = Self {
                path,
                state: ManifestState::default_for(shard_count),
            };
            m.save()?;
            Ok(m)
        }
    }

    pub fn save(&mut self) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.state)?;
        let tmp = self.path.with_extension("json.tmp");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_data()?;
        }
        fs::rename(&tmp, &self.path)?;
        if let Some(parent) = self.path.parent() {
            File::open(parent)?.sync_all()?;
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
