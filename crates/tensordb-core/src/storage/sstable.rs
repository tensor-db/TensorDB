use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{Result, TensorError};
use crate::ledger::internal_key::decode_internal_key;
use crate::ledger::record::FactValue;
use crate::native_bridge::Hasher;
use crate::storage::bloom::BloomFilter;
use crate::util::varint::{decode_u64, encode_u64};

pub const SST_MAGIC: u32 = 0x53535442; // SSTB
pub const SST_VERSION_V1: u32 = 1;
pub const SST_VERSION_V2: u32 = 2; // V2: LZ4-compressed blocks
pub const SST_VERSION: u32 = SST_VERSION_V2;
pub const FOOTER_MAGIC: u32 = 0x53534654; // SSFT

#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub last_key: Vec<u8>,
    pub offset: u64,
    pub len: u32,
}

#[derive(Debug, Clone)]
pub struct SsTableLookup {
    pub value: Option<Vec<u8>>,
    pub bloom_hit: bool,
    pub block_read: Option<usize>,
    pub commit_ts: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SsTableBuildOutput {
    pub path: PathBuf,
    pub entry_count: usize,
}

pub fn build_sstable(
    path: impl AsRef<Path>,
    entries: &[(Vec<u8>, Vec<u8>)],
    block_size: usize,
    bloom_bits_per_key: usize,
    hasher: &dyn Hasher,
) -> Result<SsTableBuildOutput> {
    let path = path.as_ref();
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(&SST_MAGIC.to_le_bytes())?;
    writer.write_all(&SST_VERSION.to_le_bytes())?;
    writer.write_all(&(block_size as u32).to_le_bytes())?;

    let mut index = Vec::new();
    let mut block = Vec::with_capacity(block_size.max(1024));
    let mut block_last_key: Option<Vec<u8>> = None;
    let mut block_offset = 12u64;

    let mut bloom_keys = Vec::with_capacity(entries.len());

    for (k, v) in entries {
        if let Ok(decoded) = decode_internal_key(k) {
            bloom_keys.push(decoded.user_key);
        }
        let mut entry_buf = Vec::with_capacity(k.len() + v.len() + 16);
        encode_u64(k.len() as u64, &mut entry_buf);
        encode_u64(v.len() as u64, &mut entry_buf);
        entry_buf.extend_from_slice(k);
        entry_buf.extend_from_slice(v);

        if !block.is_empty() && block.len() + entry_buf.len() > block_size {
            let compressed = lz4_flex::compress_prepend_size(&block);
            let on_disk_len = compressed.len() as u32;
            writer.write_all(&on_disk_len.to_le_bytes())?;
            writer.write_all(&compressed)?;
            let last_key = block_last_key.take().ok_or_else(|| {
                TensorError::SstableFormat("block missing last key during flush".to_string())
            })?;
            index.push(IndexEntry {
                last_key,
                offset: block_offset,
                len: on_disk_len,
            });
            block_offset += 4 + on_disk_len as u64;
            block.clear();
        }

        block.extend_from_slice(&entry_buf);
        block_last_key = Some(k.clone());
    }

    if !block.is_empty() {
        let compressed = lz4_flex::compress_prepend_size(&block);
        let on_disk_len = compressed.len() as u32;
        writer.write_all(&on_disk_len.to_le_bytes())?;
        writer.write_all(&compressed)?;
        let last_key = block_last_key.take().ok_or_else(|| {
            TensorError::SstableFormat("block missing last key at finalize".to_string())
        })?;
        index.push(IndexEntry {
            last_key,
            offset: block_offset,
            len: on_disk_len,
        });
    }

    writer.flush()?;

    let mut f = writer.into_inner().map_err(|e| e.into_error())?;
    let index_offset = f.stream_position()?;
    write_index_block(&mut f, &index)?;
    let bloom_offset = f.stream_position()?;

    let bloom = BloomFilter::new_for_keys(&bloom_keys, bloom_bits_per_key, hasher);
    let bloom_bytes = bloom.encode();
    f.write_all(&bloom_bytes)?;

    f.write_all(&index_offset.to_le_bytes())?;
    f.write_all(&bloom_offset.to_le_bytes())?;
    f.write_all(&FOOTER_MAGIC.to_le_bytes())?;
    f.sync_data()?;

    Ok(SsTableBuildOutput {
        path: path.to_path_buf(),
        entry_count: entries.len(),
    })
}

fn write_index_block(mut out: impl Write, index: &[IndexEntry]) -> Result<()> {
    out.write_all(&(index.len() as u32).to_le_bytes())?;
    for entry in index {
        let mut tmp = Vec::new();
        encode_u64(entry.last_key.len() as u64, &mut tmp);
        out.write_all(&tmp)?;
        out.write_all(&entry.last_key)?;
        out.write_all(&entry.offset.to_le_bytes())?;
        out.write_all(&entry.len.to_le_bytes())?;
    }
    Ok(())
}

fn parse_index_block(bytes: &[u8]) -> Result<Vec<IndexEntry>> {
    if bytes.len() < 4 {
        return Err(TensorError::SstableFormat(
            "index block too short".to_string(),
        ));
    }
    let mut countb = [0u8; 4];
    countb.copy_from_slice(&bytes[0..4]);
    let count = u32::from_le_bytes(countb) as usize;
    let mut idx = 4usize;
    let mut out = Vec::with_capacity(count);

    for _ in 0..count {
        let klen = decode_u64(bytes, &mut idx)? as usize;
        if idx + klen + 12 > bytes.len() {
            return Err(TensorError::SstableFormat(
                "index entry out of range".to_string(),
            ));
        }
        let key = bytes[idx..idx + klen].to_vec();
        idx += klen;
        let mut o = [0u8; 8];
        o.copy_from_slice(&bytes[idx..idx + 8]);
        idx += 8;
        let mut l = [0u8; 4];
        l.copy_from_slice(&bytes[idx..idx + 4]);
        idx += 4;

        out.push(IndexEntry {
            last_key: key,
            offset: u64::from_le_bytes(o),
            len: u32::from_le_bytes(l),
        });
    }
    Ok(out)
}

pub struct SsTableReader {
    pub path: PathBuf,
    mmap: Mmap,
    pub block_size: u32,
    version: u32,
    pub index: Vec<IndexEntry>,
    pub bloom: BloomFilter,
}

impl SsTableReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 12 + 20 {
            return Err(TensorError::SstableFormat("sstable too short".to_string()));
        }
        let mut b = [0u8; 4];
        b.copy_from_slice(&mmap[0..4]);
        let magic = u32::from_le_bytes(b);
        if magic != SST_MAGIC {
            return Err(TensorError::SstableFormat("sstable bad magic".to_string()));
        }
        b.copy_from_slice(&mmap[4..8]);
        let version = u32::from_le_bytes(b);
        if version != SST_VERSION_V1 && version != SST_VERSION_V2 {
            return Err(TensorError::SstableFormat(format!(
                "unsupported SSTable version: {version}"
            )));
        }
        b.copy_from_slice(&mmap[8..12]);
        let block_size = u32::from_le_bytes(b);

        let footer_start = mmap.len() - 20;
        let mut u8b = [0u8; 8];
        u8b.copy_from_slice(&mmap[footer_start..footer_start + 8]);
        let index_offset = u64::from_le_bytes(u8b) as usize;
        u8b.copy_from_slice(&mmap[footer_start + 8..footer_start + 16]);
        let bloom_offset = u64::from_le_bytes(u8b) as usize;
        b.copy_from_slice(&mmap[footer_start + 16..footer_start + 20]);
        let footer_magic = u32::from_le_bytes(b);
        if footer_magic != FOOTER_MAGIC {
            return Err(TensorError::SstableFormat(
                "footer magic mismatch".to_string(),
            ));
        }

        if index_offset >= bloom_offset || bloom_offset > footer_start {
            return Err(TensorError::SstableFormat(
                "footer offsets out of range".to_string(),
            ));
        }

        let index = parse_index_block(&mmap[index_offset..bloom_offset])?;
        let bloom = BloomFilter::decode(&mmap[bloom_offset..footer_start]).ok_or_else(|| {
            TensorError::SstableFormat("invalid bloom block encoding".to_string())
        })?;

        Ok(Self {
            path,
            mmap,
            block_size,
            version,
            index,
            bloom,
        })
    }

    pub fn path_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher as StdHasher};
        let mut h = DefaultHasher::new();
        self.path.hash(&mut h);
        h.finish()
    }

    pub fn get_visible(
        &self,
        user_key: &[u8],
        as_of: u64,
        valid_at: Option<u64>,
        hasher: &dyn Hasher,
        block_cache: Option<&crate::storage::cache::BlockCache>,
        index_cache: Option<&crate::storage::cache::IndexCache>,
    ) -> Result<SsTableLookup> {
        if !self.bloom.may_contain(user_key, hasher) {
            return Ok(SsTableLookup {
                value: None,
                bloom_hit: false,
                block_read: None,
                commit_ts: None,
            });
        }

        let mut best: Option<(u64, Vec<u8>)> = None;
        let mut block_read = None;

        let mut target = user_key.to_vec();
        target.push(0);

        let mut start_idx = 0usize;
        if !self.index.is_empty() {
            start_idx = match self
                .index
                .binary_search_by(|probe| probe.last_key.as_slice().cmp(target.as_slice()))
            {
                Ok(i) | Err(i) => i,
            };
        }

        let _ = index_cache; // reserved for future index caching

        for (i, entry) in self.index.iter().enumerate().skip(start_idx) {
            let block_start = entry.offset as usize;
            if block_start + 4 + entry.len as usize > self.mmap.len() {
                break;
            }
            block_read.get_or_insert(i);

            let block_len = entry.len as usize;
            let block_data: Arc<Vec<u8>> = if let Some(cache) = block_cache {
                let ph = self.path_hash();
                if let Some(cached) = cache.get(ph, entry.offset) {
                    cached
                } else {
                    let data = self.read_block_data(block_start, block_len)?;
                    cache.insert(ph, entry.offset, data.clone());
                    Arc::new(data)
                }
            } else {
                Arc::new(self.read_block_data(block_start, block_len)?)
            };
            let block = block_data.as_slice();
            let mut idx = 0usize;
            while idx < block.len() {
                let klen = decode_u64(block, &mut idx)? as usize;
                let vlen = decode_u64(block, &mut idx)? as usize;
                if idx + klen + vlen > block.len() {
                    return Err(TensorError::SstableFormat(
                        "entry out of bounds".to_string(),
                    ));
                }
                let key = &block[idx..idx + klen];
                idx += klen;
                let value = &block[idx..idx + vlen];
                idx += vlen;

                let decoded_key = decode_internal_key(key)?;
                match decoded_key.user_key.as_slice().cmp(user_key) {
                    std::cmp::Ordering::Less => continue,
                    std::cmp::Ordering::Greater => {
                        break;
                    }
                    std::cmp::Ordering::Equal => {
                        if decoded_key.commit_ts > as_of {
                            continue;
                        }
                        let fact = FactValue::decode(value)?;
                        if let Some(valid_ts) = valid_at {
                            if !(fact.valid_from <= valid_ts && valid_ts < fact.valid_to) {
                                continue;
                            }
                        }
                        match &best {
                            Some((ts, _)) if *ts >= decoded_key.commit_ts => {}
                            _ => best = Some((decoded_key.commit_ts, fact.doc)),
                        }
                    }
                }
            }
            if entry.last_key.as_slice() > user_key && best.is_some() {
                break;
            }
        }

        let (value, commit_ts) = match best {
            Some((ts, v)) => (Some(v), Some(ts)),
            None => (None, None),
        };

        Ok(SsTableLookup {
            value,
            bloom_hit: true,
            block_read,
            commit_ts,
        })
    }

    fn read_block_data(&self, block_start: usize, block_len: usize) -> Result<Vec<u8>> {
        let raw = &self.mmap[block_start + 4..block_start + 4 + block_len];
        if self.version == SST_VERSION_V2 {
            lz4_flex::decompress_size_prepended(raw)
                .map_err(|e| TensorError::SstableFormat(format!("LZ4 decompress error: {e}")))
        } else {
            Ok(raw.to_vec())
        }
    }

    pub fn iter_all_entries(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut out = Vec::new();
        for entry in &self.index {
            let block_start = entry.offset as usize;
            let block_len = entry.len as usize;
            if block_start + 4 + block_len > self.mmap.len() {
                return Err(TensorError::SstableFormat(
                    "index offset out of range".to_string(),
                ));
            }
            let block_data = self.read_block_data(block_start, block_len)?;
            let block = block_data.as_slice();
            let mut idx = 0usize;
            while idx < block.len() {
                let klen = decode_u64(block, &mut idx)? as usize;
                let vlen = decode_u64(block, &mut idx)? as usize;
                if idx + klen + vlen > block.len() {
                    return Err(TensorError::SstableFormat(
                        "entry out of bounds".to_string(),
                    ));
                }
                let key = block[idx..idx + klen].to_vec();
                idx += klen;
                let val = block[idx..idx + vlen].to_vec();
                idx += vlen;
                out.push((key, val));
            }
        }
        Ok(out)
    }
}

pub fn read_footer(path: impl AsRef<Path>) -> Result<(u64, u64)> {
    let mut f = File::open(path)?;
    let len = f.seek(SeekFrom::End(0))?;
    if len < 20 {
        return Err(TensorError::SstableFormat(
            "file too short for footer".to_string(),
        ));
    }
    f.seek(SeekFrom::End(-20))?;
    let mut buf = [0u8; 20];
    f.read_exact(&mut buf)?;
    let mut o = [0u8; 8];
    o.copy_from_slice(&buf[0..8]);
    let index = u64::from_le_bytes(o);
    o.copy_from_slice(&buf[8..16]);
    let bloom = u64::from_le_bytes(o);
    Ok((index, bloom))
}
