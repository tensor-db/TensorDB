use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crc32fast::Hasher;

use crate::error::{Result, SpectraError};
use crate::ledger::record::FactWrite;

pub const WAL_MAGIC: u32 = 0x5357414c; // SWAL
const MAX_WAL_RECORD_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

pub struct Wal {
    path: PathBuf,
    file: File,
    fsync_every_n_records: usize,
    pending_since_sync: usize,
}

impl Wal {
    pub fn open(path: impl AsRef<Path>, fsync_every_n_records: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            fsync_every_n_records,
            pending_since_sync: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&mut self, write: &FactWrite) -> Result<()> {
        let payload = write.encode_payload();
        let len = payload.len() as u32;
        let mut crc_hasher = Hasher::new();
        crc_hasher.update(&payload);
        let crc = crc_hasher.finalize();

        self.file.write_all(&WAL_MAGIC.to_le_bytes())?;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.pending_since_sync += 1;

        if self.pending_since_sync >= self.fsync_every_n_records {
            self.sync()?;
        }
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.sync_data()?;
        self.pending_since_sync = 0;
        Ok(())
    }

    pub fn truncate(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.sync_data()?;
        self.pending_since_sync = 0;
        Ok(())
    }

    pub fn replay(path: impl AsRef<Path>) -> Result<Vec<FactWrite>> {
        let mut file = OpenOptions::new().read(true).open(path)?;
        file.seek(SeekFrom::Start(0))?;
        let mut out = Vec::new();

        loop {
            let mut hdr = [0u8; 12];
            match file.read_exact(&mut hdr) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(SpectraError::Io(e)),
            }

            let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
            if magic != WAL_MAGIC {
                return Err(SpectraError::WalMagicMismatch);
            }
            let len = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
            if len > MAX_WAL_RECORD_BYTES {
                // Corrupt or unreasonable length; stop replay.
                break;
            }
            let crc = u32::from_le_bytes(hdr[8..12].try_into().unwrap());

            let mut payload = vec![0u8; len];
            match file.read_exact(&mut payload) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Torn tail; deterministic stop.
                    break;
                }
                Err(e) => return Err(SpectraError::Io(e)),
            }

            let mut crc_hasher = Hasher::new();
            crc_hasher.update(&payload);
            if crc_hasher.finalize() != crc {
                // Treat CRC mismatch same as torn tail: stop replay at last good record.
                break;
            }

            out.push(FactWrite::decode_payload(&payload)?);
        }

        Ok(out)
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        let _ = self.sync();
    }
}
