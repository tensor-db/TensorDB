//! io_uring-based async I/O for WAL writes and SSTable reads.
//!
//! This module is only available on Linux with the `io-uring` feature flag.
//! It provides:
//! - `UringWalWriter`: batched WAL writes with IORING_OP_WRITE + IORING_OP_FSYNC
//! - `UringBlockReader`: async block reads from SSTables
//!
//! When this feature is not enabled, SpectraDB uses standard synchronous I/O.

use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use io_uring::IoUring;

use crate::error::{Result, SpectraError};

/// io_uring-based WAL writer.
pub struct UringWalWriter {
    ring: IoUring,
    file: File,
}

impl UringWalWriter {
    pub fn open(path: impl AsRef<Path>, queue_depth: u32) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let ring =
            IoUring::new(queue_depth).map_err(|e| SpectraError::Io(std::io::Error::from(e)))?;
        Ok(Self { ring, file })
    }

    /// Submit a write + fsync pair via io_uring.
    pub fn write_and_sync(&mut self, data: &[u8]) -> Result<()> {
        let fd = io_uring::types::Fd(self.file.as_raw_fd());

        // Submit write
        let write_e = io_uring::opcode::Write::new(fd, data.as_ptr(), data.len() as u32)
            .build()
            .user_data(1);

        unsafe {
            self.ring
                .submission()
                .push(&write_e)
                .map_err(|_| SpectraError::Io(std::io::Error::other("SQ full")))?;
        }

        // Submit fsync
        let fsync_e = io_uring::opcode::Fsync::new(fd).build().user_data(2);

        unsafe {
            self.ring
                .submission()
                .push(&fsync_e)
                .map_err(|_| SpectraError::Io(std::io::Error::other("SQ full")))?;
        }

        // Wait for both completions
        self.ring
            .submit_and_wait(2)
            .map_err(|e| SpectraError::Io(std::io::Error::from(e)))?;

        let mut cq = self.ring.completion();
        for _ in 0..2 {
            if let Some(cqe) = cq.next() {
                if cqe.result() < 0 {
                    return Err(SpectraError::Io(std::io::Error::from_raw_os_error(
                        -cqe.result(),
                    )));
                }
            }
        }

        Ok(())
    }
}

/// io_uring-based block reader for SSTables.
pub struct UringBlockReader {
    ring: IoUring,
}

impl UringBlockReader {
    pub fn new(queue_depth: u32) -> Result<Self> {
        let ring =
            IoUring::new(queue_depth).map_err(|e| SpectraError::Io(std::io::Error::from(e)))?;
        Ok(Self { ring })
    }

    /// Read a block from an SSTable file at the given offset.
    pub fn read_block(&mut self, file: &File, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        let fd = io_uring::types::Fd(file.as_raw_fd());

        let read_e =
            io_uring::opcode::Read::new(fd, buf.as_mut_ptr(), len as u32)
                .offset(offset)
                .build()
                .user_data(1);

        unsafe {
            self.ring
                .submission()
                .push(&read_e)
                .map_err(|_| SpectraError::Io(std::io::Error::other("SQ full")))?;
        }

        self.ring
            .submit_and_wait(1)
            .map_err(|e| SpectraError::Io(std::io::Error::from(e)))?;

        if let Some(cqe) = self.ring.completion().next() {
            if cqe.result() < 0 {
                return Err(SpectraError::Io(std::io::Error::from_raw_os_error(
                    -cqe.result(),
                )));
            }
            let bytes_read = cqe.result() as usize;
            buf.truncate(bytes_read);
        }

        Ok(buf)
    }
}
