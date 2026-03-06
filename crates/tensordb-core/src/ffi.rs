//! C FFI interface for embedding TensorDB in non-Rust applications.
//!
//! Provides `extern "C"` functions for opening a database, executing SQL,
//! and freeing results. Use with `cbindgen` to generate a C header.
//!
//! # Usage (C)
//!
//! ```c
//! TensorDB *db = tensordb_open("/tmp/mydb");
//! char *result = tensordb_sql(db, "SELECT 1 + 1 AS answer");
//! // result is a JSON string
//! tensordb_free_string(result);
//! tensordb_close(db);
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::config::Config;
use crate::engine::db::Database;

/// Opaque handle to a TensorDB database.
pub struct TensorDBHandle {
    db: Database,
}

/// Open a TensorDB database at the given path.
/// Returns a handle pointer, or null on error.
///
/// # Safety
/// `path` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn tensordb_open(path: *const c_char) -> *mut TensorDBHandle {
    if path.is_null() {
        return ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match Database::open(path_str, Config::default()) {
        Ok(db) => Box::into_raw(Box::new(TensorDBHandle { db })),
        Err(_) => ptr::null_mut(),
    }
}

/// Execute a SQL query and return the result as a JSON string.
/// Returns null on error. The caller must free the result with `tensordb_free_string`.
///
/// # Safety
/// `handle` must be a valid pointer from `tensordb_open`.
/// `query` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn tensordb_sql(
    handle: *mut TensorDBHandle,
    query: *const c_char,
) -> *mut c_char {
    if handle.is_null() || query.is_null() {
        return ptr::null_mut();
    }
    let handle = unsafe { &*handle };
    let c_str = unsafe { CStr::from_ptr(query) };
    let query_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let result = match handle.db.sql(query_str) {
        Ok(result) => {
            let json = match result {
                crate::sql::exec::SqlResult::Rows(rows) => {
                    let parsed: Vec<serde_json::Value> = rows
                        .iter()
                        .filter_map(|r| serde_json::from_slice(r).ok())
                        .collect();
                    serde_json::json!({ "rows": parsed }).to_string()
                }
                crate::sql::exec::SqlResult::Affected { rows, message, .. } => serde_json::json!({
                    "affected_rows": rows,
                    "message": message
                })
                .to_string(),
                crate::sql::exec::SqlResult::Explain(text) => {
                    serde_json::json!({ "explain": text }).to_string()
                }
            };
            json
        }
        Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
    };

    match CString::new(result) {
        Ok(c) => c.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by `tensordb_sql`.
///
/// # Safety
/// `s` must be a pointer previously returned by `tensordb_sql`, or null.
#[no_mangle]
pub unsafe extern "C" fn tensordb_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Close a TensorDB database and free the handle.
///
/// # Safety
/// `handle` must be a pointer previously returned by `tensordb_open`, or null.
#[no_mangle]
pub unsafe extern "C" fn tensordb_close(handle: *mut TensorDBHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Get the last error message, if any.
/// Returns null if no error. The caller must free the result with `tensordb_free_string`.
///
/// # Safety
/// `handle` must be a valid pointer from `tensordb_open`.
#[no_mangle]
pub unsafe extern "C" fn tensordb_version() -> *mut c_char {
    let version = env!("CARGO_PKG_VERSION");
    match CString::new(version) {
        Ok(c) => c.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn ffi_open_close() {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        unsafe {
            let handle = tensordb_open(path.as_ptr());
            assert!(!handle.is_null());
            tensordb_close(handle);
        }
    }

    #[test]
    fn ffi_sql_query() {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let query = CString::new("SELECT 1 + 1 AS answer").unwrap();
        unsafe {
            let handle = tensordb_open(path.as_ptr());
            assert!(!handle.is_null());
            let result = tensordb_sql(handle, query.as_ptr());
            assert!(!result.is_null());
            let result_str = CStr::from_ptr(result).to_str().unwrap();
            assert!(result_str.contains("rows"));
            tensordb_free_string(result);
            tensordb_close(handle);
        }
    }

    #[test]
    fn ffi_null_safety() {
        unsafe {
            assert!(tensordb_open(ptr::null()).is_null());
            assert!(tensordb_sql(ptr::null_mut(), ptr::null()).is_null());
            tensordb_free_string(ptr::null_mut()); // Should not crash
            tensordb_close(ptr::null_mut()); // Should not crash
        }
    }

    #[test]
    fn ffi_version() {
        unsafe {
            let ver = tensordb_version();
            assert!(!ver.is_null());
            let ver_str = CStr::from_ptr(ver).to_str().unwrap();
            assert!(!ver_str.is_empty());
            tensordb_free_string(ver);
        }
    }
}
