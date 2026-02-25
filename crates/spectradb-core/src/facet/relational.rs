use crate::error::{Result, SpectraError};
use serde::{Deserialize, Serialize};

const TABLE_META_PREFIX: &str = "__meta/table";
const VIEW_META_PREFIX: &str = "__meta/view";
const INDEX_META_PREFIX: &str = "__meta/index";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableColumnMetadata {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableSchemaMetadata {
    #[serde(default = "default_pk_name")]
    pub pk: String,
    #[serde(default = "default_doc_type")]
    pub doc: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub columns: Vec<TableColumnMetadata>,
    #[serde(default = "default_schema_mode")]
    pub schema_mode: String,
}

impl Default for TableSchemaMetadata {
    fn default() -> Self {
        Self {
            pk: default_pk_name(),
            doc: default_doc_type(),
            columns: Vec::new(),
            schema_mode: default_schema_mode(),
        }
    }
}

fn default_schema_mode() -> String {
    "legacy".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewMetadata {
    pub name: String,
    pub query: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub name: String,
    pub table: String,
    #[serde(default)]
    pub columns: Vec<String>,
    #[serde(default)]
    pub unique: bool,
}

fn default_pk_name() -> String {
    "pk".to_string()
}

fn default_doc_type() -> String {
    "json".to_string()
}

pub fn table_meta_key(table: &str) -> Vec<u8> {
    format!("{TABLE_META_PREFIX}/{table}").into_bytes()
}

pub fn view_meta_key(view: &str) -> Vec<u8> {
    format!("{VIEW_META_PREFIX}/{view}").into_bytes()
}

pub fn index_meta_key(table: &str, index: &str) -> Vec<u8> {
    format!("{INDEX_META_PREFIX}/{table}/{index}").into_bytes()
}

pub fn encode_schema_metadata(meta: &TableSchemaMetadata) -> Result<Vec<u8>> {
    validate_identifier(&meta.pk)?;
    for column in &meta.columns {
        validate_identifier(&column.name)?;
    }
    Ok(serde_json::to_vec(meta)?)
}

pub fn parse_schema_metadata(bytes: &[u8]) -> Result<TableSchemaMetadata> {
    let mut meta: TableSchemaMetadata = serde_json::from_slice(bytes)?;
    if meta.pk.is_empty() {
        meta.pk = default_pk_name();
    }
    if meta.doc.is_empty() {
        meta.doc = default_doc_type();
    }
    validate_identifier(&meta.pk)?;
    for column in &meta.columns {
        validate_identifier(&column.name)?;
    }
    Ok(meta)
}

pub fn row_key(table: &str, pk: &str) -> Vec<u8> {
    format!("table/{table}/{pk}").into_bytes()
}

pub fn validate_identifier(name: &str) -> Result<()> {
    validate_identifier_kind("identifier", name)
}

pub fn validate_table_name(name: &str) -> Result<()> {
    validate_identifier_kind("table", name)
}

pub fn validate_view_name(name: &str) -> Result<()> {
    validate_identifier_kind("view", name)
}

pub fn validate_index_name(name: &str) -> Result<()> {
    validate_identifier_kind("index", name)
}

fn validate_identifier_kind(kind: &str, name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(SpectraError::SqlExec(format!(
            "{kind} name cannot be empty"
        )));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(SpectraError::SqlExec(format!(
            "{kind} name must be ascii alnum/_/-"
        )));
    }
    Ok(())
}

pub fn validate_json_bytes(doc: &[u8]) -> Result<()> {
    let _: serde_json::Value = serde_json::from_slice(doc)?;
    Ok(())
}

pub fn validate_pk(pk: &str) -> Result<()> {
    if pk.is_empty() {
        return Err(SpectraError::SqlExec("pk cannot be empty".to_string()));
    }
    if pk.contains('\0') {
        return Err(SpectraError::SqlExec(
            "pk must not contain null bytes".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_key_helpers_are_stable() {
        assert_eq!(table_meta_key("users"), b"__meta/table/users".to_vec());
        assert_eq!(
            view_meta_key("active_users"),
            b"__meta/view/active_users".to_vec()
        );
        assert_eq!(
            index_meta_key("users", "users_by_email"),
            b"__meta/index/users/users_by_email".to_vec()
        );
    }

    #[test]
    fn parse_legacy_schema_metadata() {
        let meta = parse_schema_metadata(br#"{"pk":"pk","doc":"json"}"#).unwrap();
        assert_eq!(meta, TableSchemaMetadata::default());
    }

    #[test]
    fn encode_and_parse_schema_metadata_round_trip() {
        let input = TableSchemaMetadata {
            pk: "id".to_string(),
            doc: "json".to_string(),
            columns: vec![
                TableColumnMetadata {
                    name: "id".to_string(),
                    type_name: "text".to_string(),
                },
                TableColumnMetadata {
                    name: "doc".to_string(),
                    type_name: "json".to_string(),
                },
            ],
            schema_mode: "legacy".to_string(),
        };
        let bytes = encode_schema_metadata(&input).unwrap();
        let output = parse_schema_metadata(&bytes).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn identifier_validation_rejects_invalid_names() {
        assert!(validate_identifier("user_1").is_ok());
        assert!(validate_identifier("bad/name").is_err());
        let err = validate_table_name("").unwrap_err();
        assert_eq!(
            err.to_string(),
            "sql execution error: table name cannot be empty"
        );
    }
}
