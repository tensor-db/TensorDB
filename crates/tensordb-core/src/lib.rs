pub mod ai;
pub mod auth;
pub mod cdc;
pub mod cluster;
pub mod config;
pub mod engine;
pub mod error;
pub mod facet;
pub mod io;
pub mod ledger;
pub mod native_bridge;
pub mod sql;
pub mod storage;
pub mod util;

pub use crate::ai::access_stats::AccessStats;
pub use crate::ai::cache_advisor::{CacheAdvisor, CountMinSketch};
pub use crate::ai::compaction_advisor::{CompactionAdvisor, CompactionDecision};
pub use crate::ai::query_advisor::{AccessPathHint, QueryAdvisor};
pub use crate::ai::{AiCorrelationRef, AiInsight, AiInsightProvenance, AiRuntimeStats};
pub use crate::config::Config;
pub use crate::engine::db::{
    BenchOptions, BenchReport, Database, DbStats, ExplainRow, PrefixScanRow,
};
pub use crate::engine::shard::{ChangeEvent, GetResult, ShardReadHandle, WriteBatchItem};
pub use crate::error::{Result, TensorError};
pub use crate::sql::exec::{ColumnStatistics, PreparedStatement, TableStats};
pub use crate::sql::planner::{explain_plan, generate_plan, PlanNode};
pub use crate::sql::vectorized::{
    vectorized_filter, vectorized_hash_aggregate, vectorized_hash_join, vectorized_limit,
    vectorized_project, vectorized_sort, AggFn, AggSpec, BatchSchema, ColumnDef, ColumnType,
    ColumnVector, CompareOp, RecordBatch,
};
