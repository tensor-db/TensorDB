//! TensorDB Distributed — horizontal scaling layer for TensorDB.
//!
//! Provides distributed routing, 2PC transactions, online shard
//! rebalancing, and cluster lifecycle management.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              Client (SQL / API)                  │
//! ├─────────────────────────────────────────────────┤
//! │  DistributedDatabase                             │
//! │   ├─ ShardRouter (ConsistentHashRing)           │
//! │   ├─ TxnCoordinator (2PC)                       │
//! │   └─ ClusterNode                                │
//! │       ├─ GossipDiscovery                         │
//! │       ├─ HealthChecker (φ-accrual)              │
//! │       └─ ShardMigrator                          │
//! ├─────────────────────────────────────────────────┤
//! │  tensordb-core (local Database)                  │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! The proto file at `proto/tensor_cluster.proto` defines the gRPC service
//! interface for network transport (requires `protoc` for codegen).

pub mod config;
pub mod error;
pub mod node;

pub mod discovery;
pub mod rebalance;
pub mod routing;
pub mod txn;
