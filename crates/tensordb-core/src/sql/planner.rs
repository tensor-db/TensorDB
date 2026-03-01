use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::sql::parser::{
    extract_pk_eq_literal, is_aggregate_function, select_items_contain_aggregate, BinOperator,
    Expr, JoinSpec, JoinType, SelectItem, Statement, TableRef,
};

/// A logical query plan node produced by the planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanNode {
    /// Single-key point read: O(log n) via memtable + bloom filter
    PointLookup {
        table: String,
        pk: String,
        estimated_cost: f64,
    },
    /// Prefix-bounded scan using BTreeMap::range: O(k log n)
    PrefixScan {
        table: String,
        prefix: Option<String>,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Secondary index scan: O(log n + k) where k = matching rows
    IndexScan {
        table: String,
        index_name: String,
        columns: Vec<String>,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Full table scan: O(n)
    FullScan {
        table: String,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Filter applied after a child scan
    Filter {
        child: Box<PlanNode>,
        predicate_display: String,
        estimated_selectivity: f64,
    },
    /// Hash join on pk equality
    HashJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_type: String,
        estimated_cost: f64,
    },
    /// Nested loop join
    NestedLoopJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_type: String,
        estimated_cost: f64,
    },
    /// Index nested loop join (INL): probe right table's index for each left row
    IndexNestedLoopJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        index_name: String,
        join_type: String,
        estimated_cost: f64,
    },
    /// Aggregation
    Aggregate {
        child: Box<PlanNode>,
        group_by_count: usize,
        aggregate_count: usize,
    },
    /// Sort
    Sort {
        child: Box<PlanNode>,
        order_by_count: usize,
        estimated_cost: f64,
    },
    /// Limit
    Limit { child: Box<PlanNode>, count: u64 },
    /// Project columns
    Project {
        child: Box<PlanNode>,
        column_count: usize,
    },
    /// Vector similarity search using HNSW or IVF-PQ index
    VectorSearch {
        table: String,
        column: String,
        k: u64,
        metric: String,
        estimated_cost: f64,
    },
}

impl PlanNode {
    /// Estimated cost in arbitrary units (lower = better).
    pub fn cost(&self) -> f64 {
        match self {
            PlanNode::PointLookup { estimated_cost, .. } => *estimated_cost,
            PlanNode::PrefixScan { estimated_cost, .. } => *estimated_cost,
            PlanNode::IndexScan { estimated_cost, .. } => *estimated_cost,
            PlanNode::FullScan { estimated_cost, .. } => *estimated_cost,
            PlanNode::Filter { child, .. } => child.cost(),
            PlanNode::HashJoin { estimated_cost, .. } => *estimated_cost,
            PlanNode::NestedLoopJoin { estimated_cost, .. } => *estimated_cost,
            PlanNode::IndexNestedLoopJoin { estimated_cost, .. } => *estimated_cost,
            PlanNode::Aggregate { child, .. } => child.cost() * 1.1,
            PlanNode::Sort { estimated_cost, .. } => *estimated_cost,
            PlanNode::Limit { child, .. } => child.cost(),
            PlanNode::Project { child, .. } => child.cost(),
            PlanNode::VectorSearch { estimated_cost, .. } => *estimated_cost,
        }
    }

    /// Pretty-print the plan as an indented tree.
    pub fn display(&self, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match self {
            PlanNode::PointLookup {
                table,
                pk,
                estimated_cost,
            } => format!("{pad}PointLookup table={table} pk={pk} cost={estimated_cost:.1}"),
            PlanNode::PrefixScan {
                table,
                prefix,
                estimated_rows,
                estimated_cost,
            } => {
                let pfx = prefix.as_deref().unwrap_or("*");
                format!("{pad}PrefixScan table={table} prefix={pfx} est_rows={estimated_rows} cost={estimated_cost:.1}")
            }
            PlanNode::IndexScan {
                table,
                index_name,
                columns,
                estimated_rows,
                estimated_cost,
            } => {
                let cols = columns.join(",");
                format!("{pad}IndexScan table={table} index={index_name} cols=[{cols}] est_rows={estimated_rows} cost={estimated_cost:.1}")
            }
            PlanNode::FullScan {
                table,
                estimated_rows,
                estimated_cost,
            } => format!(
                "{pad}FullScan table={table} est_rows={estimated_rows} cost={estimated_cost:.1}"
            ),
            PlanNode::Filter {
                child,
                predicate_display,
                estimated_selectivity,
            } => format!(
                "{pad}Filter sel={estimated_selectivity:.2} pred=[{predicate_display}]\n{}",
                child.display(indent + 1)
            ),
            PlanNode::HashJoin {
                left,
                right,
                join_type,
                estimated_cost,
            } => format!(
                "{pad}HashJoin type={join_type} cost={estimated_cost:.1}\n{}\n{}",
                left.display(indent + 1),
                right.display(indent + 1)
            ),
            PlanNode::NestedLoopJoin {
                left,
                right,
                join_type,
                estimated_cost,
            } => format!(
                "{pad}NestedLoopJoin type={join_type} cost={estimated_cost:.1}\n{}\n{}",
                left.display(indent + 1),
                right.display(indent + 1)
            ),
            PlanNode::IndexNestedLoopJoin {
                left,
                right,
                index_name,
                join_type,
                estimated_cost,
            } => format!(
                "{pad}IndexNestedLoopJoin type={join_type} index={index_name} cost={estimated_cost:.1}\n{}\n{}",
                left.display(indent + 1),
                right.display(indent + 1)
            ),
            PlanNode::Aggregate {
                child,
                group_by_count,
                aggregate_count,
            } => format!(
                "{pad}Aggregate groups={group_by_count} aggs={aggregate_count}\n{}",
                child.display(indent + 1)
            ),
            PlanNode::Sort {
                child,
                order_by_count,
                estimated_cost,
            } => format!(
                "{pad}Sort keys={order_by_count} cost={estimated_cost:.1}\n{}",
                child.display(indent + 1)
            ),
            PlanNode::Limit { child, count } => {
                format!("{pad}Limit n={count}\n{}", child.display(indent + 1))
            }
            PlanNode::Project {
                child,
                column_count,
            } => format!(
                "{pad}Project cols={column_count}\n{}",
                child.display(indent + 1)
            ),
            PlanNode::VectorSearch {
                table,
                column,
                k,
                metric,
                estimated_cost,
            } => format!(
                "{pad}VectorSearch table={table} column={column} k={k} metric={metric} cost={estimated_cost:.1}"
            ),
        }
    }
}

/// Persisted per-column statistics collected by `ANALYZE`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub name: String,
    pub distinct_count: u64,
    pub null_count: u64,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    /// Top-N most frequent values (value, count).
    pub top_values: Vec<(String, u64)>,
}

/// Enhanced table statistics for cost-based planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTableStats {
    pub row_count: u64,
    pub approx_byte_size: u64,
    pub avg_row_bytes: u64,
    pub last_updated_ms: u64,
    pub column_stats: Vec<ColumnStats>,
}

/// Estimate selectivity of a WHERE clause predicate.
fn estimate_selectivity(expr: &Expr, row_count: u64) -> f64 {
    if row_count == 0 {
        return 1.0;
    }
    match expr {
        // pk = 'literal' → single row
        Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        } => {
            if matches!(left.as_ref(), Expr::Column(c) if c.eq_ignore_ascii_case("pk"))
                && matches!(right.as_ref(), Expr::StringLit(_) | Expr::NumberLit(_))
            {
                return 1.0 / row_count as f64;
            }
            // column = literal: assume 10% selectivity as heuristic
            0.1
        }
        Expr::BinOp {
            op: BinOperator::Lt | BinOperator::Gt | BinOperator::LtEq | BinOperator::GtEq,
            ..
        } => 0.33,
        Expr::BinOp {
            op: BinOperator::And,
            left,
            right,
        } => estimate_selectivity(left, row_count) * estimate_selectivity(right, row_count),
        Expr::BinOp {
            op: BinOperator::Or,
            left,
            right,
        } => {
            let s1 = estimate_selectivity(left, row_count);
            let s2 = estimate_selectivity(right, row_count);
            (s1 + s2 - s1 * s2).min(1.0)
        }
        Expr::BinOp {
            op: BinOperator::Like | BinOperator::ILike,
            ..
        } => 0.25,
        Expr::Not(inner) => 1.0 - estimate_selectivity(inner, row_count),
        Expr::IsNull { negated, .. } => {
            if *negated {
                0.95
            } else {
                0.05
            }
        }
        Expr::Between { negated, .. } => {
            if *negated {
                0.75
            } else {
                0.25
            }
        }
        Expr::InList { list, negated, .. } => {
            let sel = (list.len() as f64 * 0.1).min(1.0);
            if *negated {
                1.0 - sel
            } else {
                sel
            }
        }
        _ => 0.5,
    }
}

/// Cost constants (in arbitrary units, calibrated to TensorDB operations).
const POINT_LOOKUP_COST: f64 = 1.0;
const _INDEX_SCAN_COST: f64 = 2.0; // Base cost for index lookup
const _INDEX_SCAN_PER_ROW: f64 = 0.3; // Per matching row
const _PREFIX_SCAN_PER_ROW: f64 = 0.5;
const FULL_SCAN_PER_ROW: f64 = 1.0;
const SORT_PER_ROW_LOG: f64 = 2.0; // n * log2(n) * this factor
const HASH_JOIN_BUILD: f64 = 1.5; // per row to build hash table
const HASH_JOIN_PROBE: f64 = 1.0; // per row to probe
const NL_JOIN_PER_PAIR: f64 = 0.5; // per row-pair in nested loop

/// Default row count estimate when no stats are available.
const DEFAULT_ROW_ESTIMATE: u64 = 1000;

/// Extract (left_column, right_column) from a join ON clause for planning.
fn extract_join_right_column(
    on_clause: &Option<Expr>,
    left_table: &str,
    right_table: &str,
) -> Option<(String, String)> {
    match on_clause {
        Some(Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        }) => {
            let resolve = |e: &Expr| -> Option<(String, String)> {
                match e {
                    Expr::Column(c) => Some((String::new(), c.clone())),
                    Expr::FieldAccess { column, path } if path.len() == 1 => {
                        Some((column.clone(), path[0].clone()))
                    }
                    _ => None,
                }
            };
            let (ltbl, lcol) = resolve(left)?;
            let (rtbl, rcol) = resolve(right)?;

            if (ltbl.is_empty() || ltbl.eq_ignore_ascii_case(left_table))
                && (rtbl.is_empty() || rtbl.eq_ignore_ascii_case(right_table))
            {
                Some((lcol, rcol))
            } else if (ltbl.is_empty() || ltbl.eq_ignore_ascii_case(right_table))
                && (rtbl.is_empty() || rtbl.eq_ignore_ascii_case(left_table))
            {
                Some((rcol, lcol))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// INL join cost constant: O(n * log(m)) per left row
const INL_JOIN_PER_ROW: f64 = 3.0;

/// Table statistics passed to the planner for cost-based optimization.
pub struct PlannerStats {
    pub table_row_counts: HashMap<String, u64>,
    pub indexed_columns: HashMap<String, Vec<String>>,
}

impl Default for PlannerStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PlannerStats {
    pub fn new() -> Self {
        Self {
            table_row_counts: HashMap::new(),
            indexed_columns: HashMap::new(),
        }
    }

    pub fn row_count(&self, table: &str) -> u64 {
        *self
            .table_row_counts
            .get(&table.to_lowercase())
            .unwrap_or(&DEFAULT_ROW_ESTIMATE)
    }

    pub fn has_index(&self, table: &str, column: &str) -> bool {
        self.indexed_columns
            .get(&table.to_lowercase())
            .is_some_and(|cols| cols.iter().any(|c| c.eq_ignore_ascii_case(column)))
    }
}

/// Build a logical plan for a SELECT statement.
#[allow(clippy::too_many_arguments)]
fn plan_select(
    from: &TableRef,
    items: &[SelectItem],
    joins: &[JoinSpec],
    filter: &Option<Expr>,
    group_by: &Option<Vec<Expr>>,
    order_by: &Option<Vec<(Expr, OrderDirection)>>,
    limit: &Option<u64>,
    stats: Option<&PlannerStats>,
) -> Option<PlanNode> {
    let table = match from {
        TableRef::Named(t) => t.clone(),
        TableRef::Subquery { .. } | TableRef::TableFunction { .. } => return None, // dynamic subqueries/table functions — skip planning
    };

    let est_rows = stats
        .map(|s| s.row_count(&table))
        .unwrap_or(DEFAULT_ROW_ESTIMATE);

    // Determine access path
    let pk_from_filter = filter.as_ref().and_then(|f| extract_pk_eq_literal(Some(f)));

    let mut node = if let Some(ref pk) = pk_from_filter {
        PlanNode::PointLookup {
            table: table.clone(),
            pk: pk.clone(),
            estimated_cost: POINT_LOOKUP_COST,
        }
    } else {
        PlanNode::FullScan {
            table: table.clone(),
            estimated_rows: est_rows,
            estimated_cost: est_rows as f64 * FULL_SCAN_PER_ROW,
        }
    };

    // Add filter node if WHERE clause present (and not fully resolved by point lookup)
    if let Some(ref f) = filter {
        if pk_from_filter.is_none() {
            let sel = estimate_selectivity(f, est_rows);
            node = PlanNode::Filter {
                child: Box::new(node),
                predicate_display: format!("{f:?}").chars().take(80).collect(),
                estimated_selectivity: sel,
            };
        }
    }

    // Add join nodes (N-way)
    for js in joins {
        let right_est = stats
            .map(|s| s.row_count(&js.right_table))
            .unwrap_or(DEFAULT_ROW_ESTIMATE);
        let right_node = PlanNode::FullScan {
            table: js.right_table.clone(),
            estimated_rows: right_est,
            estimated_cost: right_est as f64 * FULL_SCAN_PER_ROW,
        };

        let join_type_str = match js.join_type {
            JoinType::Inner => "INNER",
            JoinType::Left => "LEFT",
            JoinType::Right => "RIGHT",
            JoinType::Cross => "CROSS",
        };

        // Check for equi-join and index availability
        let is_equi = js.on_clause.as_ref().is_some_and(|c| {
            matches!(
                c,
                Expr::BinOp {
                    op: BinOperator::Eq,
                    ..
                }
            )
        });

        // Detect INL-able joins when stats are available
        let inl_index = if is_equi {
            extract_join_right_column(&js.on_clause, &table, &js.right_table).and_then(
                |(_left_col, right_col)| {
                    stats.and_then(|s| {
                        if s.has_index(&js.right_table, &right_col) {
                            Some(format!("idx_{}", right_col))
                        } else {
                            None
                        }
                    })
                },
            )
        } else {
            None
        };

        if let Some(idx_name) = inl_index {
            let cost = est_rows as f64 * INL_JOIN_PER_ROW;
            node = PlanNode::IndexNestedLoopJoin {
                left: Box::new(node),
                right: Box::new(right_node),
                index_name: idx_name,
                join_type: join_type_str.to_string(),
                estimated_cost: cost,
            };
        } else if is_equi {
            let cost = est_rows as f64 * HASH_JOIN_BUILD + right_est as f64 * HASH_JOIN_PROBE;
            node = PlanNode::HashJoin {
                left: Box::new(node),
                right: Box::new(right_node),
                join_type: join_type_str.to_string(),
                estimated_cost: cost,
            };
        } else {
            let cost = est_rows as f64 * right_est as f64 * NL_JOIN_PER_PAIR;
            node = PlanNode::NestedLoopJoin {
                left: Box::new(node),
                right: Box::new(right_node),
                join_type: join_type_str.to_string(),
                estimated_cost: cost,
            };
        }
    }

    // Add aggregate node
    let has_agg = select_items_contain_aggregate(items);
    let has_group = group_by.is_some();
    if has_agg || has_group {
        let agg_count = items
            .iter()
            .filter(|i| {
                matches!(i, SelectItem::Expr { expr: Expr::Function { name, .. }, .. }
                    if is_aggregate_function(name))
            })
            .count();
        node = PlanNode::Aggregate {
            child: Box::new(node),
            group_by_count: group_by.as_ref().map_or(0, |g| g.len()),
            aggregate_count: agg_count,
        };
    }

    // Add sort node
    if let Some(ref orders) = order_by {
        let n = est_rows as f64;
        let sort_cost = if n > 1.0 {
            n * n.log2() * SORT_PER_ROW_LOG
        } else {
            0.0
        };
        let child_cost = node.cost();
        node = PlanNode::Sort {
            child: Box::new(node),
            order_by_count: orders.len(),
            estimated_cost: sort_cost + child_cost,
        };
    }

    // Add limit node
    if let Some(n) = limit {
        node = PlanNode::Limit {
            child: Box::new(node),
            count: *n,
        };
    }

    // Add project node
    let col_count = items.len();
    if col_count > 0 {
        node = PlanNode::Project {
            child: Box::new(node),
            column_count: col_count,
        };
    }

    Some(node)
}

/// Attach a query plan to a statement. The plan is informational — execution
/// still follows the original statement-based dispatch. However, EXPLAIN now
/// uses the plan tree for rich output.
pub fn plan(stmt: Statement) -> Result<Statement> {
    // The planner is purely advisory in this version — it produces plan trees
    // for EXPLAIN and EXPLAIN ANALYZE, but doesn't yet rewrite the AST.
    // Future versions will use the plan to drive execution order.
    Ok(stmt)
}

/// Generate a plan tree for a statement (used by EXPLAIN).
pub fn generate_plan(stmt: &Statement) -> Option<PlanNode> {
    generate_plan_with_stats(stmt, None)
}

/// Generate a plan tree with optional ANALYZE statistics for accurate cost estimates.
pub fn generate_plan_with_stats(
    stmt: &Statement,
    stats: Option<&PlannerStats>,
) -> Option<PlanNode> {
    match stmt {
        Statement::Select {
            from,
            items,
            joins,
            filter,
            group_by,
            order_by,
            limit,
            ..
        } => plan_select(from, items, joins, filter, group_by, order_by, limit, stats),
        _ => None,
    }
}

/// Format a plan tree as human-readable EXPLAIN output.
pub fn explain_plan(stmt: &Statement) -> String {
    explain_plan_with_stats(stmt, None)
}

/// Format a plan tree with optional stats.
pub fn explain_plan_with_stats(stmt: &Statement, stats: Option<&PlannerStats>) -> String {
    match generate_plan_with_stats(stmt, stats) {
        Some(node) => {
            let mut out = String::from("Query Plan:\n");
            out.push_str(&node.display(1));
            out.push_str(&format!("\nEstimated cost: {:.1}", node.cost()));
            out
        }
        None => format!("No plan available for: {stmt:?}"),
    }
}

use crate::sql::parser::OrderDirection;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_point_lookup() {
        let stmt = Statement::Select {
            ctes: vec![],
            from: TableRef::Named("users".to_string()),
            items: vec![SelectItem::AllColumns],
            joins: vec![],
            filter: Some(Expr::BinOp {
                left: Box::new(Expr::Column("pk".to_string())),
                op: BinOperator::Eq,
                right: Box::new(Expr::StringLit("alice".to_string())),
            }),
            as_of: None,
            valid_at: None,
            as_of_epoch: None,
            temporal: vec![],
            group_by: None,
            having: None,
            order_by: None,
            limit: None,
        };

        let plan = generate_plan(&stmt).unwrap();
        assert!(plan.cost() <= POINT_LOOKUP_COST + 0.01);
        let display = plan.display(0);
        assert!(display.contains("PointLookup"));
        assert!(display.contains("alice"));
    }

    #[test]
    fn plan_full_scan_with_filter() {
        let stmt = Statement::Select {
            ctes: vec![],
            from: TableRef::Named("orders".to_string()),
            items: vec![SelectItem::AllColumns],
            joins: vec![],
            filter: Some(Expr::BinOp {
                left: Box::new(Expr::Column("amount".to_string())),
                op: BinOperator::Gt,
                right: Box::new(Expr::NumberLit(100.0)),
            }),
            as_of: None,
            valid_at: None,
            as_of_epoch: None,
            temporal: vec![],
            group_by: None,
            having: None,
            order_by: None,
            limit: None,
        };

        let plan = generate_plan(&stmt).unwrap();
        let display = plan.display(0);
        assert!(display.contains("FullScan"));
        assert!(display.contains("Filter"));
    }

    #[test]
    fn plan_with_order_by_and_limit() {
        let stmt = Statement::Select {
            ctes: vec![],
            from: TableRef::Named("events".to_string()),
            items: vec![SelectItem::AllColumns],
            joins: vec![],
            filter: None,
            as_of: None,
            valid_at: None,
            as_of_epoch: None,
            temporal: vec![],
            group_by: None,
            having: None,
            order_by: Some(vec![(Expr::Column("ts".to_string()), OrderDirection::Desc)]),
            limit: Some(10),
        };

        let plan = generate_plan(&stmt).unwrap();
        let display = plan.display(0);
        assert!(display.contains("Sort"));
        assert!(display.contains("Limit n=10"));
    }

    #[test]
    fn plan_aggregate_with_group_by() {
        let stmt = Statement::Select {
            ctes: vec![],
            from: TableRef::Named("sales".to_string()),
            items: vec![
                SelectItem::Expr {
                    expr: Expr::Column("region".to_string()),
                    alias: None,
                },
                SelectItem::Expr {
                    expr: Expr::Function {
                        name: "SUM".to_string(),
                        args: vec![Expr::Column("amount".to_string())],
                    },
                    alias: Some("total".to_string()),
                },
            ],
            joins: vec![],
            filter: None,
            as_of: None,
            valid_at: None,
            as_of_epoch: None,
            temporal: vec![],
            group_by: Some(vec![Expr::Column("region".to_string())]),
            having: None,
            order_by: None,
            limit: None,
        };

        let plan = generate_plan(&stmt).unwrap();
        let display = plan.display(0);
        assert!(display.contains("Aggregate groups=1 aggs=1"));
    }

    #[test]
    fn explain_plan_output() {
        let stmt = Statement::Select {
            ctes: vec![],
            from: TableRef::Named("t".to_string()),
            items: vec![SelectItem::AllColumns],
            joins: vec![],
            filter: None,
            as_of: None,
            valid_at: None,
            as_of_epoch: None,
            temporal: vec![],
            group_by: None,
            having: None,
            order_by: None,
            limit: None,
        };

        let output = explain_plan(&stmt);
        assert!(output.contains("Query Plan:"));
        assert!(output.contains("FullScan table=t"));
        assert!(output.contains("Estimated cost:"));
    }

    #[test]
    fn selectivity_and_expr() {
        let sel = estimate_selectivity(
            &Expr::BinOp {
                left: Box::new(Expr::BinOp {
                    left: Box::new(Expr::Column("a".to_string())),
                    op: BinOperator::Eq,
                    right: Box::new(Expr::NumberLit(1.0)),
                }),
                op: BinOperator::And,
                right: Box::new(Expr::BinOp {
                    left: Box::new(Expr::Column("b".to_string())),
                    op: BinOperator::Gt,
                    right: Box::new(Expr::NumberLit(5.0)),
                }),
            },
            1000,
        );
        // AND: 0.1 * 0.33 = 0.033
        assert!(sel < 0.04);
        assert!(sel > 0.03);
    }
}
