use crate::error::Result;
use crate::sql::parser::Statement;

pub fn plan(stmt: Statement) -> Result<Statement> {
    // MVP planner: syntax tree is already executable plan.
    Ok(stmt)
}
