use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::history::DefaultHistory;
use rustyline::validate::{
    MatchingBracketValidator, ValidationContext, ValidationResult, Validator,
};
use rustyline::{CompletionType, Config as RustylineConfig, Context, Editor, Helper};
use spectradb_core::sql::exec::SqlResult;
use spectradb_core::{BenchOptions, Config, Database, Result, SpectraError};

#[derive(Parser, Debug)]
#[command(name = "spectradb")]
#[command(about = "SpectraDB CLI")]
struct Cli {
    #[arg(long, default_value = "./data")]
    path: String,

    #[arg(long)]
    wal_fsync_every_n_records: Option<usize>,

    #[arg(long)]
    memtable_max_bytes: Option<usize>,

    #[arg(long)]
    sstable_block_bytes: Option<usize>,

    #[arg(long)]
    bloom_bits_per_key: Option<usize>,

    #[arg(long)]
    shard_count: Option<usize>,

    #[command(subcommand)]
    cmd: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Init,
    Put {
        key: String,
        value: String,
        #[arg(long, default_value_t = 0)]
        valid_from: u64,
        #[arg(long, default_value_t = u64::MAX)]
        valid_to: u64,
    },
    Get {
        key: String,
        #[arg(long)]
        as_of: Option<u64>,
        #[arg(long)]
        valid_at: Option<u64>,
    },
    Sql {
        query: String,
    },
    Explain {
        query: String,
    },
    Stats,
    Bench {
        #[arg(long, default_value_t = 50_000)]
        write_ops: usize,
        #[arg(long, default_value_t = 25_000)]
        read_ops: usize,
        #[arg(long, default_value_t = 10_000)]
        keyspace: usize,
        #[arg(long, default_value_t = 0.10)]
        read_miss_ratio: f64,
    },
    Shell {
        #[arg(long)]
        history_file: Option<String>,
        #[arg(long, default_value = "table", value_parser = ["table", "line", "json"])]
        mode: String,
        #[arg(long, default_value_t = false)]
        timer: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Table,
    Line,
    Json,
}

impl OutputMode {
    fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("line") {
            Self::Line
        } else if s.eq_ignore_ascii_case("json") {
            Self::Json
        } else {
            Self::Table
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Line => "line",
            Self::Json => "json",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShellAction {
    Continue,
    Exit,
}

struct ReplHelper {
    hinter: HistoryHinter,
    validator: MatchingBracketValidator,
    sql_keywords: Vec<String>,
    meta_commands: Vec<String>,
    table_names: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
        Self {
            hinter: HistoryHinter {},
            validator: MatchingBracketValidator::new(),
            sql_keywords: vec![
                "SELECT".to_string(),
                "FROM".to_string(),
                "WHERE".to_string(),
                "JOIN".to_string(),
                "ON".to_string(),
                "GROUP".to_string(),
                "BY".to_string(),
                "ORDER".to_string(),
                "ASC".to_string(),
                "DESC".to_string(),
                "LIMIT".to_string(),
                "AS".to_string(),
                "OF".to_string(),
                "VALID".to_string(),
                "AT".to_string(),
                "COUNT".to_string(),
                "CREATE".to_string(),
                "TABLE".to_string(),
                "VIEW".to_string(),
                "INDEX".to_string(),
                "ALTER".to_string(),
                "ADD".to_string(),
                "COLUMN".to_string(),
                "INSERT".to_string(),
                "INTO".to_string(),
                "VALUES".to_string(),
                "SHOW".to_string(),
                "TABLES".to_string(),
                "DESCRIBE".to_string(),
                "DROP".to_string(),
                "EXPLAIN".to_string(),
                "BEGIN".to_string(),
                "COMMIT".to_string(),
                "ROLLBACK".to_string(),
            ],
            meta_commands: vec![
                ".help".to_string(),
                ".exit".to_string(),
                ".quit".to_string(),
                ".tables".to_string(),
                ".schema".to_string(),
                ".stats".to_string(),
                ".mode".to_string(),
                ".timer".to_string(),
                ".clear".to_string(),
            ],
            table_names: Vec::new(),
        }
    }

    fn refresh_catalog(&mut self, db: &Database) {
        self.table_names.clear();
        if let Ok(SqlResult::Rows(rows)) = db.sql("SHOW TABLES;") {
            for row in rows {
                if let Ok(name) = String::from_utf8(row) {
                    self.table_names.push(name);
                }
            }
        }
        self.table_names.sort_unstable();
        self.table_names.dedup();
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (start, token) = extract_completion_token(line, pos);
        let token_lc = token.to_ascii_lowercase();

        let mut candidates = BTreeSet::new();
        let dot_mode = line.trim_start().starts_with('.') || token.starts_with('.');
        if dot_mode {
            for cmd in &self.meta_commands {
                if cmd.starts_with(&token) {
                    candidates.insert(cmd.clone());
                }
            }
        } else {
            for kw in &self.sql_keywords {
                if kw.to_ascii_lowercase().starts_with(&token_lc) {
                    candidates.insert(kw.clone());
                }
            }
            for table in &self.table_names {
                if table.to_ascii_lowercase().starts_with(&token_lc) {
                    candidates.insert(table.clone());
                }
            }
        }

        let out = candidates
            .into_iter()
            .map(|candidate| Pair {
                display: candidate.clone(),
                replacement: candidate,
            })
            .collect();
        Ok((start, out))
    }
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<Self::Hint> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl Highlighter for ReplHelper {}

impl Validator for ReplHelper {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        self.validator.validate(ctx)
    }

    fn validate_while_typing(&self) -> bool {
        self.validator.validate_while_typing()
    }
}

impl Helper for ReplHelper {}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let mut cfg = Config::default();
    if let Some(v) = cli.wal_fsync_every_n_records {
        cfg.wal_fsync_every_n_records = v;
    }
    if let Some(v) = cli.memtable_max_bytes {
        cfg.memtable_max_bytes = v;
    }
    if let Some(v) = cli.sstable_block_bytes {
        cfg.sstable_block_bytes = v;
    }
    if let Some(v) = cli.bloom_bits_per_key {
        cfg.bloom_bits_per_key = v;
    }
    if let Some(v) = cli.shard_count {
        cfg.shard_count = v;
    }

    match cli.cmd.unwrap_or(Command::Shell {
        history_file: None,
        mode: "table".to_string(),
        timer: false,
    }) {
        Command::Init => {
            let db = Database::open(&cli.path, cfg)?;
            println!("initialized at {}", db.root().display());
        }
        Command::Put {
            key,
            value,
            valid_from,
            valid_to,
        } => {
            let db = Database::open(&cli.path, cfg)?;
            let ts = db.put(
                key.as_bytes(),
                value.into_bytes(),
                valid_from,
                valid_to,
                Some(1),
            )?;
            println!("ok commit_ts={ts}");
        }
        Command::Get {
            key,
            as_of,
            valid_at,
        } => {
            let db = Database::open(&cli.path, cfg)?;
            match db.get(key.as_bytes(), as_of, valid_at)? {
                Some(v) => println!("{}", String::from_utf8_lossy(&v)),
                None => println!("null"),
            }
        }
        Command::Sql { query } => {
            let db = Database::open(&cli.path, cfg)?;
            execute_and_print_sql(&db, &query, OutputMode::Line)?;
        }
        Command::Explain { query } => {
            let db = Database::open(&cli.path, cfg)?;
            let q = if query
                .trim_start()
                .to_ascii_uppercase()
                .starts_with("EXPLAIN")
            {
                query
            } else {
                format!("EXPLAIN {query}")
            };
            execute_and_print_sql(&db, &q, OutputMode::Line)?;
        }
        Command::Stats => {
            let db = Database::open(&cli.path, cfg)?;
            print_stats(&db)?;
        }
        Command::Bench {
            write_ops,
            read_ops,
            keyspace,
            read_miss_ratio,
        } => {
            let db = Database::open(&cli.path, cfg)?;
            let rep = db.bench(BenchOptions {
                write_ops,
                read_ops,
                keyspace,
                read_miss_ratio,
            })?;
            println!("write_ops_per_sec={:.2}", rep.write_ops_per_sec);
            println!("read_p50_us={}", rep.read_p50_us);
            println!("read_p95_us={}", rep.read_p95_us);
            println!("read_p99_us={}", rep.read_p99_us);
            println!(
                "requested_read_miss_ratio={:.4}",
                rep.requested_read_miss_ratio
            );
            println!(
                "observed_read_miss_ratio={:.4}",
                rep.observed_read_miss_ratio
            );
            println!("bloom_miss_rate={:.4}", rep.bloom_miss_rate);
            println!("mmap_reads={}", rep.mmap_reads);
            println!("fsync_every_n_records={}", rep.fsync_every_n_records);
            println!("hasher={}", rep.hasher_impl);
        }
        Command::Shell {
            history_file,
            mode,
            timer,
        } => {
            let db = Database::open(&cli.path, cfg)?;
            run_shell(
                &db,
                history_file.as_deref(),
                OutputMode::parse(&mode),
                timer,
            )?;
        }
    }

    Ok(())
}

fn run_shell(
    db: &Database,
    history_file: Option<&str>,
    mut mode: OutputMode,
    mut timer: bool,
) -> Result<()> {
    let readline_cfg = RustylineConfig::builder()
        .completion_type(CompletionType::List)
        .build();
    let mut editor: Editor<ReplHelper, DefaultHistory> = Editor::with_config(readline_cfg)
        .map_err(|e| SpectraError::SqlExec(format!("shell initialization failed: {e}")))?;

    let mut helper = ReplHelper::new();
    helper.refresh_catalog(db);
    editor.set_helper(Some(helper));

    let history_path = history_file
        .map(PathBuf::from)
        .unwrap_or_else(|| db.root().join(".spectradb_history"));
    let history_path_str = history_path.to_string_lossy().to_string();
    let _ = editor.load_history(&history_path_str);

    print_shell_welcome(mode, timer);

    let mut pending_sql = String::new();
    loop {
        let prompt = if pending_sql.is_empty() {
            "spectradb> "
        } else {
            "      ...> "
        };

        match editor.readline(prompt) {
            Ok(line) => {
                let trimmed = line.trim();
                if pending_sql.is_empty() {
                    if trimmed.is_empty() {
                        continue;
                    }

                    if trimmed.starts_with('.') {
                        let _ = editor.add_history_entry(trimmed);
                        match handle_meta_command(trimmed, db, &mut mode, &mut timer)? {
                            ShellAction::Continue => {}
                            ShellAction::Exit => break,
                        }
                        if let Some(h) = editor.helper_mut() {
                            h.refresh_catalog(db);
                        }
                        continue;
                    }
                }

                pending_sql.push_str(&line);
                pending_sql.push('\n');
                if !is_sql_statement_complete(&pending_sql) {
                    continue;
                }

                let sql = pending_sql.trim().to_string();
                pending_sql.clear();
                if sql.is_empty() {
                    continue;
                }

                let _ = editor.add_history_entry(sql.as_str());
                let started = Instant::now();
                if let Err(e) = execute_and_print_sql(db, &sql, mode) {
                    eprintln!("error: {e}");
                }
                if timer {
                    println!("(elapsed: {:.3?})", started.elapsed());
                }

                if let Some(h) = editor.helper_mut() {
                    h.refresh_catalog(db);
                }
            }
            Err(ReadlineError::Interrupted) => {
                if pending_sql.is_empty() {
                    println!("^C (type .exit to quit)");
                } else {
                    println!("^C (cleared pending statement)");
                    pending_sql.clear();
                }
            }
            Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(e) => {
                return Err(SpectraError::SqlExec(format!("shell read failure: {e}")));
            }
        }
    }

    let _ = editor.save_history(&history_path_str);
    Ok(())
}

fn handle_meta_command(
    cmd: &str,
    db: &Database,
    mode: &mut OutputMode,
    timer: &mut bool,
) -> Result<ShellAction> {
    let mut parts = cmd.split_whitespace();
    let Some(head) = parts.next() else {
        return Ok(ShellAction::Continue);
    };

    match head {
        ".help" => {
            print_shell_help();
        }
        ".exit" | ".quit" => return Ok(ShellAction::Exit),
        ".tables" => execute_and_print_sql(db, "SHOW TABLES;", *mode)?,
        ".schema" => {
            if let Some(table) = parts.next() {
                let query = format!("DESCRIBE {table};");
                execute_and_print_sql(db, &query, *mode)?;
            } else {
                eprintln!("usage: .schema <table>");
            }
        }
        ".stats" => {
            print_stats(db)?;
        }
        ".mode" => {
            if let Some(m) = parts.next() {
                *mode = OutputMode::parse(m);
                println!("output mode: {}", mode.as_str());
            } else {
                eprintln!("usage: .mode <table|line|json>");
            }
        }
        ".timer" => {
            if let Some(v) = parts.next() {
                if v.eq_ignore_ascii_case("on") {
                    *timer = true;
                } else if v.eq_ignore_ascii_case("off") {
                    *timer = false;
                } else {
                    eprintln!("usage: .timer <on|off>");
                }
            } else {
                *timer = !*timer;
            }
            println!("timer: {}", if *timer { "on" } else { "off" });
        }
        ".clear" => {
            print!("\x1B[2J\x1B[H");
            let _ = io::stdout().flush();
        }
        _ => {
            eprintln!("unknown meta command: {head} (use .help)");
        }
    }

    Ok(ShellAction::Continue)
}

fn execute_and_print_sql(db: &Database, query: &str, mode: OutputMode) -> Result<()> {
    match db.sql(query)? {
        SqlResult::Affected {
            rows,
            commit_ts,
            message,
        } => {
            println!("{message}; rows={rows}; commit_ts={commit_ts:?}");
        }
        SqlResult::Rows(rows) => {
            print_rows(rows, mode);
        }
        SqlResult::Explain(plan) => {
            println!("{plan}");
        }
    }
    Ok(())
}

fn print_rows(rows: Vec<Vec<u8>>, mode: OutputMode) {
    if rows.is_empty() {
        println!("(0 rows)");
        return;
    }

    match mode {
        OutputMode::Line => {
            for row in rows {
                println!("{}", String::from_utf8_lossy(&row));
            }
        }
        OutputMode::Json => {
            for row in rows {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&row) {
                    match serde_json::to_string_pretty(&v) {
                        Ok(pretty) => println!("{pretty}"),
                        Err(_) => println!("{}", String::from_utf8_lossy(&row)),
                    }
                } else {
                    println!("{}", String::from_utf8_lossy(&row));
                }
            }
        }
        OutputMode::Table => {
            if let Some((headers, data)) = rows_to_table(&rows) {
                print_ascii_table(&headers, &data);
            } else {
                for row in rows {
                    println!("{}", String::from_utf8_lossy(&row));
                }
            }
        }
    }
}

fn rows_to_table(rows: &[Vec<u8>]) -> Option<(Vec<String>, Vec<Vec<String>>)> {
    let mut parsed_json = Vec::with_capacity(rows.len());
    for row in rows {
        let value = serde_json::from_slice::<serde_json::Value>(row).ok();
        parsed_json.push(value);
    }

    if parsed_json.iter().all(Option::is_none) {
        let data = rows
            .iter()
            .map(|row| vec![String::from_utf8_lossy(row).to_string()])
            .collect::<Vec<_>>();
        return Some((vec!["value".to_string()], data));
    }

    if !parsed_json
        .iter()
        .all(|v| matches!(v, Some(serde_json::Value::Object(_))))
    {
        return None;
    }

    let mut headers = BTreeSet::new();
    for value in parsed_json.iter().flatten() {
        if let serde_json::Value::Object(map) = value {
            for key in map.keys() {
                headers.insert(key.to_string());
            }
        }
    }
    let headers = headers.into_iter().collect::<Vec<_>>();
    if headers.is_empty() {
        return None;
    }

    let mut data = Vec::with_capacity(parsed_json.len());
    for value in parsed_json.into_iter().flatten() {
        let mut row = Vec::with_capacity(headers.len());
        if let serde_json::Value::Object(map) = value {
            for header in &headers {
                let cell = map
                    .get(header)
                    .map(value_to_cell)
                    .unwrap_or_else(|| "null".to_string());
                row.push(cell);
            }
            data.push(row);
        }
    }
    Some((headers, data))
}

fn value_to_cell(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        _ => value.to_string(),
    }
}

fn print_ascii_table(headers: &[String], rows: &[Vec<String>]) {
    let mut widths = headers.iter().map(|h| display_width(h)).collect::<Vec<_>>();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(display_width(cell));
            }
        }
    }

    let rule = table_rule(&widths);
    println!("{rule}");
    print_table_row(headers, &widths);
    println!("{rule}");
    for row in rows {
        print_table_row(row, &widths);
    }
    println!("{rule}");
    println!("({} rows)", rows.len());
}

fn table_rule(widths: &[usize]) -> String {
    let mut out = String::from("+");
    for width in widths {
        out.push_str(&"-".repeat(*width + 2));
        out.push('+');
    }
    out
}

fn print_table_row(cells: &[String], widths: &[usize]) {
    let mut line = String::from("|");
    for (i, width) in widths.iter().enumerate() {
        let raw = cells.get(i).cloned().unwrap_or_default();
        let cell = raw.replace('\n', "\\n");
        let cell_width = display_width(&cell);
        line.push(' ');
        line.push_str(&cell);
        if *width > cell_width {
            line.push_str(&" ".repeat(*width - cell_width));
        }
        line.push(' ');
        line.push('|');
    }
    println!("{line}");
}

fn display_width(s: &str) -> usize {
    s.chars().count()
}

fn print_stats(db: &Database) -> Result<()> {
    let s = db.stats()?;
    println!(
        "shards={} puts={} gets={} flushes={} compactions={} bloom_negatives={} mmap_block_reads={}",
        s.shard_count, s.puts, s.gets, s.flushes, s.compactions, s.bloom_negatives, s.mmap_block_reads
    );
    Ok(())
}

fn extract_completion_token(line: &str, pos: usize) -> (usize, String) {
    let upto = &line[..pos];
    let start = upto
        .rfind(|c: char| c.is_whitespace() || "(),;=.".contains(c))
        .map_or(0, |idx| idx + 1);
    (start, upto[start..].to_string())
}

fn is_sql_statement_complete(sql: &str) -> bool {
    let mut in_string = false;
    let mut escaped = false;
    let mut ended_with_semicolon = false;

    for ch in sql.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '\'' {
                in_string = false;
            }
            continue;
        }

        if ch == '\'' {
            in_string = true;
            continue;
        }

        if ch == ';' {
            ended_with_semicolon = true;
            continue;
        }

        if !ch.is_whitespace() {
            ended_with_semicolon = false;
        }
    }

    !in_string && ended_with_semicolon
}

fn print_shell_welcome(mode: OutputMode, timer: bool) {
    println!("SpectraDB Interactive Shell");
    println!("type .help for commands; terminate SQL statements with ';'");
    println!(
        "mode={} timer={}",
        mode.as_str(),
        if timer { "on" } else { "off" }
    );
}

fn print_shell_help() {
    println!("meta commands:");
    println!("  .help                     show this help");
    println!("  .exit | .quit             exit shell");
    println!("  .tables                   list tables");
    println!("  .schema <table>           describe table schema");
    println!("  .stats                    show engine stats");
    println!("  .mode <table|line|json>   set output format");
    println!("  .timer [on|off]           toggle or set per-query timing");
    println!("  .clear                    clear screen");
    println!();
    println!("tips:");
    println!("  - autocomplete with TAB for SQL keywords, meta commands, and table names");
    println!("  - use Up/Down for command history");
    println!("  - SQL examples:");
    println!("    SELECT doc FROM t WHERE pk='k';");
    println!("    SELECT pk, doc FROM l JOIN r ON l.pk=r.pk ORDER BY pk;");
    println!("    SELECT pk, count(*) FROM t GROUP BY pk ORDER BY pk DESC LIMIT 10;");
}
