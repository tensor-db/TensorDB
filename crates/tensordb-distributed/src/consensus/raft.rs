//! Basic Raft consensus implementation for leader election and metadata replication.
//!
//! This implements the core Raft state machine: leader election via RequestVote,
//! and log replication via AppendEntries. Suitable for coordinating metadata
//! operations (schema changes, shard assignments) across the cluster.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::node::NodeRole;

/// Raft log entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub command: RaftCommand,
}

/// Commands replicated via Raft.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RaftCommand {
    /// No-op entry appended on leader election.
    Noop,
    /// Assign shard to a node.
    AssignShard { shard_id: u32, node_id: String },
    /// Remove shard assignment.
    UnassignShard { shard_id: u32, node_id: String },
    /// Update cluster configuration.
    ConfigChange {
        add_node: Option<String>,
        remove_node: Option<String>,
    },
}

/// RequestVote RPC arguments.
#[derive(Debug, Clone)]
pub struct VoteRequest {
    pub term: u64,
    pub candidate_id: String,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

/// RequestVote RPC response.
#[derive(Debug, Clone)]
pub struct VoteResponse {
    pub term: u64,
    pub vote_granted: bool,
}

/// AppendEntries RPC arguments.
#[derive(Debug, Clone)]
pub struct AppendEntriesRequest {
    pub term: u64,
    pub leader_id: String,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    pub leader_commit: u64,
}

/// AppendEntries RPC response.
#[derive(Debug, Clone)]
pub struct AppendEntriesResponse {
    pub term: u64,
    pub success: bool,
}

/// Core Raft state machine.
pub struct RaftState {
    /// This node's ID.
    pub node_id: String,
    /// Current term.
    current_term: AtomicU64,
    /// Who we voted for in the current term.
    voted_for: RwLock<Option<String>>,
    /// Current role.
    role: RwLock<NodeRole>,
    /// Raft log (in-memory for now).
    log: RwLock<Vec<LogEntry>>,
    /// Index of highest committed entry.
    commit_index: AtomicU64,
    /// Index of highest applied entry.
    pub last_applied: AtomicU64,
    /// For leader: next index to send to each follower.
    next_index: RwLock<HashMap<String, u64>>,
    /// For leader: highest replicated index per follower.
    match_index: RwLock<HashMap<String, u64>>,
    /// Current leader ID (if known).
    leader_id: RwLock<Option<String>>,
}

impl RaftState {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            current_term: AtomicU64::new(0),
            voted_for: RwLock::new(None),
            role: RwLock::new(NodeRole::Follower),
            log: RwLock::new(Vec::new()),
            commit_index: AtomicU64::new(0),
            last_applied: AtomicU64::new(0),
            next_index: RwLock::new(HashMap::new()),
            match_index: RwLock::new(HashMap::new()),
            leader_id: RwLock::new(None),
        }
    }

    pub fn current_term(&self) -> u64 {
        self.current_term.load(Ordering::SeqCst)
    }

    pub fn role(&self) -> NodeRole {
        *self.role.read().unwrap()
    }

    pub fn leader_id(&self) -> Option<String> {
        self.leader_id.read().unwrap().clone()
    }

    pub fn commit_index(&self) -> u64 {
        self.commit_index.load(Ordering::SeqCst)
    }

    pub fn last_log_index(&self) -> u64 {
        self.log.read().unwrap().last().map_or(0, |e| e.index)
    }

    pub fn last_log_term(&self) -> u64 {
        self.log.read().unwrap().last().map_or(0, |e| e.term)
    }

    /// Start an election: increment term, vote for self, become candidate.
    pub fn start_election(&self) -> VoteRequest {
        let new_term = self.current_term.fetch_add(1, Ordering::SeqCst) + 1;
        *self.voted_for.write().unwrap() = Some(self.node_id.clone());
        *self.role.write().unwrap() = NodeRole::Candidate;
        *self.leader_id.write().unwrap() = None;

        VoteRequest {
            term: new_term,
            candidate_id: self.node_id.clone(),
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        }
    }

    /// Handle a RequestVote RPC.
    pub fn handle_vote_request(&self, req: &VoteRequest) -> VoteResponse {
        let current_term = self.current_term.load(Ordering::SeqCst);

        // Reject if our term is higher
        if req.term < current_term {
            return VoteResponse {
                term: current_term,
                vote_granted: false,
            };
        }

        // If request has higher term, step down
        if req.term > current_term {
            self.current_term.store(req.term, Ordering::SeqCst);
            *self.voted_for.write().unwrap() = None;
            *self.role.write().unwrap() = NodeRole::Follower;
        }

        // Check if we can vote for this candidate
        let voted_for = self.voted_for.read().unwrap();
        let can_vote = voted_for.is_none() || voted_for.as_deref() == Some(&req.candidate_id);

        // Check log freshness
        let our_last_term = self.last_log_term();
        let our_last_index = self.last_log_index();
        let log_ok = req.last_log_term > our_last_term
            || (req.last_log_term == our_last_term && req.last_log_index >= our_last_index);

        if can_vote && log_ok {
            drop(voted_for);
            *self.voted_for.write().unwrap() = Some(req.candidate_id.clone());
            VoteResponse {
                term: req.term,
                vote_granted: true,
            }
        } else {
            VoteResponse {
                term: req.term,
                vote_granted: false,
            }
        }
    }

    /// Become leader: initialize next_index and match_index for all peers.
    pub fn become_leader(&self, peers: &[String]) {
        *self.role.write().unwrap() = NodeRole::Leader;
        *self.leader_id.write().unwrap() = Some(self.node_id.clone());

        let next = self.last_log_index() + 1;
        let mut next_index = self.next_index.write().unwrap();
        let mut match_index = self.match_index.write().unwrap();
        next_index.clear();
        match_index.clear();
        for peer in peers {
            next_index.insert(peer.clone(), next);
            match_index.insert(peer.clone(), 0);
        }

        // Append no-op entry to commit entries from previous terms
        drop(next_index);
        drop(match_index);
        self.append_entry(RaftCommand::Noop);
    }

    /// Handle an AppendEntries RPC (follower).
    pub fn handle_append_entries(&self, req: &AppendEntriesRequest) -> AppendEntriesResponse {
        let current_term = self.current_term.load(Ordering::SeqCst);

        // Reject if our term is higher
        if req.term < current_term {
            return AppendEntriesResponse {
                term: current_term,
                success: false,
            };
        }

        // Accept leader's term
        if req.term > current_term {
            self.current_term.store(req.term, Ordering::SeqCst);
            *self.voted_for.write().unwrap() = None;
        }
        *self.role.write().unwrap() = NodeRole::Follower;
        *self.leader_id.write().unwrap() = Some(req.leader_id.clone());

        let mut log = self.log.write().unwrap();

        // Check log consistency
        if req.prev_log_index > 0 {
            match log.iter().find(|e| e.index == req.prev_log_index) {
                Some(entry) if entry.term == req.prev_log_term => {}
                _ => {
                    return AppendEntriesResponse {
                        term: req.term,
                        success: false,
                    };
                }
            }
        }

        // Append new entries (removing conflicting ones)
        for entry in &req.entries {
            // Remove any conflicting entry at this index
            log.retain(|e| e.index != entry.index || e.term == entry.term);
            if !log
                .iter()
                .any(|e| e.index == entry.index && e.term == entry.term)
            {
                log.push(entry.clone());
            }
        }

        // Update commit index
        if req.leader_commit > self.commit_index.load(Ordering::SeqCst) {
            let last = log.last().map_or(0, |e| e.index);
            let new_commit = req.leader_commit.min(last);
            self.commit_index.store(new_commit, Ordering::SeqCst);
        }

        AppendEntriesResponse {
            term: req.term,
            success: true,
        }
    }

    /// Append a new entry to the log (leader only).
    pub fn append_entry(&self, command: RaftCommand) -> u64 {
        let term = self.current_term.load(Ordering::SeqCst);
        let mut log = self.log.write().unwrap();
        let index = log.last().map_or(1, |e| e.index + 1);
        log.push(LogEntry {
            term,
            index,
            command,
        });
        index
    }

    /// Get entries from the log starting at `from_index`.
    pub fn get_entries_from(&self, from_index: u64) -> Vec<LogEntry> {
        let log = self.log.read().unwrap();
        log.iter()
            .filter(|e| e.index >= from_index)
            .cloned()
            .collect()
    }

    /// Get the log length.
    pub fn log_length(&self) -> usize {
        self.log.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn election_increments_term() {
        let state = RaftState::new("node1".to_string());
        assert_eq!(state.current_term(), 0);
        let req = state.start_election();
        assert_eq!(req.term, 1);
        assert_eq!(state.current_term(), 1);
        assert_eq!(state.role(), NodeRole::Candidate);
    }

    #[test]
    fn vote_granted_to_first_candidate() {
        let voter = RaftState::new("voter".to_string());
        let req = VoteRequest {
            term: 1,
            candidate_id: "candidate1".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };
        let resp = voter.handle_vote_request(&req);
        assert!(resp.vote_granted);
    }

    #[test]
    fn vote_rejected_for_lower_term() {
        let voter = RaftState::new("voter".to_string());
        voter.current_term.store(5, Ordering::SeqCst);
        let req = VoteRequest {
            term: 3,
            candidate_id: "candidate1".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };
        let resp = voter.handle_vote_request(&req);
        assert!(!resp.vote_granted);
        assert_eq!(resp.term, 5);
    }

    #[test]
    fn become_leader_appends_noop() {
        let state = RaftState::new("leader".to_string());
        state.current_term.store(1, Ordering::SeqCst);
        state.become_leader(&["follower1".to_string(), "follower2".to_string()]);
        assert_eq!(state.role(), NodeRole::Leader);
        assert_eq!(state.log_length(), 1);
    }

    #[test]
    fn append_entries_accepted() {
        let follower = RaftState::new("follower".to_string());
        let req = AppendEntriesRequest {
            term: 1,
            leader_id: "leader1".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 1,
                index: 1,
                command: RaftCommand::Noop,
            }],
            leader_commit: 1,
        };
        let resp = follower.handle_append_entries(&req);
        assert!(resp.success);
        assert_eq!(follower.log_length(), 1);
        assert_eq!(follower.commit_index(), 1);
        assert_eq!(follower.leader_id(), Some("leader1".to_string()));
    }

    #[test]
    fn append_entries_rejected_for_lower_term() {
        let follower = RaftState::new("follower".to_string());
        follower.current_term.store(5, Ordering::SeqCst);
        let req = AppendEntriesRequest {
            term: 3,
            leader_id: "leader1".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };
        let resp = follower.handle_append_entries(&req);
        assert!(!resp.success);
        assert_eq!(resp.term, 5);
    }
}
