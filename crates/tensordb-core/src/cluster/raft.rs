use std::collections::HashMap;

/// Raft consensus state for a single node.
/// This is a simplified Raft implementation for leader election and log replication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaftState {
    Follower,
    Candidate,
    Leader,
}

/// A log entry in the Raft log.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub kind: LogEntryKind,
    pub data: Vec<u8>,
}

/// Kind of log entry.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LogEntryKind {
    /// Normal data replication entry.
    Data,
    /// Cluster configuration change.
    ConfigChange,
    /// No-op entry (used during leader election).
    Noop,
}

/// Vote request message.
#[derive(Debug, Clone)]
pub struct VoteRequest {
    pub term: u64,
    pub candidate_id: String,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

/// Vote response message.
#[derive(Debug, Clone)]
pub struct VoteResponse {
    pub term: u64,
    pub vote_granted: bool,
}

/// Append entries request (log replication).
#[derive(Debug, Clone)]
pub struct AppendEntriesRequest {
    pub term: u64,
    pub leader_id: String,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    pub leader_commit: u64,
}

/// Append entries response.
#[derive(Debug, Clone)]
pub struct AppendEntriesResponse {
    pub term: u64,
    pub success: bool,
    pub match_index: u64,
}

/// A Raft node — manages consensus state for one node in the cluster.
pub struct RaftNode {
    pub node_id: String,
    pub state: RaftState,
    pub current_term: u64,
    pub voted_for: Option<String>,
    pub log: Vec<LogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    // Leader state
    pub next_index: HashMap<String, u64>,
    pub match_index: HashMap<String, u64>,
    // Cluster membership
    pub peers: Vec<String>,
    // Election state
    pub votes_received: usize,
    pub election_timeout_ms: u64,
    pub last_heartbeat_ms: u64,
}

impl RaftNode {
    pub fn new(node_id: &str, peers: Vec<String>) -> Self {
        RaftNode {
            node_id: node_id.to_string(),
            state: RaftState::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            peers,
            votes_received: 0,
            election_timeout_ms: 150 + (hash_node_id(node_id) % 150), // 150-300ms
            last_heartbeat_ms: current_timestamp_ms(),
        }
    }

    /// Start an election — transition to Candidate.
    pub fn start_election(&mut self) -> VoteRequest {
        self.state = RaftState::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.node_id.clone());
        self.votes_received = 1; // Vote for self

        let (last_log_index, last_log_term) = self.last_log_info();

        VoteRequest {
            term: self.current_term,
            candidate_id: self.node_id.clone(),
            last_log_index,
            last_log_term,
        }
    }

    /// Handle a vote request from another candidate.
    pub fn handle_vote_request(&mut self, req: &VoteRequest) -> VoteResponse {
        // If request term is higher, step down
        if req.term > self.current_term {
            self.current_term = req.term;
            self.state = RaftState::Follower;
            self.voted_for = None;
        }

        let vote_granted = if req.term < self.current_term
            || (self.voted_for.is_some() && self.voted_for.as_deref() != Some(&req.candidate_id))
        {
            false
        } else {
            // Check if candidate's log is at least as up-to-date
            let (my_last_index, my_last_term) = self.last_log_info();
            if req.last_log_term > my_last_term
                || (req.last_log_term == my_last_term && req.last_log_index >= my_last_index)
            {
                self.voted_for = Some(req.candidate_id.clone());
                self.last_heartbeat_ms = current_timestamp_ms();
                true
            } else {
                false
            }
        };

        VoteResponse {
            term: self.current_term,
            vote_granted,
        }
    }

    /// Handle a vote response.
    pub fn handle_vote_response(&mut self, resp: &VoteResponse) {
        if resp.term > self.current_term {
            self.current_term = resp.term;
            self.state = RaftState::Follower;
            self.voted_for = None;
            return;
        }

        if self.state != RaftState::Candidate {
            return;
        }

        if resp.vote_granted {
            self.votes_received += 1;
            let total_nodes = self.peers.len() + 1;
            let majority = total_nodes / 2 + 1;
            if self.votes_received >= majority {
                self.become_leader();
            }
        }
    }

    /// Transition to leader.
    fn become_leader(&mut self) {
        self.state = RaftState::Leader;
        let next = self.log.len() as u64 + 1;
        for peer in &self.peers {
            self.next_index.insert(peer.clone(), next);
            self.match_index.insert(peer.clone(), 0);
        }

        // Append a no-op entry to commit entries from previous terms
        self.log.push(LogEntry {
            term: self.current_term,
            index: self.log.len() as u64 + 1,
            kind: LogEntryKind::Noop,
            data: Vec::new(),
        });
    }

    /// Handle append entries (leader → follower).
    pub fn handle_append_entries(&mut self, req: &AppendEntriesRequest) -> AppendEntriesResponse {
        if req.term < self.current_term {
            return AppendEntriesResponse {
                term: self.current_term,
                success: false,
                match_index: 0,
            };
        }

        // Step down if we see a higher term
        if req.term > self.current_term {
            self.current_term = req.term;
            self.voted_for = None;
        }
        self.state = RaftState::Follower;
        self.last_heartbeat_ms = current_timestamp_ms();

        // Check log consistency
        if req.prev_log_index > 0 {
            if let Some(entry) = self.log.get((req.prev_log_index - 1) as usize) {
                if entry.term != req.prev_log_term {
                    // Conflict: delete this entry and all after it
                    self.log.truncate((req.prev_log_index - 1) as usize);
                    return AppendEntriesResponse {
                        term: self.current_term,
                        success: false,
                        match_index: self.log.len() as u64,
                    };
                }
            } else {
                return AppendEntriesResponse {
                    term: self.current_term,
                    success: false,
                    match_index: self.log.len() as u64,
                };
            }
        }

        // Append new entries
        for entry in &req.entries {
            let idx = (entry.index - 1) as usize;
            if idx < self.log.len() {
                if self.log[idx].term != entry.term {
                    self.log.truncate(idx);
                    self.log.push(entry.clone());
                }
            } else {
                self.log.push(entry.clone());
            }
        }

        // Update commit index
        if req.leader_commit > self.commit_index {
            self.commit_index = req.leader_commit.min(self.log.len() as u64);
        }

        AppendEntriesResponse {
            term: self.current_term,
            success: true,
            match_index: self.log.len() as u64,
        }
    }

    /// Append a client entry (leader only).
    pub fn append_client_entry(&mut self, data: Vec<u8>) -> Option<u64> {
        if self.state != RaftState::Leader {
            return None;
        }
        let index = self.log.len() as u64 + 1;
        self.log.push(LogEntry {
            term: self.current_term,
            index,
            kind: LogEntryKind::Data,
            data,
        });
        Some(index)
    }

    /// Create append entries request for a follower.
    pub fn create_append_entries(&self, peer_id: &str) -> Option<AppendEntriesRequest> {
        if self.state != RaftState::Leader {
            return None;
        }

        let next_idx = self.next_index.get(peer_id).copied().unwrap_or(1);
        let prev_log_index = next_idx.saturating_sub(1);
        let prev_log_term = if prev_log_index > 0 {
            self.log
                .get((prev_log_index - 1) as usize)
                .map(|e| e.term)
                .unwrap_or(0)
        } else {
            0
        };

        let entries: Vec<LogEntry> = self
            .log
            .iter()
            .skip((next_idx - 1) as usize)
            .cloned()
            .collect();

        Some(AppendEntriesRequest {
            term: self.current_term,
            leader_id: self.node_id.clone(),
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: self.commit_index,
        })
    }

    /// Handle append entries response from a follower (leader side).
    pub fn handle_append_entries_response(&mut self, peer_id: &str, resp: &AppendEntriesResponse) {
        if resp.term > self.current_term {
            self.current_term = resp.term;
            self.state = RaftState::Follower;
            self.voted_for = None;
            return;
        }

        if self.state != RaftState::Leader {
            return;
        }

        if resp.success {
            self.next_index
                .insert(peer_id.to_string(), resp.match_index + 1);
            self.match_index
                .insert(peer_id.to_string(), resp.match_index);
            self.advance_commit_index();
        } else {
            // Decrement next_index and retry
            let next = self.next_index.get(peer_id).copied().unwrap_or(1);
            if next > 1 {
                self.next_index.insert(peer_id.to_string(), next - 1);
            }
        }
    }

    /// Advance commit index based on majority match.
    fn advance_commit_index(&mut self) {
        for n in (self.commit_index + 1)..=(self.log.len() as u64) {
            if let Some(entry) = self.log.get((n - 1) as usize) {
                if entry.term != self.current_term {
                    continue;
                }
            }
            let mut match_count = 1; // Leader itself
            for &mi in self.match_index.values() {
                if mi >= n {
                    match_count += 1;
                }
            }
            let total_nodes = self.peers.len() + 1;
            let majority = total_nodes / 2 + 1;
            if match_count >= majority {
                self.commit_index = n;
            }
        }
    }

    /// Check if election timeout has elapsed.
    pub fn election_timeout_elapsed(&self) -> bool {
        let now = current_timestamp_ms();
        now.saturating_sub(self.last_heartbeat_ms) >= self.election_timeout_ms
    }

    /// Get entries that are committed but not yet applied.
    pub fn entries_to_apply(&self) -> Vec<&LogEntry> {
        let start = self.last_applied as usize;
        let end = self.commit_index as usize;
        if start >= end {
            return Vec::new();
        }
        self.log[start..end].iter().collect()
    }

    /// Mark entries as applied.
    pub fn mark_applied(&mut self, up_to: u64) {
        self.last_applied = up_to;
    }

    fn last_log_info(&self) -> (u64, u64) {
        match self.log.last() {
            Some(entry) => (entry.index, entry.term),
            None => (0, 0),
        }
    }
}

fn hash_node_id(id: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node_is_follower() {
        let node = RaftNode::new("node1", vec!["node2".into(), "node3".into()]);
        assert_eq!(node.state, RaftState::Follower);
        assert_eq!(node.current_term, 0);
        assert!(node.voted_for.is_none());
    }

    #[test]
    fn test_start_election() {
        let mut node = RaftNode::new("node1", vec!["node2".into(), "node3".into()]);
        let req = node.start_election();
        assert_eq!(node.state, RaftState::Candidate);
        assert_eq!(node.current_term, 1);
        assert_eq!(node.voted_for, Some("node1".to_string()));
        assert_eq!(node.votes_received, 1);
        assert_eq!(req.term, 1);
        assert_eq!(req.candidate_id, "node1");
    }

    #[test]
    fn test_vote_granted() {
        let mut voter = RaftNode::new("node2", vec!["node1".into(), "node3".into()]);
        let req = VoteRequest {
            term: 1,
            candidate_id: "node1".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };
        let resp = voter.handle_vote_request(&req);
        assert!(resp.vote_granted);
        assert_eq!(voter.voted_for, Some("node1".to_string()));
    }

    #[test]
    fn test_vote_denied_already_voted() {
        let mut voter = RaftNode::new("node2", vec!["node1".into(), "node3".into()]);
        voter.current_term = 1;
        voter.voted_for = Some("node3".to_string());

        let req = VoteRequest {
            term: 1,
            candidate_id: "node1".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };
        let resp = voter.handle_vote_request(&req);
        assert!(!resp.vote_granted);
    }

    #[test]
    fn test_leader_election_with_majority() {
        let mut node = RaftNode::new("node1", vec!["node2".into(), "node3".into()]);
        node.start_election();

        // Receive one vote — not yet majority (need 2 of 3)
        node.handle_vote_response(&VoteResponse {
            term: 1,
            vote_granted: true,
        });
        assert_eq!(node.state, RaftState::Leader);
        // With 3 nodes, majority = 2. Self-vote + 1 = 2 = majority
    }

    #[test]
    fn test_leader_appends_noop() {
        let mut node = RaftNode::new("node1", vec!["node2".into(), "node3".into()]);
        node.start_election();
        node.handle_vote_response(&VoteResponse {
            term: 1,
            vote_granted: true,
        });
        assert_eq!(node.state, RaftState::Leader);
        assert_eq!(node.log.len(), 1);
        assert_eq!(node.log[0].kind, LogEntryKind::Noop);
    }

    #[test]
    fn test_append_client_entry() {
        let mut node = RaftNode::new("node1", vec!["node2".into(), "node3".into()]);
        node.start_election();
        node.handle_vote_response(&VoteResponse {
            term: 1,
            vote_granted: true,
        });

        let idx = node.append_client_entry(b"hello".to_vec());
        assert_eq!(idx, Some(2)); // 1 = noop, 2 = data
        assert_eq!(node.log.len(), 2);
    }

    #[test]
    fn test_follower_cannot_append() {
        let mut node = RaftNode::new("node1", vec!["node2".into()]);
        let idx = node.append_client_entry(b"hello".to_vec());
        assert_eq!(idx, None);
    }

    #[test]
    fn test_log_replication() {
        // Leader creates entries
        let mut leader = RaftNode::new("leader", vec!["follower".into()]);
        leader.start_election();
        leader.handle_vote_response(&VoteResponse {
            term: 1,
            vote_granted: true,
        });
        leader.append_client_entry(b"data1".to_vec());
        leader.append_client_entry(b"data2".to_vec());

        // Leader sends to follower
        let req = leader.create_append_entries("follower").unwrap();
        assert_eq!(req.entries.len(), 3); // noop + 2 data

        // Follower handles
        let mut follower = RaftNode::new("follower", vec!["leader".into()]);
        let resp = follower.handle_append_entries(&req);
        assert!(resp.success);
        assert_eq!(follower.log.len(), 3);
    }

    #[test]
    fn test_commit_index_advances() {
        let mut leader = RaftNode::new("leader", vec!["f1".into(), "f2".into()]);
        leader.start_election();
        leader.handle_vote_response(&VoteResponse {
            term: 1,
            vote_granted: true,
        });
        leader.handle_vote_response(&VoteResponse {
            term: 1,
            vote_granted: true,
        });
        leader.append_client_entry(b"data".to_vec());

        // Simulate follower f1 matching
        leader.handle_append_entries_response(
            "f1",
            &AppendEntriesResponse {
                term: 1,
                success: true,
                match_index: 2,
            },
        );

        // With leader + f1 matched, majority achieved
        assert_eq!(leader.commit_index, 2);
    }

    #[test]
    fn test_step_down_on_higher_term() {
        let mut node = RaftNode::new("node1", vec!["node2".into()]);
        node.start_election();
        assert_eq!(node.state, RaftState::Candidate);

        let req = AppendEntriesRequest {
            term: 5,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };
        node.handle_append_entries(&req);
        assert_eq!(node.state, RaftState::Follower);
        assert_eq!(node.current_term, 5);
    }

    #[test]
    fn test_entries_to_apply() {
        let mut node = RaftNode::new("node1", vec![]);
        node.log.push(LogEntry {
            term: 1,
            index: 1,
            kind: LogEntryKind::Data,
            data: b"a".to_vec(),
        });
        node.log.push(LogEntry {
            term: 1,
            index: 2,
            kind: LogEntryKind::Data,
            data: b"b".to_vec(),
        });
        node.commit_index = 2;
        node.last_applied = 0;

        let to_apply = node.entries_to_apply();
        assert_eq!(to_apply.len(), 2);

        node.mark_applied(2);
        assert!(node.entries_to_apply().is_empty());
    }
}
