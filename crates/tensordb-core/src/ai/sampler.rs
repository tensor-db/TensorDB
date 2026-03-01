//! Token sampling strategies: greedy, top-p (nucleus), temperature, and repetition penalty.

/// Token sampler with configurable strategy.
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    rng_state: u64,
}

impl Sampler {
    /// Create a new sampler.
    ///
    /// - `temperature`: Controls randomness. 0.0 = greedy (argmax). Higher = more random.
    /// - `top_p`: Nucleus sampling threshold. 1.0 = disabled. 0.9 = sample from top 90% probability mass.
    /// - `repetition_penalty`: Penalize recently generated tokens. 1.0 = no penalty.
    /// - `seed`: RNG seed for reproducibility.
    pub fn new(temperature: f32, top_p: f32, repetition_penalty: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_p,
            repetition_penalty,
            rng_state: if seed == 0 { 0xdeadbeefcafe1234 } else { seed },
        }
    }

    /// Create a greedy sampler (always picks the highest-probability token).
    pub fn greedy() -> Self {
        Self::new(0.0, 1.0, 1.0, 0)
    }

    /// Sample a token from the logits distribution.
    ///
    /// Applies repetition penalty, temperature scaling, and top-p filtering.
    pub fn sample(&mut self, logits: &mut [f32], recent_tokens: &[u32]) -> u32 {
        // Apply repetition penalty
        if self.repetition_penalty != 1.0 {
            for &token in recent_tokens {
                if (token as usize) < logits.len() {
                    let logit = &mut logits[token as usize];
                    if *logit > 0.0 {
                        *logit /= self.repetition_penalty;
                    } else {
                        *logit *= self.repetition_penalty;
                    }
                }
            }
        }

        // Greedy: just return argmax
        if self.temperature <= 0.0 {
            return Self::argmax(logits);
        }

        // Temperature scaling
        if self.temperature != 1.0 {
            let inv_temp = 1.0 / self.temperature;
            for logit in logits.iter_mut() {
                *logit *= inv_temp;
            }
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for logit in logits.iter_mut() {
                *logit *= inv_sum;
            }
        }

        // Top-p (nucleus) sampling
        if self.top_p < 1.0 {
            self.sample_top_p(logits)
        } else {
            self.sample_categorical(logits)
        }
    }

    /// Argmax: return the index of the largest value.
    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }

    /// Sample from probabilities using top-p filtering.
    fn sample_top_p(&mut self, probs: &mut [f32]) -> u32 {
        // Build sorted (prob, index) pairs — only non-negligible probabilities
        let mut candidates: Vec<(f32, u32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-9)
            .map(|(i, &p)| (p, i as u32))
            .collect();

        // Sort descending by probability
        candidates
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find the cutoff where cumulative probability exceeds top_p
        let mut cumulative = 0.0f32;
        let mut cutoff_idx = candidates.len();
        for (i, &(p, _)) in candidates.iter().enumerate() {
            cumulative += p;
            if cumulative >= self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Truncate to top-p candidates
        candidates.truncate(cutoff_idx);

        // Re-normalize
        let sum: f32 = candidates.iter().map(|(p, _)| p).sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for (p, _) in &mut candidates {
                *p *= inv_sum;
            }
        }

        // Sample from truncated distribution
        let r = self.random_f32();
        let mut cumulative = 0.0f32;
        for &(p, idx) in &candidates {
            cumulative += p;
            if r <= cumulative {
                return idx;
            }
        }

        // Fallback to last candidate
        candidates.last().map(|(_, idx)| *idx).unwrap_or(0)
    }

    /// Sample from a categorical distribution (after softmax).
    fn sample_categorical(&mut self, probs: &[f32]) -> u32 {
        let r = self.random_f32();
        let mut cumulative = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }

    /// Simple xorshift64 PRNG — fast, no external dependency.
    fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        // Convert to [0, 1) float
        (self.rng_state >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut sampler = Sampler::greedy();
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let token = sampler.sample(&mut logits, &[]);
        assert_eq!(token, 1); // index 1 has highest logit
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(Sampler::argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(Sampler::argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn repetition_penalty_reduces_repeated() {
        let mut sampler = Sampler::new(0.0, 1.0, 2.0, 42);
        let mut logits = vec![5.0, 5.0, 1.0];
        // Penalize token 0 → its logit should be reduced
        let token = sampler.sample(&mut logits, &[0]);
        assert_eq!(token, 1); // token 1 should now be preferred
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut sampler = Sampler::new(0.0, 1.0, 1.0, 123);
        let mut logits = vec![1.0, 10.0, 2.0];
        let token = sampler.sample(&mut logits, &[]);
        assert_eq!(token, 1);
    }

    #[test]
    fn sample_with_temperature() {
        let mut sampler = Sampler::new(1.0, 1.0, 1.0, 42);
        // All same logits → should sample somewhat uniformly
        let mut counts = [0u32; 3];
        for _ in 0..300 {
            let mut logits = vec![0.0, 0.0, 0.0];
            let token = sampler.sample(&mut logits, &[]);
            counts[token as usize] += 1;
        }
        // Each should get roughly 100 hits (allow wide margin)
        for (i, &count) in counts.iter().enumerate() {
            assert!(count > 30, "token {i} got only {count} samples out of 300");
        }
    }

    #[test]
    fn top_p_filters() {
        let mut sampler = Sampler::new(1.0, 0.5, 1.0, 42);
        // One token dominates — top-p should mostly pick it
        let mut dominant_count = 0u32;
        for _ in 0..100 {
            let mut logits = vec![10.0, 0.0, 0.0, 0.0];
            let token = sampler.sample(&mut logits, &[]);
            if token == 0 {
                dominant_count += 1;
            }
        }
        assert!(
            dominant_count > 80,
            "dominant token should be picked most often, got {dominant_count}/100"
        );
    }

    #[test]
    fn rng_deterministic() {
        let mut s1 = Sampler::new(1.0, 1.0, 1.0, 42);
        let mut s2 = Sampler::new(1.0, 1.0, 1.0, 42);
        for _ in 0..10 {
            assert_eq!(s1.random_f32(), s2.random_f32());
        }
    }
}
