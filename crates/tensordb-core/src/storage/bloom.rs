use crate::native_bridge::Hasher;

#[derive(Debug, Clone)]
pub struct BloomFilter {
    pub m_bits: u32,
    pub k_hashes: u8,
    pub bits: Vec<u8>,
}

impl BloomFilter {
    pub fn new_for_keys(keys: &[Vec<u8>], bits_per_key: usize, hasher: &dyn Hasher) -> Self {
        let key_count = keys.len().max(1);
        let m_bits = (key_count * bits_per_key).max(64) as u32;
        let m_bytes = m_bits.div_ceil(8) as usize;
        let mut bits = vec![0u8; m_bytes];
        let k_hashes = ((bits_per_key as f64 * 0.69).round() as u8).clamp(1, 12);

        for key in keys {
            let h1 = hasher.hash64(key);
            let h2 = hasher.hash64(&h1.to_le_bytes());
            for i in 0..k_hashes {
                let bit = (h1.wrapping_add((i as u64).wrapping_mul(h2)) % m_bits as u64) as usize;
                bits[bit / 8] |= 1u8 << (bit % 8);
            }
        }

        Self {
            m_bits,
            k_hashes,
            bits,
        }
    }

    pub fn may_contain(&self, key: &[u8], hasher: &dyn Hasher) -> bool {
        let h1 = hasher.hash64(key);
        let h2 = hasher.hash64(&h1.to_le_bytes());

        for i in 0..self.k_hashes {
            let bit = (h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.m_bits as u64) as usize;
            if (self.bits[bit / 8] & (1u8 << (bit % 8))) == 0 {
                return false;
            }
        }
        true
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(9 + self.bits.len());
        out.extend_from_slice(&self.m_bits.to_le_bytes());
        out.push(self.k_hashes);
        out.extend_from_slice(&(self.bits.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.bits);
        out
    }

    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 9 {
            return None;
        }
        let mut m_bits = [0u8; 4];
        m_bits.copy_from_slice(&data[0..4]);
        let k_hashes = data[4];
        let mut lenb = [0u8; 4];
        lenb.copy_from_slice(&data[5..9]);
        let len = u32::from_le_bytes(lenb) as usize;
        if 9 + len > data.len() {
            return None;
        }
        Some(Self {
            m_bits: u32::from_le_bytes(m_bits),
            k_hashes,
            bits: data[9..9 + len].to_vec(),
        })
    }
}
