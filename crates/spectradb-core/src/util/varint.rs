use crate::error::{Result, SpectraError};

pub fn encode_u64(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

pub fn decode_u64(input: &[u8], idx: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    while *idx < input.len() && shift < 64 {
        let b = input[*idx];
        *idx += 1;
        value |= ((b & 0x7f) as u64)
            .checked_shl(shift)
            .ok_or(SpectraError::InvalidVarint)?;
        if (b & 0x80) == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    Err(SpectraError::InvalidVarint)
}

pub fn encode_bytes(bytes: &[u8], out: &mut Vec<u8>) {
    encode_u64(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

pub fn decode_bytes(input: &[u8], idx: &mut usize) -> Result<Vec<u8>> {
    let len = decode_u64(input, idx)? as usize;
    if *idx + len > input.len() {
        return Err(SpectraError::InvalidVarint);
    }
    let out = input[*idx..*idx + len].to_vec();
    *idx += len;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        let nums = [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX - 1];
        for n in nums {
            let mut b = Vec::new();
            encode_u64(n, &mut b);
            let mut i = 0;
            let got = decode_u64(&b, &mut i).unwrap();
            assert_eq!(got, n);
            assert_eq!(i, b.len());
        }
    }
}
