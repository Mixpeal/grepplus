use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub const NUM_HASHES: usize = 64;

pub struct MinHasher {
    a: [u64; NUM_HASHES],
    b: [u64; NUM_HASHES],
    prime: u64,
}

impl MinHasher {
    pub fn new() -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_C0DE);
        let mut a = [0u64; NUM_HASHES];
        let mut b = [0u64; NUM_HASHES];
        for i in 0..NUM_HASHES {
            a[i] = rng.gen::<u64>() | 1;
            b[i] = rng.gen::<u64>();
        }
        Self {
            a,
            b,
            prime: (1u64 << 61) - 1,
        }
    }

    /// Compute MinHash signature over trigram token shingles of `text`.
    pub fn signature(&self, text: &str) -> [u32; NUM_HASHES] {
        let shingles = shingles(text);
        let mut sig = [u32::MAX; NUM_HASHES];
        for sh in shingles {
            let h = fxhash(sh.as_bytes());
            for k in 0..NUM_HASHES {
                let v = ((self.a[k].wrapping_mul(h).wrapping_add(self.b[k])) % self.prime) as u32;
                if v < sig[k] {
                    sig[k] = v;
                }
            }
        }
        sig
    }
}

impl Default for MinHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimated Jaccard similarity between two signatures.
pub fn jaccard(a: &[u32; NUM_HASHES], b: &[u32; NUM_HASHES]) -> f32 {
    let mut eq = 0;
    for k in 0..NUM_HASHES {
        if a[k] == b[k] {
            eq += 1;
        }
    }
    eq as f32 / NUM_HASHES as f32
}

fn shingles(text: &str) -> Vec<String> {
    let toks: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    if toks.len() < 3 {
        return toks.iter().map(|t| t.to_lowercase()).collect();
    }
    toks.windows(3)
        .map(|w| w.join(" ").to_lowercase())
        .collect()
}

fn fxhash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_texts_high_jaccard() {
        let mh = MinHasher::new();
        let a = mh.signature("fn main() { println!(\"hello\"); }");
        let b = mh.signature("fn main() { println!(\"hello\"); }");
        assert!(jaccard(&a, &b) > 0.9);
    }
}
