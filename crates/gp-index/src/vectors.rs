//! Compact vector storage: 4-bit quantization + asymmetric dot scoring.

use gp_core::traits::Q4Code;

/// Encode and score embedding vectors stored in the index.
#[derive(Debug, Clone)]
pub struct VectorCodec {
    pub dim: usize,
}

impl VectorCodec {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn project(&self, vec: &[f32]) -> Q4Code {
        let d = self.dim.min(vec.len());
        quantize_q4(&vec[..d])
    }

    pub fn score(&self, query: &[f32], code: &Q4Code) -> f32 {
        asym_dot(&query[..code.dim as usize], code)
    }
}

/// Symmetric per-vector 4-bit quantization (2 nibbles per byte).
pub fn quantize_q4(vec: &[f32]) -> Q4Code {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in vec {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let range = (hi - lo).max(1e-8);
    let scale = range / 15.0;
    let bias = lo;

    let mut bytes = vec![0u8; (vec.len() + 1) / 2];
    for (i, &x) in vec.iter().enumerate() {
        let q = (((x - bias) / scale).round().clamp(0.0, 15.0)) as u8;
        if i % 2 == 0 {
            bytes[i / 2] = q;
        } else {
            bytes[i / 2] |= q << 4;
        }
    }
    Q4Code {
        bytes,
        dim: vec.len() as u16,
        scale,
        bias,
    }
}

/// Asymmetric dot product: full-precision query vs dequantized 4-bit code.
#[inline]
pub fn asym_dot(query: &[f32], code: &Q4Code) -> f32 {
    debug_assert_eq!(query.len(), code.dim as usize);
    let mut acc = 0f32;
    for i in 0..code.dim as usize {
        let byte = code.bytes[i / 2];
        let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let dq = code.bias + (nib as f32) * code.scale;
        acc += query[i] * dq;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_projects_and_scores() {
        let codec = VectorCodec::new(4);
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let code = codec.project(&v);
        let s = codec.score(&v[..4], &code);
        assert!(s.is_finite());
    }
}
