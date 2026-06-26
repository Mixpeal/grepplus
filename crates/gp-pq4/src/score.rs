use gp_core::traits::Q4Code;

/// Asymmetric dot product: full-precision query vs dequantized 4-bit code.
/// Both query and (pre-projection) doc are L2-normalized, so dot ≈ cosine.
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
    use crate::quant::quantize_q4;

    #[test]
    fn dot_with_self_is_positive() {
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let code = quantize_q4(&v);
        let score = asym_dot(&v, &code);
        assert!(score > 0.0);
    }
}
