use gp_core::traits::Q4Code;

/// Symmetric per-vector 4-bit quantization.
/// Maps each component to 0..=15 via affine (scale,bias). Packs 2 per byte.
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

/// Dequantize back to f32 (used for scoring + tests).
pub fn dequantize_q4(code: &Q4Code) -> Vec<f32> {
    let mut out = Vec::with_capacity(code.dim as usize);
    for i in 0..code.dim as usize {
        let byte = code.bytes[i / 2];
        let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        out.push(code.bias + (nib as f32) * code.scale);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_approximate() {
        let v = vec![0.1, -0.5, 0.9, 0.0, 1.0];
        let code = quantize_q4(&v);
        let back = dequantize_q4(&code);
        assert_eq!(back.len(), v.len());
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.2);
        }
    }
}
