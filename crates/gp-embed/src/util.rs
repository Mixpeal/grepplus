pub fn mrl_truncate_normalize(v: &[f32], dim: usize) -> Vec<f32> {
    let d = dim.min(v.len());
    let mut out = v[..d].to_vec();
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in &mut out {
            *x /= norm;
        }
    }
    out
}
