use crate::{pca, quant, score};
use gp_core::traits::{ProjectionBackend, Q4Code};

/// Stage A: identity/MRL slice projection + Q4. The ablation baseline.
pub struct BaselineQ4 {
    pub proj_dim: usize,
}

impl ProjectionBackend for BaselineQ4 {
    fn project(&self, vec: &[f32]) -> Q4Code {
        let d = self.proj_dim.min(vec.len());
        let sliced = &vec[..d];
        quant::quantize_q4(sliced)
    }
    fn score(&self, query: &[f32], code: &Q4Code) -> f32 {
        score::asym_dot(&query[..code.dim as usize], code)
    }
    fn id(&self) -> &str {
        "baseline-q4"
    }
}

/// Stage A.5: PCA projection (linear matrix) + Q4.
pub struct PcaQ4 {
    matrix: Vec<f32>,
    mean: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl PcaQ4 {
    pub fn fit(samples: &[Vec<f32>], out_dim: usize) -> Self {
        let in_dim = samples[0].len();
        let (mean, flat) = pca::fit_pca(samples, out_dim);
        Self {
            matrix: flat,
            mean,
            out_dim,
            in_dim,
        }
    }

    pub fn from_parts(matrix: Vec<f32>, mean: Vec<f32>, out_dim: usize, in_dim: usize) -> Self {
        Self {
            matrix,
            mean,
            out_dim,
            in_dim,
        }
    }

    pub fn matrix(&self) -> &[f32] {
        &self.matrix
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    fn project_f32(&self, vec: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; self.out_dim];
        for o in 0..self.out_dim {
            let row = &self.matrix[o * self.in_dim..(o + 1) * self.in_dim];
            let mut acc = 0f32;
            for i in 0..self.in_dim {
                acc += row[i] * (vec[i] - self.mean[i]);
            }
            y[o] = acc;
        }
        pca::normalize(&mut y);
        y
    }
}

impl ProjectionBackend for PcaQ4 {
    fn project(&self, vec: &[f32]) -> Q4Code {
        let y = self.project_f32(vec);
        quant::quantize_q4(&y)
    }
    fn score(&self, query: &[f32], code: &Q4Code) -> f32 {
        let qproj = self.project_f32(query);
        score::asym_dot(&qproj, code)
    }
    fn id(&self) -> &str {
        "pca-q4"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_q4_projects_and_scores() {
        let backend = BaselineQ4 { proj_dim: 4 };
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let code = backend.project(&v);
        let s = backend.score(&v[..4], &code);
        assert!(s.is_finite());
    }
}
