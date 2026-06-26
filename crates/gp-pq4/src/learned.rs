use crate::{pca, quant, score};
use gp_core::traits::{ProjectionBackend, Q4Code};
use serde::{Deserialize, Serialize};

/// Track 2: margin-trained linear projection + Q4 (learned on AgentCode triplets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredLearnedQ4 {
    pub matrix: Vec<f32>,
    pub mean: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

#[derive(Debug)]
pub struct LearnedQ4 {
    matrix: Vec<f32>,
    mean: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl LearnedQ4 {
    pub fn from_parts(matrix: Vec<f32>, mean: Vec<f32>, out_dim: usize, in_dim: usize) -> Self {
        Self {
            matrix,
            mean,
            out_dim,
            in_dim,
        }
    }

    pub fn from_stored(s: &StoredLearnedQ4) -> Self {
        Self::from_parts(s.matrix.clone(), s.mean.clone(), s.out_dim, s.in_dim)
    }

    pub fn to_stored(&self) -> StoredLearnedQ4 {
        StoredLearnedQ4 {
            matrix: self.matrix.clone(),
            mean: self.mean.clone(),
            out_dim: self.out_dim,
            in_dim: self.in_dim,
        }
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

impl ProjectionBackend for LearnedQ4 {
    fn project(&self, vec: &[f32]) -> Q4Code {
        let y = self.project_f32(vec);
        quant::quantize_q4(&y)
    }

    fn score(&self, query: &[f32], code: &Q4Code) -> f32 {
        let qproj = self.project_f32(query);
        score::asym_dot(&qproj, code)
    }

    fn id(&self) -> &str {
        "learned-pq4"
    }
}
