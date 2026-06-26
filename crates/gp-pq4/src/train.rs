use crate::learned::LearnedQ4;
use crate::{pca, quant, score};
use gp_core::error::{GpError, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Triplet {
    pub anchor: Vec<f32>,
    pub positive: Vec<f32>,
    pub negative: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub out_dim: usize,
    pub epochs: usize,
    pub lr: f32,
    pub margin: f32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            out_dim: 64,
            epochs: 40,
            lr: 0.05,
            margin: 0.1,
            seed: 0x9E3779B9_7F4A7C15,
        }
    }
}

/// Train learned PQ4 projection with triplet margin loss (straight-through Q4).
pub fn train_learned_pq4(triplets: &[Triplet], cfg: &TrainConfig) -> Result<LearnedQ4> {
    if triplets.is_empty() {
        return Err(GpError::Training("no triplets".into()));
    }
    let in_dim = triplets[0].anchor.len();
    if triplets.iter().any(|t| t.anchor.len() != in_dim) {
        return Err(GpError::Training("inconsistent embedding dims".into()));
    }
    let out_dim = cfg.out_dim.min(in_dim);

    let samples: Vec<Vec<f32>> = triplets
        .iter()
        .flat_map(|t| [&t.anchor, &t.positive, &t.negative])
        .map(|v| v.to_vec())
        .collect();
    let (mean, flat) = pca::fit_pca(&samples, out_dim);
    let mut matrix = flat;
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);

    for _ in 0..cfg.epochs {
        let mut order: Vec<usize> = (0..triplets.len()).collect();
        order.shuffle(&mut rng);
        for &idx in &order {
            let t = &triplets[idx];
            let grad = triplet_grad(&matrix, &mean, in_dim, out_dim, t, cfg.margin);
            for (w, g) in matrix.iter_mut().zip(grad.iter()) {
                *w -= cfg.lr * g;
            }
        }
    }

    Ok(LearnedQ4::from_parts(matrix, mean, out_dim, in_dim))
}

fn triplet_grad(
    matrix: &[f32],
    mean: &[f32],
    in_dim: usize,
    out_dim: usize,
    t: &Triplet,
    margin: f32,
) -> Vec<f32> {
    let qa = project(matrix, mean, in_dim, out_dim, &t.anchor);
    let qp = project(matrix, mean, in_dim, out_dim, &t.positive);
    let qn = project(matrix, mean, in_dim, out_dim, &t.negative);

    let cp = quant::quantize_q4(&qp);
    let cn = quant::quantize_q4(&qn);
    let sp = score::asym_dot(&qa, &cp);
    let sn = score::asym_dot(&qa, &cn);

    if sp + margin > sn {
        return vec![0.0; matrix.len()];
    }

    let mut grad = vec![0f32; matrix.len()];
    for o in 0..out_dim {
        for i in 0..in_dim {
            let idx = o * in_dim + i;
            grad[idx] += (t.negative[i] - mean[i]) - (t.positive[i] - mean[i]);
        }
    }
    grad
}

fn project(matrix: &[f32], mean: &[f32], in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; out_dim];
    for o in 0..out_dim {
        let row = &matrix[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0f32;
        for i in 0..in_dim {
            acc += row[i] * (x[i] - mean[i]);
        }
        y[o] = acc;
    }
    pca::normalize(&mut y);
    y
}

pub fn save_learned(path: &Path, model: &LearnedQ4) -> Result<()> {
    let stored = model.to_stored();
    std::fs::write(path, serde_json::to_string_pretty(&stored)?)
        .map_err(|e| GpError::Index(e.to_string()))?;
    Ok(())
}

pub fn load_learned(path: &Path) -> Result<LearnedQ4> {
    let raw = std::fs::read_to_string(path).map_err(|e| GpError::Index(e.to_string()))?;
    let stored = serde_json::from_str(&raw)?;
    Ok(LearnedQ4::from_stored(&stored))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: usize) -> Vec<f32> {
        let mut out = vec![0.0; 8];
        out[v % 8] = 1.0;
        out
    }

    #[test]
    fn empty_triplets_error() {
        assert!(train_learned_pq4(&[], &TrainConfig::default()).is_err());
    }

    #[test]
    fn training_improves_triplet_order() {
        let triplets = vec![
            Triplet {
                anchor: unit(0),
                positive: unit(1),
                negative: unit(7),
            },
            Triplet {
                anchor: unit(2),
                positive: unit(3),
                negative: unit(6),
            },
        ];
        let model = train_learned_pq4(&triplets, &TrainConfig::default()).expect("train");
        let backend: &dyn gp_core::traits::ProjectionBackend = &model;
        let q = unit(0);
        let cp = backend.project(&unit(1));
        let cn = backend.project(&unit(7));
        assert!(backend.score(&q, &cp) >= backend.score(&q, &cn));
    }
}
