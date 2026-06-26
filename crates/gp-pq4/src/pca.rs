use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

const PQ4_SEED: u64 = 0x9E3779B97F4A7C15;

/// Returns (mean[in_dim], components[out_dim * in_dim] row-major).
pub fn fit_pca(samples: &[Vec<f32>], out_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n = samples.len();
    let d = samples[0].len();
    assert!(n > 0 && d > 0);

    let mut mean = vec![0f32; d];
    for s in samples {
        for i in 0..d {
            mean[i] += s[i];
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }

    let centered: Vec<Vec<f32>> = samples
        .iter()
        .map(|s| s.iter().zip(&mean).map(|(x, m)| x - m).collect())
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(PQ4_SEED);
    let mut components = Vec::<Vec<f32>>::new();

    for _ in 0..out_dim {
        let mut v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        normalize(&mut v);
        for _iter in 0..100 {
            let mut w = vec![0f32; d];
            for x in &centered {
                let dot: f32 = x.iter().zip(&v).map(|(a, b)| a * b).sum();
                for i in 0..d {
                    w[i] += x[i] * dot;
                }
            }
            for wi in &mut w {
                *wi /= n as f32;
            }
            for comp in &components {
                let proj: f32 = w.iter().zip(comp).map(|(a, b)| a * b).sum();
                for i in 0..d {
                    w[i] -= proj * comp[i];
                }
            }
            normalize(&mut w);
            let delta: f32 = w.iter().zip(&v).map(|(a, b)| (a - b).abs()).sum();
            v = w;
            if delta < 1e-6 {
                break;
            }
        }
        components.push(v);
    }
    let flat: Vec<f32> = components.into_iter().flatten().collect();
    (mean, flat)
}

pub fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_produces_components() {
        let samples: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 * 0.1, (i as f32 * 0.2).sin(), 1.0])
            .collect();
        let (mean, flat) = fit_pca(&samples, 2);
        assert_eq!(mean.len(), 3);
        assert_eq!(flat.len(), 6);
    }
}
