use crate::manifest::{ModelManifest, Pooling};
use crate::util::mrl_truncate_normalize;
use gp_core::error::{GpError, Result};
use gp_core::traits::Embedder;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Maps a model's declared ONNX input name to tensors we know how to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeedKind {
    InputIds,
    AttentionMask,
    TokenTypeIds,
    PositionIds,
}

fn feed_kind_for_input(name: &str) -> Option<FeedKind> {
    match name {
        "input_ids" => Some(FeedKind::InputIds),
        "attention_mask" => Some(FeedKind::AttentionMask),
        "token_type_ids" | "segment_ids" => Some(FeedKind::TokenTypeIds),
        "position_ids" => Some(FeedKind::PositionIds),
        _ => None,
    }
}

pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dim: usize,
    native_dim: usize,
    model_id: String,
    pooling: Pooling,
    query_instruct: String,
    max_len: usize,
    /// ONNX input names in session order — drives dynamic tensor feeding.
    input_names: Vec<String>,
    max_batch: usize,
}

impl OnnxEmbedder {
    pub fn load(model_dir: &Path, dim: usize, query_instruct: String) -> Result<Self> {
        let cfg = ModelManifest::read(&model_dir.join("manifest.json"))?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| GpError::Model(e.to_string()))?;
        let session = crate::ort_session::open_session(&model_dir.join(&cfg.model_file))?;
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        for name in &input_names {
            if feed_kind_for_input(name).is_none() {
                return Err(GpError::Model(format!(
                    "model {id} requires unsupported ONNX input `{name}`",
                    id = cfg.id
                )));
            }
        }
        let pooling = cfg.pooling_mode();
        let max_batch = cfg.effective_max_batch();
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim,
            native_dim: cfg.native_dim,
            model_id: cfg.id,
            pooling,
            query_instruct,
            max_len: cfg.max_len.min(8192),
            input_names,
            max_batch,
        })
    }

    fn build_input_tensors(
        &self,
        encodings: &[tokenizers::Encoding],
        batch: usize,
        seq: usize,
    ) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>, Array2<i64>)> {
        let mut ids = Array2::<i64>::zeros((batch, seq));
        let mut mask = Array2::<i64>::zeros((batch, seq));
        let mut type_ids = Array2::<i64>::zeros((batch, seq));
        let mut positions = Array2::<i64>::zeros((batch, seq));
        for (i, enc) in encodings.iter().enumerate() {
            for (j, (&id, &m)) in enc
                .get_ids()
                .iter()
                .zip(enc.get_attention_mask().iter())
                .take(seq)
                .enumerate()
            {
                ids[[i, j]] = id as i64;
                mask[[i, j]] = m as i64;
            }
            for (j, &tid) in enc.get_type_ids().iter().take(seq).enumerate() {
                type_ids[[i, j]] = tid as i64;
            }
            for j in 0..seq {
                positions[[i, j]] = j as i64;
            }
        }
        Ok((ids, mask, type_ids, positions))
    }

    fn run_session<'a>(
        &self,
        session: &'a mut Session,
        ids: &Array2<i64>,
        mask: &Array2<i64>,
        type_ids: &Array2<i64>,
        positions: &Array2<i64>,
    ) -> Result<ort::session::SessionOutputs<'a>> {
        let mut id_val = Some(
            Value::from_array(ids.clone()).map_err(|e| GpError::Model(e.to_string()))?,
        );
        let mut mask_val = Some(
            Value::from_array(mask.clone()).map_err(|e| GpError::Model(e.to_string()))?,
        );
        let mut type_val = Some(
            Value::from_array(type_ids.clone()).map_err(|e| GpError::Model(e.to_string()))?,
        );
        let mut pos_val = Some(
            Value::from_array(positions.clone()).map_err(|e| GpError::Model(e.to_string()))?,
        );

        let mut inputs: Vec<(Cow<'_, str>, ort::session::SessionInputValue<'_>)> =
            Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let value = match feed_kind_for_input(name) {
                Some(FeedKind::InputIds) => id_val.take().ok_or_else(|| {
                    GpError::Model("duplicate input_ids in ONNX graph".into())
                })?,
                Some(FeedKind::AttentionMask) => mask_val.take().ok_or_else(|| {
                    GpError::Model("duplicate attention_mask in ONNX graph".into())
                })?,
                Some(FeedKind::TokenTypeIds) => type_val.take().ok_or_else(|| {
                    GpError::Model("duplicate token_type_ids in ONNX graph".into())
                })?,
                Some(FeedKind::PositionIds) => pos_val.take().ok_or_else(|| {
                    GpError::Model("duplicate position_ids in ONNX graph".into())
                })?,
                None => {
                    return Err(GpError::Model(format!(
                        "unsupported ONNX input `{name}`"
                    )))
                }
            };
            inputs.push((Cow::from(name.as_str()), value.into()));
        }
        session
            .run(inputs)
            .map_err(|e| GpError::Model(e.to_string()))
    }

    fn run_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| GpError::Model(e.to_string()))?;

        let batch = encodings.len();
        let seq = encodings
            .iter()
            .map(|e| e.len())
            .max()
            .unwrap_or(0)
            .min(self.max_len);

        let (ids, mask, type_ids, positions) = self.build_input_tensors(&encodings, batch, seq)?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| GpError::Model(e.to_string()))?;
        let outputs = self.run_session(&mut session, &ids, &mask, &type_ids, &positions)?;

        let output = outputs
            .values()
            .next()
            .ok_or_else(|| GpError::Model("missing model output".into()))?;
        let hidden = extract_hidden(&output, batch, seq, self.native_dim)?;
        let hidden_dim = hidden.ncols();
        let already_pooled = hidden.nrows() == batch;

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let pooled = if already_pooled {
                hidden.row(b).to_vec()
            } else {
                self.pool(&hidden, &mask, b, seq, hidden_dim)
            };
            let truncated = mrl_truncate_normalize(&pooled, self.dim);
            out.push(truncated);
        }
        Ok(out)
    }

    fn pool(
        &self,
        hidden: &Array2<f32>,
        mask: &Array2<i64>,
        b: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let base = b * seq;
        match self.pooling {
            Pooling::Cls => hidden.row(base).to_vec(),
            Pooling::Last => {
                let mut last = 0usize;
                for j in 0..seq {
                    if mask[[b, j]] == 1 {
                        last = j;
                    }
                }
                hidden.row(base + last).to_vec()
            }
            Pooling::Mean => {
                let mut acc = vec![0f32; hidden_dim];
                let mut count = 0f32;
                for j in 0..seq {
                    if mask[[b, j]] == 1 {
                        let row = hidden.row(base + j);
                        for d in 0..hidden_dim {
                            acc[d] += row[d];
                        }
                        count += 1.0;
                    }
                }
                if count > 0.0 {
                    for d in 0..hidden_dim {
                        acc[d] /= count;
                    }
                }
                acc
            }
        }
    }
}

fn extract_hidden(
    output: &ort::value::ValueRef<'_>,
    batch: usize,
    seq: usize,
    native_dim: usize,
) -> Result<Array2<f32>> {
    if let Ok((shape, data)) = output.try_extract_tensor::<f32>() {
        return tensor_to_hidden(&shape, data.to_vec(), batch, seq, native_dim);
    }
    if let Ok((shape, data)) = output.try_extract_tensor::<half::f16>() {
        let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        return tensor_to_hidden(&shape, f32_data, batch, seq, native_dim);
    }
    Err(GpError::Model(
        "model output must be f32 or f16 tensor".into(),
    ))
}

fn tensor_to_hidden(
    shape: &[i64],
    data: Vec<f32>,
    batch: usize,
    seq: usize,
    native_dim: usize,
) -> Result<Array2<f32>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    match dims.as_slice() {
        [b, hidden] if *b == batch => {
            Array2::from_shape_vec((batch, *hidden), data).map_err(|e| GpError::Model(e.to_string()))
        }
        [b, s, hidden] if *b == batch => {
            let rows = batch * s;
            Array2::from_shape_vec((rows, *hidden), data).map_err(|e| GpError::Model(e.to_string()))
        }
        [b, s, hidden] => {
            let rows = b * s;
            Array2::from_shape_vec((rows, *hidden), data).map_err(|e| {
                GpError::Model(format!("output shape {dims:?}: {e}"))
            })
        }
        _ if data.len() == batch * seq * native_dim => {
            Array2::from_shape_vec((batch * seq, native_dim), data)
                .map_err(|e| GpError::Model(e.to_string()))
        }
        _ => Err(GpError::Model(format!(
            "unexpected ONNX output shape {dims:?} (batch={batch}, seq={seq}, native_dim={native_dim}, len={})",
            data.len()
        ))),
    }
}

#[cfg(test)]
mod tensor_tests {
    use super::*;

    #[test]
    fn parses_pooled_batch_output() {
        let data = vec![1.0; 1024];
        let hidden = tensor_to_hidden(&[1, 1024], data, 1, 22, 1024).unwrap();
        assert_eq!(hidden.shape(), &[1, 1024]);
    }

    #[test]
    fn parses_sequence_output() {
        let data = vec![1.0; 22 * 1024];
        let hidden = tensor_to_hidden(&[1, 22, 1024], data, 1, 22, 1024).unwrap();
        assert_eq!(hidden.shape(), &[22, 1024]);
    }
}

#[cfg(test)]
mod onnx_live_tests {
    use super::*;
    use gp_core::config::Config;
    use std::path::PathBuf;

    fn harrier_dir() -> Option<PathBuf> {
        let dir = gp_core::config::Config::models_dir().join("harrier-oss-v1-0.6b");
        if dir.join("manifest.json").exists() {
            Some(dir)
        } else {
            None
        }
    }

    fn bge_dir() -> Option<PathBuf> {
        let dir = gp_core::config::Config::models_dir().join("bge-small-en-v1.5");
        if dir.join("manifest.json").exists() {
            Some(dir)
        } else {
            None
        }
    }

    #[test]
    #[ignore = "requires local nomic fp16 install"]
    fn nomic_fp16_live_probe() {
        let dir = gp_core::config::Config::models_dir().join("nomic-embed-text-v1.5-model_fp16");
        if !dir.join("manifest.json").exists() {
            panic!("nomic fp16 not installed");
        }
        let cfg = Config::default();
        let emb = OnnxEmbedder::load(&dir, cfg.embedder.dim, cfg.embedder.query_instruct.clone())
            .expect("load nomic fp16");
        emb.embed_query("test").expect("embed");
    }

    #[test]
    #[ignore = "requires local bge install"]
    fn bge_live_probe() {
        let dir = bge_dir().expect("bge not installed");
        let cfg = Config::default();
        let emb = OnnxEmbedder::load(&dir, cfg.embedder.dim, cfg.embedder.query_instruct.clone())
            .expect("load");
        eprintln!("inputs: {:?}", emb.input_names);
        emb.embed_query("deduct wallet before charging")
            .expect("embed_query");
    }

    #[test]
    #[ignore = "requires local harrier install"]
    fn harrier_live_probe() {
        let dir = harrier_dir().expect("harrier not installed");
        let cfg = Config::default();
        let emb = OnnxEmbedder::load(&dir, cfg.embedder.dim, cfg.embedder.query_instruct.clone())
            .expect("load");
        eprintln!(
            "max_len={} native_dim={} pooling={:?} inputs={:?}",
            emb.max_len, emb.native_dim, emb.pooling, emb.input_names
        );
        let session = emb.session.lock().unwrap();
        for input in session.inputs() {
            eprintln!("input: {}", input.name());
        }
        drop(session);
        emb.embed_query("deduct wallet before charging")
            .expect("embed_query");
        emb.embed(&["fn billing() {}".into()]).expect("embed");
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.max_batch.max(1)) {
            out.extend(self.run_batch(chunk)?);
        }
        Ok(out)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let text = match self.pooling {
            Pooling::Last => format!("Instruct: {}\nQuery:{}", self.query_instruct, query),
            Pooling::Cls | Pooling::Mean => query.to_string(),
        };
        Ok(self.run_batch(&[text])?.pop().unwrap())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}
