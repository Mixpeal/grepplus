use std::collections::HashMap;

pub struct Bm25 {
    postings: HashMap<String, Vec<(u32, u32)>>,
    doc_len: Vec<u32>,
    avg_len: f32,
    n_docs: usize,
    k1: f32,
    b: f32,
}

impl Bm25 {
    pub fn build(chunk_texts: &[String]) -> Self {
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut doc_len = Vec::with_capacity(chunk_texts.len());
        for (id, text) in chunk_texts.iter().enumerate() {
            let mut tf: HashMap<String, u32> = HashMap::new();
            let toks = tokenize(text);
            doc_len.push(toks.len() as u32);
            for t in toks {
                *tf.entry(t).or_default() += 1;
            }
            for (term, f) in tf {
                postings.entry(term).or_default().push((id as u32, f));
            }
        }
        let total: u64 = doc_len.iter().map(|&x| x as u64).sum();
        let n = chunk_texts.len();
        Self {
            postings,
            doc_len,
            avg_len: if n > 0 { total as f32 / n as f32 } else { 0.0 },
            n_docs: n,
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Return top-k (chunk_global_id, score), restricted to `candidates` if Some.
    pub fn search(&self, query: &str, k: usize, candidates: Option<&[u32]>) -> Vec<(u32, f32)> {
        let cand_set: Option<std::collections::HashSet<u32>> =
            candidates.map(|c| c.iter().copied().collect());
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for term in tokenize(query) {
            if let Some(plist) = self.postings.get(&term) {
                let df = plist.len() as f32;
                let idf = ((self.n_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
                for &(doc, tf) in plist {
                    if let Some(set) = &cand_set {
                        if !set.contains(&doc) {
                            continue;
                        }
                    }
                    let dl = self.doc_len[doc as usize] as f32;
                    let denom = tf as f32 + self.k1 * (1.0 - self.b + self.b * dl / self.avg_len);
                    let s = idf * (tf as f32 * (self.k1 + 1.0)) / denom;
                    *scores.entry(doc).or_default() += s;
                }
            }
        }
        let mut v: Vec<(u32, f32)> = scores.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v.truncate(k);
        v
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_ranks_matching_doc_higher() {
        let texts = vec![
            "circuit breaker fallback handler".into(),
            "unrelated database migration".into(),
        ];
        let idx = Bm25::build(&texts);
        let hits = idx.search("circuit breaker", 2, None);
        assert_eq!(hits[0].0, 0);
    }
}
