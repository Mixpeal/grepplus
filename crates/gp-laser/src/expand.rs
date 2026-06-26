/// Static synonym dictionary, loaded from embedded expand.toml.
pub struct Expander {
    map: std::collections::HashMap<String, Vec<String>>,
}

impl Expander {
    pub fn builtin() -> Self {
        let raw = include_str!("../expand.toml");
        let parsed: std::collections::HashMap<String, Vec<String>> =
            toml::from_str(raw).expect("valid expand.toml");
        Self { map: parsed }
    }

    /// Expand a query into a set of grep terms:
    /// 1. raw tokens
    /// 2. synonym expansions
    /// 3. identifier splits (camelCase / snake_case)
    pub fn expand(&self, query: &str) -> Vec<String> {
        let mut terms = std::collections::BTreeSet::new();
        for tok in tokenize(query) {
            terms.insert(tok.clone());
            if let Some(syns) = self.map.get(&tok.to_lowercase()) {
                for s in syns {
                    terms.insert(s.clone());
                }
            }
            for part in split_identifier(&tok) {
                if part.len() >= 3 {
                    terms.insert(part);
                }
            }
        }
        terms.into_iter().collect()
    }
}

/// "handleSessionRefresh" -> ["handle","Session","Refresh"]
/// "session_refresh"      -> ["session","refresh"]
pub fn split_identifier(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for ch in s.chars() {
        if ch == '_' || ch == '-' || ch == '.' {
            if !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
            continue;
        }
        if ch.is_uppercase() && prev_lower {
            if !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
        }
        cur.push(ch.to_ascii_lowercase());
        prev_lower = ch.is_lowercase();
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

fn tokenize(q: &str) -> Vec<String> {
    q.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_camel_case() {
        let parts = split_identifier("handleSessionRefresh");
        assert!(parts.contains(&"handle".to_string()));
        assert!(parts.contains(&"session".to_string()));
        assert!(parts.contains(&"refresh".to_string()));
    }

    #[test]
    fn expands_retry_synonyms() {
        let e = Expander::builtin();
        let terms = e.expand("retry logic");
        assert!(terms.iter().any(|t| t == "backoff"));
    }
}
