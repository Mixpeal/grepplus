//! Shared query classification for routing and lexical fast-paths.

/// True when the query is a symbol/identifier that should use exact grep (-F).
pub fn is_literal_query(query: &str) -> bool {
    let q = query.trim();
    if q.is_empty() {
        return false;
    }
    if has_natural_language_cue(q) {
        return false;
    }
    let stripped = strip_quotes(q);
    if stripped.contains(' ') {
        return false;
    }
    is_identifier_token(stripped)
}

/// True when the query should bypass embedding and route to grep.
pub fn is_grep_route_query(query: &str) -> bool {
    let q = query.trim();
    if q.is_empty() {
        return false;
    }
    if has_regex_metacharacters(q) {
        return true;
    }
    if is_quoted(q) {
        return true;
    }
    is_literal_query(q)
}

pub fn strip_quotes(q: &str) -> &str {
    let q = q.trim();
    if (q.starts_with('"') && q.ends_with('"') && q.len() >= 2)
        || (q.starts_with('\'') && q.ends_with('\'') && q.len() >= 2)
    {
        &q[1..q.len() - 1]
    } else {
        q
    }
}

pub fn is_quoted(q: &str) -> bool {
    let q = q.trim();
    (q.starts_with('"') && q.ends_with('"') && q.len() >= 2)
        || (q.starts_with('\'') && q.ends_with('\'') && q.len() >= 2)
}

pub fn is_identifier_token(token: &str) -> bool {
    if token.len() < 2 {
        return false;
    }
    if has_camel_case(token) {
        return true;
    }
    if token.contains('_')
        && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && token.len() >= 3
    {
        return true;
    }
    let first = token.chars().next().unwrap_or('\0');
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if !token
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
    {
        return false;
    }
    token.len() >= 3
}

pub fn has_regex_metacharacters(q: &str) -> bool {
    q.contains(".*") || q.contains('[') || q.contains('(') || q.contains('|') || q.contains('\\')
}

pub fn has_camel_case(token: &str) -> bool {
    let has_upper = token.chars().any(|c| c.is_uppercase());
    let has_lower = token.chars().any(|c| c.is_lowercase());
    has_upper && has_lower
}

pub fn has_natural_language_cue(q: &str) -> bool {
    let lower = q.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    const CUES: &[&str] = &[
        "where",
        "what",
        "how",
        "find",
        "logic",
        "when",
        "why",
        "which",
        "explain",
        "show",
        "does",
        "code",
        "implementation",
    ];
    if words.len() >= 2 {
        return CUES.iter().any(|w| words.contains(w));
    }
    // Single-token queries are identifiers unless they are clearly NL question words.
    matches!(
        words.first(),
        Some(&"where" | &"what" | &"how" | &"why" | &"which")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_symbols() {
        assert!(is_literal_query("paymentWebhook"));
        assert!(is_literal_query("authGuard"));
        assert!(is_literal_query("verify_webhook_signature"));
        assert!(!is_literal_query("where is retry logic"));
    }

    #[test]
    fn grep_route_includes_regex() {
        assert!(is_grep_route_query("foo.*bar"));
        assert!(is_grep_route_query("handleSessionRefresh"));
    }
}
