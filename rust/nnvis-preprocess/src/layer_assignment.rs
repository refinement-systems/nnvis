//! Layer assignment — ports `assign_nodes_to_layers()` and its helpers from
//! extract.py (lines 482–619).
//!
//! For each ONNX node the algorithm tries, in order:
//!  1. **Metadata-scope matching**: look for a torch name-scope in the node's
//!     `metadata_props`, parse it, and match it against layer aliases using the
//!     longest-prefix-wins rule.
//!  2. **Text heuristic matching**: try the node name, inputs, and outputs
//!     against the same prefix matcher and a set of keyword heuristics.
//!  3. **Unassigned fallback**: if nothing matched, assign to `"unassigned"`.

use std::collections::HashMap;

use regex::Regex;

use crate::config_parser::LayerDef;
use crate::onnx_parser::OnnxNode;

// ─── Public output types ─────────────────────────────────────────────────────

/// The result of assigning a single ONNX node to a conceptual layer.
#[derive(Debug, Clone)]
pub struct NodeAssignment {
    pub layer_id: String,
    /// Canonical scope string extracted from node metadata (may be `None`).
    pub scope: Option<String>,
    /// The alias or keyword that caused the match (may be `None` for unassigned).
    pub matched_prefix: Option<String>,
    /// How the match was made (see `MatchedVia` constants below).
    pub matched_via: String,
}

/// String constants for the `matched_via` field — mirrors extract.py values.
pub mod matched_via {
    pub const METADATA_SCOPE: &str = "metadata-scope";
    pub const PREFIX_TEXT: &str = "prefix-text";
    pub const HEURISTIC_EMBEDDINGS: &str = "heuristic-embeddings";
    pub const HEURISTIC_POOLER: &str = "heuristic-pooler";
    pub const HEURISTIC_CLASSIFIER: &str = "heuristic-classifier";
    pub const HEURISTIC_LAYER_INDEX: &str = "heuristic-layer-index";
    pub const UNMATCHED: &str = "unmatched";
}

/// Result returned by `assign_nodes_to_layers`.
pub struct AssignmentResult {
    pub assignments: HashMap<String, NodeAssignment>,
    pub unmatched_nodes: Vec<String>,
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Assign every node in `onnx_nodes` to a conceptual layer from `layer_defs`.
///
/// Ports `assign_nodes_to_layers()` from extract.py (lines 581–619).
pub fn assign_nodes_to_layers(
    onnx_nodes: &[OnnxNode],
    layer_defs: &[LayerDef],
) -> AssignmentResult {
    // Pre-compile the layer-index regex once for the whole batch.
    let layer_idx_re =
        Regex::new(r"(?:^|[./_])layer[./_](\d+)(?:$|[./_])").expect("valid regex");

    let mut assignments: HashMap<String, NodeAssignment> = HashMap::new();
    let mut unmatched_nodes: Vec<String> = Vec::new();

    for node in onnx_nodes {
        let scope = canonical_scope(node);
        let mut matched: Option<MatchResult> = None;

        // ── Step 1: metadata-scope match ─────────────────────────────────────
        if let Some(ref scope_str) = scope {
            if let Some(m) = match_layer_by_prefix(&scope_str.to_lowercase(), layer_defs) {
                matched = Some(MatchResult {
                    layer_id: m.layer_id,
                    matched_prefix: Some(m.matched_prefix),
                    matched_via: matched_via::METADATA_SCOPE.to_string(),
                });
            }
        }

        // ── Step 2: text heuristic match ─────────────────────────────────────
        if matched.is_none() {
            // Try node name first, then inputs, then outputs — same order as Python.
            let texts = std::iter::once(node.name.as_str())
                .chain(node.inputs.iter().map(|s| s.as_str()))
                .chain(node.outputs.iter().map(|s| s.as_str()));

            for text in texts {
                if let Some(m) = guess_layer_from_text(text, layer_defs, &layer_idx_re) {
                    matched = Some(m);
                    break;
                }
            }
        }

        // ── Step 3: unassigned fallback ───────────────────────────────────────
        let assignment = match matched {
            Some(m) => NodeAssignment {
                layer_id: m.layer_id,
                scope,
                matched_prefix: m.matched_prefix,
                matched_via: m.matched_via,
            },
            None => {
                unmatched_nodes.push(node.id.clone());
                NodeAssignment {
                    layer_id: "unassigned".to_string(),
                    scope,
                    matched_prefix: None,
                    matched_via: matched_via::UNMATCHED.to_string(),
                }
            }
        };

        assignments.insert(node.id.clone(), assignment);
    }

    AssignmentResult {
        assignments,
        unmatched_nodes,
    }
}

// ─── Internal types ────────────────────────────────────────────────────────

/// Intermediate result from a prefix or heuristic match.
struct MatchResult {
    layer_id: String,
    matched_prefix: Option<String>,
    matched_via: String,
}

/// Returned by `match_layer_by_prefix` when a match is found.
struct PrefixMatch {
    layer_id: String,
    matched_prefix: String,
}

// ─── Core helpers ─────────────────────────────────────────────────────────────

/// Extract the canonical scope string from a node's metadata_props.
///
/// Ports `_canonical_scope()` (extract.py lines 509–522).
fn canonical_scope(node: &OnnxNode) -> Option<String> {
    const SCOPE_KEYS: &[&str] = &[
        "pkg.torch.onnx.name_scopes",
        "pkg.torch.onnx.scope",
        "namespace",
        "scope",
    ];

    for &key in SCOPE_KEYS {
        if let Some(raw) = node.metadata_props.get(key) {
            if raw.is_empty() {
                continue;
            }
            let parts = parse_name_scopes(raw);
            if !parts.is_empty() {
                let joined = parts.join(".");
                let trimmed = joined.trim_matches('.').to_string();
                if !trimmed.is_empty() {
                    return Some(trimmed);
                }
            }
        }
    }
    None
}

/// Parse a raw scope string into individual name parts.
///
/// Ports `_parse_name_scopes()` (extract.py lines 482–506).
///
/// Strategy (in order):
/// 1. Try JSON — if the result is a list, use its items; if a string, continue
///    with that string.
/// 2. Split on `>`, `/`, or `::` (first separator found).
/// 3. Split on `.`.
/// 4. Return the whole string as a single-element list.
fn parse_name_scopes(raw: &str) -> Vec<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return vec![];
    }

    // 1. Try JSON parse.
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw) {
        if let Some(arr) = val.as_array() {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|v| {
                    let s = v.as_str().unwrap_or("").trim().to_string();
                    if s.is_empty() { None } else { Some(s) }
                })
                .collect();
            if !parts.is_empty() {
                return parts;
            }
        }
        // JSON string — unwrap and proceed with the inner string.
        if let Some(inner) = val.as_str() {
            let inner = inner.trim().to_string();
            if inner.is_empty() {
                return vec![];
            }
            return parse_name_scopes_raw(&inner);
        }
    }

    parse_name_scopes_raw(raw)
}

/// Split a (non-JSON) scope string by separator heuristics.
fn parse_name_scopes_raw(raw: &str) -> Vec<String> {
    for sep in &[">", "/", "::"] {
        if raw.contains(sep) {
            return raw
                .split(sep)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    if raw.contains('.') {
        return raw
            .split('.')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    vec![raw.to_string()]
}

/// Longest-prefix-wins layer lookup.
///
/// Ports `_match_layer_by_prefix()` (extract.py lines 525–544).
///
/// `text` should already be lowercase before calling this.
fn match_layer_by_prefix(text: &str, layer_defs: &[LayerDef]) -> Option<PrefixMatch> {
    let normalized = text.trim_matches('.');
    if normalized.is_empty() {
        return None;
    }

    let mut best: Option<(usize, String, String)> = None; // (len, layer_id, alias)

    for layer in layer_defs {
        for alias in &layer.aliases {
            let alias_norm = alias.trim_matches('.');
            if alias_norm.is_empty() {
                continue;
            }
            if normalized == alias_norm || normalized.starts_with(&format!("{}.", alias_norm)) {
                let len = alias_norm.len();
                match &best {
                    Some((best_len, _, _)) if len <= *best_len => {}
                    _ => {
                        best = Some((len, layer.id.clone(), alias_norm.to_string()));
                    }
                }
            }
        }
    }

    best.map(|(_, layer_id, matched_prefix)| PrefixMatch {
        layer_id,
        matched_prefix,
    })
}

/// Try all heuristics to match a single text string to a layer.
///
/// Ports `_guess_layer_from_text()` (extract.py lines 547–578).
fn guess_layer_from_text(
    text: &str,
    layer_defs: &[LayerDef],
    layer_idx_re: &Regex,
) -> Option<MatchResult> {
    if text.is_empty() {
        return None;
    }

    let lowered = text.to_lowercase();

    // 1. Prefix match (case-insensitive).
    if let Some(m) = match_layer_by_prefix(&lowered, layer_defs) {
        return Some(MatchResult {
            layer_id: m.layer_id,
            matched_prefix: Some(m.matched_prefix),
            matched_via: matched_via::PREFIX_TEXT.to_string(),
        });
    }

    // 2. Keyword heuristics.
    if lowered.contains("embed") {
        if let Some(layer) = layer_defs.iter().find(|l| l.id.ends_with(".embeddings")) {
            return Some(MatchResult {
                layer_id: layer.id.clone(),
                matched_prefix: Some("embed".to_string()),
                matched_via: matched_via::HEURISTIC_EMBEDDINGS.to_string(),
            });
        }
    }

    if lowered.contains("pool") {
        return Some(MatchResult {
            layer_id: "pooler".to_string(),
            matched_prefix: Some("pool".to_string()),
            matched_via: matched_via::HEURISTIC_POOLER.to_string(),
        });
    }

    if ["classif", "logits", "label", "pred"]
        .iter()
        .any(|kw| lowered.contains(kw))
    {
        return Some(MatchResult {
            layer_id: "classifier".to_string(),
            matched_prefix: Some("classif".to_string()),
            matched_via: matched_via::HEURISTIC_CLASSIFIER.to_string(),
        });
    }

    // 3. Encoder-block index regex: `layer[./_]{digits}`.
    if let Some(cap) = layer_idx_re.captures(&lowered) {
        let layer_index = &cap[1];
        let target_suffix = format!(".encoder.layer.{}", layer_index);
        if let Some(layer) = layer_defs.iter().find(|l| l.id.ends_with(&target_suffix)) {
            return Some(MatchResult {
                layer_id: layer.id.clone(),
                matched_prefix: Some(format!("layer.{}", layer_index)),
                matched_via: matched_via::HEURISTIC_LAYER_INDEX.to_string(),
            });
        }
    }

    None
}

// ─── Serialisation ────────────────────────────────────────────────────────────

impl NodeAssignment {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "layer_id": self.layer_id,
            "scope": self.scope,
            "matched_prefix": self.matched_prefix,
            "matched_via": self.matched_via,
        })
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn test_model_dir() -> PathBuf {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        PathBuf::from(manifest)
            .parent()
            .unwrap() // rust/
            .parent()
            .unwrap() // repo root
            .join("models")
            .join("MoritzLaurer_mDeBERTa-v3-base-mnli-xnli")
    }

    fn model_available() -> bool {
        test_model_dir().join("config.json").exists()
            && test_model_dir().join("onnx").join("model.onnx").exists()
    }

    /// Load both the parsed graph and layer defs from the test model.
    fn load_test_data() -> (Vec<OnnxNode>, Vec<LayerDef>) {
        let dir = test_model_dir();
        let config = crate::config_parser::extract_config_summary(&dir).unwrap();
        let layer_defs = crate::config_parser::generate_layer_names(&config);
        let bundle = crate::onnx_parser::extract_onnx_graph(&dir).unwrap();
        (bundle.graph, layer_defs)
    }

    // ── parse_name_scopes ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_name_scopes_json_array() {
        let result = parse_name_scopes(r#"["deberta", "encoder", "layer"]"#);
        assert_eq!(result, vec!["deberta", "encoder", "layer"]);
    }

    #[test]
    fn test_parse_name_scopes_json_string() {
        let result = parse_name_scopes(r#""deberta.encoder.layer""#);
        assert_eq!(result, vec!["deberta", "encoder", "layer"]);
    }

    #[test]
    fn test_parse_name_scopes_gt_separator() {
        let result = parse_name_scopes("deberta>encoder>layer");
        assert_eq!(result, vec!["deberta", "encoder", "layer"]);
    }

    #[test]
    fn test_parse_name_scopes_slash_separator() {
        let result = parse_name_scopes("deberta/encoder/layer");
        assert_eq!(result, vec!["deberta", "encoder", "layer"]);
    }

    #[test]
    fn test_parse_name_scopes_doublecolon_separator() {
        let result = parse_name_scopes("deberta::encoder::layer");
        assert_eq!(result, vec!["deberta", "encoder", "layer"]);
    }

    #[test]
    fn test_parse_name_scopes_dot_separator() {
        let result = parse_name_scopes("deberta.encoder.layer");
        assert_eq!(result, vec!["deberta", "encoder", "layer"]);
    }

    #[test]
    fn test_parse_name_scopes_single_word() {
        let result = parse_name_scopes("classifier");
        assert_eq!(result, vec!["classifier"]);
    }

    #[test]
    fn test_parse_name_scopes_empty() {
        assert!(parse_name_scopes("").is_empty());
        assert!(parse_name_scopes("   ").is_empty());
    }

    #[test]
    fn test_parse_name_scopes_json_array_empty_items_filtered() {
        let result = parse_name_scopes(r#"["deberta", "", "layer"]"#);
        assert_eq!(result, vec!["deberta", "layer"]);
    }

    // ── match_layer_by_prefix ──────────────────────────────────────────────────

    fn make_simple_layers() -> Vec<LayerDef> {
        vec![
            LayerDef {
                id: "deberta.embeddings".to_string(),
                aliases: vec![
                    "deberta.embeddings".to_string(),
                    "model.embeddings".to_string(),
                    "embeddings".to_string(),
                ],
                description: "Token Embeddings".to_string(),
                sub_components: vec![],
            },
            LayerDef {
                id: "deberta.encoder.layer.0".to_string(),
                aliases: vec![
                    "deberta.encoder.layer.0".to_string(),
                    "layer.0".to_string(),
                    "layer_0".to_string(),
                ],
                description: "Encoder Block 0".to_string(),
                sub_components: vec!["attention".to_string()],
            },
            LayerDef {
                id: "pooler".to_string(),
                aliases: vec!["pooler".to_string(), "deberta.pooler".to_string()],
                description: "Pooler Layer".to_string(),
                sub_components: vec![],
            },
            LayerDef {
                id: "classifier".to_string(),
                aliases: vec![
                    "classifier".to_string(),
                    "cls".to_string(),
                    "classification_head".to_string(),
                ],
                description: "Classification Head".to_string(),
                sub_components: vec![],
            },
        ]
    }

    #[test]
    fn test_prefix_match_exact() {
        let layers = make_simple_layers();
        let m = match_layer_by_prefix("deberta.embeddings", &layers).unwrap();
        assert_eq!(m.layer_id, "deberta.embeddings");
        assert_eq!(m.matched_prefix, "deberta.embeddings");
    }

    #[test]
    fn test_prefix_match_subpath() {
        let layers = make_simple_layers();
        let m =
            match_layer_by_prefix("deberta.embeddings.word_embeddings", &layers).unwrap();
        assert_eq!(m.layer_id, "deberta.embeddings");
    }

    #[test]
    fn test_prefix_match_longest_wins() {
        // "deberta.embeddings" is longer than "embeddings" — should win.
        let layers = make_simple_layers();
        let m = match_layer_by_prefix("deberta.embeddings.weight", &layers).unwrap();
        assert_eq!(m.matched_prefix, "deberta.embeddings");
    }

    #[test]
    fn test_prefix_match_no_match() {
        let layers = make_simple_layers();
        assert!(match_layer_by_prefix("unknown.stuff", &layers).is_none());
    }

    #[test]
    fn test_prefix_match_empty() {
        let layers = make_simple_layers();
        assert!(match_layer_by_prefix("", &layers).is_none());
        assert!(match_layer_by_prefix(".", &layers).is_none());
    }

    // ── guess_layer_from_text ──────────────────────────────────────────────────

    fn re() -> Regex {
        Regex::new(r"(?:^|[./_])layer[./_](\d+)(?:$|[./_])").unwrap()
    }

    #[test]
    fn test_guess_layer_prefix_text() {
        let layers = make_simple_layers();
        let m = guess_layer_from_text("deberta.embeddings.weight", &layers, &re()).unwrap();
        assert_eq!(m.matched_via, matched_via::PREFIX_TEXT);
        assert_eq!(m.layer_id, "deberta.embeddings");
    }

    #[test]
    fn test_guess_layer_embed_heuristic() {
        let layers = make_simple_layers();
        let m = guess_layer_from_text("token_embeddings_output", &layers, &re()).unwrap();
        assert_eq!(m.matched_via, matched_via::HEURISTIC_EMBEDDINGS);
        assert_eq!(m.layer_id, "deberta.embeddings");
    }

    #[test]
    fn test_guess_layer_pool_heuristic() {
        let layers = make_simple_layers();
        let m = guess_layer_from_text("some_pooler_output", &layers, &re()).unwrap();
        assert_eq!(m.matched_via, matched_via::HEURISTIC_POOLER);
        assert_eq!(m.layer_id, "pooler");
    }

    #[test]
    fn test_guess_layer_classifier_heuristic() {
        let layers = make_simple_layers();
        // Use strings that contain the keywords but don't match any alias by prefix.
        for text in &["logits_output", "pred_label", "label_ids"] {
            let m = guess_layer_from_text(text, &layers, &re()).unwrap();
            assert_eq!(m.matched_via, matched_via::HEURISTIC_CLASSIFIER, "text={}", text);
        }
    }

    #[test]
    fn test_guess_layer_index_heuristic() {
        let layers = make_simple_layers();
        // Use strings that contain layer/0 but don't prefix-match any alias directly.
        for text in &["/layer/0/output", "encoder_layer_0_out"] {
            let m = guess_layer_from_text(text, &layers, &re()).unwrap();
            assert_eq!(
                m.matched_via,
                matched_via::HEURISTIC_LAYER_INDEX,
                "text={}",
                text
            );
            assert_eq!(m.layer_id, "deberta.encoder.layer.0");
        }
    }

    #[test]
    fn test_guess_layer_no_match() {
        let layers = make_simple_layers();
        assert!(guess_layer_from_text("random_op_xyz", &layers, &re()).is_none());
        assert!(guess_layer_from_text("", &layers, &re()).is_none());
    }

    // ── assign_nodes_to_layers (integration) ─────────────────────────────────

    #[test]
    fn test_assign_match_rate() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let (nodes, layer_defs) = load_test_data();
        let total = nodes.len();
        let result = assign_nodes_to_layers(&nodes, &layer_defs);

        let unmatched = result.unmatched_nodes.len();
        let match_rate = (total - unmatched) as f64 / total as f64 * 100.0;

        eprintln!(
            "total={}, unmatched={}, match_rate={:.1}%",
            total, unmatched, match_rate
        );

        // Python produces 83 unmatched out of 4113 (98.0%).
        assert_eq!(
            unmatched, 83,
            "expected 83 unmatched (like Python), got {}",
            unmatched
        );
    }

    #[test]
    fn test_assign_via_breakdown() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let (nodes, layer_defs) = load_test_data();
        let result = assign_nodes_to_layers(&nodes, &layer_defs);

        let mut via_counts: HashMap<&str, usize> = HashMap::new();
        for assignment in result.assignments.values() {
            *via_counts.entry(assignment.matched_via.as_str()).or_insert(0) += 1;
        }

        eprintln!("via breakdown: {:?}", via_counts);

        // Mirror the Python breakdown exactly.
        assert_eq!(via_counts.get(matched_via::HEURISTIC_LAYER_INDEX).copied().unwrap_or(0), 4000);
        assert_eq!(via_counts.get(matched_via::HEURISTIC_EMBEDDINGS).copied().unwrap_or(0), 19);
        assert_eq!(via_counts.get(matched_via::HEURISTIC_POOLER).copied().unwrap_or(0), 10);
        assert_eq!(via_counts.get(matched_via::HEURISTIC_CLASSIFIER).copied().unwrap_or(0), 1);
        assert_eq!(via_counts.get(matched_via::UNMATCHED).copied().unwrap_or(0), 83);
    }

    #[test]
    fn test_assign_all_nodes_present() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let (nodes, layer_defs) = load_test_data();
        let node_ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        let result = assign_nodes_to_layers(&nodes, &layer_defs);

        // Every node should appear in assignments exactly once.
        for id in &node_ids {
            assert!(
                result.assignments.contains_key(*id),
                "node '{}' missing from assignments",
                id
            );
        }
        assert_eq!(result.assignments.len(), nodes.len());
    }

    #[test]
    fn test_assign_encoder_block_distribution() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let (nodes, layer_defs) = load_test_data();
        let result = assign_nodes_to_layers(&nodes, &layer_defs);

        // With 12 encoder blocks and 4000 layer-index hits,
        // each block should get exactly 4000/12 ≈ 333 nodes, and all 12 are
        // present.
        let mut per_block: HashMap<String, usize> = HashMap::new();
        for (_, assignment) in &result.assignments {
            if assignment
                .layer_id
                .contains(".encoder.layer.")
            {
                *per_block.entry(assignment.layer_id.clone()).or_insert(0) += 1;
            }
        }
        assert_eq!(per_block.len(), 12, "expected 12 encoder blocks with nodes, got {}", per_block.len());
        // 4000 / 12 nodes should all be accounted for.
        let total_encoder: usize = per_block.values().sum();
        assert_eq!(total_encoder, 4000);
    }

    #[test]
    fn test_assign_empty_input() {
        let result = assign_nodes_to_layers(&[], &[]);
        assert!(result.assignments.is_empty());
        assert!(result.unmatched_nodes.is_empty());
    }
}
