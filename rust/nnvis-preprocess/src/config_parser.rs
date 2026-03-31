//! Config parser — ports `extract_config_summary()` and `generate_layer_names()`
//! from extract.py (lines 26–38, 422–479).
//!
//! Reads `<model_dir>/config.json` as generic JSON and produces:
//! - [`ConfigSummary`]: the key architectural fields needed for layer generation
//! - [`LayerDef`]: a list of conceptual transformer layers (embeddings, encoder
//!   blocks, pooler, classifier)

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;

// ─── Public output types ─────────────────────────────────────────────────────

/// Architectural metadata extracted from `config.json`.
#[derive(Debug, Clone, Default)]
pub struct ConfigSummary {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub vocab_size: Option<u32>,
    pub hidden_size: Option<u32>,
    pub num_hidden_layers: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub max_position_embeddings: Option<u32>,
    /// id2label mapping as `(id, label)` pairs, ordered by id.
    pub id2label: Vec<(String, String)>,
}

/// A conceptual transformer layer with positional aliases used to map ONNX
/// node scopes to the layer.
#[derive(Debug, Clone)]
pub struct LayerDef {
    pub id: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub sub_components: Vec<String>,
}

// ─── Entry points ─────────────────────────────────────────────────────────────

/// Read `<model_dir>/config.json` and return a [`ConfigSummary`].
/// Returns a default (all-empty) summary if the file does not exist.
pub fn extract_config_summary(model_dir: &Path) -> Result<ConfigSummary> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Ok(ConfigSummary::default());
    }

    let bytes = std::fs::read(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let root: JsonValue = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing JSON from {}", config_path.display()))?;

    let model_type = root["model_type"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();

    let architectures: Vec<String> = root["architectures"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let vocab_size = root["vocab_size"].as_u64().map(|v| v as u32);
    let hidden_size = root["hidden_size"].as_u64().map(|v| v as u32);
    let num_hidden_layers = root["num_hidden_layers"].as_u64().map(|v| v as u32);
    let num_attention_heads = root["num_attention_heads"].as_u64().map(|v| v as u32);
    let intermediate_size = root["intermediate_size"].as_u64().map(|v| v as u32);
    let max_position_embeddings = root["max_position_embeddings"].as_u64().map(|v| v as u32);

    let id2label: Vec<(String, String)> = if let Some(obj) = root["id2label"].as_object() {
        let mut pairs: Vec<(String, String)> = obj
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();
        // Sort by key so output is deterministic.
        pairs.sort_by(|a, b| {
            a.0.parse::<u64>()
                .unwrap_or(u64::MAX)
                .cmp(&b.0.parse::<u64>().unwrap_or(u64::MAX))
        });
        pairs
    } else {
        vec![]
    };

    Ok(ConfigSummary {
        model_type,
        architectures,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        max_position_embeddings,
        id2label,
    })
}

/// Generate conceptual layer definitions from a [`ConfigSummary`].
///
/// Ports `_module_roots_for_model()` + `generate_layer_names()` from extract.py.
///
/// Layer order:
/// 1. Embeddings
/// 2. N encoder blocks  (`num_hidden_layers`, or 0 if absent)
/// 3. Pooler
/// 4. Classifier
pub fn generate_layer_names(config: &ConfigSummary) -> Vec<LayerDef> {
    let roots = module_roots_for_model(config);
    let primary_root = roots.first().map(|s| s.as_str()).unwrap_or("model");

    let mut layers: Vec<LayerDef> = Vec::new();

    // ── 1. Embeddings ─────────────────────────────────────────────────────────
    let emb_aliases: Vec<String> = roots
        .iter()
        .map(|r| format!("{}.embeddings", r))
        .chain(std::iter::once("embeddings".to_string()))
        .collect();
    layers.push(LayerDef {
        id: format!("{}.embeddings", primary_root),
        aliases: emb_aliases,
        description: "Token Embeddings".to_string(),
        sub_components: vec![],
    });

    // ── 2. Encoder blocks ─────────────────────────────────────────────────────
    let num_hidden_layers = config.num_hidden_layers.unwrap_or(0) as usize;
    for i in 0..num_hidden_layers {
        let mut layer_aliases: Vec<String> = roots
            .iter()
            .map(|r| format!("{}.encoder.layer.{}", r, i))
            .collect();
        // If the primary root is literally "encoder", add the bare form too.
        if primary_root == "encoder" {
            layer_aliases.push(format!("encoder.layer.{}", i));
        }
        layer_aliases.push(format!("layer.{}", i));
        layer_aliases.push(format!("layer_{}", i));

        layers.push(LayerDef {
            id: format!("{}.encoder.layer.{}", primary_root, i),
            aliases: layer_aliases,
            description: format!("Encoder Block {}", i),
            sub_components: vec![
                "attention".to_string(),
                "intermediate".to_string(),
                "output".to_string(),
            ],
        });
    }

    // ── 3. Pooler ─────────────────────────────────────────────────────────────
    layers.push(LayerDef {
        id: "pooler".to_string(),
        aliases: vec![
            "pooler".to_string(),
            format!("{}.pooler", primary_root),
        ],
        description: "Pooler Layer".to_string(),
        sub_components: vec![],
    });

    // ── 4. Classifier ─────────────────────────────────────────────────────────
    layers.push(LayerDef {
        id: "classifier".to_string(),
        aliases: vec![
            "classifier".to_string(),
            "cls".to_string(),
            "classification_head".to_string(),
            "head".to_string(),
        ],
        description: "Classification Head".to_string(),
        sub_components: vec![],
    });

    layers
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Ports `_module_roots_for_model()` from extract.py.
///
/// Chooses a prioritised list of module-name prefixes based on model_type and
/// architectures.  The first entry is the "primary" root.
fn module_roots_for_model(config: &ConfigSummary) -> Vec<String> {
    let model_type_lower = config.model_type.to_lowercase();
    let arch_lower: Vec<String> = config
        .architectures
        .iter()
        .map(|s| s.to_lowercase())
        .collect();

    if model_type_lower.contains("deberta")
        || arch_lower.iter().any(|s| s.contains("deberta"))
    {
        vec![
            "deberta".to_string(),
            "model".to_string(),
            "bert".to_string(),
            "roberta".to_string(),
        ]
    } else {
        vec![
            "model".to_string(),
            "encoder".to_string(),
            "transformer".to_string(),
        ]
    }
}

// ─── Serialisation ────────────────────────────────────────────────────────────

impl ConfigSummary {
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        let id2label: serde_json::Map<String, serde_json::Value> = self
            .id2label
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect();
        json!({
            "model_type": self.model_type,
            "architectures": self.architectures,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "id2label": id2label,
        })
    }
}

impl LayerDef {
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        json!({
            "id": self.id,
            "aliases": self.aliases,
            "description": self.description,
            "sub_components": self.sub_components,
        })
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
    }

    // ── extract_config_summary ─────────────────────────────────────────────

    #[test]
    fn test_config_model_type() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        assert_eq!(cfg.model_type, "deberta-v2");
    }

    #[test]
    fn test_config_architectures() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        assert!(
            cfg.architectures.contains(&"DebertaV2ForSequenceClassification".to_string()),
            "architectures: {:?}",
            cfg.architectures
        );
    }

    #[test]
    fn test_config_numeric_fields() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        assert_eq!(cfg.hidden_size, Some(768));
        assert_eq!(cfg.num_hidden_layers, Some(12));
        assert_eq!(cfg.num_attention_heads, Some(12));
        assert_eq!(cfg.intermediate_size, Some(3072));
        assert_eq!(cfg.max_position_embeddings, Some(512));
        assert_eq!(cfg.vocab_size, Some(251000));
    }

    #[test]
    fn test_config_id2label() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        // The test model has 3 labels: entailment, neutral, contradiction
        assert_eq!(cfg.id2label.len(), 3, "id2label: {:?}", cfg.id2label);
        let labels: Vec<&str> = cfg.id2label.iter().map(|(_, v)| v.as_str()).collect();
        assert!(labels.contains(&"entailment"));
        assert!(labels.contains(&"neutral"));
        assert!(labels.contains(&"contradiction"));
    }

    #[test]
    fn test_missing_config_returns_default() {
        let cfg = extract_config_summary(Path::new("/nonexistent/path"))
            .expect("should return Ok(default)");
        assert_eq!(cfg.model_type, "");
        assert!(cfg.architectures.is_empty());
        assert!(cfg.num_hidden_layers.is_none());
    }

    // ── generate_layer_names ───────────────────────────────────────────────

    #[test]
    fn test_layer_count() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        let layers = generate_layer_names(&cfg);
        let num_hidden = cfg.num_hidden_layers.unwrap_or(0) as usize;
        // embeddings + num_hidden_layers + pooler + classifier
        assert_eq!(
            layers.len(),
            num_hidden + 3,
            "expected {} layers, got {}",
            num_hidden + 3,
            layers.len()
        );
    }

    #[test]
    fn test_layer_ids_deberta() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        let layers = generate_layer_names(&cfg);

        // For deberta-v2 the primary root should be "deberta"
        assert_eq!(layers[0].id, "deberta.embeddings");
        assert_eq!(layers[1].id, "deberta.encoder.layer.0");
        assert_eq!(layers[12].id, "deberta.encoder.layer.11");
        assert_eq!(layers[13].id, "pooler");
        assert_eq!(layers[14].id, "classifier");
    }

    #[test]
    fn test_layer_aliases_contain_primary() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        let layers = generate_layer_names(&cfg);

        // Every layer's `id` should appear in its own alias list (or be the only
        // alias, for pooler/classifier which use different ids).
        for layer in &layers {
            // pooler and classifier use their id directly as the first alias.
            assert!(
                !layer.aliases.is_empty(),
                "layer '{}' has no aliases",
                layer.id
            );
        }
    }

    #[test]
    fn test_encoder_block_sub_components() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        let layers = generate_layer_names(&cfg);

        // Layers 1..=num_hidden_layers are encoder blocks and should have 3 sub-components.
        for layer in layers.iter().skip(1).take(cfg.num_hidden_layers.unwrap_or(0) as usize) {
            assert_eq!(
                layer.sub_components,
                vec!["attention", "intermediate", "output"],
                "layer '{}' sub_components wrong: {:?}",
                layer.id,
                layer.sub_components
            );
        }
    }

    #[test]
    fn test_non_deberta_roots() {
        // Synthesise a GPT-like config summary.
        let cfg = ConfigSummary {
            model_type: "gpt2".to_string(),
            architectures: vec!["GPT2LMHeadModel".to_string()],
            num_hidden_layers: Some(3),
            ..Default::default()
        };
        let layers = generate_layer_names(&cfg);
        // Primary root should be "model" for non-deberta.
        assert_eq!(layers[0].id, "model.embeddings");
        assert_eq!(layers[1].id, "model.encoder.layer.0");
    }

    #[test]
    fn test_json_serialisation() {
        if !model_available() {
            eprintln!("model absent, skip");
            return;
        }
        let cfg = extract_config_summary(&test_model_dir()).expect("parse failed");
        let json = cfg.to_json();
        assert_eq!(json["model_type"], "deberta-v2");
        assert!(json["architectures"].is_array());
        assert_eq!(json["hidden_size"], 768);
        assert_eq!(json["num_hidden_layers"], 12);

        let layers = generate_layer_names(&cfg);
        for layer in &layers {
            let lj = layer.to_json();
            assert!(lj["id"].is_string());
            assert!(lj["aliases"].is_array());
            assert!(lj["description"].is_string());
        }
    }
}
