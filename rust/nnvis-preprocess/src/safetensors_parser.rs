use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use serde_json::Value;

pub struct SafeTensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
}

pub fn extract_safetensors_metadata(model_dir: &Path) -> Result<Vec<SafeTensorMeta>> {
    let sf_path = model_dir.join("model.safetensors");
    if !sf_path.exists() {
        return Ok(Vec::new());
    }

    let mut f = File::open(&sf_path).context("failed to open safetensors file")?;
    
    // Safetensors format: 8 bytes (uint64 little endian) for JSON header size
    let mut header_size_bytes = [0u8; 8];
    f.read_exact(&mut header_size_bytes).context("failed to read 8-byte header size")?;
    
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;
    
    // Read the JSON header
    let mut header_bytes = vec![0u8; header_size];
    f.read_exact(&mut header_bytes).context("failed to read safetensors JSON header")?;
    
    let header: HashMap<String, Value> = serde_json::from_slice(&header_bytes)
        .context("failed to parse safetensors JSON header")?;

    let mut tensors = Vec::new();
    for (key, value) in header {
        if key == "__metadata__" {
            continue;
        }

        let dtype = value.get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let shape = value.get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|n| n.as_i64())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(Vec::new);

        tensors.push(SafeTensorMeta {
            name: key,
            dtype,
            shape,
        });
    }

    // Sort to ensure deterministic output
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_safetensors_metadata() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_dir = manifest_dir.parent().unwrap().parent().unwrap().join("models").join("MoritzLaurer_mDeBERTa-v3-base-mnli-xnli");
        
        if !model_dir.exists() {
            println!("Skipping test: model directory {:?} does not exist", model_dir);
            return;
        }

        let tensors = extract_safetensors_metadata(&model_dir).expect("failed to extract safetensors");
        assert!(!tensors.is_empty(), "expected some tensors");

        // Verify some specific expected tensors
        let embeddings = tensors.iter().find(|t| t.name == "deberta.embeddings.word_embeddings.weight");
        assert!(embeddings.is_some(), "expected word embeddings tensor");
        
        let emb = embeddings.unwrap();
        assert_eq!(emb.dtype, "F16", "expected dtype F16 for embeddings");
        assert_eq!(emb.shape, vec![251000, 768], "expected specific shape for word embeddings");
        
        let metadata_tensor = tensors.iter().find(|t| t.name == "__metadata__");
        assert!(metadata_tensor.is_none(), "should not have parsed __metadata__ node");
    }
}
