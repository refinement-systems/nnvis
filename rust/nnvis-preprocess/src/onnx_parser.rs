//! ONNX protobuf parser — ports `extract_onnx_graph()` from extract.py.
//!
//! Reads `<model_dir>/onnx/model.onnx` as raw bytes, decodes the protobuf
//! message with prost-generated types (proto2, so all scalar optional fields
//! are `Option<T>`), and returns a structured [`OnnxBundle`].

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use prost::Message;
use serde_json::{json, Value as JsonValue};

use crate::onnx_proto::onnx::{
    attribute_proto::AttributeType, AttributeProto, ModelProto,
};

// ─── Public output types ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub id: String,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Attribute values serialised to JSON (matches Python's `_attr_to_python`).
    pub attributes: HashMap<String, JsonValue>,
    pub doc_string: String,
    pub metadata_props: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct TensorShape(pub Vec<JsonValue>);

#[derive(Debug, Clone)]
pub struct InitializerMeta {
    pub data_type: i32,
    pub dims: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct ValueInfoMeta {
    pub shape: Option<TensorShape>,
    pub metadata_props: HashMap<String, String>,
}

#[derive(Debug)]
pub struct OnnxBundle {
    pub graph: Vec<OnnxNode>,
    pub graph_metadata: HashMap<String, String>,
    pub model_metadata: ModelMetadata,
    pub initializers: HashMap<String, InitializerMeta>,
    pub value_info: HashMap<String, ValueInfoMeta>,
}

#[derive(Debug)]
pub struct ModelMetadata {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub metadata_props: HashMap<String, String>,
    pub opset_import: Vec<OpsetEntry>,
}

#[derive(Debug)]
pub struct OpsetEntry {
    pub domain: String,
    pub version: i64,
}

// ─── Entry point ─────────────────────────────────────────────────────────────

pub fn extract_onnx_graph(model_dir: &Path) -> Result<OnnxBundle> {
    let onnx_path = model_dir.join("onnx").join("model.onnx");

    if !onnx_path.exists() {
        return Ok(OnnxBundle {
            graph: vec![],
            graph_metadata: HashMap::new(),
            model_metadata: empty_model_metadata(),
            initializers: HashMap::new(),
            value_info: HashMap::new(),
        });
    }

    let bytes =
        std::fs::read(&onnx_path).with_context(|| format!("reading {}", onnx_path.display()))?;

    let model = ModelProto::decode(bytes.as_slice())
        .with_context(|| format!("decoding protobuf from {}", onnx_path.display()))?;

    let graph_proto = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("ModelProto has no graph field"))?;

    // ── Initializers ────────────────────────────────────────────────────────
    let initializers: HashMap<String, InitializerMeta> = graph_proto
        .initializer
        .iter()
        .map(|init| {
            (
                init.name.clone().unwrap_or_default(),
                InitializerMeta {
                    data_type: init.data_type.unwrap_or(0),
                    dims: init.dims.clone(),
                },
            )
        })
        .collect();

    // ── Value info (inputs + outputs + intermediates) ────────────────────────
    let mut value_info: HashMap<String, ValueInfoMeta> = HashMap::new();
    for vi in graph_proto
        .input
        .iter()
        .chain(graph_proto.output.iter())
        .chain(graph_proto.value_info.iter())
    {
        let name = vi.name.clone().unwrap_or_default();
        let shape = tensor_shape_from_type_proto(vi.r#type.as_ref());
        let metadata_props = metadata_props_to_map(&vi.metadata_props);
        value_info.insert(name, ValueInfoMeta { shape, metadata_props });
    }

    // ── Graph nodes ──────────────────────────────────────────────────────────
    let graph: Vec<OnnxNode> = graph_proto
        .node
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let name = node.name.clone().unwrap_or_default();
            let op_type = node.op_type.clone().unwrap_or_default();
            let id = if name.is_empty() {
                format!("{}_{}", op_type, i)
            } else {
                name.clone()
            };

            let attributes: HashMap<String, JsonValue> = node
                .attribute
                .iter()
                .map(|attr| {
                    let attr_name = attr.name.clone().unwrap_or_default();
                    (attr_name, attr_to_json(attr))
                })
                .collect();

            OnnxNode {
                id,
                name,
                op_type,
                domain: node.domain.clone().unwrap_or_default(),
                inputs: node.input.clone(),
                outputs: node.output.clone(),
                attributes,
                doc_string: node.doc_string.clone().unwrap_or_default(),
                metadata_props: metadata_props_to_map(&node.metadata_props),
            }
        })
        .collect();

    // ── Model-level metadata ─────────────────────────────────────────────────
    let model_metadata = ModelMetadata {
        ir_version: model.ir_version.unwrap_or(0),
        producer_name: model.producer_name.clone().unwrap_or_default(),
        producer_version: model.producer_version.clone().unwrap_or_default(),
        domain: model.domain.clone().unwrap_or_default(),
        model_version: model.model_version.unwrap_or(0),
        doc_string: model.doc_string.clone().unwrap_or_default(),
        metadata_props: metadata_props_to_map(&model.metadata_props),
        opset_import: model
            .opset_import
            .iter()
            .map(|op| OpsetEntry {
                domain: op.domain.clone().unwrap_or_default(),
                version: op.version.unwrap_or(0),
            })
            .collect(),
    };

    let graph_metadata = metadata_props_to_map(&graph_proto.metadata_props);

    Ok(OnnxBundle {
        graph,
        graph_metadata,
        model_metadata,
        initializers,
        value_info,
    })
}

// ─── Internal helpers ────────────────────────────────────────────────────────

fn empty_model_metadata() -> ModelMetadata {
    ModelMetadata {
        ir_version: 0,
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        metadata_props: HashMap::new(),
        opset_import: vec![],
    }
}

/// Convert a proto2 repeated `StringStringEntryProto` list to a `HashMap`.
/// Both key and value are optional in proto2 but always present in practice.
fn metadata_props_to_map(
    props: &[crate::onnx_proto::onnx::StringStringEntryProto],
) -> HashMap<String, String> {
    props
        .iter()
        .map(|p| (
            p.key.clone().unwrap_or_default(),
            p.value.clone().unwrap_or_default(),
        ))
        .collect()
}

/// Extract the tensor shape from a `TypeProto`, mirroring
/// `_tensor_shape_from_value_info` in extract.py.
fn tensor_shape_from_type_proto(
    type_proto: Option<&crate::onnx_proto::onnx::TypeProto>,
) -> Option<TensorShape> {
    use crate::onnx_proto::onnx::type_proto::Value as TypeValue;
    use crate::onnx_proto::onnx::tensor_shape_proto::dimension::Value as DimValue;

    let tp = type_proto?;
    let tensor_type = match tp.value.as_ref()? {
        TypeValue::TensorType(tt) => tt,
        _ => return None,
    };

    let shape_proto = tensor_type.shape.as_ref()?;

    let dims: Vec<JsonValue> = shape_proto
        .dim
        .iter()
        .map(|dim| match dim.value.as_ref() {
            Some(DimValue::DimValue(v)) => json!(*v),
            Some(DimValue::DimParam(s)) => json!(s),
            _ => JsonValue::Null,
        })
        .collect();

    Some(TensorShape(dims))
}

/// Convert an `AttributeProto` to a JSON value, mirroring `_attr_to_python`.
/// All scalar fields are `Option<T>` in the proto2-generated code.
fn attr_to_json(attr: &AttributeProto) -> JsonValue {
    // `r#type` is Option<i32>; 0 → Undefined
    let attr_type = attr
        .r#type
        .and_then(|t| AttributeType::try_from(t).ok())
        .unwrap_or(AttributeType::Undefined);

    match attr_type {
        AttributeType::Float   => json!(attr.f.unwrap_or(0.0)),
        AttributeType::Int     => json!(attr.i.unwrap_or(0)),
        AttributeType::String  => {
            let s = attr.s.as_deref()
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_default();
            json!(s)
        }
        AttributeType::Floats  => json!(attr.floats),
        AttributeType::Ints    => json!(attr.ints),
        AttributeType::Strings => {
            let strs: Vec<String> = attr
                .strings
                .iter()
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .collect();
            json!(strs)
        }
        AttributeType::Tensor  => {
            match &attr.t {
                Some(t) => json!({
                    "tensor_name": t.name.clone().unwrap_or_default(),
                    "data_type": t.data_type.unwrap_or(0),
                    "dims": t.dims,
                }),
                None => JsonValue::Null,
            }
        }
        _ => JsonValue::Null,
    }
}

// ─── Serialisation ───────────────────────────────────────────────────────────

impl OnnxNode {
    pub fn to_json(&self) -> JsonValue {
        json!({
            "id": self.id,
            "name": self.name,
            "op_type": self.op_type,
            "domain": self.domain,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
            "doc_string": self.doc_string,
            "metadata_props": self.metadata_props,
        })
    }
}

impl OnnxBundle {
    pub fn to_json(&self) -> JsonValue {
        let graph: Vec<JsonValue> = self.graph.iter().map(|n| n.to_json()).collect();

        let initializers: serde_json::Map<String, JsonValue> = self
            .initializers
            .iter()
            .map(|(k, v)| (k.clone(), json!({ "data_type": v.data_type, "dims": v.dims })))
            .collect();

        let value_info: serde_json::Map<String, JsonValue> = self
            .value_info
            .iter()
            .map(|(k, v)| {
                let shape = v.shape.as_ref().map(|s| json!(s.0)).unwrap_or(JsonValue::Null);
                (k.clone(), json!({ "shape": shape, "metadata_props": v.metadata_props }))
            })
            .collect();

        let opset: Vec<JsonValue> = self
            .model_metadata
            .opset_import
            .iter()
            .map(|op| json!({ "domain": op.domain, "version": op.version }))
            .collect();

        let model_metadata = json!({
            "ir_version": self.model_metadata.ir_version,
            "producer_name": self.model_metadata.producer_name,
            "producer_version": self.model_metadata.producer_version,
            "domain": self.model_metadata.domain,
            "model_version": self.model_metadata.model_version,
            "doc_string": self.model_metadata.doc_string,
            "metadata_props": self.model_metadata.metadata_props,
            "opset_import": opset,
        });

        json!({
            "graph": graph,
            "graph_metadata": self.graph_metadata,
            "model_metadata": model_metadata,
            "initializers": initializers,
            "value_info": value_info,
        })
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_model_dir() -> PathBuf {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        PathBuf::from(manifest)
            .parent().unwrap()  // rust/
            .parent().unwrap()  // repo root
            .join("models")
            .join("MoritzLaurer_mDeBERTa-v3-base-mnli-xnli")
    }

    fn model_available() -> bool {
        test_model_dir().join("onnx").join("model.onnx").exists()
    }

    #[test]
    fn test_node_count_nonzero() {
        if !model_available() { eprintln!("model absent, skip"); return; }
        let bundle = extract_onnx_graph(&test_model_dir()).expect("parse failed");
        assert!(!bundle.graph.is_empty(), "expected nodes, got 0");
        eprintln!("node count: {}", bundle.graph.len());
    }

    #[test]
    fn test_node_ids_and_op_types() {
        if !model_available() { eprintln!("model absent, skip"); return; }
        let bundle = extract_onnx_graph(&test_model_dir()).expect("parse failed");
        for node in &bundle.graph {
            assert!(!node.id.is_empty(), "node has empty id");
            assert!(!node.op_type.is_empty(), "node {} has empty op_type", node.id);
        }
    }

    #[test]
    fn test_known_op_types_present() {
        if !model_available() { eprintln!("model absent, skip"); return; }
        let bundle = extract_onnx_graph(&test_model_dir()).expect("parse failed");
        let op_types: std::collections::HashSet<&str> =
            bundle.graph.iter().map(|n| n.op_type.as_str()).collect();
        for expected in &["MatMul", "Add", "Gather"] {
            assert!(
                op_types.contains(*expected),
                "op_type '{}' not found. Present: {:?}",
                expected, op_types
            );
        }
    }

    #[test]
    fn test_edge_connectivity() {
        if !model_available() { eprintln!("model absent, skip"); return; }
        let bundle = extract_onnx_graph(&test_model_dir()).expect("parse failed");

        let mut producer: HashMap<&str, &str> = HashMap::new();
        for node in &bundle.graph {
            for out in &node.outputs {
                if !out.is_empty() { producer.insert(out.as_str(), node.id.as_str()); }
            }
        }

        let internal_edges: usize = bundle.graph.iter()
            .flat_map(|n| &n.inputs)
            .filter(|inp| !inp.is_empty() && producer.contains_key(inp.as_str()))
            .count();

        assert!(internal_edges > 0, "Expected internal edges, found 0");
        eprintln!("internal edge count: {}", internal_edges);
    }

    #[test]
    fn test_json_serialisation() {
        if !model_available() { eprintln!("model absent, skip"); return; }
        let bundle = extract_onnx_graph(&test_model_dir()).expect("parse failed");
        let json = bundle.to_json();
        assert!(json["graph"].is_array());
        assert!(!json["graph"].as_array().unwrap().is_empty());
        assert!(json["model_metadata"]["ir_version"].is_number());
        assert!(json["initializers"].is_object());
        assert!(json["value_info"].is_object());
    }

    #[test]
    fn test_missing_model_returns_empty() {
        let bundle = extract_onnx_graph(Path::new("/nonexistent/path"))
            .expect("should return Ok(empty bundle)");
        assert!(bundle.graph.is_empty());
        assert!(bundle.initializers.is_empty());
    }
}
