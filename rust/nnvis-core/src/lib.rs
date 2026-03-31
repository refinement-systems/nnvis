//! nnvis-core — shared types and binary format for the .nnvis file.
//!
//! The FlatBuffers-generated code lives in `generated::model_bundle_generated`.
//! Consumers should import through the public re-exports in this module.

// Include the generated FlatBuffers accessor code.
#[allow(
    clippy::all,
    warnings,
    unused,
    non_snake_case,
    non_camel_case_types,
    dead_code
)]
pub mod generated {
    include!("generated/model_bundle_generated.rs");
}

pub use generated::nnvis::{
    ConfigSummary, ConfigSummaryArgs, ConfigSummaryBuilder,
    Edge, EdgeArgs, EdgeBuilder,
    GroupNode, GroupNodeArgs, GroupNodeBuilder,
    LayerDef, LayerDefArgs, LayerDefBuilder,
    ModelBundle, ModelBundleArgs, ModelBundleBuilder,
    NodeAssignment, NodeAssignmentArgs, NodeAssignmentBuilder,
    OnnxNode, OnnxNodeArgs, OnnxNodeBuilder,
    TensorMeta, TensorMetaArgs, TensorMetaBuilder,
    root_as_model_bundle, root_as_model_bundle_with_opts,
    finish_model_bundle_buffer, finish_size_prefixed_model_bundle_buffer,
};

/// Magic bytes written at the start of every `.nnvis` file.
pub const MAGIC: &[u8; 4] = b"NNVS";
/// Format version encoded as two little-endian bytes.
pub const VERSION: u16 = 1;

/// Write the 6-byte header (`NNVS` magic + version) followed by the raw
/// FlatBuffer bytes and return the complete `.nnvis` payload.
pub fn encode_nnvis(flatbuffer_bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(6 + flatbuffer_bytes.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(flatbuffer_bytes);
    out
}

/// Strip and validate the 6-byte header, returning a slice into the
/// FlatBuffer payload.  Returns `Err` on magic mismatch or version
/// incompatibility.
pub fn decode_nnvis(data: &[u8]) -> Result<&[u8], &'static str> {
    if data.len() < 6 {
        return Err("file too short to contain .nnvis header");
    }
    if &data[0..4] != MAGIC {
        return Err("invalid magic bytes — not a .nnvis file");
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != VERSION {
        return Err("unsupported .nnvis version");
    }
    Ok(&data[6..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatbuffers::FlatBufferBuilder;

    /// Build a minimal ModelBundle, round-trip it through encode/decode, and
    /// verify that the accessor returns the values we wrote.
    #[test]
    fn roundtrip_model_bundle() {
        let mut fbb = FlatBufferBuilder::with_capacity(1024);

        // --- ConfigSummary ---
        let model_type = fbb.create_string("deberta");
        let archs_vec = {
            let a = fbb.create_string("DebertaForSequenceClassification");
            fbb.create_vector(&[a])
        };
        let config = ConfigSummary::create(
            &mut fbb,
            &ConfigSummaryArgs {
                model_type: Some(model_type),
                architectures: Some(archs_vec),
                vocab_size: 250002,
                hidden_size: 768,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                intermediate_size: 3072,
                max_position_embeddings: 512,
            },
        );

        // --- TensorMeta ---
        let t_name = fbb.create_string("embeddings.word_embeddings.weight");
        let t_dtype = fbb.create_string("F32");
        let shape_data: Vec<i64> = vec![250002, 768];
        let shape_vec = fbb.create_vector(&shape_data);
        let tensor = TensorMeta::create(
            &mut fbb,
            &TensorMetaArgs {
                name: Some(t_name),
                dtype: Some(t_dtype),
                shape: Some(shape_vec),
            },
        );
        let tensors_vec = fbb.create_vector(&[tensor]);

        // --- OnnxNode ---
        let n_id = fbb.create_string("node_0");
        let n_op = fbb.create_string("MatMul");
        let node = OnnxNode::create(
            &mut fbb,
            &OnnxNodeArgs {
                id: Some(n_id),
                op_type: Some(n_op),
                ..Default::default()
            },
        );
        let nodes_vec = fbb.create_vector(&[node]);

        // --- Edge ---
        let e_src = fbb.create_string("node_0");
        let e_tgt = fbb.create_string("node_1");
        let edge = Edge::create(
            &mut fbb,
            &EdgeArgs {
                source: Some(e_src),
                target: Some(e_tgt),
            },
        );
        let edges_vec = fbb.create_vector(&[edge]);

        // --- ModelBundle ---
        let bundle = ModelBundle::create(
            &mut fbb,
            &ModelBundleArgs {
                config: Some(config),
                tensors: Some(tensors_vec),
                graph_nodes: Some(nodes_vec),
                group_edges: Some(edges_vec),
                ..Default::default()
            },
        );
        finish_model_bundle_buffer(&mut fbb, bundle);
        let fb_bytes = fbb.finished_data();

        // Encode to .nnvis container.
        let nnvis_bytes = encode_nnvis(fb_bytes);

        // Validate header.
        assert_eq!(&nnvis_bytes[0..4], b"NNVS");
        assert_eq!(u16::from_le_bytes([nnvis_bytes[4], nnvis_bytes[5]]), 1);

        // Decode and verify.
        let payload = decode_nnvis(&nnvis_bytes).expect("decode_nnvis failed");
        let decoded = root_as_model_bundle(payload).expect("invalid FlatBuffer");

        let cfg = decoded.config().expect("missing config");
        assert_eq!(cfg.model_type(), Some("deberta"));
        assert_eq!(cfg.hidden_size(), 768);
        assert_eq!(cfg.num_hidden_layers(), 12);

        let tensors = decoded.tensors().expect("missing tensors");
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors.get(0).name(), Some("embeddings.word_embeddings.weight"));
        assert_eq!(tensors.get(0).dtype(), Some("F32"));
        let shape = tensors.get(0).shape().expect("missing shape");
        assert_eq!(shape.get(0), 250002i64);
        assert_eq!(shape.get(1), 768i64);

        let nodes = decoded.graph_nodes().expect("missing nodes");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes.get(0).id(), Some("node_0"));
        assert_eq!(nodes.get(0).op_type(), Some("MatMul"));

        let edges = decoded.group_edges().expect("missing edges");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges.get(0).source(), Some("node_0"));
        assert_eq!(edges.get(0).target(), Some("node_1"));
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let bad = b"BAD!0000";
        assert!(decode_nnvis(bad).is_err());
    }

    #[test]
    fn decode_rejects_short_input() {
        assert!(decode_nnvis(b"NNV").is_err());
    }
}
