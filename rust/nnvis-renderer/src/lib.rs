//! nnvis-renderer — WASM library that deserialises .nnvis files and exposes
//! node/edge data to JavaScript via wasm-bindgen.
//!
//! This is a skeleton.  Full layout computation will be implemented in
//! issue #9 (force-directed layout) and issue #10 (JS bindings).

// Only pull in wasm-bindgen when compiling for wasm32.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use nnvis_core::decode_nnvis;

/// Parse an `.nnvis` byte payload and return the number of ONNX graph nodes
/// it contains.  This is the initial smoke-test binding that proves the
/// wasm-bindgen / nnvis-core integration works before the full renderer
/// (issue #10) is wired up.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn node_count(data: &[u8]) -> u32 {
    let payload = match decode_nnvis(data) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let bundle = match nnvis_core::root_as_model_bundle(payload) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    bundle.graph_nodes().map(|v| v.len() as u32).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatbuffers::FlatBufferBuilder;
    use nnvis_core::{
        encode_nnvis, finish_model_bundle_buffer, ModelBundle, ModelBundleArgs, OnnxNode,
        OnnxNodeArgs,
    };

    #[test]
    fn node_count_empty_bundle() {
        let mut fbb = FlatBufferBuilder::with_capacity(64);
        let bundle = ModelBundle::create(&mut fbb, &ModelBundleArgs::default());
        finish_model_bundle_buffer(&mut fbb, bundle);
        let bytes = encode_nnvis(fbb.finished_data());
        assert_eq!(node_count(&bytes), 0);
    }

    #[test]
    fn node_count_returns_correct_value() {
        let mut fbb = FlatBufferBuilder::with_capacity(256);

        let id0 = fbb.create_string("node_0");
        let n0 = OnnxNode::create(&mut fbb, &OnnxNodeArgs { id: Some(id0), ..Default::default() });
        let id1 = fbb.create_string("node_1");
        let n1 = OnnxNode::create(&mut fbb, &OnnxNodeArgs { id: Some(id1), ..Default::default() });
        let nodes = fbb.create_vector(&[n0, n1]);

        let bundle = ModelBundle::create(
            &mut fbb,
            &ModelBundleArgs {
                graph_nodes: Some(nodes),
                ..Default::default()
            },
        );
        finish_model_bundle_buffer(&mut fbb, bundle);
        let bytes = encode_nnvis(fbb.finished_data());
        assert_eq!(node_count(&bytes), 2);
    }
}
