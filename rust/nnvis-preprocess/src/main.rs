// nnvis-preprocess — native CLI that reads model files and writes .nnvis
//
// Issue #3: ONNX protobuf parsing      — done
// Issue #5: config.json + layer names  — done
// Remaining work: #4 (SafeTensors), #6 (layer assignment), #7 (grouped graph).

mod onnx_proto;
mod onnx_parser;
mod config_parser;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

/// Extract a HuggingFace model into the compact .nnvis binary format.
#[derive(Parser, Debug)]
#[command(name = "nnvis-preprocess", version, about, long_about = None)]
struct Cli {
    /// Path to the model directory (must contain model.safetensors and
    /// onnx/model.onnx).
    model_dir: PathBuf,

    /// Output path for the generated .nnvis file.
    /// Defaults to <model_dir>/model.nnvis.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Dump the parsed ONNX graph as JSON to stdout (for debugging).
    #[arg(long)]
    dump_json: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let output = cli
        .output
        .unwrap_or_else(|| cli.model_dir.join("model.nnvis"));

    eprintln!("nnvis-preprocess: model_dir = {}", cli.model_dir.display());
    eprintln!("nnvis-preprocess: output    = {}", output.display());

    // ── Issue #5: Parse config.json and generate layer definitions ────────────
    eprintln!("nnvis-preprocess: parsing config.json …");
    let config = config_parser::extract_config_summary(&cli.model_dir)
        .context("config.json parsing failed")?;
    let layer_names = config_parser::generate_layer_names(&config);

    eprintln!(
        "nnvis-preprocess: model_type = {:?}, num_hidden_layers = {:?}, layers = {}",
        config.model_type,
        config.num_hidden_layers,
        layer_names.len(),
    );

    // ── Issue #3: Parse the ONNX graph ──────────────────────────────────────
    eprintln!("nnvis-preprocess: parsing ONNX graph …");
    let onnx_bundle = onnx_parser::extract_onnx_graph(&cli.model_dir)
        .context("ONNX parsing failed")?;

    eprintln!(
        "nnvis-preprocess: found {} nodes, {} initializers, {} value_info entries",
        onnx_bundle.graph.len(),
        onnx_bundle.initializers.len(),
        onnx_bundle.value_info.len(),
    );

    if cli.dump_json {
        let layers_json: Vec<serde_json::Value> =
            layer_names.iter().map(|l| l.to_json()).collect();
        let mut json = onnx_bundle.to_json();
        json["config"] = config.to_json();
        json["layer_names"] = serde_json::Value::Array(layers_json);
        println!("{}", serde_json::to_string_pretty(&json)?);
        return Ok(());
    }

    // ── Write placeholder .nnvis (full FlatBuffer encoding in issues #6/#7) ─
    use flatbuffers::FlatBufferBuilder;
    use nnvis_core::{encode_nnvis, finish_model_bundle_buffer, ModelBundle, ModelBundleArgs,
                     OnnxNode as FbOnnxNode, OnnxNodeArgs};

    let mut fbb = FlatBufferBuilder::with_capacity(64 * 1024);

    // Encode the parsed graph nodes into FlatBuffers.
    let fb_nodes: Vec<_> = onnx_bundle.graph.iter().map(|n| {
        let id    = fbb.create_string(&n.id);
        let op    = fbb.create_string(&n.op_type);
        let name  = fbb.create_string(&n.name);
        let attrs = fbb.create_string(&serde_json::to_string(&n.attributes).unwrap_or_default());
        let inputs_fb: Vec<_>  = n.inputs.iter().map(|s| fbb.create_string(s)).collect();
        let outputs_fb: Vec<_> = n.outputs.iter().map(|s| fbb.create_string(s)).collect();
        let inp_vec = fbb.create_vector(&inputs_fb);
        let out_vec = fbb.create_vector(&outputs_fb);
        FbOnnxNode::create(&mut fbb, &OnnxNodeArgs {
            id: Some(id),
            op_type: Some(op),
            name: Some(name),
            attributes_json: Some(attrs),
            inputs: Some(inp_vec),
            outputs: Some(out_vec),
            ..Default::default()
        })
    }).collect();

    let nodes_vec = fbb.create_vector(&fb_nodes);

    let bundle = ModelBundle::create(&mut fbb, &ModelBundleArgs {
        graph_nodes: Some(nodes_vec),
        ..Default::default()
    });
    finish_model_bundle_buffer(&mut fbb, bundle);
    let payload = encode_nnvis(fbb.finished_data());

    std::fs::write(&output, &payload).context("failed to write output file")?;
    eprintln!(
        "nnvis-preprocess: wrote {} bytes to {} ({} nodes encoded)",
        payload.len(),
        output.display(),
        onnx_bundle.graph.len(),
    );

    Ok(())
}
