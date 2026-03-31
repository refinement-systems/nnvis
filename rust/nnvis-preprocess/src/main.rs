// nnvis-preprocess — native CLI that reads model files and writes .nnvis
//
// This is a skeleton.  Full extraction logic will be implemented in
// issues #3 (ONNX parsing), #4 (SafeTensors), #5 (config.json), #6 (layer
// assignment), and #7 (grouped graph).

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
}

fn main() {
    let cli = Cli::parse();

    let output = cli
        .output
        .unwrap_or_else(|| cli.model_dir.join("model.nnvis"));

    eprintln!("nnvis-preprocess: model_dir = {}", cli.model_dir.display());
    eprintln!("nnvis-preprocess: output    = {}", output.display());
    eprintln!("nnvis-preprocess: extraction logic not yet implemented (see issues #3–#7)");

    // Placeholder: write an empty (but valid header) .nnvis file so the
    // scaffold can be tested end-to-end.
    use flatbuffers::FlatBufferBuilder;
    use nnvis_core::{encode_nnvis, finish_model_bundle_buffer, ModelBundle, ModelBundleArgs};

    let mut fbb = FlatBufferBuilder::with_capacity(64);
    let bundle = ModelBundle::create(&mut fbb, &ModelBundleArgs::default());
    finish_model_bundle_buffer(&mut fbb, bundle);
    let payload = encode_nnvis(fbb.finished_data());

    std::fs::write(&output, &payload).expect("failed to write output file");
    eprintln!(
        "nnvis-preprocess: wrote {} bytes to {}",
        payload.len(),
        output.display()
    );
}
