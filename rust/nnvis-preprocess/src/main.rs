// nnvis-preprocess — native CLI that reads model files and writes .nnvis
//
// Issue #3: ONNX protobuf parsing      — done
// Issue #5: config.json + layer names  — done
// Issue #6: node-to-layer assignment   — done
// Remaining work: #4 (SafeTensors), #7 (grouped graph).

mod onnx_proto;
mod onnx_parser;
mod config_parser;
mod layer_assignment;
mod safetensors_parser;
mod grouped_graph;

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

    // ── Issue #6: Assign nodes to layers ───────────────────────────────────
    eprintln!("nnvis-preprocess: assigning nodes to layers …");
    let assignment_result =
        layer_assignment::assign_nodes_to_layers(&onnx_bundle.graph, &layer_names);

    // ── Issue #4: Parse SafeTensors metadata ──────────────────────────────
    eprintln!("nnvis-preprocess: parsing safetensors …");
    let safetensors = safetensors_parser::extract_safetensors_metadata(&cli.model_dir)
        .context("safetensors parsing failed")?;
    eprintln!(
        "nnvis-preprocess: found {} tensors in safetensors",
        safetensors.len()
    );

    // ── Issue #7: Build grouped graph ──────────────────────────────────────
    eprintln!("nnvis-preprocess: building grouped graph …");
    let grouped_graph = grouped_graph::build_grouped_graph(
        &onnx_bundle.graph,
        &assignment_result.assignments,
        &layer_names,
    );
    eprintln!(
        "nnvis-preprocess: built grouped graph with {} nodes and {} edges",
        grouped_graph.group_nodes.len(),
        grouped_graph.group_edges.len()
    );

    if cli.dump_json {
        let layers_json: Vec<serde_json::Value> =
            layer_names.iter().map(|l| l.to_json()).collect();
        let assignments_json: serde_json::Map<String, serde_json::Value> =
            assignment_result
                .assignments
                .iter()
                .map(|(k, v)| (k.clone(), v.to_json()))
                .collect();
        let tensors_json: serde_json::Map<String, serde_json::Value> = safetensors
            .iter()
            .map(|t| {
                let mut obj = serde_json::Map::new();
                obj.insert("dtype".to_string(), serde_json::Value::String(t.dtype.clone()));
                obj.insert(
                    "shape".to_string(),
                    serde_json::Value::Array(
                        t.shape
                            .iter()
                            .map(|&s| serde_json::Value::Number(serde_json::Number::from(s)))
                            .collect(),
                    ),
                );
                (t.name.clone(), serde_json::Value::Object(obj))
            })
            .collect();

        let mut json = onnx_bundle.to_json();
        json["config"] = config.to_json();
        json["layer_names"] = serde_json::Value::Array(layers_json);
        json["tensor_names"] = serde_json::Value::Object(tensors_json);
        json["node_assignments"] = serde_json::Value::Object(assignments_json);
        json["unmatched_nodes"] =
            serde_json::Value::Array(
                assignment_result.unmatched_nodes.into_iter()
                    .map(serde_json::Value::String)
                    .collect()
            );
        println!("{}", serde_json::to_string_pretty(&json)?);
        return Ok(());
    }

    // ── Write .nnvis FlatBuffer ─────────────────────────────────────────────
    use flatbuffers::FlatBufferBuilder;
    use nnvis_core::{
        encode_nnvis, finish_model_bundle_buffer, ConfigSummary as FbConfig,
        ConfigSummaryArgs, LayerDef as FbLayerDef, LayerDefArgs, ModelBundle,
        ModelBundleArgs, NodeAssignment as FbAssignment, NodeAssignmentArgs,
        OnnxNode as FbOnnxNode, OnnxNodeArgs, TensorMeta as FbTensorMeta,
        TensorMetaArgs, Edge as FbEdge, EdgeArgs as FbEdgeArgs,
        GroupNode as FbGroupNode, GroupNodeArgs as FbGroupNodeArgs,
    };

    let mut fbb = FlatBufferBuilder::with_capacity(1024 * 1024);

    // 1. Encode ConfigSummary
    let cf_model_type = fbb.create_string(&config.model_type);
    let cf_archs_fb: Vec<_> = config
        .architectures
        .iter()
        .map(|s| fbb.create_string(s))
        .collect();
    let cf_archs_vec = fbb.create_vector(&cf_archs_fb);
    let fb_config = FbConfig::create(
        &mut fbb,
        &ConfigSummaryArgs {
            model_type: Some(cf_model_type),
            architectures: Some(cf_archs_vec),
            vocab_size: config.vocab_size.unwrap_or(0),
            hidden_size: config.hidden_size.unwrap_or(0),
            num_hidden_layers: config.num_hidden_layers.unwrap_or(0),
            num_attention_heads: config.num_attention_heads.unwrap_or(0),
            intermediate_size: config.intermediate_size.unwrap_or(0),
            max_position_embeddings: config.max_position_embeddings.unwrap_or(0),
        },
    );

    // 2. Encode LayerDefs
    let fb_layers: Vec<_> = layer_names
        .iter()
        .map(|l| {
            let id = fbb.create_string(&l.id);
            let desc = fbb.create_string(&l.description);
            let aliases_fb: Vec<_> = l.aliases.iter().map(|s| fbb.create_string(s)).collect();
            let sub_fb: Vec<_> = l.sub_components.iter().map(|s| fbb.create_string(s)).collect();
            let aliases_vec = fbb.create_vector(&aliases_fb);
            let sub_vec = fbb.create_vector(&sub_fb);
            FbLayerDef::create(
                &mut fbb,
                &LayerDefArgs {
                    id: Some(id),
                    description: Some(desc),
                    aliases: Some(aliases_vec),
                    sub_components: Some(sub_vec),
                },
            )
        })
        .collect();
    let layers_vec = fbb.create_vector(&fb_layers);

    // 3. Encode OnnxNodes
    let fb_nodes: Vec<_> = onnx_bundle
        .graph
        .iter()
        .map(|n| {
            let id = fbb.create_string(&n.id);
            let op = fbb.create_string(&n.op_type);
            let name = fbb.create_string(&n.name);
            let domain = fbb.create_string(&n.domain);
            let attrs = fbb.create_string(&serde_json::to_string(&n.attributes).unwrap_or_default());
            let doc = fbb.create_string(&n.doc_string);

            let inputs_fb: Vec<_> = n.inputs.iter().map(|s| fbb.create_string(s)).collect();
            let outputs_fb: Vec<_> = n.outputs.iter().map(|s| fbb.create_string(s)).collect();
            let inp_vec = fbb.create_vector(&inputs_fb);
            let out_vec = fbb.create_vector(&outputs_fb);

            // Metadata props
            let mut keys: Vec<_> = n.metadata_props.keys().collect();
            keys.sort(); // Deterministic order
            let k_fb: Vec<_> = keys.iter().map(|k| fbb.create_string(k)).collect();
            let v_fb: Vec<_> = keys
                .iter()
                .map(|k| fbb.create_string(n.metadata_props.get(*k).unwrap()))
                .collect();
            let k_vec = fbb.create_vector(&k_fb);
            let v_vec = fbb.create_vector(&v_fb);

            FbOnnxNode::create(
                &mut fbb,
                &OnnxNodeArgs {
                    id: Some(id),
                    op_type: Some(op),
                    name: Some(name),
                    domain: Some(domain),
                    attributes_json: Some(attrs),
                    inputs: Some(inp_vec),
                    outputs: Some(out_vec),
                    doc_string: Some(doc),
                    metadata_keys: Some(k_vec),
                    metadata_values: Some(v_vec),
                },
            )
        })
        .collect();
    let nodes_vec = fbb.create_vector(&fb_nodes);

    // 4. Encode NodeAssignments
    let fb_assignments: Vec<_> = assignment_result
        .assignments
        .iter()
        .map(|(node_id, a)| {
            let n_id = fbb.create_string(node_id);
            let l_id = fbb.create_string(&a.layer_id);
            let scope = fbb.create_string(a.scope.as_deref().unwrap_or(""));
            let prefix = fbb.create_string(a.matched_prefix.as_deref().unwrap_or(""));
            let via = fbb.create_string(&a.matched_via);

            FbAssignment::create(
                &mut fbb,
                &NodeAssignmentArgs {
                    node_id: Some(n_id),
                    layer_id: Some(l_id),
                    scope: Some(scope),
                    matched_prefix: Some(prefix),
                    matched_via: Some(via),
                },
            )
        })
        .collect();
    let assignments_vec = fbb.create_vector(&fb_assignments);

    // 5. Build ModelBundle
    let fb_tensors: Vec<_> = safetensors
        .iter()
        .map(|t| {
            let name_fb = fbb.create_string(&t.name);
            let dtype_fb = fbb.create_string(&t.dtype);
            let shape_fb = fbb.create_vector(&t.shape);
            FbTensorMeta::create(
                &mut fbb,
                &TensorMetaArgs {
                    name: Some(name_fb),
                    dtype: Some(dtype_fb),
                    shape: Some(shape_fb),
                },
            )
        })
        .collect();
    let tensors_vec = fbb.create_vector(&fb_tensors);

    // 6. Encode GroupNodes
    let fb_group_nodes: Vec<_> = grouped_graph
        .group_nodes
        .iter()
        .map(|g| {
            let id = fbb.create_string(&g.id);
            let label = fbb.create_string(&g.label);
            let desc = fbb.create_string(&g.description);
            let members_fb: Vec<_> = g.members.iter().map(|s| fbb.create_string(s)).collect();
            let members_vec = fbb.create_vector(&members_fb);

            let mut hist_keys: Vec<_> = g.op_type_histogram.keys().collect();
            hist_keys.sort();
            let keys_fb: Vec<_> = hist_keys.iter().map(|k| fbb.create_string(k)).collect();
            let counts: Vec<_> = hist_keys
                .iter()
                .map(|k| *g.op_type_histogram.get(*k).unwrap())
                .collect();
            let keys_vec = fbb.create_vector(&keys_fb);
            let counts_vec = fbb.create_vector(&counts);

            FbGroupNode::create(
                &mut fbb,
                &FbGroupNodeArgs {
                    id: Some(id),
                    label: Some(label),
                    description: Some(desc),
                    members: Some(members_vec),
                    member_count: g.member_count,
                    histogram_keys: Some(keys_vec),
                    histogram_counts: Some(counts_vec),
                },
            )
        })
        .collect();
    let group_nodes_vec = fbb.create_vector(&fb_group_nodes);

    // 7. Encode GroupEdges
    let fb_group_edges: Vec<_> = grouped_graph
        .group_edges
        .iter()
        .map(|e| {
            let src = fbb.create_string(&e.source);
            let tgt = fbb.create_string(&e.target);
            FbEdge::create(
                &mut fbb,
                &FbEdgeArgs {
                    source: Some(src),
                    target: Some(tgt),
                },
            )
        })
        .collect();
    let group_edges_vec = fbb.create_vector(&fb_group_edges);

    let bundle = ModelBundle::create(
        &mut fbb,
        &ModelBundleArgs {
            config: Some(fb_config),
            graph_nodes: Some(nodes_vec),
            layer_defs: Some(layers_vec),
            node_assignments: Some(assignments_vec),
            tensors: Some(tensors_vec),
            group_nodes: Some(group_nodes_vec),
            group_edges: Some(group_edges_vec),
            ..Default::default()
        },
    );
    finish_model_bundle_buffer(&mut fbb, bundle);
    let payload = encode_nnvis(fbb.finished_data());

    std::fs::write(&output, &payload).context("failed to write output file")?;
    eprintln!(
        "nnvis-preprocess: wrote {} bytes to {} ({} nodes, {} assignments encoded)",
        payload.len(),
        output.display(),
        onnx_bundle.graph.len(),
        assignment_result.assignments.len(),
    );

    Ok(())
}
