import argparse
import json
import os
from transformers import AutoConfig
import safetensors
import onnx
import struct
import graphviz

def extract_config_summary(model_dir: str):
    config = AutoConfig.from_pretrained(model_dir)
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "id2label": getattr(config, "id2label", {})
    }

def read_safetensors_header(sf_path: str):
    # The safest way to get purely metadata without touching any large memory mappings
    # Safetensors format: 8 bytes (uint64) for JSON header size, followed by the JSON header
    with open(sf_path, 'rb') as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_bytes = f.read(header_size)
    
    header = json.loads(header_bytes)
    # The header is a dict mapping tensor names to {"dtype": "...", "shape": [...], "data_offsets": [...]}
    # It might also contain a "__metadata__" key
    tensor_info = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        tensor_info[key] = {
            "dtype": value.get("dtype"),
            "shape": value.get("shape")
        }
    return tensor_info

def extract_safetensors_metadata(model_dir: str):
    sf_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(sf_path):
        return read_safetensors_header(sf_path)
    return {}

def extract_onnx_graph(model_dir: str):
    onnx_path = os.path.join(model_dir, "onnx", "model.onnx")
    graph_info = []
    if os.path.exists(onnx_path):
        model = onnx.load(onnx_path, load_external_data=False)
        for node in model.graph.node:
            graph_info.append({
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output)
            })
    return graph_info

def generate_onnx_svg(onnx_graph_info):
    dot = graphviz.Digraph(comment='ONNX Model', format='svg')
    dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#2b303b', fontcolor='white', fontname='Helvetica', color='#4f5b66')
    dot.attr('edge', color='#8c9440')
    dot.attr('graph', bgcolor='transparent')
    
    created_nodes = set()
    MAX_NODES = 150
    for i, node in enumerate(onnx_graph_info):
        if i >= MAX_NODES:
            dot.node("truncated", "Graph Truncated\\njuggling 3000+ nodes crashes layout", shape="note", fillcolor="#bf616a")
            break
            
        node_id = f"node_{i}"
        # Keep label short
        label = node.get("name", "") or node["op_type"]
        if len(label) > 30:
            label = label[:27] + "..."
        op_type = node["op_type"]
        
        dot.node(node_id, f"{op_type}\\n{label}")
        created_nodes.add(node_id)
        
        for inp in node["inputs"]:
            if inp:
                if inp not in created_nodes:
                    dot.node(inp, inp[:20] + ("..." if len(inp)>20 else ""), shape='ellipse', fillcolor='#343d46', fontcolor='#c0c5ce', color='#4f5b66')
                    created_nodes.add(inp)
                dot.edge(inp, node_id)
                
        for out in node["outputs"]:
            if out:
                if out not in created_nodes:
                    dot.node(out, out[:20] + ("..." if len(out)>20 else ""), shape='ellipse', fillcolor='#343d46', fontcolor='#c0c5ce', color='#4f5b66')
                    created_nodes.add(out)
                dot.edge(node_id, out)
                
    try:
        print("Rendering Graphviz SVG (this may take a moment)...")
        svg_bytes = dot.pipe()
        return svg_bytes.decode('utf-8')
    except Exception as e:
        print(f"Warning: Failed to render graphviz SVG: {e}")
        return ""

def generate_layer_names(config_summary):
    # Conceptual blocks mapped based on DeBERTa's model architecture logic.
    layers = []
    layers.append({"id": "embeddings", "description": "Token Embeddings (Word, Position, Token Type)"})
    
    num_layers = config_summary.get("num_hidden_layers", 0)
    for i in range(num_layers):
        layers.append({
            "id": f"encoder.layer.{i}",
            "description": f"Encoder Block {i}",
            "sub_components": ["attention (query_proj, key_proj, value_proj, pos_proj)", "intermediate", "output"]
        })
    layers.append({"id": "pooler", "description": "Pooler Layer"})
    layers.append({"id": "classifier", "description": "Classification Head"})
    return layers

def main():
    parser = argparse.ArgumentParser(description="Extract JSON summary from a downloaded model directory.")
    parser.add_argument("model_dir", help="Path to the downloaded model directory (e.g., models/MoritzLaurer_mDeBERTa_v3_base_mnli_xnli)")
    args = parser.parse_args()

    print(f"Extracting info from {args.model_dir}...")
    
    summary = extract_config_summary(args.model_dir)
    safetensors_meta = extract_safetensors_metadata(args.model_dir)
    onnx_graph = extract_onnx_graph(args.model_dir)
    layer_names = generate_layer_names(summary)

    output = {
        "summary": summary,
        "layer_names": layer_names,
        "executable_graph": onnx_graph,
        "executable_graph_svg": generate_onnx_svg(onnx_graph),
        "tensor_names": safetensors_meta
    }
    
    out_file = os.path.join(args.model_dir, "model_summary.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved custom summary to {out_file}")

if __name__ == "__main__":
    main()
