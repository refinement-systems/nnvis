import argparse
import json
import os
from transformers import AutoConfig
import safetensors
import onnx
import struct
import hashlib
import math

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

def generate_layering(onnx_graph_info):
    nodes_by_name = {}
    for i, n in enumerate(onnx_graph_info):
        name = n.get("name") or f"{n['op_type']}_{i}"
        nodes_by_name[name] = n
        
    incoming_edges = {n: [] for n in nodes_by_name}
    outgoing_edges = {n: [] for n in nodes_by_name}
    
    known_outputs = {}
    for name, node in nodes_by_name.items():
        for out in node.get("outputs", []):
            if out:
                known_outputs[out] = name

    for name, node in nodes_by_name.items():
        for inp in node.get("inputs", []):
            if inp and inp in known_outputs:
                src_node = known_outputs[inp]
                outgoing_edges[src_node].append(name)
                incoming_edges[name].append(src_node)

    levels = {}
    queue = []
    
    in_degrees = {n: len(incoming_edges[n]) for n in nodes_by_name}
    for n, deg in in_degrees.items():
        if deg == 0:
            levels[n] = 0
            queue.append(n)
            
    while queue:
        curr = queue.pop(0)
        curr_level = levels[curr]
        
        for neighbor in outgoing_edges[curr]:
            if neighbor not in levels or levels[neighbor] <= curr_level:
                levels[neighbor] = curr_level + 1
            
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                queue.append(neighbor)
                
    max_l = max(levels.values()) if levels else 0
    for n in nodes_by_name:
        if n not in levels:
            levels[n] = max_l + 1
            
    return levels, outgoing_edges, nodes_by_name

def string_to_color(s):
    hash_object = hashlib.md5(s.encode())
    hex_color = hash_object.hexdigest()[:6]
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return [r, g, b]

def generate_onnx_3d_layout(onnx_graph_info):
    levels, outgoing_edges, nodes_by_name = generate_layering(onnx_graph_info)
    
    nodes_by_level = {}
    for n, l in levels.items():
        if l not in nodes_by_level:
            nodes_by_level[l] = []
        nodes_by_level[l].append(n)
        
    positions = {}
    Y_SPACING = 3.0
    XZ_SPACING = 1.5
    
    nodes_out = []
    for l, level_nodes in nodes_by_level.items():
        count = len(level_nodes)
        grid_size = math.ceil(math.sqrt(count))
        
        for i, n in enumerate(level_nodes):
            row = i // grid_size
            col = i % grid_size
            
            x = (col - (grid_size - 1) / 2.0) * XZ_SPACING
            z = (row - (grid_size - 1) / 2.0) * XZ_SPACING
            y = l * Y_SPACING
            
            positions[n] = [x, y, z]
            
            op_type = nodes_by_name[n].get("op_type", "Unknown")
            nodes_out.append({
                "id": n,
                "label": op_type,
                "op_type": op_type,
                "pos": [x, y, z],
                "color": string_to_color(op_type)
            })

    edges_out = []
    for src, targets in outgoing_edges.items():
        for tgt in targets:
            if src in positions and tgt in positions:
                edges_out.append({
                    "source": src,
                    "target": tgt,
                    "points": [positions[src], positions[tgt]]
                })
                
    return {
        "nodes": nodes_out,
        "edges": edges_out
    }

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
        "executable_graph_3d": generate_onnx_3d_layout(onnx_graph),
        "tensor_names": safetensors_meta
    }
    
    out_file = os.path.join(args.model_dir, "model_summary.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved custom summary to {out_file}")

if __name__ == "__main__":
    main()
