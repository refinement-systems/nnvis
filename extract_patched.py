# Permission to use, copy, modify, and/or distribute this software for
# any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
# AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import argparse
import hashlib
import json
import math
import os
import random
import re
import struct
from collections import deque

import onnx
from transformers import AutoConfig


def extract_config_summary(model_dir: str):
    config = AutoConfig.from_pretrained(model_dir)
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "architectures": list(getattr(config, "architectures", []) or []),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "id2label": getattr(config, "id2label", {}),
    }


def read_safetensors_header(sf_path: str):
    # Safetensors format: 8 bytes (uint64) for JSON header size, followed by the JSON header.
    # This reads metadata only and does not map or load tensor payloads.
    with open(sf_path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = f.read(header_size)

    header = json.loads(header_bytes)
    tensor_info = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        tensor_info[key] = {
            "dtype": value.get("dtype"),
            "shape": value.get("shape"),
        }
    return tensor_info


def extract_safetensors_metadata(model_dir: str):
    sf_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(sf_path):
        return read_safetensors_header(sf_path)
    return {}


def _metadata_props_to_dict(proto):
    return {prop.key: prop.value for prop in getattr(proto, "metadata_props", [])}


def _tensor_shape_from_value_info(value_info_proto):
    tensor_type = value_info_proto.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def _attr_to_python(attr):
    attr_type = attr.type
    if attr_type == onnx.AttributeProto.FLOAT:
        return attr.f
    if attr_type == onnx.AttributeProto.INT:
        return attr.i
    if attr_type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if attr_type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if attr_type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if attr_type == onnx.AttributeProto.STRINGS:
        return [s.decode("utf-8", errors="replace") for s in attr.strings]
    if attr_type == onnx.AttributeProto.TENSOR:
        return {
            "tensor_name": attr.t.name,
            "data_type": int(attr.t.data_type),
            "dims": list(attr.t.dims),
        }
    return None


def extract_onnx_graph(model_dir: str):
    onnx_path = os.path.join(model_dir, "onnx", "model.onnx")
    if not os.path.exists(onnx_path):
        return {
            "graph": [],
            "graph_metadata": {},
            "model_metadata": {},
            "initializers": {},
            "value_info": {},
        }

    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph

    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = {
            "data_type": int(init.data_type),
            "dims": list(init.dims),
        }

    value_info = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        value_info[vi.name] = {
            "shape": _tensor_shape_from_value_info(vi),
            "metadata_props": _metadata_props_to_dict(vi),
        }

    graph_info = []
    for i, node in enumerate(graph.node):
        node_id = node.name or f"{node.op_type}_{i}"
        graph_info.append(
            {
                "id": node_id,
                "name": node.name,
                "op_type": node.op_type,
                "domain": node.domain,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {attr.name: _attr_to_python(attr) for attr in node.attribute},
                "doc_string": node.doc_string,
                "metadata_props": _metadata_props_to_dict(node),
            }
        )

    model_metadata = {
        "ir_version": int(getattr(model, "ir_version", 0)),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "domain": model.domain,
        "model_version": int(getattr(model, "model_version", 0)),
        "doc_string": model.doc_string,
        "metadata_props": _metadata_props_to_dict(model),
        "opset_import": [
            {"domain": imp.domain, "version": int(imp.version)} for imp in model.opset_import
        ],
    }

    return {
        "graph": graph_info,
        "graph_metadata": _metadata_props_to_dict(graph),
        "model_metadata": model_metadata,
        "initializers": initializers,
        "value_info": value_info,
    }


def _node_id(node, index):
    return node.get("id") or node.get("name") or f"{node.get('op_type', 'Unknown')}_{index}"


def build_connectivity(graph_nodes):
    nodes_by_id = {}
    for i, node in enumerate(graph_nodes):
        node_id = _node_id(node, i)
        if "id" not in node:
            node["id"] = node_id
        nodes_by_id[node_id] = node

    incoming_edges = {node_id: [] for node_id in nodes_by_id}
    outgoing_edges = {node_id: [] for node_id in nodes_by_id}
    producer_by_tensor = {}

    for node_id, node in nodes_by_id.items():
        for output_name in node.get("outputs", []):
            if output_name:
                producer_by_tensor[output_name] = node_id

    for node_id, node in nodes_by_id.items():
        for input_name in node.get("inputs", []):
            if not input_name:
                continue
            src_node_id = producer_by_tensor.get(input_name)
            if src_node_id is None:
                continue
            outgoing_edges[src_node_id].append(node_id)
            incoming_edges[node_id].append(src_node_id)

    return nodes_by_id, incoming_edges, outgoing_edges, producer_by_tensor


def generate_layering(graph_nodes):
    nodes_by_id, incoming_edges, outgoing_edges, _ = build_connectivity(graph_nodes)

    levels = {}
    in_degrees = {node_id: len(srcs) for node_id, srcs in incoming_edges.items()}
    queue = deque()

    for node_id, degree in in_degrees.items():
        if degree == 0:
            levels[node_id] = 0
            queue.append(node_id)

    while queue:
        curr = queue.popleft()
        curr_level = levels[curr]

        for neighbor in outgoing_edges[curr]:
            next_level = curr_level + 1
            if neighbor not in levels or levels[neighbor] < next_level:
                levels[neighbor] = next_level

            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                queue.append(neighbor)

    max_level = max(levels.values()) if levels else 0
    for node_id in nodes_by_id:
        if node_id not in levels:
            levels[node_id] = max_level + 1

    return levels, outgoing_edges, nodes_by_id


def string_to_color(value):
    hash_object = hashlib.md5(value.encode("utf-8"))
    hex_color = hash_object.hexdigest()[:6]
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return [r, g, b]


def generate_graph_3d_layout(graph_nodes, graph_edges=None, color_key="op_type", label_key="label"):
    if graph_edges is None:
        levels, outgoing_edges, nodes_by_id = generate_layering(graph_nodes)
    else:
        nodes_by_id = {}
        for i, node in enumerate(graph_nodes):
            node_copy = dict(node)
            node_copy.setdefault("id", _node_id(node_copy, i))
            nodes_by_id[node_copy["id"]] = node_copy

        outgoing_edges = {node_id: [] for node_id in nodes_by_id}
        incoming_edges = {node_id: [] for node_id in nodes_by_id}
        for edge in graph_edges:
            src = edge["source"]
            tgt = edge["target"]
            if src in nodes_by_id and tgt in nodes_by_id:
                outgoing_edges[src].append(tgt)
                incoming_edges[tgt].append(src)

        levels = {}
        in_degrees = {node_id: len(srcs) for node_id, srcs in incoming_edges.items()}
        queue = deque()
        for node_id, degree in in_degrees.items():
            if degree == 0:
                levels[node_id] = 0
                queue.append(node_id)

        while queue:
            curr = queue.popleft()
            curr_level = levels[curr]
            for neighbor in outgoing_edges[curr]:
                next_level = curr_level + 1
                if neighbor not in levels or levels[neighbor] < next_level:
                    levels[neighbor] = next_level
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(neighbor)

        max_level = max(levels.values()) if levels else 0
        for node_id in nodes_by_id:
            if node_id not in levels:
                levels[node_id] = max_level + 1

    nodes_by_level = {}
    for node_id, level in levels.items():
        nodes_by_level.setdefault(level, []).append(node_id)

    positions = {}
    y_spacing = 5.0
    xz_spacing = 3.0

    random.seed(42)
    for level, level_nodes in nodes_by_level.items():
        count = len(level_nodes)
        grid_size = max(1, math.ceil(math.sqrt(count)))
        for index, node_id in enumerate(level_nodes):
            row = index // grid_size
            col = index % grid_size
            x = (col - (grid_size - 1) / 2.0) * xz_spacing + random.uniform(-0.1, 0.1)
            z = (row - (grid_size - 1) / 2.0) * xz_spacing + random.uniform(-0.1, 0.1)
            y = level * y_spacing
            positions[node_id] = [x, y, z]

    outgoing_edges_set = {node_id: set() for node_id in levels}
    for src, tgts in outgoing_edges.items():
        for tgt in tgts:
            if src in levels and tgt in levels:
                outgoing_edges_set[src].add(tgt)

    iterations = 50
    attraction_factor = 0.05
    repulsion_factor = 1.0
    repulsion_radius = xz_spacing * 1.5

    for iteration in range(iterations):
        displacements = {node_id: [0.0, 0.0] for node_id in levels}

        for level, level_nodes in nodes_by_level.items():
            for i in range(len(level_nodes)):
                n1 = level_nodes[i]
                p1 = positions[n1]
                for j in range(i + 1, len(level_nodes)):
                    n2 = level_nodes[j]
                    p2 = positions[n2]

                    dx = p1[0] - p2[0]
                    dz = p1[2] - p2[2]
                    dist_sq = dx * dx + dz * dz
                    if dist_sq < 0.0001:
                        dx = random.uniform(-0.1, 0.1)
                        dz = random.uniform(-0.1, 0.1)
                        dist_sq = dx * dx + dz * dz

                    if dist_sq < repulsion_radius * repulsion_radius:
                        dist = math.sqrt(dist_sq)
                        force = repulsion_factor * (repulsion_radius - dist) / dist
                        fx = dx * force
                        fz = dz * force
                        displacements[n1][0] += fx
                        displacements[n1][1] += fz
                        displacements[n2][0] -= fx
                        displacements[n2][1] -= fz

        for node_id, tgts in outgoing_edges_set.items():
            if node_id not in positions:
                continue
            p1 = positions[node_id]
            for tgt in tgts:
                if tgt not in positions:
                    continue
                p2 = positions[tgt]
                dx = p2[0] - p1[0]
                dy = abs(p2[1] - p1[1])
                dz = p2[2] - p1[2]
                distance_ratio = max(1.0, dy / y_spacing)
                decayed_attraction = attraction_factor / (distance_ratio * distance_ratio)
                fx = dx * decayed_attraction
                fz = dz * decayed_attraction
                displacements[node_id][0] += fx
                displacements[node_id][1] += fz
                displacements[tgt][0] -= fx
                displacements[tgt][1] -= fz

        cooling = max(0.05, 1.0 - (iteration / iterations))
        for node_id, displacement in displacements.items():
            positions[node_id][0] += displacement[0] * cooling
            positions[node_id][2] += displacement[1] * cooling

    nodes_out = []
    for node_id in levels:
        node = nodes_by_id[node_id]
        label = node.get(label_key) or node.get("label") or node.get("op_type") or node_id
        color_basis = str(node.get(color_key) or label)
        x, y, z = positions[node_id]
        nodes_out.append(
            {
                "id": node_id,
                "label": label,
                "pos": [x, y, z],
                "color": string_to_color(color_basis),
            }
        )

    edges_out = []
    for src, targets in outgoing_edges.items():
        for tgt in targets:
            if src in positions and tgt in positions:
                edges_out.append(
                    {
                        "source": src,
                        "target": tgt,
                        "points": [positions[src], positions[tgt]],
                    }
                )

    return {"nodes": nodes_out, "edges": edges_out}


def generate_onnx_3d_layout(onnx_graph_info):
    graph_nodes = []
    for i, node in enumerate(onnx_graph_info):
        node_copy = dict(node)
        node_copy.setdefault("id", _node_id(node_copy, i))
        node_copy.setdefault("label", node_copy.get("op_type", "Unknown"))
        graph_nodes.append(node_copy)
    return generate_graph_3d_layout(graph_nodes, graph_edges=None, color_key="op_type", label_key="label")


def _module_roots_for_model(config_summary):
    model_type = (config_summary.get("model_type") or "").lower()
    architectures = [str(x).lower() for x in (config_summary.get("architectures") or [])]

    roots = []
    if "deberta" in model_type or any("deberta" in item for item in architectures):
        roots.extend(["deberta", "model", "bert", "roberta"])
    else:
        roots.extend(["model", "encoder", "transformer"])
    return roots


def generate_layer_names(config_summary):
    roots = _module_roots_for_model(config_summary)
    primary_root = roots[0] if roots else "model"

    layers = []
    layers.append(
        {
            "id": f"{primary_root}.embeddings",
            "aliases": [f"{root}.embeddings" for root in roots] + ["embeddings"],
            "description": "Token Embeddings",
        }
    )

    num_layers = config_summary.get("num_hidden_layers", 0) or 0
    for i in range(num_layers):
        layer_aliases = [f"{root}.encoder.layer.{i}" for root in roots]
        if primary_root == "encoder":
            layer_aliases.append(f"encoder.layer.{i}")
        layers.append(
            {
                "id": f"{primary_root}.encoder.layer.{i}",
                "aliases": layer_aliases + [f"layer.{i}", f"layer_{i}"],
                "description": f"Encoder Block {i}",
                "sub_components": [
                    "attention",
                    "intermediate",
                    "output",
                ],
            }
        )

    layers.append(
        {
            "id": "pooler",
            "aliases": ["pooler", f"{primary_root}.pooler"],
            "description": "Pooler Layer",
        }
    )
    layers.append(
        {
            "id": "classifier",
            "aliases": ["classifier", "cls", "classification_head", "head"],
            "description": "Classification Head",
        }
    )
    return layers


def _parse_name_scopes(raw_value):
    if not raw_value:
        return []

    raw = str(raw_value).strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, str):
            raw = parsed.strip()
    except Exception:
        pass

    for separator in (">", "/", "::"):
        if separator in raw:
            return [part.strip() for part in raw.split(separator) if part.strip()]

    if "." in raw:
        return [part.strip() for part in raw.split(".") if part.strip()]

    return [raw]


def _canonical_scope(node):
    metadata = node.get("metadata_props", {}) or {}
    scope_keys = [
        "pkg.torch.onnx.name_scopes",
        "pkg.torch.onnx.scope",
        "namespace",
        "scope",
    ]
    for key in scope_keys:
        if key in metadata and metadata[key]:
            parts = _parse_name_scopes(metadata[key])
            if parts:
                return ".".join(parts).strip(".")
    return None


def _match_layer_by_prefix(text, layer_defs):
    if not text:
        return None
    normalized = text.strip(".")
    if not normalized:
        return None

    candidates = []
    for layer in layer_defs:
        for alias in layer.get("aliases", [layer["id"]]):
            alias = alias.strip(".")
            if not alias:
                continue
            if normalized == alias or normalized.startswith(alias + "."):
                candidates.append((len(alias), layer["id"], alias))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, layer_id, alias = candidates[0]
    return {"layer_id": layer_id, "matched_prefix": alias}


def _guess_layer_from_text(text, layer_defs):
    if not text:
        return None

    lowered = text.lower()
    matched = _match_layer_by_prefix(lowered, layer_defs)
    if matched:
        return {**matched, "matched_via": "prefix-text"}

    if "embed" in lowered:
        for layer in layer_defs:
            if layer["id"].endswith(".embeddings"):
                return {"layer_id": layer["id"], "matched_prefix": "embed", "matched_via": "heuristic-embeddings"}

    if "pool" in lowered:
        return {"layer_id": "pooler", "matched_prefix": "pool", "matched_via": "heuristic-pooler"}

    if any(token in lowered for token in ("classif", "logits", "label", "pred")):
        return {"layer_id": "classifier", "matched_prefix": "classif", "matched_via": "heuristic-classifier"}

    match = re.search(r"(?:^|[./_])layer[./_](\d+)(?:$|[./_])", lowered)
    if match:
        layer_index = match.group(1)
        for layer in layer_defs:
            if layer["id"].endswith(f".encoder.layer.{layer_index}"):
                return {
                    "layer_id": layer["id"],
                    "matched_prefix": f"layer.{layer_index}",
                    "matched_via": "heuristic-layer-index",
                }

    return None


def assign_nodes_to_layers(onnx_nodes, layer_defs):
    assignments = {}
    unmatched_nodes = []

    for node in onnx_nodes:
        scope = _canonical_scope(node)
        matched = None

        if scope:
            matched = _match_layer_by_prefix(scope.lower(), layer_defs)
            if matched:
                matched["matched_via"] = "metadata-scope"

        if matched is None:
            texts = [node.get("name") or ""]
            texts.extend(node.get("inputs", []))
            texts.extend(node.get("outputs", []))
            for text in texts:
                matched = _guess_layer_from_text(text, layer_defs)
                if matched is not None:
                    break

        if matched is None:
            matched = {
                "layer_id": "unassigned",
                "matched_prefix": None,
                "matched_via": "unmatched",
            }
            unmatched_nodes.append(node["id"])

        assignments[node["id"]] = {
            "layer_id": matched["layer_id"],
            "scope": scope,
            "matched_prefix": matched.get("matched_prefix"),
            "matched_via": matched.get("matched_via"),
            "metadata_props": node.get("metadata_props", {}),
        }

    return assignments, unmatched_nodes


def build_grouped_graph(onnx_nodes, assignments, layer_defs):
    layer_map = {layer["id"]: layer for layer in layer_defs}
    nodes_by_id, _, _, producer_by_tensor = build_connectivity(onnx_nodes)

    groups = {}
    for layer_id, layer in layer_map.items():
        groups[layer_id] = {
            "id": layer_id,
            "label": layer.get("description") or layer_id,
            "description": layer.get("description") or layer_id,
            "aliases": layer.get("aliases", []),
            "members": [],
            "member_count": 0,
            "op_type_histogram": {},
        }

    if any(assignment["layer_id"] == "unassigned" for assignment in assignments.values()):
        groups["unassigned"] = {
            "id": "unassigned",
            "label": "Unassigned Nodes",
            "description": "Nodes that could not be mapped to a conceptual layer",
            "aliases": [],
            "members": [],
            "member_count": 0,
            "op_type_histogram": {},
        }

    for node_id, node in nodes_by_id.items():
        layer_id = assignments[node_id]["layer_id"]
        group = groups.setdefault(
            layer_id,
            {
                "id": layer_id,
                "label": layer_id,
                "description": layer_id,
                "aliases": [],
                "members": [],
                "member_count": 0,
                "op_type_histogram": {},
            },
        )
        group["members"].append(node_id)
        group["member_count"] += 1
        op_type = node.get("op_type", "Unknown")
        group["op_type_histogram"][op_type] = group["op_type_histogram"].get(op_type, 0) + 1

    grouped_edges = set()
    for node_id, node in nodes_by_id.items():
        dst_layer = assignments[node_id]["layer_id"]
        for input_name in node.get("inputs", []):
            src_node_id = producer_by_tensor.get(input_name)
            if src_node_id is None:
                continue
            src_layer = assignments[src_node_id]["layer_id"]
            if src_layer != dst_layer:
                grouped_edges.add((src_layer, dst_layer))

    group_nodes = list(groups.values())
    group_edges = [{"source": src, "target": dst} for src, dst in sorted(grouped_edges)]
    grouped_graph_3d = generate_graph_3d_layout(group_nodes, group_edges, color_key="id", label_key="label")

    return {
        "nodes": group_nodes,
        "edges": group_edges,
        "layout_3d": grouped_graph_3d,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract JSON summary from a downloaded model directory.")
    parser.add_argument(
        "model_dir",
        help="Path to the downloaded model directory (e.g., models/MoritzLaurer_mDeBERTa_v3_base_mnli_xnli)",
    )
    args = parser.parse_args()

    print(f"Extracting info from {args.model_dir}...")

    summary = extract_config_summary(args.model_dir)
    safetensors_meta = extract_safetensors_metadata(args.model_dir)
    onnx_bundle = extract_onnx_graph(args.model_dir)
    onnx_graph = onnx_bundle["graph"]
    layer_names = generate_layer_names(summary)
    node_assignments, unmatched_nodes = assign_nodes_to_layers(onnx_graph, layer_names)
    grouped_graph = build_grouped_graph(onnx_graph, node_assignments, layer_names)

    output = {
        "summary": summary,
        "layer_names": layer_names,
        "tensor_names": safetensors_meta,
        "model_metadata": onnx_bundle["model_metadata"],
        "graph_metadata": onnx_bundle["graph_metadata"],
        "onnx_initializers": onnx_bundle["initializers"],
        "onnx_value_info": onnx_bundle["value_info"],
        "executable_graph": onnx_graph,
        "executable_graph_3d": generate_onnx_3d_layout(onnx_graph),
        "node_assignments": node_assignments,
        "grouped_graph": grouped_graph,
        "unmatched_nodes": unmatched_nodes,
    }

    out_file = os.path.join(args.model_dir, "model_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved custom summary to {out_file}")


if __name__ == "__main__":
    main()
