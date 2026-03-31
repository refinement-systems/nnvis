use std::collections::{HashMap, HashSet};
use crate::onnx_parser::OnnxNode;
use crate::config_parser::LayerDef;
use crate::layer_assignment::NodeAssignment;

#[derive(Debug)]
pub struct GroupNodeInfo {
    pub id: String,
    pub label: String,
    pub description: String,
    pub members: Vec<String>,
    pub member_count: u32,
    pub op_type_histogram: HashMap<String, u32>,
}

#[derive(Debug)]
pub struct Edge {
    pub source: String,
    pub target: String,
}

#[derive(Debug)]
pub struct GroupedGraph {
    pub group_nodes: Vec<GroupNodeInfo>,
    pub group_edges: Vec<Edge>,
}

pub fn build_grouped_graph(
    onnx_nodes: &[OnnxNode],
    assignments: &HashMap<String, NodeAssignment>,
    layer_defs: &[LayerDef],
) -> GroupedGraph {
    let mut producer_by_tensor = HashMap::new();

    // 1. Build producer map
    for node in onnx_nodes {
        for out in &node.outputs {
            producer_by_tensor.insert(out.clone(), node.id.clone());
        }
    }

    // 2. Initialize layer groups
    let mut groups: HashMap<String, GroupNodeInfo> = HashMap::new();
    for layer in layer_defs {
        groups.insert(layer.id.clone(), GroupNodeInfo {
            id: layer.id.clone(),
            label: if layer.description.is_empty() { layer.id.clone() } else { layer.description.clone() },
            description: if layer.description.is_empty() { layer.id.clone() } else { layer.description.clone() },
            members: Vec::new(),
            member_count: 0,
            op_type_histogram: HashMap::new(),
        });
    }

    // 3. Unassigned nodes group
    let has_unassigned = assignments.values().any(|a| a.layer_id == "unassigned");
    if has_unassigned {
        groups.insert("unassigned".to_string(), GroupNodeInfo {
            id: "unassigned".to_string(),
            label: "Unassigned Nodes".to_string(),
            description: "Nodes that could not be mapped to a conceptual layer".to_string(),
            members: Vec::new(),
            member_count: 0,
            op_type_histogram: HashMap::new(),
        });
    }

    // 4. Assign nodes to groups
    for node in onnx_nodes {
        if let Some(assignment) = assignments.get(&node.id) {
            let l_id = &assignment.layer_id;
            if let Some(group) = groups.get_mut(l_id) {
                group.members.push(node.id.clone());
                group.member_count += 1;
                *group.op_type_histogram.entry(node.op_type.clone()).or_insert(0) += 1;
            }
        }
    }

    // 5. Inter-layer edges
    let mut grouped_edges_set = HashSet::new();
    for node in onnx_nodes {
        if let Some(dst_assignment) = assignments.get(&node.id) {
            let dst_layer = &dst_assignment.layer_id;
            for inp in &node.inputs {
                if let Some(src_node_id) = producer_by_tensor.get(inp) {
                    if let Some(src_assignment) = assignments.get(src_node_id) {
                        let src_layer = &src_assignment.layer_id;
                        if src_layer != dst_layer {
                            grouped_edges_set.insert((src_layer.clone(), dst_layer.clone()));
                        }
                    }
                }
            }
        }
    }

    // Build the nodes vec and sort by ID mapping index (just to keep determinism)
    let mut group_nodes: Vec<_> = groups.into_values().collect();
    group_nodes.sort_by(|a, b| a.id.cmp(&b.id));

    // Sort edges for determinism
    let mut group_edges: Vec<_> = grouped_edges_set.into_iter().map(|(source, target)| Edge { source, target }).collect();
    group_edges.sort_by(|a, b| (&a.source, &a.target).cmp(&(&b.source, &b.target)));

    GroupedGraph {
        group_nodes,
        group_edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_build_grouped_graph() {
        // Mock some nodes
        let n1 = OnnxNode {
            id: "node1".to_string(),
            name: "node1".to_string(),
            op_type: "MatMul".to_string(),
            domain: "".to_string(),
            inputs: vec!["in1".to_string()],
            outputs: vec!["out1".to_string()],
            doc_string: "".to_string(),
            attributes: HashMap::new(),
            metadata_props: HashMap::new(),
        };
        let n2 = OnnxNode {
            id: "node2".to_string(),
            name: "node2".to_string(),
            op_type: "Add".to_string(),
            domain: "".to_string(),
            inputs: vec!["out1".to_string()],
            outputs: vec!["out2".to_string()],
            doc_string: "".to_string(),
            attributes: HashMap::new(),
            metadata_props: HashMap::new(),
        };
        let onnx_nodes = vec![n1, n2];

        // Assignments
        let mut assignments = HashMap::new();
        assignments.insert("node1".to_string(), NodeAssignment {
            layer_id: "layer1".to_string(),
            scope: None,
            matched_prefix: None,
            matched_via: "match".to_string(),
        });
        assignments.insert("node2".to_string(), NodeAssignment {
            layer_id: "layer2".to_string(),
            scope: None,
            matched_prefix: None,
            matched_via: "match".to_string(),
        });

        // Layers
        let layer_defs = vec![
            LayerDef {
                id: "layer1".to_string(),
                aliases: vec![],
                description: "Layer 1".to_string(),
                sub_components: vec![],
            },
            LayerDef {
                id: "layer2".to_string(),
                aliases: vec![],
                description: "Layer 2".to_string(),
                sub_components: vec![],
            },
        ];

        let graph = build_grouped_graph(&onnx_nodes, &assignments, &layer_defs);

        assert_eq!(graph.group_nodes.len(), 2);
        
        let g1 = graph.group_nodes.iter().find(|g| g.id == "layer1").unwrap();
        assert_eq!(g1.member_count, 1);
        assert_eq!(g1.op_type_histogram.get("MatMul"), Some(&1));
        
        // Edge out1 connects layer1 -> layer2
        assert_eq!(graph.group_edges.len(), 1);
        assert_eq!(graph.group_edges[0].source, "layer1");
        assert_eq!(graph.group_edges[0].target, "layer2");
    }
}
