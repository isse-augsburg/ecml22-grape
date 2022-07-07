from common.component_identifier import ComponentIdentifier
from models.word_to_vec.embedding import Embedding
from preprocessing.graph import Graph
from preprocessing.component import Component
import options as op

import copy
import dgl
import logging
import torch
from typing import List, Tuple

logger = logging.getLogger(__name__)


def construct_instances(graph: Graph,
                        min_size_partial_graph: int = 3,
                        max_size_partial_graph: int = 15) -> List[Tuple[Graph, Component]]:
    unique_instances = []
    queue = [graph]
    seen_graphs = [graph]
    while queue:
        # get fist element and delete it from queue
        g = queue[0]
        queue.pop(0)
        if not (min_size_partial_graph < len(g.get_nodes()) <= max_size_partial_graph):
            continue
        for leaf in g.get_leaf_nodes():
            leaf_node_id = leaf.get_id()
            partial_graph = copy.deepcopy(g)
            partial_graph.remove_leaf_node_by_id(leaf_node_id)
            if partial_graph not in seen_graphs:
                # add partial graph to queue if not seen before
                queue.append(partial_graph)
                seen_graphs.append(partial_graph)
            new_instance = (partial_graph, leaf.get_component())
            if new_instance not in unique_instances:
                unique_instances.append(new_instance)
    return unique_instances


def instance_to_dgl_instance(instance: Tuple[Graph, Component],
                             embedding: Embedding,
                             identifier: ComponentIdentifier,
                             save_sparse: bool = False)\
        -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """
    Transforms a given instance (composed of a graph and a target component) into a tensor-based representation for the
    DGL library.
    :param instance: instance to transform (composed of a graph and a target component)
    :param embedding: Embedding containing vector representations for components
    :param identifier: identifier of the components
    :param save_sparse:
    :return: the transformed instance composed of a DGL Graph (or a list of DGL Graphs) and the target component
      represented as tensor.
    """
    partial_graph, target = instance
    # number nodes from 0 to n-1
    node2number = {}
    graph_nodes = list(partial_graph.get_nodes())
    for node_idx in range(len(graph_nodes)):
        node2number[graph_nodes[node_idx]] = node_idx

    # transform edges and collect node features
    embedding_dimension = 1 if save_sparse else embedding.get_embedding_dimension()
    node_features = torch.zeros((len(graph_nodes), embedding_dimension), dtype=float)
    target2id = embedding.get_word_to_index()
    embedded_vectors = embedding.get_embedded_vectors()

    edges_source = []
    edges_sink = []
    graph_edges = partial_graph.get_edges()
    for node in graph_nodes:
        for connected_node in graph_edges[node]:
            edges_source.append(node2number[node])
            edges_sink.append(node2number[connected_node])

        # add node features
        node_embedding = embedded_vectors[node.get_component().get(identifier)]
        node_features[node2number[node], :] = (torch.argmax(node_embedding) if save_sparse else node_embedding).cpu()

    dgl_graph = dgl.graph((torch.tensor(edges_source, dtype=torch.int32),
                           torch.tensor(edges_sink, dtype=torch.int32)))
    dgl_graph.ndata[op.EMBEDDING_KEYWORD] = node_features
    dgl_graph.construction_id = partial_graph.get_construction_id()  # important for final model evaluation

    # get id for target component
    target_identifier_value = target.get(identifier)
    target_id = torch.tensor(target2id[target_identifier_value])

    return dgl_graph, target_id
