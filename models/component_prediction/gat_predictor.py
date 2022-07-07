"""
The actual pytorch/DGL model, a graph attention network (GAT) to predict components.
"""

import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
from dgl.nn.pytorch import GATConv
from typing import List, Tuple, Union
import options as op


class GatPredictor(nn.Module):
    """
        A basic GAT-based model to predict components given an initial graph of embeddings.
        :param node_feature_dim: Size of the node features, i.e. size of embedding.
        :param hidden_dim: Dimension of each hidden layer.
        :param num_heads: Number of heads in Multi-Head Attention.
        :param num_hidden_layers: Number of hidden layers.
        :param num_classes: Number of target classes, i.e. dimension of final FC-layer.
        :param feat_drop: Dropout rate on node features.
        :param attn_drop: Dropout rate on attention weights.
        :param residual: Boolean if residual connections should be used in each hidden layer.
    """

    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_hidden_layers: int,
                 num_classes: int,
                 feat_drop: float = 0.0,
                 attn_drop: float = 0.0,
                 residual: bool = False):
        super(GatPredictor, self).__init__()
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.embedding_keyword = op.EMBEDDING_KEYWORD

        self.activation = F.elu

        # input projection (no residual). Do not use dropout in input layer
        self.input_layer = GATConv(node_feature_dim, hidden_dim, num_heads, 0.0, 0.0, residual=False)
        self.layers.append(self.input_layer)

        # hidden layers
        for i in range(num_hidden_layers):     # used to  be range(1, num_hidden_layers)
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop,
                                       residual=residual))

        # for the read-out from the hidden features of the graph
        self.classify = nn.Linear(hidden_dim * num_heads, num_classes)

    def get_num_classes(self) -> int:
        """ Returns the number of classes to predict from. """
        return self.num_classes

    def get_num_heads(self) -> int:
        return self.num_heads

    def forward(self, graph: dgl.DGLGraph, get_attention: bool = False) -> Union[torch.Tensor,
                                                                                 Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forwards a DGL graph through the network.
        :param graph: graph sample
        :param get_attention: weather the attention weights of last layer should be returned as well
                            (typically used for evaluation of the model)
        :return: tensor of unnormalized logits (and attention if get_attention is True)
        """
        # the input to the GAT are the component embeddings
        current_device = graph.device
        h = graph.ndata[self.embedding_keyword].type(torch.FloatTensor).to(current_device)
        for i, layer in enumerate(self.layers):
            # only request attention for last layer (saves time)
            get_att_for_layer = get_attention and i == len(self.layers) - 1
            res = layer(graph, h, get_att_for_layer)
            if get_att_for_layer:
                h, attention_weights = res
            else:
                h = res
            h = self.activation(h).flatten(1)

        # update all nodes
        graph.ndata['h'] = h

        # Read-out process: average all node representations of the last layer
        graph_summary = dgl.mean_nodes(graph, 'h')
        logits = self.classify(graph_summary)

        if get_attention:
            return logits, attention_weights   # shape attention weights (#edges, #heads, 1)
        return logits

    def collate(self, samples: List):
        """
        Collates a list of samples. Needed for batched training
        :param samples: a list of pairs (graph, label). Needed for batched training
        :return:
        """
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)
