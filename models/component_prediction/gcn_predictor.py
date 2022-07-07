"""
The actual pytorch/DGL model, a graph convolutional network (GCN) to predict components.
"""

from dgl.nn.pytorch import GraphConv
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
import options as op


class GcnPredictor(nn.Module):
    """
        A basic GCN-based model to predict components given an initial graph of embeddings.
        :param node_feature_dim: Size of the node features, i.e. size of embedding.
        :param hidden_dim: Dimension of each hidden layer.
        :param num_hidden_layers: Number of hidden layers.
        :param num_classes: Number of target classes, i.e. dimension of final FC-layer.
        :param dropout_rate: Dropout rate on node features.
        """

    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 hidden_dim: int,
                 num_hidden_layers: int,
                 num_classes: int,
                 dropout_rate: float = 0.0):
        super(GcnPredictor, self).__init__()
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        self.embedding_keyword = op.EMBEDDING_KEYWORD

        self.activation = F.elu

        # input layer
        self.input_layer = GraphConv(node_feature_dim, hidden_dim, bias=True)
        self.layers.append(self.input_layer)

        # hidden layers
        for i in range(num_hidden_layers):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, bias=True))

        # for the read-out from the hidden features of the graph
        self.classify = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def get_num_classes(self) -> int:
        """ Returns the number of classes to predict from. """
        return self.num_classes

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Forwards a DGL graph through the network.
        :param graph: graph sample
        :return: tensor of unnormalized logits
        """
        # the input to the GCN are the component embeddings
        current_device = graph.device
        h = graph.ndata[self.embedding_keyword].type(torch.FloatTensor).to(current_device)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)   # do not use dropout in first layer (input layer)
            h = self.activation(layer(graph, h))
        # update all nodes
        graph.ndata['h'] = h

        # Read-out process: average all node representations of the last layer
        graph_summary = dgl.mean_nodes(graph, 'h')
        logits = self.classify(graph_summary)
        return logits

    def collate(self, samples: List):
        """
        Collates a list of samples. Needed for batched training
        :param samples: a list of pairs (graph, label).
        :return:
        """
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)
