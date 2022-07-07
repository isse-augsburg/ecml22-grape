import dgl
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import entropy
import time
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Union

from util.abstract_os_handler import AbstractOSHandler


def collate(samples: List):
    """
    Collates a list of samples. Needed for batched training
    :param samples: a list of pairs (graph, label).
    :return:
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class PredictionEvaluator:
    """
        A class to evaluate a prediction model regarding the performance measure top-k hits on the given eval set.
    """

    def __init__(self,
                 eval_set: Dataset,
                 model: torch.nn.Module,
                 model_output_file: str,
                 os_handler: AbstractOSHandler):
        self.eval_set = eval_set
        self.model = model
        self.model_output_file = model_output_file
        self.os_handler = os_handler
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.batch_size = 1024
        self.entropy_list_per_head = []
        self.uniform_entropy_list_per_head = []

    def evaluate(self, k_values_to_test: Tuple[int] = (1, 2, 3, 5, 10, 15, 20), investigate_attention: bool = False):
        """
        Evaluates the top-k rate of the model's predictions for the given list of values for k.
        Optionally, it compares the distribution of the model's attention weights to the uniform distribution.
        :param k_values_to_test: list of k-values for top-k rate to evaluate model on
        :param investigate_attention: boolean indicating if the distribution of the attention weights should be compared
            to uniform distribution (only possible for GatPredictor models)
        :return:
        """
        logging.info(f'Evaluation with {len(self.eval_set)} graphs')
        self.model.load_state_dict(torch.load(self.model_output_file))
        self.model.to(self.device)  # modifies the module in-place

        # prepare eval state --> relevant for layers that behave differently in training mode (e.g. dropout)
        self.model.eval()
        label_in_top_k = torch.zeros(len(k_values_to_test), dtype=torch.int)

        data_loader = DataLoader(self.eval_set,
                                 batch_size=self.batch_size,
                                 num_workers=self.os_handler.get_num_workers(),
                                 persistent_workers=self.os_handler.get_num_workers() != 0,
                                 collate_fn=collate)

        # evaluate top k hit rate
        max_k = max(k_values_to_test)
        if investigate_attention:
            entropy_list_per_head = [[] for _ in range(self.model.get_num_heads())]
            uniform_entropy_list_per_head = [[] for _ in range(self.model.get_num_heads())]
        start_time = time.time()

        for batch_idx, (batched_graphs, batched_labels) in enumerate(data_loader):
            batched_graphs, batched_labels = batched_graphs.to(self.device), batched_labels.to(self.device)
            if investigate_attention:
                batched_logits, attention_weights = self.model(batched_graphs, investigate_attention)
                attention_matrix_list = preprocess_attention(attention_weights, batched_graphs)
                # A[h][sink, source] contains the attention of the corresponding edge in head h.

                for head_idx in range(len(attention_matrix_list)):
                    for node_idx in range(batched_graphs.number_of_nodes()):
                        attention_distribution = attention_matrix_list[head_idx].getrow(node_idx)
                        dist = np.ravel(attention_distribution[attention_distribution.nonzero()].toarray())
                        # Ignore distributions with only one element since here the attention distribution
                        #     equals the uniform distribution
                        if len(dist) == 1:
                            continue

                        entropy_, uniform_entropy = calculate_entropies(dist)

                        entropy_list_per_head[head_idx].append(entropy_)
                        uniform_entropy_list_per_head[head_idx].append(uniform_entropy)

            else:
                batched_logits = self.model(batched_graphs)  # shape (batch_size, output_size)

            batched_top_k_indices = torch.topk(batched_logits, max_k, dim=1, largest=True,
                                               sorted=True).indices  # shape (batch_size, max_k)

            for sample_idx in range(len(batched_labels)):
                label_in_top_k += torch.Tensor(
                    [batched_labels[sample_idx] in
                     torch.narrow(batched_top_k_indices, dim=1, start=0, length=k_value)[sample_idx]
                     for k_value in k_values_to_test]
                ).long()

        if investigate_attention:
            self.entropy_list_per_head = entropy_list_per_head
            self.uniform_entropy_list_per_head = uniform_entropy_list_per_head

        print(f'Time for Eval: {time.time() - start_time} s')
        hit_rates = label_in_top_k.detach().cpu().numpy() / len(self.eval_set)

        keys = []
        values = []
        pos = 0
        for (k_, hit_rate) in zip(k_values_to_test, hit_rates):
            pos = pos + 1
            logging.info(f'Hit-rate with top k (k = {k_}) out of {self.model.get_num_classes()} '
                         f'elements: {hit_rate:.6f}')
            keys.append(f'{pos} Hit-rate {k_}')
            values.append(hit_rate)
        return dict(zip(keys, values))

    def save_entropy_histograms(self, histogram_plot_prefix: str, num_bins: int = 20):
        for head_idx in range(len(self.entropy_list_per_head)):
            entropy_list = self.entropy_list_per_head[head_idx]
            uf_entropy_list = self.uniform_entropy_list_per_head[head_idx]

            max_value = max(entropy_list + uf_entropy_list)
            bar_width = (max_value / num_bins)   # * (1.0 if uniform_distribution else 0.75)
            hist_values, hist_bins = np.histogram(entropy_list, bins=num_bins, range=(0.0, max_value))
            hist_values_uf, hist_bins_uf = np.histogram(uf_entropy_list, bins=num_bins, range=(.0, max_value))

            plt.figure()
            plt.bar(hist_bins[:num_bins], hist_values[:num_bins], width=bar_width, color='tab:blue')
            plt.bar(hist_bins_uf[:num_bins] + 0.3 * bar_width, hist_values_uf[:num_bins], width=bar_width,
                    color='tab:orange', alpha=0.8)
            plt.xlabel(f'entropy bins')
            plt.ylabel(f'# of nodes')
            plt.legend(['attention', 'uniform'])
            plt.title(f'Entropy Histogram of head {head_idx}')
            plt.savefig(f'{histogram_plot_prefix}_{head_idx}.png')
        plt.show()


def preprocess_attention(edge_atten: torch.Tensor, graph: dgl.DGLGraph) -> List[np.ndarray]:
    """Organize attentions in the form of csr sparse adjacency matrices from attention on edges.
    Code from https://github.com/dmlc/dgl/issues/608
    Parameters
    ----------
    edge_atten : numpy.array of shape (# edges, # heads, 1)
        Un-normalized attention on edges.
    graph : dgl.DGLGraph.
    Returns
    ---------
    A list with size number of heads of the model containing kind of an adjacency matrix with the attention weights.
    """
    num_nodes = graph.number_of_nodes()
    num_heads = edge_atten.shape[1]
    all_head_attention: List = [lil_matrix((num_nodes, num_nodes)) for _ in range(num_heads)]
    for source_node_idx in range(num_nodes):
        predecessors = list(graph.predecessors(source_node_idx).detach().cpu())
        edges_id = graph.edge_ids(predecessors, source_node_idx).long()
        for head_idx in range(num_heads):
            all_head_attention[head_idx][source_node_idx, predecessors] = \
                edge_atten[edges_id, head_idx, 0].data.detach().cpu().numpy()
    # A[h][i, j] contains the attention of edge j -> i in head h. For non-existing edges this value will be zero.
    #    So, the columns contain the attention weights per node.
    return all_head_attention   # List of shape(#nodes, #nodes)


def calculate_entropies(distribution: Union[List[float], np.ndarray]) -> Tuple[float, float]:
    """
    Calculates entropy of given distribution as well as entropy of corresponding uniform distribution.
    :param distribution: List of probabilities.
    :return: entropy of current distribution, entropy of corresponding uniform distribution
    """
    entropy_value = entropy(distribution, base=2)

    # entropy value for uniform distribution
    length = len(distribution)
    entropy_uniform = np.log2(length)

    return entropy_value, entropy_uniform
