""" A non-ML model for component prediction used as baseline model. """

from common.catalog import Catalog
from common.component_identifier import ComponentIdentifier
from data_set_generator.partial_graphs import construct_instances
from preprocessing.graph import Graph
from preprocessing.component import Component
from util.file_handler import deserialize, serialize
import options as op

from collections import Counter, OrderedDict
from functools import reduce
import multiprocessing as mp
import numpy as np
import os
import tqdm
from typing import Dict, List, Tuple


def extend_dict(dictionary: Dict[object, List[str]], key: object, values: List[str]) -> Dict[object, List[str]]:
    current_values = dictionary.get(key, [])
    current_values.extend(values)
    dictionary[key] = current_values
    return dictionary


class FrequencyBasedPredictor:
    """
        Baseline model for component recommendation.
        :param identifier: identifier of the components
        :param mapping:
    """

    def __init__(self,
                 identifier: ComponentIdentifier,
                 mapping: 'OrderedDict[Graph, Counter[str]]' = None):

        self.__identifier = identifier
        self.__mapping: 'OrderedDict[Graph, Counter[str]]' = mapping

    def create_mapping_from(self, training_data: List[Tuple[Graph, Component]]):
        print(f'preprocess {len(training_data)} partial graphs')

        mapping = dict()
        additional_instances: List[Tuple[Graph, Component]] = []
        for g, p in training_data:
            if len(g.get_nodes()) == 3:
                additional_instances.extend(construct_instances(g, min_size_partial_graph=1))  # min-size = 1

            mapping = extend_dict(mapping, g, [p.get(self.__identifier)])

        print(f'preprocess {len(additional_instances)} additional instances with size < 3')
        for g, p in additional_instances:
            mapping = extend_dict(mapping, g, [p.get(self.__identifier)])

        # transform value list to Counter with relative frequencies
        # mapping: Graph -> Counter of ComponentIdentifiers
        print('transform mapping')
        for key_ in mapping.keys():
            counter = Counter(mapping[key_])
            for k in counter.keys():
                counter_sum = sum(counter.values())
                counter[k] = counter[k] / counter_sum   # turn absolute into relative frequencies
            mapping[key_] = counter
        # sort keys (graphs) by their number of nodes
        self.__mapping = OrderedDict(sorted(mapping.items(), key=lambda i: -len(i[0].get_nodes())))
        print(f'# keys: {len(mapping.keys())}')

    def save_mapping(self, filename: str):
        serialize(self.__mapping, filename)

    def mapping_size(self):
        return len(self.__mapping.keys())

    def predict_best_k(self, input_graph: Graph, k_value: int) -> List[str]:
        graph_size = len(input_graph.get_nodes())

        subgraphs = []
        for seen_graph in self.__mapping.keys():
            # ignore bigger seen graphs
            if len(seen_graph.get_nodes()) > graph_size:
                continue
            # check if seen_graph is a subgraph of input_graph
            if input_graph.is_subgraph(seen_graph):
                # do not include the seen_graph to the list of subgraphs if we already found a different seen_graph
                #    the current seen_graph is a subgraph of
                if not any([s.is_subgraph(seen_graph) for s in subgraphs]):
                    subgraphs.append(seen_graph)
                    if input_graph.edge_subsumption(subgraphs):
                        break

        if not subgraphs:
            return []

        # predict components
        total_counter = Counter()
        for subgraph in subgraphs:
            counter = self.__mapping[subgraph]
            for k in counter.keys():
                total_counter[k] = max(total_counter[k], counter[k]) if k in total_counter.keys() else counter[k]

        return [elem for elem, count in total_counter.most_common(k_value)]


def init_worker(m_file, max_size, comp_identifier, kvalues):
    global model, size, component_identifier, k_values_to_test
    size = max_size
    component_identifier = comp_identifier
    k_values_to_test = kvalues
    print(f'Process {os.getpid()}: loading mapping...')
    mapping = deserialize(m_file)
    model = FrequencyBasedPredictor(component_identifier, mapping)
    print(f'Process {os.getpid()}: Initialized.')


def evaluate_sample(sample):
    idx, (graph, component) = sample

    label = component.get(component_identifier)
    predictions = model.predict_best_k(graph, max(k_values_to_test))

    # evaluate
    if label in predictions:
        return predictions.index(label)


if __name__ == "__main__":

    catalog = Catalog.A
    component_identifier = ComponentIdentifier.COMPONENT
    print(f'Working on catalog {catalog.value} and component ID {component_identifier.value}.')
    k_values_to_test = [1, 2, 3, 5, 10, 15, 20]
    eval_set = 'test'   # 'val'
    process_count = 20  # nbr of processes

    mapping_file = os.path.join(op.DATA_LOCATION, f'baseline_mapping_{component_identifier.value}_{catalog.value}_max.dat')
    if not os.path.exists(mapping_file):
        print(f'Process {os.getpid()} (MASTER): create new mapping')
        train_data = deserialize(os.path.join(op.DATA_LOCATION, f'{catalog.value}_partial_graphs_train.dat'))
        print('Train data loaded.')
        # To ensure comparability between the baseline and GNN models, the frequency-based model receives
        # only the training set to train.

        model = FrequencyBasedPredictor(component_identifier)
        model.create_mapping_from(train_data)
        model.save_mapping(mapping_file)
        del model, train_data
    else:
        print('Mapping already existing.')

    eval_data: List[Tuple[Graph, Component]] = deserialize(os.path.join(op.DATA_LOCATION,
                                                           f'{catalog.value}_partial_graphs_{eval_set}.dat'))
    print(f'Process {os.getpid()} (MASTER): start evaluation of {len(eval_data)} {eval_set.upper()} samples '
          f'on {process_count} processes...')

    mp.set_start_method('spawn')
    pool = mp.Pool(processes=process_count, initializer=init_worker,
                   initargs=[mapping_file, len(eval_data), component_identifier, k_values_to_test])

    predictions_idx = []
    for idx in tqdm.tqdm(pool.imap_unordered(evaluate_sample, enumerate(eval_data), chunksize=10)):
        predictions_idx.append(idx)
    pool.close()
    pool.join()

    hits_per_k = reduce(lambda a, idx: a + np.array([1 if idx is not None and k >= idx else 0
                                                     for k in k_values_to_test]),
                        predictions_idx,
                        np.zeros(len(k_values_to_test)))
    hits_per_k = hits_per_k / len(eval_data)

    print(f'Process {os.getpid()} (MASTER): Computed results:')
    print(str(['{:.2%}'.format(rate) for rate in hits_per_k]))
    print(f'Process {os.getpid()} (MASTER): Finished. Bye!')
