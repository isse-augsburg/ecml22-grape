"""
This class represents a perfect/ ideal model for component prediction that is used to give an estimate from above for
the top-k rates.
"""

from common.catalog import Catalog
from common.component_identifier import ComponentIdentifier
from preprocessing.graph import Graph
from preprocessing.component import Component
from util.file_handler import deserialize, serialize
import options as op

from collections import Counter
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


class UpperBoundPredictor:
    """
        Calculates the upper bound of the top-k rate a perfect, but non-oracle, model could reach.
        :param identifier: identifier of the components
        :param mapping:
    """

    def __init__(self,
                 identifier: ComponentIdentifier,
                 mapping: Dict[Graph, 'Counter[str]'] = None):

        self.__identifier = identifier
        self.__mapping: Dict[Graph, 'Counter[str]'] = mapping

    def create_mapping_from(self, training_data: List[Tuple[Graph, Component]]):
        print(f'preprocess {len(training_data)} {"test".upper()} graphs')

        mapping = dict()
        for g, p in training_data:
            mapping = extend_dict(mapping, g, [p.get(self.__identifier)])

        # transform value list to Counter with absolute frequencies
        # mapping: Graph -> Counter of ComponentIdentifier
        print('transform mapping')
        for key_ in mapping.keys():
            mapping[key_] = Counter(mapping[key_])
        self.__mapping = mapping
        print(f'# keys: {len(mapping.keys())}')

    def save_mapping(self, filename: str):
        serialize(self.__mapping, filename)

    def mapping_size(self):
        return len(self.__mapping.keys())

    def predict_best_k(self, input_graph: Graph, k_value: int) -> List[str]:
        counter = self.__mapping.get(input_graph)
        return [elem for elem, count in counter.most_common(k_value)]


def init_worker(m_file, max_size, comp_identifier, kvalues):
    global model, size, component_identifier, k_values_to_test
    size = max_size
    component_identifier = comp_identifier
    k_values_to_test = kvalues
    print(f'Process {os.getpid()}: loading mapping...')
    mapping = deserialize(m_file)
    model = UpperBoundPredictor(component_identifier, mapping)
    print(f'Process {os.getpid()}: Initialized.')


def evaluate_sample(sample):
    idx, (graph, component) = sample

    label = component.get(component_identifier)
    predictions = model.predict_best_k(graph, max(k_values_to_test))

    # evaluate
    if label in predictions:
        return predictions.index(label)


if __name__ == "__main__":

    catalog = Catalog.A  # Catalogs A, B and C
    component_identifier = ComponentIdentifier.COMPONENT
    print(f'Working on catalog {catalog.value} and Component ID {component_identifier.value}.')
    k_values_to_test = [1, 2, 3, 5, 10, 15, 20]
    eval_set = 'test'   # 'val'
    process_count = 8  # nbr of processes

    mapping_file = os.path.join(op.DATA_LOCATION, f'ub_mapping_{component_identifier.value}_{catalog.value}.dat')
    if not os.path.exists(mapping_file):
        print(f'Process {os.getpid()} (MASTER): create new mapping')
        data = deserialize(os.path.join(op.DATA_LOCATION, f'{catalog.value}_partial_graphs_{eval_set}.dat'))
        print('"Train" data loaded.')

        model = UpperBoundPredictor(component_identifier)
        model.create_mapping_from(data)
        model.save_mapping(mapping_file)
        del model, data
    else:
        print(f'Using already existing mapping {mapping_file}.')

    eval_data: List[Tuple[Graph, Component]] = deserialize(os.path.join(op.DATA_LOCATION,
                                                           f'{catalog.value}_partial_graphs_{eval_set}.dat'))
    print(f'Process {os.getpid()} (MASTER): start evaluation of {len(eval_data)} {eval_set.upper()} samples '
          f'on {process_count} processes...')

    mp.set_start_method('spawn')
    pool = mp.Pool(processes=process_count, initializer=init_worker,
                   initargs=[mapping_file, len(eval_data), component_identifier, k_values_to_test])

    predictions_idx = []
    for idx in tqdm.tqdm(pool.imap_unordered(evaluate_sample, enumerate(eval_data), chunksize=100)):
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
