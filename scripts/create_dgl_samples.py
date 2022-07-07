from common.catalog import Catalog
from common.component_identifier import ComponentIdentifier
from data_set_generator.partial_graphs import instance_to_dgl_instance
from models.word_to_vec.embedding import Embedding
from models.word_to_vec.embedding_handler import EmbeddingHandler
from preprocessing.graph import Graph
from preprocessing.component import Component
from util.file_handler import deserialize

import h5py
import logging
import numpy as np
import options as op
import pickle
import sys
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


if __name__ == "__main__":
    # PARAMs
    identifier = ComponentIdentifier.COMPONENT
    data_path = op.DATA_LOCATION
    logger.info(f'data path {data_path}')

    for catalog in [Catalog.A, Catalog.B, Catalog.C]:
        print('\nCatalog', catalog.value)

        for embedding_size in [20, 100, 'one_hot']:
            embedding_name = f'embedding_{embedding_size}' if str(embedding_size).isdigit() \
                else f'{embedding_size}_embedding'
            embedding_file = f'{data_path}{catalog.value}_{identifier.value}_{embedding_name}.dat'

            # load embedding and partial graphs data set
            embedding: Embedding = EmbeddingHandler().load_embedding(embedding_file)

            for set_name in ['train', 'val', 'test']:

                logger.info(f'Processing {set_name} set')
                partial_graphs_file = f'{data_path}{catalog.value}_partial_graphs_{set_name}.dat'
                partial_graphs: List[Tuple[Graph, Component]] = deserialize(partial_graphs_file)
                logger.info(f'{len(partial_graphs)} graphs.')

                serial_dgl_instances: List[bytes] = [pickle.dumps(instance_to_dgl_instance(inst, embedding, identifier,
                                                                                           embedding_size == 'one_hot'))
                                                     for inst in partial_graphs]

                with h5py.File(f'{data_path}{catalog.value}_{identifier.value}_{embedding_size}_dgl.hdf5', 'a') as f:
                    d = f.create_dataset(set_name, data=np.bytes_(serial_dgl_instances))
                    d.attrs['size'] = len(serial_dgl_instances)

        logger.info('Done.')
