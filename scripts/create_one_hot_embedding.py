from common.catalog import Catalog
from common.component_identifier import ComponentIdentifier
from models.word_to_vec.embedding_handler import EmbeddingHandler
from util.file_handler import deserialize
import options as op

import logging
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == '__main__':
    for catalog in [Catalog.A, Catalog.B, Catalog.C]:
        identifier = ComponentIdentifier.COMPONENT
        logger.info(f'Create One-Hot-Embedding for catalog {catalog.value} '
                    f'and PartIdentifier {identifier.value}')

        data_path = op.DATA_LOCATION

        # load vocabulary of all constructions
        vocabulary = deserialize(f'{data_path}{catalog.value}_vocabulary.dat')
        pid_vocabulary = [component.get(identifier) for component in vocabulary]

        handler = EmbeddingHandler()
        handler.create_one_hot_embedding(pid_vocabulary)

        # save embedding
        embedding_file = f'{data_path}{catalog.value}_{identifier.value}_one_hot_embedding.dat'
        handler.save_embedding(embedding_file)

    print('Done')
