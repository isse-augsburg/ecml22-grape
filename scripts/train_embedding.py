from common.catalog import Catalog
from common.component_identifier import ComponentIdentifier
from data_set_generator.grams_from_graphs import GramsFromGraphs
from models.word_to_vec.embedding_handler import EmbeddingHandler
import options as op
from util import file_handler

import itertools
import logging
import numpy
import os
import sys
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

torch.manual_seed(0)
numpy.random.seed(0)

if __name__ == '__main__':
    identifier = ComponentIdentifier.COMPONENT
    catalog = Catalog.A   # Catalog.B, Catalog.C
    logger.info(f'{catalog.value}')
    embedding_sizes = [20, 100]
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]

    training_epochs = 1000
    create_new_grams = False

    data_path = op.DATA_LOCATION
    logger.info(f'data path {data_path}')
    data_path = f'{data_path}{catalog.value}_'
    # ------------------------------------------------------------------------------------------------------------------
    # Create samples for comp2vec
    train_grams_file = f'{data_path}{identifier.value}_train_grams.txt'
    val_grams_file = f'{data_path}{identifier.value}_val_grams.txt'

    if not os.path.exists(train_grams_file) or create_new_grams:
        logger.info('Creating new grams')
        # Use training and validation data from challenge for training; test data for validation
        train_graphs = file_handler.deserialize(f'{data_path}graphs_train.dat')

        # add graphs from train and validation data from challenge for training
        train_graphs = list(itertools.chain(train_graphs, file_handler.deserialize(f'{data_path}graphs_val.dat')))

        # same for validation data: use test data from challenge for validation
        val_graphs = file_handler.deserialize(f'{data_path}graphs_test.dat')
        logger.info('Graph files loaded.')

        # Transform list of graphs to data instances (grams)
        generator = GramsFromGraphs(identifier=identifier)
        generator.graphs_to_data_set(train_graphs, train_grams_file)
        logger.info('Train grams created')
        generator = GramsFromGraphs(identifier=identifier)
        generator.graphs_to_data_set(val_graphs, val_grams_file)
        logger.info('Val grams created')

    # Use vocabulary of all components to train an embedding. If we would be only using the components contained in
    # train and val set, we could be confronted with unknown components in the test set.
    vocabulary = [component.get(identifier) for component in file_handler.deserialize(f'{data_path}vocabulary.dat')]
    # Train embeddings for each defined embedding size based on created grams
    for embedding_size in embedding_sizes:
        handler = EmbeddingHandler()
        handler.tune_embedding_by_loss(training_file_path=train_grams_file, validation_file_path=val_grams_file,
                                       max_epochs=training_epochs, embedding_sizes=[embedding_size],
                                       learning_rates=learning_rates, vocabulary=vocabulary)

        embedding_file = f'{data_path}{identifier.value}_embedding_{embedding_size}.dat'
        handler.save_embedding(embedding_file)
    logger.info('Done')
