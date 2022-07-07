import logging
from typing import List, Union

from models.word_to_vec.embedding import Embedding
from util import file_handler

logger = logging.getLogger(__name__)


class EmbeddingHandler:
    def __init__(self):
        self.__embedding = None

    def tune_embedding_by_loss(self, training_file_path: str,
                               validation_file_path: str,
                               max_epochs: int,
                               embedding_sizes: Union[int, list],
                               learning_rates: Union[float, list],
                               vocabulary: List[str] = None):
        """
         This function generates a tuned word embedding based on the loss of the model.
         :param training_file_path:  Path to the file containing the training samples (grams)
         :param validation_file_path:  Path to the file containing the validation samples (grams)
         :param max_epochs: Maximal number of epochs due to the usage of early-stopping
         :param embedding_sizes: Hyperparameter embedding dimension
         :param learning_rates: Hyperparameter learning rate
         :param vocabulary: (optional) vocabulary to use
         :return:
         """
        self.__embedding = Embedding.tune_embedding_by_loss(training_file_path, validation_file_path, max_epochs,
                                                            embedding_sizes, learning_rates, vocabulary)

    def create_one_hot_embedding(self, vocabulary: List[str]):
        """
        Generates a one-hot word embedding based on the provided vocabulary.
        :param vocabulary: List of words
        """
        self.__embedding = Embedding.create_one_hot(vocabulary)

    def load_embedding(self, filename: str) -> Embedding:
        """
        Loads an embedding from a given file.
        Parameters:
            filename (str): path of the file to load
        Returns:
            embedding (Embedding): embedding
        """
        logger.info(f'Embedding loaded from {filename}')
        self.__embedding = file_handler.deserialize(filename)
        return self.__embedding

    def save_embedding(self, filename: str):
        """
        Saves the internal embedding into a file.
        Parameters:
            filename (str): The name of the file in which the embedding model
                is to be stored.
        """
        file_handler.serialize(self.__embedding, filename)
        logger.info(f'Embedding saved to {filename}')

    def set_embedding(self, embedding: Embedding):
        """ Sets the internal embedding. Only use for testing!  """
        self.__embedding = embedding

    def get_embedding(self):
        """ Returns the internal embedding (Embedding) """
        return self.__embedding
