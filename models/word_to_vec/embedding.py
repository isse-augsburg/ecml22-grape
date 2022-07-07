import itertools
import logging
import numpy
import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Union, Tuple

from models.word_to_vec.grams_dataset import GramsDataset
from models.word_to_vec.word_to_vec_model import WordToVecModel
from util import file_handler

logger = logging.getLogger(__name__)


def collate_fn(batch_list: List[Tuple[torch.Tensor, torch.Tensor]]):
    input = torch.cat([instance[0] for instance in batch_list])
    target = torch.cat([instance[1] for instance in batch_list])
    return input, target


class Embedding:

    def __init__(self):
        self.__model = None
        self.__vocabulary: List[str] = None
        self.__word2index: Dict[str, int] = None
        self.__index2word: Dict[int, str] = None
        self.__training_dataset: Dataset = None
        self.__validation_dataset: Dataset = None
        self.__dimension: int = None
        self.__embedded_vectors: Dict[str, torch.Tensor] = None

        self.__batch_size: int = 1024
        # On Windows, 0 worker should be used for DataLoader
        self.__num_worker = 0 if os.name == 'nt' else 4

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle self.__embedded_vectors, they can be recomputed
        del state["_Embedding__embedded_vectors"]  # Use pop for safe delete
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add self.__embedded_vectors
        self.__embedded_vectors = None

        self.__batch_size: int = 1024
        self.__num_worker = 0 if os.name == 'nt' else 4

    def get_word_to_index(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping each word of the vocabulary
        to its index in the bag of word representation.
        Returns:
            word2index (dict): dictionary containing the mapping
        """
        return self.__word2index

    def word_to_index(self, word: str) -> int:
        """
        Returns the index of a given word in the bag of word representation.
        Parameters:
            word (str): word
        Returns:
            index (int): Index of the given word if it is in the vocabulary
        """
        if word in self.__vocabulary:
            return self.__word2index[word]
        return -1

    def get_index_to_word(self) -> Dict[int, str]:
        """
        Returns a dictionary mapping each index of a word in the bag of word
        representation to the corresponding word in the vocabulary.
        Returns:
            index2word (dict): dictionary containing the mapping
        """
        return self.__index2word

    def index_to_word(self, index: int) -> str:
        """
        Returns the word of the vocabulary for a given index in the bag of word representation.
        Parameters:
            index (int):
        Returns:
            word (str): Index of the given word if it is in the vocabulary
        """
        if index in self.__vocabulary:
            return self.__index2word[index]
        return None

    def set_model(self, model: torch.nn.Module):
        """ Sets the internal variable 'model'. Only use for testing!  """
        self.__model = model

    def get_model(self) -> torch.nn.Module:
        """ Returns the ml model """
        return self.__model

    def set_vocabulary(self, vocabulary: List[str]):
        """ Sets the internal variable 'vocabulary'. Only use for testing!  """
        self.__vocabulary = vocabulary

    def get_vocabulary(self) -> List[str]:
        """ Returns the vocabulary """
        return self.__vocabulary

    def __get_embedded_vectors(self, model: WordToVecModel) -> Dict[str, torch.Tensor]:
        """
        Returns the vector coordinates in the embedding space of all words in the vocabulary for a given embedding
        model.
        Parameters:
            model (WordToVecModel): embedding model
        """
        return {word: model.get_embedding_for_one_hot_input(index) for (index, word) in enumerate(self.__vocabulary)}

    def get_embedded_vectors(self) -> Dict[str, torch.Tensor]:
        """
        Returns the vector coordinates in the embedding space of all words in the vocabulary, they are newly calculated
        if None.
        """
        if self.__embedded_vectors is None:
            self.__embedded_vectors = self.__get_embedded_vectors(self.__model)
        return self.__embedded_vectors

    def get_embedding_dimension(self) -> int:
        """ Returns the dimension of the embedded vectors. """
        return self.__dimension

    def __create_index_word_maps(self):
        """ Creates the maps __index2word and __word2index based on the private attribute __vocabulary. """
        self.__word2index = {word: index for (index, word) in enumerate(self.__vocabulary)}
        self.__index2word = {index: word for (index, word) in enumerate(self.__vocabulary)}

    def __prepare_samples__(self, training_file_path: str, validation_file_path: str = None,
                            vocabulary: List[str] = None, separator: str = '|||'):
        """
        This function obtains pre-generated training data from a file with lines of the form:
            center-word <separator> context-word
        Every line forms a new training sample.
        :param training_file_path: Path to the file containing the training samples (grams)
        :param validation_file_path (optional): Path to the file containing the validation samples (grams)
        :param vocabulary: vocabulary to use
        :param separator: The string by which each line in the file containing the training data should be split.
        """
        training_file_data = file_handler.load_string(training_file_path)
        training_samples = [line.split(separator) for line in training_file_data.split('\n')]
        cleaned_training_samples = [tuple(sample) for sample in training_samples if len(sample) == 2]
        logger.info('Loaded %d clean training samples (of %d samples).',
                    len(cleaned_training_samples), len(training_samples))

        if validation_file_path:
            validation_file_data = file_handler.load_string(validation_file_path)
            validation_samples = [line.split(separator) for line in validation_file_data.split('\n')]
            cleaned_validation_samples = [tuple(sample) for sample in validation_samples if len(sample) == 2]
            logger.info('Loaded %d clean validation samples (of %d samples).',
                        len(cleaned_validation_samples), len(validation_samples))
        else:
            cleaned_validation_samples = []   # empty validation samples if no file was given

        # Extract the vocabulary
        if vocabulary:
            # assure the given vocabulary contains all words of train and val grams
            # (i.e. check if all tokens in the train and val grams files are included in the given vocabulary)
            assert set(itertools.chain.from_iterable(cleaned_training_samples + cleaned_validation_samples)) \
                .issubset(set(vocabulary)), \
                'Some tokens of the given train and val gram files are not included in the given vocabulary. '
            self.__vocabulary = vocabulary
            logger.info('Given vocabulary set. Vocabulary size: %d.', len(vocabulary))
        else:
            self.__vocabulary = list(set(itertools.chain.from_iterable(
                cleaned_training_samples + cleaned_validation_samples)))
            logger.info('Vocabulary extracted. Vocabulary size: %d.', len(self.__vocabulary))

        # Indices are generated in terms of vocabulary position.
        self.__create_index_word_maps()

        if validation_file_path:
            intersection = set(cleaned_training_samples).intersection(set(cleaned_validation_samples))
            logger.info(f'{len(intersection) / len(cleaned_validation_samples) * 100:.2f}% of validation data '
                        f'is included in training data.')
            self.__validation_dataset = GramsDataset(cleaned_validation_samples, self.__vocabulary,
                                                     self.__word2index, self.__index2word)

        # Transform raw training samples into pytorch readable data
        self.__training_dataset = GramsDataset(cleaned_training_samples, self.__vocabulary,
                                               self.__word2index, self.__index2word)
        logger.info(f'Training{" and validation" if validation_file_path else ""} data prepared.')

    def __transform_samples(self, cleaned_samples):
        """ Transforms the given string-based samples into PyTorch readable samples (tensors).
        :param cleaned_samples: List of string-based samples
        :return: list of tuples representing the words as tensors.
        """
        data = []
        for (input_word, target_word) in cleaned_samples:
            input_one_hot = torch.zeros(1, len(self.__vocabulary))
            input_one_hot[0][self.__word2index[input_word]] = 1.0

            # Generate the index for the context word and store it in an 1D-tensor of type long.
            # NLLLoss from the Torch-API doesn't accept a second one-hot-vector
            # as a target, but only a classification number of type long.
            target_word_index = self.__word2index[target_word]
            target_tensor = torch.tensor(target_word_index).view(1).long()

            data.append((input_one_hot, target_tensor))
        return data

    def __train_embedding_by_loss(self, embedding_size: int, max_epochs: int, learning_rate: float):
        """
        This function generates a word embedding specified by the given hyperparameters. The training is continued until
        the maximal number of epochs is reached or if no progress in learning is achieved within the last 20 epochs.
        Parameters:
        :param embedding_size (int): Dimensionality of the word embeddings (can be freely chosen, but should be
                                    considerably smaller than the expected vocabulary size).
        :param max_epochs (int): The maximal amount of times the neural net is trained with the training data.
        :param learning_rate (float): The learning rate for the optimizer that trains the model.
        :return: A python dictionary where the keys are the words in the vocabulary and the values are the
                corresponding word embeddings as 1D-Tensors.
        """
        assert max_epochs > 0, 'Number of training epochs needs to be positive.'
        assert 0 < learning_rate < 1, 'Learning rate needs to lie in the open interval (0, 1).'

        assert self.__validation_dataset is not None, \
            'For training an embedding by loss, a validation set needs to be given.'

        logger.info(f'Training Word2Vec model with {embedding_size} dimensions and learning rate {learning_rate} ...')

        model = WordToVecModel(len(self.__vocabulary), embedding_size)
        model = torch.nn.DataParallel(model)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        training_losses = []
        validation_losses = []

        best_model = None
        best_epoch = None
        best_selection_loss = numpy.infty

        logger.info(f'Training on {len(self.__training_dataset)} training '
                    f'and validating on {len(self.__validation_dataset)} validation samples.')

        # The loss function in use is the negative log-likelihood
        log_transform = torch.nn.LogSoftmax(dim=1)  # dim specifies batch size
        loss_function = torch.nn.NLLLoss()
        # The used optimization algorithm is Stochastic Gradient Descent
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        data_loader_train = DataLoader(self.__training_dataset, shuffle=True, batch_size=self.__batch_size,
                                       num_workers=self.__num_worker, collate_fn=collate_fn)
        data_loader_val = DataLoader(self.__validation_dataset, shuffle=True, batch_size=self.__batch_size,
                                     num_workers=self.__num_worker, collate_fn=collate_fn)

        for epoch in range(1, max_epochs + 1):
            if epoch % 100 == 0:
                logger.info('Processing epoch %d', epoch)

            sum_loss = 0.0
            for batch_idx, (input_vec, target_idx) in enumerate(data_loader_train):
                # if batch_idx % 10 == 0:
                #     logger.info('Processing training batch %d', batch_idx)
                input_vec, target_idx = input_vec.to(device), target_idx.to(device)
                # gradient buffers need to be empty in order to not distort the result
                optimizer.zero_grad()
                output_activation = model(input_vec)

                # The reason that log_softmax is used instead of softmax,
                # is that NLLLoss from the Torch-API assumes log_softmax to be used.
                output_vec = log_transform(output_activation)
                loss = loss_function(output_vec, target_idx)
                sum_loss += loss.item()

                # Backpropagation and weight adaption
                loss.backward()
                optimizer.step()

            # evaluate current model on training and validation set
            with torch.no_grad():
                model.eval()
                train_loss = self.__evaluate_on(model, data_loader_train, loss_function, log_transform)
                training_losses.append(train_loss)
                val_loss = self.__evaluate_on(model, data_loader_val, loss_function, log_transform)
                validation_losses.append(val_loss)

                selection_loss = train_loss + numpy.abs(val_loss - train_loss)
                if selection_loss < best_selection_loss:
                    best_model = model.module  # get model of DataParallel
                    best_epoch = epoch
                    best_selection_loss = selection_loss

                # logger.info('train loss', train_loss, '; val loss', val_loss)
                model.train()

            if best_epoch + 20 < epoch:
                # stop tuning if model is not improving since 20 epochs
                break

            logger.debug('Loss value: %f', sum_loss / len(self.__training_dataset))
        logger.info('Training completed.')

        logger.info(f'training_losses {training_losses}')
        logger.info(f'validation_losses {validation_losses}')
        logger.info(f'Best Epoch {best_epoch}')

        return best_model, selection_loss

    def __evaluate_on(self, model: torch.nn.Module, data_set: DataLoader, loss_function, log_transform):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        summed_loss = 0.0
        for _, (input_vec, target_idx) in enumerate(data_set):
            input_vec, target_idx = input_vec.to(device), target_idx.to(device)
            prediction = model(input_vec)
            # transform prediction scores to log probabilities
            output = log_transform(prediction)
            loss = loss_function(output, target_idx)
            summed_loss += loss.item()
        return summed_loss / len(data_set)

    @staticmethod
    def tune_embedding_by_loss(training_file_path: str,
                               validation_file_path: str,
                               max_epochs: int,
                               embedding_sizes: Union[int, List[int]],
                               learning_rates: Union[float, List[float]],
                               vocabulary: List[str]) -> 'Embedding':
        """
         This function generates a tuned word embedding based on the loss of the model. For each given hyper-parameter
         (embedding size, learning rate) combination a model is trained, among which the best performing one according
         to the loss is chosen as tuned embedding model.
         :param training_file_path: Path to the file containing the training samples (grams)
         :param validation_file_path: Path to the file containing the validation samples (grams)
         :param max_epochs: Maximal number of epochs due to the usage of early-stopping
         :param embedding_sizes: List of hyper-parameter values for the embedding dimension
         :param learning_rates: List of hyper-parameter values for the learning rate
         :param vocabulary: vocabulary of words
         :return:
         """
        self = Embedding()  # create new embedding instance
        self.__prepare_samples__(training_file_path, validation_file_path, vocabulary)
        self.__training_params__(embedding_sizes=embedding_sizes, learning_rates=learning_rates)
        logger.info('Tuning new embedding ...')

        best_model = None
        best_parameters = None
        best_model_loss = numpy.infty

        for embedding_size in self.embedding_sizes:
            for learning_rate in self.learning_rates:
                trained_model, model_loss = self.__train_embedding_by_loss(embedding_size, max_epochs, learning_rate)
                assert trained_model is not None, 'Trained model should not be None.'

                if model_loss < best_model_loss:
                    best_model = trained_model
                    best_model_loss = model_loss
                    best_parameters = {'embedding_size': embedding_size, 'learning_rate': learning_rate}

        self.__model = best_model if best_model is not None else trained_model   # assure self.__model is not None
        self.__dimension = self.__model.get_embedding_size()
        # torch.save(trained_model, f'word2vec_model_{best_parameters["embedding_size"]}.dat')

        logger.info('Found best embedding with hyperparameters: %s',
                    f'Embedding Dimension: {best_parameters["embedding_size"]} '
                    f'and Learning Rate: {best_parameters["learning_rate"]}')
        return self

    def __training_params__(self, embedding_sizes: Union[int, list], learning_rates: Union[float, list]):
        """
            User is able to input GridSearch parameters or SingleLearning parameters. Such as;
            GridSearch is a sequential data type e.g. list = [a, b, c].
            SingleLearning  is a non-sequential data type float and int
        Parameters:
            embedding_sizes (int, list): Input hyper parameter embedding dimensions
            learning_rates (float, list): Input hyper parameter learning rates
        """
        assert isinstance(embedding_sizes, list) or isinstance(embedding_sizes, int),\
            'Input type should either be list or int'
        assert isinstance(learning_rates, list) or isinstance(learning_rates, float),\
            'Input type should either be list or float'
        input_parameters = (embedding_sizes, learning_rates)
        list_of_params = []
        for parameter in input_parameters:
            # All possible inputs types are conditioned
            if isinstance(parameter, list):
                range_of_parameter = parameter
            else:
                range_of_parameter = [parameter]
                assert len(range_of_parameter) == 1, 'Non-sequential data types should be converted to a single-list'
            list_of_params.append(range_of_parameter)

        self.embedding_sizes = list_of_params[0]
        self.learning_rates = list_of_params[1]

    @staticmethod
    def create_one_hot(vocabulary: List[str]) -> 'Embedding':
        """
        This class function creates a one-hot embedding (the embedding vectors are one-hot vectors) based on a given
        vocabulary. No model needs to be trained for this.
        :param vocabulary: List of words that should be represented as one-hot-vectors.
        :return: the created one-hot embedding
        """
        self = Embedding()  # create new embedding instance
        logger.info('Creating one-hot embedding ...')

        # sort vocabulary to create deterministic one-hot-encoding for same vocabulary
        sorted_vocabulary = sorted(set(vocabulary))
        self.__vocabulary = sorted_vocabulary
        self.__dimension = len(sorted_vocabulary)

        # create one-hot-encoding
        self.__create_index_word_maps()

        layer = torch.nn.Linear(self.__dimension, self.__dimension)
        layer.bias = torch.nn.Parameter(torch.zeros(layer.out_features))
        layer.weight = torch.nn.Parameter(torch.eye(layer.in_features))
        self.__model = WordToVecModel.create_by_layer(layer)

        # No need to set datasets since they are only used in train_embedding() and tune_embedding()
        self.__training_dataset = None
        self.__validation_dataset = None
        logger.info('One-hot embedding created.')

        return self
