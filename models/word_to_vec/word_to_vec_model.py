import torch


class WordToVecModel(torch.nn.Module):
    """
    This model is a neural network containing exactly one hidden layer.
    The model is trained to guess an output word for the respective input word.
    In case of a skip gram model, context words are guessed from center words; for a CBOW model vice versa.
    The word embeddings we actually want to obtain will be the output of the hidden layer.
    """

    def __init__(self, vocabulary_size: int, embedding_size: int):
        """
        Initializes the network used for training word2vec.
        Parameters:
            vocabulary_size (int): Number of available words.
            embedding_size (int): The desired dimensionality of the vector embeddings.
        """
        self.__vocabulary_size = vocabulary_size
        self.__embedding_size = embedding_size
        super(WordToVecModel, self).__init__()
        # the model has two fully connected layers (fc1 and fc2)
        self.fc1 = torch.nn.Linear(vocabulary_size, embedding_size)
        self.fc2 = torch.nn.Linear(embedding_size, vocabulary_size)

    @staticmethod
    def create_by_layer(layer: torch.nn.Linear):
        """
        Creates a Word2VecModel based on the given fully-connected layer as first hidden layer.
        :param layer: first fully-connected layer of the Word2VecModel
        :return: created Word2VecModel
        """
        self = WordToVecModel(layer.in_features, layer.out_features)
        self.fc1 = layer
        return self

    def get_embedding_size(self) -> int:
        """
        Returns the size of the hidden layer (embedding size).
        :return: size of embedding
        """
        return self.__embedding_size

    def get_vocabulary_size(self) -> int:
        """
        Returns the size of the vocabulary (size of input and output layer).
        :return: size of vocabulary
        """
        return self.__vocabulary_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward the input Tensor through the network. """
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_embedding_for_one_hot_input(self, input_index: int) -> torch.Tensor:
        """
        Returns the hidden activation (embedding) for a given one-hot input represented by the index of the 1 in the
        one-hot vector.
        :param input_index: index of the 1 in the one-hot vector as input
            (e.g. index is 1 for one-hot vector [0, 1, 0, ...])
        :return: hidden activation (embedding)
        """
        # Since the dimension of the weight matrix is (embedding_size, vocabulary_size), we need to transpose if first.
        weight_matrix = self.fc1.weight.clone().t()
        bias_vector = self.fc1.bias.clone()
        return (weight_matrix[input_index] + bias_vector).detach()
