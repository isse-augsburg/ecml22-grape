import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


class GramsDataset(Dataset):
    """ A PyTorch Dataset containing samples of the form <center, context>."""

    def __init__(self, grams_tuples: List[Tuple[str, str]],
                 vocabulary: List[str],
                 word2index: Dict[str, int],
                 index2word: Dict[int, str]):
        self.__grams_tuples = grams_tuples
        self.__vocabulary = vocabulary
        self.__word2index = word2index
        self.__index2word = index2word

    def __len__(self) -> int:
        return len(self.__grams_tuples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.__grams_tuples[index]
        if len(sample) != 2:
            return None

        # transform to pytorch-readable data
        input_word, target_word = sample
        input_one_hot = torch.zeros(1, len(self.__vocabulary))
        input_one_hot[0][self.__word2index[input_word]] = 1.0

        target_word_index = self.__word2index[target_word]
        target_tensor = torch.tensor(target_word_index).view(1).long()

        return input_one_hot, target_tensor
