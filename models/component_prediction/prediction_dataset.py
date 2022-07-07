import dgl
import h5py
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple

import options as op


class PredictionDataset(Dataset):
    """
        A PyTorch map-style dataset of component prediction samples (partial graph, component).
        :param hdf5_filename: path to hdf5 file
        :param dataset_name: name of dataset (allowed values: 'train', 'val', 'test')
        :param node_feature_size (optional): Resulting size of node features. Used for one-hot embeddings to transform
            index to dense representation with one-hot vector.
    """
    def __init__(self,
                 hdf5_filename: str,
                 dataset_name: str,
                 node_feature_size: int = None):

        super(PredictionDataset).__init__()
        self.__hdf5_filename: str = hdf5_filename
        self.__dataset_name: str = dataset_name
        self.__embedding_keyword: str = op.EMBEDDING_KEYWORD
        self.__node_feature_size: int = node_feature_size

        with h5py.File(self.__hdf5_filename, 'r') as f:
            self.__size: int = f[self.__dataset_name].attrs["size"]

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        with h5py.File(self.__hdf5_filename, 'r') as f:
            serial = f[self.__dataset_name][idx].tobytes()

        graph, label = pickle.loads(serial)

        if self.__node_feature_size:
            node_features = graph.ndata[self.__embedding_keyword]   # shape (len(graph_nodes), 1)
            graph.ndata[self.__embedding_keyword] = F.one_hot(node_features.squeeze(1).long(),
                                                              num_classes=self.__node_feature_size)
        return graph, label

    def __len__(self) -> int:
        return self.__size
