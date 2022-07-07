from abc import ABC, abstractmethod
import torch.multiprocessing as mp


class AbstractOSHandler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_mp_queue(self) -> mp.Queue:
        """ Creates a torch.multiprocessing queue. """
        ...

    @abstractmethod
    def get_num_workers(self) -> int:
        """ Returns the number of workers usable for e.g. DataLoader. """
        ...
