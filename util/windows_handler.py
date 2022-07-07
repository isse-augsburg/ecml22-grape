from util.abstract_os_handler import AbstractOSHandler
import torch.multiprocessing as mp


class WindowsHandler(AbstractOSHandler):
    """ Concrete OSHander for Windows."""

    def create_mp_queue(self) -> mp.Queue:
        return mp.Queue()

    def get_num_workers(self) -> int:
        return 0
