from util.abstract_os_handler import AbstractOSHandler

import os
import torch.multiprocessing as mp


class LinuxHandler(AbstractOSHandler):
    """ Concrete OSHander for Linux."""

    def create_mp_queue(self) -> mp.Queue:
        return mp.get_context('spawn').Queue()

    def get_num_workers(self) -> int:
        return int(os.getenv("NUM_WORKER", 1)) - 1
