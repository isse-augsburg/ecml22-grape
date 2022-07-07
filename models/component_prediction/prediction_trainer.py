import dgl
import logging
import matplotlib.pyplot as plt
import numpy as np
import time as time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple

from util.abstract_os_handler import AbstractOSHandler


logger = logging.getLogger(__name__)


def collate(samples: List):
    """
    Collates a list of samples. Needed for batched training
    :param samples: a list of pairs (graph, label).
    :return:
    """
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels)


class PredictionTrainer:
    """
        A class to train a prediction model based on early-stopping.
    """

    def __init__(self,
                 model: nn.Module,
                 train_set: Dataset,
                 val_set: Dataset,
                 model_name: str,
                 max_epochs: int,
                 learning_rate: float,
                 os_handler: AbstractOSHandler):

        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.os_handler = os_handler

        logger.info(f'Training with {len(self.train_set)} graphs. Validating on {len(self.val_set)} graphs.')

        self.batch_size = 1024

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []
        self.best_epoch = None

    def train(self, device: torch.device) -> Tuple[float, float, int]:
        """
        The training method.
        :param device: torch.device used for training (cpu or gpu)
        """

        num_workers = self.os_handler.get_num_workers()
        data_loader_train = DataLoader(self.train_set,
                                       batch_size=self.batch_size,
                                       collate_fn=collate,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       persistent_workers=num_workers != 0,
                                       pin_memory=True)
        data_loader_val = DataLoader(self.val_set,
                                     batch_size=self.batch_size,
                                     collate_fn=collate,
                                     shuffle=False,  # no need to shuffle val data
                                     num_workers=num_workers,
                                     persistent_workers=num_workers != 0,
                                     pin_memory=True)

        # activate train mode (would be relevant for dropout/batch norm)
        self.model.train()
        best_selection_loss = np.infty

        for epoch in range(1, self.max_epochs + 1):  # start counting at 1
            print(f'Epoch {epoch}')

            # TRAIN
            epoch_train_loss = 0.0
            train_time = time.time()
            for _, (batched_graphs, batched_labels) in enumerate(data_loader_train):
                # data to device
                batched_graphs, batched_labels = batched_graphs.to(device), batched_labels.to(device)
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self.model(batched_graphs)
                loss = self.loss_function(prediction, batched_labels)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.detach()

            # sum up train loss and average per instance
            print(f'Train time {(time.time() - train_time) / 60:.2f}min')
            train_loss_per_instance = epoch_train_loss.item() / len(self.train_set)
            self.train_losses.append(train_loss_per_instance)

            # VAL
            self.model.eval()  # activate eval mode
            epoch_val_loss = 0.0
            val_time = time.time()
            for _, (batched_graphs, batched_labels) in enumerate(data_loader_val):
                batched_graphs, batched_labels = batched_graphs.to(device), batched_labels.to(device)
                predictions = self.model(batched_graphs)
                loss = self.loss_function(predictions, batched_labels)
                epoch_val_loss += loss.detach()

            # sum up val loss and average per instance
            print(f'Val time {(time.time() - val_time) / 60:.2f}min')
            val_loss_per_instance = epoch_val_loss.item() / len(self.val_set)
            self.val_losses.append(val_loss_per_instance)

            # early stopping
            selection_loss = train_loss_per_instance + np.abs(val_loss_per_instance - train_loss_per_instance)
            if selection_loss < best_selection_loss:
                best_epoch = epoch
                best_selection_loss = selection_loss

            self.model.train()  # re-activate train mode

            print(f'{self.model_name} Epoch {epoch} | train loss {epoch_train_loss:.4f} | '
                  f'val loss {epoch_val_loss:.4f}')

            if best_epoch + 20 < epoch:
                print(f'break training: {best_epoch + 20} < {epoch}', device)
                break

        self.best_epoch = best_epoch

        return self.train_losses[self.best_epoch - 1], self.val_losses[self.best_epoch - 1], self.best_epoch

    def save(self, model_output_file, model_plot_file):
        # save it for later
        torch.save(self.model.state_dict(), model_output_file)

        plt.figure()
        plt.title(f'cross entropy averaged over samples of {self.model_name}')
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="train_loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="val_loss")
        plt.vlines(x=self.best_epoch, colors='c', ymin=min(self.train_losses + self.val_losses),
                   ymax=max(self.train_losses + self.val_losses), label='best epoch')
        plt.legend(loc=0)
        plt.savefig(model_plot_file)
        plt.show()
