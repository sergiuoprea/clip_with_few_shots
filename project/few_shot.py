"""Implementation of FewShot classifier.
"""
# Standard Python imports
from argparse import ArgumentParser

# Pytorch-related imports
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

# CLIP
import clip

# For the confussion matrix
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Utils
from .utils import classification_block

# Name position determines class idx
CLASS_NAMES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'truck']

class BasicClassifier(nn.Module):
    """Simple 3-layer classifier.
    """
    def __init__(self, out_classes: int = 8, input_dim: int = 512, hidden_dim: int = 256):
        """Initializing the basic classifier.

        Args:
            out_classes (int, optional): number of output classes represented in CLASS_NAMES.
                Defaults to 8.
            input_dim (int, optional): feature dimension extracted from CLIP network.
                Defaults to 512.
            hidden_dim (int, optional): first hidden dimension. Defaults to 256.
        """
        super().__init__()

        self.block0 = classification_block(input_dim, hidden_dim,
                                                dropout=0.2, batch_norm=True)
        self.block1 = classification_block(hidden_dim, hidden_dim // 2,
                                                dropout=0.2, batch_norm=True)
        self.block2 = classification_block(hidden_dim // 2, out_classes)

    def forward(self, inp: torch.HalfTensor) -> torch.HalfTensor:
        """Forward pass.

        Args:
            inp (torch.HalfTensor): input features from CLIP net.

        Returns:
            torch.HalfTensor: classification net output.
        """
        inp = self.block0(inp)
        inp = self.block1(inp)
        inp = self.block2(inp)

        return inp

class FewShot(pl.LightningModule):
    """FewShot classification model.
    """

    def __init__(self, backbone: str = "ViT-B/16", num_classes: int = 8,
                learning_rate: float = 1e-3, log_freq: int = 10):
        """Initializing the model.

        Args:
            backbone (str, optional): CLIP model used as backbone. Defaults to ViT-B/16.
            num_classes (int, optional): number of output classes. Defaults to 8.
            learning_rate (float, optional): learning rate used in the training process. Defaults to 1e-3.
            log_freq (int, optional): log stuff each log_freq batches. Defaults to 1.
        """
        super(FewShot, self).__init__()

        print("====> initializing FewShot classifier model...")
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.log_freq = log_freq

        # Instance of CLIP model used as a backbone
        self.backbone, self.preprocess = clip.load(backbone)

        # Freeze the backbone. We don't want to train the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Instance of a basic classifier
        self.classifier = BasicClassifier(out_classes= num_classes)

        # Classification loss
        self.loss = nn.CrossEntropyLoss()

        # Accudarcy metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy(compute_on_step=False)

        # For confussion matrix
        self.outs = []
        self.targs = []

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Model specific-hyperparameters.

        Args:
            parent_parser (ArgumentParser): parent parser.

        Returns:
            ArgumentParser: parser updated with the new arguments.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--no_logger', action='store_true',
                            help='if true, log stuff in Neptune.')
        parser.add_argument('--max_epochs', type=int, default=300,
                            help='maximum number of training epochs.')
        parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='learning rate used for training.')

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx) -> torch.FloatTensor:
        """Training forward pass.

        Args:
            batch ([type]): input batch of images and its corresponding classes.
            batch_idx ([type]): batch global index.

        Returns:
            loss [torch.FloatTensor]: training loss value.
        """
        _x, _y = batch
        _z = self.backbone.encode_image(_x)

        _out = self.classifier(_z)
        _loss = self.loss(_out, _y)

        self.train_accuracy.update(_out, _y)

        if batch_idx % self.log_freq == 0:
            self.logger.experiment.log_metric('train/loss', _loss)

        return _loss

    def on_train_epoch_end(self) -> None:
        """Callback executed at the end of each training epoch.
        After each epoch. Metrics are reseted automatically.
        """
        # get the accuracy over all batches
        _global_acc = self.train_accuracy.compute()
        print(f"Global training accuracy: {_global_acc}")
        self.logger.experiment.log_metric('train/acc_per_epoch', _global_acc)

    def validation_step(self, batch, batch_idx) -> torch.FloatTensor:
        """Validation forward step.

        Args:
            batch ([type]): input batch of images and its corresponding classes.
            batch_idx ([type]): batch global index.

        Returns:
            loss [torch.FloatTensor]: validation loss value.
        """
        _x, _y = batch
        _z = self.backbone.encode_image(_x)

        _out = self.classifier(_z)
        _loss = self.loss(_out, _y)

        self.valid_accuracy.update(_out, _y)

        if batch_idx % self.log_freq == 0:
            self.logger.experiment.log_metric('valid/loss', _loss)

        return _loss

    def on_validation_epoch_end(self) -> None:
        """Callback executed at the end of each validation epoch.
        After each epoch. Metrics are reseted automatically.
        """
        # get the accuracy over all batches
        _global_acc = self.valid_accuracy.compute()
        self.log('valid_global_acc', _global_acc)
        print(f"Global validation accuracy: {_global_acc}")
        self.logger.experiment.log_metric('valid/acc_per_epoch', _global_acc)

    def test_step(self, batch, batch_idx) -> None:
        """Test step.

        Args:
            batch ([type]): input batch of images and its corresponding classes.
            batch_idx ([type]): batch global index.
        """
        _x, _y = batch
        _z = self.backbone.encode_image(_x)

        _out = self.classifier(_z)
        self.test_accuracy.update(_out, _y)

        self.targs.extend(_y.cpu().numpy())
        self.outs.extend(torch.argmax(_out, dim=1).cpu().numpy())

    def on_test_epoch_end(self) -> None:
        """Callback executed at the end of testing epoch.
        """
        # get the accuracy over all batches
        _global_acc = self.test_accuracy.compute()
        print(f"Global test accuracy: {_global_acc}")
        self.logger.experiment.log_metric('test/acc_per_epoch', _global_acc)

        ## Get the confussion matrix
        _fig, _ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true=self.targs, y_pred=self.outs, labels=CLASS_NAMES, ax=_ax)
        self.logger.experiment.log_image('test/confussion_matrix', _fig)
