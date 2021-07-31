"""Some utilities.
"""
from torch import nn

def classification_block(inp_dim, out_dim, dropout=0.0,
                             batch_norm=False) -> nn.Sequential:
    """Creates a classification block.

    Args:
        inp_dim ([type]): input dimension to the block.
        out_dim ([type]): output dimension to the block.
        dropout (float, optional): if greater than 0.0, adds a dropout layer with
            given probability. Defaults to 0.0.
        batch_norm (bool, optional): if true, adds a batchnorm layer. Defaults to False.

    Returns:
        [nn.Sequential]: block combining Linear + BatchNorm + ReLU + Dropout
    """
    layers = []

    layers.append(nn.Linear(in_features=inp_dim, out_features=out_dim))

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))

    layers.append(nn.ReLU(inplace=True))

    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)
