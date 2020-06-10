from typing import Optional
import torch.nn as nn


class FullyConnectedBlock(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            normalization: Optional[str] = None,
            activation: Optional[str] = 'ReLU',
            preactivation: bool = False,
            dropout: float = 0,
            ):
        super().__init__()

        block = nn.ModuleList()
        fc_layer = nn.Linear(in_size, out_size)

        norm_layer = None
        if normalization is not None:
            class_name = '{}Norm1d'.format(normalization.capitalize())
            norm_class = getattr(nn, class_name)
            num_features = in_size if preactivation else out_size
            norm_layer = norm_class(num_features)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        if preactivation:
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)
            self.add_if_not_none(block, fc_layer)
        else:
            self.add_if_not_none(block, fc_layer)
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)

        if dropout:
            dropout_class = nn.Dropout
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)


class DenseNet(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size_list: list,
            normalization: Optional[str],
            preactivation: bool = False,
            activation: Optional[str] = 'ReLU',
            dropout: float = 0,
            ):
        super().__init__()

        fc_blocks = nn.ModuleList()
        nb_lin = len(out_size_list)

        for ii, out_size in enumerate(out_size_list,1):
            if ii==nb_lin:
                activation=None
            fc_block = FullyConnectedBlock(
                in_size,
                out_size,
                normalization=normalization,
                preactivation=preactivation,
                activation=activation,
                dropout=dropout,
            )
            fc_blocks.append(fc_block)
            in_size = out_size
        self.fc_blocks = nn.Sequential(*fc_blocks)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc_blocks(x)
