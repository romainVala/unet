from typing import Optional
import torch.nn as nn
import numpy as np
from .encoding import Encoder
from .base_net import BaseNet, clean_locals
from .dense import DenseNet


class ConvNet(BaseNet):
    def __init__(
            self,
            in_size: tuple,
            in_channels: int = 1,
            out_classes: int = 2,
            dimensions: int = 2,
            num_encoding_blocks: int = 5,
            out_channels_first_layer: int = 64,
            encoder_out_channel_lists: list = None,
            linear_out_size_list: list = None,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            monte_carlo_dropout: float = 0,
            final_activation: Optional[str] = None,
    ):
        super().__init__(**clean_locals(**locals()))

        if encoder_out_channel_lists is None:
            encoder_out_channel_lists = []
            for _ in range(num_encoding_blocks):
                if dimensions == 2:
                    out_channels_second_layer = out_channels_first_layer
                else:
                    out_channels_second_layer = 2 * out_channels_first_layer
                encoder_out_channel_lists.append([out_channels_first_layer, out_channels_second_layer])
                out_channels_first_layer *= 2
        else:
            if num_encoding_blocks != len(encoder_out_channel_lists):
                raise ValueError('Number of encoding blocks and length of output channels\' list do not match.')

        if linear_out_size_list is None:
            linear_out_size_list = []

        linear_out_size_list = linear_out_size_list + [out_classes]

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            encoder_out_channel_lists,
            dimensions,
            pooling_type,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
        )

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, 'Dropout{}d'.format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Fully connected layers
        linear_in_size = self.get_linear_in_size(in_size, encoder_out_channel_lists, padding)

        self.dense = DenseNet(
            linear_in_size,
            linear_out_size_list,
            normalization,
            preactivation=preactivation,
            activation=activation,
            dropout=dropout,
        )

        if final_activation is not None:
            self.final_activation_layer = getattr(nn, final_activation)()

    def forward(self, x):
        _, x = self.encoder(x)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        x = self.dense(x)
        if self.final_activation_layer is not None:
            x = self.final_activation_layer(x)
        return x

    @staticmethod
    def get_linear_in_size(in_size, conv_lists, padding):
        size_reduction = 2 * (1 - padding)
        for convs in conv_lists:
            in_size = tuple(map(lambda s: (s - len(convs) * size_reduction) // 2, in_size))
        return np.product(in_size) * conv_lists[-1][-1]


class ConvNet2D(ConvNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {'dimensions': 2, 'num_encoding_blocks': 5, 'out_channels_first_layer': 64}
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class ConvNet3D(ConvNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {'dimensions': 3, 'num_encoding_blocks': 4, 'out_channels_first_layer': 32, 'normalization': 'batch'}
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
