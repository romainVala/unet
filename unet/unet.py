# -*- coding: utf-8 -*-

"""Main module."""

from typing import Optional
import torch.nn as nn
from .encoding import Encoder, EncodingBlock
from .decoding import Decoder
from .conv import ConvolutionalBlock
from .base_net import BaseNet, clean_locals

__all__ = ['UNet', 'UNet2D', 'UNet3D']


class UNet(BaseNet):
    def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 2,
            dimensions: int = 2,
            num_encoding_blocks: int = 5,
            out_channels_first_layer: int = 64,
            encoder_out_channel_lists: list = None,
            decoder_out_channel_lists: list = None,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            upsampling_type: str = 'conv',
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dilation: Optional[int] = None,
            dropout: float = 0,
            monte_carlo_dropout: float = 0,
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

        skip_connection_channel_list = [out_channel_list[-1] for out_channel_list in encoder_out_channel_lists[:-1]][::-1]

        if decoder_out_channel_lists is None:
            decoder_out_channel_lists = []
            for i, out_channel_list in enumerate(encoder_out_channel_lists[:-1]):
                decoder_out_channel_lists.append(
                    list(reversed(encoder_out_channel_lists[i+1]))[1:] + [out_channel_list[-1]]
                )
                decoder_out_channel_lists.reverse()

        else:
            if num_encoding_blocks - 1 != len(decoder_out_channel_lists):
                raise ValueError('Number of decoding block and length of output channels\' list do not match.')

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            encoder_out_channel_lists[:-1],
            dimensions,
            pooling_type,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            encoder_out_channel_lists[-1],
            dimensions,
            normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Decoder
        in_channels = self.bottom_block.out_channels

        self.decoder = Decoder(
            in_channels,
            decoder_out_channel_lists,
            skip_connection_channel_list,
            dimensions,
            upsampling_type,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, 'Dropout{}d'.format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        in_channels = decoder_out_channel_lists[-1][-1]
        self.classifier = ConvolutionalBlock(
            dimensions, in_channels, out_classes,
            kernel_size=1, activation=None,
        )

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        return self.classifier(x)


class UNet2D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 2
        kwargs['num_encoding_blocks'] = 5
        kwargs['out_channels_first_layer'] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 3
        kwargs['num_encoding_blocks'] = 4
        kwargs['out_channels_first_layer'] = 32
        kwargs['normalization'] = 'batch'
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
