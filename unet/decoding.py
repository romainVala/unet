from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvolutionalBlock

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = (
    'nearest',
    'linear',
    'bilinear',
    'bicubic',
    'trilinear',
)


class Decoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channel_lists: list,
            skip_connection_channel_list: list,
            dimensions: int,
            upsampling_type: str,
            normalization: Optional[str],
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dilation: Optional[int] = None,
            dropout: float = 0,
            ):
        super().__init__()
        upsampling_type = fix_upsampling_type(upsampling_type, dimensions)
        self.decoding_blocks = nn.ModuleList()
        self.dilation = dilation
        for skip_connection_channels, out_channel_list in zip(skip_connection_channel_list, out_channel_lists):
            decoding_block = DecodingBlock(
                in_channels,
                out_channel_list,
                skip_connection_channels,
                dimensions,
                upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            in_channels = out_channel_list[-1]
            self.decoding_blocks.append(decoding_block)
            # if self.dilation is not None:
            #     self.dilation //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channel_list: list,
            skip_connection_channels: int,
            dimensions: int,
            upsampling_type: str,
            normalization: Optional[str],
            preactivation: bool = True,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dilation: Optional[int] = None,
            dropout: float = 0,
            ):
        super().__init__()
        self.residual = residual

        if upsampling_type == 'conv':
            self.upsample = get_conv_transpose_layer(
                dimensions, in_channels, in_channels)
        else:
            self.upsample = get_upsampling_layer(upsampling_type)
        in_channels = in_channels + skip_connection_channels

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels,
                out_channel_list[-1],
                kernel_size=1,
                normalization=None,
                activation=None,
            )

        conv_layers = nn.ModuleList()
        for out_channels in out_channel_list:
            conv_layers.append(ConvolutionalBlock(
                dimensions,
                in_channels,
                out_channels,
                normalization=normalization,
                preactivation=preactivation,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=dilation,
                dropout=dropout,
            ))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        x = self.center_crop(skip_connection, x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv_layers(x)
            x += connection
        else:
            x = self.conv_layers(x)
        return x

    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2
        additional_crop = crop % 2
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = torch.stack((half_crop, half_crop + additional_crop)).t().flatten()
        x = F.pad(x, pad.tolist())
        return x


def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = (
            'Upsampling type is "{}"'
            ' but should be one of the following: {}'
        )
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    return nn.Upsample(scale_factor=2, mode=upsampling_type)


def get_conv_transpose_layer(dimensions, in_channels, out_channels):
    class_name = 'ConvTranspose{}d'.format(dimensions)
    conv_class = getattr(nn, class_name)
    conv_layer = conv_class(in_channels, out_channels, kernel_size=2, stride=2)
    return conv_layer


def fix_upsampling_type(upsampling_type: str, dimensions: int):
    if upsampling_type == 'linear':
        if dimensions == 2:
            upsampling_type = 'bilinear'
        elif dimensions == 3:
            upsampling_type = 'trilinear'
    return upsampling_type
