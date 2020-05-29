from typing import Optional
import torch.nn as nn
from .conv import ConvolutionalBlock


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channel_lists: list,
            dimensions: int,
            pooling_type: str,
            normalization: Optional[str],
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        is_first_block = True
        for out_channel_list in out_channel_lists:
            encoding_block = EncodingBlock(
                in_channels,
                out_channel_list,
                dimensions,
                normalization,
                pooling_type,
                preactivation,
                is_first_block=is_first_block,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            in_channels = out_channel_list[-1]
            if self.dilation is not None:
                self.dilation *= 2

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x

    @property
    def out_channels(self):
        return self.encoding_blocks[-1].out_channels


class EncodingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channel_list: list,
            dimensions: int,
            normalization: Optional[str],
            pooling_type: Optional[str],
            preactivation: bool = False,
            is_first_block: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dilation: Optional[int] = None,
            dropout: float = 0,
            ):
        super().__init__()

        self.preactivation = preactivation
        self.normalization = normalization

        self.residual = residual

        if is_first_block:
            normalization = None
            preactivation = None
        else:
            normalization = self.normalization
            preactivation = self.preactivation

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
            preactivation = self.preactivation
            normalization = self.normalization
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*conv_layers)

        self.downsample = None
        if pooling_type is not None:
            self.downsample = get_downsampling_layer(dimensions, pooling_type)

    def forward(self, x):
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv_layers(x)
            x += connection
        else:
            x = self.conv_layers(x)
        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
            return x, skip_connection

    @property
    def out_channels(self):
        return self.conv_layers[-1].out_channels


def get_downsampling_layer(
        dimensions: int,
        pooling_type: str,
        kernel_size: int = 2,
        ) -> nn.Module:
    class_name = '{}Pool{}d'.format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)
