import tensorflow as tf
from tensorflow.keras import layers

from model.conv_block import conv_block
import pdb

def aspp_block(
    input_layer: layers.Input,
    out_channels: int,
    atrous_rates: list,
    dropout_rate: float,
    name=None,
):

    if len(atrous_rates) != 3:
        raise ValueError("Expected 3 values for atrous_rates")

    in_proj = conv_block(
        input_layer=input_layer,
        num_filters=out_channels,
        kernel_size=(1, 1),
        use_normalization=True,
        activation='relu',
        name=name+'convs_0',
    )

    convs = {}
    for i, rate in enumerate(atrous_rates):
        x = conv_block(
            input_layer=input_layer,
            num_filters=out_channels,
            kernel_size=(3,3),
            dilation=rate,
            use_padding=rate,
            use_normalization=True,
            activation='relu',
            name=f'{name}convs_{i+1}',
        )
        convs[f'{i}'] = x

    shape = input_layer.shape.as_list()[1:-1]
    pool = layers.GlobalAvgPool2D(keepdims=True)(input_layer)
    pool = conv_block(
        input_layer=pool,
        num_filters=out_channels,
        kernel_size=(1,1),
        use_normalization=True,
        activation='relu',
        name=name+"convs_4",
    )
    pool = layers.Resizing(shape[0], shape[1], interpolation='bilinear', name=name+'resize')(pool)

    out = layers.Concatenate(axis=-1, name=name+'concat')([in_proj, convs['0'], convs['1'], convs['2'], pool])

    out = conv_block(
        input_layer=out,
        num_filters=out_channels,
        kernel_size=(1,1),
        use_normalization=True,
        activation='relu',
        name=name+"proj_",
    )

    dropout = layers.Dropout(dropout_rate)(out)

    return dropout

def segmentation_block(
    input_layer: layers.Input,
    classifier_dropout_rate: float,
    aspp_out_channels: int,
    aspp_atrous_rates: list,
    aspp_dropout_rate: float,
    name=None
):

    aspp = aspp_block(
        input_layer,
        aspp_out_channels,
        aspp_atrous_rates,
        aspp_dropout_rate,
        name + "aspp_block_",
    )

    aspp = layers.Dropout(classifier_dropout_rate)(aspp)

    return aspp