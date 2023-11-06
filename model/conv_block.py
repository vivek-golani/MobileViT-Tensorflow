from typing import Tuple

import tensorflow as tf
from keras import backend, layers
from keras.applications import imagenet_utils
from tensorflow import keras

def conv_block(
    input_layer: layers.Input,
    num_filters: int = 16,
    kernel_size: Tuple[int, int] = (3, 3),
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False,
    use_padding: int = 0, 
    use_normalization: bool = True,
    activation: str = 'swish',
    name=None,
):
    """
    3x3 Convolutional Stem Stage.

    Args:
        input_layer: input tensor
        num_filters (int): number of filters in the convolutional layer
        kernel_size (Tuple[int, int]): kernel size of the convolution layer
        stride (int): stride of the convolutional layer
        groups (int):
        padding (str):
        use_normalization (bool):
        activation: 
        name (str): name of the layer

    Returns:
        output tensor of the convolutional block
    """

    if name is None:
        name = 'conv_block'
    
    if use_padding:
        input_layer = layers.ZeroPadding2D(padding=use_padding, name=name+'pad')(input_layer)

    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=stride,
        padding='valid',
        dilation_rate=dilation,
        groups=groups,
        use_bias=bias,
        name=name+'conv',
    )(input_layer)

    if use_normalization:
        x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name=name+'bn')(x)

    if activation == 'swish' or activation == 'silu':
        x = tf.keras.activations.swish(x)
    elif activation == 'relu':
        x = tf.keras.activations.relu(x)

    return x

def inverted_residual_block(
    input_layer: layers.Input,
    expanded_channels: int,
    output_channels: int,
    stride: int = 1,
    name: str = "",
):
    """
    Inverted Residual Block.

    Args:
        input_layer: input tensor
        expanded_channels (int): number of filters in the expanded convolutional layer
        output_channels (int): number of filters in the output convolutional layer
        strides (int): stride of the convolutional layer
        name (str): name of the layer
    Returns:
        output tensor of the inverted residual block
    """
    
    # Conv to increase channels
    conv1x1_expand = conv_block(
        input_layer=input_layer,
        num_filters=expanded_channels,
        kernel_size=(1, 1),
        stride=1,
        use_normalization=True,
        activation='swish',
        name=name + '1_',
    )

    # Deptwise Conv
    conv3x3 = conv_block(
        input_layer=conv1x1_expand,
        num_filters=expanded_channels,
        kernel_size=(3, 3),
        stride=stride,
        groups=expanded_channels,
        use_padding=1, 
        use_normalization=True,
        activation='swish',
        name=name + '2_',
    )

    # Conv to reduce channels
    conv1x1_reduce = conv_block(
        input_layer=conv3x3,
        num_filters=output_channels,
        kernel_size=(1, 1),
        stride=1, 
        use_normalization=True,
        activation='None',
        name=name + '3_',
    )

    if tf.math.equal(input_layer.shape[-1], output_channels) and stride == 1:
        return layers.Add()([conv1x1_reduce, input_layer])
    
    return conv1x1_reduce
