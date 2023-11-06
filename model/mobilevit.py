
import numpy as np
from tkinter import X
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K

from configs.task_config import get_config
from model.conv_block import conv_block, inverted_residual_block
from model.mobilevit_block import mobilevit_block
from model.segmentation_block import segmentation_block

def get_mobilevit_model(
    model_name: str,
    image_shape: tuple,
    num_classes: int,
    task: str = 'segment',
) -> keras.Model:
    """
    Implements MobileViT family of models given a configuration.
    References:
        (1) https://arxiv.org/pdf/2110.02178.pdf

    Args:
        model_name (str): Name of the MobileViT model
        image_shape (tuple): Shape of the input image
        num_classes (int): number of classes of the classifier
        task(str): Classification or Segmentation Task

    Returns:
        model: Keras Model
    """
    configs = get_config(model_name, task)
    num_inverted_block = 2

    # Input Layer -> H*W*3
    input_layer = keras.Input(shape=image_shape) 
    
    # Stem Layer -> H/2*W/2*16
    x = conv_block(
        input_layer,
        num_filters=configs.out_channels[0],
        kernel_size=(3, 3),
        stride=2,
        use_padding=1, 
        use_normalization=True,
        activation='swish',
        name='stem_',
    )

    # MobileNetv2 Stack 1 -> H/2*W/2*32
    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[0] * configs.expansion_factor,
        output_channels=configs.out_channels[1],
        name='stack1_block1_deep_',
    )

    # MobileNetv2 Stack 2 Block 1 (downsampling) -> H/4*W/4*64
    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[1] * configs.expansion_factor,
        output_channels=configs.out_channels[2],
        stride=2,
        name='stack2_block1_deep_',
    )

    # MobileNetv2 Stack 2 Block 2,3 -> H/4*W/4*64
    for i in range(num_inverted_block):
        x = inverted_residual_block(
            input_layer=x,
            expanded_channels=configs.out_channels[2] * configs.expansion_factor,
            output_channels=configs.out_channels[3],
            name=f'stack2_block{i+2}_deep_',
        )

    # MobileNetv2 Stack 3 Block 1 (downsampling) -> H/8*W/8*96
    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[3] * configs.expansion_factor,
        output_channels=configs.out_channels[4],
        stride=2,
        name='stack3_block1_deep_',
    )

    # MobileViT Stack 3 Block 2,3 -> H/8*W/8*96
    x = mobilevit_block(
        input_layer=x,
        num_blocks=configs.num_blocks[0],
        projection_dim=configs.projection_dims[0],
        patch_size=4,
        name='stack3_block',
    )

    # MobileNetv2 Stack 4 Block 1 (downsampling) -> 16*16*128
    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[5] * configs.expansion_factor,
        output_channels=configs.out_channels[6],
        stride=1,
        name='stack4_block1_deep_',
    )

    # MobileViT Stack 4 Block 2,3,4,5 -> 16*16*128
    x = mobilevit_block(
        input_layer=x,
        num_blocks=configs.num_blocks[1],
        projection_dim=configs.projection_dims[1],
        patch_size=4,
        name='stack4_block',
    )

    # MobileNetv2 Stack 5 Block 1 (downsampling) -> 8*8*
    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[7] * configs.expansion_factor,
        output_channels=configs.out_channels[8],
        stride=1,
        name='stack5_block1_deep_',
    )

    # MobileViT Stack 5 Block 2,3,4
    output_layer = mobilevit_block(
        input_layer=x,
        num_blocks=configs.num_blocks[2],
        projection_dim=configs.projection_dims[2],
        patch_size=4,
        name='stack5_block',
    )

    # output_layer = conv_block(
    #     input_layer=x,
    #     num_filters=configs.out_channels[10],
    #     kernel_size=(1, 1),
    #     stride=1,
    # )

    if task == 'classify':
        # Classification Head
        output_layer = layers.GlobalAvgPool2D()(output_layer)
        output_layer = layers.Dense(num_classes, activation="softmax", name="classification_head")(output_layer)
    else:
        # Segmentation Head
        output_layer = segmentation_block(
            input_layer=output_layer,
            classifier_dropout_rate=configs.classifier_dropout_prob,
            aspp_out_channels=configs.aspp_out_channels,
            aspp_atrous_rates=configs.atrous_rates,
            aspp_dropout_rate=configs.aspp_dropout_prob,
            name='seg_head_',
        )

        output_layer = conv_block(
            input_layer=output_layer,
            num_filters=num_classes,
            kernel_size=(1, 1),
            stride=1,
            bias=True,
            use_padding=1, 
            use_normalization=False,
            activation='None',
            name='classifier_',
        )

        output_layer = layers.Resizing(image_shape[0], image_shape[1], interpolation="bilinear")(output_layer)


    return keras.Model(input_layer, output_layer, name=model_name)
