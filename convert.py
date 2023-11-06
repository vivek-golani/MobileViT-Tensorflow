import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras import layers

from model import mobilevit
from model.mobilevit_pt import get_mobilevit_pt

torch.set_grad_enabled(False)

DATASET_TO_CLASSES = {
    "imagenet-1k": 1000,
}

TF_MODEL_ROOT = "saved_models"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the PyTorch pre-trained MobileViT weights to TensorFlow."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="imagenet-1k",
        type=str,
        required=False,
        choices=["imagenet-1k", "imagenet-21k"],
        help="Name of the dataset.",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="mobilevit_s",
        type=str,
        required=False,
        choices=[
            "mobilevit_xxs",
            "mobilevit_xs",
            "mobilevit_s",
        ],
        help="Types of MobileViT models.",
    )
    parser.add_argument(
        "-r",
        "--image-resolution",
        default=256,
        type=int,
        required=False,
        help="Image resolution of the model.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        default="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        type=str,
        required=False,
        help="URL of the checkpoint to be loaded.",
    )
    parser.add_argument(
        "-t",
        "--task",
        default="segment",
        type=str,
        required=False,
        choices=["classify", "segment"],
        help="Downstream task.",
    )
    return vars(parser.parse_args())

def get_torch_model(model_name, task):
    torch_model = get_mobilevit_pt(model_name=model_name, task=task)
    torch_model.eval()  # run all component in inference mode

    return torch_model

def get_tf_model(model_name, resolution, num_classes, task):
    tf_model = mobilevit.get_mobilevit_model(
        model_name=model_name,
        image_shape=(resolution, resolution, 3),
        num_classes=num_classes,
        task=task,
    )

    return tf_model


def copy_weights(torch_model, tf_model, task):
    # Fetch the pretrained parameters
    param_list = list(torch_model.cpu().parameters())
    model_states = torch_model.cpu().state_dict()
    state_list = list(model_states.keys())

    def copy_inverted_residual(tf_layer, torch_layer):
        # inverted residual block 1 - (expand_1x1)
        inverted_residual_1_conv_1_tf = tf_model.get_layer(
            f'{tf_layer}_1_conv'
        )
        inverted_residual_1_conv_1_pt = model_states[
            f'{torch_layer}.expand_1x1.convolution.weight'
        ]
        if isinstance(inverted_residual_1_conv_1_tf, layers.Conv2D):
            print(f'--- {tf_layer}_1_conv')
            inverted_residual_1_conv_1_tf.kernel.assign(
                tf.Variable(inverted_residual_1_conv_1_pt.numpy().transpose(2, 3, 1, 0))
            )

        inverted_residual_1_bn_1_tf = tf_model.get_layer(
            f'{tf_layer}_1_bn'
        )
        if isinstance(inverted_residual_1_bn_1_tf, layers.BatchNormalization):
            print(f'--- {tf_layer}_1_bn')
            inverted_residual_1_bn_1_tf.gamma.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.expand_1x1.normalization.weight'
                    ].numpy()
                )
            )
            inverted_residual_1_bn_1_tf.beta.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.expand_1x1.normalization.bias'
                    ].numpy()
                )
            )

        # inverted residual block 1 - (conv_3x3)
        inverted_residual_1_conv_2_tf = tf_model.get_layer(
            f'{tf_layer}_2_conv'
        )
        inverted_residual_1_conv_2_pt = model_states[
            f'{torch_layer}.conv_3x3.convolution.weight'
        ]
        if isinstance(inverted_residual_1_conv_2_tf, layers.Conv2D):
            print(f'--- {tf_layer}_2_conv')
            inverted_residual_1_conv_2_tf.kernel.assign(
                tf.Variable(inverted_residual_1_conv_2_pt.numpy().transpose(2, 3, 1, 0))
            )

        inverted_residual_1_bn_2_tf = tf_model.get_layer(
            f'{tf_layer}_2_bn'
        )
        if isinstance(inverted_residual_1_bn_2_tf, layers.BatchNormalization):
            print(f'--- {tf_layer}_2_bn')
            inverted_residual_1_bn_2_tf.gamma.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.conv_3x3.normalization.weight'
                    ].numpy()
                )
            )
            inverted_residual_1_bn_2_tf.beta.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.conv_3x3.normalization.bias'
                    ].numpy()
                )
            )

        # inverted residual block - 1 (reduce_1x1)
        inverted_residual_1_conv_3_tf = tf_model.get_layer(
            f'{tf_layer}_3_conv'
        )
        inverted_residual_1_conv_3_pt = model_states[
            f'{torch_layer}.reduce_1x1.convolution.weight'
        ]
        if isinstance(inverted_residual_1_conv_3_tf, layers.Conv2D):
            print(f'--- {tf_layer}_3_conv')
            inverted_residual_1_conv_3_tf.kernel.assign(
                tf.Variable(inverted_residual_1_conv_3_pt.numpy().transpose(2, 3, 1, 0))
            )

        inverted_residual_1_bn_3_tf = tf_model.get_layer(
            f'{tf_layer}_3_bn'
        )
        if isinstance(inverted_residual_1_bn_3_tf, layers.BatchNormalization):
            print(f'--- {tf_layer}_3_bn')
            inverted_residual_1_bn_3_tf.gamma.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.reduce_1x1.normalization.weight'
                    ].numpy()
                )
            )
            inverted_residual_1_bn_3_tf.beta.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.reduce_1x1.normalization.bias'
                    ].numpy()
                )
            )

    def copy_mobilevit_block(tf_layer, torch_layer, num_blocks):
        # conv 3*3 
        mobilevit_1_conv_1_tf = tf_model.get_layer(
            f'{tf_layer}2_pre_1_conv'
        )
        mobilevit_1_conv_1_pt = model_states[
            f'{torch_layer}.conv_kxk.convolution.weight'
        ]
        if isinstance(mobilevit_1_conv_1_tf, layers.Conv2D):
            print(f'--- {tf_layer}2_pre_1_conv')
            mobilevit_1_conv_1_tf.kernel.assign(
                tf.Variable(mobilevit_1_conv_1_pt.numpy().transpose(2, 3, 1, 0))
            )

        mobilevit_1_bn_1_tf = tf_model.get_layer(
            f'{tf_layer}2_pre_1_bn'
        )
        if isinstance(mobilevit_1_bn_1_tf, layers.BatchNormalization):
            print(f'--- {tf_layer}2_pre_1_bn')
            mobilevit_1_bn_1_tf.gamma.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.conv_kxk.normalization.weight'
                    ].numpy()
                )
            )
            mobilevit_1_bn_1_tf.beta.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.conv_kxk.normalization.bias'
                    ].numpy()
                )
            )

        # conv 1*1
        mobilevit_1_conv_2_tf = tf_model.get_layer(
            f'{tf_layer}2_pre_2_conv'
        )
        mobilevit_1_conv_2_pt = model_states[
            f'{torch_layer}.conv_1x1.convolution.weight'
        ]
        if isinstance(mobilevit_1_conv_2_tf, layers.Conv2D):
            print(f'--- {tf_layer}2_pre_2_conv')
            mobilevit_1_conv_2_tf.kernel.assign(
                tf.Variable(mobilevit_1_conv_2_pt.numpy().transpose(2, 3, 1, 0))
            )


        # # transformer block
        # copy_transformer_block('', '', num_blocks)

        # conv 1*1
        mobilevit_1_conv_3_tf = tf_model.get_layer(
            f'{tf_layer}{2+num_blocks-1}_post_1_conv'
        )
        mobilevit_1_conv_3_pt = model_states[
            f'{torch_layer}.conv_projection.convolution.weight'
        ]
        if isinstance(mobilevit_1_conv_3_tf, layers.Conv2D):
            print(f'--- {tf_layer}{2+num_blocks-1}_post_1_conv')
            mobilevit_1_conv_3_tf.kernel.assign(
                tf.Variable(mobilevit_1_conv_3_pt.numpy().transpose(2, 3, 1, 0))
            )

        mobilevit_1_bn_3_tf = tf_model.get_layer(
            f'{tf_layer}{2+num_blocks-1}_post_1_bn'
        )
        if isinstance(mobilevit_1_bn_3_tf, layers.BatchNormalization):
            print(f'--- {tf_layer}{2+num_blocks-1}_post_1_bn')
            mobilevit_1_bn_3_tf.gamma.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.conv_projection.normalization.weight'
                    ].numpy()
                )
            )
            mobilevit_1_bn_3_tf.beta.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.conv_projection.normalization.bias'
                    ].numpy()
                )
            )

        # conv 3*3
        mobilevit_1_conv_4_tf = tf_model.get_layer(
            f'{tf_layer}{2+num_blocks-1}_post_2_conv'
        )
        mobilevit_1_conv_4_pt = model_states[
            f'{torch_layer}.fusion.convolution.weight'
        ]
        if isinstance(mobilevit_1_conv_4_tf, layers.Conv2D):
            print(f'--- {tf_layer}{2+num_blocks-1}_post_2_conv')
            mobilevit_1_conv_4_tf.kernel.assign(
                tf.Variable(mobilevit_1_conv_4_pt.numpy().transpose(2, 3, 1, 0))
            )

        mobilevit_1_bn_4_tf = tf_model.get_layer(
            f'{tf_layer}{2+num_blocks-1}_post_2_bn'
        )
        if isinstance(mobilevit_1_bn_4_tf, layers.BatchNormalization):
            print(f'--- {tf_layer}{2+num_blocks-1}_post_2_bn')
            mobilevit_1_bn_4_tf.gamma.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.fusion.normalization.weight'
                    ].numpy()
                )
            )
            mobilevit_1_bn_4_tf.beta.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.fusion.normalization.bias'
                    ].numpy()
                )
            )

    def copy_segmentation_block(tf_layer, torch_layer):
        def copy_conv_block(i):
            seg_head_aspp_conv_tf = tf_model.get_layer(
                f'{tf_layer}aspp_block_convs_{i}conv'
            )
            seg_head_aspp_conv_pt = model_states[
                f'{torch_layer}.aspp.convs.{i}.convolution.weight'
            ]
            if isinstance(seg_head_aspp_conv_tf, layers.Conv2D):
                print(f'--- {tf_layer}aspp_block_convs_{i}conv')
                seg_head_aspp_conv_tf.kernel.assign(
                    tf.Variable(seg_head_aspp_conv_pt.numpy().transpose(2, 3, 1, 0))
                )

            seg_head_aspp_bn_tf = tf_model.get_layer(
                f'{tf_layer}aspp_block_convs_{i}bn'
            )
            if isinstance(seg_head_aspp_bn_tf, layers.BatchNormalization):
                print(f'--- {tf_layer}aspp_block_convs_{i}bn')
                seg_head_aspp_bn_tf.gamma.assign(
                    tf.Variable(
                        model_states[
                            f'{torch_layer}.aspp.convs.{i}.normalization.weight'
                        ].numpy()
                    )
                )
                seg_head_aspp_bn_tf.beta.assign(
                    tf.Variable(
                        model_states[
                            f'{torch_layer}.aspp.convs.{i}.normalization.bias'
                        ].numpy()
                    )
                )

        copy_conv_block(0)
        copy_conv_block(1)
        copy_conv_block(2)
        copy_conv_block(3)

        # conv 4 - 1*1
        seg_head_aspp_conv_tf = tf_model.get_layer(
                f'{tf_layer}aspp_block_convs_4conv'
            )
        seg_head_aspp_conv_pt = model_states[
            f'{torch_layer}.aspp.convs.4.conv_1x1.convolution.weight'
        ]
        if isinstance(seg_head_aspp_conv_tf, layers.Conv2D):
            print(f'--- {tf_layer}aspp_block_convs_4conv')
            seg_head_aspp_conv_tf.kernel.assign(
                tf.Variable(seg_head_aspp_conv_pt.numpy().transpose(2, 3, 1, 0))
            )

        seg_head_aspp_bn_tf = tf_model.get_layer(
            f'{tf_layer}aspp_block_convs_4bn'
        )
        if isinstance(seg_head_aspp_bn_tf, layers.BatchNormalization):
            print(f'--- {tf_layer}aspp_block_convs_4bn')
            seg_head_aspp_bn_tf.gamma.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.aspp.convs.4.conv_1x1.normalization.weight'
                    ].numpy()
                )
            )
            seg_head_aspp_bn_tf.beta.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.aspp.convs.4.conv_1x1.normalization.bias'
                    ].numpy()
                )
            )

        #project
        seg_head_aspp_conv_tf = tf_model.get_layer(
                f'{tf_layer}aspp_block_proj_conv'
            )
        seg_head_aspp_conv_pt = model_states[
            f'{torch_layer}.aspp.project.convolution.weight'
        ]
        if isinstance(seg_head_aspp_conv_tf, layers.Conv2D):
            print(f'--- {tf_layer}aspp_block_proj_conv')
            seg_head_aspp_conv_tf.kernel.assign(
                tf.Variable(seg_head_aspp_conv_pt.numpy().transpose(2, 3, 1, 0))
            )

        seg_head_aspp_bn_tf = tf_model.get_layer(
            f'{tf_layer}aspp_block_proj_bn'
        )
        if isinstance(seg_head_aspp_bn_tf, layers.BatchNormalization):
            print(f'--- {tf_layer}aspp_block_proj_bn')
            seg_head_aspp_bn_tf.gamma.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.aspp.project.normalization.weight'
                    ].numpy()
                )
            )
            seg_head_aspp_bn_tf.beta.assign(
                tf.Variable(
                    model_states[
                        f'{torch_layer}.aspp.project.normalization.bias'
                    ].numpy()
                )
            )

        # #classifier
        # seg_head_aspp_conv_tf = tf_model.get_layer(
        #         f'classifier'
        #     )
        # seg_head_aspp_conv_pt = model_states[
        #     f'{torch_layer}.classifier.convolution.weight'
        # ]
        # if isinstance(seg_head_aspp_conv_tf, layers.Conv2D):
        #     print(f'--- classifier')
        #     seg_head_aspp_conv_tf.kernel.assign(
        #         tf.Variable(seg_head_aspp_conv_pt.numpy().transpose(2, 3, 1, 0))
        #     )
        #     seg_head_aspp_bn_tf.bias.assign(
        #         tf.Variable(
        #             model_states[f'{torch_layer}.classifier.convolution.bias'].numpy())
        #     )


    # Stem block
    stem_layer_conv = tf_model.get_layer("stem_conv")

    print(f'Stem Stack')
    if isinstance(stem_layer_conv, layers.Conv2D):
        print("--- stem_conv")
        stem_layer_conv.kernel.assign(
            tf.Variable(param_list[0].numpy().transpose(2, 3, 1, 0))
        )
        # stem_layer_conv.bias.assign(tf.Variable(param_list[1].numpy()))

    stem_layer_bn = tf_model.get_layer("stem_bn")
    if isinstance(stem_layer_bn, layers.BatchNormalization):
        print("--- stem_bn")
        stem_layer_bn.gamma.assign( 
            tf.Variable(
                model_states["mobilevit.conv_stem.normalization.weight"].numpy()
            )
        )
        stem_layer_bn.beta.assign(
            tf.Variable(model_states["mobilevit.conv_stem.normalization.bias"].numpy())
        )

    print(f'Stack1 - Block1') 
    copy_inverted_residual('stack1_block1_deep', 'mobilevit.encoder.layer.0.layer.0')

    print(f'Stack2 - Block1') 
    copy_inverted_residual('stack2_block1_deep', 'mobilevit.encoder.layer.1.layer.0')

    print(f'Stack2 - Block2')
    copy_inverted_residual('stack2_block2_deep', 'mobilevit.encoder.layer.1.layer.1')

    print(f'Stack2 - Block3')
    copy_inverted_residual('stack2_block3_deep', 'mobilevit.encoder.layer.1.layer.2')

    print(f'Stack3 - Block1')
    copy_inverted_residual('stack3_block1_deep', 'mobilevit.encoder.layer.2.downsampling_layer')

    print(f'Stack3 - Block2, 3')
    copy_mobilevit_block('stack3_block', 'mobilevit.encoder.layer.2', 2)

    print(f'Stack4 - Block1')
    copy_inverted_residual('stack4_block1_deep', 'mobilevit.encoder.layer.3.downsampling_layer')

    print(f'Stack4 - Block2, 3, 4, 5')
    copy_mobilevit_block('stack4_block', 'mobilevit.encoder.layer.3', 4)

    print(f'Stack5 - Block1')
    copy_inverted_residual('stack5_block1_deep', 'mobilevit.encoder.layer.4.downsampling_layer')

    print(f'Stack5 - Block2, 3, 4')
    copy_mobilevit_block('stack5_block', 'mobilevit.encoder.layer.4', 3)

    if task == 'classify':
        pass
    else:
        print('Segmentation Head')
        copy_segmentation_block('seg_head_', 'segmentation_head')

    return tf_model

def main(args):
    print(f'Model: {args["model_name"]}')
    print(f'Image resolution: {args["image_resolution"]}')
    print(f'Dataset: {args["dataset"]}')
    print(f'Checkpoint URL: {args["checkpoint_path"]}')
    print(f'Downstream Task: {args["task"]}')
    
    print("Instantiating PyTorch model and populating weights...")
    torch_model = get_torch_model(args["model_name"], args["task"])
    
    print("Instantiating TensorFlow model...")
    tf_model = get_tf_model(args["model_name"], args["image_resolution"], 64, args["task"])

    print("TensorFlow model instantiated, populating pretrained weights...")
    tf_model = copy_weights(torch_model, tf_model)

    # if args["task"] == 'classify':
    #     pass
    #     Classifier Head
    #     tf_model.layers[-1].kernel.assign(
    #         tf.Variable(model_states[state_list[-2]].numpy().transpose())
    #     )
    #     tf_model.layers[-1].bias.assign(
    #         tf.Variable(model_states[state_list[-1]].numpy())
    #     )
    # else:
    #     pass
    #     convert_segmentation_block(
    #         tf_model,
    #         param_list,
    #         model_states,
    #         state_list,
    #     )

    print("Weight population successful, serializing TensorFlow model...")
    model_name = f'{args["model_name"]}_{args["image_resolution"]}'
    save_path = os.path.join(TF_MODEL_ROOT, f'{model_name}_')
    tf_model.save(save_path)
    print(f"TensorFlow model serialized to: {save_path}...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
