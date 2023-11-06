import tensorflow as tf
from tensorflow.keras import layers

from model.conv_block import conv_block
from model.transformer_block import transformer_block


def unfolding(features, patch_size, projection_dim):
    batch_size = tf.shape(features)[0]
    orig_height = tf.shape(features)[1]
    orig_width = tf.shape(features)[2]
    channels = tf.shape(features)[3]

    patch_height, patch_width = patch_size, patch_size
    patch_area = patch_height * patch_width
    num_patch_height, num_patch_width = orig_height // patch_height, orig_width // patch_width

    num_patches = num_patch_height * num_patch_width

    # features = layers.Permute((3, 1, 2))(features)
    # patches = layers.Reshape(
    #     (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
    # )(features)
    # patches = layers.Permute((2, 1, 3))(patches)
    # patches = layers.Reshape((batch_size, channels, num_patches, patch_area))(patches)
    # patches = layers.Permute((0, 3, 2, 1))(patches)
    # patches = layers.Reshape((batch_size * patch_area, num_patches, channels))(patches)
    
    features = tf.transpose(features, [0, 3, 1, 2])
    patches = tf.reshape(
        features, (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
    )
    patches = tf.transpose(patches, [0, 2, 1, 3])
    patches = tf.reshape(patches, (batch_size, channels, num_patches, patch_area))
    patches = tf.transpose(patches, [0, 3, 2, 1])
    patches = tf.reshape(patches, (batch_size * patch_area, num_patches, channels))
    patches = tf.cast(patches, "float32")
    info_dict = {
        "orig_size": (orig_height, orig_width),
        "batch_size": batch_size,
        "channels": channels,
        "num_patches": num_patches,
        "num_patches_width": num_patch_width,
        "num_patches_height": num_patch_height,
    }
    # patches = layers.Reshape((features[0] * patch_area, num_patches, projection_dim))(features)
    
    return patches, info_dict

def folding(patches, patch_size, info_dict):
    patch_width, patch_height = patch_size, patch_size
    patch_area = int(patch_width * patch_height)

    batch_size = info_dict["batch_size"]
    channels = info_dict["channels"]
    num_patches = info_dict["num_patches"]
    num_patch_height = info_dict["num_patches_height"]
    num_patch_width = info_dict["num_patches_width"]

    # convert from shape (batch_size * patch_area, num_patches, channels)
    # back to shape (batch_size, channels, orig_height, orig_width)
    features = tf.reshape(patches, (batch_size, patch_area, num_patches, -1))
    features = tf.transpose(features, perm=(0, 3, 2, 1))
    features = tf.reshape(
        features, (batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width)
    )
    features = tf.transpose(features, perm=(0, 2, 1, 3))
    features = tf.reshape(
        features, (batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width)
    )
    features = tf.transpose(features, perm=(0, 2, 3, 1))

    return features

def mobilevit_block(
    input_layer: layers.Input,
    num_blocks: int,
    projection_dim: int,
    patch_size: int,
    num_heads : int = 2,
    name: str = 'mobvit_block',
):
    """
    MobileVIT Block.
    Reference: https://arxiv.org/abs/2110.02178

    Args:
        input_layer: input tensor
        num_blocks (int): number of blocks in the MobileVIT block
        projection_dim (int): number of filters in the expanded convolutional layer
        patch_size (int): size of the patch
        strides (int): stride of the convolutional layer

    Returns:
        output_tensor of the MobileVIT block
    """

    # Conv + BN + Act -> 32*32*96
    local_features = conv_block(
        input_layer=input_layer,
        num_filters=input_layer.shape[-1],
        kernel_size=(3, 3),
        stride=1,
        use_padding=True,
        use_normalization=True,
        activation='swish',
        name=name+'2_pre_1_',
    )

    # Conv -> 32*32*144
    local_features = conv_block(
        input_layer=local_features,
        num_filters=projection_dim,
        kernel_size=(1, 1),
        stride=1,
        use_padding=False,
        use_normalization=False,
        activation='None',
        name=name+'2_pre_2_',
    )
    
    # P=4, N=256, d=144
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)

    # Unfolding -> (4, 256, 144)
    non_overlap_patches, info_dict = unfolding(local_features, patch_size, projection_dim)
    # non_overlap_patches = layers.Reshape((patch_size, num_patches, projection_dim))(local_features)
    
    global_features = transformer_block(non_overlap_patches, num_blocks, projection_dim, num_heads, name)
    global_features = layers.LayerNormalization(epsilon=1e-5, name=name+f'{2+num_blocks-1}_post_ln')(global_features)

    # Folding -> ()
    folded_feature_map = folding(global_features, patch_size, info_dict)
    # folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(global_features)

    # Conv -> 32*32*96
    folded_feature_map = conv_block(
        input_layer=folded_feature_map,
        num_filters=input_layer.shape[-1],
        kernel_size=(1, 1),
        stride=1,
        use_padding=False,
        use_normalization=True,
        activation='swish',
        name=name+f'{2+num_blocks-1}_post_1_',
    )

    # Concat -> 32*32*192
    local_global_features = layers.Concatenate(axis=-1)([input_layer, folded_feature_map])

    # Conv -> 32*32*96
    local_global_features = conv_block(
        input_layer=local_global_features,
        num_filters=input_layer.shape[-1],
        kernel_size=(3,3),
        stride=1,
        use_padding=True,
        use_normalization=True,
        activation='swish',
        name=name+f"{2+num_blocks-1}_post_2_",
    )

    return local_global_features
