import random
import tensorflow as tf
# import tensorflow_models as tfm
from tensorflow import keras
from keras import layers, backend

# Transform dataset using tf.image
mean = tf.constant([0.5, 0.5, 0.5])
std = tf.constant([0.5, 0.5, 0.5])

def normalize(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - mean) / tf.maximum(std, backend.epsilon())
    
    return image

def tf_image_transforms(image, label, crop_size, split):
    combined = tf.concat([image, label], axis=-1)
    seed = random.randint(0, 100) 

    if split == 'Train':
        combined = tf.image.resize_with_crop_or_pad(combined, crop_size, crop_size)
        combined = tf.image.random_flip_left_right(combined, seed=seed)
        combined = tf.cast(combined, tf.float32)

        image, label = combined[:,:,:3], combined[:,:,3:]
        
        image = image / tf.constant(255, tf.float32)
        image = tf.image.random_brightness(image, 0.1, seed=seed)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
        image = tf.image.random_contrast(image, 0.5, 1.5, seed=seed)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
        image = tf.image.random_saturation(image, 0.5, 1.5, seed=seed)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
        image = tf.image.random_hue(image, 0.1, seed=seed)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    else:
        combined = tf.image.resize_with_crop_or_pad(combined, crop_size, crop_size)
        combined = tf.cast(combined, tf.float32)

        image, label = combined[:,:,:3], combined[:,:,3:]
        image = image / tf.constant(255, tf.float32)
    
    return image, label


Transform dataset using tensorflow_models.preprocess_ops
def tfm_transforms(image, label, crop_size, split):
    combined = tf.concat([image, label], axis=-1)
    seed = random.randint(0, 100) 

    if split == 'Train':
        combined = tfm.vision.preprocess_ops_3d.crop_image(combined, target_height=crop_size, target_width=crop_size, random=False, num_crops=1)
        combined, _, _ = tfm.vision.preprocess_ops.random_horizontal_flip(combined, seed=seed, prob=0.5)

        image, label = combined[:,:,:3], combined[:,:,3:]

        image = tfm.vision.preprocess_ops.color_jitter(image, brightness=0.3, contrast=0.5, saturation=0.5, seed=seed)
        image = tf.cast(image, tf.float32) 
        label = tf.cast(label, tf.float32)
    else:
        combined = tfm.vision.preprocess_ops_3d.crop_image(combined, target_height=crop_size, target_width=crop_size, random=False, num_crops=1)
        combined = tf.cast(combined, tf.float32)

        image, label = combined[:,:,:3], combined[:,:,3:]
        
    image = tfm.vision.preprocess_ops.normalize_scaled_float_image(
        image,
        offset=tfm.vision.preprocess_ops.MEAN_NORM, 
        scale=tfm.vision.preprocess_ops.STDDEV_NORM,
    )       

    return image, label


# Transform dataset using Sequential keras model
def keras_transforms(crop_size, split):
    seed = random.randint(0, 100) 
    if split == 'Train':
        image_transforms = keras.Sequential([
            layers.CenterCrop(crop_size, crop_size),
            layers.RandomFlip(mode='horizontal', seed=seed),
            layers.RandomRotation(factor=0.025, fill_mode='constant', interpolation='bilinear', fill_value=0, seed=seed),
            layers.RandomBrightness(factor=0.2, seed=seed),
            layers.RandomContrast(factor=(0.5, 1.5), seed=seed),
            layers.GaussianNoise(random.random(), seed=seed),
            layers.Rescaling(scale=1./255.),
        ])
        label_transforms = keras.Sequential([
            layers.CenterCrop(crop_size, crop_size),
            layers.RandomFlip(mode='horizontal', seed=seed),
            layers.RandomRotation(factor=0.025, fill_mode='constant', interpolation='nearest', fill_value=0, seed=seed),
        ])
    else:
        image_transforms = keras.Sequential([
            layers.CenterCrop(crop_size, crop_size),
            layers.Rescaling(scale=1./255.),
        ])
        label_transforms = keras.Sequential([
            layers.CenterCrop(crop_size, crop_size),
        ])
    
    return image_transforms, label_transforms

def apply_transform(image, label, crop_size, split, option=0):
    if option == 0:
        image, label = tf_image_transforms(image, label, crop_size, split)
    elif option == 1:
        image, label = tfm_transforms(image, label, crop_size, split)
    elif option == 2:
        image_transforms, label_transforms = keras_transforms(crop_size, split)

        image = image_transforms(image, training=True)
        label = label_transforms(label, training=True)
    
    image = tf.reverse(image, axis=[-1])

    return image, label

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape((600, 960, 3))

    return image

def load_label(seg_map_path):
    seg_map = tf.io.read_file(seg_map_path)
    seg_map = tf.image.decode_image(seg_map, channels=1)
    seg_map.set_shape((600, 960, 1))

    return seg_map

def load(image_path, seg_map_path):
    image = load_image(image_path)
    seg_map = load_label(seg_map_path)

    return image, seg_map

def preprocess(image, seg_map, crop_size, split):        
    # options: 0 - tf.image transforms, 1- tfm.preprocess_ops, 2 - keras sequential model
    image, seg_map = apply_transform(image, seg_map, crop_size, split, option=0)

    return image, seg_map
