import os
import shutil
import random
import numpy as np
from PIL import Image
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras import layers, backend

from data.pipeline import load, preprocess
from data.utils import read_config_file, examples_per_class, visualize_dataset

DATASET_CONFIG_PATH = Path(__file__).parent / "config" / "temp.py"
DATASET_CATS_PATH = Path(__file__).parent / "config" / "temp.yml"

# Define the ignore label value
IGNORE_LABEL = 255  

def diff_set_train_val(dataset_dir, split):
    image_dir = Path(dataset_dir) / 'image_dir' / split
    ann_dir = Path(dataset_dir) / 'ann_dir' / split

    if split == 'Train':
        image_paths = [str(p) for p in image_dir.rglob('*.bmp')]
        seg_map_paths = [str(p) for p in ann_dir.rglob('*_c_*.png')]
    elif split == 'Val':
        image_paths = [str(p) for p in image_dir.rglob('*.bmp')]
        seg_map_paths = [str(p) for p in ann_dir.rglob('*.png')]

    image_paths.sort()
    seg_map_paths.sort()

    return image_paths, seg_map_paths

def same_set_train_val(dataset_dir, split, train_ratio=0.9,):
    img_paths, seg_paths = diff_set_train_val(dataset_dir=dataset_dir, split='Train')
    
    # Creating dictionary of paths per class from train set itself
    class2image, class2label, count = examples_per_class(img_paths, seg_paths)

    image_paths = []
    seg_map_paths = []

    # Keeping train set as train_ratio percentage of samples and rest for validation
    for cls in class2image.keys():
        cls_len = len(class2image[cls])
        train_len = int(train_ratio * cls_len)
        val_len = cls_len - train_len

        if split == 'Train':
            image_paths.extend(class2image[cls][:train_len])
            seg_map_paths.extend(class2label[cls][:train_len])
        elif split == 'Val':
            image_paths.extend(class2image[cls][train_len:])
            seg_map_paths.extend(class2label[cls][train_len:])

    return image_paths, seg_map_paths

def create_temp_dataset(**kwargs):
    name=kwargs['dataset']
    split=kwargs['split'] 
    image_size=kwargs['']
    crop_size=kwargs['crop_size']
    batch_size=kwargs['batch_size']
    use_same_set=kwargs['use_same_set']

    if 'debug' in kwargs.keys():
        debug = kwargs['debug']
    else:
        debug = 0

    auto = tf.data.AUTOTUNE
    
    # for target names and colors - read DATASET_CATS_PATH

    config = read_config_file(DATASET_CONFIG_PATH)
    dataset_dir = config['data_root']

    # use_same_set indicates if we plan to use train, val and test set from same distribution or different distributions
    if use_same_set:
        image_paths, seg_map_paths = same_set_train_val(dataset_dir=dataset_dir, split=split)
    else:
        image_paths, seg_map_paths = diff_set_train_val(dataset_dir=dataset_dir, split=split)

    # Convert the lists to TensorFlow tensors
    image_paths_tensor = tf.constant(np.array(image_paths))
    seg_map_paths_tensor = tf.constant(np.array(seg_map_paths))

    # Create a TensorFlow dataset from the image and segmentation map file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, seg_map_paths_tensor))
    if split=='Train':
        dataset = dataset.shuffle(1000)

    if debug:
        visualize_dataset(dataset=dataset, num_samples=5, crop_size=crop_size, split=split)

    # Loading and Augmenting Data
    def load_and_preprocess_image(image_path, seg_map_path):
        image, seg_map = load(image_path, seg_map_path)
        image, seg_map = preprocess(image, seg_map, crop_size, split)
        
        return image, seg_map

    dataset = dataset.map(
        lambda image_path, label_path: load_and_preprocess_image(image_path, label_path),
        num_parallel_calls=auto,
    )

    return dataset

if __name__ == '__main__':
    dataset_kwargs = {}
    dataset_kwargs['dataset'] = 'temp'
    dataset_kwargs['batch_size'] = 2
    dataset_kwargs['patch_size'] = (600, 960)
    dataset_kwargs['crop_size'] = 512
    dataset_kwargs['use_same_set'] = 0    
    dataset_kwargs['split'] = 'Train'

    create_temp_dataset(**dataset_kwargs)
