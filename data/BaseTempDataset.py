import os
import shutil
import random
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers, backend
from data.pipeline import apply_transform
# import tensorflow_models as tfm

DATASET_CONFIG_PATH = Path(__file__).parent / "config" / "temp.py"
DATASET_CATS_PATH = Path(__file__).parent / "config" / "temp.yml"

IGNORE_LABEL = 255  # Define the ignore label value

def read_config_file(cfg_file_path):
    config_data = {}
    with open(cfg_file_path, "r") as f:
        exec(f.read(), config_data)
    return config_data

def load(image_path, seg_map_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape((600, 960, 3))

    seg_map = tf.io.read_file(seg_map_path)
    seg_map = tf.image.decode_image(seg_map, channels=1)
    seg_map.set_shape((600, 960, 1))

    return image, seg_map

def preprocess(image, seg_map, crop_size, split):        
    # options: 0 - tf.image transforms, 1- tfm.preprocess_ops, 2 - keras sequential model
    image, seg_map = apply_transform(image, seg_map, crop_size, split, option=0)

    return image, seg_map

def visualize_dataset(dataset, num_samples, crop_size, split):
    outdir1 = str(Path(__file__).parent.parent / 'visualizations' / split / 'Original')
    outdir2 = str(Path(__file__).parent.parent / 'visualizations' / split / 'Loaded')
    for outdir in [outdir1, outdir2]:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    for idx, (image_path, label_path) in enumerate(dataset.take(num_samples)):
        image, seg_map = load(image_path, label_path)
        keras.utils.save_img(outdir1 + f'/{idx}_img.jpg', image)
        keras.utils.save_img(outdir1 + f'/{idx}_label.jpg', seg_map)
        image, seg_map = preprocess(image, seg_map, crop_size, split)
        keras.utils.save_img(outdir2 + f'/{idx}_img.jpg', image*255)
        keras.utils.save_img(outdir2 + f'/{idx}_label.jpg', seg_map)

def examples_per_class(image_paths, label_paths):
    class2image = {}
    class2label = {}    
    count = 0

    for image_path, label_path in zip(image_paths, label_paths):
        image, label = load(image_path, label_path)
        unique_labels = np.unique(label)
        if len(unique_labels) > 2:
            count += 1
        
        label_class = int(unique_labels[1])
        if label_class not in class2image:
            class2image[label_class] = []
        else:
            class2image[label_class].append(image_path)
        
        if label_class not in class2label:
            class2label[label_class] = []
        else:
            class2label[label_class].append(label_path)

    return class2image, class2label, count

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

def same_set_train_val(dataset_dir, split):
    img_paths, seg_paths = diff_set_train_val(dataset_dir=dataset_dir, split='Train')
    
    # Creating dictionary of paths per class from train set itself
    class2image, class2label, count = examples_per_class(img_paths, seg_paths)

    image_paths = []
    seg_map_paths = []

    # Keeping train set as 90% of samples and 10% for validation
    for cls in class2image.keys():
        cls_len = len(class2image[cls])
        train_len = int(0.9 * cls_len)
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

    auto = tf.data.AUTOTUNE
    
    # target names and colors - read DATASET_CATS_PATH

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

    visualize_dataset(dataset=dataset, num_samples=5, crop_size=crop_size, split=split)

    # Define the function to load and preprocess the image and segmentation map
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
