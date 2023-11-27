import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from mobilevit.data.pipeline import load_label, load, preprocess

# Visualize original and preprocessed images and labels for debugging
def visualize_dataset(dataset, num_samples, crop_size, split):
    outdir1 = str(Path(__file__).parent.parent.parent / 'visualizations' / split / 'Original')
    outdir2 = str(Path(__file__).parent.parent.parent / 'visualizations' / split / 'Loaded')
    for outdir in [outdir1, outdir2]:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    for idx, (image_path, label_path) in enumerate(dataset.take(num_samples)):
        image, seg_map = load(image_path, label_path)
        keras.utils.save_img(outdir1 + f'/{idx}_img.png', image)
        keras.utils.save_img(outdir1 + f'/{idx}_label.png', seg_map)

        image, seg_map = preprocess(image, seg_map, crop_size, split)
        keras.utils.save_img(outdir2 + f'/{idx}_img.png', image*255)
        keras.utils.save_img(outdir2 + f'/{idx}_label.png', seg_map)

# Visualize outputs for debugging
def visualize_outputs(image, label, pred_label, idx, name='output'):
    vis_dir = str(Path(__file__).parent.parent.parent / 'visualizations' / name)

    if not os.path.exists(vis_dir):    
        os.makedirs(vis_dir)

    print(f'vis dir: {vis_dir}')
    keras.utils.save_img(vis_dir + f'/{idx}_rgb.png', tf.reverse(image[0], axis=[-1]))
    keras.utils.save_img(vis_dir + f'/{idx}_label.png', label[0], scale=True)
    keras.utils.save_img(vis_dir + f'/{idx}_pred_label.png', pred_label[0], scale=True)