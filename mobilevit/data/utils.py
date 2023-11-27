import os
import shutil
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from mobilevit.data.pipeline import load, preprocess

# Load config file for dataset
def read_config_file(cfg_file_path):
    config_data = {}
    with open(cfg_file_path, "r") as f:
        exec(f.read(), config_data)
    return config_data

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
