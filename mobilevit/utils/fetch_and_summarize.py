import os
import torch

import tensorflow as tf
from tensorflow import keras
from torchinfo import summary

from pathlib import Path
from datetime import datetime

from mobilevit.model import mobilevit_pt, mobilevit

def torch_summary(model, task, name='hf', resolution=256):
    logdir = str(Path(__file__).parent.parent.parent / 'model_summary' / task)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    save_path = os.path.join(logdir, datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_torch_{name}.txt'))
    stats = str(summary(model, (3, 512, 512), batch_dim = 0, col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose = 1, depth=15))
    
    with open(save_path, 'w') as f:
        f.write(stats)

def tf_summary(model, task, name='my'):
    logdir = str(Path(__file__).parent.parent.parent / 'model_summary' / task)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    save_path = os.path.join(logdir, datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_tf_{name}.txt'))

    def tf_summary(s):
        with open(save_path,'a') as f:
            print(s, file=f)

    model.summary(print_fn=tf_summary, expand_nested=True)


def get_torch_model(model_name, task, name='hf', save=False):
    torch_model = mobilevit_pt.get_mobilevit_pt(model_name=model_name, task=task, name=name)
    torch_model.eval()  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_model.to(device)

    if save:
        torch_summary(torch_model, task, name)

    return torch_model

def get_tf_model(model_name, resolution, num_classes, task, name='my', save=False):
    tf_model = mobilevit.get_mobilevit_model(
        model_name=model_name,
        image_shape=(resolution, resolution, 3),
        num_classes=num_classes,
        task=task,
    )

    if save:
        tf_summary(tf_model, task, name)

    return tf_model

def get_model(set_type, resolution, ckpt_path=''):
    models = {
        'diff_set': {
            '256': '/nfs/bigiris/vgolani/zebra/MobileViT/saved_models/mobilevit_s_segment/2023-10-12_15-06-29/epoch_60/',
            '512': 'saved_models/mobilevit_s_segment/2023-11-02_16-59-32/epoch_45',
        },
        'same_set': {
            '256': '',
            '512': 'saved_models/mobilevit_s_segment/2023-11-03_01-32-45/epoch_45',
        }
    }

    model = {} 
    if ckpt_path:
        model['model'] = keras.models.load_model(ckpt_path)
    elif set_type and resolution:
        model['model'] = keras.models.load_model(models[set_type][resolution])
    else:
        model = get_mo

    model['loss_fn'] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model['train_acc_metric'] = keras.metrics.MeanIoU(num_classes=64)
    model['val_acc_metric'] = keras.metrics.MeanIoU(num_classes=64)

    return model