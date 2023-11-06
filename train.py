import os
import pdb
import yaml
import time
import argparse
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow import keras
from datetime import datetime
from keras import layers, models, datasets

from data.factory import create_dataset
from feat_extractor import get_fe_model
from model import mobilevit
from convert import get_torch_model, copy_weights

from setproctitle import setproctitle                                                                                                                                                                                
setproctitle("user:Vivek")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training TensorFlow model."
    )
    parser.add_argument('-d', '--dataset', default="zebra", type=str, required=False, help='Name of the dataset')
    parser.add_argument('-m', '--model', default="mobilevit_s", type=str, required=False, choices=['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s'], help='Flavors of MobileViT models')
    parser.add_argument('-r', '--resolution', default=512, type=int, required=False, help='Image resolution of the model.')
    parser.add_argument('-t', '--task', default='segment', type=str, required=False, choices=['classify', 'segment'], help='Downstream task')
    parser.add_argument('-u', '--use_pretrained', action='store_true', help='Use pretrained feature extractor')
    return vars(parser.parse_args())

def load_config():
    return yaml.load(open(Path(__file__).parent / 'config.yml', 'r'), Loader=yaml.FullLoader)

def train(args):    
    # Loading configuration
    cfg = load_config()
    save_path = Path(__file__).parent / 'saved_models' / f'{args["model"]}_{args["task"]}' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print('Preparing Data..')

    dataset = args['dataset']

    # Dataset dict
    dataset_kwargs = {}
    dataset_kwargs['dataset'] = args['dataset']
    dataset_kwargs['batch_size'] = cfg[dataset]['batch_size']
    dataset_kwargs['patch_size'] = cfg[dataset]['patch_size']
    dataset_kwargs['crop_size'] = args['resolution']
    dataset_kwargs['use_same_set'] = 1
    
    batch_size = dataset_kwargs['batch_size']
    
    dataset_kwargs['split'] = 'Train'
    train_dataset = create_dataset(dataset_kwargs)
    train_ds = train_dataset.batch(batch_size, drop_remainder=True)

    dataset_kwargs['split'] = 'Val'
    val_dataset = create_dataset(dataset_kwargs)
    val_ds = val_dataset.batch(batch_size, drop_remainder=True)

    print(f'{args["dataset"]} dataset created\n')

    num_train = train_dataset.cardinality()
    num_val = val_dataset.cardinality()

    print(f'Number of training examples: {num_train}')
    print(f'Number of validation examples: {num_val}\n')

    # Instantiating model
    if args['use_pretrained']:
        print('Using PreTrained model..')
        tf_model = get_fe_model(num_classes=cfg[dataset]['num_classes'])
    else:
        print('Instantiating Model..')
        tf_model = mobilevit.get_mobilevit_model(
            model_name=args['model'],
            image_shape=(args['resolution'], args['resolution'], 3),
            num_classes=cfg[dataset]['num_classes'],
            task=args['task'],
        )

    print(f'Initializing model with pretrained weights')
    torch_model = get_torch_model(model_name=args['model'], task=args['task'])
    model = copy_weights(torch_model, tf_model, args['task'])
    print(f"{args['model']} initialized")
    
    logdir = os.path.join('logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    print(f'Created logdir: {logdir}')

    writer = tf.summary.create_file_writer(logdir)

    weight_decay = 0.01
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.00009)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = keras.metrics.MeanIoU(num_classes=64)
    val_acc_metric = keras.metrics.MeanIoU(num_classes=64)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[keras.metrics.MeanIoU(num_classes=64)],
    )
    
    print('Training..')

    train_acc_list=[]
    train_loss_list=[]
    epochs = 50

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            logits = tf.image.resize(logits, size=y.shape[1:-1], method='bilinear')
            loss_value = loss_fn(y, logits)
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        pred_labels = tf.argmax(logits, axis=-1)
        pred_labels = tf.expand_dims(pred_labels, -1)
        train_acc_metric.update_state(y, pred_labels)

        return loss_value, logits, pred_labels, grads

    @tf.function
    def test_step(x, y):
        logits = model(x, training=False)
        logits = tf.image.resize(logits, size=y.shape[1:-1], method='bilinear')
        loss_value = loss_fn(y, logits)
        pred_labels = tf.argmax(logits, axis=-1)
        pred_labels = tf.expand_dims(pred_labels, -1)
        val_acc_metric.update_state(y, pred_labels)

        return loss_value

    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        start_time = time.time()
        train_loss = []
        val_loss = []

        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            loss_value, logits, pred_labels, grads = train_step(x_batch_train, y_batch_train)

            # with writer.as_default():
            #     for layer, grad in zip(model.trainable_weights, grads):
            #         if layer.shape != grad.shape:
            #             print('Layer {layer.name} gradient not available')
            #         tf.summary.histogram(layer.name, grad, step=step)
    
            #     tf.summary.image("Training data", x_batch_train, step=epoch)
            #     tf.summary.image("Labels", y_batch_train, step=epoch)
            #     tf.summary.image("Predictions", pred_labels, step=epoch)
            
            train_loss.append(float(loss_value))
            if step % 85 == 0:
                print(f"Training: step: {step} seen_samples: {(step + 1) * batch_size} loss: {float(loss_value)}")
        
        train_loss_list.append(np.mean(train_loss))

        train_acc = train_acc_metric.result()
        train_acc_list.append(float(train_acc))
        train_acc_metric.reset_states()
        with writer.as_default():
            tf.summary.scalar('train loss', np.mean(train_loss), step=epoch)
            tf.summary.scalar('train accuracy', train_acc, step=epoch)

        # Validation
        for x_batch_val, y_batch_val in val_ds:
            loss_value = test_step(x_batch_val, y_batch_val)
            val_loss.append(float(loss_value))

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        with writer.as_default():
            tf.summary.scalar('val loss', np.mean(val_loss), step=epoch)
            tf.summary.scalar('val accuracy', val_acc, step=epoch)

        print(f"Epoch {epoch}, Loss: {np.mean(train_loss)}, Train_Acc: {train_acc}, Val_Loss:{np.mean(val_loss)}, Val_Accuracy: {val_acc}, Learning Rate: {optimizer.learning_rate.numpy()}, Time taken: {(time.time() - start_time)}")

        if epoch % 5 == 0:
            model.save(os.path.join(str(save_path), f'epoch_{epoch}'))
            print(f'Stored model {epoch} at {os.path.join(str(save_path), f"epoch_{epoch}")}')

if __name__ == '__main__':
     args = parse_args()
     train(args)




# Printing layer weights across epochs
# pdb.set_trace()
# for layer in ['stem_bn', 'stack3_block2_attn_ln']:
#     print(layer)
#     if isinstance(model.get_layer(layer), layers.BatchNormalization):
#         mean = model.get_layer(layer).moving_mean.numpy()
#         var = model.get_layer(layer).moving_variance.numpy()
#         print(f'mean: {mean}')
#         print(f'var: {var}')
    
#     gamma = model.get_layer(layer).gamma.numpy()
#     beta = model.get_layer(layer).beta.numpy()
#     print(f'gamma: {gamma}')
#     print(f'beta: {beta}')


# print(f'step: {optimizer.iterations.numpy()} lr: {optimizer.learning_rate.numpy()}')
            # with writer.as_default():
            #     tf.summary.image("Training data", x_batch_train, step=epoch)
            #     tf.summary.image("Labels", y_batch_train, step=epoch)
            #     tf.summary.image("Predictions", pred_labels, step=epoch)
