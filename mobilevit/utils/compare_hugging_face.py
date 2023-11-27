import tensorflow as tf

from mobilevit.utils.fetch_and_summarize import get_torch_model, get_tf_model

def compare():
    name = 'mobilevit_s'
    task = 'segment'
    resolution = 512
    classes = 64

    hf_model = get_torch_model(name, task, name='hf', save=True)
    my_model = get_tf_model(name, resolution, classes, task, name='my', save=True)

if __name__ == '__main__':
    compare()
    