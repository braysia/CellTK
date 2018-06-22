
import os
import imp
try:
    from tensorflow.python.keras import backend
    from tensorflow.python.keras.layers import Layer, Conv2D, MaxPooling2D
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.models import load_model
except:
    from tensorflow.contrib.keras.python.keras.engine.topology import Layer
    from tensorflow.contrib.keras.python.keras import backend
    from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.contrib.keras.python.keras.models import Sequential
    from tensorflow.contrib.keras.python.keras.models import load_model
from _dilated_pool import DilatedMaxPool2D
import numpy as np
from scipy.ndimage import imread as imread0
import tifffile as tiff


class Squeeze(Layer):
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(Squeeze, self).__init__(**kwargs)

    def call(self, x):
        x = backend.squeeze(x, axis=2)
        return backend.squeeze(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])

    # def get_config(self):
    #     config = {'output_dim': self.output_dim}
    #     base_config = super(Squeeze, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))



def convert_model_patch2full(model):
    """
    """
    dr = 1
    new_model = Sequential()
    for nl, layer in enumerate(model.layers):
        if layer.name.startswith('squeeze_'):
            continue
        if isinstance(layer, MaxPooling2D):
            newl = DilatedMaxPool2D(dilation_rate=dr)
            newl = newl.from_config(layer.get_config())
            newl.strides, newl.dilation_rate = (1, 1), dr
            new_model.add(newl)
            dr = dr * 2
            continue
        if isinstance(layer, Conv2D):
            if not layer.kernel_size == (1, 1):
                layer.dilation_rate = (dr, dr)
                new_model.add(layer)
            else:
                newl = Conv2D(layer.filters, layer.kernel_size, input_shape=layer.input_shape[1:])
                new_model.add(newl.from_config(layer.get_config()))
        else:
            new_model.add(layer)
    return new_model


def load_model_py(path):
    if path.endswith('.py'):
        fname = os.path.basename(path).split('.')[0]
        module = imp.load_source(fname, path)
        return module.model
    elif path.endswith('.hdf5'):
        return load_model(path, custom_objects={'Squeeze':Squeeze})



def make_outputdir(output):
    try:
        os.makedirs(output)
    except:
        pass


def imread_check_tiff(path):
    img = imread0(path)
    if img.dtype == 'object' or path.endswith('tif'):
        img = tiff.imread(path)
    return img


def imread(path):
    if isinstance(path, tuple) or isinstance(path, list):
        st = []
        for p in path:
            st.append(imread_check_tiff(p))
        img = np.dstack(st)
        if img.shape[2] == 1:
            np.squeeze(img, axis=2)
        return img
    else:
        return imread_check_tiff(path)


def parse_image_files(inputs):
    if "/" not in inputs:
        return (inputs, )
    store = []
    li = []
    while inputs:
        element = inputs.pop(0)
        if element == "/":
            store.append(li)
            li = []
        else:
            li.append(element)
    store.append(li)
    return zip(*store)