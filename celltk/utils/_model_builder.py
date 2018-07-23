import keras.layers
import keras.models
import tensorflow as tf

CONST_DO_RATE = 0.5

option_dict_conv = {"activation": "relu", "border_mode": "same"}
option_dict_bn = {"mode": 0, "momentum" : 0.9}


# returns a core model from input to 64 channels of the same size
def get_core(dim1, dim2, dim3):

    # assume dim1 x dim2 image with dim3 color channels

    x = keras.layers.Input(shape=(dim1, dim2, dim3))

    a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(x)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)

    a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(a)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)


    y = keras.layers.MaxPooling2D()(a)

    b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)

    b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(b)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)


    y = keras.layers.MaxPooling2D()(b)

    c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)

    c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(c)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)


    y = keras.layers.MaxPooling2D()(c)

    d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(y)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)

    d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(d)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)


    # UP

    d = keras.layers.UpSampling2D()(d)

    y = keras.layers.merge([d, c], concat_axis=3, mode="concat")

    e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(e)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.UpSampling2D()(e)


    y = keras.layers.merge([e, b], concat_axis=3, mode="concat")

    f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(f)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.UpSampling2D()(f)


    y = keras.layers.merge([f, a], concat_axis=3, mode="concat")

    y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    return [x, y]

def get_model(dim1, dim2, dim3, activation="softmax"):

    [x, y] = get_core(dim1, dim2, dim3)

    y = keras.layers.Convolution2D(3, 1, 1, **option_dict_conv)(y)

    if activation is not None:
        y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)

    return model