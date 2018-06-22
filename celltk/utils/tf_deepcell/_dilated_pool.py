from tensorflow.python.ops import nn
from tensorflow import nn

from tensorflow.python.layers.pooling import _Pooling2D
from tensorflow.python.layers import utils
try:
    from tensorflow.python.keras.layers import Layer
except:
    from tensorflow.contrib.keras.python.keras.layers import Layer

# from tensorflow.python.keras._impl.keras.utils import conv_utils


# class DilatedMaxPool2D(_Pooling2D, Layer):
#     def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
#                  data_format='channels_last', name=None,
#                  dilation_rate=2, **kwargs):
#         self.dilation_rate = dilation_rate
#         if strides is None or dilation_rate > 1:
#             strides = (1, 1)
#         super(DilatedMaxPool2D, self).__init__(
#               nn.pool, pool_size=pool_size, strides=strides,
#               padding=padding, data_format=data_format, name=name, **kwargs)

#     def call(self, inputs):
#         outputs = self.pool_function(
#             inputs, window_shape=self.pool_size, pooling_type="MAX",
#             strides=self.strides, padding=self.padding.upper(),
#             dilation_rate=(self.dilation_rate, self.dilation_rate),
#             data_format=utils.convert_data_format(self.data_format, 4))
#         return outputs

#     def get_config(self):
#         config = {
#             'pool_size': self.pool_size,
#             'padding': self.padding,
#             'strides': self.strides,
#             'data_format': self.data_format,
#             'dilation_rate': self.dilation_rate,
#         }
#         base_config = super(DilatedMaxPool2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


class DilatedMaxPool2D(_Pooling2D, Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format='channels_last', name=None,
                 dilation_rate=2, **kwargs):
        self.dilation_rate = dilation_rate
        if strides is None or dilation_rate > 1:
            strides = (1, 1)
        super(DilatedMaxPool2D, self).__init__(
              nn.pool, pool_size=pool_size, strides=strides,
              padding=padding, data_format=data_format, name=name, **kwargs)

    def call(self, inputs):
        outputs = self.pool_function(
            inputs, window_shape=self.pool_size, pooling_type="MAX",
            strides=self.strides, padding=self.padding.upper(),
            dilation_rate=(self.dilation_rate, self.dilation_rate),
            data_format=utils.convert_data_format(self.data_format, 4))
        return outputs

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
        }
        base_config = super(DilatedMaxPool2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
