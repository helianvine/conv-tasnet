from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from model.layers import NormalLayer
from model.layers import DepthwiseConv1D

EPS = 1e-8


class Decoder(tf.keras.Model):
    def __init__(self, kernel_size):
        super(Decoder, self).__init__(name="Decoder")
        self.kernel_size = (1, kernel_size)
        self.frame_shift = (1, kernel_size // 2)
        self.conv2d_trans = keras.layers.Conv2DTranspose(1,
                                                         self.kernel_size,
                                                         self.frame_shift)
        # self.kernel_size = kernel_size
        # self.frame_shift = kernel_size // 2
        # self.dense = keras.layers.Dense(kernel_size, use_bias=False)

    def call(self, inputs):
        x = inputs
        x = self.conv2d_trans(x)
        x = tf.squeeze(x, axis=-1)
        # import ipdb; ipdb.set_trace()
        # x = self.dense(x)
        # x = tf.signal.overlap_and_add(x, self.frame_shift)
        return x

    def compute_output_shape(self, input_shape):
        # output_length = (input_shape[-2] - 1) * self.frame_shift \
        #     + self.kernel_size
        # return (input_shape[0], input_shape[1], output_length)
        return self.conv2d_trans.compute_output_shape(input_shape)[0:3]


class Encoder(tf.keras.Model):
    """ A 1-D Convolutional Layer

    Args:
        filters (integer): An integer specifying the number of filters for encoder
        kernel_size (integer): An integer specifying the kernel_size of filters 
    """
    def __init__(self, filters, kernel_size):
        super(Encoder, self).__init__(name="Encoder")
        self.filters = filters
        self.kernel_size = kernel_size

        self.conv0 = keras.layers.Conv1D(filters=filters,
                                         kernel_size=kernel_size,
                                         strides=kernel_size // 2,
                                         activation=tf.nn.relu,
                                         use_bias=False,
                                         padding="valid",
                                         name="conv_0")

    def call(self, inputs):
        x = inputs
        x = self.conv0(x)
        # x = x / (tf.sqrt(tf.reduce_sum(x**2, axis=[1, 2], keep_dims=True)) + EPS)
        return x

    def compute_output_shape(self, input_shape):
        return self.conv0.compute_output_shape(input_shape)


class TemporalConvNet(tf.keras.Model):
    """ Temporal Convolutional Network for estimating the sepatation mask

    Args:
        N (integer): Number of filters in autocoder 
        B (integer): Number of the channels in bottleneck 1x1-conv block 
        H (integer): Number of channels in convolutional blocks 
        P (integer): Kernel size in convolutional blocks
        X (integer): Number of convolutional blocks in each repeat
        R (integer): Number of repeats
        C (integer): Number of speakers
    """
    def __init__(self, N, B, H, P, X, R, C, causal, norm_type="cln",
                 name=None):
        super(TemporalConvNet, self).__init__(name=name)
        self.N = N
        self.B = B
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.C = C
        self.norm_in = NormalLayer(norm_type="cln", name="layer_norm_in")
        self.causal = causal
        self.norm_type = norm_type
        self.conv_bottleneck = keras.layers.Conv1D(filters=B,
                                                   kernel_size=1,
                                                   name="bottleneck_layer")
        self.conv_blocks = []
        name_str = "mini_block_{}/conv_block_{}"
        for r in range(R):
            for n in range(X):
                dilation = 2**n
                self.conv_blocks.append(
                    TemporalConvBlock(B,
                                      H,
                                      P,
                                      dilation,
                                      causal=causal,
                                      norm_type=norm_type,
                                      name=name_str.format(r, n)))
        self.tcn_out_layer = keras.layers.Conv1D(filters=N,
                                                 kernel_size=1,
                                                 name="conv1d_output")
        self.norm_out = NormalLayer(norm_type=norm_type, name="norm_output")

        self.prelu = keras.layers.PReLU(shared_axes=[1], name="prelu_out")

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        # [batch_size, Length, N]
        x = inputs
        x = self.norm_in(x)
        x = self.conv_bottleneck(x)
        outputs = []
        for conv_block in self.conv_blocks:
            z = conv_block(x, training)
            x += z[0]
            outputs.append(z[1])
        for i in outputs:
            x += i
        # x = tf.concat(self.outputs, axis=-1)
        x = self.prelu(x)
        x = self.tcn_out_layer(x)
        # [batch_size, legnth, speakers, N]
        x = tf.reshape(x, [input_shape[0], input_shape[1], self.C, self.N])
        x = self.norm_out(x)
        # [batch_size, speakers, legnth, N]
        return tf.transpose(x, [0, 2, 1, 3])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.C, input_shape[1], self.N)


class TemporalConvBlock(tf.keras.Model):
    def __init__(self,
                 channels,
                 filters,
                 kernel_size,
                 dilation,
                 causal,
                 norm_type="gln",
                 name=None):
        super(TemporalConvBlock, self).__init__(name=name)
        self.channels = channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal

        self.conv0 = keras.layers.Conv1D(filters=filters,
                                         kernel_size=1,
                                         name='conv1d_0')
        self.prelu0 = keras.layers.PReLU(shared_axes=[1], name="prelu_0")
        self.norm0 = NormalLayer(norm_type=norm_type, name="norm_0")

        self.depthwise_conv = DepthwiseConv1D(kernel_size=kernel_size,
                                              dilation_rate=dilation,
                                              name="depthwise_conv1d")
        self.prelu1 = keras.layers.PReLU(shared_axes=[1], name="prelu_1")
        self.norm1 = NormalLayer(norm_type=norm_type, name="norm_1")
        self.conv1 = keras.layers.Conv1D(filters=channels,
                                         kernel_size=1,
                                         name='conv1d_1')
        self.conv2 = keras.layers.Conv1D(filters=channels,
                                         kernel_size=1,
                                         name='conv1d_2')

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv0(x)
        x = self.prelu0(x)
        x = self.norm0(x, training)
        padding_size = self.dilation * (self.kernel_size - 1)
        if self.causal:
            padding = [[0, 0], [padding_size, 0], [0, 0]]
        else:
            padding = [[0, 0], [padding_size // 2, padding_size // 2], [0, 0]]
        x = tf.pad(x, padding)
        x = self.depthwise_conv(x)
        x = self.prelu1(x)
        x = self.norm1(x, training)

        # shape = inputs.get_shape().as_list()
        # x = tf.slice(x, [0, 0, 0], shape)
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        return (x1, x2)

    def compute_output_shape(self, input_shape):
        return (input_shape, input_shape)
