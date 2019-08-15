import numpy as np 
import tensorflow as tf
from tensorflow import keras
from model import utils

EPS = 1e-8


class Conv1DTranspose(keras.layers.Conv1D):
    """Transposed convolution layer (sometimes called Deconvolution).

    For `same` and `valid` padding, just call `tf.contrib.nn.conv1d_transpose`.
    For `causal` padding, using alias `valid` padding to call `tf.contrib.nn.conv1d_transpose`,
    then the first `stride` points along time dimension were dropped to get right output shape.
    """

    # TODO：only support dilation_rate=1
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv1DTranspose,
              self).__init__(filters,
                             kernel_size,
                             strides=strides,
                             padding=padding,
                             data_format=data_format,
                             dilation_rate=dilation_rate,
                             activation=activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             activity_regularizer=activity_regularizer,
                             kernel_constraint=kernel_constraint,
                             bias_constraint=bias_constraint,
                             **kwargs)

        self.output_padding = output_padding
        if self.padding == 'causal':
            self.real_padding = self.padding
            self.padding = 'valid'
        else:
            self.real_padding = None
        if self.output_padding is not None:
            self.output_padding = keras.layers.conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) +
                                     ' must be greater than output padding ' +
                                     str(self.output_padding))

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                'Inputs should have rank ' + str(3) +
                '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
            self.data_format_tf = 'NCW'
        else:
            channel_axis = -1
            self.data_format_tf = 'NWC'
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = keras.engine.base_layer.InputSpec(
            ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            t_axis = 2
        else:
            t_axis = 1

        time = input_shape[t_axis]
        kernel = self.kernel_size[0]
        stride = self.strides[0]
        if self.output_padding is None:
            out_pad = None
        else:
            out_pad = self.output_padding

        # Infer the dynamic output shape:
        out_time = keras.layers.conv_utils.deconv_length(
            time, stride, kernel, self.padding, out_pad, self.dilation_rate[0])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_time)
        else:
            output_shape = (batch_size, out_time, self.filters)

        outputs = tf.contrib.nn.conv1d_transpose(
            inputs,
            self.kernel,
            output_shape,
            stride,
            padding=self.padding.upper(),
            data_format=self.data_format_tf)

        if self.real_padding == 'causal':
            outputs = outputs[:, -(out_time - stride):, :]

        if self.use_bias:
            outputs = keras.backend.bias_add(outputs,
                                             self.bias,
                                             data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, t_axis = 1, 2
        else:
            c_axis, t_axis = 2, 1

        kernel = self.kernel_size[0]
        stride = self.strides[0]
        if self.output_padding is None:
            out_pad = None
        else:
            out_pad = self.output_padding

        output_shape[c_axis] = self.filters
        out_time = keras.layers.conv_utils.deconv_length(
            output_shape[t_axis], stride, kernel, self.padding, out_pad,
            self.dilation_rate[0])
        if self.real_padding == 'causal':
            output_shape[t_axis] = out_time - stride
        else:
            output_shape[t_axis] = out_time
        return tuple(output_shape)

    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()
        config['output_padding'] = self.output_padding
        return config


class DepthwiseConv1D(keras.layers.Conv1D):
    """Depthwise 1D convolution
   
    Args:
        kernel_size (integer): An integer, specifying the length of the 1D
            convolution window
        stride (integer): An integer, specifying the stide of the 1D convolution
        padding (String): One of "valid" or "same" 

    
    Returns:
        [tensor]: An 3D tensor with shape [batch, new_lengths, filters] 
    """
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 name=None,
                 **kwargs):
        super(DepthwiseConv1D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            # depth_multiplier=depth_multiplier,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            # depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            # depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            # depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = keras.initializers.get(
            depthwise_initializer)
        self.depthwise_regularizer = keras.regularizers.get(
            depthwise_regularizer)
        self.depthwise_constraint = keras.constraints.get(depthwise_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.strides = [1, self.strides[0], 1, 1]

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError(
                'Inputs to `DepthwiseConv1D` should have rank 3. '
                'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 2
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv1D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0], 1, input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim *
                                               self.depth_multiplier, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        # self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True
        # super(DepthwiseConv1D, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, -2)
        # import ipdb; ipdb.set_trace()
        if self.data_format == "channels_first":
            data_format = "NCHW"
        else:
            data_format = "NHWC"
        outputs = tf.nn.depthwise_conv2d(inputs,
                                         self.depthwise_kernel,
                                         strides=self.strides,
                                         padding=self.padding.upper(),
                                         rate=self.dilation_rate,
                                         data_format=data_format)

        if self.use_bias:
            outputs = keras.backend.bias_add(outputs,
                                             self.bias,
                                             data_format=self.data_format)

        outputs = tf.squeeze(outputs, [-2])
        if self.activation is not None:
            return self.activation(outputs)
        # outputs = super().call(inputs_expanded)
        return outputs

    def compute_output_shape(self, input_shape):
        out_filters = input_shape[-1] * self.depth_multiplier
        length = utils.conv_output_length(input_shape[1], self.kernel_size,
                                          self.padding, self.strides)
        return (input_shape[0], length, out_filters)

    @classmethod
    def get_input_shape(self, input_shape):
        input_shape = [input_shape[0], input_shape[1], 1, input_shape[2]]
        return input_shape


class NormalLayer(tf.keras.layers.Layer):
    """ Normalization Layer
    
    Args:
        axis (integer): An integer specifying the axis that should be normalized
        norm_type (string): Feature to be added
    """
    def __init__(self, norm_type, axis=None, name=None):
        super(NormalLayer, self).__init__(name=name)
        self.axis = axis
        self.norm_type = norm_type
        if norm_type == "gln":
            self.norm_layer = GlobalNormalization()
        elif norm_type == "cln":
            self.norm_layer = ChannelwiseLayerNorm()

    def build(self, input_shape):
        self.norm_layer.build(input_shape)

    def call(self, inputs, training=False):
        # the parameter training is reserved for BN
        return self.norm_layer(inputs, training)
        # x = tf.layers.batch_normalization(inputs, axis=self.axis, training=training)
        # return x

    def compute_output_shape(self, input_shape):
        return input_shape


class GlobalNormalization(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta = self.add_weight(name='beta',
                                    shape=tf.TensorShape(input_shape[-1]),
                                    initializer=tf.zeros_initializer,
                                    trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=tf.TensorShape(input_shape[-1]),
                                     initializer=tf.ones_initializer,
                                     trainable=True)
        # Be sure to call this at the end
        super(GlobalNormalization, self).build(input_shape)

    def call(self, inputs, training):
        means, variances = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        x = self.gamma * (inputs - means) / tf.pow(variances + EPS,
                                                   0.5) + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def get_config(self):
        base_config = super(GlobalNormalization, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ChannelwiseLayerNorm(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelwiseLayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta = self.add_weight(name='beta',
                                    shape=tf.TensorShape(input_shape[-1]),
                                    initializer=tf.zeros_initializer,
                                    trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=tf.TensorShape(input_shape[-1]),
                                     initializer=tf.ones_initializer,
                                     trainable=True)
        # Be sure to call this at the end
        self.built = True

    def call(self, inputs, training):
        means, variances = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        x = self.gamma * (inputs - means) / tf.pow(variances + EPS,
                                                   0.5) + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def get_config(self):
        base_config = super(ChannelwiseLayerNorm, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
