import tensorflow as tf
from model.modules_v2 import Encoder
from model.modules_v2 import Decoder
from model.modules_v2 import TemporalConvNet

EPS = 1e-8


class TasNet(tf.keras.Model):
    """ Time-domain Audio Separateiong Network

    Args:
        N (integer): Number of filters in autocoder
        L (integer): Length of the filters (in samples)
        B (integer): Number of the channels in bottleneck 1x1-conv block
        H (integer): Number of channels in convolutional blocks
        P (integer): Kernel size in convolutional blocks
        X (integer): Number of convolutional blocks in each repeat
        R (integer): Number of repeats
    """
    def __init__(self,
                 N,
                 L,
                 B,
                 H,
                 P,
                 X,
                 R,
                 C,
                 causal,
                 norm_type="cln",
                 name=None):
        super(TasNet, self).__init__(name=name)
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.C = C
        self.causal = causal
        self.norm_type = norm_type
        self.encoder = Encoder(N, L)
        self.sep_mask_estimator = TemporalConvNet(N,
                                                  B,
                                                  H,
                                                  P,
                                                  X,
                                                  R,
                                                  C,
                                                  causal=causal,
                                                  norm_type=norm_type,
                                                  name="Separator")
        self.decoder = Decoder(L)

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        # [batch_size, length]
        x = inputs
        # [batch_size, length,  channels]
        x = self.s_mixed_en = self.encoder(x)
        # [batch_size, C(speakers), length, N(channels)]
        x = self.s_mask_sep_logits = self.sep_mask_estimator(x, training)
        x = self.s_mask = tf.nn.sigmoid(x)
        x = self.s_mask * tf.expand_dims(self.s_mixed_en, axis=1)
        # [batch_size, speakers, time]
        x = self.decoder(self.s_sep_en)
        x = self.s_sep = x[:, :, 0:input_shape[1]]

        return x

    def compute_output_shape(self, input_shape):
        shape0 = (input_shape[0], self.C, input_shape[1])
        return shape0
