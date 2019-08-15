import tensorflow as tf
import numpy as np

EPS = 1e-8


def thirdoct(fs, N_fft, numBands, mn):
    """[A CF] = THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix.
    inputs:
        FS:         samplerate
        N_FFT:      FFT size
        NUMBANDS:   number of bands
        MN:         center frequency of first 1/3 octave band
    outputs:
        A:          octave band matrix
        CF:         center frequencies
    """

    f = np.linspace(0, fs, N_fft + 1)
    f = f[:N_fft // 2 + 1]
    k = np.arange(numBands)
    cf = 2**(k / 3) * mn
    fl = np.sqrt(2**(k / 3) * mn) * np.sqrt(2**((k - 1) / 3) * mn)
    fr = np.sqrt(2**(k / 3) * mn) * np.sqrt(2**((k + 1) / 3) * mn)
    A = np.zeros((numBands, len(f)))

    for i in range(len(cf)):
        # a = np.min((f - fl[i]) ** 2)
        b = np.argmin((f - fl[i])**2)
        fl[i] = f[b]
        fl_ii = b

        # a = np.min((f - fr[i]) ** 2)
        b = np.argmin((f - fr[i])**2)
        fr[i] = f[b]
        fr_ii = b

        A[i, fl_ii:fr_ii] = 1

    rnk = np.sum(A, axis=1)
    numBands = np.where(
        (rnk[1:] >= rnk[:-1]) & ((rnk[1:] != 0) != 0) == 1)[0][-1] + 1 + 1
    A = A[:numBands, :]
    cf = cf[:numBands]
    return A, cf


def hann_window(window_length, dtype=None):
    if dtype:
        return tf.signal.hann_window(window_length + 2,
                                     periodic=False,
                                     dtype=dtype)[1:-1]
    else:
        return tf.signal.hann_window(window_length + 2,
                                     periodic=False,
                                     dtype=tf.float32)[1:-1]


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def removeSilentFrames(x, y, dyn_range, N, K):
    w = hann_window(N)
    x_frames = tf.signal.frame(x, N, K) * w
    y_frames = tf.signal.frame(y, N, K) * w
    x_energies = 20 * log10(
        tf.sqrt(tf.reduce_sum(tf.square(x_frames), axis=-1)) /
        tf.sqrt(float(N)) + 1e-10)
    mask = (x_energies - tf.reduce_max(x_energies) + dyn_range) > 0
    x_frames = tf.boolean_mask(x_frames, mask)
    y_frames = tf.boolean_mask(y_frames, mask)
    x_sil = tf.signal.overlap_and_add(x_frames, K)
    y_sil = tf.signal.overlap_and_add(y_frames, K)
    return x_sil, y_sil


def stdft(x, N, K, N_fft):
    stft = tf.signal.stft(x, N, K, N_fft, window_fn=hann_window)
    return tf.transpose(stft)


def taa_corr(x, y):
    xn = x - tf.reduce_mean(x, axis=-1, keepdims=True)
    xn = xn / tf.sqrt(tf.reduce_sum(tf.square(xn), axis=-1, keepdims=True))
    yn = y - tf.reduce_mean(y, axis=-1, keepdims=True)
    yn = yn / tf.sqrt(tf.reduce_sum(tf.square(yn), axis=-1, keepdims=True))
    rho = tf.reduce_sum(xn * yn, axis=-1)
    return tf.reduce_mean(rho)


def tfstoi(x, y, fs=16000):
    """ STOI implement with tensorflow.
    Origin stoi was implement with 10kHz, but audio resampling is not easy for tensoflow,
    so this was reimplement with 16kHz, which is actually in mostly use.
    TODO: satisfiy different sample rate.
    Only support 1-D tensor, since `removeSilentFrames` will make samples unable to batch.

    :param x: 1-D Tensor, [N,]. Reference signal.
    :param y: 1-D Tensor, same shape with x. Enhanced signal.
    :param fs: Sample rate for input signal. Default 16000.
    :return: scalar value.
    """

    fs = fs
    N_frame = 256
    N_fft = 512
    mn = 150
    Beta = -15
    dyn_range = 40

    if fs == 10000:
        J = 15
        N = 30
    elif fs == 16000:
        J = 20
        N = 48
    else:
        raise ValueError("Not support sample rate yet.")
    H, _ = thirdoct(fs, N_fft, J, mn)
    H = tf.convert_to_tensor(H.astype('float32'))

    x, y = removeSilentFrames(tf.convert_to_tensor(x), tf.convert_to_tensor(y),
                              dyn_range, N_frame, N_frame // 2)
    x_hat = stdft(x, N_frame, N_frame // 2, N_fft)
    y_hat = stdft(y, N_frame, N_frame // 2, N_fft)

    X = tf.sqrt(tf.matmul(H, tf.abs(x_hat)**2))
    Y = tf.sqrt(tf.matmul(H, tf.abs(y_hat)**2))

    X_seg = tf.transpose(tf.signal.frame(X, N, 1), (1, 0, 2))
    Y_seg = tf.transpose(tf.signal.frame(Y, N, 1), (1, 0, 2))

    alpha = tf.sqrt(
        tf.reduce_sum(tf.square(X_seg), axis=-1, keepdims=True) /
        tf.reduce_sum(tf.square(Y_seg), axis=-1, keepdims=True))
    aY_seg = tf.multiply(Y_seg, alpha)
    Y_prime = tf.minimum(aY_seg, X_seg * (1 + 10**(-Beta / 20)))

    d = taa_corr(X_seg, Y_prime)
    return d