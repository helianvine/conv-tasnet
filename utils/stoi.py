import librosa
import numpy as np
import soundfile as sf


def load_wav(path, sample_rate=None):
    wav, fs = sf.read(path)
    if sample_rate and fs != sample_rate:
        wav = librosa.resample(wav, fs, sample_rate)
    return wav


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


def hanning(N, sflag='symmetric'):
    assert sflag in ['symmetric', 'periodic']
    if sflag == 'symmetric':
        w = sym_hanning(N)  # Does not include the first and last zero sample
    else:
        w = np.concatenate([[0],
                            sym_hanning(N)])  # Includes the first zero sample
    return w


def sym_hanning(N):
    if N % 2 == 0:
        # Even length window
        half = N / 2
        w = calc_hanning(half, N)
        w = np.concatenate([w, w[::-1]])
    else:
        # Odd length window
        half = (N + 1) / 2
        w = calc_hanning(half, N)
        w = np.concatenate([w, w[-2::-1]])
    return w


def calc_hanning(m, n):
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (n + 1)))
    return w


def removeSilentFrames(x, y, dyn_range, N, K):
    """[X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K)
    X and Y are segmented with frame-length N and overlap K, where the maximum energy
    of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the
    reconstructed signals, excluding the frames, where the energy of a frame
    of X is smaller than X_MAX-RANGE
    """

    w = hanning(N)
    x_frames = np.array([w * x[i:i + N] for i in range(0, len(x) - N + 1, K)])
    y_frames = np.array([w * y[i:i + N] for i in range(0, len(x) - N + 1, K)])
    x_energies = 20 * np.log10(
        np.sqrt(np.sum(np.square(x_frames), axis=1)) / np.sqrt(N) + 1e-10)
    mask = (x_energies - np.max(x_energies) + dyn_range) > 0
    x_frames = x_frames[mask]
    y_frames = y_frames[mask]

    x_sil = np.zeros((np.sum(mask) - 1) * K + N)
    y_sil = np.zeros((np.sum(mask) - 1) * K + N)
    for i in range(x_frames.shape[0]):
        x_sil[range(i * K, i * K + N)] += x_frames[i, :]
        y_sil[range(i * K, i * K + N)] += y_frames[i, :]
    return x_sil, y_sil


def stdft(x, N, K, N_fft):
    w = hanning(N)
    x_frames = np.array([x[i:i + N] for i in range(0, len(x) - N + 1, K)])
    return np.fft.rfft(x_frames * w, N_fft).T


def taa_corr(x, y):
    xn = x - np.mean(x, axis=-1, keepdims=True)
    xn = xn / np.sqrt(np.sum(xn**2, axis=-1, keepdims=True))
    yn = y - np.mean(y, axis=-1, keepdims=True)
    yn = yn / np.sqrt(np.sum(yn**2, axis=-1, keepdims=True))
    rho = np.sum(xn * yn, axis=-1)
    return np.mean(rho)


def stoi(x, y, fs_signal=None, fs=10000):
    """short-time objective intelligibility (STOI) measure.
    d = stoi(x, y, fs_signal) returns the output of the short-time
    objective intelligibility (STOI) measure described in [1, 2], where x
    and y denote the clean and processed speech, respectively, with sample
    rate fs_signal in Hz. The output d is expected to have a monotonic
    relation with the subjective speech-intelligibility, where a higher d
    denotes better intelligible speech. See [1, 2] for more details.

    References:
       [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
       Objective Intelligibility Measure for Time-Frequency Weighted Noisy
       Speech', ICASSP 2010, Texas, Dallas.
       [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
       Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
       IEEE Transactions on Audio, Speech, and Language Processing, 2011.
    """

    fs = fs
    N_frame = 256
    K = 512
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
    H, _ = thirdoct(fs, K, J, mn)

    if isinstance(x, (list, np.ndarray)):
        if not fs_signal:
            raise Exception(
                'Signal sample rate must be given for x when np.ndarray or list were used as inputs.'
            )
        elif fs_signal != fs:
            x = librosa.resample(x, fs_signal, fs)
    elif isinstance(x, str):
        x = load_wav(x, sample_rate=fs)
    else:
        raise Exception(
            'x must be np.ndarray or list, or the filename of the speech file')

    if isinstance(y, (list, np.ndarray)):
        if not fs_signal:
            raise Exception(
                'Signal sample rate must be given for y when np.ndarray or list were used as inputs.'
            )
        elif fs_signal != fs:
            y = librosa.resample(y, fs_signal, fs)
    elif isinstance(y, str):
        y = load_wav(y, sample_rate=fs)
    else:
        raise Exception(
            'y must be np.ndarray or list, or the filename of the speech file')

    if len(np.squeeze(x).shape) > 1 or len(np.squeeze(y).shape) > 1:
        raise Exception('x and y should be 1-d vector')
    if x.shape != y.shape:
        raise Exception('x and y should have the same length')

    x, y = removeSilentFrames(x, y, dyn_range, N_frame, N_frame // 2)

    x_hat = stdft(x, N_frame, N_frame // 2, K)
    y_hat = stdft(y, N_frame, N_frame // 2, K)

    X = np.sqrt(H.dot(np.abs(x_hat)**2))
    Y = np.sqrt(H.dot(np.abs(y_hat)**2))

    X_seg = np.array([X[:, i - N:i] for i in range(N, X.shape[1] + 1)])
    Y_seg = np.array([Y[:, i - N:i] for i in range(N, X.shape[1] + 1)])

    alpha = np.sqrt(
        np.sum(X_seg**2, axis=-1, keepdims=True) /
        np.sum(Y_seg**2, axis=-1, keepdims=True))
    aY_seg = Y_seg * alpha
    Y_prime = np.minimum(aY_seg, X_seg * (1 + 10**(-Beta / 20)))
    d = taa_corr(X_seg, Y_prime)
    return d
