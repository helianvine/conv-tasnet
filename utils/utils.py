import os
import librosa
import numpy as np
import soundfile as sf


def audio_read(path, fs=16000):
    data, sr = sf.read(path, always_2d=True)
    data = data[:, 0]
    if fs != sr:
        data = librosa.resample(data, sr, fs)
    return data


def audio_write(filename, data, fs):
    """write audio data to file
    
    Args:
        filename
        data: [description]
        fs: [description]
    """
    filepath, _ = os.path.split(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    sf.write(filename, data, fs, subtype="PCM_16")
  