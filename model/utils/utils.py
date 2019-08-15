import os
import shutil
import numpy as np
import soundfile as sf

EPS = 1e-8


class AudioChecker(object):
    """AudioChecker
    
    Args:
        name (string): Name
        max_keep (integer): Maximum number of files to keep
    """
    def __init__(self, filepath, fs=16000, max_keep=5):
        self.path = filepath
        self.fs = fs
        self.max_keep = max_keep
        self.tr_filelist = []
        self.dir_dict = {"tr": os.path.join(self.path, "tr"), 
                         "cv": os.path.join(self.path, "cv"),
                         "tt": os.path.join(self.path, "tt")}
        self._idx_dict = {"tr": 0,
                          "cv": 0,
                          "tt": 0}
        self._epoch = 0

    def _update_tr_list(self, path):
        self.tr_filelist.append(path)
        if len(self.tr_filelist) > self.max_keep:
            path = self.tr_filelist.pop(0)
            try:
                shutil.rmtree(path)
            except Exception:
                # print("Remove {} failed!".format(path))
                pass

    def write(self, name_list, data, global_step, fs=16000, normalize=True, 
              stage="tt", epoch=None):
        """write
    
        Args:
            data (ndarray): The data to write with shape (channel, samples) or (samples, ) 
            fs (integer): Sampling frequency 
            global_step (integer): An integer 
            training (bool, optional): Defaults to True.
        """
        if isinstance(data, tuple) or isinstance(data, list):
            try:
                data = np.array(data)
            except Exception:
                raise ValueError

        # audioname = path + "/{}/" + self.name + "-{}-{}.wav"
        # audio_dir = self.dir_dict[stage]
        if normalize:
            data = data / max(np.max(np.abs(data)), 1e-8)
        audio_dir = self.dir_dict[stage]
        if epoch is not None:
            audio_dir = os.path.join(self.dir_dict[stage], 
                                     "epoch-{:0>3d}".format(epoch))
            audio_dir = os.path.join(audio_dir, 
                                     "{:0>6d}").format(global_step)
        
        audioname = os.path.join(audio_dir, "{:0>6d}-{}")
                                
        if stage == "tr":
            self._update_tr_list(audio_dir)
        elif stage == "cv": 
            if epoch > 0 and epoch != self._epoch:
                try:
                    shutil.rmtree(
                        os.path.join(self.dir_dict["tr"], 
                                     "epoch-{:0>3d}".format(self._epoch)))
                    shutil.rmtree(
                        os.path.join(self.dir_dict["cv"], 
                                     "epoch-{:0>3d}".format(self._epoch)))
                    self.tr_filelist = []
                    self._epoch = epoch
                    self._idx_dict["tr"] = 0
                    self._idx_dict["cv"] = 0
                except Exception:
                    pass  
        elif stage == "tt":
            pass
        else:
            print("stage must be one of 'tr', 'cv' or 'tt'.")
            raise ValueError

        # batch_size = data.shape[0]
        for i, di in enumerate(data):
            _, name = os.path.split(name_list[i]) 
            audio_write(audioname.format(self._idx_dict[stage], name), 
                        di, 
                        fs)
            self._idx_dict[stage] += 1


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


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.
    Arguments:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full", "causal"
        stride: integer.
        dilation: dilation rate, integer.
    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ['same', 'causal']:
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride
