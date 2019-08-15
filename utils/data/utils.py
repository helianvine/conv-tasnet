import os
import fnmatch
import soundfile as sf


def get_file_list(data_dir, pattern, recursive=False):
    file_list = []
    if recursive:
        for root, _, files in os.walk(data_dir):
            # print(root, files)
            files = fnmatch.filter(files, pattern)
            file_list.extend([os.path.join(root, i) for i in files])
    else:
        files = os.listdir(data_dir)
        files = fnmatch.filter(files, pattern)
        file_list.extend([os.path.join(data_dir, i) for i in files])
    file_num = len(file_list)
    assert file_num > 0, \
        "No file found in folder {} with pattren {}".format(
            data_dir, pattern)
    file_list.sort()
    return file_list


def audio_read(path):
    r"""Read audio data from file. Note that only the first channel is returned 
        if the number of channels is greater than one. 
    """
    data, _ = sf.read(path, always_2d=True)
    data = data[:, 0]
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
