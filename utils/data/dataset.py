import os
import abc
import pickle
from . utils import audio_read
from . utils import get_file_list


class BaseDataset(object):
    # complitable with keras' Sequence
    use_sequence_api = True

    @abc.abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


class SeparatedDataset(BaseDataset):
    r"""Combined speparated datasets. (eg: input features and labels are stored in 
        two separated dataset, then SeparatedDateset can be used. )
        Every `Dataset` must implement the `__getitem__` and the `__len__` methods.
        The method `__getitem__` should return a complete batch. The length of 
        data_dir_list and the length of the file_pattern_list should be same or 
        one of them is 1.

    Args:
        data_dir_list (list): directories where data stored
        file_pattern_list (list): finding files according to the filepatterns
        file_read_fn (callable object): function that decode the data file
        recursive (bool): whether find files recursively
    TODO: substitude .csv file
    """

    def __init__(self, 
                 data_dir_list, 
                 file_pattern_list=["*.wav"], 
                 dataset_parse_fn=get_file_list,
                 file_read_fn=audio_read,
                 transform_fn=None,
                 recursive=False,
                 min_size=True,
                 package=False,
                 *args,
                 **kwargs):
        self.data_dir_list = data_dir_list
        self.file_pattern_list = file_pattern_list
        self.dataset_parse_fn = dataset_parse_fn
        self.file_read_fn = file_read_fn
        self.transform_fn = transform_fn
        self.recursive = recursive
        self.min_size = min_size
        self.package = package
        file_pattern_num = len(file_pattern_list)
        data_dir_num = len(data_dir_list)
        if file_pattern_num == data_dir_num:
            self._members = file_pattern_num
        elif file_pattern_num == 1:
            self._members = data_dir_num
            self.file_pattern_list *= self._members
        elif data_dir_num == 1:
            self._members = file_pattern_num
            self.data_dir_list *= self._members
        else:
            raise ValueError
        assert callable(file_read_fn), "file_read_fn must be callable"
        self.size = None
        self.file_list = self._parse_dataset()

    def __getitem__(self, index):
        filenames = self.file_list[index]
        data_list = []
        for file in filenames:
            data = self.file_read_fn(file)
            data_list.append(data)
        sample = {"filename": filenames, "data": data_list} 
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample 

    def __len__(self):
        return self.size
    
    def _parse_dataset(self):
        file_list = []
        for data_dir, pattern in zip(self.data_dir_list, self.file_pattern_list):
            tmp_list = self.dataset_parse_fn(data_dir, pattern, self.recursive)
            file_num = len(tmp_list)
            if self.size is None:
                self.size = file_num 
            elif self.min_size:
                self.size = min(file_num, self.size)
            else:
                assert self.size == file_num
            file_list.append(tmp_list)
        file_list = list(zip(*file_list))
        return file_list
    

class Dataset(object):
    r"""Base object for fitting to a Dataset of data, such as a dataset.
    Every `Dataset` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`. The method `__getitem__` should return a complete batch.

    Args:
        data_dir (string): path to data
        file_type (string): extension name 
        file_pattern (string)
    """
    def __init__(self, data_dir, 
                 file_pattern=None, 
                 file_type=".wav", 
                 subset=None, 
                 package=False):
        self.data_dir = data_dir
        self.file_type = file_type
        self.file_pattern = file_pattern
        self.package = package
        if self.file_pattern is None:
            self.file_pattern = [file_type]
        self.file_list = self.get_file_list()
        if subset is not None:
            self.file_list = self.file_list[0:subset]
        self.size = len(self.file_list)

    def __getitem__(self, index):
        if self.package:
            filename = self.file_list[0][index] 
            return (filename, self._file_decode(filename))
        else:
            filenames = self.file_list[index] 
            paired_data = []
            for file in filenames:
                paired_data.append(self._file_decode(file))
            # if len(filenames) > 1:
            #     paired_data = []
            #     for file in filenames:
            #         paired_data.append(self._file_decode(file))
            # else:
            #     filenames = filenames[0]
            #     paired_data = self._file_decode(filenames)
            return {"filename": filenames, "data": paired_data}

    def __len__(self):
        return self.size

    def __iter__(self):
        """Create a generator that iterate over the Dataset."""
        for item in (self[i] for i in range(len(self))):
            yield item

    @staticmethod
    def _file_decode(file_name):
        _, ext = os.path.splitext(file_name)
        if ext in [".wav", ".WAV"]:
            data = audio_read(file_name)
        elif ext in [".pkl", ".dat"]:
            try:
                data = pickle.load(file_name)
            except Exception:
                print("file type is not supported")
        return data

    def get_file_list(self):
        members = len(self.file_pattern)
        file_list = [[] for i in range(members)]
        for root, _, files in os.walk(self.data_dir):
            # print(root, files)
            # files = fnmatch.filter(files, self.file_type)
            if self.file_type is not None and files is not None:
                files = [i for i in files if self.file_type in i]
            if self.file_pattern is not None and files is not None:
                for m, p in enumerate(self.file_pattern):
                    file_list[m].extend(
                        [os.path.join(root, i) for i in files if p in i])

        file_num = len(file_list[0]) 
        assert file_num > 0, "No file found"
        for l in file_list[1:]:
            assert len(l) == file_num

        for l in file_list:
            l.sort()
        file_list = [[file_list[m][n] for m in range(members)]
                     for n in range(file_num)] 
        return file_list
