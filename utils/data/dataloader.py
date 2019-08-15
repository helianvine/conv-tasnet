import random
import numpy as np
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import Event
from . sampler import BatchSampler


class DataLoader(object):
    
    def __init__(self,
                 dataset,
                 batch_size,
                 drop_last=False,
                 shuffle=False,
                 worker_num=1,
                 q_num=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.worker_num = worker_num
        self.q_num = q_num
        self.sampler = BatchSampler(dataset, batch_size, shuffle)

    def __iter__(self):
        loader_iter = LoaderIter(self)
        return iter(loader_iter)

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def on_epoch_end(self, **kwargs):
        """Method called at the end of every epoch.
        """
        self.dataset.on_epoch_end(**kwargs)


class LoaderIter(object):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.drop_last = loader.drop_last
        self.shuffle = loader.shuffle
        self.worker_num = loader.worker_num
        self.sample_iter = iter(loader.sampler)
        self.file_num = len(self.dataset)
        self.batches_outstanding = 0
        self.send_idx = 0
        self.worker_queue_idx = 0
        self.shutdown = False
        if self.dataset.package:
            self.buffer_list = []
        self.workers_done_event = Event()
        # signal.signal(signal.SIGINT, self._shutdown_workers)
        # signal.signal(signal.SIGTERM, self._shutdown_workers)

        self.index_queues = []
        self.workers = []
        if self.worker_num > 0:
            self.dataqueue = Queue()
            self.reorder_dict = {}
            self.rcvd_idx = 0
            index_queue = Queue()
            index_queue.cancel_join_thread()
            w = Process(
                target=self._worker_loop,
                args=(self._collate_fn,
                      self.dataset,
                      self.dataqueue,
                      index_queue,
                      self.workers_done_event))
            w.daemon = True
            w.start()
            self.index_queues.append(index_queue)
            self.workers.append(w)
            for _ in range(2 * self.worker_num):
                self._put_indices()

    @staticmethod
    def _worker_loop(worker_fn, dataset, dataqueue, index_queue, done_event):
        try:
            while True:
                try:
                    r = index_queue.get()
                except Exception:
                    print("Exception happend when getting index")
                    continue
                if r is None:
                    assert done_event.is_set()
                    break
                elif done_event.is_set():
                    continue

                idx, indices = r
                batch_samples = worker_fn(dataset, indices)
                dataqueue.put((idx, batch_samples))
        except KeyboardInterrupt:
            pass
        if done_event.is_set():
            dataqueue.cancel_join_thread()
            dataqueue.close()

    @staticmethod
    def _collate_fn(dataset, indices):
        batch = []
        for i in indices:
            sample = dataset[i]
            batch.append(sample) 
        # TODO: transform_fn
        return batch
        # batch_data = [[batch_data[n][m] for n in range(len(batch_data))]
        #               for m in range(len(batch_data[0]))]
        # batch_files = [[batch_files[n][m] for n in range(len(batch_files))]
        #                for m in range(len(batch_files[0]))]

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.worker_num
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.worker_num
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self):
        self.rcvd_idx += 1
        self._put_indices()
    
    def _get_batch(self):
        return self.dataqueue.get()

    def __next__(self):
        if self.worker_num == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self._collate_fn(self.dataset, indices)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            self._process_next_batch()
            return batch

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            self._process_next_batch()
            return batch

    def __iter__(self):
        if self.dataset.package:
            return self._iter_package()
        else:
            return self

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            try:
                self.workers_done_event.set()
                # Exit workers now.
                for q in self.index_queues:
                    q.put(None)
                    # Indicate that no more data will be put on this queue by the
                    # current process.
                    q.close()
                for w in self.workers:
                    w.join()
            except Exception:
                print("Exception happend when exiting child process")
                for w in self.workers:
                    w.terminate()

    def __del__(self):
        if self.worker_num > 0:
            self._shutdown_workers()

    def _load_from_queue(self):

        if self.file_idx in self.buffer_dict:
            self.buffer_list.extend(self.buffer_dict[self.file_idx])
            self.file_idx += 1
            return self.buffer_list
        else:
            while True:
                # assert not self.dataqueue.empty()
                idx, sample = self.dataqueue.get()
                if idx != self.file_idx:
                    self.buffer_dict[idx] = sample
                    continue
                self.file_idx += 1
                self.buffer_list.extend(sample)
                return self.buffer_list

    def _iter_package(self):
        while (self.file_idx + self.q_num) < self.file_num:

            for _ in range(self.q_num):
                # import ipdb; ipdb.set_trace()
                self.buffer_list = self._load_from_queue()
                # ipdb.set_trace()

            batch_in_buffer = np.floor(len(self.buffer_list) / self.batch_size)
            if self.shuffle:
                random.shuffle(self.buffer_list)

            for _ in range(batch_in_buffer):
                self.batch = self.buffer_list[0: self.batch_size]
                del self.buffer_list[0: self.batch_size]
                self.total_sample += self.batch_size
                yield self.batch

            # import ipdb; ipdb.set_trace()

        for _ in range(self.file_num - self.file_idx):
            self.buffer_list = self.load_from_queue()

        batch_in_buffer = np.floor(len(self.buffer_list) / self.batch_size)
        if self.shuffle:
            random.shuffle(self.buffer_list)

        # import ipdb; ipdb.set_trace()

        assert self.filequeue.empty(), 'File_queue is not empty'
        assert self.dataqueue.empty(), 'Data_queue is not empty'
        assert self.file_idx == self.file_num, 'file_idx does not equal file_num'
        self._shut_down_process()

        for _ in range(batch_in_buffer):
            self.batch = self.buffer_list[0: self.batch_size]
            del self.buffer_list[0: self.batch_size]
            self.total_sample += self.batch_size
            yield self.batch
        # import ipdb; ipdb.set_trace()

        if (not self.drop_last) and (len(self.buffer_list) > 0):
            self.total_sample += len(self.buffer_list)
            import ipdb
            ipdb.set_trace()
            yield self.buffer_list
