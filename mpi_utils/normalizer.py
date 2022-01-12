import threading
import numpy as np
from mpi4py import MPI

class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
    
    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)

    # extend to new observation
    def change_size(self, old_size = None, new_size = None):
        if new_size == None: # default to extend to currient setting
            new_size = self.size
        if old_size == None:
            old_size = self.size
            self.size = new_size
        assert (old_size <= new_size) and (new_size <= old_size*2), \
            f'Old size:{old_size}, New size:{new_size}'
        if old_size == new_size:
            return
        extend_length = new_size - old_size
        # local information
        self.local_sum = np.append(self.local_sum, self.local_sum[-extend_length:])
        self.local_sumsq = np.append(self.local_sumsq, self.local_sumsq[-extend_length:])
        # total sum sumsq and sum count
        self.total_sum = np.append(self.total_sum, self.total_sum[-extend_length:])
        self.total_sumsq = np.append(self.total_sumsq, self.total_sumsq[-extend_length:])
        # get the mean and std
        self.mean = np.append(self.mean, self.mean[-extend_length:])
        self.std = np.append(self.std, self.std[-extend_length:])

    def load(self, config):
        self.eps = config['eps']
        self.default_clip_range = config['default_clip_range']
        self.local_sum = config['local_sum']
        self.local_sumsq = config['local_sumsq']
        self.local_count = config['local_count']
        self.total_sum = config['total_sum']
        self.total_sumsq = config['total_sumsq']
        self.total_count = config['total_count']
        self.mean = config['mean']
        self.std = config['std']
        self.change_size(old_size = config['size'])

    def state_dict(self):
        config = {
            'size': self.size, 
            'eps': self.eps, 
            'default_clip_range': self.default_clip_range, 
            'local_sum': self.local_sum, 
            'local_sumsq': self.local_sumsq, 
            'local_count': self.local_count, 
            'total_sum': self.total_sum, 
            'total_sumsq': self.total_sumsq, 
            'total_count': self.total_count, 
            'mean': self.mean, 
            'std': self.std
        }
        return config