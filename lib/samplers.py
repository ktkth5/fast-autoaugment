
import math
import os
import random
from collections import defaultdict

from torch.utils.data import Sampler
import torch.distributed as dist



class DistributedStratifiedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, labels, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.size = len(labels)
        self.num_samples = int(math.ceil(self.size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.idx_by_lb = defaultdict(list)
        for idx, lb in enumerate(labels):
            self.idx_by_lb[lb].append(idx)
        self.size = len(labels)

        print('----- distributed stratified sampler -----')

    def __iter__(self):
        # deterministically shuffle based on epoch

        songs_list = []
        artists_list = []
        for lb, v in self.idx_by_lb.items():
            for idx in v:
                songs_list.append(idx)
                artists_list.append(lb)

        random_state = random.getstate()
        random.seed(self.epoch)
        indices = spotifyShuffle(songs_list, artists_list)
        random.setstate(random_state)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class StratifiedSampler(Sampler):
    def __init__(self, labels):
        self.idx_by_lb = defaultdict(list)
        for idx, lb in enumerate(labels):
            self.idx_by_lb[lb].append(idx)

        self.size = len(labels)

    def __len__(self):
        return self.size

    def __iter__(self):
        songs_list = []
        artists_list = []
        for lb, v in self.idx_by_lb.items():
            for idx in v:
                songs_list.append(idx)
                artists_list.append(lb)

        shuffled = spotifyShuffle(songs_list, artists_list)
        return iter(shuffled)


def fisherYatesShuffle(arr):
    """
    https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    for i from n−1 downto 1 do
     j ← random integer such that 0 ≤ j ≤ i
     exchange a[j] and a[i]
    """
    for i in range(len(arr)-1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def spotifyShuffle(songs_list, artists_list):
    artist2songs = defaultdict(list)
    for artist, song in zip(artists_list, songs_list):
        artist2songs[artist].append(song)
    songList = []
    songsLocs = []
    for artist, songs in artist2songs.items():
        songs = fisherYatesShuffle(songs)
        songList += songs
        songsLocs += get_locs(len(songs))
    return [songList[idx] for idx in argsort(songsLocs)]


def argsort(seq):
    return [i for i, j in sorted(enumerate(seq), key=lambda x:x[1])]


def get_locs(n):
    percent = 1. / n
    locs = [percent * random.random()]
    last = locs[0]
    for i in range(n - 1):
        value = last + percent * random.uniform(0.8, 1.2)  # 25% : 20~30% = 1 : 0.8x~1.2x
        locs.append(value)
        last = value
    return locs