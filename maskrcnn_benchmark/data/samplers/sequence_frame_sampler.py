import math
import torch
from torch.utils.data.sampler import Sampler
from maskrcnn_benchmark.utils.comm import get_world_size, get_rank
from maskrcnn_benchmark.utils.ulti import get_batch_size_per_gpu


class SequenceFrameSampler(Sampler):
    def __init__(self, dataset, shuffle=True, distributed=False, num_replicas=None, rank=None, args=None, cfg=None):
        self.dataset = dataset
        # this is a list of list of names.
        # the first level correspond to a video, and the second
        # one to the names of the frames in the video
        self.video_data = dataset.video_data
        self.window_size = 1
        self.batch_size_per_gpu = 1
        self.epoch = 0
        self.shuffle = shuffle
        self.distributed = distributed
        self.is_train = True

        if args is not None:
            if hasattr(args, 'window_size'):
                self.window_size = args.window_size
            if hasattr(args, 'is_train'):
                self.is_train = args.is_train
        if cfg is not None:
            self.batch_size_per_gpu = get_batch_size_per_gpu(cfg.SOLVER.IMS_PER_BATCH if self.is_train
                                                             else cfg.TEST.IMS_PER_BATCH)

        self.indices = []
        for video_id in sorted(self.video_data):
            frame_list = sorted(self.video_data[video_id])
            count = 0
            frame_ids = []
            for frame_id in sorted(frame_list):
                frame_ids.append(frame_id)
                count += 1
                if count == self.window_size:
                    self.indices.append(frame_ids)
                    frame_ids = []
                    count = 0
            if not args.is_train and count > 0:
                for i in range(self.window_size):
                    frame_ids.append(frame_id)
                    count += 1
                    if count == self.window_size:
                        self.indices.append(frame_ids)
                        frame_ids = []
                        count = 0
                        break
        self.num_samples = len(self.indices)
        self.total_size = self.num_samples
        # print(self.__len__())

        if self.distributed:
            if num_replicas is None:
                num_replicas = get_world_size()
            if rank is None:
                rank = get_rank()
            self.num_replicas = num_replicas
            self.rank = rank
            self.num_samples = int(math.ceil(self.num_samples * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # if self.distributed:
        #     raise ValueError('sequence frame sampler does not support multiple GPUs')

        batch_index = []

        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            batch_index = torch.randperm(self.total_size, generator=g).tolist()
            # batch_index = torch.randperm(len(self.indices)).tolist()
        else:
            batch_index = torch.arange(self.total_size).tolist()

        # print('num_samples', self.num_samples, 'rank', get_rank(), 'num_replicas', self.num_replicas, 'total_size', self.total_size)

        if self.distributed:
            indices = batch_index
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset: offset + self.num_samples]
            assert len(indices) == self.num_samples
            # print('dataset', self.total_size, 'begin', offset, 'end', offset + self.num_samples, 'len', len(indices))
            # print(indices[:20])
            batch_index = indices

        frame_indices = []
        if self.batch_size_per_gpu == 1:
            for i in batch_index:
                frame_indices.extend(self.indices[i])
            if len(frame_indices) != self.__len__():
                raise ValueError('Window Size errors')
        else:
            for i in range(math.floor(len(batch_index) / self.batch_size_per_gpu)):
                if i + self.batch_size_per_gpu < len(batch_index):
                    for j in range(self.window_size):
                        for k in range(self.batch_size_per_gpu):
                            frame_indices.append(self.indices[batch_index[i * self.batch_size_per_gpu + k]][j])

        return iter(frame_indices)

    def __len__(self):
        return self.total_size * self.window_size

    def set_epoch(self, epoch):
        self.epoch = epoch

