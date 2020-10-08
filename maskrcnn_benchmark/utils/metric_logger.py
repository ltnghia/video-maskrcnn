# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size, all_gather, synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str

import torch
import time, os, io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


class TensorboardXLogger(MetricLogger):
    def __init__(self, log_dir='logs', delimiter='\t', window_size=20):
        super(TensorboardXLogger, self).__init__(delimiter)
        self.writer = self._get_tensorboard_writer(log_dir, window_size)

    @staticmethod
    def _get_tensorboard_writer(log_dir, window_size=20):
        if is_main_process() and log_dir is not None:
            timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
            return TensorboardXWriter(os.path.join(log_dir, timestamp), window_size)
        else:
            return None

    def update(self, iteration, **kwargs):
        super(TensorboardXLogger, self).update(**kwargs)
        if self.writer is not None:
            self.writer.update(iteration, **kwargs)


class TensorboardXWriter:
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): The directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        try:
            from torch.utils.tensorboard import SummaryWriter
        except:
            try:
                from tensorboardX import SummaryWriter
            except:
                return None

        self._writer = SummaryWriter(log_dir, **kwargs)

    def update(self, iteration, **kwargs):
        if self._writer is not None:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                assert isinstance(v, (float, int))
                self._writer.add_scalar(k, v, iteration)

    def update_image(self, iteration, image, preds, targets):
        image = image.cpu().numpy()
        boxes = preds.bbox.cpu().numpy()
        boxes_gt = targets.bbox.cpu().numpy()
        cats = preds.get_field('labels').cpu().numpy()
        cats_gt = targets.get_field('labels').cpu().numpy()

        if self._writer:
            for cat in np.unique(np.append(cats, cats_gt)):
                if cat == 0:
                    continue
                fig, ax = plt.figure(), plt.gca()
                ax.imshow(image.transpose(1, 2, 0))

                for i in range(len(cats)):
                    if cats[i] == cat:
                        x1, y1, x2, y2 = boxes[i]
                        ax.add_patch(
                            patches.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                edgecolor='r',
                                linewidth=1,
                                fill=False
                            )
                        )

                for i in range(len(cats_gt)):
                    if cats_gt[i] == cat:
                        x1, y1, x2, y2 = boxes_gt[i]
                        ax.add_patch(
                            patches.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                edgecolor='g',
                                linewidth=1,
                                fill=False
                            )
                        )

                plt.axis('scaled')
                plt.tight_layout()

                self._writer.add_figure('train/image/{}'.format(cat), fig, iteration)

    def __del__(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()





