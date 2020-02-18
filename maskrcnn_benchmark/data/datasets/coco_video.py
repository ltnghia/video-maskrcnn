import os
import json
import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F
from maskrcnn_benchmark.data.datasets.coco import COCODataset


class COCOVideoDataset(COCODataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations=False, transforms=None
    ):
        super(COCOVideoDataset, self).__init__(ann_file=ann_file,
                                               root=root,
                                               remove_images_without_annotations=remove_images_without_annotations,
                                               transforms=transforms)
        self.img_map_to_id = {}
        for key in self.id_to_img_map.keys():
            self.img_map_to_id[self.id_to_img_map[key]] = key

        self.video_ids = []
        self.video_data = {}
        for id in self.ids:
            video_id = self.coco.imgs[id]['video_id']
            self.video_ids.append(video_id)
            if video_id not in self.video_data.keys():
                self.video_data[video_id] = []
            self.video_data[video_id].append(self.img_map_to_id[id])
        self.video_ids = np.unique(self.video_ids)

    def __getitem__(self, index):
        img, target, index = super(COCOVideoDataset, self).__getitem__(index)

        if self._transforms is None:
            img = F.to_tensor(image)

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        return img, target, {'id': index, 'filename': path, 'ann_ids': ann_ids}

    def get_img_info(self, index):
        return super(COCOVideoDataset, self).get_img_info(index)
