# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.semantic_mask import to_mask_list
from maskrcnn_benchmark.structures.prediction_value import to_value_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_infos = transposed_batch[2]

        if isinstance(targets[0], dict):
            if "roi_target" in targets[0].keys() and targets[0]["roi_target"] is not None:
                roi_targets = [target["roi_target"] for target in targets]
            else:
                roi_targets = None
            if "seg_target" in targets[0].keys() and targets[0]["seg_target"] is not None:
                seg_targets = [target["seg_target"] for target in targets]
                seg_targets = to_mask_list(seg_targets, self.size_divisible)
            else:
                seg_targets = None
            if "reg_target" in targets[0].keys() and targets[0]["reg_target"] is not None:
                reg_targets = [target["reg_target"] for target in targets]
                reg_targets = to_value_list(reg_targets, self.size_divisible)
            else:
                reg_targets = None
            targets = dict(roi_target=roi_targets, reg_target=reg_targets, seg_target=seg_targets)

        return images, targets, img_infos


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
