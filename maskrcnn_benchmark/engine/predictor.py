# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torchvision import transforms as T

import numpy as np

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.visualization.visualizer import Visualizer


class Predictor(object):

    def __init__(
        self,
        cfg,
        show_mask_heatmaps=False,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.min_image_size = min_image_size

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        self.cpu_device = torch.device("cpu")

        self.visualizer = Visualizer(categories=ro_categories.CATEGORIES,
                                cfg=cfg,
                                confidence_threshold=0.8,
                                show_mask_heatmaps=show_mask_heatmaps)

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, model, image):
        """
        Arguments:
            model: faster r-cnn or mask r-cnn
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(model, image)
        top_predictions = self.visualizer.select_top_predictions(predictions)

        result = image.copy()
        if self.visualizer.show_heatmap:
            return self.visualizer.create_mask_montage(result, top_predictions)
        result = self.visualizer.overlay_boxes(result, top_predictions)
        if self.visualizer.cfg.MODEL.MASK_ON:
            result = self.visualizer.overlay_masks(result, top_predictions)
        result = self.visualizer.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, model, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            masks = self.masker(masks, prediction)
            prediction.add_field("mask", masks)
        return prediction


if __name__ == "__main__":
    cfg.merge_from_file('/mnt/sda/ltnghia/ITS_project/Code/maskrcnn-benchmark/configs/caffe2/ltnghia/e2e_faster_rcnn_R_50_FPN_1x_caffe2_ltnghia_ro_cs.yaml')
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    dataset = data_loader.dataset
    predictions = torch.load('/mnt/sda/ltnghia/ITS_project/Code/maskrcnn-benchmark/demo/save/RO/CityScapes/FPN50-RO_CityScapes-Fine-16-0.02-15k/inference/ro_cityscapes_gtFine_train/predictions.pth')
    Predictor = Predictor(cfg=cfg, min_image_size=800)
    Predictor.visualize_results(predictions, dataset, './temp')