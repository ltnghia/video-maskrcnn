# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch

import requests
from io import BytesIO
from PIL import Image
import numpy as np

import os
import matplotlib.pyplot as plt

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.structures.keypoints import PersonKeypoints
from maskrcnn_benchmark.data.categories import ro_categories
from maskrcnn_benchmark.utils.visualization.create_palette import create_palette
from maskrcnn_benchmark.utils.visualization.cv2_util import findContours
from maskrcnn_benchmark.utils.miscellaneous import mkdir


class Visualizer(object):

    def __init__(
        self,
        cfg,
        confidence_threshold=-1,
        show_mask_heatmaps=False,
        show_mask_montage=True,
        masks_per_dim=2,
        categories=None
    ):
        self.categories = categories
        self.cfg = cfg.clone()

        mask_threshold = -1 if show_mask_montage else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        if confidence_threshold >= 0:
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_VISUALIZATION

        self.show_heatmap = show_mask_heatmaps
        self.show_mask_montage = show_mask_montage
        self.masks_per_dim = masks_per_dim

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.my_palette = []
        colors = create_palette()
        self.my_palette = []
        for color in colors:
            self.my_palette.append(color['rgb'])

    def visualize_predictions(self, predictions, dataset, output_folder):
        mkdir(output_folder)
        mask_montage_folder = os.path.join(output_folder, '..', 'mask_montage')
        mkdir(mask_montage_folder)

        root = dataset.root
        for image_id, prediction in enumerate(predictions):
            original_id = dataset.id_to_img_map[image_id]

            # TODO replace with get_img_info?
            image_width = dataset.coco.imgs[original_id]["width"]
            image_height = dataset.coco.imgs[original_id]["height"]
            file_name = dataset.coco.imgs[original_id]["file_name"]

            pil_image = Image.open(os.path.join(root, file_name)).convert("RGB")
            # convert to BGR format
            image = np.array(pil_image)[:, :, [2, 1, 0]]

            if len(prediction) == 0:
                dir_img_output = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.jpg')
                dir_folder_output = os.path.dirname(dir_img_output)
                mkdir(dir_folder_output)
                cv2.imwrite(dir_img_output, image)
                continue

            prediction = prediction.resize((image_width, image_height))

            if prediction.has_field("mask"):
                # if we have masks, paste the masks in the right position
                # in the image, as defined by the bounding boxes
                masks = prediction.get_field("mask")
                masks = self.masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

            top_predictions = self.select_top_predictions(prediction)

            result = image.copy()
            if self.show_heatmap and self.cfg.MODEL.MASK_ON:
                result = self.overlay_heatmaps(top_predictions, result)
            result = self.overlay_boxes(result, top_predictions)
            if self.cfg.MODEL.MASK_ON:
                result = self.overlay_masks(result, top_predictions)
            result = self.overlay_class_names(result, top_predictions)

            dir_img_output = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.jpg')
            dir_folder_output = os.path.dirname(dir_img_output)
            mkdir(dir_folder_output)
            cv2.imwrite(dir_img_output, result)

            if self.show_mask_montage and self.cfg.MODEL.MASK_ON:
                result = self.create_mask_montage(top_predictions)
                dir_img_output = os.path.join(mask_montage_folder, os.path.splitext(file_name)[0] + '.jpg')
                dir_folder_output = os.path.dirname(dir_img_output)
                mkdir(dir_folder_output)
                cv2.imwrite(dir_img_output, result)

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        # print(len(self.my_palette))
        # print(len(labels))
        # print(labels)
        # print(labels[:, None])
        if self.my_palette is not None:
            colors = []
            for l in labels:
                colors.append(self.my_palette[l-1])
            colors = np.asarray(colors).astype("uint8")
            # colors = colors.numpy().astype("uint8")
            # pprint(colors)
            # exit()
        else:
            colors = labels[:, None] * self.palette
            colors = (colors % 255).numpy().astype("uint8")
            # pprint(colors)
            # exit()
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        if predictions.has_field("second_labels"):
            labels = predictions.get_field("second_labels")
        else:
            labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2)

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.categories[i] for i in labels]
        colors = self.compute_colors_for_labels(predictions.get_field("labels")).tolist()
        boxes = predictions.bbox
        alpha = 0.5

        template = "{}: {:.2f}"
        for box, score, label, color in zip(boxes, scores, labels, colors):
            box = box.to(torch.int64)
            top_left = box[:2].tolist()
            bottom_right = box[2:].tolist()
            bottom_right[1] = top_left[1] - 20

            image2 = cv2.rectangle(image.copy(), tuple(top_left), tuple(bottom_right), tuple(color), -1)

            image = cv2.addWeighted(image2, alpha, image, 1-alpha, 0)

            x = (top_left[0] + bottom_right[0]) / 2 - 50
            y = (top_left[1]) - 4
            s = template.format(label, score)
            cv2.putText(image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        return image

    def overlay_masks(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def overlay_heatmaps(self, predictions, image=None, alpha=0.5):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        heatmap = 0
        for mask in masks:
            thresh = mask[0, :, :, None]
            heatmap += thresh
        if np.sum(heatmap) > 0:
            heatmap = ((heatmap - np.min(heatmap))/np.ptp(heatmap)*255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        if image is not None:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
        return heatmap

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
        """Visualizes keypoints (adapted from vis_one_image).
        kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
        """
        dataset_keypoints = PersonKeypoints.NAMES
        kp_lines = PersonKeypoints.CONNECTIONS

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # Draw mid shoulder / mid hip first for better visualization.
        mid_shoulder = (
                               kps[:2, dataset_keypoints.index('right_shoulder')] +
                               kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
        sc_mid_shoulder = np.minimum(
            kps[2, dataset_keypoints.index('right_shoulder')],
            kps[2, dataset_keypoints.index('left_shoulder')])
        mid_hip = (
                          kps[:2, dataset_keypoints.index('right_hip')] +
                          kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
        sc_mid_hip = np.minimum(
            kps[2, dataset_keypoints.index('right_hip')],
            kps[2, dataset_keypoints.index('left_hip')])
        nose_idx = dataset_keypoints.index('nose')
        if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
                color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
        if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder), tuple(mid_hip),
                color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

        # Draw the keypoints.
        for l in range(len(kp_lines)):
            i1 = kp_lines[l][0]
            i2 = kp_lines[l][1]
            p1 = kps[0, i1], kps[1, i1]
            p2 = kps[0, i2], kps[1, i2]
            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                cv2.line(
                    kp_mask, p1, p2,
                    color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            if kps[2, i1] > kp_thresh:
                cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if kps[2, i2] > kp_thresh:
                cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

    def create_mask_montage(self, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        if masks.shape[0] > 0:
            masks = torch.nn.functional.interpolate(masks.float(), scale_factor=1 / masks_per_dim).byte()
            height, width = masks.shape[-2:]
            max_masks = masks_per_dim ** 2
            masks = masks[:max_masks]
            # handle case where we have less detections than max_masks
            if len(masks) < max_masks:
                masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
                masks_padded[: len(masks)] = masks
                masks = masks_padded
            masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
            result = torch.zeros(
                (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
            )
            for y in range(masks_per_dim):
                start_y = y * height
                end_y = (y + 1) * height
                for x in range(masks_per_dim):
                    start_x = x * width
                    end_x = (x + 1) * width
                    result[start_y:end_y, start_x:end_x] = masks[y, x]
        else:
            height, width = masks.shape[-2:]
            result = torch.zeros((masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8)
        heatmap = cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)
        return heatmap

    def visualize_attentions(self, attentions, dataset, output_folder, alpha=0.5):
        mkdir(output_folder)
        root = dataset.root
        for image_id, attention in enumerate(attentions):
            original_id = dataset.id_to_img_map[image_id]
            file_name = dataset.coco.imgs[original_id]["file_name"]
            pil_image = Image.open(os.path.join(root, file_name)).convert("RGB")
            # convert to BGR format
            image = np.array(pil_image)[:, :, [2, 1, 0]]

            heatmap = attention.numpy()
            scale = 1
            heatmap = (scale*255*heatmap).astype(np.uint8)
            # if np.sum(heatmap) > 0:
            #     heatmap = ((heatmap - np.min(heatmap)) / np.ptp(heatmap) * 255).astype(np.uint8)
            # else:
            #     heatmap = (255 * heatmap).astype(np.uint8)
            # heatmap[heatmap > 10] = 255
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            image = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)

            dir_img_output = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.jpg')
            dir_folder_output = os.path.dirname(dir_img_output)
            mkdir(dir_folder_output)
            cv2.imwrite(dir_img_output, image)

            dir_img_output = os.path.join(output_folder, os.path.splitext(file_name)[0] + '_attention.jpg')
            cv2.imwrite(dir_img_output, heatmap)

    def load(url):
        """
        Given an url of an image, downloads the image and
        returns a PIL image
        """
        response = requests.get(url)
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        # convert to BGR format
        image = np.array(pil_image)[:, :, [2, 1, 0]]
        return image

    def imshow(img):
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.axis("off")


if __name__ == "__main__":
    cfg.merge_from_file('/mnt/sda/ltnghia/ITS_project/Code/maskrcnn-benchmark/configs/caffe2/ltnghia/e2e_faster_rcnn_R_50_FPN_1x_caffe2_ltnghia_ro_cs.yaml')
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    dataset = data_loader.dataset
    predictions = torch.load('/mnt/sda/ltnghia/ITS_project/Code/maskrcnn-benchmark/demo/save/RO/CityScapes/FPN50-RO_CityScapes-Fine-16-0.02-15k/inference/ro_cityscapes_gtFine_train/predictions.pth')
    visualizer = Visualizer(categories=ro_categories.CATEGORIES, cfg=cfg)
    visualizer.visualize_predictions(predictions, dataset, './temp')