import os
import torch
import numpy as np

import cv2
from maskrcnn_benchmark.utils.visualization import cv2_util
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.data.evaluation.coco import PanopticMetrics
from maskrcnn_benchmark.data.evaluation.coco import do_coco_evaluation
from maskrcnn_benchmark.data.evaluation.coco import do_coco_semantic_evaluation


def do_coco_panoptic_evaluation(
        dataset,
        predictions_detection,
        predictions_detection_retinamask,
        predictions_semantic,
        box_only,
        output_folder,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
):
    coco_detection_results = do_coco_evaluation(
        dataset.detection_dataset,
        predictions_detection,
        box_only,
        output_folder,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
    )

    coco_detection_retinamask_results = do_coco_evaluation(
        dataset.detection_dataset,
        predictions_detection_retinamask,
        box_only,
        output_folder,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
    )

    coco_semantic_results = do_coco_semantic_evaluation(
        dataset.semantic_dataset,
        predictions_semantic,
        output_folder,
    )

    pano_cls_inds = []
    panos = []
    ssegs = []

    semantic_scores = []
    for pred in predictions_semantic:
        semantic_scores.append(pred[2])

    coco = dataset.semantic_dataset.coco

    # Generate mask logits
    masker = Masker(threshold=0.5, padding=1)
    for image_id, prediction in tqdm(enumerate(predictions_detection)):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
            prediction.add_field('mask', masks)

        # Display masks
        img_id = dataset.semantic_dataset.ids[image_id]
        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(dataset.semantic_dataset.root, path), cv2.IMREAD_COLOR)
        top_predictions = select_top_predictions(prediction)
        # result = img.copy()
        # result = overlay_boxes(result, top_predictions)
        # result = overlay_mask(result, top_predictions)
        # cv2. imwrite(os.path.join(output_folder, path), result)
        # cv2.imshow("COCO detections", result)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        boxes = top_predictions._split_into_xyxy()
        labels = top_predictions.get_field('labels')
        labels = [lab - 53 for lab in labels]
        masks = top_predictions.get_field('mask')
        boxes_sorted = [[] for i in range(4)]
        labels_sorted = []
        masks_sorted = []
        # Remove stuff predictions
        for i in range(len(labels)):
            if (labels[i] >= 0):
                for j in range(len(boxes)):
                    boxes_sorted[j].append(boxes[j][i])
                labels_sorted.append(labels[i])
                masks_sorted.append(masks[i])

        img_shape = (1, image_height, image_width)
        mask_logits = np.full(img_shape, 0, dtype=np.uint8)

        minimum_pixels = 50
        idx_removed = 0
        new_labels = []

        zeros = np.zeros(img_shape, dtype=np.uint8)
        seg_logits = np.where((coco_semantic_results[image_id] > 52), zeros, coco_semantic_results[image_id])
        seg_inst_logits = np.where((coco_semantic_results[image_id] <= 52), zeros, coco_semantic_results[image_id] - 52)

        label = 53
        for idx, mask in enumerate(masks_sorted):
            label = 53 + idx - idx_removed
            calc = np.full(img_shape, label, dtype=np.uint8)
            xmin = int(boxes_sorted[0][idx])
            ymin = int(boxes_sorted[1][idx])
            xmax = int(boxes_sorted[2][idx].round() + 1)
            ymax = int(boxes_sorted[3][idx].round() + 1)

            # Select pixels of seg_inst inside te bbox and add those with the good label
            # TODO manage the inst pixels not in the instance branch
            mask_seg_inst = np.zeros(img_shape)
            mask_seg_inst[0, ymin: ymax, xmin: xmax] = seg_inst_logits[0, ymin: ymax, xmin: xmax]
            cond = (mask_seg_inst == labels_sorted[idx].item()) * (mask.numpy() == 0)
            mask = np.where(cond, mask_seg_inst, mask)
            mask = np.where((mask != 0), calc, mask)

            new_mask = np.where((mask_logits != 0), mask_logits, mask)
            cond = (mask != 0) * (mask - mask_logits == label)
            remaining_pixels = cond.sum().item()
            if remaining_pixels >= minimum_pixels:
                mask_logits = new_mask
                new_labels.append(labels_sorted[idx].item())
            else:
                idx_removed += 1

        labels_seg_inst = np.delete(np.unique(seg_inst_logits), 0)
        for lab in labels_seg_inst:
            if lab.item() not in new_labels:
                calc = np.full(img_shape, label, dtype=np.uint8)
                cond = (seg_inst_logits == lab.item()) * semantic_scores[image_id] >= 10
                mask_logits = np.where(cond, calc, mask_logits)
                new_labels.append(lab.item())
                label += 1

        # Display mask logits
        # print("MASK", np.unique(mask_logits), idx_removed)
        # cv2.imshow("Mask logits", mask_logits[0].astype(np.uint8))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # Create panoptic logits
        empty = torch.full(img_shape, 255, dtype=torch.uint8)
        pan = np.where((mask_logits == 0), empty, mask_logits)
        pan = np.where((pan == 255), seg_logits, pan)

        pano_cls_inds.append(new_labels)
        panos.append(np.squeeze(pan.astype(np.uint8)))
        ssegs.append(np.squeeze(coco_semantic_results[image_id]))

    print("Start panoptic evaluation")
    panoptic_metrics = PanopticMetrics()
    coco_panoptic_results = panoptic_metrics.evaluate_panoptic(
        panoptic_metrics.get_unified_pan_result(ssegs, panos, pano_cls_inds), output_folder,
        dataset.semantic_dataset.root)

    return coco_detection_results, coco_detection_retinamask_results, coco_semantic_results, coco_panoptic_results


def select_top_predictions(predictions):
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
    confidence_threshold = 0.7
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1], dtype=torch.float32)
    colors = labels[:, None].to(dtype=torch.float32) * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image


def overlay_mask(image, predictions):
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
    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite