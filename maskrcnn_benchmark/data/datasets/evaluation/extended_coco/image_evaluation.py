import logging
import os
import torch
import json
import numpy as np

# from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults
# from maskrcnn_benchmark.data.datasets.evaluation.extended_coco.ExtendedCOCOeval import ExtendedCOCOeval


def do_image_evaluation(
    dataset,
    predictions,
    output_folder
):
    types = ['cls']
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Preparing results for Extended COCO format")
    extended_coco_results = {}
    if "cls" in types:
        logger.info("Preparing classification results")
        extended_coco_results["cls"] = evaluate_detection(predictions, dataset)
    if "seg" in types:
        logger.info("Preparing segmentation results")
        extended_coco_results["seg"] = evaluate_segmentation(predictions, dataset)

    logger.info('---------------------------')
    logger.info(extended_coco_results)
    logger.info('---------------------------')

    if output_folder:
        torch.save(extended_coco_results, os.path.join(output_folder, "image_results.pth"))

    logger.info(dataset.root)
    logger.info(output_folder)
    logger.info('===========================')

    return extended_coco_results


def evaluate_detection(predictions, dataset):
    n_cat = len(dataset.second_categories.keys()) + 1 # for background
    raw_confusion_matrix = np.zeros((n_cat, n_cat))
    raw_confusion_matrix[0, :].fill(np.nan)
    raw_confusion_matrix[:, 0].fill(np.nan)
    # count = 0

    # print(dataset.coco.imgToRegAnns)

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        reg_target = dataset.coco.imgToRegAnns[original_id]
        # print(image_id, original_id, reg_target)

        if reg_target[0]['label'] in dataset.json_second_category_id_to_contiguous_id.keys():
            reg_target = dataset.json_second_category_id_to_contiguous_id[reg_target[0]['label']]
        else:
            reg_target = 0

        score = prediction['cls_scores'].get_tensors().item()
        label = prediction['cls_labels'].get_tensors().item()
        # mapped_label = dataset.contiguous_second_category_id_to_json_id[label]

        # print(reg_target, label)
        if np.isnan(raw_confusion_matrix[reg_target, label]):
            # print(reg_target, label)
            raw_confusion_matrix[reg_target, label] = 0
        raw_confusion_matrix[reg_target, label] += 1
        # count += 1

    count = len(dataset.coco.imgToRegAnns.keys())
    # raw_confusion_matrix = raw_confusion_matrix[1:, 1:]
    raw_confusion_matrix = raw_confusion_matrix / count

    # print(raw_confusion_matrix)
    accuracy = 0
    for i in range(raw_confusion_matrix.shape[0]):
        if not np.isnan(raw_confusion_matrix[i, i]):
            accuracy += raw_confusion_matrix[i, i]

    acc_sum = raw_confusion_matrix.sum(axis=1)
    # print(acc_sum)
    refined_confusion_matrix = (raw_confusion_matrix.T / acc_sum).T
    # print(refined_confusion_matrix)
    refined_accuracy = 0
    for i in range(refined_confusion_matrix.shape[0]):
        if not np.isnan(refined_confusion_matrix[i, i]):
            refined_accuracy += refined_confusion_matrix[i, i]

    if np.isnan(refined_confusion_matrix[0, :]).all() and np.isnan(refined_confusion_matrix[:, 0]).all():
        refined_accuracy /= (refined_confusion_matrix.shape[0]-1)
    else:
        refined_accuracy /= refined_confusion_matrix.shape[0]

    return dict(accuracy=accuracy, refined_accuracy=refined_accuracy,
                confusion_matrix=raw_confusion_matrix, refined_confusion_matrix=refined_confusion_matrix)


def evaluate_segmentation(predictions, dataset):
    return None


