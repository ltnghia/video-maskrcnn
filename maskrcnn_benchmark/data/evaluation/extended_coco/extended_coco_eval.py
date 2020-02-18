import logging
import tempfile
import os
import torch
from tqdm import tqdm
import json

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.data.evaluation.coco.coco_eval import check_expected_results, COCOResults, evaluate_box_proposals
from maskrcnn_benchmark.data.core import ExtendedCOCOeval


PROPERTY = [
    {'id': 1,
     'label': 'labels',
     'score': 'scores',
     'category': 'categories',
     'category_id': 'category_id',
     'coco_score': 'score'},
    {'id': 2,
     'label': 'second_labels',
     'score': 'second_scores',
     'category': 'second_categories',
     'category_id': 'second_category_id',
     'coco_score': 'second_score'},
    # {'id': 3,
    #  'label': 'third_labels',
    #  'score': 'third_scores',
    #  'category': 'third_categories',
    #  'category_id': 'third_category_id',
    #  'coco_score': 'third_score'},
]


def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    overall_results = {}
    overall_coco_results = {}

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = ExtendedCOCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(predictions, dataset, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return

    logger.info("Preparing results for Extended COCO format")
    extended_coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        extended_coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        extended_coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if 'keypoints' in iou_types:
        logger.info('Preparing keypoints results')
        extended_coco_results['keypoints'] = prepare_for_coco_keypoint(predictions, dataset)

    res = None
    for k, property in enumerate(PROPERTY):
        results = ExtendedCOCOResults(*iou_types)
        logger.info("Evaluating predictions: Property {}".format(property['id']))
        for iou_type in iou_types:
            with tempfile.NamedTemporaryFile() as f:
                if len(extended_coco_results[iou_type]) > 0 and property['category_id'] in extended_coco_results[iou_type][0].keys():
                    file_path = f.name
                    if output_folder:
                        file_path = os.path.join(output_folder, iou_type + "_property_{}.json".format(property['id']))
                    res = evaluate_predictions_on_extended_coco(
                        dataset.coco, extended_coco_results[iou_type], file_path, iou_type,
                        category_type=property['category_id'])
                    results.update(res)

        logger.info('---------------------------')
        logger.info(results.__repr__())
        logger.info('---------------------------')

        check_expected_results(results, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(results, os.path.join(output_folder, "coco_results_property_{}.pth".format(property['id'])))

        overall_results[property['id']] = results
        overall_coco_results[property['id']] = extended_coco_results

    logger.info(dataset.root)
    logger.info(output_folder)
    logger.info('===========================')

    return overall_results, overall_coco_results


def evaluate_predictions_on_extended_coco(coco_gt=None, coco_results=None, json_result_file=None,
                                          iou_type="bbox", category_type='category_id'):
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(str(json_result_file))
    coco_eval = ExtendedCOCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate(category_type=category_type)
    coco_eval.accumulate(category_type=category_type)
    coco_eval.summarize(category_type=category_type)
    return coco_eval


class ExtendedCOCOResults(COCOResults):
    def __init__(self, *iou_types):
        super(ExtendedCOCOResults, self).__init__(*iou_types)

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, ExtendedCOCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = ExtendedCOCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]


def prepare_for_coco_detection(predictions, dataset):
    extended_coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()

        scores = {}
        labels = {}
        mapped_labels = {}

        for property in PROPERTY:
            if not prediction.has_field(property['score']):
                scores[property['id']] = None
                continue
            scores[property['id']] = prediction.get_field(property['score']).tolist()
            labels[property['id']] = prediction.get_field(property['label']).tolist()

            if property['id'] == 1:
                mapped_labels[property['id']] = [dataset.contiguous_category_id_to_json_id[i] for i in labels[property['id']]]
            elif property['id'] == 2:
                mapped_labels[property['id']] = [dataset.contiguous_second_category_id_to_json_id[i] for i in labels[property['id']]]
            elif property['id'] == 3:
                mapped_labels[property['id']] = [dataset.json_third_category_id_to_contiguous_id[i] for i in labels[property['id']]]

        if scores[PROPERTY[1]['id']] is None:
            for k, box in enumerate(boxes):
                x = {
                    "image_id": original_id,
                    "bbox": box,
                    PROPERTY[0]['category_id']: mapped_labels[PROPERTY[0]['id']][k],
                    # PROPERTY[1]['category_id']: mapped_labels[PROPERTY[1]['id']][k],
                    # property[PROPERTY[2]['category_id']]: mapped_labels[PROPERTY[2]['id']][k],
                    PROPERTY[0]['coco_score']: scores[PROPERTY[0]['id']][k],
                    # PROPERTY[1]['coco_score']: scores[PROPERTY[1]['id']][k],
                    # PROPERTY[2]['coco_score']: scores[PROPERTY[2]['id']][k],
                    }
                extended_coco_results.extend([x])
        else:
            for k, box in enumerate(boxes):
                x = {
                    "image_id": original_id,
                    "bbox": box,
                    PROPERTY[0]['category_id']: mapped_labels[PROPERTY[0]['id']][k],
                    PROPERTY[1]['category_id']: mapped_labels[PROPERTY[1]['id']][k],
                    # property[PROPERTY[2]['category_id']]: mapped_labels[PROPERTY[2]['id']][k],
                    PROPERTY[0]['coco_score']: scores[PROPERTY[0]['id']][k],
                    PROPERTY[1]['coco_score']: scores[PROPERTY[1]['id']][k],
                    # PROPERTY[2]['coco_score']: scores[PROPERTY[2]['id']][k],
                    }
                extended_coco_results.extend([x])
    return extended_coco_results


def prepare_for_coco_segmentation(predictions, dataset, property=PROPERTY[0]):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    extended_coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        masks = masker([masks], [prediction])[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        # scores = prediction.get_field(property['score']).tolist()
        # labels = prediction.get_field(property['label']).tolist()

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]

        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
        #     rle["counts"] = mask_util.decode(rle)

        scores = {}
        labels = {}
        mapped_labels = {}

        for property in PROPERTY:
            if not prediction.has_field(property['score']):
                scores[property['id']] = None
                continue

            scores[property['id']] = prediction.get_field(property['score']).tolist()
            labels[property['id']] = prediction.get_field(property['label']).tolist()

            if property['id'] == 1:
                mapped_labels[property['id']] = [dataset.contiguous_category_id_to_json_id[i] for i in labels[property['id']]]
            elif property['id'] == 2:
                mapped_labels[property['id']] = [dataset.contiguous_second_category_id_to_json_id[i] for i in labels[property['id']]]
            elif property['id'] == 3:
                mapped_labels[property['id']] = [dataset.json_third_category_id_to_contiguous_id[i] for i in labels[property['id']]]

        if scores[PROPERTY[1]['id']] is None:
            for k, rle in enumerate(rles):
                x = {
                    "image_id": original_id,
                    "segmentation": rle,
                    PROPERTY[0]['category_id']: mapped_labels[PROPERTY[0]['id']][k],
                    # PROPERTY[1]['category_id']: mapped_labels[PROPERTY[1]['id']][k],
                    # property[PROPERTY[2]['category_id']]: mapped_labels[PROPERTY[2]['id']][k],
                    PROPERTY[0]['coco_score']: scores[PROPERTY[0]['id']][k],
                    # PROPERTY[1]['coco_score']: scores[PROPERTY[1]['id']][k],
                    # PROPERTY[2]['coco_score']: scores[PROPERTY[2]['id']][k],
                    }
                extended_coco_results.extend([x])
        else:
            for k, rle in enumerate(rles):
                x = {
                    "image_id": original_id,
                    "segmentation": rle,
                    PROPERTY[0]['category_id']: mapped_labels[PROPERTY[0]['id']][k],
                    PROPERTY[1]['category_id']: mapped_labels[PROPERTY[1]['id']][k],
                    # property[PROPERTY[2]['category_id']]: mapped_labels[PROPERTY[2]['id']][k],
                    PROPERTY[0]['coco_score']: scores[PROPERTY[0]['id']][k],
                    PROPERTY[1]['coco_score']: scores[PROPERTY[1]['id']][k],
                    # PROPERTY[2]['coco_score']: scores[PROPERTY[2]['id']][k],
                    }
                extended_coco_results.extend([x])
    return extended_coco_results


def prepare_for_coco_keypoint(predictions, dataset, property=PROPERTY[0]):
    extended_coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        keypoints = prediction.get_field('keypoints')
        keypoints = keypoints.resize((image_width, image_height))
        keypoints = keypoints.keypoints.view(keypoints.keypoints.shape[0], -1).tolist()

        scores = {}
        labels = {}
        mapped_labels = {}

        for property in PROPERTY:
            if not prediction.has_field(property['score']):
                scores[property['id']] = None
                continue

            scores[property['id']] = prediction.get_field(property['score']).tolist()
            labels[property['id']] = prediction.get_field(property['label']).tolist()

            if property['id'] == 1:
                mapped_labels[property['id']] = [dataset.contiguous_category_id_to_json_id[i] for i in labels[property['id']]]
            elif property['id'] == 2:
                mapped_labels[property['id']] = [dataset.contiguous_second_category_id_to_json_id[i] for i in labels[property['id']]]
            elif property['id'] == 3:
                mapped_labels[property['id']] = [dataset.json_third_category_id_to_contiguous_id[i] for i in labels[property['id']]]

        if scores[PROPERTY[1]['id']] is None:
            for k, keypoint in enumerate(keypoints):
                x = {
                    "image_id": original_id,
                    "keypoints": keypoint,
                    PROPERTY[0]['category_id']: mapped_labels[PROPERTY[0]['id']][k],
                    # PROPERTY[1]['category_id']: mapped_labels[PROPERTY[1]['id']][k],
                    # property[PROPERTY[2]['category_id']]: mapped_labels[PROPERTY[2]['id']][k],
                    PROPERTY[0]['coco_score']: scores[PROPERTY[0]['id']][k],
                    # PROPERTY[1]['coco_score']: scores[PROPERTY[1]['id']][k],
                    # PROPERTY[2]['coco_score']: scores[PROPERTY[2]['id']][k],
                    }
                extended_coco_results.extend([x])
        else:
            for k, keypoint in enumerate(keypoints):
                x = {
                    "image_id": original_id,
                    "keypoints": keypoint,
                    PROPERTY[0]['category_id']: mapped_labels[PROPERTY[0]['id']][k],
                    PROPERTY[1]['category_id']: mapped_labels[PROPERTY[1]['id']][k],
                    # property[PROPERTY[2]['category_id']]: mapped_labels[PROPERTY[2]['id']][k],
                    PROPERTY[0]['coco_score']: scores[PROPERTY[0]['id']][k],
                    PROPERTY[1]['coco_score']: scores[PROPERTY[1]['id']][k],
                    # PROPERTY[2]['coco_score']: scores[PROPERTY[2]['id']][k],
                    }
                extended_coco_results.extend([x])
    return extended_coco_results



