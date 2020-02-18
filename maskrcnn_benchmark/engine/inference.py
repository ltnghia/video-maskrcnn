# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import torch
import tqdm

from maskrcnn_benchmark.data.evaluation import evaluate_roi, evaluate_img
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size, all_gather, synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.config import cfg
from .bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.visualization.visualizer import Visualizer


def compute_on_dataset(model, data_loader, device, bbox_aug=False, timer=None):
    model.eval()
    roi_results_dict = {}
    img_results_dict = {}
    attention_results = []
    img_ids = []
    cpu_device = torch.device("cpu")
    if is_main_process():
        tq = tqdm.tqdm(total=len(data_loader))

    for i, batch in enumerate(data_loader):
        if is_main_process():
            tq.update(1)

        images = batch[0]
        if len(batch) > 2:
            info = batch[2]
        else:
            info = None

        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not device == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            if isinstance(output, dict):
                roi_output = output['detector_results']
                img_output = output['decoder_results']
                attention_map = output['attention_map']
            else:
                roi_output = output
                img_output = None
                attention_map = None
            roi_output = [o.to(cpu_device) for o in roi_output]

        roi_results_dict.update({img_info['id']: result for img_info, result in zip(info, roi_output)})

        if img_output is not None or attention_map is not None:
            img_ids.extend([img_info['id'] for img_info in info])
            if img_output is not None:
                for v in img_output.values():
                    v = v.to(cpu_device)
                if len(img_results_dict.keys()) == 0:
                    img_results_dict.update(img_output)
                else:
                    for k, v in img_output.items():
                        img_results_dict[k].cat(v)
            if attention_map is not None:
                attention_map = attention_map.to(cpu_device)
                attention_results.extend(torch.split(attention_map, 1))

    if is_main_process():
        tq.close()

    results = {}
    if len(img_ids) > 0 and len(img_results_dict.keys()) > 0:
        for k, v in img_results_dict.items():
            img_results_dict[k] = v.to_list()
        for i in range(len(img_ids)):
            collection = {}
            for k, v in img_results_dict.items():
                collection[k] = v[i]
            results[img_ids[i]] = collection
    img_results_dict = results

    results = {}
    if len(img_ids) > 0 and len(attention_results) > 0:
        for i in range(len(img_ids)):
            results[img_ids[i]] = attention_results[i].squeeze()
    attention_results = results

    return roi_results_dict, img_results_dict, attention_results


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        # print(len(image_ids), image_ids[-1])
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning("Number of images that were gathered from multiple processes is not a contiguous set. "
                       "Some images might be missing from the evaluation.")

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def save_as_bdd_format(preds, path, name, img_names):
    preds_bdd = []
    for j in range(len(preds)):
        pred = preds[j]
        pred_bdd = {
            'name': img_names[j],
            'labels': []
        }
        boxes = pred.bbox.numpy().tolist()
        labels = pred.get_field('labels').numpy().tolist()
        scores = pred.get_field('scores').numpy().tolist()
        for i in range(len(boxes)):
            pred_bdd['labels'] += [{
                'category': labels[i],
                'box2d': {
                    'x1': boxes[i][0],
                    'y1': boxes[i][1],
                    'x2': boxes[i][2],
                    'y2': boxes[i][3]
                },
                'score': scores[i]
            }]
        preds_bdd += [pred_bdd]
    path = os.path.join(path, '{}.json'.format(name))
    with open(path, 'w') as f:
        json.dump(preds_bdd, f)


def inference(
    model,
    data_loader,
    dataset_name,
    iou_types=("bbox",),
    box_only=False,
    device=torch.device("cuda"),
    expected_results=0,
    expected_results_sigma_tol=0,
    output_folder=None,
    cfg=None,
    bbox_aug=False,
    visualize_results=False,
    visualization_label="coco",
    only_visualization=False,
):
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    roi_predictions, img_predictions, attention_maps = compute_on_dataset(model, data_loader, device,
                                                                          bbox_aug=bbox_aug, timer=inference_timer)

    # wait for all processes to complete before measuring the time
    synchronize()

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if roi_predictions:
        roi_predictions = _accumulate_predictions_from_multiple_gpus(roi_predictions)
    if img_predictions:
        img_predictions = _accumulate_predictions_from_multiple_gpus(img_predictions)
    if attention_maps:
        attention_maps = _accumulate_predictions_from_multiple_gpus(attention_maps)

    if not is_main_process():
        return

    if roi_predictions and len(roi_predictions) > 0:
        for prediction in roi_predictions:
            if prediction.has_field("pred_scores"):
                prediction.add_field('second_scores', prediction.get_field('pred_scores'))
                del prediction.extra_fields["pred_scores"]
            if prediction.has_field("pred_labels"):
                prediction.add_field('second_labels', prediction.get_field('pred_labels'))
                del prediction.extra_fields["pred_labels"]

        if output_folder:
            torch.save(roi_predictions, os.path.join(output_folder, "roi_predictions.pth"))

        print('Visualize results')
        if output_folder and visualize_results:
            categories = import_file(
                "maskrcnn_benchmark.data.datasets.categories.{}_categories".format(visualization_label),
                os.path.join(os.path.dirname(os.path.dirname(cfg.PATHS_CATALOG)), 'data', 'categories',
                             '{}_categories.py'.format(visualization_label)),
                True
            )
            visualizer = Visualizer(categories=categories.CATEGORIES, cfg=cfg)
            visualizer.visualize_attentions(attention_maps, dataset, os.path.join(output_folder, 'attention_map'))
            visualizer.visualize_predictions(roi_predictions, dataset, os.path.join(output_folder, 'visualization'))
            if only_visualization:
                return

        extra_args = dict(
            box_only=box_only,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )

        print('ROI: Evaluate')
        evaluate_roi(dataset=dataset,
                     predictions=roi_predictions,
                     output_folder=output_folder,
                     **extra_args)

    if img_predictions and len(img_predictions) > 0:
        if output_folder:
            torch.save(img_predictions, os.path.join(output_folder, "img_predictions.pth"))
        print('IMAGE: Evaluate')
        evaluate_img(dataset=dataset,
                     predictions=img_predictions,
                     output_folder=output_folder)



