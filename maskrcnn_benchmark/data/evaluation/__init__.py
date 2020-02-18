from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation

from .extended_coco import extended_coco_evaluation, image_evaluation

from .cityscapes import abs_cityscapes_evaluation
from maskrcnn_benchmark.data.evaluation.coco.coco_eval import do_coco_evaluation as do_orig_coco_evaluation
from maskrcnn_benchmark.data.evaluation.coco.coco_eval_wrapper import do_coco_evaluation as do_wrapped_coco_evaluation
from maskrcnn_benchmark.data.datasets import AbstractDataset, COCODataset


def evaluate_roi(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.ExtendedCOCODataset):
        return extended_coco_evaluation(**args)
    elif isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.AbstractDataset):
        return abs_cityscapes_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))


def evaluate_img(dataset, predictions, output_folder):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder
    )
    if isinstance(dataset, datasets.ExtendedCOCODataset):
        return image_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))


def coco_evaluation(dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    if isinstance(dataset, COCODataset):
        return do_orig_coco_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    elif isinstance(dataset, AbstractDataset):
        return do_wrapped_coco_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    else:
        raise NotImplementedError(
            (
                    "Ground truth dataset is not a COCODataset, "
                    "nor it is derived from AbstractDataset: type(dataset)="
                    "%s" % type(dataset)
            )
        )

