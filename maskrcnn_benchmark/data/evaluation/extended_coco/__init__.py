from .extended_coco_eval import do_coco_evaluation
from .image_evaluation import do_image_evaluation


def extended_coco_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    return do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )


def image_evaluation(
    dataset,
    predictions,
    output_folder
):
    return do_image_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder
    )

