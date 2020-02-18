import os
import numpy as np
import json
import cv2

CATEGORIES = [
    "__background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def do_coco_semantic_evaluation(
        dataset,
        predictions,
        output_folder,
):
    gts, preds = [], []
    coco_results = []
    semantic_results = []
    # TODO give number of classes as an argument
    n_classes = 133
    for image_id, pred in enumerate(predictions):
        semantic_results.append(pred[0])
        # Display semantic logits
        # res = pred[0].astype(np.uint8)
        # cv2.imshow("Segments", res[0])
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        for pred_, gt_ in zip(pred[0], pred[1]):
            preds.append(pred_)
            gts.append(gt_)

    score, class_iou = scores(gts, preds, n_class=n_classes)

    file_path = os.path.join(output_folder, "segmentation" + ".json")

    coco_results.extend(
        [
            {
                k: v,
            }
            for k, v in score.items()
        ]
    )
    coco_results.extend(
        [
            {
                "IoU": class_iou,
            }
        ]
    )
    for k, v in score.items():
        print(k, v)
    for i in range(n_classes):
        if i < 52:
            print(i, class_iou[i])
        else:
            print(CATEGORIES[i - 52], class_iou[i])

    with open(file_path, "w") as f:
        json.dump(coco_results, f)

    return semantic_results


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    # import pprint
    # print("ACC")
    # pprint.pprint(acc_cls)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # print("IoU")
    # pprint.pprint(iu)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu, }, cls_iu