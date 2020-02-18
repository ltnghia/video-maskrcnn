# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .boxes import BoxList
import numpy as np
from maskrcnn_benchmark.layers import nms as _box_nms
import copy


def nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def soft_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    return None


def softer_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    return None


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2, mode='iou'):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
      mode (str): "iou" (intersection over union) or iof (intersection over foreground).

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """

    assert mode in ['iou', 'iof']

    if boxlist1.size != boxlist2.size:
        raise RuntimeError("boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    if mode == 'iou':
        iou = inter / (area1[:, None] + area2 - inter)
    else:
        iou = inter / (area1[:, None])

    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def expand_boxes(boxes, scale=1):
    # device = boxes.bbox.device
    boxes_exp = copy.deepcopy(boxes)
    boxes_exp = boxes_exp.convert("xyxy")
    im_w, im_h = boxes_exp.size
    bbox = boxes_exp.bbox#.to(dtype=torch.float)

    w_half = (bbox[:, 2] - bbox[:, 0]) * .5
    h_half = (bbox[:, 3] - bbox[:, 1]) * .5
    x_c = (bbox[:, 2] + bbox[:, 0]) * .5
    y_c = (bbox[:, 3] + bbox[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    # bbox[:, 0] = max(x_c - w_half, 0)
    # bbox[:, 2] = min(x_c + w_half, im_w-1)
    # bbox[:, 1] = max(y_c - h_half, 0)
    # bbox[:, 3] = min(y_c + h_half, im_h-1)

    bbox[:, 0] = x_c - w_half
    bbox[:, 2] = x_c + w_half
    bbox[:, 1] = y_c - h_half
    bbox[:, 3] = y_c + h_half

    bbox[bbox < 0] = 0
    bbox[bbox[:, 2] > im_w - 1, 2] = im_w - 1
    bbox[bbox[:, 3] > im_h - 1, 3] = im_h - 1

    boxes_exp.bbox = bbox#.to(dtype=torch.int32)

    # print(boxes.bbox[0, :])
    # print(boxes_exp.bbox[0, :])

    return boxes_exp


def expand(boxes, scale=1):
    if isinstance(boxes, list):
        return [expand_boxes(box, scale) for box in boxes]
    else:
        return expand_boxes(boxes, scale)


def search_neighbors_single_image(boxes, scale=1, threshold=0, output_only_index=True):
    n = len(boxes)
    if scale == 1:
        query_boxes = boxes
    else:
        query_boxes = expand(boxes, scale)

    iou = boxlist_iou(query_boxes, boxes, mode='iou')
    neighbor_indices = [[]] * n
    if not output_only_index:
        neighbors = [None] * n
    else:
        neighbors = None
    iou = iou.cpu().numpy()
    position = (threshold < iou) * (iou <= 1)
    index = np.asarray(list(range(n)))
    for i in range(n):
        iou_index = index[position[i, :]]
        # print(i, iou_index)
        neighbor_indices[i] = iou_index
        if not output_only_index:
            neighbors[i] = boxes[iou_index]
            neighbors[i] = cat_boxlist(neighbors[i])

    return neighbor_indices, neighbors


def search_neighbors(boxes, scale=1, threshold=0, output_only_index=True):
    if isinstance(boxes, list):
        neighbor_indices = []
        neighbors = []
        for box in boxes:
            index, neighbor = search_neighbors_single_image(box, scale, threshold, output_only_index)
            neighbor_indices.append(index)
            neighbors.append(neighbor)
        # print(neighbor_indices)
        return neighbor_indices, neighbors
    else:
        return search_neighbors_single_image(boxes, scale, threshold, output_only_index)






