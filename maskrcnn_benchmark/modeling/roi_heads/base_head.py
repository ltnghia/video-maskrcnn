import torch
from torch import nn
from maskrcnn_benchmark.structures.boxes import BoxList


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIBaseHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIBaseHead, self).__init__()
        self.cfg = cfg.clone()
        self.use_extended_features = False
        self.feature_extractor = None
        self.predictor = None
        self.post_processor = None
        self.loss_evaluator = None

    def forward(self, features, proposals, targets=None):
        pass

