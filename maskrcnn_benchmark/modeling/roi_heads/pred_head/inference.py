import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.boxes import BoxList


class PredPostProcessor(nn.Module):
    def __init__(self, score_thresh=0.05):
        super(PredPostProcessor, self).__init__()
        self.score_thresh = score_thresh

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_prob = F.softmax(x, -1)
        num_classes = class_prob.shape[1]
        device = class_prob.device

        boxes_per_image = [len(box) for box in boxes]
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for scores, box in zip(class_prob, boxes):
            if len(box) == 0:
                continue
            pred_scores = torch.full((len(box),), 0, dtype=torch.float, device=device)
            pred_labels = torch.full((len(box),), 0, dtype=torch.int64, device=device)

            # inds_all = scores > self.score_thresh
            val, argmax = scores[:, 1:].max(dim=1)
            argmax = argmax + 1
            for j in range(1, num_classes):
                # inds1 = inds_all[:, j]
                inds2 = (argmax == j)
                # inds = inds1 * inds2
                inds = inds2
                inds = inds.nonzero().squeeze(1)
                pred_scores[inds] = scores[inds][:, j]
                pred_labels[inds] = torch.full((len(inds),), j, dtype=torch.int64, device=device)

            box.add_field("pred_scores", pred_scores)
            box.add_field("pred_labels", pred_labels)
            results.append(box)

        return results


def make_roi_pred_post_processor(cfg):
    postprocessor = PredPostProcessor()
    return postprocessor
