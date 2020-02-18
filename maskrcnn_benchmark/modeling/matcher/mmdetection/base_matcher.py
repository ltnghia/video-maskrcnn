from abc import ABCMeta, abstractmethod
from .match_result import MatchResult
from maskrcnn_benchmark.structures.boxes import BoxList
from maskrcnn_benchmark.modeling.matcher.matcher import Matcher


class BaseMatcher(metaclass=ABCMeta):

    BELOW_LOW_THRESHOLD = Matcher.BELOW_LOW_THRESHOLD
    BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        return None

    def __call__(self, anchor, target, target_ignore=None):
        """
        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        bboxes = anchor.convert("xyxy")
        gt_bboxes = target.convert("xyxy")

        if target.has_field("labels"):
            gt_labels = target.get_field('labels')
        else:
            gt_labels = None

        if target_ignore:
            gt_bboxes_ignore = target_ignore.convert("xyxy")
        else:
            gt_bboxes_ignore = None

        assign_result = self.assign(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels)
        matches = assign_result.assigned_gt_inds
        return assign_result, matches
