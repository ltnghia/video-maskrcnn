from .poolers import Pooler


class ContextualPooler(Pooler):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, poolers, canonical_level=4):
        super(ContextualPooler, self).__init__(output_size, scales, poolers, canonical_level)

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        result = super(ContextualPooler, self).forward(x, boxes)
        return result


