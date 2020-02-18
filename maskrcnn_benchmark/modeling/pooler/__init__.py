from .poolers import Pooler
from .contextual_poolers import ContextualPooler

from maskrcnn_benchmark.layers import ROIAlign
from maskrcnn_benchmark.layers import PrRoIPool2D
from maskrcnn_benchmark.layers import ROIPool

from maskrcnn_benchmark.modeling import registry


@registry.POOLER.register("ROIPool")
def build_roi_pooling(cfg, head_name):
    """
    Arguments:
        output_size (list[tuple[int]] or list[int]): output size for the pooled region
        scales (list[float]): scales for each Pooler
    """
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    output_size = (resolution, resolution)
    use_torchvision = cfg.MODEL[head_name].USE_TORCH_VISION_POOLING

    poolers = []
    for scale in scales:
        pooler = ROIPool(output_size, spatial_scale=scale, use_torchvision=use_torchvision)
        poolers.append(pooler)
    return poolers


@registry.POOLER.register("ROIAlign")
def build_roi_align(cfg, head_name):
    """
    Arguments:
        output_size (list[tuple[int]] or list[int]): output size for the pooled region
        scales (list[float]): scales for each Pooler
        sampling_ratio (int): sampling ratio for ROIAlign
    """
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    output_size = (resolution, resolution)
    use_torchvision = cfg.MODEL[head_name].USE_TORCH_VISION_POOLING

    poolers = []
    for scale in scales:
        pooler = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, use_torchvision=use_torchvision)
        poolers.append(pooler)
    return poolers


@registry.POOLER.register("PrRoIPool2D")
def build_precise_roi_pooling(cfg, head_name):
    """
    Arguments:
        output_size (list[tuple[int]] or list[int]): output size for the pooled region
        scales (list[float]): scales for each Pooler
    """
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    output_size = (resolution, resolution)
    use_torchvision = cfg.MODEL[head_name].USE_TORCH_VISION_POOLING

    poolers = []
    for scale in scales:
        pooler = PrRoIPool2D(output_size, spatial_scale=scale, use_torchvision=use_torchvision)
        poolers.append(pooler)
    return poolers


def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    canonical_level = cfg.MODEL[head_name].CANONICAL_LEVEL
    poolers = registry.POOLER[cfg.MODEL[head_name].POOLER_TYPE](cfg, head_name)

    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        poolers=poolers,
        canonical_level=canonical_level,
    )
    return pooler


def make_contextual_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    canonical_level = cfg.MODEL[head_name].CANONICAL_LEVEL
    poolers = registry.POOLER[cfg.MODEL[head_name].POOLER_TYPE](cfg, head_name)

    pooler = ContextualPooler(
        output_size=(resolution, resolution),
        scales=scales,
        poolers=poolers,
        canonical_level=canonical_level,
    )
    return pooler