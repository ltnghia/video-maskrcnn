import cv2
import torch
from maskrcnn_benchmark.layers.misc import interpolate
from pycocotools import mask as maskUtils
from maskrcnn_benchmark.utils.visualization import cv2_util
from maskrcnn_benchmark.structures.image_list import ImageList

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class SemanticMaskList(object):

    def __init__(self, mask, size=None):

        if isinstance(mask, torch.Tensor):
            # The raw data representation is passed as argument
            mask = mask.clone()
        elif isinstance(mask, SemanticMaskList):
            # just hard copy the BinaryMaskList instance's underlying data
            mask = mask.mask.clone()
            size = mask.size
        else:
            RuntimeError(
                "Type of `masks` argument could not be interpreted:%s" % type(mask)
            )

        self.mask = mask
        if size is not None:
            self.size = tuple(size)
        else:
            self.size = None

    def transpose(self, method):
        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_masks = self.mask.flip(dim)
        return SemanticMaskList(flipped_masks, self.size)

    def crop(self, box=None):
        if box is None:
            return SemanticMaskList(self.mask)
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = round(float(bbox))

        assert xmin <= xmax and ymin <= ymax
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_mask = self.mask[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return SemanticMaskList(cropped_mask, cropped_size)

    def resize(self, size=None):
        if box is None:
            return SemanticMaskList(self.mask)
        width, height = map(int, size)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_mask = interpolate(
            input=self.mask.float(),
            size=(height, width),
            mode="nearest",
            align_corners=False,
        )[0].type_as(self.mask)
        resized_size = width, height
        return SemanticMaskList(resized_mask, resized_size)

    def to(self, device):
        self.mask = self.value.to(device)
        return self

    def _findContours(self):
        contours = []
        mask = self.mask.detach().numpy()
        mask = cv2.UMat(mask)
        contour, hierarchy = cv2_util.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )
        reshaped_contour = []
        for entity in contour:
            assert len(entity.shape) == 3
            assert entity.shape[1] == 1, "Hierarchical contours are not allowed"
            reshaped_contour.append(entity.reshape(-1).tolist())
        contours.append(reshaped_contour)
        return contours

    def get_tensors(self):
        return self.mask

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s

    @staticmethod
    def to_device(x, device):
        if x is None:
            return None
        if isinstance(x, (tuple, list)):
            if x[0] is None:
                return None
            output = SemanticMaskList(x[0])
            output.mask = [xi.mask for xi in x]
            output.mask = torch.stack(output.mask)
            output = output.to(device)
        else:
            output = x.to(device)
        return output.mask

    @staticmethod
    def annToRLE(segm, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = segm
        return rle

    @staticmethod
    def annToMask(ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = SemanticMaskList.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def to_mask_list(tensors, size_divisible=0, ignore_value=255):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new_full(*batch_shape, ignore_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_mask_list: {}".format(type(tensors)))
