import torch
from torchvision.transforms import functional as F

from maskrcnn_benchmark.structures.boxes import BoxList
from maskrcnn_benchmark.structures.masks import SegmentationMask
from maskrcnn_benchmark.structures.keypoints import PersonKeypoints

from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.data.core import ExtendedCOCO


class ExtendedCOCODataset(COCODataset):
    def __init__(self, ann_file, root, remove_images_without_annotations=False, transforms=None):
        super(ExtendedCOCODataset, self).__init__(ann_file=ann_file,
                                                  root=root,
                                                  remove_images_without_annotations=remove_images_without_annotations,
                                                  transforms=transforms)
        self.coco = ExtendedCOCO(ann_file)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        #
        # # filter images without detection annotations
        # if remove_images_without_annotations:
        #     ids = []
        #     for img_id in self.ids:
        #         ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        #         anno = self.coco.loadAnns(ann_ids)
        #         if has_valid_annotation(anno):
        #             ids.append(img_id)
        #         self.ids = ids
        #
        # self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        #
        # self.json_category_id_to_contiguous_id = {
        #     v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        # }
        # self.contiguous_category_id_to_json_id = {
        #     v: k for k, v in self.json_category_id_to_contiguous_id.items()
        # }
        # self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        # self._transforms = transforms

        if self.coco.getCatIds(category_type='second_categories'):
            self.second_categories = {cat['id']: cat['name'] for cat in self.coco.second_cats.values()}
            self.json_second_category_id_to_contiguous_id = {
                v: i + 1 for i, v in enumerate(self.coco.getCatIds(category_type='second_categories'))
            }
            self.contiguous_second_category_id_to_json_id = {
                v: k for k, v in self.json_second_category_id_to_contiguous_id.items()
            }
        else:
            self.second_categories = None
            self.json_second_category_id_to_contiguous_id = None
            self.contiguous_second_category_id_to_json_id = None

        if self.coco.getCatIds(category_type='third_categories'):
            self.third_categories = {cat['id']: cat['name'] for cat in self.coco.third_cats.values()}
            self.json_third_category_id_to_contiguous_id = {
                v: i + 1 for i, v in enumerate(self.coco.getCatIds(category_type='third_categories'))
            }
            self.contiguous_third_category_id_to_json_id = {
                v: k for k, v in self.json_third_category_id_to_contiguous_id.items()
            }
        else:
            self.third_categories = None
            self.json_third_category_id_to_contiguous_id = None
            self.contiguous_third_category_id_to_json_id = None

    def __getitem__(self, index):
        img, anno = super(COCODataset, self).__getitem__(index)

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        roi_target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        first_cat = []
        second_cat = []
        third_cat = []
        for obj in anno:
            if "category_id" in obj.keys():
                if obj["category_id"] is not None:
                    label1 = self.json_category_id_to_contiguous_id[obj["category_id"]]
                    first_cat.append(label1)

            if "second_category_id" in obj.keys():
                if obj["second_category_id"] is not None:
                    label2 = self.json_second_category_id_to_contiguous_id[obj["second_category_id"]]
                    second_cat.append(label2)

            if "third_category_id" in obj.keys():
                if obj["third_category_id"] is not None:
                    label3 = self.json_third_category_id_to_contiguous_id[obj["third_category_id"]]
                    third_cat.append(label3)

        if len(first_cat) > 0:
            first_cat = [c for c in first_cat]
            first_cat = torch.tensor(first_cat)
            roi_target.add_field("labels", first_cat)

        if len(second_cat) > 0:
            second_cat = [c for c in second_cat]
            second_cat = torch.tensor(second_cat)
            roi_target.add_field("second_labels", second_cat)

        if len(third_cat) > 0:
            third_cat = [c for c in third_cat]
            third_cat = torch.tensor(third_cat)
            roi_target.add_field("third_labels", third_cat)

        if anno and "segmentation" in anno[0]:
            try:
                masks = [obj["segmentation"] for obj in anno]
                masks = SegmentationMask(masks, img.size, mode='poly')
                roi_target.add_field("masks", masks)
            except:
                masks = None

        if anno and "keypoints" in anno[0]:
            try:
                keypoints = [obj["keypoints"] for obj in anno]
                keypoints = PersonKeypoints(keypoints, img.size)
                roi_target.add_field("keypoints", keypoints)
            except:
                keypoints = None

        roi_target = roi_target.clip_to_image(remove_empty=True)

        reg_target = self.coco.imgToRegAnns[img_id]
        seg_target = self.coco.imgToSegAnns[img_id]

        if len(reg_target) > 0:
            if reg_target[0]['label'] in self.json_second_category_id_to_contiguous_id.keys():
                reg_target = torch.tensor(self.json_second_category_id_to_contiguous_id[reg_target[0]['label']])
            else:
                reg_target = torch.tensor(0)
        else:
            reg_target = None

        if len(seg_target) > 0:
            seg_target = F.to_tensor(seg_target[0]['label'])
        else:
            seg_target = None

        if self._transforms is not None:
            if seg_target:
                img, roi_target, seg_target = self._transforms(img, roi_target, seg_target)
            else:
                img, roi_target = self._transforms(img, roi_target)

        info = self.get_img_info(index)

        # return img, roi_target, dict(id=index, info=info)
        return img, dict(roi_target=roi_target, reg_target=reg_target, seg_target=seg_target), dict(id=index, info=info)
