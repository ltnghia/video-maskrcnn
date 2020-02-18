import torch.utils.data as data


class COCOPanopticDataset(data.Dataset):
    def __init__(self, detection_dataset, semantic_dataset, transforms):
        self.detection_dataset = detection_dataset
        self.semantic_dataset = semantic_dataset
        self.transforms = transforms

    def __getitem__(self, index):
        img, target_detection, _ = self.detection_dataset.__getitem__(index)
        img, target_semantic, _ = self.semantic_dataset.__getitem__(index)

        # Check if there are not inconsistency with transforms
        if (self.transforms is not None):
            img, target_detection, target_semantic = self.transforms(img, target_detection, target_semantic)

        return img, target_detection, target_semantic, index

    # TODO manage correctly the length
    def __len__(self):
        return len(self.detection_dataset.ids)

    def get_img_info(self, index):
        img_id = self.detection_dataset.id_to_img_map[index]
        img_data = self.detection_dataset.coco.imgs[img_id]
        return img_data
