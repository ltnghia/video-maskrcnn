# class-balanced loss based on effective number of samples, CVPR 2019

import torch
import numpy as np
import json
import os
from maskrcnn_benchmark.utils.miscellaneous import load_json, write_json
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
import pprint
# from .utils import add_weights_and_normalize
from maskrcnn_benchmark.layers.loss.utils import add_weights_and_normalize


class ClassBalanceLoss:
    def __init__(self, device=torch.device('cpu'), num_class_list=None,
                 alpha=1, beta=0.999):
        # num_class_list: [num_classes] a list of sampler numbers for each class

        self.alpha = alpha
        self.beta = beta
        self.num_classes = len(num_class_list) + 1

        #######################################
        # Add background class: number of samples are equal total possible samples
        num_negative_samples = np.array([N for N in num_class_list]).sum()
        num_negative_samples = np.round(self.alpha * num_negative_samples)
        if num_class_list:
            num_class_list.insert(0, num_negative_samples)
        #######################################

        if num_class_list and device:
            self.weight_list = np.array([0 if N == 0 else (1 - self.beta) / (1 - self.beta ** N) for N in num_class_list])
            self.weight_list = torch.Tensor(self.weight_list / np.sum(self.weight_list)).to(device, dtype=torch.float)

    def get_weights(self, label):
        weight = self.weight_list[label.long()]
        # weight = self.weight_list.index_select(0, label)
        return weight

    def add_weights(self, loss, label, reduction='sum'):
        # label: [batch_size]
        weight = self.weight_list[label]
        # weight = self.weight_list.index_select(0, label)
        return add_weights_and_normalize(loss,
                                         label=label,
                                         weight=weight,
                                         reduction=reduction)

    @staticmethod
    def compute_class_samples(annotation_file, outputfile=None, sample=None):
        dataset = load_json(annotation_file)
        if not dataset:
            return

        if sample is None:
            sample = {
                'category': {},
                'second_category': {},
                'third_category': {},
                'image_ids': [],
                'recognition_category': {},
            }
            annotations = dataset['annotations']
            categories = dataset['categories']
            for cat in categories:
                sample['category'][cat['id']] = 0
            if "second_categories" in dataset.keys():
                second_categories = dataset['second_categories']
                for cat in second_categories:
                    sample['second_category'][cat['id']] = 0
                    sample['recognition_category'][cat['id']] = 0
            if "third_categories" in dataset.keys():
                third_categories = dataset['third_categories']
                for cat in third_categories:
                    sample['third_category'][cat['id']] = 0

        image_ids = []
        for ann in annotations:
            image_ids.append(ann['image_id'])
            category_id = ann["category_id"]
            sample['category'][category_id] += 1
            if "second_category_id" in ann.keys() and len(sample['second_category'].keys()) > 0:
                second_category_id = ann['second_category_id']
                sample['second_category'][second_category_id] += 1
            if "third_category_id" in ann.keys() and len(sample['third_category'].keys()) > 0:
                third_category_id = ann['third_category_id']
                sample['third_category'][third_category_id] += 1
        if 'recognition_annotations' in dataset.keys():
            for ann in dataset['recognition_annotations']:
                sample['recognition_category'][ann['label']] += 1
        image_ids = list(set(image_ids))
        sample['image_ids'].extend(image_ids)
        if outputfile:
            mkdir(os.path.dirname(outputfile))
            write_json(sample, outputfile)
        return sample

    @staticmethod
    def load_class_samples(filename, category_type='category'):
        data = load_json(filename)
        if category_type not in data.keys():
            category_type = 'category'
        data = data[category_type]
        num_class_list = []
        for key in sorted(data.keys()):
            num_class_list.append(data[key])
        return num_class_list


if __name__ == '__main__':

    annotation_file = DatasetCatalog.DATASETS['accident_dad_small_train_v2']['ann_file']

    outputfile = os.path.join(os.path.dirname(annotation_file),
                              os.path.splitext(os.path.basename(annotation_file))[0] + '_cbw.json')

    sample = ClassBalanceLoss.compute_class_samples(annotation_file=annotation_file,
                                                    outputfile=outputfile)

    # num_class_list = ClassBalanceLoss.load_class_samples(outputfile, category_type='category')
    #
    # cbl = ClassBalanceLoss(num_class_list=num_class_list)
    #
    # label = np.array([5,4,3,6,2,1,3,4,5,3,2,4,6,0,3])
    # label = torch.Tensor(label).to(dtype=torch.long)
    # print(label)
    #
    # weight_list = cbl.weight_list
    # weight = weight_list[label]
    #
    # print(weight)
    #
    # print(weight_list.index_select(0, label))

    # print(sample)

    print('done')
