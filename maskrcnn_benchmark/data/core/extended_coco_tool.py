import json
import time
import numpy as np
import copy
from collections import defaultdict
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO, _isArrayLike
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import itertools
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


class ExtendedCOCO(COCO):
    def __init__(self, annotation_file=None):
        super(ExtendedCOCO, self).__init__(annotation_file=annotation_file)

    def createIndex(self):
        super(ExtendedCOCO, self).createIndex()

        reg_anns, seg_anns, second_cats, third_cats = {}, {}, {}, {}

        if 'second_categories' in self.dataset:
            for cat in self.dataset['second_categories']:
                second_cats[cat['id']] = cat
        self.second_cats = second_cats

        if 'third_categories' in self.dataset:
            for cat in self.dataset['third_categories']:
                third_cats[cat['id']] = cat
        self.third_cats = third_cats

        try:
            secondCatToImgs = defaultdict(list)
            if 'annotations' in self.dataset and 'second_categories' in self.dataset:
                for ann in self.dataset['annotations']:
                    secondCatToImgs[ann['second_category_id']].append(ann['image_id'])
            self.secondCatToImgs = secondCatToImgs
        except:
            self.secondCatToImgs = secondCatToImgs

        try:
            thirdCatToImgs = defaultdict(list)
            if 'annotations' in self.dataset and 'third_categories' in self.dataset:
                for ann in self.dataset['annotations']:
                    thirdCatToImgs[ann['third_category_id']].append(ann['image_id'])
            self.thirdCatToImgs = thirdCatToImgs
        except:
            self.thirdCatToImgs = thirdCatToImgs

        imgToRegAnns = defaultdict(list)
        if 'recognition_annotations' in self.dataset:
            for ann in self.dataset['recognition_annotations']:
                if ann is None:
                    pass
                reg_anns[ann['id']] = ann
                imgToRegAnns[ann['image_id']].append(ann)
        self.reg_anns = reg_anns
        self.imgToRegAnns = imgToRegAnns

        imgToSegAnns = defaultdict(list)
        if 'segmentation_annotations' in self.dataset:
            for ann in self.dataset['segmentation_annotations']:
                seg_anns[ann['id']] = ann
                imgToSegAnns[ann['image_id']].append(ann)
        self.seg_anns = seg_anns
        self.imgToSegAnns = imgToSegAnns

    def getCatIds(self, catNms=[], supNms=[], catIds=[], category_type='categories'):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        if category_type not in self.dataset.keys():
            return None

        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset[category_type]
        else:
            cats = self.dataset[category_type]
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getCats(self, catNms=[], supNms=[], catIds=[], category_type='categories'):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        if category_type not in self.dataset.keys():
            return None

        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset[category_type]
        else:
            cats = self.dataset[category_type]
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        return cats

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None, category_type='category_id'):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann[category_type] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = ExtendedCOCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            if 'second_categories' in self.dataset.keys():
                res.dataset['second_categories'] = copy.deepcopy(self.dataset['second_categories'])
            if 'third_categories' in self.dataset.keys():
                res.dataset['third_categories'] = copy.deepcopy(self.dataset['third_categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def loadNumpyAnnotations(self, data):
        raise Exception("Not support")
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            x = {
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }
            if len(data) >= 8:
                x.update({'second_category_id': int(data[i, 7])})
            if len(data) >= 9:
                x.update({'third_category_id': int(data[i, 8])})
            ann += [x]
        return ann



