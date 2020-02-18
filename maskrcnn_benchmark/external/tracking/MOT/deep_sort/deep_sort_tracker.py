import os
import cv2
import numpy as np
import shutil
from .deep_sort import DeepSort
from .util import draw_bboxes, draw_bboxes0
import json
import tqdm


class DeepSortTracker(object):
    def __init__(self,
                 dir_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), "deep/checkpoint/ckpt.t7"),
                 output_kf_bbox=True):
        self.dir_model = dir_model
        self.output_kf_bbox = output_kf_bbox
        self.deepsort = DeepSort(dir_model, output_kf_bbox=output_kf_bbox)

    def reset(self):
        self.deepsort = DeepSort(self.dir_model, output_kf_bbox=self.output_kf_bbox)

    def write_json(self, info, file):
        outfile = open(file, 'w')
        x = json.dumps(info)
        outfile.write(x)
        outfile.close()
        return file

    def load_json(self, dir_json):
        f = open(dir_json)
        data = json.load(f)

        if isinstance(data, dict):
            data = data.values()

        bbox_xywh = []
        cls_conf = []
        cls_ids = []

        for x in data:
            bbox = np.array([x['bbox']['x'], x['bbox']['y'], x['bbox']['w'], x['bbox']['h']]).astype(np.float)
            # bbox = np.array([int(x['bbox']['x']),
            #                  int(x['bbox']['y']),
            #                  int(x['bbox']['w']),
            #                  int(x['bbox']['h'])]).astype(np.float)
            bbox[:2] += bbox[2:] / 2
            bbox_xywh.append(bbox)
            if 'score' in x.keys():
                cls_conf.append(x['score'])
            else:
                cls_conf.append(1.0)
            cls_ids.append(x['category_id'])
        return np.array(bbox_xywh), np.array(cls_conf), np.array(cls_ids)

    def track(self, dir_images, dir_jsons, dir_output_img=None, dir_output_json=None, visualization=False, tq_display=False, reverse=False):

        list_images = os.listdir(dir_images)
        list_images = sorted(list_images, reverse=reverse)
        list_jsons = [os.path.join(dir_jsons, img.replace('.jpg', '.json')) for img in list_images]
        list_images = [os.path.join(dir_images, img) for img in list_images]

        if dir_output_img:
            os.makedirs(dir_output_img, exist_ok=True)

        if dir_output_json:
            if os.path.exists(dir_output_json):
                shutil.rmtree(dir_output_json)
            os.makedirs(dir_output_json, exist_ok=True)

        if tq_display:
            tq = tqdm.tqdm(total=len(list_images))
        else:
            tq = None

        for i, (dir_image, dir_json) in enumerate(zip(list_images, list_jsons)):
            if tq is not None:
                tq.set_description('Video frame: {}'.format(os.path.basename(dir_image)))
                tq.update(1)

            ori_im = cv2.imread(dir_image)
            im = ori_im[:, :, (2, 1, 0)]

            bbox_xyxy = []
            bbox_xyxy2 = []
            identities = []
            if os.path.isfile(dir_json):
                bbox_xywh, cls_conf, cls_ids = self.load_json(dir_json)

                if bbox_xywh is not None:

                    bbox_xyxy2 = np.array(bbox_xywh)
                    bbox_xyxy2[:, :2] = bbox_xywh[:, :2] - bbox_xywh[:, 2:] / 2
                    bbox_xyxy2[:, 2:] = bbox_xywh[:, :2] + bbox_xywh[:, 2:] / 2

                    # mask = cls_ids == 3
                    mask = cls_ids > 0
                    cls_ids = cls_ids[mask]
                    mask = mask.nonzero()
                    bbox_xywh = bbox_xywh[mask]

                    id = (bbox_xywh[:, 2] > 0) & (bbox_xywh[:, 3] > 0)

                    bbox_xywh = bbox_xywh[id, :]
                    ebbox_xywh = bbox_xywh.astype(np.float)
                    # ebbox_xywh[:, 3] *= 1.2

                    cls_conf = cls_conf[mask]
                    cls_conf = cls_conf[id]

                    outputs = self.deepsort.update(ebbox_xywh, cls_conf, im)
                    # outputs = []
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]

                        tracklet = [{'bbox_xyxy': [int(x) for x in box], 'id': int(id)} for box, id in zip(bbox_xyxy, identities)]
                        # tracklet = {'bbox_xyxy': [[int(x) for x in box] for box in bbox_xyxy], 'id': [int(id) for id in identities]}
                        self.write_json(tracklet, os.path.join(dir_output_json, os.path.basename(dir_json)))
                    else:
                        bbox_xyxy = bbox_xywh
                        bbox_xyxy[:, :2] = bbox_xywh[:, :2] - bbox_xywh[:, 2:] / 2
                        bbox_xyxy[:, 2:] = bbox_xywh[:, :2] + bbox_xywh[:, 2:]
                        identities = None

            ori_im = draw_bboxes(ori_im, bbox_xyxy, identities=identities)
            ori_im = draw_bboxes0(ori_im, bbox_xyxy2, color=(144,238,144))
            cv2.imwrite(os.path.join(dir_output_img, os.path.basename(dir_image)), ori_im)
            if visualization:
                cv2.imshow('demo', ori_im)
                cv2.waitKey(1)

    def track_by_class(self, dir_images, dir_jsons, dir_output_img=None, dir_output_json=None, dir_cat_json=None,
                       visualization=False, tq_display=False, reverse=False):

        with open(dir_cat_json) as f:
            data = json.load(f)
        cat_ids = [cat['id'] for cat in data['category']]

        list_images = os.listdir(dir_images)
        list_images = sorted(list_images, reverse=reverse)
        list_jsons = [os.path.join(dir_jsons, img.replace('.jpg', '.json')) for img in list_images]
        list_images = [os.path.join(dir_images, img) for img in list_images]

        if dir_output_img:
            os.makedirs(dir_output_img, exist_ok=True)

        if dir_output_json:
            if os.path.exists(dir_output_json):
                shutil.rmtree(dir_output_json)
            os.makedirs(dir_output_json, exist_ok=True)

        if tq_display:
            tq = tqdm.tqdm(total=len(list_images)*len(cat_ids))
        else:
            tq = None

        max_id = 0
        max_id_tmp = 0
        for k, cat_id in enumerate(cat_ids):
            self.reset()
            max_id = max_id_tmp + 1

            for i, (dir_image, dir_json) in enumerate(zip(list_images, list_jsons)):
                if tq is not None:
                    tq.set_description('Video frame: {}'.format(os.path.basename(dir_image)))
                    tq.update(1)

                ori_im = cv2.imread(dir_image)
                im = ori_im[:, :, (2, 1, 0)]

                bbox_xyxy = []
                bbox_xyxy2 = []
                identities = []
                if os.path.isfile(dir_json):
                    bbox_xywh, cls_conf, cls_ids = self.load_json(dir_json)

                    if bbox_xywh is not None:

                        bbox_xyxy2 = np.array(bbox_xywh)
                        if bbox_xyxy2.shape[0] > 0:
                            bbox_xyxy2[:, :2] = bbox_xywh[:, :2] - bbox_xywh[:, 2:] / 2
                            bbox_xyxy2[:, 2:] = bbox_xywh[:, :2] + bbox_xywh[:, 2:] / 2

                            mask = (cls_ids == cat_id)
                            # mask = cls_ids > 0
                            cls_ids = cls_ids[mask]
                            mask = mask.nonzero()
                            bbox_xywh = bbox_xywh[mask]

                            id = (bbox_xywh[:, 2] > 0) & (bbox_xywh[:, 3] > 0)

                            bbox_xywh = bbox_xywh[id, :]
                            ebbox_xywh = bbox_xywh.astype(np.float)
                            # ebbox_xywh[:, 3] *= 1.2

                            cls_conf = cls_conf[mask]
                            cls_conf = cls_conf[id]

                            outputs = self.deepsort.update(ebbox_xywh, cls_conf, im)
                            if len(outputs) > 0:
                                bbox_xyxy = outputs[:, :4]
                                identities = outputs[:, -1]

                                tracklet = []
                                if k > 0 and os.path.isfile(os.path.join(dir_output_json, os.path.basename(dir_json))):
                                    with open(os.path.join(dir_output_json, os.path.basename(dir_json))) as f:
                                        tracklet = json.load(f)

                                identities += max_id
                                max_id_tmp = max(np.max(identities), max_id)

                                tracklet.extend([{'bbox_xyxy': [int(x) for x in box], 'id': int(id)} for box, id in zip(bbox_xyxy, identities)])

                                # if i == 1:
                                #     print([x['id'] for x in tracklet])

                                self.write_json(tracklet, os.path.join(dir_output_json, os.path.basename(dir_json)))
                            else:
                                bbox_xyxy = bbox_xywh
                                bbox_xyxy[:, :2] = bbox_xywh[:, :2] - bbox_xywh[:, 2:] / 2
                                bbox_xyxy[:, 2:] = bbox_xywh[:, :2] + bbox_xywh[:, 2:]
                                identities = None

                if k > 0 and os.path.isfile(os.path.join(dir_output_img, os.path.basename(dir_image))):
                    ori_im = cv2.imread(os.path.join(dir_output_img, os.path.basename(dir_image)))
                ori_im = draw_bboxes(ori_im, bbox_xyxy, identities=identities)
                ori_im = draw_bboxes0(ori_im, bbox_xyxy2, color=(144,238,144))
                cv2.imwrite(os.path.join(dir_output_img, os.path.basename(dir_image)), ori_im)
                if visualization:
                    cv2.imshow('demo', ori_im)
                    cv2.waitKey(0)