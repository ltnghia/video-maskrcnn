import os
import torch
import numpy as np
import json
import torch.multiprocessing as multiprocessing
import time
import cv2

from PIL import Image, ImageDraw
from collections import defaultdict

# panoptic visualization
vis_panoptic = False
log = True


class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'iou': iou, 'tp': tp, 'fp': fp,
                                        'fn': fn}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


class PanopticMetrics():

    def evaluate_panoptic(self, pred_pans_2ch, output_dir, path):

        from lib.dataset.panopticapi.utils import IdGenerator

        def get_gt(pan_gt_json_file=None, pan_gt_folder=None):
            if pan_gt_json_file is None:
                pan_gt_json_file = self.panoptic_json_file
            if pan_gt_folder is None:
                pan_gt_folder = self.panoptic_gt_folder

            with open(pan_gt_json_file, 'r') as f:
                pan_gt_json = json.load(f)
            files = [item['file_name'] for item in pan_gt_json['images']]
            cpu_num = 1  # multiprocessing.cpu_count()
            files_split = np.array_split(files, cpu_num)
            # workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            pan_gt_all = []
            for proc_id, files_set in enumerate(files_split):
                p = PanopticMetrics._load_image_single_core(proc_id, files_set, pan_gt_folder)
                # p = workers.apply_async(PanopticMetrics._load_image_single_core, (proc_id, files_set, pan_gt_folder))
                processes.append(p)
            # workers.close()
            # workers.join()
            pan_gt_all = []
            for p in processes:
                # pan_gt_all.extend(p.get())
                pan_gt_all.extend(p)
            categories = pan_gt_json['categories']
            categories = {el['id']: el for el in categories}
            color_generator = IdGenerator(categories)

            return pan_gt_all, pan_gt_json, categories, color_generator

        def get_pred(pan_2ch_all, color_gererator, cpu_num=None):
            if cpu_num is None:
                cpu_num = 1  # multiprocessing.cpu_count()
            pan_2ch_split = np.array_split(pan_2ch_all, cpu_num)
            # workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
                # p = workers.apply_async(PanopticMetrics._converter_2ch_single_core, (proc_id, pan_2ch_set, color_gererator))
                p = PanopticMetrics._converter_2ch_single_core(proc_id, pan_2ch_set, color_gererator)
                processes.append(p)
            # workers.close()
            # workers.join()
            annotations, pan_all = [], []
            for p in processes:
                # p = p.get()
                annotations.extend(p[0])
                pan_all.extend(p[1])
            pan_json = {'annotations': annotations}
            return pan_all, pan_json

        def save_image(images, save_folder, gt_json, colors=None):
            os.makedirs(save_folder, exist_ok=True)
            names = [os.path.join(save_folder,
                                  item['file_name'].replace('_leftImg8bit', '').replace('jpg', 'png').replace('jpeg',
                                                                                                              'png'))
                     for item in gt_json['images']]
            cpu_num = 1  # multiprocessing.cpu_count()
            images_split = np.array_split(images, cpu_num)
            names_split = np.array_split(names, cpu_num)
            # workers = multiprocessing.Pool(processes=cpu_num)
            for proc_id, (images_set, names_set) in enumerate(zip(images_split, names_split)):
                PanopticMetrics._save_image_single_core(proc_id, images_set, names_set, colors)
                # workers.apply_async(PanopticMetrics._save_image_single_core, (proc_id, images_set, names_set, colors))
            # workers.close()
            # workers.join()

        def pq_compute(gt_jsons, pred_jsons, gt_pans, pred_pans, categories):
            start_time = time.time()
            # from json and from numpy
            gt_image_jsons = gt_jsons['images']
            gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
            cpu_num = 1  # multiprocessing.cpu_count()
            gt_jsons_split, pred_jsons_split = np.array_split(gt_jsons, cpu_num), np.array_split(pred_jsons, cpu_num)
            gt_pans_split, pred_pans_split = np.array_split(gt_pans, cpu_num), np.array_split(pred_pans, cpu_num)
            gt_image_jsons_split = np.array_split(gt_image_jsons, cpu_num)

            # workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, (gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set) in enumerate(
                    zip(gt_jsons_split, pred_jsons_split, gt_pans_split, pred_pans_split, gt_image_jsons_split)):
                # p = workers.apply_async(PanopticMetrics._pq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories))
                p = PanopticMetrics._pq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set,
                                                            pred_pans_set, gt_image_jsons_set, categories)
                processes.append(p)
            # workers.close()
            # workers.join()
            pq_stat = PQStat()
            for p in processes:
                pq_stat += p
                # pq_stat += p.get()
            metrics = [("All", None), ("Things", True), ("Stuff", False)]
            results = {}
            for name, isthing in metrics:
                results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
                if name == 'All':
                    results['per_class'] = per_class_results

            if log:
                # pq, sq, rq = [], [], []
                # for idx, result in results['per_class'].items():
                #    pq.append(100 * result['pq'])
                #    sq.append(100 * result['sq'])
                #    rq.append(100 * result['rq'])
                # print("PQ", pq)
                # print("SQ", sq)
                # print("RQ", rq)

                print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
                print("-" * (10 + 7 * 4))
                for name, _isthing in metrics:
                    print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(name, 100 * results[name]['pq'],
                                                                           100 * results[name]['sq'],
                                                                           100 * results[name]['rq'],
                                                                           results[name]['n']))

                print("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}".format("IDX", "PQ", "SQ", "RQ", "IoU",
                                                                                       "TP", "FP", "FN"))
                for idx, result in results['per_class'].items():
                    print("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}".format(idx, 100 * result['pq'],
                                                                                             100 * result['sq'],
                                                                                             100 * result['rq'],
                                                                                             result['iou'],
                                                                                             result['tp'],
                                                                                             result['fp'],
                                                                                             result['fn']))

            t_delta = time.time() - start_time
            print("Time elapsed: {:0.2f} seconds".format(t_delta))
            return results

        # if eval for test0-dev, since there is no gt we simply retrieve image names from image_info json files
        # with open(self.panoptic_json_file, 'r') as f:
        #     gt_json = json.load(f)
        #     gt_json['images'] = sorted(gt_json['images'], key=lambda x: x['id'])
        # other wise:

        pan_gt_json_file = path.replace('val2017', 'annotations/panoptic_val2017_rerange.json').replace('overfit2017',
                                                                                                        'annotations/panoptic_overfit2017.json').replace(
            'val_small2017', 'annotations/panoptic_val_small2017_rerange.json')
        pan_gt_folder = path.replace('val2017', 'annotations/panoptic_val2017').replace('overfit2017',
                                                                                        'annotations/panoptic_overfit2017').replace(
            'val_small2017', 'annotations/panoptic_val_small2017')
        print("Get ground truth")
        gt_pans, gt_json, categories, color_generator = get_gt(pan_gt_json_file, pan_gt_folder)
        print("Get predictions")
        pred_pans, pred_json = get_pred(pred_pans_2ch, color_generator)

        print("Save images")
        save_image(pred_pans_2ch, os.path.join(output_dir, 'pan_2ch'), gt_json)
        save_image(pred_pans, os.path.join(output_dir, 'pan'), gt_json)
        print("Dump JSON file")
        json.dump(gt_json, open(os.path.join(output_dir, 'gt.json'), 'w'))
        json.dump(pred_json, open(os.path.join(output_dir, 'pred.json'), 'w'))
        print("Compute PQ")
        results = pq_compute(gt_json, pred_json, gt_pans, pred_pans, categories)

        return results

    def get_unified_pan_result(self, segs, pans, cls_inds, stuff_area_limit=4 * 64 * 64):
        pred_pans_2ch = []

        for (seg, pan, cls_ind) in zip(segs, pans, cls_inds):
            pan_seg = pan.copy()
            pan_ins = pan.copy()
            id_last_stuff = 133 - 81
            # id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes
            ids = np.unique(pan)
            ids_ins = ids[ids > id_last_stuff]
            pan_ins[pan_ins <= id_last_stuff] = 0
            for idx, id in enumerate(ids_ins):
                region = (pan_ins == id)
                if id == 255:
                    pan_seg[region] = 255
                    pan_ins[region] = 0
                    continue
                cls, cnt = np.unique(seg[region], return_counts=True)
                if cls[np.argmax(cnt)] == cls_ind[id - id_last_stuff - 1] + id_last_stuff:
                    pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                    pan_ins[region] = idx + 1
                else:
                    if np.max(cnt) / np.sum(cnt) >= 0.5 and cls[np.argmax(cnt)] <= id_last_stuff:
                        pan_seg[region] = cls[np.argmax(cnt)]
                        pan_ins[region] = 0
                    else:
                        pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                        pan_ins[region] = idx + 1

            idx_sem = np.unique(pan_seg)
            for i in range(idx_sem.shape[0]):
                if idx_sem[i] <= id_last_stuff:
                    area = pan_seg == idx_sem[i]
                    if (area).sum() < stuff_area_limit:
                        pan_seg[area] = 255

            pan_2ch = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)
            pan_2ch[:, :, 0] = pan_seg
            pan_2ch[:, :, 1] = pan_ins
            pred_pans_2ch.append(pan_2ch)
        return pred_pans_2ch

    @staticmethod
    def _load_image_single_core(proc_id, files_set, folder):
        images = []
        for working_idx, file in enumerate(files_set):
            try:
                file = file.replace("jpg", "png")
                image = np.array(Image.open(os.path.join(folder, file)))
                images.append(image)
            except Exception:
                pass
        return images

    @staticmethod
    def _converter_2ch_single_core(proc_id, pan_2ch_set, color_generator):
        from lib.dataset.panopticapi.utils import rgb2id
        OFFSET = 1000
        VOID = 255
        annotations, pan_all = [], []
        for idx in range(len(pan_2ch_set)):
            pan_2ch = np.uint32(pan_2ch_set[idx])
            pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 1]
            pan_format = np.zeros((pan_2ch.shape[0], pan_2ch.shape[1], 3), dtype=np.uint8)

            l = np.unique(pan)
            segm_info = []
            for el in l:
                sem = el // OFFSET
                if sem == VOID:
                    continue
                mask = pan == el
                if vis_panoptic:
                    color = color_generator.categories[sem]['color']
                else:
                    color = color_generator.get_color(sem)
                pan_format[mask] = color
                index = np.where(mask)
                x = index[1].min()
                y = index[0].min()
                width = index[1].max() - x
                height = index[0].max() - y
                segm_info.append({"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)),
                                  "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()})
            annotations.append({"segments_info": segm_info})
            if vis_panoptic:
                pan_format = Image.fromarray(pan_format)
                draw = ImageDraw.Draw(pan_format)
                for el in l:
                    sem = el // OFFSET
                    if sem == VOID:
                        continue
                    if color_generator.categories[sem]['isthing'] and el % OFFSET != 0:
                        mask = ((pan == el) * 255).astype(np.uint8)
                        _, contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for c in contour:
                            c = c.reshape(-1).tolist()
                            if len(c) < 4:
                                print('warning: invalid contour')
                                continue
                            draw.line(c, fill='white', width=2)
                pan_format = np.array(pan_format)
            pan_all.append(pan_format)
        return annotations, pan_all

    @staticmethod
    def _pq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set,
                                categories):
        OFFSET = 256 * 256 * 256
        VOID = 0
        pq_stat = PQStat()
        for idx, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(
                zip(gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set)):
            # if idx % 100 == 0:
            #     logger.info('Compute pq -> Core: {}, {} from {} images processed'.format(proc_id, idx, len(gt_jsons_set)))
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256

            gt_segms = {el['id']: el for el in gt_json['segments_info']}
            pred_segms = {el['id']: el for el in pred_json['segments_info']}

            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError(
                        'In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(
                            gt_ann['image_id'], label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(
                        gt_ann['image_id'], label, pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(
                        gt_ann['image_id'], list(pred_labels_set)))

            # confusion matrix calculation
            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()
            tp = 0
            fp = 0
            fn = 0

            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                if gt_label not in gt_segms:
                    continue
                if pred_label not in pred_segms:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    continue

                union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                    (VOID, pred_label), 0)
                iou = intersection / union
                if iou > 0.5:
                    pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    tp += 1

            # count false positives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                pq_stat[gt_info['category_id']].fn += 1
                fn += 1

            # count false positives
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                pq_stat[pred_info['category_id']].fp += 1
                fp += 1
        # logger.info('Compute pq -> Core: {}, all {} images processed'.format(proc_id, len(gt_jsons_set)))
        return pq_stat

    @staticmethod
    def _save_image_single_core(proc_id, images_set, names_set, colors=None):
        def colorize(gray, palette):
            # gray: numpy array of the label and 1*3N size list palette
            color = Image.fromarray(gray.astype(np.uint8)).convert('P')
            color.putpalette(palette)
            return color

        for working_idx, (image, name) in enumerate(zip(images_set, names_set)):
            if colors is not None:
                image = colorize(image, colors)
            else:
                image = Image.fromarray(image)
            os.makedirs(os.path.dirname(name), exist_ok=True)
            image.save(name)