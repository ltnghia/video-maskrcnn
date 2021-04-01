import torch
import torchvision
import os
import tqdm
import ulti
import json
import numpy as np


def init_tracklet():
    info = ulti.load_json()
    ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Tracklet'))
    videonames = os.listdir(os.path.join(info['dataset_dir'], 'Images'))
    videonames = sorted(videonames)
    tq = tqdm.tqdm(total=len(videonames))
    for videoname in videonames:
        tq.set_description('Video {}'.format(videoname))
        tq.update(1)

        dir_track = os.path.join(info['dataset_dir'], info['experiment'], 'Track', 'DeepSort', 'Json', videoname)
        dir_track_reverse = os.path.join(info['dataset_dir'], info['experiment'], 'Track', 'DeepSort_Reverse', 'Json', videoname)

        tracklet_by_id = {}
        tracklet_by_frame = {}
        list_images = os.listdir(dir_track)
        list_images = sorted(list_images)
        for i, image in enumerate(list_images):
            filename = os.path.splitext(image)[0]
            if filename not in tracklet_by_frame.keys():
                tracklet_by_frame[filename] = {}
            with open(os.path.join(dir_track, image)) as f:
                data = json.load(f)
                for x in data:
                    tracklet_by_frame[filename][x['id']] = {}
                    tracklet_by_frame[filename][x['id']]['track_bbox_xyxy'] = x['bbox_xyxy']
                    if x['id'] not in tracklet_by_id.keys():
                        tracklet_by_id[x['id']] = {}
                    tracklet_by_id[x['id']][filename] = {}
                    tracklet_by_id[x['id']][filename]['track_bbox_xyxy'] = x['bbox_xyxy']
        tracklet_by_id = dict(sorted(tracklet_by_id.items()))
        for key in tracklet_by_id.keys():
            tracklet_by_id[key] = dict(sorted(tracklet_by_id[key].items()))
        tracklet_by_frame = dict(sorted(tracklet_by_frame.items()))
        for key in tracklet_by_frame.keys():
            tracklet_by_frame[key] = dict(sorted(tracklet_by_frame[key].items()))

        tracklet_by_id2 = {}
        tracklet_by_frame2 = {}
        list_reverse_images = os.listdir(dir_track_reverse)
        list_reverse_images = sorted(list_reverse_images)
        for i, image in enumerate(list_reverse_images):
            filename = os.path.splitext(image)[0]
            if filename not in tracklet_by_frame2.keys():
                tracklet_by_frame2[filename] = {}
            with open(os.path.join(dir_track_reverse, image)) as f:
                data = json.load(f)
                for x in data:
                    tracklet_by_frame2[filename][x['id']] = {}
                    tracklet_by_frame2[filename][x['id']]['track_bbox_xyxy'] = x['bbox_xyxy']
                    if x['id'] not in tracklet_by_id2.keys():
                        tracklet_by_id2[x['id']] = {}
                    tracklet_by_id2[x['id']][filename] = {}
                    tracklet_by_id2[x['id']][filename]['track_bbox_xyxy'] = x['bbox_xyxy']
            if image in list_images:
                break
        tracklet_by_id2 = dict(sorted(tracklet_by_id2.items()))
        for key in tracklet_by_id2.keys():
            tracklet_by_id2[key] = dict(sorted(tracklet_by_id2[key].items()))
        tracklet_by_frame2 = dict(sorted(tracklet_by_frame2.items()))
        for key in tracklet_by_frame2.keys():
            tracklet_by_frame2[key] = dict(sorted(tracklet_by_frame2[key].items()))

        if len(list(tracklet_by_frame.keys())) > 0 and len(list(tracklet_by_frame2.keys())) > 0:

            boxes1 = np.zeros((len(tracklet_by_frame[list(tracklet_by_frame.keys())[0]]), 4))
            for i, box in enumerate(list(tracklet_by_frame[list(tracklet_by_frame.keys())[0]].items())):
                boxes1[i, :] = box[1]['track_bbox_xyxy']

            boxes2 = np.zeros((len(tracklet_by_frame2[list(tracklet_by_frame2.keys())[-1]]), 4))
            for i, box in enumerate(list(tracklet_by_frame2[list(tracklet_by_frame2.keys())[-1]].items())):
                boxes2[i, :] = box[1]['track_bbox_xyxy']

            iou = torchvision.ops.box_iou(torch.from_numpy(boxes1), torch.from_numpy(boxes2))
            iou = iou.detach().numpy()
            iou_threshold = 0.5
            iou[iou < iou_threshold] = 0

            max_iou = 1
            while max_iou != 0:
                max_iou = np.amax(iou)

                if max_iou == 0:
                    break
                (i, j) = np.where(iou == max_iou)
                i = i[0]
                j = j[0]
                # i  boxes1 tracklet_by_frame[list(tracklet_by_frame.keys())[0]]
                # j  boxes2 tracklet_by_frame2[list(tracklet_by_frame2.keys())[-1]]
                id1 = list(tracklet_by_frame[list(tracklet_by_frame.keys())[0]].keys())[i]
                id2 = list(tracklet_by_frame2[list(tracklet_by_frame2.keys())[-1]].keys())[j]

                # if videoname == '000017' and id1 == 4:
                #     print(videoname, max_iou, id1, id2)

                for k in tracklet_by_id2[id2].keys():
                    tracklet_by_id[id1][k] = tracklet_by_id2[id2][k]
                    tracklet_by_id2[id2][k] = None

                for k in tracklet_by_frame2.keys():
                    if k not in tracklet_by_frame.keys():
                        tracklet_by_frame[k] = {}
                    if id2 in tracklet_by_frame2[k].keys():
                        tracklet_by_frame[k][id1] = tracklet_by_frame2[k][id2]
                        tracklet_by_frame2[k][id2] = None

                iou[i, :] = 0
                iou[:, j] = 0

            tracklet_by_id = dict(sorted(tracklet_by_id.items()))
            for key in tracklet_by_id.keys():
                tracklet_by_id[key] = dict(sorted(tracklet_by_id[key].items()))
            tracklet_by_frame = dict(sorted(tracklet_by_frame.items()))
            for key in tracklet_by_frame.keys():
                tracklet_by_frame[key] = dict(sorted(tracklet_by_frame[key].items()))

        ulti.write_json({'tracklet_by_id': tracklet_by_id, 'tracklet_by_frame': tracklet_by_frame},
                        file=os.path.join(info['dataset_dir'], info['experiment'], 'Tracklet', videoname + '.json'))


def create_tubelet():
    info = ulti.load_json()
    ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Tubelet', 'Tubelet'))
    videonames = os.listdir(os.path.join(info['dataset_dir'], 'Images'))
    videonames = sorted(videonames)
    tq = tqdm.tqdm(total=len(videonames))
    for videoname in videonames:
        tq.set_description('Video {}'.format(videoname))
        tq.update(1)

        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Tubelet', 'Videos', videoname))
        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Tubelet', 'Visualization', videoname))

        dir_tracklet = os.path.join(info['dataset_dir'], info['experiment'], 'Tracklet', videoname + '.json')
        dir_detection = os.path.join(info['dataset_dir'], info['experiment'], 'Detection', 'Json', videoname)

        with open(dir_tracklet) as f:
            tracklet_data = json.load(f)
        tracklet_by_id = tracklet_data['tracklet_by_id']
        tracklet_by_frame = tracklet_data['tracklet_by_frame']

        tubelet_by_frame = {}
        tubelet_by_id = {}

        for frame in tracklet_by_frame.keys():
            tubelet_data = []
            with open(os.path.join(dir_detection, frame + '.json')) as f:
                detection_data = json.load(f)
            detection_bbox = np.zeros((len(detection_data), 4))
            for i, box in enumerate(detection_data):
                detection_bbox[i, 0] = box['bbox']['x']
                detection_bbox[i, 1] = box['bbox']['y']
                detection_bbox[i, 2] = box['bbox']['x'] + box['bbox']['w']
                detection_bbox[i, 3] = box['bbox']['y'] + box['bbox']['h']
            track_bbox = np.zeros((len(tracklet_by_frame[frame]), 4))
            for i, key in enumerate(tracklet_by_frame[frame].keys()):
                box = tracklet_by_frame[frame][key]
                track_bbox[i, :] = box['track_bbox_xyxy']

            if detection_bbox.shape[0] == 0 or track_bbox.shape[0] == 0:
                continue
            iou = torchvision.ops.box_iou(torch.from_numpy(detection_bbox), torch.from_numpy(track_bbox))
            iou = iou.detach().numpy()
            iou_threshold = 0.7
            iou[iou < iou_threshold] = 0

            max_iou = 1
            while max_iou != 0:
                max_iou = np.amax(iou)

                if max_iou == 0:
                    break
                (i, j) = np.where(iou == max_iou)
                i = i[0]
                j = j[0]

                detection_data[i]['track_id'] = list(tracklet_by_frame[frame].keys())[j]
                detection_data[i]['track_bbox_xyxy'] = tracklet_by_frame[frame][detection_data[i]['track_id']]['track_bbox_xyxy']
                tubelet_data.append(detection_data[i])

                iou[i, :] = 0
                iou[:, j] = 0

            if len(tubelet_data) > 0:
                ulti.write_json(tubelet_data, os.path.join(info['dataset_dir'], info['experiment'], 'Tubelet', 'Videos', videoname, frame + '.json'))
                tubelet_by_frame[frame] = {}
                for x in tubelet_data:
                    tubelet_by_frame[frame][x['track_id']] = x
                    if x['track_id'] not in tubelet_by_id:
                        tubelet_by_id[x['track_id']] = {}
                    tubelet_by_id[x['track_id']][frame] = x
        ulti.write_json({'tubelet_by_frame': tubelet_by_frame, 'tubelet_by_id': tubelet_by_id},
                        os.path.join(info['dataset_dir'], info['experiment'], 'Tubelet', 'Tubelet', videoname + '.json'))


def most_frequent(ids, weights=None):
    if weights and len(weights) == len(ids):
        weights = np.array(weights)
        ids = np.array(ids)
        val = np.unique(ids)
        freq = {}
        x_max = 0
        y_max = 0
        y_mean = 0
        for key in val:
            x = ids[ids == key]
            y = weights[ids == key]
            freq[key] = np.sum(y)
            if y_max < freq[key]:
                y_max = freq[key]
                x_max = int(key)
                y_mean = np.mean(y)
        return x_max, y_mean
    else:
        return max(set(ids), key=ids.count), None


def smoothen_label():
    info = ulti.load_json()
    videonames = os.listdir(os.path.join(info['dataset_dir'], 'Images'))
    videonames = sorted(videonames)
    tq = tqdm.tqdm(total=len(videonames))
    for videoname in videonames:
        tq.set_description('Video {}'.format(videoname))
        tq.update(1)

        dir_tubelet = os.path.join(info['dataset_dir'], info['experiment'], 'Tubelet', 'Tubelet', videoname + '.json')

        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Smooth_label', 'Videos', videoname))
        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Smooth_label', 'Tubelet'))
        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Smooth_label', 'Visualization', videoname))

        with open(dir_tubelet) as f:
            tubelet = json.load(f)
        tubelet_by_frame = tubelet['tubelet_by_frame']
        tubelet_by_id = tubelet['tubelet_by_id']

        for id in tubelet_by_id.keys():
            tubelet = tubelet_by_id[id]
            category_ids = []
            category_scores = []
            for frame in tubelet.keys():
                bbox = tubelet[frame]
                category_ids.append(bbox['category_id'])
                category_scores.append(bbox['score'])
            # if len(np.unique(np.array(category_ids))) > 1:
            element, score = most_frequent(category_ids, category_scores)
            # print(frame, id, element, ':', category_ids)
            for frame in tubelet.keys():
                if score:
                    tubelet_by_id[id][frame]['score'] = score
                    tubelet_by_frame[frame][id]['score'] = score
                tubelet_by_id[id][frame]['category_id'] = element
                tubelet_by_frame[frame][id]['category_id'] = element

        ulti.write_json({'tubelet_by_frame': tubelet_by_frame, 'tubelet_by_id': tubelet_by_id},
                        os.path.join(info['dataset_dir'], info['experiment'], 'Smooth_label', 'Tubelet', videoname + '.json'))

        for frame in tubelet_by_frame.keys():
            tubelet = tubelet_by_frame[frame]
            ulti.write_json(tubelet,
                            os.path.join(info['dataset_dir'], info['experiment'], 'Smooth_label', 'Videos', videoname, frame + '.json'))


if __name__ == "__main__":
    init_tracklet()
    create_tubelet()
    smoothen_label()
