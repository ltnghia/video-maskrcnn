import os
import tqdm
import ulti
import json
import cv2
import copy
from maskrcnn_benchmark.external.tracking.SOT.SiamDW.run_tracker import init_tracker, run_tracker


def xywh_to_xyxy(rect):
    box = [rect[0],
           rect[1],
           rect[0] + rect[2],
           rect[1] + rect[3]]
    return box


def bb_intersection_over_union(rectA, rectB):
    boxA = xywh_to_xyxy(rectA)
    boxB = xywh_to_xyxy(rectB)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def run_trackers():
    info = ulti.load_json()
    videonames = os.listdir(os.path.join(info['dataset_dir'], 'Images'))
    videonames = sorted(videonames)
    tq = tqdm.tqdm(total=len(videonames))
    alpha = 0.5
    frequency_threshold = 5
    iou_threshold = 0.3
    iou_threshold2 = 0.8
    border = 8
    for videoname in videonames:
        tq.set_description('Video {}'.format(videoname))
        tq.update(1)

        dir_tubelet = os.path.join(info['dataset_dir'], info['experiment'], 'Filter_data', 'Tubelet',
                                   videoname + '.json')

        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Add_instances', 'Videos', videoname))
        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Add_instances', 'Tubelet'))

        with open(dir_tubelet) as f:
            tubelet = json.load(f)
        tubelet_by_frame = tubelet['tubelet_by_frame']
        tubelet_by_id = tubelet['tubelet_by_id']

        filenames = os.listdir(os.path.join(info['dataset_dir'], 'Images', videoname))
        filenames = sorted(filenames)
        frames = []
        for i, frame in enumerate(filenames):
            frame = os.path.splitext(frame)[0]
            frames.append(frame)

        img = cv2.imread(os.path.join(info['dataset_dir'], 'Images', videoname, filenames[0]))
        height, width, channels = img.shape

        for id in list(tubelet_by_id.keys()):
            tubelet = tubelet_by_id[id]
            if len(tubelet.keys()) >= frequency_threshold:
                keys = sorted(list(tubelet.keys()))

                # forward process
                prev_idx = frames.index(keys[0])
                tracker = None
                for idx in range(frames.index(keys[1]), len(frames)):
                    if frames[idx] not in keys:
                        # begin tracking
                        if tracker is None and frames[prev_idx] in tubelet.keys():
                            prev_bbox = tubelet[frames[prev_idx]]['bbox']

                            if prev_bbox['x'] >= width / border and (prev_bbox['x'] + prev_bbox['w']) <= width * (
                                    border - 1) / border:
                                tracker = init_tracker(
                                    init_image_file=os.path.join(info['dataset_dir'], 'Images', videoname,
                                                                 filenames[prev_idx]),
                                    init_rect=[prev_bbox['x'], prev_bbox['y'], prev_bbox['w'], prev_bbox['h']],
                                    device='cuda')
                                prev_rect = [prev_bbox['x'], prev_bbox['y'], prev_bbox['w'], prev_bbox['h']]
                                iou = -1

                        if tracker:
                            rect, tracker = run_tracker(
                                [os.path.join(info['dataset_dir'], 'Images', videoname, filenames[idx])],
                                tracker, visualization=False)
                            rect = rect[0]
                            if iou < 0 or iou >= iou_threshold:
                                iou = bb_intersection_over_union(rect, prev_rect)
                            prev_rect = rect

                            if os.path.splitext(filenames[idx])[0] in tubelet_by_frame.keys():
                                other_boxes = tubelet_by_frame[os.path.splitext(filenames[idx])[0]]
                                other_iou = 0
                                for other_key in other_boxes.keys():
                                    other_box = other_boxes[other_key]['bbox']
                                    other_rect = [other_box['x'], other_box['y'], other_box['w'], other_box['h']]
                                    other_iou = bb_intersection_over_union(rect, other_rect)
                                    if other_iou >= iou_threshold2:
                                        break
                                if other_iou >= iou_threshold2:
                                    break

                            if iou >= iou_threshold:
                                tubelet[frames[idx]] = copy.deepcopy(tubelet[list(tubelet.keys())[0]])
                                tubelet[frames[idx]]['bbox'] = {'x': rect[0],
                                                                'y': rect[1],
                                                                'w': rect[2],
                                                                'h': rect[3]}
                                tubelet[frames[idx]]['track_bbox_xyxy'] = xywh_to_xyxy(rect)
                                tubelet[frames[idx]]['add_by_tracker'] = True

                                if frames[idx] not in tubelet_by_frame.keys():
                                    tubelet_by_frame[frames[idx]] = {}
                                tubelet_by_frame[frames[idx]][id] = tubelet[frames[idx]]
                            else:
                                break
                    else:
                        # stop tracking
                        if tracker:
                            tracker = None
                    prev_idx = idx

        ulti.write_json({'tubelet_by_frame': tubelet_by_frame, 'tubelet_by_id': tubelet_by_id},
                        os.path.join(info['dataset_dir'], info['experiment'], 'Add_instances', 'Tubelet',
                                     videoname + '.json'))

        for frame in tubelet_by_frame.keys():
            tubelet = tubelet_by_frame[frame]
            ulti.write_json(tubelet,
                            os.path.join(info['dataset_dir'], info['experiment'], 'Add_instances', 'Videos', videoname,
                                         frame + '.json'))


if __name__ == "__main__":
    run_trackers()
