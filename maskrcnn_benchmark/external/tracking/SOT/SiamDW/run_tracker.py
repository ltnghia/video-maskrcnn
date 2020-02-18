import os
import numpy as np
import cv2
from easydict import EasyDict as edict

from .lib.models import models
from .lib.tracker.siamfc import SiamFC
from .lib.tracker.siamrpn import SiamRPN
from .lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou


def init_tracker(init_image_file, init_rect,
                 # dir_model='/mnt/sdb/ltnghia/ITS_project/Code/tracking/SOT/SiamDW/snapshot/CIResNet22_RPN.pth',
                 dir_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), "snapshot/CIResNet22_RPN.pth"),
                 device='cuda'):

    info = edict()
    info.arch = 'SiamRPNRes22'
    info.dataset = ''
    info.epoch_test = True

    if 'FC' in info.arch:
        net = models.__dict__[info.arch]()
        tracker = SiamFC(info)
    else:
        net = models.__dict__[info.arch](anchors_nums=5)
        tracker = SiamRPN(info)

    model = load_pretrain(net, dir_model)
    model.eval().to(device)

    lx, ly, w, h = init_rect
    target_pos = np.array([lx + w / 2, ly + h / 2])
    target_sz = np.array([w, h])
    frame = cv2.imread(init_image_file)

    state = tracker.init(frame, target_pos, target_sz, model)  # init tracker

    return tracker, state


def run_tracker(tracking_image_files,
                tracker,
                dir_output=None,
                visualization=False):

    state = tracker[1]
    tracker = tracker[0]

    res = []
    for file in tracking_image_files:  # track
        im = cv2.imread(file)

        state = tracker.track(state, im)  # track
        rect = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

        rect = [int(r) for r in rect]
        res.append(rect)

        if visualization:
            cv2.rectangle(im, tuple(rect), (0, 255, 0), 2)
            cv2.imshow('track', im)
            cv2.waitKey(1)

    # # save result
    if dir_output:
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)
        result_path = os.path.join(dir_output, 'results.txt')
        with open(result_path, 'w') as f:
            for x in res:
                f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')
    return res, [tracker, state]


if __name__ == '__main__':

    dir_input = '/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Images/000001'
    files = os.listdir(dir_input)
    files = sorted(files)
    init_image_file = os.path.join(dir_input, files[0])
    init_rect = [781, 400, 23, 43]  # xywh

    tracking_image_files = [os.path.join(dir_input, f) for f in files[1:]]

    tracker = init_tracker(init_image_file=init_image_file,
                           init_rect=init_rect,
                           device='cuda')

    for file in tracking_image_files:
        res, tracker = run_tracker([file], tracker=tracker, visualization=True)

    print('done')

