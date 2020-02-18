"""
Module for cv2 utility functions and maintaining version compatibility
between 3.x and 4.x
"""
import cv2
import os
import json
from maskrcnn_benchmark.utils.visualization.create_palette import create_palette
from maskrcnn_benchmark.data.categories.ro_categories import CATEGORIES as RO_CATEGORIES
from maskrcnn_benchmark.data.categories.accident_categories import CATEGORIES as AC_CATEGORIES


COLORS  =   [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


def visualize_object_class(dir_image, dir_json, dir_output=None,
                           min_threshold=0, max_threshold=1,
                           thick=5, colors=None,
                           display_caption=True, CATEGORIES=[]):
    if colors is None:
        palette = create_palette()
        colors = {}
        for x in palette:
            colors[x['id']] = x['rgb']

    image = cv2.imread(dir_image)
    if os.path.exists(dir_json):
        with open(dir_json) as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [x for x in data.values()]

        for bbox in data:
            if 'score' not in bbox.keys() or (bbox['score'] >= min_threshold and bbox['score'] <= max_threshold):
                top_left = (int(bbox['bbox']['x']), int(bbox['bbox']['y']))
                bottom_right = (int(bbox['bbox']['x']) + int(bbox['bbox']['w']), int(bbox['bbox']['y']) + int(bbox['bbox']['h']))
                image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(colors[bbox['category_id']]), thick)

            if display_caption and CATEGORIES:
                txt = CATEGORIES[int(bbox['category_id'])]
                t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.putText(image, txt,
                            (int(bbox['bbox']['x'] + bbox['bbox']['w'] / 2 - t_size[0] / 2), int(bbox['bbox']['y'] + t_size[1] + 10)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    if dir_output is not None:
        cv2.imwrite(dir_output, image)
    return image


def visualize_characteristics(dir_image, dir_json, dir_output=None,
                              min_threshold=0, max_threshold=1,
                              thick=5, colors=None,
                              display_caption=True, CATEGORIES=[]):
    if colors is None:
        palette = create_palette()
        colors = {}
        for x in palette:
            colors[x['id']] = x['rgb']

    image = cv2.imread(dir_image)
    if os.path.exists(dir_json):
        with open(dir_json) as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [x for x in data.values()]

        for bbox in data:
            if 'score' not in bbox.keys() or (bbox['score'] >= min_threshold and bbox['score'] <= max_threshold):
                top_left = (int(bbox['bbox']['x']), int(bbox['bbox']['y']))
                bottom_right = (int(bbox['bbox']['x']) + int(bbox['bbox']['w']), int(bbox['bbox']['y']) + int(bbox['bbox']['h']))
                image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right),
                                      tuple(colors[bbox['second_category_id']+1]), thick)

            if display_caption and CATEGORIES:
                txt = CATEGORIES[int(bbox['second_category_id']+1)]
                t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.putText(image, txt,
                            (int(bbox['bbox']['x'] + bbox['bbox']['w'] / 2 - t_size[0] / 2), int(bbox['bbox']['y'] + t_size[1] + 10)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, [255, 255, 255], 2)

    if dir_output is not None:
        cv2.imwrite(dir_output, image)
    return image


def visualize_tracking_id(dir_image, dir_json, dir_output=None,
                          min_threshold=0, max_threshold=1,
                          thick=5, display_caption=True):
    image = cv2.imread(dir_image)
    if os.path.exists(dir_json):
        with open(dir_json) as f:
            data = json.load(f)

        for bbox in data:
            if 'score' not in bbox.keys() or (bbox['score'] >= min_threshold and bbox['score'] <= max_threshold):
                top_left = (int(bbox['bbox']['x']), int(bbox['bbox']['y']))
                bottom_right = (int(bbox['bbox']['x']) + int(bbox['bbox']['w']), int(bbox['bbox']['y']) + int(bbox['bbox']['h']))

                if 'track_id' not in bbox.keys() or int(bbox['track_id']) < 0:
                    color = [0, 0, 0]
                else:
                    color = COLORS[int(bbox['track_id']) % len(COLORS)]

                image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), color, thick)
                if display_caption:
                    txt = '#{}'.format(int(bbox['track_id']))
                    t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.putText(image, txt,
                                (int(bbox['bbox']['x'] + bbox['bbox']['w'] / 2 - t_size[0] / 2), int(bbox['bbox']['y'] + t_size[1] + 10)),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, [255, 255, 255], 2)

    if dir_output is not None:
        cv2.imwrite(dir_output, image)
    return image


def visualize_tracking_object_class(dir_image, dir_json, dir_output=None,
                                    min_threshold=0, max_threshold=1,
                                    thick=5, colors=None,
                                    display_caption=True, CATEGORIES=[]):
    if colors is None:
        palette = create_palette()
        colors = {}
        for x in palette:
            colors[x['id']] = x['rgb']

    image = cv2.imread(dir_image)
    if os.path.exists(dir_json):
        with open(dir_json) as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [x for x in data.values()]

        for bbox in data:
            if 'score' not in bbox.keys() or (bbox['score'] >= min_threshold and bbox['score'] <= max_threshold):
                top_left = (int(bbox['bbox']['x']), int(bbox['bbox']['y']))
                bottom_right = (int(bbox['bbox']['x']) + int(bbox['bbox']['w']), int(bbox['bbox']['y']) + int(bbox['bbox']['h']))
                image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(colors[bbox['category_id']]), thick)

            if display_caption and CATEGORIES:
                txt = '#{} {}'.format(int(bbox['track_id']), CATEGORIES[int(bbox['category_id'])])
                t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.putText(image, txt,
                            (int(bbox['bbox']['x'] + bbox['bbox']['w'] / 2 - t_size[0] / 2), int(bbox['bbox']['y'] + t_size[1] + 10)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    if dir_output is not None:
        cv2.imwrite(dir_output, image)
    return image


def visualize_object_video(dir_image, dir_json, dir_output1, thick=3, first_categories=RO_CATEGORIES):
    if not os.path.exists(dir_output1):
        os.makedirs(dir_output1)

    files = sorted(os.listdir(dir_image))
    for file in files:
        dir_image0 = os.path.join(dir_image, file)
        dir_json0 = os.path.join(dir_json, os.path.splitext(file)[0] + '.json')

        visualize_object_class(dir_image0, dir_json0, dir_output=os.path.join(dir_output1, os.path.splitext(file)[0] + '.jpg'),
                               min_threshold=0, max_threshold=1,
                               thick=thick, colors=None,
                               display_caption=True, CATEGORIES=first_categories)


def visualize_tracking_object_video(dir_image, dir_json, dir_output1, thick=3, first_categories=RO_CATEGORIES):
    if not os.path.exists(dir_output1):
        os.makedirs(dir_output1)

    files = sorted(os.listdir(dir_image))
    for file in files:
        dir_image0 = os.path.join(dir_image, file)
        dir_json0 = os.path.join(dir_json, os.path.splitext(file)[0] + '.json')

        visualize_tracking_object_class(dir_image0, dir_json0, dir_output=os.path.join(dir_output1, os.path.splitext(file)[0] + '.jpg'),
                                        min_threshold=0, max_threshold=1,
                                        thick=thick, colors=None,
                                        display_caption=True, CATEGORIES=first_categories)


def visualize_video(dir_image, dir_json, dir_output1, dir_output2, dir_output3,
                    thick=3, first_categories=RO_CATEGORIES, second_categories=AC_CATEGORIES):
    if not os.path.exists(dir_output1):
        os.makedirs(dir_output1)

    if not os.path.exists(dir_output2):
        os.makedirs(dir_output2)

    if not os.path.exists(dir_output3):
        os.makedirs(dir_output3)

    files = sorted(os.listdir(dir_image))
    for file in files:
        dir_image0 = os.path.join(dir_image, file)
        dir_json0 = os.path.join(dir_json, os.path.splitext(file)[0] + '.json')

        visualize_object_class(dir_image0, dir_json0, dir_output=os.path.join(dir_output1, os.path.splitext(file)[0] + '.jpg'),
                               min_threshold=0, max_threshold=1,
                               thick=thick, colors=None,
                               display_caption=True, CATEGORIES=first_categories)

        visualize_characteristics(dir_image0, dir_json0, dir_output=os.path.join(dir_output2, os.path.splitext(file)[0] + '.jpg'),
                                  min_threshold=0, max_threshold=1,
                                  thick=thick, colors=None,
                                  display_caption=True, CATEGORIES=second_categories)

        visualize_tracking_id(dir_image0, dir_json0, dir_output=os.path.join(dir_output3, os.path.splitext(file)[0] + '.jpg'),
                              min_threshold=0, max_threshold=1,
                              thick=thick, display_caption=True)


if __name__ == '__main__':
    # pass

    # dir_image = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo_dataset/Images/000001"
    # dir_json = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Tracking_ID/Tubelet/Videos/000001"
    # dir_output1 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/temp/Object/000001"
    # dir_output2 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/temp/Accident/000001"
    # dir_output3 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/temp/Tracking/000001"
    # visualize_video(dir_image, dir_json, dir_output1, dir_output2, dir_output3, 3, RO_CATEGORIES, AC_CATEGORIES)

    dir_image = '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_1/Add_instances/Images/20171215_100225_001_Camera1'
    dir_json = '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_1/Add_instances/Videos/20171215_100225_001_Camera1'
    dir_output1 = '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_1/Add_instances/Visualization2/20171215_100225_001_Camera1'
    visualize_tracking_object_video(dir_image, dir_json, dir_output1, thick=3, first_categories=RO_CATEGORIES)

