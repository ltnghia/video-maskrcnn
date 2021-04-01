import os
import tqdm
import ulti
import json
import cv2


def check_condition(bbox, height, width):
    h_threshold = 50
    s_threshold2 = 0.02

    if (bbox['bbox']['y'] + bbox['bbox']['h']) >= (height-h_threshold) and \
       (bbox['bbox']['w'] * bbox['bbox']['h'] / height / width) < s_threshold2:
        return None
    return bbox


def filter_data():
    info = ulti.load_json()
    videonames = os.listdir(os.path.join(info['dataset_dir'], 'Images'))
    videonames = sorted(videonames)
    tq = tqdm.tqdm(total=len(videonames))
    frequency_threshold = 4
    for videoname in videonames:
        tq.set_description('Video {}'.format(videoname))
        tq.update(1)
        dir_tubelet = os.path.join(info['dataset_dir'], info['experiment'], 'Smooth_label', 'Tubelet',
                                   videoname + '.json')
        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Filter_data', 'Videos', videoname))
        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Filter_data', 'Tubelet'))
        ulti.make_dir(os.path.join(info['dataset_dir'], info['experiment'], 'Filter_data', 'Visualization', videoname))

        with open(dir_tubelet) as f:
            tubelet = json.load(f)
        tubelet_by_frame = tubelet['tubelet_by_frame']
        tubelet_by_id = tubelet['tubelet_by_id']

        for id in list(tubelet_by_id.keys()):
            tubelet = tubelet_by_id[id]
            if len(tubelet.keys()) < frequency_threshold:
                for frame in tubelet.keys():
                    del tubelet_by_frame[frame][id]
                del tubelet_by_id[id]
                continue

            for frame in list(tubelet.keys()):
                img = cv2.imread(os.path.join(info['dataset_dir'], 'Images', videoname, frame + '.jpg'))
                height, width, channels = img.shape

                bbox = tubelet[frame]
                bbox = check_condition(bbox, height, width)

                if bbox is None:
                    del tubelet_by_id[id][frame]
                    del tubelet_by_frame[frame][id]

                if len(tubelet_by_frame[frame].keys()) == 0:
                    del tubelet_by_frame[frame]

            if len(tubelet_by_id[id].keys()) == 0:
                del tubelet_by_id[id]

        ulti.write_json({'tubelet_by_frame': tubelet_by_frame, 'tubelet_by_id': tubelet_by_id},
                        os.path.join(info['dataset_dir'], info['experiment'], 'Filter_data', 'Tubelet',
                                     videoname + '.json'))

        for frame in tubelet_by_frame.keys():
            tubelet = tubelet_by_frame[frame]
            ulti.write_json(tubelet,
                            os.path.join(info['dataset_dir'], info['experiment'], 'Filter_data', 'Videos', videoname,
                                         frame + '.json'))


if __name__ == "__main__":
    filter_data()


