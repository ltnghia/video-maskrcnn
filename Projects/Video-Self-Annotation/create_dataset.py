import os
import ulti
import cv2
from glob import glob
import tqdm
import copy
import PIL.Image as Image
import numpy as np


def copy_data():
    info = ulti.load_json()
    dir_images = os.path.join(info['dataset_dir'], '..', 'Raw')
    dir_output = os.path.join(info['dataset_dir'], 'Images')
    videos = os.listdir(dir_images)
    for video in sorted(videos):
        if 'frankfurt' != video:
            pass
            for id in range(1000):
                prex = (video + '_%06d') % id
                files = glob(os.path.join(dir_images, video, prex) + '_*')
                if len(files) > 0:
                    files = sorted(files)
                    if not os.path.exists(os.path.join(dir_output, prex)):
                        os.makedirs(os.path.join(dir_output, prex))
                    for file in files:
                        img = cv2.imread(os.path.join(dir_images, video, file))
                        print(os.path.join(dir_output, prex, os.path.splitext(os.path.basename(file))[0] + '.jpg'))
                        cv2.imwrite(os.path.join(dir_output, prex, os.path.splitext(os.path.basename(file))[0] + '.jpg'), img)
        else:
            for id in range(1000):
                prex = video + '_%06d' % id
                files = glob(os.path.join(dir_images, video, prex) + '_*')
                files = [os.path.splitext(file)[0] for file in files]
                files = list(dict.fromkeys(files))
                if len(files) > 0:
                    files = sorted(files)
                    count = 0
                    for file in files:
                        # print('{} {}'.format(count//30, count%30))
                        new_prex = prex + '_%06d' % ((count) // 30)
                        if (count % 30) == 0:
                            if not os.path.exists(os.path.join(dir_output, new_prex)):
                                os.makedirs(os.path.join(dir_output, new_prex))
                        if os.path.isfile(os.path.join(dir_images, video, file + '.jpg')):
                            img = cv2.imread(os.path.join(dir_images, video, file + '.jpg'))
                        else:
                            img = cv2.imread(os.path.join(dir_images, video, file + '.png'))

                        cv2.imwrite(os.path.join(dir_output, new_prex, os.path.splitext(os.path.basename(file))[0] + '.jpg'), img)
                        files_tmp = os.listdir(os.path.join(dir_output, new_prex))
                        count += 1


def create_attributes():
    info = ulti.load_json()
    output_dir = os.path.join(info['dataset_dir'], 'Categories')
    ulti.make_dir(output_dir)
    info = {}
    info['id'] = 1
    info['name'] = 'Road_Objects'
    category = [{"id": 1, "name": "pedestrian", "type": "thing", "supercategory": "person"},
                {"id": 2, "name": "rider", "type": "thing", "supercategory": "person"},
                {"id": 3, "name": "car", "type": "thing", "supercategory": "vehicle"},
                {"id": 4, "name": "truck", "type": "thing", "supercategory": "vehicle"},
                {"id": 5, "name": "bus", "type": "thing", "supercategory": "vehicle"},
                {"id": 6, "name": "motorcycle", "type": "thing", "supercategory": "vehicle"},
                {"id": 7, "name": "bicycle", "type": "thing", "supercategory": "vehicle"},
                ]
    info['category'] = category
    outfile = ulti.write_json(info, file=os.path.join(output_dir, info['name'] + '.json'))


def create_dataset_info(img_id_start=0, video_id_start=0, ann_id_start=0):
    info = ulti.load_json()
    dataset_dir = os.path.join(info['dataset_dir'])
    dataset_name = info['dataset_name']

    img_id = img_id_start
    video_id = video_id_start
    ann_id = ann_id_start

    dataset = {}
    dataset['id'] = 1

    info = {}
    info['name'] = dataset_name
    info['root_dir'] = dataset_dir + '/'
    info['type'] = 'video'  # 'video' or 'image'
    info['ann_dir'] = ''
    info['extension'] = 'jpg'
    dataset['info'] = info

    videos = []
    data_dir = os.path.join(info['root_dir'], 'Images')
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    for folder in sorted(folders):
        video = {}
        video['id'] = video_id
        video['name'] = folder
        data_dir = os.path.join(info['root_dir'], 'Images', video['name'])
        files = [f for f in os.listdir(data_dir) if
                 os.path.isfile(os.path.join(data_dir, f)) and f.endswith(info['extension'])]
        video['n_frames'] = len(files)
        videos.append(video)
        video_id += 1
    dataset['videos'] = videos

    images = []
    tq = tqdm.tqdm(total=len(videos))
    for video in videos:
        tq.update(1)
        data_dir = os.path.join(info['root_dir'], 'Images', video['name'])
        files = [f for f in os.listdir(data_dir) if
                 os.path.isfile(os.path.join(data_dir, f)) and f.endswith(info['extension'])]
        sub_images = []
        sub_dataset = copy.deepcopy(dataset)
        for f in sorted(files):
            image = {}
            image['id'] = img_id
            image['has_gt'] = False
            image['video_id'] = video['id']
            image['file_name'] = os.path.join('Images', video['name'], f)
            image['seg_file_name'] = ''
            img = Image.open(os.path.join(info['root_dir'], image['file_name'])).convert("RGB")
            img = np.array(img)[:, :, [2, 1, 0]]
            img = img.copy()
            image['width'] = img.shape[1]
            image['height'] = img.shape[0]
            img_id += 1
            images.append(image)
            sub_images.append(image)

            tq.set_description('Video {}'.format(image['file_name']))

        sub_dataset['images'] = sub_images

        dir_output = dataset_dir + '/Info'
        ulti.make_dir(dir_output)
        outfile = ulti.write_json(sub_dataset, file=(dir_output + '/' + video['name'] + '.json'))

    dataset['images'] = images

    dir_output = dataset_dir + '/Info/'
    ulti.make_dir(dir_output)
    outfile = ulti.write_json(dataset, file=(dir_output + '/' + info['name'] + '.json'))


def create_train():
    info = ulti.load_json()
    dir_input = info['dataset_dir']
    dir_output = os.path.join(info['dataset_dir'], 'RCNN_data')
    video = info['annotated_video']
    only_use_true_gt = False
    path = os.path.join(dir_input, 'Info', video + '.json')

    dataset = ulti.load_json(path)
    images = dataset['images']
    videos = dataset['videos']

    categories = ulti.load_json(dir_input + '/Categories/Road_Objects.json')
    categories = categories['category']

    dataset = {'categories': categories, 'annotations': [], 'videos': [], 'images': []}

    video_names = []
    video_ids = []
    if video == info['dataset_name']:
        for vid in videos:
            dataset['videos'].append(vid)
            video_names.append(vid['name'])
            video_ids.append(vid['id'])
    else:
        for vid in videos:
            if vid['name'] == video:
                dataset['videos'].append(vid)
                video_names.append(vid['name'])
                video_ids.append(vid['id'])

    for image in images:
        image['file_name'] = image['file_name'].replace('Images\\\\', '')
        image['file_name'] = image['file_name'].replace('Images/', '')
        image['file_name'] = image['file_name'].replace('Images\\', '')

    ann_files = []
    list_images = []
    annotations = []
    ins_id = 0
    tq = tqdm.tqdm(total=len(video_names))
    for id, video in zip(video_ids, video_names):
        tq.update(1)
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(info['dataset_dir'], info['experiment'], 'Detection', 'Json', video)):
            ann_temp = []
            img_temp = []

            for file in filenames:
                if file.endswith('.json'):
                    data = ulti.load_json(os.path.join(dirpath, file))
                    if len(data) > 0:
                        ann_files.append(file)
                        ann_temp.append(file)

            img_temp = sorted(img_temp)
            ann_temp = sorted(ann_temp)
            for image in images:
                if image['video_id'] == id:
                    file_name = os.path.splitext(os.path.basename(image['file_name']))[0] + '.json'
                    if file_name in ann_temp:
                        if (image['has_gt'] and only_use_true_gt) or not only_use_true_gt:
                            # print(file_name)
                            list_images.append(image)
                            img_temp.append(image)

                            json_file = os.path.splitext(os.path.basename(image['file_name']))[0] + '.json'

                            if json_file in ann_temp:
                                data = ulti.load_json(os.path.join(dirpath, json_file))
                                for ann in data:
                                    ann['bbox'] = [ann['bbox']['x'], ann['bbox']['y'], ann['bbox']['w'], ann['bbox']['h']]
                                    ann['id'] = ins_id
                                    ann['image_id'] = image['id']
                                    ins_id += 1
                                annotations.extend(data)
                                tq.set_description('Video {}'.format(os.path.join(video, file_name)))

    dataset['annotations'] = annotations
    dataset['videos'] = videos
    dataset['images'] = list_images
    ulti.make_dir(dir_output)
    outfile = ulti.write_json(dataset, file=os.path.join(dir_output, 'train.json'))


def create_test():
    info = ulti.load_json()
    dir_input = info['dataset_dir']
    dir_output = os.path.join(info['dataset_dir'], 'RCNN_data')
    video = info['annotated_video']
    exclude_true_gt_frames = False
    path = os.path.join(dir_input, 'Info', video + '.json')
    dataset = ulti.load_json(dir_input + '/Info/' + video + '.json')
    videos = dataset['videos']

    images = []
    if os.path.isfile(path):
        data = ulti.load_json(path)
        if exclude_true_gt_frames:
            for image in data['images']:
                if not image['has_gt']:
                    images.append(image)
        else:
            images.extend(data['images'])

    for image in images:
        image['file_name'] = image['file_name'].replace('Images\\\\', '')
        image['file_name'] = image['file_name'].replace('Images/', '')
        image['file_name'] = image['file_name'].replace('Images\\', '')

    categories = ulti.load_json(dir_input + '/Categories/Road_Objects.json')
    categories = categories['category']
    dataset = {'categories': categories, 'annotations': [], 'videos': [], 'images': []}
    dataset['videos'] = videos
    dataset['images'] = images

    ulti.make_dir(dir_output)
    outfile = ulti.write_json(dataset, file=os.path.join(dir_output, 'test.json'))


if __name__ == "__main__":
    copy_data()
    create_attributes()
    create_dataset_info()
    create_train()
    create_test()
