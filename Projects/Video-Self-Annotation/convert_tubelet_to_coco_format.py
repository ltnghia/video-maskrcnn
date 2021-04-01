import json
import os
import ulti
import tqdm


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
    for vid in videos:
        dataset['videos'].append(vid)
        video_names.append(vid['name'])
        video_ids.append(vid['id'])

    dict_images = {}
    for image in images:
        image['file_name'] = image['file_name'].replace('Images\\\\', '')
        image['file_name'] = image['file_name'].replace('Images/', '')
        image['file_name'] = image['file_name'].replace('Images\\', '')
        key = os.path.join(os.path.basename(os.path.dirname(image['file_name'])), os.path.splitext(os.path.basename(image['file_name']))[0])
        dict_images[key] = image

    list_images = []
    annotations = []
    ins_id = 0
    tq = tqdm.tqdm(total=len(video_names))
    for id, video in zip(video_ids, video_names):
        # print(video)
        tq.update(1)

        dir_tubelet = os.path.join(info['dataset_dir'], info['experiment'], 'Add_instances', 'Tubelet', video + '.json')
        with open(dir_tubelet) as f:
            tubelet = json.load(f)
        tubelet_by_frame = tubelet['tubelet_by_frame']
        tubelet_by_id = tubelet['tubelet_by_id']

        for filename in tubelet_by_frame.keys():
            tubelet = tubelet_by_frame[filename]
            if len(tubelet.keys()) > 0:
                image = dict_images[os.path.join(video, filename)]
                if (image['has_gt'] and only_use_true_gt) or not only_use_true_gt:
                    list_images.append(image)

                    for key in tubelet.keys():
                        ann = tubelet[key]
                        ann['bbox'] = [ann['bbox']['x'], ann['bbox']['y'], ann['bbox']['w'], ann['bbox']['h']]
                        ann['id'] = ins_id
                        ann['image_id'] = image['id']
                        ins_id += 1
                        annotations.append(ann)
                    tq.set_description('Video {}'.format(os.path.join(video, filename)))
    dataset['annotations'] = annotations
    dataset['videos'] = videos
    dataset['images'] = list_images
    print('train: ', len(list_images))
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
    # pprint(dataset)
    videos = dataset['videos']
    images = []
    path = dir_input + '/Info/' + video + '.json'
    if os.path.isfile(path):
        data = ulti.load_json(path)
        if exclude_true_gt_frames:
            for image in data['images']:
                if not image['has_gt']:
                    images.append(image)
        else:
            images.extend(data['images'])

    tq = tqdm.tqdm(total=len(images))
    for image in images:
        tq.update(1)
        image['file_name'] = image['file_name'].replace('Images\\\\', '')
        image['file_name'] = image['file_name'].replace('Images/', '')
        image['file_name'] = image['file_name'].replace('Images\\', '')

    categories = ulti.load_json(dir_input + '/Categories/Road_Objects.json')
    categories = categories['category']
    dataset = {'categories': categories, 'annotations': [], 'videos': [], 'images': []}
    dataset['videos'] = videos
    dataset['images'] = images
    print('test: ', len(images))
    # pprint(dataset)
    ulti.make_dir(dir_output)
    outfile = ulti.write_json(dataset, file=os.path.join(dir_output, 'test.json'))


if __name__ == "__main__":
    create_train()
    create_test()
