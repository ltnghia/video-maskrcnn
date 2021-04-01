import os
import ulti
import tqdm


def generate_ann(threshold=None, trackable_threshold=0):

    if threshold is None:
        threshold = [0.85, 0.550, 0.950, 0.950, 0.800, 0.800, 0.700]

    info = ulti.load_json()
    video = info['annotated_video']
    dataset_dir = info['dataset_dir']

    output_dir = os.path.join(dataset_dir, info['experiment'], 'Detection')
    ulti.make_dir(output_dir)

    print(output_dir)

    input_path = os.path.join(dataset_dir, info['experiment'], 'Raw_Detection', 'bbox.json')

    prediction = ulti.load_json(input_path)
    category = ulti.load_json(os.path.join(dataset_dir, 'Categories', 'Road_Objects.json'))
    dataset = ulti.load_json(os.path.join(dataset_dir, 'Info', video + '.json'))

    list_img_id_dataset = []
    list_img_id_prediction = []
    ann_id = 0
    for ann in prediction:
        list_img_id_prediction.append(ann['image_id'])

    tq = tqdm.tqdm(total=len(dataset['images']))
    for img in dataset['images']:
        list_img_id_dataset.append(img['id'])
        tq.set_description('Video Frame {}'.format(img['id']))
        tq.update(1)
        index = [i for i, x in enumerate(list_img_id_prediction) if
                 x == img['id'] and prediction[i]['score'] >= threshold[prediction[i]['category_id'] - 1]]
        if len(index) > 0:
            # print(img['id'], index)
            annotations = []
            for ind in index:
                ann = {}
                ann['id'] = ann_id
                ann['image_id'] = prediction[ind]['image_id']
                ann['track_id'] = -1
                ann['category_id'] = prediction[ind]['category_id']
                ann['score'] = prediction[ind]['score']
                ann['second_category_id'] = None
                ann['second_score'] = 0
                ann['third_category_id'] = None
                ann['third_score'] = 0
                ann['iscrowd'] = 0  # always 0
                ann['bbox'] = {
                    'x': int(prediction[ind]['bbox'][0]),
                    'y': int(prediction[ind]['bbox'][1]),
                    'w': int(prediction[ind]['bbox'][2]),
                    'h': int(prediction[ind]['bbox'][3])
                }
                if ann['bbox']['w'] * ann['bbox']['h'] >= trackable_threshold:
                    ann['trackable'] = True
                else:
                    ann['trackable'] = False
                ann['segmentation'] = []
                ann['area'] = 0
                ann['score'] = prediction[ind]['score']
                annotations.append(ann)
                ann_id += 1
            if len(annotations) > 0:
                fullpath = os.path.join(dataset_dir, 'Annotations', category['name'],
                                        os.path.basename(os.path.dirname(img['file_name'])))
                ulti.make_dir(fullpath)
                fullpath = os.path.join(output_dir, 'Json', os.path.basename(os.path.dirname(img['file_name'])))
                ulti.make_dir(fullpath)
                path = os.path.join(fullpath, os.path.splitext(os.path.basename(img['file_name']))[0] + '.json')
                ulti.write_json(annotations, path)


if __name__ == "__main__":
    generate_ann()






