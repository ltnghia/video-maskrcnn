import os
import json
import cv2
import tqdm


from maskrcnn_benchmark.data.categories.ro_categories import CATEGORIES
from maskrcnn_benchmark.utils.visualization.create_palette import create_palette

root_file = './root_dir.json'


def write_json(info, file=root_file):
    outfile = open(file, 'w')
    x = json.dumps(info)
    outfile.write(x)
    outfile.close()
    return file


def load_json(file=root_file):
    info = json.load(open(file))
    return info


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_folder(input, output, index=0):
    if output and not os.path.exists(output):
        os.makedirs(output)

    files = os.listdir(input)
    for file in sorted(files):
        img = cv2.imread(os.path.join(input, file))
        cv2.imwrite(os.path.join(output, '%06d.jpg' % index), img)
        index += 1
    return index


def copy_model():
    info = load_json()
    from tools import convert_network
    default_dir = os.path.join(info['dataset_dir'], '../Initial_model/initial_model.pth') # pre-train on BDD dataset
    if info['iter'] == 0:
        dir_input = default_dir
    else:
        dir_input = os.path.join(info['training_dir'], info['dataset_name'], 'Models', 'last_checkpoint')
        if not os.path.isfile(dir_input):
            dir_input = ''
        else:
            with open(dir_input) as f:
                dir_input = f.readline()
                if not os.path.isfile(dir_input):
                    dir_input = ''
    dir_output = os.path.join(info['dataset_dir'], info['experiment'], 'Detector', 'Iter{}.pth'.format(info['iter']))
    if not os.path.exists(os.path.dirname(dir_output)):
        os.makedirs(os.path.dirname(dir_output))
    convert_network.convert_pytorch_to_pytorch(dir_input, dir_output)


def visualize(dir_root, display=False, category_file=None):
    info = load_json()
    palette = create_palette()
    colors = {}
    for x in palette:
        colors[x['id']] = x['rgb']

    if category_file is not None and os.path.isfile(category_file):
        data = load_json(category_file)
        data = data['category']
        categories = ['__background']
        categories.extend([cat['name'] for cat in data])
    else:
        categories = CATEGORIES

    videonames = os.listdir(os.path.join(info['dataset_dir'], 'Images'))
    videonames = sorted(videonames)

    tq = tqdm.tqdm(total=len(videonames))

    alpha = 0.6

    for videoname in videonames:
        tq.set_description('Video {}'.format(videoname))
        tq.update(1)

        dir_input = os.path.join(dir_root, videoname)
        dir_output = os.path.join(dir_root, '..', 'Visualization', videoname)

        make_dir(dir_output)

        files = os.listdir(os.path.join(info['dataset_dir'], 'Images', videoname))
        files = sorted(files)

        for file in files:
            filename = os.path.splitext(file)[0]
            ori_im = cv2.imread(os.path.join(info['dataset_dir'], 'Images', videoname, file))
            image = ori_im.copy()

            if os.path.isfile(os.path.join(dir_input, filename + '.json')):
                with open(os.path.join(dir_input, filename + '.json')) as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    data = data.values()

                for bbox in data:
                    top_left = (int(bbox['bbox']['x']), int(bbox['bbox']['y']))
                    bottom_right = (
                    int(bbox['bbox']['x']) + int(bbox['bbox']['w']), int(bbox['bbox']['y']) + int(bbox['bbox']['h']))
                    image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right),
                                          tuple(colors[bbox['category_id']]), 5)

                    # label = categories[int(bbox['category_id'])]
                    # txt = label
                    # t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
                    # image = cv2.rectangle(image, tuple(top_left),
                    #                       tuple((top_left[0] + t_size[0] + 5, top_left[1] + int(t_size[1] * 1) + 10)),
                    #                       tuple(colors[bbox['category_id']]), -1)
                    # cv2.putText(image, txt, (int(top_left[0] + 5), top_left[1] + t_size[1] + 5), cv2.FONT_HERSHEY_PLAIN,
                    #             3, [0, 0, 0], 2)

            cv2.imwrite(os.path.join(dir_output, filename + '.jpg'), image)
            if display:
                image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
                cv2.imshow('demo', image)
                cv2.waitKey(0)


if __name__ == "__main__":
    info = load_json()
    visualize(os.path.join(info['dataset_dir'], info['experiment'], 'Raw_Detection', 'Raw', 'Json'))
