import os
import imageio


def image2gif(dir_input, dir_output):
    # if not os.path.exists(dir_output):
    #     os.makedirs(dir_output)
    files = os.listdir(dir_input)
    files.sort()
    with imageio.get_writer(dir_output, mode='I') as writer:
        for filename in files:
            image = imageio.imread(os.path.join(dir_input, filename))
            writer.append_data(image)


def convert_dataset(dir_input, dir_output):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    videos = os.listdir(dir_input)
    videos.sort()

    for i, video in enumerate(videos):
        print(i + 1, len(videos), video)
        image2gif(os.path.join(dir_input, video), os.path.join(dir_output, video + '.gif'))


if __name__ == "__main__":
    print('done')