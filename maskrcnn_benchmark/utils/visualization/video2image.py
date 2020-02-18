import os
import cv2


def video2image(dir_input, dir_output, format='frame%06d.jpg', rotate=False):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    vidcap = cv2.VideoCapture(dir_input)
    count = 0
    while(True):
        success, image = vidcap.read()
        if not success:
            break
        if rotate:
            (h, w) = image.shape[:2]
            center = (h / 2, w / 2)
            angle90 = 270
            scale = 1.0
            M = cv2.getRotationMatrix2D(center, angle90, scale)
            image = cv2.warpAffine(image, M, (h, w))
        cv2.imwrite(os.path.join(dir_output, format % (count+1)), image)
        count += 1


def convert_dataset(dir_input, dir_output, format='frame%06d.jpg', rotate=False):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    videos = os.listdir(dir_input)
    if len(videos) > 0:
        videos.sort()

    for i, video in enumerate(videos):
        print(i + 1, len(videos), video)
        videoname = os.path.splitext(video)[0]
        video2image(os.path.join(dir_input, video), os.path.join(dir_output, videoname), format, rotate)


def convert_d2_city(dir_input, dir_output):
    print(dir_input)
    folders = os.listdir(dir_input)
    if len(folders) > 0:
        folders.sort()

    for folder in folders:
        if not os.path.exists(os.path.join(dir_output, folder)):
            os.makedirs(os.path.join(dir_output, folder))
        convert_dataset(os.path.join(dir_input, folder), os.path.join(dir_output, folder))


def convert_bdd(dir_input, dir_output, format='frame%06d.jpg'):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    videos = os.listdir(dir_input)
    if len(videos) > 0:
        videos.sort()

    for i, video in enumerate(videos):
        print(i + 1, len(videos), video)
        videoname = os.path.splitext(video)[0]
        if not os.path.exists(os.path.join(dir_output, videoname)):
            os.makedirs(os.path.join(dir_output, videoname))
        cmd = "ffmpeg -i " + os.path.join(dir_input, video) + " -vf fps=1 " + os.path.join(dir_output, videoname, format)
        os.system(cmd)
        # exit()


if __name__ == "__main__":
    # convert_d2_city('/mnt/sdb/ltnghia/Dataset/D2City/train/video', '/mnt/sdb/ltnghia/Dataset/D2City/train/image')
    # convert_d2_city('/mnt/sdb/ltnghia/Dataset/D2City/val/video', '/mnt/sdb/ltnghia/Dataset/D2City/val/image')
    # convert_d2_city('/mnt/sdb/ltnghia/Dataset/D2City/test0/video', '/mnt/sdb/ltnghia/Dataset/D2City/test0/image')

    # convert_bdd('/mnt/sdb/ltnghia/Dataset/BDD100K/videos/100k/train',
    #                 '/mnt/sdb/ltnghia/Dataset/BDD100K/videos/image_per_second/train')
    # convert_bdd('/mnt/sdb/ltnghia/Dataset/BDD100K/videos/100k/val',
    #                 '/mnt/sdb/ltnghia/Dataset/BDD100K/videos/image_per_second/val')
    # convert_bdd('/mnt/sdb/ltnghia/Dataset/BDD100K/videos/100k/test0',
    #                 '/mnt/sdb/ltnghia/Dataset/BDD100K/videos/image_per_second/test0')

    convert_d2_city("/mnt/sdb/ltnghia/Dataset/DAD/videos/train", "/mnt/sdb/ltnghia/Dataset/DAD/images/train")
    convert_d2_city("/mnt/sdb/ltnghia/Dataset/DAD/videos/test0", "/mnt/sdb/ltnghia/Dataset/DAD/images/test0")


