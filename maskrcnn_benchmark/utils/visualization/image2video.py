import os
import cv2
import shutil


def image2video(dir_input, dir_output, format='frame%06d.jpg'):
    # if not os.path.exists(dir_output):
    #     os.makedirs(dir_output)
    # os.system("ffmpeg -r 24 -i " + os.path.join(dir_input, format) + " -vcodec mpeg4 -y " + dir_output)
    os.system("ffmpeg -r 24 -i " + os.path.join(dir_input, format) + " -c:v libx264 -crf 10 -c:a copy " + dir_output)


def convert_dataset(dir_input, dir_output, format='frame%06d.jpg'):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    videos = os.listdir(dir_input)
    videos.sort()

    for i, video in enumerate(videos):
        print(i+1, len(videos), video)
        image2video(os.path.join(dir_input, video), os.path.join(dir_output, video + '.mp4'), format)


def dataset2video(dir_input, dir_temp, dir_output, format='frame%06d.jpg'):
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    index = 1
    videos = os.listdir(dir_input)
    for video in sorted(videos):
        files = os.listdir(os.path.join(dir_input, video))
        for file in sorted(files):
            source = os.path.join(dir_input, video, file)
            destination = os.path.join(dir_temp, format % index)
            shutil.copyfile(source, destination)
            index += 1

    image2video(dir_temp, dir_output, format)


if __name__ == "__main__":
    image2video('/mnt/sdc/Backup_From_sdb/Accident/Kyushu/video/images/20171215_100225_001_Camera1',
                '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/video/20171215_100225_001_Camera1.mp4',
                'frame%06d.jpg')

    # convert_dataset('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/Images',
    #                 '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/Videos')
    # dataset2video('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/Images',
    #               '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/tmp',
    #               '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/demo.mp4')

    # convert_dataset('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo2/Images',
    #                 '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo2/Videos')
    # dataset2video('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo2/Images',
    #               '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo2/tmp',
    #               '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo2/demo.mp4')

    # convert_dataset('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/Images',
    #                 '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/Videos')
    # dataset2video('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/Images',
    #               '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/tmp',
    #               '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/demo.mp4')

    # convert_dataset('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/Images',
    #                 '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/Videos')
    # dataset2video('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/Images',
    #               '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/tmp',
    #               '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/demo.mp4')

    # convert_dataset('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo1/Images',
    #                 '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo1/Videos')
    # dataset2video('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo1/Images',
    #               '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo1/tmp',
    #               '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo1/demo.mp4')

    # convert_dataset('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo3/Images',
    #                 '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo3/Videos')
    # dataset2video('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo3/Images',
    #               '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo3/tmp',
    #               '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo3/demo.mp4')
    #
    # convert_dataset('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo2/Images',
    #                 '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo2/Videos')
    # dataset2video('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo2/Images',
    #               '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo2/tmp',
    #               '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo2/demo.mp4')
    #
    # convert_dataset('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo4/Images',
    #                 '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo4/Videos')
    # dataset2video('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo4/Images',
    #               '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo4/tmp',
    #               '/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo4/demo.mp4')
