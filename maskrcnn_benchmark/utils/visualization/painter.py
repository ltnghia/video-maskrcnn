import os
import cv2
from shutil import copyfile
from PIL import Image, ImageFont, ImageDraw
from maskrcnn_benchmark.utils.visualization.cv2_util import visualize_video, visualize_object_video


def combine_2_images(files=[], captions=[], output='', height=720, width=1280, h_dis=60, w_dis=80, font_size=36):

    result = Image.new("RGB", (width*2+w_dis, height*2+h_dis))

    for index, file in enumerate(files):
        path = os.path.expanduser(file)
        if os.path.isfile(path):
            img = Image.open(path)
            img = img.resize((width, height), Image.ANTIALIAS)
            x = index % 2 * (width + w_dis)
            y = int(result.size[1] / 4)
            w, h = img.size
            result.paste(img, (x, y, x + w, y + h))

    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("Preview.otf", font_size)
    for index, caption in enumerate(captions):
        if caption:
            draw.text(((width + w_dis) * (index % 2) + width / 2 - int(len(caption) / 2) * 25,
                       int(result.size[1] / 4) + height + h_dis / 2 - 0),
                      caption,
                      (255, 255, 255),
                      font=font)

    result.save(os.path.expanduser(output))


def combine_4_images(files=[], captions=[], output='', height=720, width=1280, h_dis=60, w_dis=80, font_size=36):

    result = Image.new("RGB", (width * 2 + w_dis, height * 2 + h_dis * 2))

    for index, file in enumerate(files):
        path = os.path.expanduser(file)
        if os.path.isfile(path):
            img = Image.open(path)
            img = img.resize((width, height), Image.ANTIALIAS)
            x = index % 2 * (width + w_dis)
            y = index // 2 * (height + h_dis)
            w, h = img.size
            result.paste(img, (x, y, x + w, y + h))

    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("Preview.otf", font_size)
    for index, caption in enumerate(captions):
        if caption:
            draw.text(((width + w_dis) * (index % 2) + width / 2 - int(len(caption) / 2) * 25,
                       (height + h_dis) * (index // 2) + height + h_dis / 2 - 20),
                      caption,
                      (255, 255, 255),
                      font=font)

    result.save(os.path.expanduser(output))


def convert_4_images_to_video(video_dirs, captions, output_dir, height=720, width=1280, h_dis=60, w_dis=80, font_size=36):
    images = os.listdir(video_dirs[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image in enumerate(sorted(images)):
        files = []
        for video_dir in video_dirs:
            if os.path.exists(video_dir):
                files.append(os.path.join(video_dir, image))
            else:
                files.append('')

        combine_4_images(
            files=files,
            captions=captions,
            output=os.path.join(output_dir, 'frame%06d.jpg' % i),
            height=height, width=width, h_dis=h_dis, w_dis=w_dis, font_size=font_size)


def convert_2_images_to_video(video_dirs, captions, output_dir, height=720, width=1280, h_dis=60, w_dis=80, font_size=36):
    images = os.listdir(video_dirs[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image in enumerate(sorted(images)):
        files = []
        for video_dir in video_dirs:
            if os.path.exists(video_dir):
                files.append(os.path.join(video_dir, image))
            else:
                files.append('')

        combine_2_images(
            files=files,
            captions=captions,
            output=os.path.join(output_dir, 'frame%06d.jpg' % i),
            height=height, width=width, h_dis=h_dis, w_dis=w_dis, font_size=font_size)


def run_dad_gt():
    dir_image0 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo_dataset/Images"
    dir_json0 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Tracking_ID/Tubelet/Videos"
    dir_output0 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/temp"

    videos = os.listdir(dir_json0)
    for video in sorted(videos):
        print(video)
        dir_image = os.path.join(dir_image0, video)
        dir_json = os.path.join(dir_json0, video)
        dir_output1 = os.path.join(dir_output0, 'Object', video)
        dir_output2 = os.path.join(dir_output0, 'Accident', video)
        dir_output3 = os.path.join(dir_output0, 'Tracking', video)

        visualize_video(dir_image, dir_json, dir_output1, dir_output2, dir_output3, 3)

        convert_4_images_to_video(
            video_dirs=[
                dir_image,
                dir_output3,
                dir_output1,
                dir_output2],
            captions=['Dashboard Camera',
                      'Generated Tracking Ground-Truth',
                      'Our Object Ground-Truth',
                      'Our Characteristics Ground-Truth', ],
            output_dir=os.path.join('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/Images', video),
            height=720, width=1280, h_dis=60, w_dis=80, font_size=36)


def run_dad_ara():
    dir_image0 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo_dataset/Images"
    dir_json0 = "/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_2/Filter_data/Videos"
    dir_output0 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo2/temp/Iter_2_Filter_data"
    dir_output1 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo2/temp/Human"

    videos = os.listdir(dir_json0)
    for video in sorted(videos):
        print(video)
        dir_image = os.path.join(dir_image0, video)
        dir_json = os.path.join(dir_json0, video)
        dir_output = os.path.join(dir_output0, video)

        visualize_object_video(dir_image, dir_json, dir_output, 3)

        if os.path.exists(os.path.join(dir_output1, video)):
            convert_4_images_to_video(
                video_dirs=[
                    dir_image,
                    '',
                    dir_output,
                    os.path.join(dir_output1, video)],
                captions=['Dashboard Camera',
                          '',
                          'Our Self-Learning Method',
                          'Human Annotation', ],
                output_dir=os.path.join('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo2/Images', video),
                height=720, width=1280, h_dis=60, w_dis=80, font_size=36)


def run_dad_ara2():
    dir_image0 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo_dataset/Images"
    dir_json1 = "/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_2/Add_instances/Videos"
    dir_json2 = "/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_2/Filter_data/Videos"
    dir_json3 = "/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_3/Filter_data/Videos"
    dir_output1 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/temp/Iter_2_Add_instances"
    dir_output2 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/temp/Iter_2_Filter_data"
    dir_output3 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/temp/Iter_3_Filter_data"

    videos = os.listdir(dir_json1)
    for video in sorted(videos):
        print(video)
        dir_image = os.path.join(dir_image0, video)
        visualize_object_video(dir_image, os.path.join(dir_json1, video), os.path.join(dir_output1, video), 3)
        visualize_object_video(dir_image, os.path.join(dir_json2, video), os.path.join(dir_output2, video), 3)
        visualize_object_video(dir_image, os.path.join(dir_json3, video), os.path.join(dir_output3, video), 3)

        if os.path.exists(os.path.join(dir_output1, video)):
            convert_4_images_to_video(
                video_dirs=[
                    dir_image,
                    os.path.join(dir_output1, video),
                    os.path.join(dir_output2, video),
                    os.path.join(dir_output3, video)],
                captions=['Dashboard Camera',
                          'Iter_2_Add_instances',
                          'Iter_2_Filter_data',
                          'Iter_3_Filter_data', ],
                output_dir=os.path.join('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo3/Images',
                                        video),
                height=720, width=1280, h_dis=60, w_dis=80, font_size=36)


def run_dad_ara3():
    dir_image0 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo_dataset/Images"
    dir_json1 = '/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_0/Raw_Detection/Raw/Json'
    dir_json2 = "/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_2/Filter_data/Videos"
    dir_output0 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/temp/Human"
    dir_output1 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/temp/Iter_0_Raw_Detection"
    dir_output2 = "/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/temp/Iter_2_Filter_data"

    videos = os.listdir(dir_json1)
    for video in sorted(videos):
        print(video)
        dir_image = os.path.join(dir_image0, video)
        visualize_object_video(dir_image, os.path.join(dir_json1, video), os.path.join(dir_output1, video), 3)
        visualize_object_video(dir_image, os.path.join(dir_json2, video), os.path.join(dir_output2, video), 3)

        if os.path.exists(os.path.join(dir_output0, video)):
            convert_4_images_to_video(
                video_dirs=[
                    dir_image,
                    os.path.join(dir_output0, video),
                    os.path.join(dir_output1, video),
                    os.path.join(dir_output2, video)],
                captions=['Dashboard Camera',
                          'Human Annotation',
                          'Pre-Trained Model',
                          'Our Self-Learning Method',],
                output_dir=os.path.join('/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo4/Images',
                                        video),
                height=720, width=1280, h_dis=60, w_dis=80, font_size=36)


def run_cs_ara():
    dir_image0 = "/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images/"
    dir_json1 = '/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Tutorial_v2/Entire_dataset/0.88/Iter_0/Raw_Detection/Raw/Json'
    dir_json2 = "/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Tutorial_v2/Entire_dataset/0.88/Iter_2/Add_instances/Videos"
    dir_output1 = "/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo1/temp/Iter_0_Raw_Detection"
    dir_output2 = "/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo1/temp/Iter_2_Add_instances"

    videos = os.listdir(dir_json1)
    for video in sorted(videos):
        print(video)
        dir_image = os.path.join(dir_image0, video)
        # visualize_object_video(dir_image, os.path.join(dir_json1, video), os.path.join(dir_output1, video), 7)
        # visualize_object_video(dir_image, os.path.join(dir_json2, video), os.path.join(dir_output2, video), 7)

        if os.path.exists(os.path.join(dir_output1, video)):
            convert_4_images_to_video(
                video_dirs=[
                    dir_image,
                    '',
                    os.path.join(dir_output1, video),
                    os.path.join(dir_output2, video)],
                captions=['CityScapes Dataset',
                          '',
                          'Pre-Trained Model',
                          'Our Self-Learning Method', ],
                output_dir=os.path.join('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo1/Images', video),
                height=720, width=1280, h_dis=60, w_dis=80, font_size=36)


def run_cs_ara2():
    dir_image0 = "/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images/"
    dir_json1 = "/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Tutorial_v2/Entire_dataset/0.88/Iter_2/Add_instances/Videos"
    dir_output1 = "/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo2/temp/Iter_2_Add_instances"

    videos = os.listdir(dir_json1)
    for video in sorted(videos):
        print(video)
        dir_image = os.path.join(dir_image0, video)
        # visualize_object_video(dir_image, os.path.join(dir_json1, video), os.path.join(dir_output1, video), 7)

        if os.path.exists(os.path.join(dir_output1, video)):
            convert_2_images_to_video(
                video_dirs=[
                    dir_image,
                    os.path.join(dir_output1, video)],
                captions=['CityScapes Dataset',
                          'Our Self-Learning Method'],
                output_dir=os.path.join('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo2/Images', video),
                height=720, width=1280, h_dis=60, w_dis=80, font_size=36)


def run_cs_ara3():
    dir_image0 = "/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images_with_GT/"
    dir_json1 = '/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Tutorial_v2/Entire_dataset/0.88/Iter_0/Raw_Detection/Raw/Json'
    dir_json2 = "/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Tutorial_v2/Entire_dataset/0.88/Iter_2/Add_instances/Videos"
    dir_output1 = "/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo3/temp/Iter_0_Raw_Detection"
    dir_output2 = "/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo3/temp/Iter_2_Add_instances"

    videos = os.listdir(dir_json1)
    for video in sorted(videos):
        print(video)
        dir_image = os.path.join(dir_image0, video)
        # visualize_object_video(dir_image, os.path.join(dir_json1, video), os.path.join(dir_output1, video), 7)
        # visualize_object_video(dir_image, os.path.join(dir_json2, video), os.path.join(dir_output2, video), 7)

        if os.path.exists(os.path.join(dir_output1, video)):
            convert_4_images_to_video(
                video_dirs=[
                    dir_image,
                    '',
                    os.path.join(dir_output1, video),
                    os.path.join(dir_output2, video)],
                captions=['CityScapes Dataset',
                          '',
                          'Pre-Trained Model',
                          'Our Self-Learning Method', ],
                output_dir=os.path.join('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo3/Images', video),
                height=720, width=1280, h_dis=60, w_dis=80, font_size=36)


def run_cs_ara4():
    dir_image0 = "/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images_with_GT/"
    dir_json1 = "/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Tutorial_v2/Entire_dataset/0.88/Iter_2/Add_instances/Videos"
    dir_output1 = "/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo4/temp/Iter_2_Add_instances"

    videos = os.listdir(dir_json1)
    for video in sorted(videos):
        print(video)
        dir_image = os.path.join(dir_image0, video)
        # visualize_object_video(dir_image, os.path.join(dir_json1, video), os.path.join(dir_output1, video), 7)

        if os.path.exists(os.path.join(dir_output1, video)):
            convert_2_images_to_video(
                video_dirs=[
                    dir_image,
                    os.path.join(dir_output1, video)],
                captions=['CityScapes Dataset',
                          'Our Self-Learning Method'],
                output_dir=os.path.join('/mnt/sdc/Backup_From_sdb/Accident/CityScapes/Demo/Demo4/Images', video),
                height=720, width=1280, h_dis=60, w_dis=80, font_size=36)


if __name__ == "__main__":
    # combine_4_images(files=['/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo_dataset/Images/000001/frame000001.jpg',
    #                         '/mnt/sdb/ltnghia/ITS_project/Accident/Experiment/DAD/ISA/Tracking_ID/Tubelet/Visualization/000001/frame000001.jpg',
    #                         '/mnt/sdb/ltnghia/ITS_project/Accident/Experiment/DAD/ISA/Tracking_ID/Tubelet/Visualization/000001/frame000001.jpg',
    #                         '/mnt/sdb/ltnghia/ITS_project/Accident/Experiment/DAD/ISA/Tracking_ID/Tubelet/Visualization/000001/frame000001.jpg'],
    #                  captions=['Dashboard Camera',
    #                            'Pre-trained Model',
    #                            'Our Self-Learning Method',
    #                            'Human Verification'],
    #                  output='/mnt/sdc/output.jpg',
    #                  height=720, width=1280, h_dis=60, w_dis=80, font_size=36)
    #
    # combine_4_images(
    #     files=['/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images/frankfurt_000000_000000/frankfurt_000000_000275_leftImg8bit.jpg', '',
    #            '/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images/frankfurt_000000_000000/frankfurt_000000_000275_leftImg8bit.jpg',
    #            '/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images/frankfurt_000000_000000/frankfurt_000000_000275_leftImg8bit.jpg'],
    #     captions=['CityScape Video', '',
    #               'Pre-trained Model',
    #               'Our Self-Learning Method'],
    #     output='/mnt/sdc/output2.jpg',
    #     height=720, width=1280, h_dis=60, w_dis=80, font_size=36)
    #
    # combine_2_images(
    #     files=[
    #         '/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images/frankfurt_000000_000000/frankfurt_000000_000275_leftImg8bit.jpg',
    #         '/mnt/sdb/ltnghia/ITS_project/Accident/CityScapes/val/Demo_dataset/Images/frankfurt_000000_000000/frankfurt_000000_000275_leftImg8bit.jpg'],
    #     captions=['CityScape Video',
    #               'Our Self-Learning Method'],
    #     output='/mnt/sdc/output3.jpg',
    #     height=720, width=1280, h_dis=60, w_dis=80, font_size=36)
    #
    # combine_4_images(
    #     files=[
    #         '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Images/20171215_100225_001_Camera1/frame000324.jpg',
    #         '',
    #         '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_0/Raw_Detection/Raw/Visualization0/20171215_100225_001_Camera1/frame000324.jpg',
    #         '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_1/Add_instances/Visualization/20171215_100225_001_Camera1/frame000324.jpg'],
    #     captions=['Kyushu Video', '',
    #               'Pre-trained Model',
    #               'Our Self-Learning Method'],
    #     output='/mnt/sdc/output4.jpg',
    #     height=480, width=640, h_dis=60, w_dis=80, font_size=36)

    convert_4_images_to_video(
        video_dirs=[
            '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Images/20171215_100225_001_Camera1',
            '',
            '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_0/Raw_Detection/Raw/Visualization0/20171215_100225_001_Camera1',
            '/mnt/sdc/Backup_From_sdb/Accident/Kyushu/Demo_dataset/Tutorial_v2/Entire_dataset/Iter_1/Add_instances/Visualization2/20171215_100225_001_Camera1'],
        captions=['Kyushu Video', '',
                  'Pre-trained Model',
                  'Our Self-Learning Method'],
        output_dir='/mnt/sdc/Backup_From_sdb/Accident/Kyushu/video/images/20171215_100225_001_Camera1',
        height=480, width=640, h_dis=60, w_dis=80, font_size=36)

    # convert_4_images_to_video(
    #     video_dirs=[
    #         '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo_dataset/Images/000001',
    #         '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/temp/Tracking/000001',
    #         '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/temp/Object/000001',
    #         '/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/temp/Accident/000001',],
    #     captions=['Dashboard Camera',
    #               'Generated Tracking Ground-Truth',
    #               'Our Object Ground-Truth',
    #               'Our Characteristics Ground-Truth',],
    #     output_dir='/mnt/sdc/Backup_From_sdb/Accident/Experiment/DAD/ISA/Demo/Demo1/Images/000001',
    #     height=720, width=1280, h_dis=60, w_dis=80, font_size=36)

    # run_dad_gt()
    # run_dad_ara()
    # run_dad_ara2()
    # run_dad_ara3()
    # run_cs_ara()
    # run_cs_ara2()
    # run_cs_ara3()
    # run_cs_ara4()

    pass



