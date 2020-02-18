import os
import tqdm
from maskrcnn_benchmark.external.tracking.MOT.deep_sort.deep_sort_tracker import DeepSortTracker

if __name__ == "__main__":

    videonames = os.listdir('/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Images')
    videonames = sorted(videonames)

    tq = tqdm.tqdm(total=len(videonames))

    for videoname in videonames:
        tq.set_description('Video {}'.format(videoname))
        tq.update(1)
        tracker = DeepSortTracker()
        tracker.track(os.path.join('/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Images', videoname),
                  os.path.join('/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Annotations/Road_Objects', videoname),
                  os.path.join('/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Track/DeepSort_Reverse/Visualization', videoname),
                  os.path.join('/mnt/sdb/ltnghia/ITS_project/Accident/DAD/Demo_dataset/Track/DeepSort_Reverse/Json', videoname),
                  visualization=True,
                  reverse=True,)
        # break

