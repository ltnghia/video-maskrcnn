import os
import tqdm
import ulti
from maskrcnn_benchmark.external.tracking.MOT.deep_sort.deep_sort_tracker import DeepSortTracker

if __name__ == "__main__":
    info = ulti.load_json()
    videonames = os.listdir(os.path.join(info['dataset_dir'], 'Images'))
    videonames = sorted(videonames)

    output_kf_bbox = False

    if info['annotated_video'] == info['dataset_name']:
        tq = tqdm.tqdm(total=len(videonames))
        for videoname in videonames:
            tq.set_description('Video {}'.format(videoname))
            tq.update(1)
            tracker = DeepSortTracker(output_kf_bbox=output_kf_bbox)
            tracker.track(os.path.join(info['dataset_dir'], 'Images', videoname),
                          os.path.join(info['dataset_dir'], info['experiment'], 'Detection', 'Json', videoname),
                          None,
                          os.path.join(info['dataset_dir'], info['experiment'], 'Track', 'DeepSort', 'Json', videoname),
                          visualization=False,
                          tq_display=False,
                          reverse=False)
    else:
        videoname = info['annotated_video']
        tracker = DeepSortTracker(output_kf_bbox=output_kf_bbox)
        tracker.track(os.path.join(info['dataset_dir'], 'Images', videoname),
                      os.path.join(info['dataset_dir'], info['experiment'], 'Detection', 'Json', videoname),
                      None,
                      os.path.join(info['dataset_dir'], info['experiment'], 'Track', 'DeepSort', 'Json', videoname),
                      visualization=False,
                      tq_display=True,
                      reverse=False)

    if info['annotated_video'] == info['dataset_name']:
        tq = tqdm.tqdm(total=len(videonames))
        for videoname in videonames:
            tq.set_description('Video {}'.format(videoname))
            tq.update(1)
            tracker = DeepSortTracker(output_kf_bbox=output_kf_bbox)
            tracker.track(os.path.join(info['dataset_dir'], 'Images', videoname),
                          os.path.join(info['dataset_dir'], info['experiment'], 'Detection', 'Json', videoname),
                          None,
                          os.path.join(info['dataset_dir'], info['experiment'], 'Track', 'DeepSort_Reverse', 'Json', videoname),
                          visualization=False,
                          tq_display=False,
                          reverse=True)
    else:
        videoname = info['annotated_video']
        tracker = DeepSortTracker(output_kf_bbox=output_kf_bbox)
        tracker.track(os.path.join(info['dataset_dir'], 'Images', videoname),
                      os.path.join(info['dataset_dir'], info['experiment'], 'Detection', 'Json', videoname),
                      None,
                      os.path.join(info['dataset_dir'], info['experiment'], 'Track', 'DeepSort_Reverse', 'Json', videoname),
                      visualization=False,
                      tq_display=True,
                      reverse=True)




