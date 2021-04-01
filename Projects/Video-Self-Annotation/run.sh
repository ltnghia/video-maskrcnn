#!/usr/bin/env bash

python init_root_dirs.py --iter 0
python create_dataset.py

python clear_dir.py
python run_rcnn.py --test true
python init_root_dirs.py --iter 0
python generate_raw_ann_from_rcnn_results.py
python generate_adaptive_detection.py
python track_instances.py
python create_tubelets.py
python filter_data.py
python add_instances.py
python convert_tubelet_to_coco_format.py

python clear_dir.py
python run_rcnn.py --train true
python run_rcnn.py --test true
python init_root_dirs.py --iter 1
python generate_raw_ann_from_rcnn_results.py
python generate_adaptive_detection.py
python track_instances.py
python create_tubelets.py
python filter_data.py
python add_instances.py
python convert_tubelet_to_coco_format.py

python clear_dir.py
python run_rcnn.py --train true
python run_rcnn.py --test true
python init_root_dirs.py --iter 2
python generate_raw_ann_from_rcnn_results.py
python generate_adaptive_detection.py
python track_instances.py
python create_tubelets.py
python filter_data.py
python add_instances.py
python convert_tubelet_to_coco_format.py

python clear_dir.py
python run_rcnn.py --train true
python run_rcnn.py --test true
python init_root_dirs.py --iter 3
python generate_raw_ann_from_rcnn_results.py
python ulti.py
