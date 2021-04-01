import argparse
import os
import torch
import json
import ulti
import math

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.engine import network
from maskrcnn_benchmark.utils.ulti import get_num_gpus, seed_torch

seed_torch()
info = ulti.load_json()


def main(args):

    num_gpus = get_num_gpus()
    args.config_file = os.path.join(info['training_dir'], 'e2e_faster_rcnn_R_50_FPN_Xconv1fc_1x_gn.yaml')

    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.OUTPUT_DIR = os.path.join(info['training_dir'], args.sub_dataset)
    cfg.MODEL.WEIGHT = os.path.join(info['dataset_dir'], info['experiment'], 'Detector', 'Iter{}.pth'.format(info['iter']))
    cfg.SOLVER.IMS_PER_BATCH = num_gpus * 4
    cfg.TEST.IMS_PER_BATCH = num_gpus * 16
    cfg.SOLVER.BASE_LR = 0.002
    cfg.freeze()

    mkdir(cfg.OUTPUT_DIR)

    if args.sub_dataset is None:
        args.sub_dataset = ""

    if args.vis_title is None:
        args.vis_title = os.path.basename(cfg.OUTPUT_DIR)

    logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    DatasetCatalog = None
    train_dataset = cfg.DATASETS.TRAIN[0]
    test_dataset = cfg.DATASETS.TEST[0]
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )

    if args.sub_dataset != "":
        DatasetCatalog = paths_catalog.DatasetCatalog

        DatasetCatalog.DATASETS[train_dataset]['img_dir'] = os.path.join(
            info['dataset_dir'], 'Images')
        DatasetCatalog.DATASETS[train_dataset]['ann_file'] = os.path.join(
            info['dataset_dir'], 'RCNN_data', 'train.json')

        DatasetCatalog.DATASETS[test_dataset]['img_dir'] = os.path.join(
            info['dataset_dir'], 'Images')
        DatasetCatalog.DATASETS[test_dataset]['ann_file'] = os.path.join(
            info['dataset_dir'], 'RCNN_data', 'test.json')

        data = json.load(open(DatasetCatalog.DATASETS[train_dataset]['ann_file']))
    else:
        data = json.load(open(paths_catalog.DatasetCatalog.DATASETS[train_dataset]['ann_file']))

    iters_per_epoch = len(data['images'])
    iters_per_epoch = math.ceil(iters_per_epoch / cfg.SOLVER.IMS_PER_BATCH)
    args.iters_per_epoch = iters_per_epoch

    cfg.defrost()
    cfg.SOLVER.MAX_ITER = round(args.epochs * args.scale * iters_per_epoch)
    cfg.SOLVER.STEPS = (round(8 * args.scale * iters_per_epoch),
                        round(11 * args.scale * iters_per_epoch),
                        round(16 * args.scale * iters_per_epoch))
    cfg.freeze()

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    logger.info(DatasetCatalog)

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    if args.train:
        args.skip_train = False
        model = network.train(cfg, args, DatasetCatalog)

    if args.test:
        network.test(cfg, args, model=None, DatasetCatalog=DatasetCatalog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--config-file",
        default=os.path.join(info['training_dir'], 'e2e_faster_rcnn_R_50_FPN_Xconv1fc_1x_gn.yaml'),
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--sub-dataset",
        dest="sub_dataset",
        help="Video Name",
        default=info['annotated_video'],
        # default=None,
        type=str,
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        help="Number of Epochs",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        help="Scale of Epochs",
        default=0.5,
        type=int,
    )
    parser.add_argument(
        "--train",
        dest="train",
        help="Train model",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--test",
        dest="test",
        help="Test the final model",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--visualize-loss",
        dest="visualize_loss",
        help="Draw loss function",
        default="", # "visdom", "tensorboardx", "tensorboard_logger"
        type=str,
    )
    parser.add_argument(
        "--visualize-results",
        dest="visualize_results",
        help="Visualize results",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--only-visualization",
        dest="only_visualization",
        help="Only visualize results, not evaluate results",
        default=True,
        type=bool
    )
    parser.add_argument(
        "--category",
        dest="category",
        help="dataset having categories",
        default="ro",
        type=str
    )
    parser.add_argument(
        "--visualization-title",
        dest="vis_title",
        help="Visualization Title",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--ckpt",
        dest="ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )

    parser.add_argument(
        "--copy-weight-from-head-box",
        dest="copy_weight_from_head_box",
        help="Copy weight from ROI head box to other branches if they are initialized",
        default=False,
        type=bool
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    main(args)

