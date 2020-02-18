import argparse
import os
import torch
import json
import math

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.engine import network
from maskrcnn_benchmark.utils.ulti import get_num_gpus, seed_torch

# seed_torch()


def main(args):
    cfg.merge_from_file(args.config_file)

    num_gpus = get_num_gpus()
    DatasetCatalog = None

    # train_dataset = cfg.DATASETS.TRAIN[0]
    # paths_catalog = import_file(
    #     "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    # )
    # data = json.load(open(paths_catalog.DatasetCatalog.DATASETS[train_dataset]['ann_file']))
    # iters_per_epoch = len(data['images'])
    # cfg.defrost()
    # iters_per_epoch = math.ceil(iters_per_epoch / cfg.SOLVER.IMS_PER_BATCH)
    # cfg.SOLVER.MAX_ITER = round(args.epochs * args.scale * iters_per_epoch)
    # cfg.SOLVER.STEPS = (round(8 * args.scale * iters_per_epoch),
    #                     round(11 * args.scale * iters_per_epoch),
    #                     round(16 * args.scale * iters_per_epoch))
    # cfg.SOLVER.IMS_PER_BATCH = num_gpus * 4
    # cfg.TEST.IMS_PER_BATCH = num_gpus * 16
    # cfg.freeze()

    mkdir(cfg.OUTPUT_DIR)

    if args.vis_title is None:
        args.vis_title = os.path.basename(cfg.OUTPUT_DIR)

    logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    logger.info(DatasetCatalog)

    model = network.train(cfg, args, DatasetCatalog)
    network.test(cfg, args, model=model, DatasetCatalog=DatasetCatalog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--config-file",
        default="e2e_faster_rcnn_R_50_FPN_Xconv1fc_1x_gn.yaml",
        metavar="FILE",
        help="path to config file",
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
        default=1,
        type=int,
    )

    parser.add_argument(
        "--visualize-loss",
        dest="visualize_loss",
        help="Draw loss function",
        default="visdom", # "visdom", "tensorboardx", "tensorboard_logger"
        type=str,
    )
    parser.add_argument(
        "--visualize-results",
        dest="visualize_results",
        help="Visualize results",
        default=True,
        type=bool
    )
    parser.add_argument(
        "--only-visualization",
        dest="only_visualization",
        help="Only visualize results, not evaluate results",
        default=False,
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

    args = parser.parse_args()

    num_gpus = get_num_gpus()
    args.distributed = num_gpus > 1

    # print(num_gpus, args.local_rank, args.distributed)
    # exit()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    main(args)


