# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import torch
import os
import math
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import TensorboardXLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.ulti import get_device
from maskrcnn_benchmark.utils.comm import is_main_process
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size, all_gather, synchronize

from apex import amp

import tqdm


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    cfg,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    output_dir='',
    visualize_loss='',
    vis_title='',
    iters_per_epoch=0
):
    max_iter = len(data_loader)
    # arguments["iteration"] = max_iter
    start_iter = arguments["iteration"]
    if start_iter >= max_iter:
        checkpointer.save("model_{:07d}".format(start_iter), **arguments)
        checkpointer.save("model_epoch_{:07d}".format(int(math.ceil(start_iter / iters_per_epoch))), **arguments)
        return

    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")

    meters = TensorboardXLogger(log_dir=os.path.join(output_dir, 'tensorboardX'), delimiter="  ")

    model.train()
    start_training_time = time.time()
    end = time.time()

    mkdir(output_dir)

    if visualize_loss == "visdom" and is_main_process():
        from maskrcnn_benchmark.utils.visualization.visdom_visualizer import VisdomVisualizer
        vis_legend = None
        visualizer = VisdomVisualizer()
    else:
        visualizer = None

    scheduler.step(start_iter-1)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    if is_main_process():
        tq = tqdm.tqdm(total=len(data_loader), initial=start_iter)
    for iteration, batch in enumerate(data_loader, start_iter):
        images = batch[0]
        if len(batch) > 2:
            info = batch[2]
        else:
            info = None
        reg_targets = None
        seg_targets = None
        if isinstance(batch[1], dict):
            roi_targets = batch[1]["roi_target"]
            if "reg_target" in batch[1].keys():
                reg_targets = batch[1]["reg_target"]
            if "seg_target" in batch[1].keys():
                seg_targets = batch[1]["seg_target"]
        else:
            roi_targets = batch[1]

        # for target in roi_targets:
        #     print('labels: ', target.extra_fields['labels'])
        #     print('second_labels: ', target.extra_fields['second_labels'])

        if any(len(target) < 1 for target in roi_targets):
            roi_targets = None
        #     logger.error(f"Iteration={iteration+1} || Image Ids used for training {_} || targets Length={[len(target) for target in roi_targets]}")
        #     continue

        # print(info)
        # print(roi_targets)

        data_time = time.time() - end

        iteration = iteration + 1
        arguments["iteration"] = iteration

        if is_main_process():
            tq.set_description('Iteration {}'.format(iteration))
            tq.update(1)

        images = images.to(device)
        if roi_targets is not None:
            roi_targets = [target.to(device) for target in roi_targets]
        if reg_targets is not None:
            reg_targets = reg_targets.to(device)
        if seg_targets is not None:
            seg_targets = seg_targets.to(device)
        global_targets = dict(reg_targets=reg_targets, seg_targets=seg_targets)
        loss_dict = model(images, roi_targets, global_targets=global_targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(iteration, loss=losses_reduced, **loss_dict_reduced)

        losses = sum(loss for loss in loss_dict.values())

        if visualizer:
            if vis_legend is None:
                vis_legend = [key for key in sorted(loss_dict_reduced.keys())]
                vis_legend.append('Total loss')
                iter_plot = visualizer.create_vis_plot('Iteration', 'loss', vis_title, vis_legend)
            for key in sorted(loss_dict_reduced.keys()):
                visualizer.update_vis_plot(iteration=iteration,
                                           loss=loss_dict_reduced[key],
                                           window=iter_plot,
                                           name=key,
                                           update_type='append')
            visualizer.update_vis_plot(iteration=iteration,
                                       loss=losses_reduced.data,
                                       window=iter_plot,
                                       name='Total loss',
                                       update_type='append')

        optimizer.zero_grad()

        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        if device == get_device(str_device='cuda'):
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
        else:
            # pooling doesn't support CPU
            # losses.backward()
            pass

        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(iteration, time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # tq.set_postfix(
        #     log="max mem: {.0f}, lr: {.6f}, loss: {.6f}".format(
        #         float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
        #         float(optimizer.param_groups[0]["lr"]),
        #         float(losses_reduced)
        #     )
        # )

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iters_per_epoch > 0:
            if iteration % iters_per_epoch == 0:
                checkpointer.save("model_epoch_{:07d}".format(int(math.ceil(iteration / iters_per_epoch))), **arguments)

        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = TensorboardXLogger(log_dir=None, delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

    iteration = len(data_loader)
    if is_main_process():
        tq.close()

    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    checkpointer.save("model_epoch_{:07d}".format(int(math.ceil(iteration / iters_per_epoch))), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))

