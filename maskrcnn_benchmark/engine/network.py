import os
import torch
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.checkpoint.detection_checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.inference import inference


# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def train(cfg, args, DatasetCatalog=None):
    if len(cfg.DATASETS.TRAIN) == 0 or not args.train:
        return None

    local_rank = args.local_rank
    distributed = args.distributed

    model = build_detection_model(cfg)

    # for key, value in model.named_parameters():
    #     print(key, value.requires_grad)

    if hasattr(args, 'train_last_layer'):
        if args.train_last_layer:
            listofkeys = ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight',
                          'mask_fcn_logits.bias', 'mask_fcn_logits.weight']
            for key, value in model.named_parameters():
                value.requires_grad = False
                for k in listofkeys:
                    if k in key:
                        value.requires_grad = True
            # for key, value in model.named_parameters():
            #     print(key, value.requires_grad)

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    if cfg.MODEL.DEVICE == 'cuda':
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    save_to_disk = get_rank() == 0

    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk)

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT,
                                              force_load_external_checkpoint=False,
                                              copy_weight_from_head_box=args.copy_weight_from_head_box)

    arguments = {}
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        args,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        DatasetCatalog=DatasetCatalog,
    )

    if cfg.SOLVER.TEST_PERIOD > 0:
        data_loader_val = make_data_loader(cfg,
                                           args,
                                           is_train=False,
                                           is_distributed=distributed,
                                           is_for_period=True,
                                           start_iter=arguments["iteration"],
                                           DatasetCatalog=DatasetCatalog,
                                           )
    else:
        data_loader_val = None

    do_train(
        model,
        cfg,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        cfg.SOLVER.CHECKPOINT_PERIOD,
        cfg.SOLVER.TEST_PERIOD,
        arguments,
        cfg.OUTPUT_DIR,
        args.visualize_loss,
        args.vis_title,
        args.iters_per_epoch,
    )

    return model


def test(cfg, args, model=None, DatasetCatalog=None):
    # if args.distributed:
    #     model = model.module
    # torch.cuda.empty_cache()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_dir = cfg.OUTPUT_DIR
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(output_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    else:
        output_dir = None

    device = torch.device(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    # if cfg.MODEL.DEVICE == 'cuda':
    #     use_mixed_precision = cfg.DTYPE == 'float16'
    #     amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    if model is None:
        model = build_detection_model(cfg)
        model.to(device)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
        ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
        _ = checkpointer.load(ckpt, force_load_external_checkpoint=(args.ckpt is not None))

    data_loaders_val = make_data_loader(
        cfg,
        args,
        is_train=False,
        is_distributed=args.distributed,
        DatasetCatalog=DatasetCatalog,
    )
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=device,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            cfg=cfg,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            visualize_results=args.visualize_results,
            visualization_label=args.category,
            only_visualization=args.only_visualization,
        )
        synchronize()



