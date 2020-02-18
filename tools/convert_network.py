import os
import torch

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.checkpoint.converted_checkpoint import ConvertedCheckpointer, REMOVED_KEYS
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.checkpoint.model_zoo import cache_url


def remove_keys_in_model(d, listofkeys=REMOVED_KEYS):
    d = dict(d)
    keys = sorted(list(d.keys()))
    for key in listofkeys:
        for model_key in keys:
            if key in model_key and 'rpn' not in model_key:
                print('key: {} is removed'.format(model_key))
                del d[model_key]
    return d


def transfer_pretrained_weights(cfg=None, path_output=None):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpointer = ConvertedCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.convert(cfg.MODEL.WEIGHT,
                         use_gn_fpn=True,
                         use_gn_head_box=True,
                         use_gn_head_mask=True, )

    checkpointer.save(path_output)
    print("write model to: " + path_output)
    if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, 'Models/last_checkpoint')):
        os.remove(os.path.join(cfg.OUTPUT_DIR, 'Models/last_checkpoint'))


def convert_caffe2_to_pytorch(config_file, weight_dir, output_name):
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHT = weight_dir
    cfg.freeze()

    mkdir(cfg.OUTPUT_DIR)

    f = cfg.MODEL.WEIGHT
    if f.startswith("catalog://"):
        paths_catalog = import_file(
            "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])

    # download url files
    if f.startswith("http"):
        f = cache_url(f)

    if output_name != "":
        path_output = os.path.join(cfg.OUTPUT_DIR, output_name)
    elif cfg.MODEL.WEIGHT.endswith(".pkl"):
        path_output = os.path.join(cfg.OUTPUT_DIR, os.path.basename(f))
    else:
        str_output = cfg.MODEL.WEIGHT.split('/')
        str_output = str_output[-1]
        path_output = os.path.join(cfg.OUTPUT_DIR, str_output + '')

    transfer_pretrained_weights(cfg, path_output)


def convert_pytorch_to_pytorch(dir_input, dir_output, init_classifier=False):
    if not os.path.isfile(dir_input):
        return

    model = torch.load(dir_input)
    model.pop('optimizer', None)
    model.pop('scheduler', None)
    model.pop('iteration', None)

    if init_classifier:
        model['model'] = remove_keys_in_model(model['model'])
    torch.save(model, dir_output)


def convert_pytorch_to_new_format(config_file, dir_input, dir_output, init_classifier=False):
    if not os.path.isfile(dir_input):
        return

    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHT = dir_input
    cfg.freeze()

    mkdir(os.path.dirname(dir_output))

    f = cfg.MODEL.WEIGHT
    if f.startswith("catalog://"):
        paths_catalog = import_file(
            "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])

    # download url files
    if f.startswith("http"):
        f = cache_url(f)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpointer = ConvertedCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.convert(cfg.MODEL.WEIGHT,
                         use_gn_fpn=False,
                         use_gn_head_box=True,
                         use_gn_head_mask=True,
                         keys=[])

    model = {}
    model["model"] = checkpointer.model.state_dict()

    if init_classifier:
        model["model"] = remove_keys_in_model(model["model"])

    torch.save(model, dir_output)
    print("write model to: " + dir_output)
    if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, 'last_checkpoint')):
        os.remove(os.path.join(cfg.OUTPUT_DIR, 'last_checkpoint'))





