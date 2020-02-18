# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
from .comm import is_main_process
import json
import logging


def write_json(info, file):
    outfile = open(file, 'w')
    x = json.dumps(info)
    outfile.write(x)
    outfile.close()
    return file


def load_json(file):
    if not os.path.isfile(file):
        return None
    info = json.load(open(file))
    return info


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


def save_labels(dataset_list, output_dir):
    if is_main_process():
        logger = logging.getLogger(__name__)

        ids_to_labels = {}
        for dataset in dataset_list:
            if hasattr(dataset, 'categories'):
                ids_to_labels.update(dataset.categories)
            else:
                logger.warning("Dataset [{}] has no categories attribute, labels.json file won't be created".format(
                    dataset.__class__.__name__))

        if ids_to_labels:
            labels_file = os.path.join(output_dir, 'labels.json')
            logger.info("Saving labels mapping into {}".format(labels_file))
            with open(labels_file, 'w') as f:
                json.dump(ids_to_labels, f, indent=2)

