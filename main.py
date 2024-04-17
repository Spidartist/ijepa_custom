# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import sys

import pprint
import yaml

from src.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(fname, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[0].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f'called-params {fname}')

    trainer = Trainer(cfg_path=fname)
    trainer.train()



if __name__ == '__main__':
    print(sys.executable)
    args = parser.parse_args()

    process_main(args.fname, args.devices)
