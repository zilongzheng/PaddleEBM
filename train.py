import os
import sys
import argparse

from utils.config import get_config
from utils.setup import setup
from utils.logger import get_logger

from trainers.builder import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleEBM')
    parser.add_argument('-c',
                        '--config-file',
                        metavar="FILE",
                        help='config file path')
    # cuda setting
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    # checkpoint and log
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--load',
                        type=str,
                        default=None,
                        help='put the path to resuming file if needed')
    # for evaluation
    parser.add_argument('--val-interval',
                        type=int,
                        default=1,
                        help='run validation every interval')
    parser.add_argument('--evaluate-only',
                        action='store_true',
                        default=False,
                        help='skip validation during training')
    # config options
    parser.add_argument('opts',
                        help='See config for all options',
                        default=None,
                        nargs=argparse.REMAINDER)

    #for inference
    parser.add_argument("--source_path",
                        default="",
                        metavar="FILE",
                        help="path to source image")
    parser.add_argument("--reference_dir",
                        default="",
                        help="path to reference images")
    parser.add_argument("--model_path", default=None, help="model for loading")

    args = parser.parse_args()

    return args

def main(args, cfg):
    # init environment, include logger, dynamic graph, seed, device, train or test mode...
    setup(args, cfg)
    logger = get_logger()
    logger.info(cfg)
    
    # build trainer
    trainer = build_trainer(cfg)

    # continue train or evaluate, checkpoint need contain epoch and optimizer info
    if args.resume:
        trainer.resume(args.resume)
    # evaluate or finute, only load generator weights
    elif args.load:
        trainer.load(args.load)

    if args.evaluate_only:
        trainer.test()
        return
    # training, when keyboard interrupt save weights
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        if trainer.by_epoch:
            trainer.save(trainer.current_epoch)
    trainer.close()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.config_file)

    main(args, cfg)
    