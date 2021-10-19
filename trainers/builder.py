import copy
import paddle

from utils.registry import Registry

TRAINER = Registry("TRAINER")


def build_trainer(cfg):
    if hasattr(cfg, 'trainer'):
        name = getattr(cfg, 'trainer')
    else:
        name = 'Trainer'
    trainer = TRAINER.get(name)(cfg)
    return trainer