
import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
import json

from .mydataloader_mix import RMOT_Dataset_mix
from .mydataloader import RMOT_Dataset

class mySampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas= None,
                 rank=None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset,num_replicas,rank,shuffle,seed,drop_last)
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.dataset.set_epoch(epoch)
        # self.__init__(self.dataset,self.num_replicas,self.rank,self.shuffle,self.seed,self.drop_last)

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = mySampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )


    sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val,shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        #persistent_workers = True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers = True,

    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, None


def build_dataset(is_train, config):
    if config.DATA.DATASET.startswith('kitti') or config.DATA.DATASET=='dance': 
        func = RMOT_Dataset
    else:
        func = RMOT_Dataset_mix
    if config.EVAL_MODE:
        dataset = func(mode='test',opt=config)
    else:
        if is_train:
            dataset = func(mode='train',opt=config)
        else:
            dataset = func(mode='val',opt=config)
    

    nb_classes = 2


    return dataset, nb_classes

