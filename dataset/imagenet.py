import time
import os
from functools import partial

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.multiprocessing import Process

from .multiproc import init_process


def imagenet_main(config):

    dataset_path = config["path"]
    batch_size = config.getint("batch_size")
    num_workers = config.getint("num_workers")
    bench_iter = config.getint("bench_iter")

    trainset = torchvision.datasets.ImageNet(
        root=dataset_path,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]),
    )

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              sampler=train_sampler,
                                              shuffle=(train_sampler is None),
                                              num_workers=num_workers,
                                              drop_last=True)

    num_samples = 0

    torch.distributed.barrier()
    start_ts = time.time()
    end_ts = time.time()
    total_time = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        torch.distributed.barrier()
        if i > 5:
            num_samples += inputs.size(0)
            total_time += time.time() - end_ts
        if i > bench_iter:
            break
        end_ts = time.time()

    samples_per_second = torch.FloatTensor([num_samples / total_time])

    torch.distributed.all_reduce(samples_per_second)

    if torch.distributed.get_rank() == 0:
        print(samples_per_second)


def bench_imagenet(config):
    if not config.getboolean('enable'):
        return
    world_size = config.getint("multi_proc")
    processes = []
    for rank in range(world_size):
        p = Process(target=init_process,
                    args=(rank, world_size, config, imagenet_main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()