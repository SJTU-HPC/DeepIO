import time
import logging
from multiprocessing import Process, Manager

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from ..multiproc import init_process
from ..report import PerfRecord


def random_main(config, return_dict):

    dataset_path = config["path"]
    batch_size = config.getint("batch_size")
    num_workers = config.getint("num_workers")
    bench_iter = config.getint("bench_iter")

    trainset = torchvision.datasets.ImageFolder(
        root=dataset_path,
        transform=transforms.Compose([
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
    timestamp = time.time()
    total_time = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        torch.distributed.barrier()
        if i > 5:
            num_samples += inputs.size(0)
            total_time += time.time() - timestamp
        if i > bench_iter:
            break
        timestamp = time.time()

    samples_per_second_ = torch.FloatTensor([num_samples / total_time])

    torch.distributed.all_reduce(samples_per_second_)

    if torch.distributed.get_rank() == 0:
        return_dict[0] = samples_per_second_.item()


def bench_random(config, size):
    if not config.getboolean('enable'):
        return

    logging.info(f"Start benchmark of Ramdom ({size}).")

    manager = Manager()
    return_dict = manager.dict()

    world_size = config.getint("multi_proc")
    processes = []
    for rank in range(world_size):
        p = Process(target=init_process,
                    args=(rank, world_size, config, return_dict,
                          random_main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    PerfRecord.add_item(f"Random ({size})", return_dict.values()[0])