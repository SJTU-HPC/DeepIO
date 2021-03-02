from .dataset.imagenet import bench_imagenet
from .dataset.random import bench_random

def bench_dataset(dataset_name, config):
    if dataset_name == "imagenet":
        bench_imagenet(config)
    if dataset_name == "small_random":
        bench_random(config, "small")
    if dataset_name == "medium_random":
        bench_random(config, "medium")
    if dataset_name == "large_random":
        bench_random(config, "large")
