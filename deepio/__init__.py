from .dataset.imagenet import bench_imagenet
from .report import PerfRecord

def bench_dataset(dataset_name, config):
    if dataset_name == "imagenet":
        bench_imagenet(config)

    PerfRecord.print_results()