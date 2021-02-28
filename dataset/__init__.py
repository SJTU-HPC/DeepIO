from .imagenet import bench_imagenet

def bench_dataset(dataset_name, config):
    if dataset_name == "imagenet":
        bench_imagenet(config)