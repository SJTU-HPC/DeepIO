# DeepIO

IO Benchmark for Deep Learning Application on HPC cluster.

## Requirements

```shell
pip3 install -r requirements.txt
```

## Usage

Please refer to help message:

```
python3 bench.py --help
```

### Example

* Generate Random Dataset

```shell
python3 generate.py
```

* Run

```shell
python3 bench.py --config ./config.ini
```