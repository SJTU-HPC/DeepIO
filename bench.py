import argparse
import logging
import configparser
from deepio import bench_dataset
from deepio.report import PerfRecord


logging.basicConfig(format='[%(levelname)s]\t: %(message)s',
                    level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        dest='config_path',
                        default='./config.ini')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)
    logging.info(f"Read in config file from {args.config_path}.")
    for section_name in config.sections():
        bench_dataset(section_name, config[section_name])

    PerfRecord.print_results()


if __name__ == '__main__':
    main()