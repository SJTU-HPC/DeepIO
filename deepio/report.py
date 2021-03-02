import logging

from .texttable import Texttable

class PerfRecord():
    __conf = {
        "results": {}
    }

    @staticmethod
    def add_item(bench_name, sample_per_second):
        PerfRecord.__conf["results"][bench_name] = sample_per_second

    @staticmethod
    def print_results():
        if PerfRecord.__conf["results"]:
            table = Texttable()
            rows = [["Dataset Name", "Sample Per Second"]]
            
            for dataset_name in PerfRecord.__conf["results"]:
                rows.append([dataset_name, PerfRecord.__conf["results"][dataset_name]])

            table.add_rows(rows)

            logging.info("DeepIO Benchmark Summary:")
            print(table.draw())