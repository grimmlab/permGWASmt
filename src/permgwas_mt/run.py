from pathlib import Path
import pandas as pd
import torch
import time

from permgwas_mt.utils.input_config import InputConfig
from permgwas_mt.utils.result_types import ResultType
from permgwas_mt.preprocessing.data_loader import Dataset

def run(config:InputConfig):

    print(f"Running analysis on device {config.device}\n\nStart loading data.")
    start = time.time()
    dataset = Dataset(config)
    have_data = time.time()
    print("Have dataset, elapsed time: ", have_data - start)
    print(f"Dataset size: genotype {dataset.X.shape}, phenotype {dataset.y.shape}")

    pval_file = config.resolve_output_file(ResultType.P_VALUES)
    summary_file = config.resolve_output_file(ResultType.SUMMARY_STATS)
    print(pval_file)
    print(summary_file)