from pathlib import Path
import pandas as pd
import numpy as np
import torch
import time

from permgwas_mt.utils.input_config import InputConfig
from permgwas_mt.utils.result_types import ResultType
from permgwas_mt.preprocessing.data_loader import Dataset
from permgwas_mt.models.bivariate_gwas import BivariateGWAS

def run(config:InputConfig):

    print(f"Running analysis on device {config.device}\n\nStart loading data.")
    start = time.time()
    dataset = Dataset(config)
    have_data = time.time()
    print("Have dataset, elapsed time: ", have_data - start)
    print(f"Dataset size: genotype {dataset.X.shape}, phenotype {dataset.y.shape}")

    # pval_file = config.resolve_output_file(ResultType.P_VALUES)
    # summary_file = config.resolve_output_file(ResultType.SUMMARY_STATS)
    # print(pval_file)
    # print(summary_file)

    A_cov = torch.eye(2)

    def spectral_decomp(K: torch.Tensor):
        """
        Compute spectral decomposition of kinship matrix K=UDU^T

        :param K:
        :return: eigenvalues and U^T
        """
        # TODO move to data loader
        eigenvals, U = torch.linalg.eigh(K.to(device=self.device, dtype=torch.float64))
        return eigenvals, U.t()

    solver = BivariateGWAS(Y=dataset.y, K=dataset.K, Z=dataset.fixed,  A_cov=A_cov, device=config.device)

    print("Fitting Null Model (REML)...")
    solver.fit_null_model(save_intermediate=True)
    print(f"Null Model Beta: \n{solver.beta_null}")

    # ---------------------------------------------------------
    # 3. RUN GWAS SCANS
    # ---------------------------------------------------------

    # --- Test A: "Any Effect" (Standard 2-df test) ---
    # A_alt = Identity matrix [[1, 0], [0, 1]]
    f_any, p_any, beta_any = solver.run_full_gwas_scan(dataset.X, batch_size=5000)


    # --- Test B: "Common Effect" (1-df test) ---
    # Testing if the SNP has the exact same effect on both traits
    A_comm = torch.ones((2, 1))
    f_comm, p_comm, beta_comm = solver.run_full_gwas_scan(dataset.X, batch_size=5000, A_alt=A_comm)

    # --- Test C: "Nested Test" (Interaction Test) ---
    # Is 'Any Effect' significantly better than 'Common Effect'?
    # This identifies SNPs where traits respond differently.
    f_nest, p_nest, beta_nest = solver.run_full_gwas_scan(dataset.X, batch_size=5000, A_alt=torch.eye(2), A_null=A_comm)

    # ---------------------------------------------------------
    # 4. VIEW RESULTS
    # ---------------------------------------------------------
    best_snp_idx = np.argmin(p_any)
    print(f"\n--- Top SNP Results (Index {best_snp_idx}) ---")
    print(f"Any Effect P-value:    {p_any[best_snp_idx]:.2e}")
    print(f"Common Effect P-value: {p_comm[best_snp_idx]:.2e}")
    print(f"Interaction P-value:   {p_nest[best_snp_idx]:.2e}")
    print(f"Beta (Trait 1, Trait 2): {beta_any[best_snp_idx]}")
