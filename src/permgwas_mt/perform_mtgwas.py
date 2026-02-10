import torch

from permgwas_mt.utils.input_config import InputConfig
from permgwas_mt.utils.result_types import ResultType
from permgwas_mt.preprocessing.data_loader import Dataset
from permgwas_mt.models.bivariate_gwas import BivariateGWAS
from permgwas_mt.utils.save_results import save_p_values, save_max_test_stats, save_gwas_summary
from permgwas_mt.utils.get_time import Timer, timed
import permgwas_mt.utils.helpers as helpers


def spectral_decomp(K: torch.Tensor, device):
    """
    Compute spectral decomposition of kinship matrix K=UDU^T

    :param K:
    :return: eigenvalues and U^T
    """
    # TODO move to data loader
    eigenvals, U = torch.linalg.eigh(K.to(device=device, dtype=torch.float64))
    return eigenvals, U.t()

# TODO load full X vs. batchwise
def run(input_config:InputConfig):
    device = torch.device(input_config.device)
    timer = Timer()
    print(f"Running analysis on device {input_config.device}\n\nStart loading data.")
    with timed(timer, "load data", use_cuda=(device.type == "cuda")):
        dataset = Dataset(input_config)
    timer.log("Loaded data.")
    print(f"Dataset size: genotype {dataset.X.shape}, phenotype {dataset.y.shape}")

    print("Start fitting null model.")
    with timed(timer, "fit data", use_cuda=(device.type == "cuda")):
        # TODO put in dataloader
        eigenvals, Ut = spectral_decomp(dataset.K, input_config.device)
        # TODO put in cli args
        A_cov = torch.eye(2)
        solver = BivariateGWAS(Y=dataset.y, Ut=Ut, eigenvals=eigenvals, Z=dataset.fixed, A_cov=A_cov,
                               device=input_config.device, dtype=torch.float32)
        solver.fit_null_model(full_wald=True)
    timer.log("Have null model.")

    # TODO put in cli args
    hypothesis_type = "any"
    print(f"Start GWAS scan. Test for {hypothesis_type} effects.")
    A_alt, A_null = helpers.get_bivariate_hypotheses(hypothesis_type)

    # TODO test on gpu
    with timed(timer, "gwas scan", use_cuda=(device.type == "cuda")):
        """f_stats, p_values, betas = solver.run_full_gwas_scan(genotype_data=dataset.X,
                                                             n_snps=dataset.n_snps,
                                                             A_alt=A_alt,
                                                             A_null=A_null,
                                                             batch_size=5000)"""

        f_stats, p_values, betas = helpers.robust_gwas_executor(solver.run_full_gwas_scan,
                                                                 genotype_data=dataset.X,
                                                                 n_snps=dataset.n_snps,
                                                                 A_alt=A_alt,
                                                                 A_null=A_null,
                                                                 batch_size=5000
                                                                 )
    timer.log("Have p-values.")

    # TODO in cli args
    master_seed = 142
    if input_config.n_permutations > 0:
        with timed(timer, "permutations", use_cuda=(device.type == "cuda")):
            print("Start permutations.")
            # TODO test on gpu
            """max_stats, perm_seeds = solver.run_permutation_gwas(genotype_data=dataset.X,
                                                                n_perms=input_config.n_permutations,
                                                                n_snps=dataset.n_snps,
                                                                master_seed=master_seed,
                                                                A_alt=A_alt,
                                                                A_null=A_null,
                                                                batch_size=5000,
                                                                perm_batch_size=100
                                                                )"""

            max_stats, perm_seeds = helpers.robust_gwas_executor(solver.run_permutation_gwas,
                                                            genotype_data=dataset.X,
                                                            n_perms=input_config.n_permutations,
                                                            n_snps=dataset.n_snps,
                                                            master_seed=master_seed,
                                                            A_alt=A_alt,
                                                            A_null=A_null,
                                                            batch_size=5000,
                                                            perm_batch_size=100
                                                            )
        timer.log("Have max test statistics")
    else:
        max_stats, perm_seeds = None, None


    print("Finished GWAS scan. Save results")
    pval_file = input_config.resolve_output_file(ResultType.P_VALUES)
    summary_file = input_config.resolve_output_file(ResultType.SUMMARY_STATS)
    result_df = save_p_values(filepath=pval_file,
                              f_stats=f_stats,
                              p_values=p_values,
                              betas=betas,
                              chromosomes=dataset.chromosomes,
                              positions=dataset.positions)
    save_gwas_summary(filepath=summary_file,
                      genotype_file=input_config.genotype_file,
                      phenotype_file=input_config.phenotype_file,
                      traits=input_config.traits,
                      n_samples=dataset.n_samples,
                      n_snps=dataset.n_snps,
                      maf_threshold=input_config.maf_threshold,
                      l_G=solver.l_G.detach().cpu().numpy().tolist(),
                      l_R=solver.l_R.detach().cpu().numpy().tolist(),
                      hypothesis_type=hypothesis_type,
                      n_perm=input_config.n_permutations,
                      master_seed=master_seed,
                      max_stats=max_stats,
                      kinship_file=input_config.kinship_file,
                      covariate_file=input_config.covariate_file,
                      covariate_list=input_config.covariate_list,
                      )
    if input_config.n_permutations > 0:
        max_test_stat_file = input_config.resolve_output_file(ResultType.MAX_TEST_STATS)
        save_max_test_stats(filepath=max_test_stat_file, max_test_stats=max_stats, seeds=perm_seeds)

    timer.report()

