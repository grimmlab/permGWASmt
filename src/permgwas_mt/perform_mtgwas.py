import torch
import numpy as np

from permgwas_mt.utils.get_input_config import InputConfig
from permgwas_mt.utils.result_types import ResultType
from permgwas_mt.preprocessing.data_loader import Dataset
from permgwas_mt.models.bivariate_gwas import BivariateGWAS
from permgwas_mt.utils.save_results import save_p_values, save_max_test_stats, save_gwas_summary
from permgwas_mt.utils.get_time import Timer, timed
import permgwas_mt.utils.helpers as helpers


# TODO test on gpu / M1 - timer??
# TODO PLINK
# TODO encoding check genotype
# todo allow 2 pheno files


def run(input_config:InputConfig):
    device = torch.device(input_config.device)
    dtype = helpers.get_dtype(input_config.dtype)
    timer = Timer()
    with timed(timer, "load data", use_cuda=(device.type == "cuda")):
        print(f"Running analysis on device {input_config.device}\n\nStart loading data.")
        dataset = Dataset(input_config, dtype=dtype)
    timer.log("Loaded data.")

    with timed(timer, "fit data", use_cuda=(device.type == "cuda")):
        print(f"\nStart fitting null model for {dataset.n_samples} samples and {dataset.fixed.shape[1]} fixed effect(s).")
        A_cov = helpers.get_trait_design(input_config.trait_design)
        solver = BivariateGWAS(Y=dataset.y, Ut=dataset.Ut, eigenvals=dataset.evals, Z=dataset.fixed, A_cov=A_cov,
                               device=input_config.device, dtype=dtype)
        solver.fit_null_model(full_wald=True)
        metrics = solver.get_metrics()
    timer.log("Have null model.")

    perm_thres = {}
    if not input_config.no_scan:
        if input_config.hypothesis_type == "all":
            hypothesis_type = ["any", "common", "specific"]
        else:
            hypothesis_type = [input_config.hypothesis_type]

        for test_type in hypothesis_type:
            with timed(timer, f"gwas scan ({test_type})", use_cuda=(device.type == "cuda")):
                print(f"\nStart GWAS scan. Test {dataset.n_snps} SNPs for {test_type} effects.")
                A_alt, A_null = helpers.get_bivariate_hypotheses(test_type)
                f_stats, p_values, betas, ses = helpers.robust_gwas_executor(solver.run_full_gwas_scan,
                                                                         genotype_data=dataset.get_genotype_stream(),
                                                                         n_snps=dataset.n_snps,
                                                                         A_alt=A_alt,
                                                                         A_null=A_null,
                                                                         batch_size=input_config.batch_size,
                                                                         )
                pval_file = input_config.resolve_output_file(ResultType.P_VALUES, test_type)
                result_df = save_p_values(filepath=pval_file,
                                          f_stats=f_stats,
                                          p_values=p_values,
                                          betas=betas,
                                          ses=ses,
                                          chromosomes=dataset.metadata['chr'],
                                          positions=dataset.metadata['pos'],
                                          maf=dataset.metadata['maf'])
            timer.log("Have p-values.")


            if input_config.n_permutations > 0:
                with timed(timer, f"permutations ({test_type})", use_cuda=(device.type == "cuda")):
                    print(f"Compute min p-values for {input_config.n_permutations} permutations.")
                    max_stats, min_p, perm_seeds = helpers.robust_gwas_executor(solver.run_permutation_gwas,
                                                                        genotype_data=dataset.get_genotype_stream(),
                                                                        n_perms=input_config.n_permutations,
                                                                        n_snps=dataset.n_snps,
                                                                        master_seed=input_config.master_seed,
                                                                        A_alt=A_alt,
                                                                        A_null=A_null,
                                                                        batch_size=input_config.batch_size,
                                                                        perm_batch_size=input_config.perm_batch_size,
                                                                        )
                    max_test_stat_file = input_config.resolve_output_file(ResultType.MAX_TEST_STATS, test_type)
                    save_max_test_stats(filepath=max_test_stat_file,
                                        max_test_stats=max_stats,
                                        min_p_vals=min_p,
                                        seeds=perm_seeds)
                    perm_thres[test_type] = float(np.percentile(min_p, 5))
                timer.log("Have min p-values.")

    summary_file = input_config.resolve_output_file(ResultType.SUMMARY_STATS)
    save_gwas_summary(filepath=summary_file,
                      genotype_file=input_config.genotype_file,
                      phenotype_file=input_config.phenotype_file,
                      traits=input_config.traits,
                      n_samples=dataset.n_samples,
                      n_snps=dataset.n_snps,
                      maf_threshold=input_config.maf_threshold,
                      l_G=solver.l_G.detach().cpu().numpy().tolist(),
                      l_R=solver.l_R.detach().cpu().numpy().tolist(),
                      metrics=metrics,
                      trait_design=input_config.trait_design,
                      trait_corr=dataset.trait_corr,
                      n_perm=input_config.n_permutations,
                      master_seed=input_config.master_seed,
                      hypothesis_type=input_config.hypothesis_type,
                      perm_thres=perm_thres,
                      kinship_file=input_config.kinship_file,
                      covariate_file=input_config.covariate_file,
                      covariate_list=input_config.covariate_list,
                      )

    if input_config.time_report:
        timer.report()
