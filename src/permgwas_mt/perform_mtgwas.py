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
        helpers.print_setup(input_config)
        dataset = Dataset(input_config, dtype=dtype)
    timer.log("Loaded data.")

    with timed(timer, "fit data", use_cuda=(device.type == "cuda")):
        print(f"\nStart fitting null model for {dataset.n_samples} samples and {dataset.fixed.shape[1]} fixed effect(s).")
        A_cov = helpers.get_trait_design(input_config.trait_design)
        solver = BivariateGWAS(Y=dataset.y, Ut=dataset.Ut, eigenvals=dataset.evals, Z=dataset.fixed, A_cov=A_cov,
                               device=input_config.device, dtype=dtype)
        final_loss = solver.fit()
        metrics = solver.get_variance_components(y_std=dataset.y_std)
    timer.log("Have null model.")

    perm_thres = {}
    genomic_control = {}
    if not input_config.no_scan:
        if input_config.hypothesis_type == "all":
            hypothesis_type = ["any", "common", "specific"]
        else:
            hypothesis_type = [input_config.hypothesis_type]

        for test_type in hypothesis_type:
            with timed(timer, f"gwas scan ({test_type})", use_cuda=(device.type == "cuda")):
                print(f"\nStart GWAS scan. Test {dataset.n_snps} SNPs for {test_type} effects.")
                A_alt, A_null = helpers.get_bivariate_hypotheses(test_type)
                test_stats, p_values, betas, ses, lambda_gc = helpers.robust_gwas_executor(
                    solver.run_full_gwas_scan,
                    lambda bs: dataset.get_genotype_stream(batch_size=bs),
                    n_snps=dataset.n_snps,
                    A_alt=A_alt,
                    A_null=A_null,
                    batch_size=input_config.batch_size,
                )
                beta_original, se_original = helpers.backtransform_effects(torch.from_numpy(betas),
                                                                           torch.from_numpy(ses), A_alt, dataset.y_std)
                genomic_control[test_type] = lambda_gc
            timer.log("Have p-values.")
            pval_file = input_config.resolve_output_file(ResultType.P_VALUES, test_type)
            result_df = save_p_values(filepath=pval_file,
                                      test_stats=test_stats,
                                      p_values=p_values,
                                      betas=beta_original,
                                      ses=se_original,
                                      chromosomes=dataset.metadata['chr'],
                                      positions=dataset.metadata['pos'],
                                      maf=dataset.metadata['maf'])


            if input_config.n_permutations > 0:
                with timed(timer, f"permutations ({test_type})", use_cuda=(device.type == "cuda")):
                    print(f"\nCompute min p-values for {input_config.n_permutations} permutations.")
                    max_stats, min_p, perm_seeds = helpers.robust_gwas_executor(
                        solver.run_permutation_gwas,
                        lambda bs: dataset.get_genotype_stream(batch_size=bs),
                        n_perms=input_config.n_permutations,
                        n_snps=dataset.n_snps,
                        master_seed=input_config.master_seed,
                        A_alt=A_alt,
                        A_null=A_null,
                        lambda_gc=lambda_gc,
                        batch_size=input_config.batch_size,
                        perm_batch_size=input_config.perm_batch_size,
                    )
                    perm_thres[test_type] = float(np.percentile(min_p, 5))
                timer.log("Have min p-values.")
                max_test_stat_file = input_config.resolve_output_file(ResultType.MAX_TEST_STATS, test_type)
                save_max_test_stats(filepath=max_test_stat_file,
                                    max_test_stats=max_stats,
                                    min_p_vals=min_p,
                                    seeds=perm_seeds)

    summary_file = input_config.resolve_output_file(ResultType.SUMMARY_STATS)
    save_gwas_summary(filepath=summary_file,
                      genotype_file=input_config.genotype_file,
                      phenotype_file=input_config.phenotype_file,
                      traits=input_config.traits,
                      n_samples=dataset.n_samples,
                      n_snps=dataset.n_snps,
                      maf_threshold=input_config.maf_threshold,
                      metrics=metrics,
                      final_loss=final_loss.item(),
                      trait_design=input_config.trait_design,
                      trait_corr=dataset.trait_corr,
                      n_perm=input_config.n_permutations,
                      master_seed=input_config.master_seed,
                      hypothesis_type=input_config.hypothesis_type,
                      genomic_control=genomic_control,
                      perm_thres=perm_thres,
                      kinship_file=input_config.kinship_file,
                      covariate_file=input_config.covariate_file,
                      covariate_list=input_config.covariate_list,
                      )

    if input_config.time_report:
        timer.report()
