import yaml
import pandas as pd
from datetime import datetime


def save_p_values(filepath, test_stats, p_values, betas, ses, chromosomes, positions, maf):
    n_effects = betas.shape[1] if betas.ndim > 1 else 1

    data = {
        'CHR': chromosomes,
        'POS': positions,
        'MAF': maf,
        'TEST_STAT': test_stats,
        'P-VAL': p_values
    }

    # 3. Dynamically Add Beta Columns
    # If 1 effect: BETA. If >1: BETA_1, BETA_2, etc.
    if n_effects == 1:
        data['BETA'] = betas.flatten()
        data['SE'] = ses.flatten()
    else:
        for i in range(n_effects):
            data[f'BETA_{i + 1}'] = betas[:,i]
            data[f'SE_{i + 1}']= ses[:,i]

    # 4. Create DataFrame and Save
    results_df = pd.DataFrame(data)
    results_df = results_df.sort_values(['CHR', 'POS'])

    results_df.to_csv(filepath, index=False)
    print(f"\nSaved {test_stats.shape[0]} SNPs with {n_effects} effect column(s) to {filepath}")
    return results_df


def save_max_test_stats(filepath, max_test_stats, min_p_vals, seeds):
    df = pd.DataFrame({'SEED': seeds,
                       'MAX_F_STAT': max_test_stats,
                       'MIN_P_VAL': min_p_vals,})
    df.to_csv(filepath, index=False)
    print(f"\nSaved min p-values for {max_test_stats.shape[0]} permutations to {filepath}")


def save_gwas_summary(filepath, genotype_file, phenotype_file, traits, n_samples, n_snps, maf_threshold,
                      metrics, final_loss, trait_design, trait_corr, n_perm, master_seed,
                      hypothesis_type=None, genomic_control=None, perm_thres=None, kinship_file=None,
                      covariate_file=None, covariate_list=None):

    #Saves a comprehensive summary of the GWAS run including
    #model parameters, hardware settings, and significance thresholds.

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "genotype_file": genotype_file.name,
        "phenotype_file": phenotype_file.name,
        "traits": traits,
    }
    if kinship_file is not None:
        summary["kinship_file"] = kinship_file.name
    if covariate_file is not None:
        summary["covariate_file"] = covariate_file.name
    if covariate_list is not None:
        summary["covariates"] = covariate_list
    summary.update({"maf_threshold": maf_threshold,
                    "trait_design": trait_design,
                    "n_samples": n_samples,
                    "n_snps": n_snps,
                    "trait_correlation": trait_corr,
                    "optimized parameters": {
                        "genetic covariance G": metrics["G"],
                        "residual covariance R": metrics["R"],
                        "final REML loss": final_loss,
                    },
                    "heritability": {
                        "trait_1": metrics["h2_1"],
                        "trait_2": metrics["h2_2"],
                    },
                    "genetic_correlation": metrics["rg"],
                    "residual_correlation": metrics["re"],
    })
    if hypothesis_type is not None:
        summary["hypothesis_type"] = hypothesis_type
        for key in genomic_control:
            summary[f"lambda_gc_{key}"] = genomic_control[key]
        summary["Bonferroni_threshold_05"] = 0.05/n_snps
        if len(perm_thres) > 0:
            summary["n_permutations"] = n_perm
            summary["perm_master_seed"] = int(master_seed)
            for key in perm_thres:
                summary[f"permutation_threshold_{key}_05"] = perm_thres[key]

    # 2. Print to Console (Formatted)
    print("\n" + "=" * 40)
    print("GWAS RUN SUMMARY")
    print("=" * 40)
    # This renders the dict as a clean YAML string for the terminal
    print(yaml.dump(summary, default_flow_style=False, sort_keys=False))
    print("=" * 40 + "\n")

    with open(filepath, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"Summary saved to {filepath}")
