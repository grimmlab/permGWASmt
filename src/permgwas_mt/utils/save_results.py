import yaml
import numpy as np
import pandas as pd
from datetime import datetime


def save_p_values(filepath, f_stats, p_values, betas, ses, chromosomes, positions):
    n_effects = betas.shape[1] if betas.ndim > 1 else 1

    data = {
        'CHR': chromosomes,
        'POS': positions,
        'F_STAT': f_stats,
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
    print(f"Saved {f_stats.shape[0]} SNPs with {n_effects} effect column(s) to {filepath}")
    return results_df


def save_max_test_stats(filepath, max_test_stats, seeds):
    df = pd.DataFrame({'SEED': seeds,
                       'MAX_F_STAT': max_test_stats})
    df.to_csv(filepath, index=False)
    print(f"Saved max test statistics for {max_test_stats.shape[0]} permutations to {filepath}")


def save_gwas_summary(filepath, genotype_file, phenotype_file, traits, n_samples, n_snps, maf_threshold, l_G, l_R,
                      metrics, trait_design, trait_corr, n_perm, master_seed, hypothesis_type=None, perm_thres=None,
                      kinship_file=None, covariate_file=None, covariate_list=None):

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
        summary["covariate_list"] = covariate_list
    summary.update({"n_samples": n_samples,
                    "n_snps": n_snps,
                    "maf_threshold": maf_threshold,
                    "trait_design": trait_design,
                    "trait_correlation": trait_corr,
                    "variance_components": {
                        "l_G": l_G,
                        "l_R": l_R,
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
        summary["Bonferroni_threshold_05"]: 0.05/n_snps
        if len(perm_thres) > 0:
            summary["n_perm"] = n_perm
            summary["master_seed"] = int(master_seed)
            for key in perm_thres:
                summary[f"permutation_threshold_{key}_05"] = perm_thres[key]

    # 2. Print to Console (Formatted)
    print("\n" + "=" * 30)
    print("GWAS RUN SUMMARY")
    print("=" * 30)
    # This renders the dict as a clean YAML string for the terminal
    print(yaml.dump(summary, default_flow_style=False, sort_keys=False))
    print("=" * 30 + "\n")

    with open(filepath, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"Summary saved to {filepath}")