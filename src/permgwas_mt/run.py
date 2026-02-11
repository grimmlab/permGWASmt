import argparse
import sys
from permgwas_mt.utils.input_config import InputConfig
from permgwas_mt.perform_mtgwas import run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-trait permGWAS analysis."
    )

    # File paths
    parser.add_argument("-x", "--genotype_file", type=str,
                        help="Genotype file (.h5, .csv, .ped/.map, .bed/.bim/.fam)")
    parser.add_argument("-y", "--phenotype_file", type=str,
                        help="Phenotype file (.csv, .txt, .pheno)")
    parser.add_argument("-cov", "--covariate_file", type=str, default=None,
                        help="Optional covariate file (.csv)")
    parser.add_argument("-k", "--kinship_file", type=str, default=None,
                        help="Optional kinship matrix file (.h5, .csv) "
                             "If not provided, compute realized relationship kernel.")

    # Phenotype and covariates
    parser.add_argument("-traits", "--traits","--phenotypes", nargs=2,
                        help="Two phenotype names in the phenotype file")
    parser.add_argument("-cov_list", "--covariate_list", nargs="+", default=None,
                        help="Optional list of covariates to use from covariate file")

    # General options
    parser.add_argument("-maf", "--maf_threshold", type=int, choices=range(0, 31), default=None,
                        help="Optional minor allele frequency threshold")
    parser.add_argument("-perm", "--n_permutations", type=int, default=None,
                        help="Number of permutations")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output folder (default: ./results)")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Result file name (default: pheno1_pheno2)")
    parser.add_argument("--hypothesis_type", type=str, default=None,
                        help="Optional hypothesis type. Valid options are 'any', 'common', 'specific' and 'all'. "
                             "(default: 'any')")
    parser.add_argument("--trait_design", type=str, default=None,
                        help="Optional trait design matrix type. Currently only support 'identity'.")

    # Compute settings
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cpu or cuda:N), (default: cpu)")
    parser.add_argument("--load_genotype", action="store_true", default=None,
                        help="Load genotype file completely during preprocessing. Otherwise load genotype batch-wise "
                             "for computations to save memory. Batch-wise loading only possible for .h5 genotype files "
                             "and if kinship is provided.")
    parser.add_argument('-batch', '--batch_size', type=int, default=None,
                        help='Number of SNPs to work on simultaneously (default: 10000)')
    parser.add_argument('-batch_perm', '--perm_batch_size', type=int, default=None,
                        help='Number of SNPs to work on simultaneously for permutations (default: 1000)')
    parser.add_argument('--master_seed', type=int, default=None,
                        help='Master seed used for permutations (default: will be randomly generated)')

    # Config file
    parser.add_argument("--config", "--config_file", type=str, default=None,
                        help="YAML config file")

    args = parser.parse_args()
    return vars(args)

# TODO A_cov,
# TODO dytpe/precision

def main():
    print('Start parsing arguments')
    cli_args = parse_args()
    config_path = cli_args.pop("config", None)
    try:
        config = InputConfig.from_sources(cli_args=cli_args, config_path=config_path)
    except Exception as e:
        print(f"Error in input configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # run main workflow
    run(config)


if __name__ == "__main__":
    main()