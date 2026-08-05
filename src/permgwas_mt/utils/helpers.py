import torch
import gc
import logging

from permgwas_mt.utils.get_input_config import InputConfig


def print_setup(input_config: InputConfig):
    print("\n" + "=" * 50)
    print("|" + " " * 19 + "permGWASmt"+ " " * 19 + "|")
    print("=" * 50)
    print(f"Run multi-trait GWAS analysis on device {input_config.device}")
    print(f"- Genotype file: {input_config.genotype_file.name}")
    print(f"- Phenotype file: {input_config.phenotype_file.name}")
    print(f"- Traits: {input_config.traits[0]}, {input_config.traits[1]}")
    if input_config.covariate_file is not None:
        print(f"- Covariate file: {input_config.covariate_file.name}")
        print(f"- Covariates: {','.join(input_config.covariate_list)}")
    if input_config.kinship_file is not None:
        print(f"- Kinship file: {input_config.kinship_file.name}")
    if input_config.maf_threshold > 0:
        print(f"- Minor allele frequency filtering: {input_config.maf_threshold}")
    if not input_config.no_scan:
        print(f"- Test hypothesis: {input_config.hypothesis_type}")
    if input_config.n_permutations > 0:
        print(f"- Number of permutations: {input_config.n_permutations}")
    print("=" * 50 + "\n\nStart loading data")


def get_dtype(dtype: str):
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise RuntimeError(f"Unsupported dtype {dtype}")

def get_trait_design(trait_design="identity"):
    if trait_design == "identity":
        return torch.eye(2)
    else:
        raise ValueError(f"Unknown trait design matrix: {trait_design}")

def get_bivariate_hypotheses(test_type="any"):
    """
    Generates A_alt and A_null matrices for common bivariate tests.

    Types:
    - 'any': Tests if the SNP affects Trait 1 OR Trait 2 (2-df).
    - 'common': Tests if the SNP has the SAME effect on both traits (1-df).
    - 'pleiotropy': Alternative is 'any', Null is 'common'.
                   Tests if effects are significantly different (1-df).
    """
    # Any effect (Identity matrix)
    A_any = torch.eye(2)

    # Common effect (Single shared coefficient)
    A_common = torch.tensor([[1.0], [1.0]])

    if test_type == "any":
        return A_any, None
    elif test_type == "common":
        return A_common, None
    elif test_type == "specific":
        # Tests if the 2-parameter model is better than the 1-parameter model
        return A_any, A_common
    else:
        raise ValueError(f"Unknown test type: {test_type}")

def backtransform_effects(beta_alt, se_alt, A_alt, y_std):
    """
    Transform effect sizes and SEs from GWAS scan back to original scale

    :param beta_alt: (n_snps, k) - beta_alt from gwas scan
    :param se_alt:   (n_snps, k) - SEs from gwas scan
    :param A_alt:    (2, k) - trait design matrix
    :param y_std:    (2,) - standard deviation of traits
    :return: beta_original (n_snps, 2), se_original (n_snps, 2)
    """
    A_alt = A_alt.to(dtype=beta_alt.dtype)
    beta_std_full = beta_alt @ A_alt.t()              # (n_snps, 2)
    var_alt = se_alt.pow(2)                            # (n_snps, k)
    var_std_full = var_alt @ (A_alt.t() ** 2)           # (n_snps, 2)

    beta_original = beta_std_full * y_std.unsqueeze(0)               # (n_snps, 2)
    se_original = torch.sqrt(var_std_full) * y_std.unsqueeze(0)      # (n_snps, 2)

    return beta_original, se_original

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GWAS_Executor")

def robust_gwas_executor(func, *args, **kwargs):
    """
    Tries to run a solver function. If it hits OOM, it clears cache,
    reduces the relevant batch size, and retries.
    """
    try:
        return func(*args, **kwargs)

    except torch.cuda.OutOfMemoryError:
        # 1. Force release of all possible memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 2. Logic to determine which parameter to cut
        # Priority 1: Reduce permutation batch size (most effective for OOM)
        if 'perm_batch_size' in kwargs and kwargs['perm_batch_size'] > 1:
            old_val = kwargs['perm_batch_size']
            kwargs['perm_batch_size'] //= 2
            logger.warning(f"CUDA OOM: Reducing 'perm_batch_size' from {old_val} to {kwargs['perm_batch_size']}.")

        # Priority 2: Reduce SNP chunk size
        elif 'batch_size' in kwargs and kwargs['batch_size'] > 500:
            old_val = kwargs['batch_size']
            kwargs['batch_size'] //= 2
            logger.warning(f"CUDA OOM: Reducing 'batch_size' from {old_val} to {kwargs['batch_size']}.")

        else:
            logger.error("CUDA OOM: Cannot reduce batch sizes any further. Task failed.")
            raise RuntimeError("Out of Memory at minimum batch settings.")

        # 3. Recursive retry
        return robust_gwas_executor(func, *args, **kwargs)
