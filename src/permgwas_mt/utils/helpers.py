import torch
import gc
import logging

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
    elif test_type == "pleiotropy":
        # Tests if the 2-parameter model is better than the 1-parameter model
        return A_any, A_common
    else:
        raise ValueError(f"Unknown test type: {test_type}")

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


