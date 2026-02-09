import yaml
import torch
import pathlib
import importlib
import inspect
import numpy as np


def get_bivariate_hypotheses(test_type="any"):
    """
    Generates A_alt and A_null matrices for common bivariate tests.

    Types:
    - 'any': Tests if the SNP affects Trait 1 OR Trait 2 (2-df).
    - 'common': Tests if the SNP has the SAME effect on both traits (1-df).
    - 'trait1_only': Tests if the SNP affects Trait 1, ignoring Trait 2 (1-df).
    - 'pleiotropy': Alternative is 'any', Null is 'common'.
                   Tests if effects are significantly different (1-df).
    """
    # Any effect (Identity matrix)
    A_any = torch.tensor([[1.0, 0.0],
                          [0.0, 1.0]])

    # Common effect (Single shared coefficient)
    A_common = torch.tensor([[1.0],
                             [1.0]])

    # Trait 1 only
    A_t1 = torch.tensor([[1.0],
                         [0.0]])

    # Trait 2 only
    A_t2 = torch.tensor([[0.0],
                         [1.0]])

    if test_type == "any":
        return A_any, None
    elif test_type == "common":
        return A_common, None
    elif test_type == "trait1_only":
        return A_t1, None
    elif test_type == "pleiotropy":
        # Tests if the 2-parameter model is better than the 1-parameter model
        return A_any, A_common
    else:
        raise ValueError(f"Unknown test type: {test_type}")