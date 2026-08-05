import torch


def univariate_reml(y, trait_idx, eigenvals, n_samples, n_fixed, tol=1e-5, max_iter=100):
    """
    1D Golden Section Search for univariate REML initialization.
    Finds the optimal delta = var_e / var_g.
    """
    y_sq = y[:, trait_idx, 0].pow(2)
    n_p = n_samples - n_fixed

    # Golden ratio
    invphi = 0.6180339887
    invphi2 = 0.3819660113

    # Search range for delta (var_e / var_g)
    # From 1e-4 (high heritability) to 1e4 (low heritability)
    a, b = -5.0, 5.0  # We search in log10 space

    def get_ll(log_delta):
        delta = 10 ** log_delta
        w = 1.0 / (eigenvals + delta)
        # Analytical sigma_g^2
        s2g = (y_sq * w).sum() / n_p
        # REML Log-Likelihood
        ll = -0.5 * (n_p * torch.log(s2g) + torch.log(eigenvals + delta).sum() + n_p)
        return -ll  # Minimize negative log-likelihood

    # Standard 1D optimization loop
    h = b - a
    x1, x2 = a + invphi2 * h, a + invphi * h
    f1, f2 = get_ll(x1), get_ll(x2)

    for _ in range(max_iter):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            h = b - a
            x1 = a + invphi2 * h
            f1 = get_ll(x1)
        else:
            a, x1, f1 = x1, x2, f2
            h = b - a
            x2 = a + invphi * h
            f2 = get_ll(x2)
        if h < tol: break

    # Final parameters
    opt_log_delta = (a + b) / 2
    delta = 10 ** opt_log_delta
    w = 1.0 / (eigenvals + delta)
    s2g = (y_sq * w).sum() / n_p
    s2e = s2g * delta
    return s2g, s2e