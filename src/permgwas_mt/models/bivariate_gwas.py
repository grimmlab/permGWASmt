import torch
import gc
from tqdm import tqdm
import scipy.stats as stats
import numpy as np


class BivariateGWAS:

    def __init__(self, Y: torch.Tensor, Ut: torch.Tensor, eigenvals: torch.Tensor, Z: torch.Tensor, A_cov: torch.Tensor,
                 device: str, dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_samples = Y.shape[0]
        # eigenvalues and transposed vectors from kinship spectral decomposition
        self.eigenvals = eigenvals.to(device=self.device, dtype=self.dtype)
        self.Ut = Ut.to(device=self.device, dtype=self.dtype)
        # transform y and fixed effects
        self.y = self._transform_phenotype(Y=Y.to(device=self.device, dtype=self.dtype))
        self.X_batch, self.n_fixed = self._transform_covariates(Z=Z.to(device=self.device, dtype=self.dtype),
                                                                A_cov=A_cov.to(device=self.device, dtype=self.dtype))
        # initialize Cholesky factors
        self.l_G = torch.zeros(3, device=self.device, dtype=self.dtype, requires_grad=True)
        self.l_R = torch.zeros(3, device=self.device, dtype=self.dtype, requires_grad=True)
        self._initialize_params()
        # initialize remaining variables
        self.beta_null = None
        self.V_inv_null = None  # (n,4)
        self.XV_null = None  # (n,2*c)
        self.XVX_null = None  # (c,c)
        self.Vres = None  # (n,2)

        self._clear_vram()

    def _transform_phenotype(self, Y: torch.Tensor):
        """
        Transform phenotype matrix Y with Ut and reshape (n,2,1)

        :param Y:
        :return:
        """
        y = torch.mm(self.Ut, Y)
        return y.unsqueeze(-1)

    def _transform_covariates(self, Z: torch.Tensor, A_cov: torch.Tensor):
        """
        Transform covariates with U^T and compute Kronecker structure

        :param Z:
        :param A_cov:
        :return: X_batch and number of fixed effects
        """
        # transform core covariate matrix
        Z_trans = torch.mm(self.Ut, Z)  # (n,c)
        # create X_batch using Kronecker product logic
        # for each individual have X_i = A \otimes z_i
        # z_i is row vector (1,c)
        # X_i is (2,k*c)
        X_expanded = A_cov.to(self.device).unsqueeze(0).unsqueeze(3) * Z_trans.unsqueeze(1).unsqueeze(2)  # (n,2,k,c)
        # reshape to (n,2,k*c)
        X_batch = X_expanded.reshape(self.n_samples, 2, -1)
        return X_batch, X_batch.shape[2]

    def _initialize_params(self):
        """
        Initialize Cholesky factors L_G, L_R using univariate REML variances as a warm start

        :return: L_G, L_R
        """
        with torch.no_grad():
            # get univariate variances
            s2g1, s2e1 = self._univariate_reml(trait_idx=0)
            s2g2, s2e2 = self._univariate_reml(trait_idx=1)

            # phenotypic correlation of residuals y * 1/(evals + delta)
            res1 = self.y[:, 0, 0] * (1.0 / (self.eigenvals + s2e1 / s2g1))
            res2 = self.y[:, 1, 0] * (1.0 / (self.eigenvals + s2e2 / s2g2))

            # Standard Pearson correlation
            cos = torch.nn.CosineSimilarity(dim=0)
            rho = cos(res1 - res1.mean(), res2 - res2.mean())
            rho = torch.clamp(rho, -0.9, 0.9)  # Keep it stable

            #Initialize Cholesky parameters
            self.l_G[0], self.l_R[0] = torch.sqrt(s2g1), torch.sqrt(s2e1)
            # Covariance (l1): rho * sqrt(var1 * var2)
            self.l_G[1] = rho * torch.sqrt(s2g2)
            self.l_R[1] = rho * torch.sqrt(s2e2)

            # For Trait 2 (l2): sqrt(var2 - l1^2) to satisfy L*L.T = Var
            # We use a small epsilon to ensure the square root is valid
            self.l_G[2] = torch.sqrt(torch.clamp(s2g2 - self.l_G[1] ** 2, min=1e-6))
            self.l_R[2] = torch.sqrt(torch.clamp(s2e2 - self.l_R[1] ** 2, min=1e-6))

    def _univariate_reml(self, trait_idx, tol=1e-5, max_iter=100):
        """
        1D Golden Section Search for univariate REML initialization.
        Finds the optimal delta = var_e / var_g.
        """
        y_sq = self.y[:, trait_idx, 0].pow(2)
        n_p = self.n_samples - self.n_fixed

        # Golden ratio
        invphi = 0.6180339887
        invphi2 = 0.3819660113

        # Search range for delta (var_e / var_g)
        # From 1e-4 (high heritability) to 1e4 (low heritability)
        a, b = -5.0, 5.0  # We search in log10 space

        def get_ll(log_delta):
            delta = 10 ** log_delta
            w = 1.0 / (self.eigenvals + delta)
            # Analytical sigma_g^2
            s2g = (y_sq * w).sum() / n_p
            # REML Log-Likelihood
            ll = -0.5 * (n_p * torch.log(s2g) + torch.log(self.eigenvals + delta).sum() + n_p)
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
        w = 1.0 / (self.eigenvals + delta)
        s2g = (y_sq * w).sum() / n_p
        s2e = s2g * delta
        return s2g, s2e

    def _clear_vram(self):
        """
        Cleanup for PyTorch caching allocator
        """
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    @staticmethod
    def vec_to_sym_matrix(l: torch.Tensor):
        """
        compute C = LL^T for lower triangular L

        :param l: 3 dim vector containing elements of L
        :return: LL^T
        """
        L = torch.zeros((2,2), device=l.device)
        L[0,0], L[1,0], L[1,1] = l[0], l[1], l[2]
        return L @ L.t()

    def reml_loss(self):
        """
        compute REML log likelihood for optimization

        :return: REML loss, beta_hat, V_inv
        """
        G = self.vec_to_sym_matrix(self.l_G)
        R = self.vec_to_sym_matrix(self.l_R)

        # V_i = lambda_i*G + R
        V_batch = self.eigenvals.view(self.n_samples, 1, 1) * G + R
        V_inv = torch.inverse(V_batch)

        # X^T * V^-1 * X and X^T * V^-1 * y
        XV = torch.bmm(self.X_batch.transpose(1,2), V_inv)  # (n,c,2)
        XVX = torch.bmm(XV, self.X_batch).sum(dim=0)  # (c,c)
        XVy = torch.bmm(XV, self.y).sum(dim=0)  # (c,1)

        #  generalized least squares beta
        beta_hat = torch.linalg.solve(XVX, XVy)

        # residuals
        res = self.y - torch.bmm(self.X_batch, beta_hat.expand(self.n_samples, -1, -1))

        # log-likelihood
        logdet_V = torch.logdet(V_batch).sum()
        logdet_XVX = torch.logdet(XVX)
        resVres = torch.bmm(res.transpose(1,2), torch.bmm(V_inv, res)).sum()

        return 0.5 * (logdet_V + logdet_XVX + resVres), beta_hat, V_inv, XV, XVX, res

    def fit_null_model(self, full_wald=True):
        """
        optimize REML loss for null model, optionally save intermediate results X^TV^-1 and (X^TV^-1X)^-1
        :param full_wald: boolean, whether to save intermediate results for full Wald Chi square scan
        """
        optimizer = torch.optim.LBFGS([self.l_G, self.l_R], lr=0.05, max_iter=1000)
        def closure():
            optimizer.zero_grad()
            loss, _, _, _, _, _ = self.reml_loss()
            loss.backward()
            return loss
        optimizer.step(closure)

        _, self.beta_null, V_inv_null, XV, XVX, res = self.reml_loss()

        # precalculate and store constants for all future scans
        # weighted residuals under null V^-1 * res with V_inv_null: (n,2,2), res: (n,2,1)
        self.Vres = torch.bmm(V_inv_null, res).squeeze(-1)  # (n,2)
        # flat inverse variance
        self.V_inv_null = V_inv_null.reshape(self.n_samples, 4)  # (n,4)
        if full_wald:
            # compute (X^T * V^-1 * X)^-1 and (X^TV^-1)^T for null model
            self.XVX_null = torch.linalg.inv(XVX)
            self.XV_null = XV.transpose(1, 2).reshape(self.n_samples, -1)  # (n,2*c)

        del V_inv_null, XV, XVX, res
        self._clear_vram()

    def get_metrics(self):
        """Returns heritability (h2), genetic correlation (rg) and residual correlation (re)."""
        G, R = self.vec_to_sym_matrix(self.l_G), self.vec_to_sym_matrix(self.l_R)
        h2 = [G[i, i] / (G[i, i] + R[i, i]) for i in range(2)]
        rg = G[0, 1] / torch.sqrt(G[0, 0] * G[1, 1])
        re = R[0, 1] / torch.sqrt(R[0, 0] * R[1, 1])
        return {"h2_1": h2[0].item(), "h2_2": h2[1].item(), "rg": rg.item(), "re": re.item()}

    def _gwas_scan(self, g_batch: torch.Tensor, A_alt: torch.Tensor, A_null: torch.Tensor = None):
        """
        Performs Bivariate GWAS Scan using the F-test

        :param g_batch: batch of SNPs (n, num_snps)
        :param A_alt: design matrix for alternative model
        :param A_null: design matrix for null model
        :return:test statistics, p-values, betas
        """

        # base score: g^T * V^-1 * res and base information matrix g^T * V^-1 * g^T
        gVres = torch.mm(g_batch.t(), self.Vres)  # (n_snps, 2)
        gVg = torch.mm(g_batch.pow(2).t(), self.V_inv_null).reshape(-1, 2, 2)  # (n_snps,2,2)

        # correct for full Wald test
        if self.XVX_null is not None:
            # gXV (n_snps,c,2)
            gXV = torch.mm(g_batch.t(), self.XV_null).reshape(-1, self.n_fixed, 2)
            gVg -= torch.matmul(torch.matmul(gXV.transpose(1, 2), self.XVX_null), gXV)
            del gXV
        del g_batch

        # project to null space and compute Wald Chi-square: (gVres)^T * (gVg)^-1 * gVres
        if A_null is not None:
            gVres_null = torch.matmul(gVres, A_null)
            gVg_null = torch.matmul(A_null.t(), torch.matmul(gVg, A_null))
            chi_null = torch.bmm(gVres_null.unsqueeze(1),
                                 torch.linalg.solve(gVg_null, gVres_null.unsqueeze(-1))).flatten()  # (n_snps,)
            del gVres_null, gVg_null
        else:
            chi_null = None

        # project to alternative space and compute Wald Chi-square: (gVres)^T * (gVg)^-1 * gVres
        gVres = torch.matmul(gVres, A_alt)
        gVg = torch.matmul(A_alt.t(), torch.matmul(gVg, A_alt))
        gVg = torch.linalg.inv(gVg)  # get inverse gVg^-1
        beta_alt = torch.bmm(gVg, gVres.unsqueeze(-1))  # (n_snps,k,1)
        chi_alt = torch.bmm(gVres.unsqueeze(1), beta_alt).flatten()  # (n_snps,)

        # get standard errors sqrt(diag(gVg^-1))
        if gVg.shape[1] == 1:
            se_alt = torch.sqrt(gVg.reshape(-1,1))
        else:
            se_alt = torch.sqrt(torch.diagonal(gVg, dim1=-2, dim2=-1))

        del gVres, gVg

        if chi_null is not None:
            f_stat = torch.clamp(chi_alt - chi_null, min=0.0)
            df1 = float(A_alt.shape[1] - A_null.shape[1])
        else:
            f_stat = chi_alt
            df1 = float(A_alt.shape[1])

        # compute F-statistic
        f_stat = f_stat / df1
        f_stat = f_stat.cpu().numpy()
        df2 = (2 * self.n_samples) - self.n_fixed - A_alt.shape[1]
        p_vals = stats.f.sf(f_stat, df1, df2)
        return f_stat, p_vals, beta_alt.squeeze(-1).cpu().numpy(), se_alt.cpu().numpy()

    @staticmethod
    def _check_genotypes(genotype_data, n_snps, batch_size):
        """
        check if genotypes are tensor or generator, set up iterable batches
        :return:
        """
        is_tensor = isinstance(genotype_data, torch.Tensor)
        if is_tensor:
            n_snps = genotype_data.shape[1]
            iterable = range(0, n_snps, batch_size)
        else:
            iterable = genotype_data
        n_batches = int(np.ceil(n_snps / batch_size))
        return is_tensor, iterable, n_batches

    def run_full_gwas_scan(self, genotype_data, batch_size: int, n_snps: int, A_null: torch.Tensor = None,
                           A_alt: torch.Tensor = None):
        """
        Run full GWAS scan over batches of SNPs. Accept genotype_data as a tensor containing full SNP matrix or
        as a generator, loading SNPs batch-wise from file.

        :param genotype_data:
        :param batch_size:
        :param n_snps:
        :param A_null:
        :param A_alt:
        :return:
        """

        if A_alt is None:
            A_alt = torch.eye(2, device=self.device, dtype=self.dtype)
        else:
            A_alt = A_alt.to(device=self.device, dtype=self.dtype)
        if A_null is not None:
            A_null = A_null.to(device=self.device, dtype=self.dtype)
        all_f, all_p, all_betas, all_se = [], [], [], []

        # check if genotypes is tensor or generator
        is_tensor, iterable, n_batches = self._check_genotypes(genotype_data, n_snps, batch_size)
        pbar = tqdm(iterable, total=n_batches, desc="GWAS Scan")

        with torch.no_grad():
            for entry in pbar:
                if is_tensor:
                    g_batch = genotype_data[:, entry : entry + batch_size].to(device=self.device, dtype=self.dtype)
                else:
                    g_batch = entry.to(device=self.device, dtype=self.dtype, non_blocking=True)
                g_batch = torch.mm(self.Ut, g_batch)
                test_stats, p_vals, betas, se = self._gwas_scan(g_batch=g_batch, A_alt=A_alt, A_null=A_null)
                all_f.append(test_stats)
                all_p.append(p_vals)
                all_betas.append(betas)
                all_se.append(se)

                self._clear_vram()

        return np.concatenate(all_f), np.concatenate(all_p), np.concatenate(all_betas), np.concatenate(all_se)

    def _permutation_scan(self, g_batch: torch.Tensor, seeds: np.array, A_alt: torch.Tensor,
                          A_null: torch.Tensor = None):
        """
        Permutation GWAS scan over batch of SNPs and batch of permutations

        :param g_batch:
        :param seeds:
        :param A_alt:
        :param A_null:
        :return:
        """
        n_perm = len(seeds)
        n_snps = g_batch.shape[1]

        # Set predefined seeds and generate local batch indices
        idx_list = []
        for s in seeds:
            torch.manual_seed(int(s))
            idx_list.append(torch.randperm(self.n_samples, device=self.device))
        idx_list = torch.stack(idx_list)

        # Get shuffled SNPs
        g_perm = g_batch[idx_list, :]  # (p,n,m)
        del idx_list,

        # Compute Score (B, M, 2) and Information (B, M, 2, 2)
        gVres = torch.matmul(g_perm.transpose(1, 2), self.Vres)  # (p,m,2)
        gVg = torch.matmul(g_perm.pow(2).transpose(1, 2),
                         self.V_inv_null).reshape(n_perm, n_snps, 2, 2)  # (p,m,2,2)

        # Confounding Correction
        if self.XVX_null is not None:
            gXV = torch.matmul(g_perm.transpose(1, 2),
                               self.XV_null).reshape(n_perm, n_snps, -1, 2)  # (p,m,c,2)
            gVg -= torch.matmul(gXV.transpose(-2, -1), torch.matmul(self.XVX_null, gXV))  # (p,m,2,2)
            del gXV
        del g_perm

        # compute chi squared for specific null
        if A_null is not None:
            gVres_null = torch.matmul(gVres, A_null)
            gVg_null = torch.matmul(A_null.t(), torch.matmul(gVg, A_null))
            chi_null = torch.matmul(torch.matmul(gVres_null.unsqueeze(-2), torch.inverse(gVg_null)),
                                    gVres_null.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            del gVres_null, gVg_null
        else:
            chi_null = None

        # compute chi square for alternative
        gVres = torch.matmul(gVres, A_alt)
        gVg = torch.matmul(A_alt.t(), torch.matmul(gVg, A_alt))
        chi_alt = torch.matmul(torch.matmul(gVres.unsqueeze(-2), torch.inverse(gVg)),
                               gVres.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        del gVres, gVg

        if chi_null is not None:
            stat = torch.clamp(chi_alt - chi_null, min=0.0)
        else:
            stat = chi_alt
        # Return max across SNPs (dim=1)
        return torch.max(stat, dim=1)[0]


    def run_permutation_gwas(self, genotype_data, batch_size: int, n_snps: int, n_perms: int, perm_batch_size: int,
                             A_null: torch.Tensor = None, A_alt: torch.Tensor = None, master_seed: int=142):
        """
        Compute Westfall & Young permutation-based threshold.

        """
        if A_alt is None:
            A_alt = torch.eye(2, device=self.device, dtype=self.dtype)
        else:
            A_alt = A_alt.to(device=self.device, dtype=self.dtype)
        if A_null is not None:
            A_null = A_null.to(device=self.device, dtype=self.dtype)
            df1 = float(A_alt.shape[1] - A_null.shape[1])
        else:
            df1 = float(A_alt.shape[1])

        # get list of seeds for permutations
        rng = np.random.default_rng(master_seed)
        perm_seeds = rng.integers(low=0, high=2 ** 31, size=n_perms)

        max_stats = torch.zeros(n_perms, device=self.device, dtype=self.dtype)

        is_tensor, iterable, n_batches = self._check_genotypes(genotype_data, n_snps, batch_size)
        pbar = tqdm(iterable, total=n_batches, desc="Permutation Scan")

        with torch.no_grad():
            for entry in pbar:
                if is_tensor:
                    g_batch = genotype_data[:, entry: entry + batch_size].to(device=self.device, dtype=self.dtype)
                else:
                    g_batch = entry.to(device=self.device, dtype=self.dtype, non_blocking=True)
                g_batch = torch.mm(self.Ut, g_batch)

                # go through perm sub-batches
                for p_idx in range(0, n_perms, perm_batch_size):
                    n_perm_batch = min(perm_batch_size, n_perms - p_idx)
                    batch_seeds = perm_seeds[p_idx:p_idx + n_perm_batch]
                    # get max test stats for current perm and SNP batch
                    perm_batch_max = self._permutation_scan(g_batch=g_batch,seeds=batch_seeds,
                                                            A_alt=A_alt, A_null=A_null)
                    # update global max tensor
                    max_stats[p_idx:p_idx + n_perm_batch] = torch.max(max_stats[p_idx:p_idx + n_perm_batch],
                                                                      perm_batch_max)
                del g_batch
                self._clear_vram()

        # compute F-statistic and p-values
        max_stats = max_stats / df1
        max_stats = max_stats.cpu().numpy()
        df2 = (2 * self.n_samples) - self.n_fixed - A_alt.shape[1]
        min_p_vals = stats.f.sf(max_stats, df1, df2)

        return max_stats, min_p_vals, perm_seeds

