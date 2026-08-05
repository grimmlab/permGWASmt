import torch
import gc
from tqdm import tqdm
import scipy.stats as stats
import numpy as np

from permgwas_mt.models.univariate_reml import univariate_reml


class BivariateGWAS:

    EPS_D = 1e-6  # numerical safety for D
    RHO_MAX = 0.995  # keeps rho_r away from +-1

    def __init__(self, Y: torch.Tensor, Ut: torch.Tensor, eigenvals: torch.Tensor, Z: torch.Tensor, A_cov: torch.Tensor,
                 device: str, dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_samples = Y.shape[0]

        # eigenvalues and transposed vectors from kinship spectral decomposition
        # transform y and fixed effects: y' = U^Ty, X' = U^TX
        self.eigenvals = eigenvals.to(device=self.device, dtype=self.dtype)
        self.Ut = Ut.to(device=self.device, dtype=self.dtype)
        self.y = self._transform_phenotype(Y=Y.to(device=self.device, dtype=self.dtype))
        self.X_batch, self.n_fixed = self._transform_covariates(Z=Z.to(device=self.device, dtype=self.dtype),
                                                                A_cov=A_cov.to(device=self.device, dtype=self.dtype))
        # initialize covariance matrices
        self.l_R = None
        self.l_D = None
        self.theta = None
        self.eps_var = None  # floor of residual variance
        self._initialize_params()

        # initialize remaining variables
        self.beta_null = None
        self.Phi = None
        self.diag_null = None
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

    @staticmethod
    def _inverse_softplus(z):
        """
        # compute inverse of softplus, since z needs to be > 0
        # log(exp(z) - 1) = z + log(1-exp(-z))
        """
        return z + torch.log1p(-torch.exp(-z))

    def _initialize_params(self):
        """
        Initialize parameters using univariate REML variances as a warm start (l_R, theta, l_D)
        """
        with torch.no_grad():
            # get univariate variances
            s2g1, s2e1 = univariate_reml(y=self.y, trait_idx=0, eigenvals=self.eigenvals,
                                         n_samples=self.n_samples, n_fixed=self.n_fixed)
            s2g2, s2e2 = univariate_reml(y=self.y, trait_idx=1, eigenvals=self.eigenvals,
                                         n_samples=self.n_samples, n_fixed=self.n_fixed)

            # floor residual variance
            frac_floor = 1e-3  # make sure that h^2 < 1 - frac_floor
            self.eps_var = torch.stack([frac_floor * (s2g1 + s2e1), frac_floor * (s2g2 + s2e2),
                                        ]).to(device=self.device, dtype=self.dtype)

            # phenotypic correlation of residuals y * 1/(evals + delta)
            res1 = self.y[:, 0, 0] * (1.0 / (self.eigenvals + s2e1 / s2g1))
            res2 = self.y[:, 1, 0] * (1.0 / (self.eigenvals + s2e2 / s2g2))

            # Standard Pearson correlation
            cos = torch.nn.CosineSimilarity(dim=0)
            rho = cos(res1 - res1.mean(), res2 - res2.mean())
            rho = torch.clamp(rho, -0.9, 0.9)

            # initialize R and G
            R_init = torch.stack([
                torch.stack([s2e1, rho * torch.sqrt(s2e1 * s2e2)]),
                torch.stack([rho * torch.sqrt(s2e1 * s2e2), s2e2]),
            ])
            G_init = torch.stack([
                torch.stack([s2g1, rho * torch.sqrt(s2g1 * s2g2)]),
                torch.stack([rho * torch.sqrt(s2g1 * s2g2), s2g2]),
            ])

            # get variances from R_init
            r1 = torch.clamp(R_init[0, 0], min=self.eps_var[0] * 1.01)
            r2 = torch.clamp(R_init[1, 1], min=self.eps_var[1] * 1.01)
            cov_r = R_init[0, 1]
            rho_r = torch.clamp(cov_r / torch.sqrt(r1 * r2), -self.RHO_MAX * 0.98, self.RHO_MAX * 0.98)

            x_r1 = self._inverse_softplus(r1 - self.eps_var[0])
            x_r2 = self._inverse_softplus(r2 - self.eps_var[1])
            x_rho_r = torch.atanh(rho_r / self.RHO_MAX)
            self.l_R = torch.stack([x_r1, x_r2, x_rho_r]).to(device=self.device, dtype=self.dtype
                                                             ).clone().requires_grad_(True)

            # transform G_init with R basis: G' = C^-1 G C^-T
            C_init = torch.linalg.cholesky(R_init)
            tmp = torch.linalg.solve(C_init, G_init)  # C^-1 G_init
            G_prime = torch.linalg.solve(C_init.t(), tmp.t()).t()  # (C^-1 G_init) C^-T
            G_prime = 0.5 * (G_prime + G_prime.t())  # symmetric
            # get eigen decomposition
            eigvals, eigvecs = torch.linalg.eigh(G_prime)
            d1, d2 = eigvals[0], eigvals[1]
            Q_init = eigvecs

            # sanity check for Q -> det=1
            if torch.det(Q_init) < 0:
                Q_init = torch.stack([Q_init[:, 0], -Q_init[:, 1]], dim=1)

            # get eigenvalues of G_prime, and angle theta of rotation matrix Q
            theta_init = torch.atan2(Q_init[1, 0], Q_init[0, 0])

            x_d1 = self._inverse_softplus(torch.clamp(d1, min=self.EPS_D * 1.01))
            x_d2 = self._inverse_softplus(torch.clamp(d2, min=self.EPS_D * 1.01))

            self.l_D = torch.stack([x_d1, x_d2]).to(device=self.device, dtype=self.dtype).clone().requires_grad_(True)
            self.theta = theta_init.to(device=self.device, dtype=self.dtype).clone().requires_grad_(True)

    def _clear_vram(self):
        """
        Cleanup for PyTorch caching allocator
        """
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()

    def build_R_chol(self):
        """ build Cholesky decomp of residual covariance matrix"""
        r1 = torch.nn.functional.softplus(self.l_R[0]) + self.eps_var[0]
        r2 = torch.nn.functional.softplus(self.l_R[1]) + self.eps_var[1]
        rho_r = torch.tanh(self.l_R[2]) * self.RHO_MAX

        sqrt_r1 = torch.sqrt(r1)
        sqrt_r2 = torch.sqrt(r2)
        c10 = rho_r * sqrt_r2
        c11 = sqrt_r2 * torch.sqrt(1.0 - rho_r ** 2)

        C = torch.stack([
            torch.stack([sqrt_r1, torch.zeros_like(sqrt_r1)]),
            torch.stack([c10, c11]),
        ])
        logdet_C = torch.log(sqrt_r1) + torch.log(c11)
        return C, logdet_C

    def build_Q(self):
        """ build Q matrix with G' = QDQ^T"""
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        Q = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ])
        return Q

    def build_D(self):
        """ build diagonal matrix D with G' = QDQ^T"""
        d1 = torch.nn.functional.softplus(self.l_D[0]) + self.EPS_D
        d2 = torch.nn.functional.softplus(self.l_D[1]) + self.EPS_D
        return torch.stack([d1, d2])

    def reml_loss(self):
        """
        compute REML log likelihood for optimization with canonic parametrization

        :return: loss, beta_hat, Phi, diag_i (n,2), weighted_Xz, res_z, XVX
        """
        C, logdet_C = self.build_R_chol()  # (2,2)
        Q = self.build_Q()  # (2,2)
        D = self.build_D()  # (2,)

        # Phi = C^{-T} Q  ->  solve C^T Phi = Q
        Phi = torch.linalg.solve(C.t(), Q)  # (2,2)
        # lambda_i*D + I
        diag_i = self.eigenvals.view(-1, 1) * D.view(1, 2) + 1.0  # (n,2)
        inv_diag = 1.0 / diag_i  # (n,2)

        # Transform X and y in canonical space: yz = Phi^T y, Xz = Phi^T X
        Phi_t = Phi.t()
        Xz = torch.matmul(Phi_t.unsqueeze(0), self.X_batch)  # (n,2,c)
        yz = torch.matmul(Phi_t.unsqueeze(0), self.y)  # (n,2,1)

        # X^T V^-1 X and X^T V^-1 y via (diagonal) canonic covariance
        weighted_Xz = Xz * inv_diag.unsqueeze(-1)  # (n,2,c)
        XVX = torch.einsum('nkc,nkd->cd', weighted_Xz, Xz)  # (c,c)
        XVy = torch.einsum('nkc,nk->c', weighted_Xz, yz.squeeze(-1)).unsqueeze(-1)  # (c,1)

        #  generalized least squares beta
        beta_hat = torch.linalg.solve(XVX, XVy)  # (c,1)

        # residuals
        res_z = yz - torch.matmul(Xz, beta_hat)  # (n,2,1)
        resVres = torch.einsum('nk,nk->', res_z.squeeze(-1).pow(2), inv_diag)

        # logdet(V) = sum_i [ 2*logdet(C) + sum_k log(lambda_i*d_k + 1) ]
        logdet_V = self.n_samples * 2.0 * logdet_C + torch.sum(torch.log(diag_i))
        logdet_XVX = torch.logdet(XVX)

        loss = 0.5 * (logdet_V + logdet_XVX + resVres)
        return loss, beta_hat, Phi, diag_i, weighted_Xz, res_z, XVX

    def _optimize_reml(self, line_search, lr, max_rounds=5, iters_per_round=2000, stable_rounds_needed=2,
                       loss_tol=1e-5):
        """
        Repeat REML optimization for null model over several rounds to check convergence
        """
        prev_summary, stable_count = None, 0
        final_loss = None
        for _ in range(max_rounds):
            optimizer = torch.optim.LBFGS([self.l_R, self.theta, self.l_D], lr=lr, max_iter=iters_per_round,
                                          line_search_fn=line_search)

            def closure():
                optimizer.zero_grad()
                loss, *_ = self.reml_loss()
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                final_loss, *_ = self.reml_loss()
                D = self.build_D()
                C, _ = self.build_R_chol()
                R = C @ C.t()
                summary = torch.cat([D, R.flatten(), self.theta.view(1)])

            if prev_summary is not None and (summary - prev_summary).abs().max().item() < loss_tol:
                stable_count += 1
                if stable_count >= stable_rounds_needed:
                    break
            else:
                stable_count = 0
            prev_summary = summary

        return final_loss.item()

    def fit_null_model(self,
                       configs=(('strong_wolfe', 2.0),
                                ('strong_wolfe', 1.0),
                                (None,           0.05)),
                       max_iter=5000, max_rounds_single_fit=5, stable_rounds=2, loss_tol=1e-5):
        """
        optimize REML loss for null model, optionally save intermediate results X^TV^-1 and (X^TV^-1X)^-1
        """
        l_R_init, theta_init, l_D_init = self.l_R.clone(), self.theta.clone(), self.l_D.clone()
        best_loss, best_state = float('inf'), None

        # try out different line_search and lr params to optimize l_R, l_D and theta
        for line_search, lr in configs:
            with torch.no_grad():
                self.l_R = l_R_init.clone().requires_grad_(True)
                self.theta = theta_init.clone().requires_grad_(True)
                self.l_D = l_D_init.clone().requires_grad_(True)

                final_loss = self._optimize_reml(line_search=line_search, lr=lr, max_rounds=max_rounds_single_fit,
                                                 iters_per_round=max_iter // max_rounds_single_fit,
                                                 stable_rounds_needed=stable_rounds, loss_tol=loss_tol)

                if final_loss < best_loss:
                    best_loss = final_loss
                    best_state = (self.l_R.detach().clone(), self.theta.detach().clone(),
                                  self.l_D.detach().clone())

        # finalize best result
        self.l_R = best_state[0].requires_grad_(True)
        self.theta = best_state[1].requires_grad_(True)
        self.l_D = best_state[2].requires_grad_(True)

        with torch.no_grad():
            final_loss, beta_null, Phi, diag_i, weighted_Xz, res_z, XVX = self.reml_loss()

        self.beta_null = beta_null
        self.Phi = Phi
        self.diag_null = diag_i
        inv_diag = 1.0 / diag_i

        # Vres Phi (inv_diag * res_z), res_z = Phi^T @ res_original
        self.Vres = torch.einsum('ik,nk->ni', Phi, inv_diag * res_z.squeeze(-1))  # (n,2)

        # compute (X^T * V^-1 * X)^-1 for null model
        self.XVX_null = torch.linalg.inv(XVX) # (c,c)
        # compute X^T V^-1 = weighted_Xz_i^T Phi^T
        XV = torch.einsum('nkf,jk->nfj', weighted_Xz, Phi)  # (n,c,2)
        self.XV_null = XV.reshape(self.n_samples, -1)  # (n,c*2)

        self._clear_vram()
        return final_loss

    def get_variance_components(self, y_std=None):
        """
        Compute heritability, genetic and residual correlation
        """
        with torch.no_grad():
            C ,_ = self.build_R_chol()
            R = C @ C.t()
            Q = self.build_Q()
            D = self.build_D()

            # G = C Q diag(D) Q^T C^T
            G = C @ Q @ torch.diag(D) @ Q.t() @ C.t()

            g1, g2 = G[0, 0], G[1, 1]
            r1, r2 = R[0, 0], R[1, 1]

            h2_1 = g1 / (g1 + r1)
            h2_2 = g2 / (g2 + r2)
            rho_g = G[0, 1] / torch.sqrt(g1 * g2)
            rho_r = R[0, 1] / torch.sqrt(r1 * r2)

            if y_std is not None:
                scale = torch.outer(y_std, y_std)  # (2,2)
                G = (G * scale)
                R = (R * scale)

            return {
                "h2_1": h2_1.item(),
                "h2_2": h2_2.item(),
                "rg": rho_g.item(),
                "re": rho_r.item(),
                "G": G.cpu().numpy().tolist(),
                "R": R.cpu().numpy().tolist(),
                "D_canonical": D.cpu().numpy(),
                "theta": self.theta.item(),
            }

    def _compute_df(self, A_alt, A_null=None):
        """
        compute degrees of freedom
        :param A_alt: trait design matrix of alternative model (2,k)
        :param A_null: trait design matrix of null model (2,j) or None
        :return: df1
        """

        if A_null is not None:
            Q_full, _ = torch.linalg.qr(torch.cat([A_null, A_alt], dim=1))
            P = Q_full[:, A_null.shape[1]:A_null.shape[1] + (A_alt.shape[1] - A_null.shape[1])]
        else:
            P, _ = torch.linalg.qr(A_alt)

        return P.shape[1]

    @staticmethod
    def _apply_genomic_control(test_stats:np.array, df1:int, lambda_gc:float =None, warn_threshold:float =0.1):
        """
        Correct test statistics by genomic control value lambda_gc, and compute p-values

        :param test_stats:
        :param df1:
        :param warn_threshold:
        :return: p-values, lambda_gc
        """
        if lambda_gc is None:
            lambda_gc = np.median(test_stats) / stats.chi2.ppf(0.5, df1)
            if abs(lambda_gc - 1.0) > warn_threshold:
                print(f"WARNING: lambda_GC={lambda_gc:.3f}. Applied genomic control.")
        lambda_gc_applied = max(lambda_gc, 1.0)
        corrected_stats = test_stats / lambda_gc_applied
        p_vals = stats.chi2.sf(corrected_stats, df1)
        return p_vals, lambda_gc

    def _gwas_scan(self, g_batch: torch.Tensor, A_alt: torch.Tensor, A_null: torch.Tensor = None):
        """
        Performs Bivariate GWAS Scan using the F-test

        :param g_batch: batch of SNPs (n, num_snps)
        :param A_alt: design matrix for alternative model
        :param A_null: design matrix for null model
        :return:test statistic, beta, se
        """

        inv_diag = 1.0 / self.diag_null  # (n_snps,2)

        # base score: g^T * V^-1 * res and base information matrix Phi(g^T * D^-1 * g^T)Phi^T = g^T * V^-1 * g^T
        gVres = torch.mm(g_batch.t(), self.Vres)  # (n_snps, 2)
        gVg = torch.mm(g_batch.pow(2).t(), inv_diag)  # (n_snps,2)
        gVg = torch.einsum('ik,nk,jk->nij', self.Phi, gVg, self.Phi)  # (n_snps,2,2)

        # correct for full Wald test
        gXV = torch.mm(g_batch.t(), self.XV_null).reshape(-1, self.n_fixed, 2)  # (n_snps,c,2)
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

        test_stat = torch.clamp(chi_alt - chi_null, min=0.0) if chi_null is not None else chi_alt
        return test_stat.cpu().numpy(), beta_alt.squeeze(-1).cpu().numpy(), se_alt.cpu().numpy()

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
        all_stats, all_betas, all_se = [], [], []

        df1 = self._compute_df(A_alt=A_alt, A_null=A_null)

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
                test_stats, betas, se = self._gwas_scan(g_batch=g_batch, A_alt=A_alt, A_null=A_null)
                all_stats.append(test_stats)
                all_betas.append(betas)
                all_se.append(se)
                self._clear_vram()

        # get p-values
        stats_all = np.concatenate(all_stats)
        p_vals, lambda_gc = self._apply_genomic_control(stats_all, df1)

        return stats_all, p_vals, np.concatenate(all_betas), np.concatenate(all_se), lambda_gc.item()

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

        inv_diag = 1.0 / self.diag_null  # (n,2)

        # Compute score (p, m, 2)
        gVres = torch.matmul(g_perm.transpose(1, 2), self.Vres)  # (p,m,2)
        # Compute information (p, m, 2, 2)
        gVg = torch.matmul(g_perm.pow(2).transpose(1, 2), inv_diag)  # (p,m,2)
        gVg = torch.einsum('ik,pmk,jk->pmij', self.Phi, gVg, self.Phi)  # (p,m,2,2)

        #gVg = torch.matmul(g_perm.pow(2).transpose(1, 2),
        #                 self.V_inv_null).reshape(n_perm, n_snps, 2, 2)  # (p,m,2,2)

        # Confounding Correction
        gXV = torch.matmul(g_perm.transpose(1, 2),
                           self.XV_null).reshape(n_perm, n_snps, -1, 2)  # (p,m,c,2)
        gVg -= torch.matmul(gXV.transpose(-2, -1), torch.matmul(self.XVX_null, gXV))  # (p,m,2,2)
        del gXV
        del g_perm

        # compute chi squared for specific null
        if A_null is not None:
            gVres_null = torch.matmul(gVres, A_null)
            gVg_null = torch.matmul(A_null.t(), torch.matmul(gVg, A_null))
            chi_null = torch.matmul(torch.matmul(gVres_null.unsqueeze(-2), torch.linalg.inv(gVg_null)),
                                    gVres_null.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            del gVres_null, gVg_null
        else:
            chi_null = None

        # compute chi square for alternative
        gVres = torch.matmul(gVres, A_alt)
        gVg = torch.matmul(A_alt.t(), torch.matmul(gVg, A_alt))
        chi_alt = torch.matmul(torch.matmul(gVres.unsqueeze(-2), torch.linalg.inv(gVg)),
                               gVres.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        del gVres, gVg

        if chi_null is not None:
            stat = torch.clamp(chi_alt - chi_null, min=0.0)
        else:
            stat = chi_alt
        # Return max across SNPs (dim=1)
        return torch.max(stat, dim=1)[0]


    def run_permutation_gwas(self, genotype_data, batch_size: int, n_snps: int, n_perms: int, perm_batch_size: int,
                             A_null: torch.Tensor = None, A_alt: torch.Tensor = None, lambda_gc: float = 1.0,
                             master_seed: int=142):
        """
        Compute Westfall & Young permutation-based threshold.

        """
        if A_alt is None:
            A_alt = torch.eye(2, device=self.device, dtype=self.dtype)
        else:
            A_alt = A_alt.to(device=self.device, dtype=self.dtype)
        if A_null is not None:
            A_null = A_null.to(device=self.device, dtype=self.dtype)
        df1 = self._compute_df(A_alt=A_alt, A_null=A_null)

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

        # compute min p-values
        max_stats = max_stats.cpu().numpy()
        min_p_vals, _ = self._apply_genomic_control(max_stats, df1, lambda_gc)

        return max_stats, min_p_vals, perm_seeds

