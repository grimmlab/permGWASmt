import torch
import numpy as np
from torch.xpu import device


class BivariateGWAS:

    def __init__(self, Y: torch.Tensor, K: torch.Tensor, Z: torch.Tensor, A_cov: torch.Tensor,
                 device: torch.device):
        self.device = device
        self.n_samples = Y.shape[0]
        self.n_covariates = Z.shape[1]
        self.Z_raw = Z
        self.eigenvals, self.Ut = self._spectral_decomp(K=K)
        # transform y and fixed effects
        self.y = self._transform_phenotype(Y=Y)
        self.X_batch = self._transform_covariates(Z=self.Z_raw, A_cov=A_cov)
        # initialize Cholesky factors
        self.l_G = torch.zeros(3, device=self.device, requires_grad=True)
        self.l_R = torch.zeros(3, device=self.device, requires_grad=True)
        self._initialize_params(Y=Y)
        # initialize remaining variables
        self.final_loss = None
        self.beta_null = None
        self.V_inv_null = None

    def _spectral_decomp(self, K: torch.Tensor):
        """
        Compute spectral decomposition of kinship matrix K

        :param K:
        :return:
        """
        eigenvals, U = torch.linalg.eigh(K.to(self.device))
        return eigenvals, U.t()

    def _transform_phenotype(self, Y: torch.Tensor):
        """
        Transform phenotype matrix Y with Ut and stack columns

        :param Y:
        :return:
        """
        y = torch.mm(self.Ut, Y.to(self.device))
        return y.unsqueeze(-1)

    def _transform_covariates(self, Z: torch.Tensor, A_cov: torch.Tensor):
        # transform core covariate matrix
        Z_trans = torch.mm(self.Ut, Z.to(self.device))  # (n,c)
        k = A_cov.shape[1]
        # create X_batch using Kronecker product logic
        # for each individual have X_i = A \otimes z_i
        # z_i is row vector (1,c)
        # X_i is (2,k*c)
        Z_expanded = A_cov.to(self.device).unsqueeze(0).unsqueeze(3) * Z_trans.unsqueeze(1).unsqueeze(2)  # (n,2,k,c)
        # reshape to (n,2,k*c)
        return Z_expanded.reshape(self.n_samples, k, self.n_covariates, 2, k * self.n_covariates)

    def _initialize_params(self, Y: torch.Tensor):
        """
        Initialize Cholesky factors L_G, L_R

        :param Y: phenotype matrix
        :return: L_G, L_R
        """
        # TODO change to univariate GWAS
        # initial guess: 50% heritability
        var_y = torch.var(Y, dim=0)
        with torch.no_grad():
            self.l_G.copy_(torch.tensor([torch.sqrt(var_y[0] * 0.5), 0.0, torch.sqrt(var_y[1] * 0.5)]))
            self.l_R.copy_(torch.tensor([torch.sqrt(var_y[0] * 0.5), 0.0, torch.sqrt(var_y[1] * 0.5)]))

    @staticmethod
    def vec_to_sym_matrix(l: torch.Tensor):
        """
        compute C = LL^T for lower triangular L

        :param l: vector containing elements of L
        :return:
        """
        L = torch.zeros((2,2), device=l.device)
        L[0,0], L[1,0], L[1,1] = l[0], l[1], l[2]
        return L @ L.t()

    def reml_loss(self):
        G = self.vec_to_sym_matrix(self.l_G)
        R = self.vec_to_sym_matrix(self.l_R)
        # TODO

    def fit_null_model(self):
        # TODO
        return None

    def get_metrics(self):
        # TODO
        return None

    def gwas_scan(self):
        # TODO
        return None

    def reset(self, Y:torch.Tensor=None, A_cov: torch.Tensor=None):
        """
        Resets parameters and allows changing the phenotypes, covariates and design matrix

        :return:
        """
        # TODO use this to change trait design matrix A_cov or to change phenotypes or if optimization stuck
        print("Resetting parameters...")
        # 1. Update design matrix A_cov and covariates Z if provided
        if A_cov is not None:
            print("Reconstructing covariate batch...")
            self.X_batch = self._transform_covariates(Z=self.Z_raw, A_cov=A_cov)

        # 2. Update phenotypes if provided
        if Y is not None:
            print("Updating phenotypes...")
            self.y = self._transform_phenotype(Y=Y)

        # 3. Clear gradients
        for param in [self.l_G, self.l_R]:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        # 4. Re-initialize variance components (warm start)
        with torch.no_grad():
            # Estimate variance from current y_star
            self._initialize_params(Y=self.y.squeeze(-1))

        self.beta_null = None
        self.V_inv_null = None
        print("Reset complete.")