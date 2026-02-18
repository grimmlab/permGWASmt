import numpy as np
import pandas as pd
import torch
import h5py

from permgwas_mt.utils.get_input_config import InputConfig
from permgwas_mt.utils.file_types import (GenotypeFileType, KinshipFileType)
from permgwas_mt.preprocessing.read_gfile import H5Reader, CSVReader


class Dataset:
    def __init__(self, input_config: InputConfig, dtype=torch.float32):
        self.device = torch.device(input_config.device)
        self.dtype = dtype
        self.batch_size = input_config.batch_size
        # get genotype reader
        self.reader = self._get_reader(genotype_file=input_config.genotype_file,
                                       genotype_type=input_config.genotype_type)

        # intersect sample IDs
        self.geno_indices, self.y, covars, self.common_ids = self._match_samples(geno_ids=self.reader.get_ids(),
                                                                            phenotype_file=input_config.phenotype_file,
                                                                            traits=input_config.traits,
                                                                            kinship_file=input_config.kinship_file,
                                                                            kinship_type=input_config.kinship_type,
                                                                            covariate_file=input_config.covariate_file,
                                                                            covariate_list=input_config.covariate_list)
        self.n_samples = len(self.common_ids)
        self.fixed = self._prepare_covariates(covars)
        self.trait_corr = torch.corrcoef(self.y.t())[0, 1].item()

        # filter SNPs and get kinship
        self.valid_snp_indices, K, self.metadata = self._perform_unified_setup(maf_threshold=input_config.maf_threshold,
                                                                               kinship_file=input_config.kinship_file,
                                                                               kinship_type=input_config.kinship_type)
        self.n_snps = len(self.valid_snp_indices)


        # Compute spectral decomposition
        self.evals, self.U = self._spectral_decomposition(K)
        del K
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()



    def _match_samples(self, geno_ids, phenotype_file, traits, kinship_file=None, kinship_type=None,
                       covariate_file=None, covariate_list=None):
        geno_ids = np.array([str(x).strip() for x in geno_ids])
        # load phenotypes
        p_df = self._load_dataframe(filename=phenotype_file, target_cols=traits)
        common = np.intersect1d(geno_ids, p_df['ID'].values)

        # load covariates
        c_df = None
        if covariate_file is not None:
            c_df = self._load_dataframe(filename=covariate_file, target_cols=covariate_list)
            common = np.intersect1d(common, c_df['ID'].values)

        # load Kinship IDs
        if kinship_file is not None:
            if kinship_type == KinshipFileType.HDF5:
                with h5py.File(kinship_file, 'r') as f:
                    k_ids = f['sample_ids'][:].astype(str)
            elif kinship_type == KinshipFileType.CSV:
                k_ids = pd.read_csv(kinship_file, sep=None, engine='python', usecols=[0]).iloc[:, 0].values.astype(str)
            else:
                raise ValueError("Unknown kinship type")
            common = np.intersect1d(common, k_ids)

        # Final ordering based on genotype file sequence
        common = np.array([idx for idx in geno_ids if idx in set(common)])
        p_df = p_df.set_index('ID').loc[common]
        c_df = c_df.set_index('ID').loc[common] if c_df is not None else None

        geno_indices = np.where(np.isin(geno_ids, common))[0]
        y = torch.from_numpy(p_df.values).to(dtype=self.dtype)
        covars = torch.from_numpy(c_df.values).to(dtype=self.dtype) if c_df is not None else None
        return geno_indices, y, covars, np.array(common)


    @staticmethod
    def _load_dataframe(filename, target_cols=None):

        # Load with flexible separator (sniffs , \t or spaces)
        df = pd.read_csv(filename, sep=None, engine='python')
        # Identify the ID column (assume it is the first one)
        df = df.rename(columns={df.columns[0]: 'ID'})
        df['ID'] = df['ID'].astype(str).str.strip()
        # Get columns
        if target_cols:
            # Ensure we keep ID even if not in the target list
            cols = ['ID'] + [c for c in target_cols if c in df.columns]
            df = df[cols]
        # Handle Replicates: Compute mean for duplicate IDs
        if df['ID'].duplicated().any():
            df = df.groupby('ID').mean().reset_index()
        # Drop rows with missing entries
        df = df.dropna(subset=target_cols)
        return df

    def _perform_unified_setup(self, maf_threshold, kinship_file=None, kinship_type=None):
        """Get SNP filters and meta data and kinship matrix"""
        if kinship_file is not None:
            K = self._load_kinship_matrix(kinship_file, kinship_type)
        else:
            K = torch.zeros((self.n_samples, self.n_samples), device=self.device)

        valid_idx, chrs, poss, mafs, total_snps = [], [], [], [], 0

        for batch in self.reader.stream_batches(self.geno_indices, self.batch_size):
            snps = batch.pop('genotypes')
            m = np.mean(snps, axis=0) / 2
            m = np.minimum(m, 1 - m)
            mask = (m >= (maf_threshold / 100)) & (np.var(snps, axis=0) > 1e-6)


            if not np.any(mask): continue

            # Registry
            valid_idx.extend(np.where(mask)[0] + batch['start_idx'])
            chrs.extend(batch['chrs'][mask])
            poss.extend(batch['pos'][mask])
            mafs.extend(m[mask])

            # Accumulate GRM only if not provided
            if kinship_file is None:
                snps = torch.from_numpy(snps[:, mask]).to(device=self.device, dtype=self.dtype)
                snps -= snps.mean(0)
                snps /= snps.std(0).clamp(min=1e-6)
                snps = snps.to(self.dtype)
                K = torch.addmm(K, snps, snps.t())
                total_snps += snps.shape[1]

                del snps
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        if kinship_file is not None:
            K /= total_snps
        return np.array(valid_idx), K, {'chr': chrs, 'pos': poss, 'maf': np.array(mafs)}


    def _load_kinship_matrix(self, kinship_file, kinship_type):
        """Load kinship from file"""
        if kinship_type == KinshipFileType.HDF5:
            with h5py.File(kinship_file, 'r') as f:
                k_ids = f['sample_ids'][:].astype(str)
                id_map = {id_: i for i, id_ in enumerate(k_ids)}
                idx = [id_map[id_] for id_ in self.common_ids]
                K = f['kinship'][idx, :][:, idx]
        elif kinship_type == KinshipFileType.CSV:
            df = pd.read_csv(kinship_file, sep=None, engine='python', header=None)
            # check for header
            if df.shape[0] == df.shape[1]:
                df.columns = df.iloc[0]
                df = df.drop(df.index[0])
            df.iloc[:, 0] = df.iloc[:, 0].astype(str)
            df = df.rename(columns={df.columns[0]: 'ID'}).set_index('ID')
            K = df.loc[self.common_ids, self.common_ids].values.astype(np.float64)
        else:
            raise ValueError("Unknown kinship type")
        return torch.from_numpy(K).to(device=self.device)


    def _spectral_decomposition(self, K):
        """
            Applies a diagonal nudge and performs spectral decomposition.
            """
        # Gower-style Centering
        K = K.to(dtype=torch.float64, device=self.device)
        row_means = K.mean(dim=0, keepdim=True)
        col_means = K.mean(dim=1, keepdim=True)
        grand_mean = K.mean()
        K = K - row_means - col_means + grand_mean
        # Scaling
        scaling_factor = torch.diag(K).mean()
        if scaling_factor > 1e-8:
            K = K / scaling_factor
        # Forces absolute symmetry and adds epsilon to the diagonal for stability
        K = (K + K.t()) / 2
        nudge = torch.eye(self.n_samples, device=K.device) * 1e-6
        # Perform Decomposition
        evals, U = torch.linalg.eigh(K + nudge)
        # Eigenvalue Clamp to force them to be positive
        evals = torch.clamp(evals, min=1e-7)
        return evals.to(self.dtype), U.t().to(dtype=self.dtype)

    def _prepare_covariates(self, covars):
        fixed = torch.ones((self.n_samples, 1))
        if covars is not None:
            fixed = torch.cat([fixed, covars], dim=1)
        return fixed

    def get_genotype_stream(self):
        """Pass 2: The actual GWAS scan."""
        return self.reader.stream_valid_batches(self.geno_indices, self.valid_snp_indices, self.batch_size)

    @staticmethod
    def _get_reader(genotype_file, genotype_type):
        if genotype_type == GenotypeFileType.HDF5:
            return H5Reader(genotype_file)
        elif genotype_type == GenotypeFileType.CSV:
            return CSVReader(genotype_file)
        else:
            raise ValueError('Unknown genotype type')