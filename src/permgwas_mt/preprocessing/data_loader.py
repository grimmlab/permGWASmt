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
        self.y, self.fixed, self.sample_ids, self.geno_index = self._match_samples(geno_ids=self.reader.get_ids(),
                                                                            phenotype_file=input_config.phenotype_file,
                                                                            traits=input_config.traits,
                                                                            kinship_file=input_config.kinship_file,
                                                                            kinship_type=input_config.kinship_type,
                                                                            covariate_file=input_config.covariate_file,
                                                                            covariate_list=input_config.covariate_list)
        self.n_samples = len(self.sample_ids)
        self.trait_corr = torch.corrcoef(self.y.t())[0, 1].item()

        # filter SNPs and get kinship
        self.valid_snp_indices, K, self.metadata = self._perform_unified_setup(maf_threshold=input_config.maf_threshold,
                                                                               kinship_file=input_config.kinship_file,
                                                                               kinship_type=input_config.kinship_type)
        self.n_snps = len(self.valid_snp_indices)


        # Compute spectral decomposition
        self.evals, self.Ut = self._spectral_decomposition(K)
        del K
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()


    def _match_samples(self, geno_ids, phenotype_file, traits, kinship_file=None, kinship_type=None,
                       covariate_file=None, covariate_list=None):
        geno_ids = np.array([str(x).strip() for x in geno_ids])
        # load phenotypes (and covariates and kinship if available)
        p_df = self._load_dataframe(filename=phenotype_file, target_cols=traits)
        c_df = None
        if covariate_file is not None:
            c_df = self._load_dataframe(filename=covariate_file, target_cols=covariate_list)
        k_ids = None
        if kinship_file is not None:
            if kinship_type == KinshipFileType.HDF5:
                with h5py.File(kinship_file, 'r') as f:
                    k_ids = f['sample_ids'][:].astype(str)
            elif kinship_type == KinshipFileType.CSV:
                k_ids = pd.read_csv(kinship_file, sep=None, engine='python', usecols=[0], dtype=str)
                k_ids = k_ids.iloc[:,0].values.astype(str)
            else:
                raise ValueError("Unknown kinship type")
            k_ids = [s.strip() for s in k_ids]
        # get common IDs ordered same as genotype
        common_ids = set(geno_ids) & set(p_df.index)
        if c_df is not None: common_ids &= set(c_df.index)
        if k_ids is not None: common_ids &= set(k_ids)
        sample_ids = [x for x in geno_ids if x in common_ids]

        # get phenotype
        y = torch.from_numpy(p_df.loc[sample_ids].values).to(self.dtype)
        # get covariates
        fixed = np.ones((len(sample_ids), 1))
        if c_df is not None:
            fixed = torch.from_numpy(np.hstack([fixed, c_df.loc[sample_ids].values])).to(self.dtype)
        else:
            fixed = torch.from_numpy(fixed).to(self.dtype)

        geno_index = np.where(np.isin(geno_ids, sample_ids))[0]
        return y, fixed, np.array(sample_ids), geno_index

    @staticmethod
    def _load_dataframe(filename, target_cols=None):

        # Load with flexible separator (sniffs , \t or spaces)
        df = pd.read_csv(filename, sep=None, engine='python', index_col=0)
        df.index = df.index.astype(str).str.strip()
        # Get columns
        if target_cols is not None:
            # Ensure we keep ID even if not in the target list
            df = df[[c for c in target_cols if c in df.columns]]
        # Handle Replicates: Compute mean for duplicate IDs
        if df.index.duplicated().any():
            df = df.groupby(level=0).mean()
        # Drop rows with missing entries
        df = df.dropna()
        return df

    def _perform_unified_setup(self, maf_threshold, kinship_file=None, kinship_type=None):
        """Get SNP filters and meta data and kinship matrix"""
        dtype = torch.float32 if self.device.type == "mps" else torch.float64
        if kinship_file is not None:
            K = self._load_kinship_matrix(kinship_file, kinship_type)
        else:
            K = torch.zeros((self.n_samples, self.n_samples), device=self.device, dtype=dtype)

        valid_idx, chrs, poss, mafs, total_snps = [], [], [], [], 0

        for batch in self.reader.stream_batches(self.geno_index, self.batch_size):
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
                snps = snps.to(dtype=dtype)
                K = torch.addmm(K, snps, snps.t())
                total_snps += snps.shape[1]

                del snps
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    torch.mps.empty_cache()

        if kinship_file is None:
            K /= total_snps
        return np.array(valid_idx), K, {'chr': chrs, 'pos': poss, 'maf': np.array(mafs)}


    def _load_kinship_matrix(self, kinship_file, kinship_type):
        """Load kinship from file"""
        if kinship_type == KinshipFileType.HDF5:
            with h5py.File(kinship_file, 'r') as f:
                k_ids = f['sample_ids'][:].astype(str)
                id_map = {id_: i for i, id_ in enumerate(k_ids)}
                # Get original indices
                unsorted_idx = [id_map[id_] for id_ in self.sample_ids]
                # Create a sorting map to get back to original order later
                sorted_idx = np.sort(unsorted_idx)
                rank = np.argsort(np.argsort(unsorted_idx))
                # Slice using the sorted indices
                K = f['kinship'][sorted_idx, :][:, sorted_idx]
                # Re-arrange the matrix back to the 'sample_ids' order
                K = K[rank, :][:, rank]
        elif kinship_type == KinshipFileType.CSV:
            df = pd.read_csv(kinship_file, sep=None, engine='python', index_col=0)
            df.index = df.index.astype(str).str.strip()
            df.columns = df.index
            K = df.loc[self.sample_ids, self.sample_ids].apply(pd.to_numeric, errors='coerce').values
        else:
            raise ValueError("Unknown kinship type")
        dtype = torch.float32 if self.device == "mps" else torch.float64
        return torch.from_numpy(K).to(dtype=dtype, device=self.device)

    def _spectral_decomposition(self, K):
        """
            Applies a diagonal nudge and performs spectral decomposition.
            """
        # Gower-style Centering
        device = torch.device("cpu") if self.device.type == "mps" else self.device
        K = K.to(device=device)
        K = K.to(dtype=torch.float64)
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

    def get_genotype_stream(self):
        """Pass 2: The actual GWAS scan."""
        return self.reader.stream_valid_batches(self.geno_index, self.valid_snp_indices, self.batch_size)

    @staticmethod
    def _get_reader(genotype_file, genotype_type):
        if genotype_type == GenotypeFileType.HDF5:
            return H5Reader(genotype_file)
        elif genotype_type == GenotypeFileType.CSV:
            return CSVReader(genotype_file)
        else:
            raise ValueError('Unknown genotype type')
