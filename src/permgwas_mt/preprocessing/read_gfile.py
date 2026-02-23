import numpy as np
import pandas as pd
import torch
import h5py

# TODO PLINKReader
# TODO sanity checks and encoding

class H5Reader:
    def __init__(self, genotype_file):
        self.genotype_file = genotype_file

    def get_ids(self):
        with h5py.File(self.genotype_file, 'r') as f:
            return f['sample_ids'][:].astype(str)

    def stream_batches(self, sample_indices, batch_size):
        """Pass 1: Stream EVERYTHING (Genotypes + Metadata) for setup."""
        with h5py.File(self.genotype_file, 'r') as f:
            geno_ds = f['snps']
            chr_ds, pos_ds = f['chr_index'], f['position_index']
            n_snps = geno_ds.shape[1]

            for i in range(0, n_snps, batch_size):
                end = min(i + batch_size, n_snps)
                yield {
                    'genotypes': geno_ds[:, i:end][sample_indices, :],
                    'chrs': chr_ds[i:end].astype(str),
                    'pos': pos_ds[i:end].astype(int),
                    'start_idx': i
                }

    def stream_valid_batches(self, sample_indices, valid_snp_indices, batch_size):
        """Pass 2: Stream only SNPs that passed filters (Selective I/O)."""
        with h5py.File(self.genotype_file, 'r') as f:
            ds = f['snps']
            for i in range(0, len(valid_snp_indices), batch_size):
                targets = np.sort(valid_snp_indices[i: i + batch_size])
                yield torch.from_numpy(ds[:, targets][sample_indices, :]).float()


class CSVReader:
    def __init__(self, genotype_file):
        # Load once into RAM; parse CHR_POS from headers
        df = pd.read_csv(genotype_file, index_col=0)
        self.sample_ids = df.index.values.astype(str)
        self.full_matrix = df.values.astype(np.float32)
        rsids = df.columns.values.astype(str)

        # Split "CHR_POS" into metadata arrays
        meta = [s.split('_') for s in rsids]
        self.chrs = np.array([m[0] for m in meta])
        self.pos = np.array([int(m[1]) for m in meta])

    def get_ids(self):
        return self.sample_ids

    def stream_batches(self, sample_indices, batch_size):
        """Pass 1: Mimic stream by slicing RAM matrix."""
        aligned = self.full_matrix[sample_indices, :]
        for i in range(0, aligned.shape[1], batch_size):
            end = i + batch_size
            yield {
                'genotypes': aligned[:, i:end],
                'chrs': self.chrs[i:end],
                'pos': self.pos[i:end],
                'start_idx': i
            }

    def stream_valid_batches(self, sample_indices, valid_snp_indices, batch_size):
        """Pass 2: High-speed RAM slice for valid SNPs."""
        valid_data = self.full_matrix[sample_indices, :][:, valid_snp_indices]
        for i in range(0, valid_data.shape[1], batch_size):
            yield torch.from_numpy(valid_data[:, i: i + batch_size]).float()
