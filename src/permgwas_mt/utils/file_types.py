from enum import Enum
from pathlib import Path

class GenotypeFileType(Enum):
    HDF5 = {".h5", ".hdf5", ".h5py"}
    CSV = {".csv"}
    PLINK_TEXT = {".ped", ".map"}
    PLINK_BINARY = {".bed", ".bim", ".fam"}

    @classmethod
    def from_path(cls, path: Path):
        suffix = path.suffix.lower()
        for filetype in cls:
            if suffix in filetype.value:
                return filetype
        raise ValueError(f"Unsupported genotype file type: {path.suffix}")


class PhenotypeFileType(Enum):
    CSV = {".csv"}
    TXT = {".txt"}
    PHENO = {".pheno"}

    @classmethod
    def from_path(cls, path: Path):
        suffix = path.suffix.lower()
        for filetype in cls:
            if suffix in filetype.value:
                return filetype
        raise ValueError(f"Unsupported phenotype file type: {path.suffix}")


class CovariateFileType(Enum):
    CSV = {".csv"}

    @classmethod
    def from_path(cls, path: Path):
        suffix = path.suffix.lower()
        for filetype in cls:
            if suffix in filetype.value:
                return filetype
        raise ValueError(f"Unsupported covariate file type: {path.suffix}")


class KinshipFileType(Enum):
    HDF5 = {".h5", ".hdf5", ".h5py"}
    CSV = {".csv"}

    @classmethod
    def from_path(cls, path: Path):
        suffix = path.suffix.lower()
        for filetype in cls:
            if suffix in filetype.value:
                return filetype
        raise ValueError(f"Unsupported genotype file type: {path.suffix}")