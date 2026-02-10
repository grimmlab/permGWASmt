import warnings
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from .result_types import ResultType
from .file_types import (GenotypeFileType, PhenotypeFileType, CovariateFileType, KinshipFileType)


@dataclass(frozen=True)
class InputConfig:
    genotype_file: Path
    phenotype_file: Path
    traits: list

    kinship_file: Path | None = None
    covariate_file: Path | None = None
    covariate_list: list | None = None
    outdir: Path | None = None
    outfile: str | None = None

    maf_threshold: int = 0
    n_permutations: int = 0
    perm_method: str = "x"

    load_genotype: bool = False
    batch_size: int = 10000
    perm_batch_size: int = 1000
    device: str = "cpu"

    genotype_type: GenotypeFileType = field(init=False)
    phenotype_type: PhenotypeFileType = field(init=False)
    covariate_type: CovariateFileType | None = field(init=False)
    kinship_type: KinshipFileType | None = field(init=False)

    @classmethod
    def from_sources(cls, cli_args: dict, config_path: str=None):
        """
        Merge CLI args and optional YAML config, return resolved InputConfig
        ________________
        :param cli_args: CLI arguments
        :param config_path: path to YAML config file
        :return: resolved InputConfig
        """
        import yaml

        config = {}
        if config_path is not None:
            config_file = Path(config_path)
            if not config_file.is_file():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with config_file.open("r") as f:
                config = yaml.safe_load(f) or {}

        # CLI args override config file
        merged = config.copy()
        for key, value in cli_args.items():
            if value is not None:
                merged[key] = value

        return cls(**merged)

    def __post_init__(self):
        self._normalize_paths()
        self._infer_file_types()
        self._validate_plink_files()
        self._validate_inputs()
        self._validate_phenotypes()
        self._validate_covariates()
        self._prepare_output()
        self._validate_device()

    #------------------------ INTERNAL METHODS ------------------------

    def _normalize_paths(self):
        object.__setattr__(self, "genotype_file", Path(self.genotype_file))
        object.__setattr__(self, "phenotype_file", Path(self.phenotype_file))
        if self.covariate_file is not None:
            object.__setattr__(self, "covariate_file", Path(self.covariate_file))
        if self.kinship_file is not None:
            object.__setattr__(self, "kinship_file", Path(self.kinship_file))
        if self.outdir is not None:
            object.__setattr__(self, "outdir", Path(self.outdir))

    def _infer_file_types(self):
        object.__setattr__(
            self,
            "genotype_type",
            GenotypeFileType.from_path(self.genotype_file),
        )
        object.__setattr__(
            self,
            "phenotype_type",
            PhenotypeFileType.from_path(self.phenotype_file),
        )
        if self.covariate_file is not None:
            object.__setattr__(
                self,
                "covariate_type",
                CovariateFileType.from_path(self.covariate_file),
            )
        else:
            object.__setattr__(self, "covariate_type", None)
        if self.kinship_file is not None:
            object.__setattr__(
                self,
                "kinship_type",
                KinshipFileType.from_path(self.kinship_file),
            )
        else:
            object.__setattr__(self, "kinship_type", None)

    def _validate_plink_files(self):
        if self.genotype_type not in {
            GenotypeFileType.PLINK_TEXT,
            GenotypeFileType.PLINK_BINARY,
        }:
            return

        prefix = self.genotype_file.with_suffix("")

        if self.genotype_type == GenotypeFileType.PLINK_TEXT:
            required = {".ped", ".map"}
        else:
            required = {".bed", ".bim", ".fam"}

        found = {
            p.suffix.lower()
            for p in prefix.parent.glob(prefix.name + ".*")
        }

        missing = required - found
        if missing:
            raise FileNotFoundError(
                f"Incomplete {self.genotype_type.name.replace('_', ' ').lower()} fileset. "
                f"Missing: {', '.join(sorted(missing))} "
                f"for prefix '{prefix}'."
            )
        object.__setattr__(self, "genotype_file", prefix)

    def _validate_inputs(self):
        # check input files
        for path in ([self.genotype_file, self.phenotype_file] +
                     ([self.kinship_file] if self.kinship_file is not None else [])):
            if not path.is_file():
                raise FileNotFoundError(f"Input file not found: {path}")

        # check batch-wise loading
        if not self.load_genotype:
            if (self.genotype_type != GenotypeFileType.HDF5) or (self.kinship_file is None):
                object.__setattr__(self, "load_genotype", True)
                warnings.warn(
                    "Full genotype matrix will be loaded during preprocessing. Batch-wise loading is only "
                    "supported for genotype file type HDF5 and if precomputed kinship matrix is provided.",
                    UserWarning,
                )

        # check permutations
        if self.n_permutations > 0:
            if self.perm_method not in ("x", "y"):
                raise NotImplementedError(f"Permutation method {self.perm_method} not supported")

    def _validate_phenotypes(self):
        if len(self.traits) != 2:
            raise ValueError("Exactly two phenotypes must be specified!")
        if self.traits[0] == self.traits[1]:
            raise ValueError("Traits must be different!")
        df = pd.read_csv(self.phenotype_file, sep=None, engine="python", nrows=0)
        id_col = df.columns[0]
        expected_id_names = {"id", "iid", "fid", "sample_id", "accession_id"}
        if id_col.lower() not in expected_id_names:
            warnings.warn(
                f"First column '{id_col}' in phenotype file "
                f"is assumed to contain sample IDs. "
                "If this is incorrect, results may be invalid.",
                UserWarning,
            )
        for pheno in self.traits:
            if pheno not in df.columns[1:]:
                raise ValueError(f"Phenotype '{pheno}' not found in phenotype file.")

    def _validate_covariates(self):
        if self.covariate_file is None:
            if self.covariate_list is not None:
                raise ValueError("Covariates were specified but no covariate file was provided.")
            return

        if not self.covariate_file.is_file():
            raise FileNotFoundError("Covariate file not found")

        df = pd.read_csv(self.covariate_file, sep=None, engine="python", nrows=0)
        id_col = df.columns[0]
        expected_id_names = {"id", "iid", "sample_id", "accession_id"}

        if id_col.lower() not in expected_id_names:
            warnings.warn(
                f"First column '{id_col}' in covariate file "
                f"is assumed to contain sample IDs.",
                UserWarning,
            )

        data_cols = df.columns[1:]
        if self.covariate_list is None:
            object.__setattr__(self, "covariate_list", tuple(data_cols))
            return

        else:
            missing = set(self.covariate_list) - set(data_cols)
            if missing:
                raise ValueError("The following covariate columns were not found in covariate file: "
                                 + ", ".join(sorted(missing)))
            object.__setattr__(self, "covariate_list", tuple(self.covariate_list))

    def _prepare_output(self):
        # prepare output directory
        outdir = self.outdir or Path.cwd() / "results"
        outdir.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "outdir", outdir)

        # prepare output filename
        tmp_file = self.outfile if self.outfile is not None else "_".join(self.traits)
        out_path = self.outdir / f"p_values_{tmp_file}.csv"

        if not out_path.exists():
            object.__setattr__(self, "outfile", tmp_file)
        else:
            i = 1
            while True:
                candidate = self.outdir / f"p_values_{tmp_file}({i}).csv"
                if not candidate.exists():
                    object.__setattr__(self, "outfile", f"{tmp_file}({i})")
                    break
                i += 1

    def _validate_device(self):
        import torch

        device = self.device.lower()

        if device == "cpu":
            object.__setattr__(self, "device", "cpu")
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError(f"CUDA device requested ({device}) but CUDA is not available.")
            # check if the device index is valid
            try:
                idx = int(device.split(":")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid CUDA device format: {device}. Use 'cuda:0', 'cuda:1', etc.")
            if idx < 0 or idx >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested CUDA device index {idx} is out of range. "
                    f"Available devices: 0-{torch.cuda.device_count() - 1}"
                )
            # normalized device string
            object.__setattr__(self, "device", f"cuda:{idx}")
        else:
            raise ValueError(f"Unknown device option: {device}. Must be 'cpu' or 'cuda:N'")

    # ----------------------------- PUBLIC METHODS ------------------------------

    def resolve_output_file(self, result_type:ResultType) -> Path:
        """
        Return output file path for given ResultType
        :param result_type: ResultType of output file
        :return: output file path
        """
        base = f"{result_type.stem}_{self.outfile}"
        out_path = self.outdir / f"{base}{result_type.suffix}"
        return out_path
