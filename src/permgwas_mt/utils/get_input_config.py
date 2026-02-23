import warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
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

    no_scan: bool = False
    hypothesis_type: str = "any"
    trait_design: str = "identity"
    maf_threshold: int = 0
    n_permutations: int = 0

    batch_size: int = 10000
    perm_batch_size: int = 100
    device: str = "cpu"
    dtype: str = "float32"
    master_seed: int | None = None
    time_report: bool = False

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

        if self.no_scan:
            object.__setattr__(self, "hypothesis_type", None)
        else:
            if self.hypothesis_type is None:
                object.__setattr__(self, "hypothesis_type", "any")
            elif self.hypothesis_type not in ("any", "common", "specific", "all"):
                raise ValueError(f"Unknown hypothesis test type: {self.hypothesis_type}")

        if self.trait_design != "identity":
            raise ValueError(f"Unknown trait design matrix: {self.trait_design}")

        if self.master_seed is None:
            object.__setattr__(self, "master_seed", np.random.randint(0, 2**31 - 1))

        if self.batch_size is None:
            object.__setattr__(self, "batch_size", 10000)

        if self.perm_batch_size is None:
            object.__setattr__(self, "perm_batch_size", 1000)

        if self.n_permutations is None:
            object.__setattr__(self, "n_permutations", 0)

        # TODO sanity checks for device and dtype -- mps only 32
        if self.device is None:
            object.__setattr__(self, "device", "cpu")

        if self.dtype is None:
            object.__setattr__(self, "dtype", "float32")

        if self.maf_threshold is None:
            object.__setattr__(self, "maf_threshold", 0)


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

        data_cols = df.columns[1:].tolist()
        if self.covariate_list is None:
            object.__setattr__(self, "covariate_list", data_cols)
        else:
            missing = set(self.covariate_list) - set(data_cols)
            if missing:
                raise ValueError("The following covariate columns were not found in covariate file: "
                                 + ", ".join(sorted(missing)))

    def _prepare_output(self):
        # prepare output directory
        outdir = self.outdir or Path.cwd() / "results"
        outdir.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "outdir", outdir)

        # prepare output filename
        tmp_file = self.outfile if self.outfile is not None else "_".join(self.traits)
        out_path = self.outdir / f"summary_stats_{tmp_file}.yaml"

        if not out_path.exists():
            object.__setattr__(self, "outfile", tmp_file)
        else:
            i = 1
            while True:
                candidate = self.outdir / f"summary_stats_{tmp_file}({i}).yaml"
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
        elif device == "mps":
            if torch.backends.mps.is_available():
                object.__setattr__(self, "device", "mps")
            else:
                print("WARNING: MPS not available, using CPU instead.")
                object.__setattr__(self, "device", "cpu")
        else:
            raise ValueError(f"Unknown device option: {device}. Must be 'cpu' or 'cuda:N'")

    # ----------------------------- PUBLIC METHODS ------------------------------

    def resolve_output_file(self, result_type: ResultType, hypothesis_type: str=None) -> Path:
        """
        Return output file path for given ResultType
        :param hypothesis_type:
        :param result_type: ResultType of output file
        :return: output file path
        """
        if result_type.stem == "summary_stats":
            base = f"{result_type.stem}_{self.outfile}"
        else:
            base = f"{result_type.stem}_{hypothesis_type}_{self.outfile}"
        out_path = self.outdir / f"{base}{result_type.suffix}"
        return out_path
