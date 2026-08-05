# permGWASmt
Multi-trait GWAS solver with permutation-based thresholds. 

## Quick Start
### Build the Docker Image
Build the Docker image using the Dockerfile provided in the `docker` directory:
```shell
docker build -t IMAGENAME .
```

### Start the Interactive Docker Container
To run the analysis, mount your local project directory to the container. 
```shell
docker run -it -v $(pwd):/app/permGWASmt --gpus device=0 --name CONTAINERNAME IMAGENAME
```

### Run the Pipeline
Once inside the container terminal, navigate to the project root. The `PYTHONPATH` is 
preconfigured to find the `permgwas_mt` module inside the `src` folder.
```shell
cd /app/permGWASmt
python3 -m permgwas_mt.run --config data/config.yaml
```

### Troubleshooting
If you receive a `ModuleNotFoundError`, ensure your `PYTHONPATH` includes the source directory:
```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

## Minimal Requirements
### Config File
permGWASmt accepts `YAML` config files that contain all flags and options:
```YAML
---
genotype_file: "./data/x_matrix.h5"
phenotype_file: "./data/y_matrix.csv"
traits:
  - "trait1"
  - "trait2"
```

For permGWASmt to run, you need to provide the paths to your **genotype** and **phenotype files** 
and the names of **two traits** contained in the phenotype file. 

## Data Guide
### Genotype File
#### HDF5 / H5 / H5PY
The file has to contain the following keys:
- snps: genotype matrix, additively encoded (012)
- sample_ids: vector containing corresponding sample ids
- position_index: vector containing the positions of all SNPs
- chr_index: vector containing the corresponding chromosome number

#### CSV
The **first column** should be the **sample ids**. The **column names** should be the **SNP identifiers** in the form 
"CHR_POSITION" (e.g. Chr1_657). The values should be the genotype matrix in **additive encoding**. 

#### (binary) PLINK
**NOT YET IMPLEMENTED—WILL COME SOON**

### Phenotype File (CSV, TXT, PHENO)
Here the **first column** should contain the **sample ids**. The remaining columns should 
contain the phenotype values with the phenotype name as column name. Both traits need to be in the same file. 
Samples where at least one trait value is missing will be removed.

### Kinship File
permGWASmt computes the realized relationship kernel as kinship matrix. If you want to use a different kinship matrix,
you can provide a kinship file containing a symmetric kinship matrix:

**HDF5 / H5 / H5PY:** The file needs to contain the following keys:
- kinship: kinship matrix values
- sample_ids: vector containing corresponding sample ids

**CSV:** The first column should contain the sample ids.

### Covariates File (CSV)
It is possible to run permGWASmt with additional covariates. 
If no covariates file is provided, only the intercept will be used as fixed effect. 
Here the first column should contain the sample ids and the header should contain the names of the covariates.

## Further Flags and Options
See help for more information.
### Data
| **Flag**       | **Description**                                                                        |
|----------------|----------------------------------------------------------------------------------------|
| genotype_file  | Path to genotype                                                                       |
| phenotype_file | Path to phenotype                                                                      |
| covariate_file | Path to covariates                                                                     |
| kinship_file   | Path to kisnhip                                                                        |
| traits         | List with two trait names                                                              |
| covariate_list | List of covariates to use from covariate file. If not provided will use all covariates |

### General Options
| **Flag**        | **Description**                                                                          |
|-----------------|------------------------------------------------------------------------------------------|
| maf_threshold   | Minor allele frequency threshold (as int value) to filter SNPs                           |
| n_permutations  | Number of permutations for permutation-based threshold                                   |
| outdir          | Path to results folder (default: ./results)                                              |
| outfile         | Postfix for result files (default: TRAITNAME1_TRAITNAME2)                                |
| hypothesis_type | Hypothesis type.Valid options are 'any', 'common', 'specific' and 'all' (default: 'any') |
| trait_design    | Trait design matrix. Currently only support 'identity'                                   |

### Compute Settings
| **Flag**        | **Description**                                                                 |
|-----------------|---------------------------------------------------------------------------------|
| device          | Device to run on (cpu or cuda:N), (default: cpu)                                |
| dtype           | Optional dtype to use for computations (float32 or float64), (default: float32) |
| no_scan         | Only determine and save variance components without running a full GWAS scan    |
| batch_size      | Number of SNPs to work on simultaneously (default: 10000)                       |
| perm_batch_size | Number of permutations to work on simultaneously (default: 100)                 |
| master_seed     | Master seed used for permutations (default: will be randomly generated)         |
| time_report     | Print time report at the end of analysis                                        |