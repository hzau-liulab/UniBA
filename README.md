# Description

**UniBA** is a unified multimodal framework for protein complex binding affinity prediction. It integrates paired sequence representations with multi-scale structural modeling, including residue-level interface graphs and coarse-grained molecular graphs, through a dynamic fusion strategy to predict binding affinity across diverse protein interaction systems.
![image](img/UniBA.png)  
  
# Usage
## 1. Clone the repository
Clone the UniBA repository and enter the project directory:

```bash
git clone https://github.com/hzau-liulab/UniBA.git
cd UniBA
```

## 2. Set up the environment
We recommend creating a dedicated conda environment for running UniBA and installing the required dependencies within this environment.
Create and activate a new conda environment:

```bash
conda create -n uniba python=3.9
conda activate uniba
```
The main dependencies used in the reported experiments are listed below.

| Package | Version | Package | Version | Package | Version | Package | Version |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Python | 3.9 | PyTorch | 1.12.0 | CUDA | 11.3 | PyTorch Geometric | 2.3.1 |
| fair-esm | 2.0.1 | Biopython | 1.85 | ANARCI | 1.3 | torch-scatter | 2.1.0 | 
| scikit-learn | 1.3.2 | MDAnalysis | 2.7.0 | NetworkX | 3.1 | GraphRicciCurvature | 0.5.3.1 |
| NumPy | 1.26.4 | SciPy | 1.12.0 | pdbfixer | 1.9 | Pandas | 2.2.3 |

## 3. Install third-party software
UniBA requires several external tools for structural feature generation and coarse-grained modeling. Please install these tools separately and configure their paths before running the feature generation pipeline.

| Software | Purpose | Version |
| --- | --- | --- |
| DSSP | Secondary structure and solvent accessibility calculation | 3.0.0 |
| PLIP | Protein interaction feature extraction | 2.4.0 |
| Open Babel | Molecular structure processing | 3.1.1 |
| GHECOM | Pocket descriptor generation | 2021-12-01 |
| Gromacs | Martini-based coarse-grained modeling and energy minimization | 2022.4 |
| Martini 2.2 | Coarse-grained structure generation | 2.2 |

## 4. Download datasets and pre-trained models 
UniBA requires protein complex structures, affinity labels, and pretrained model checkpoints for feature generation and prediction. The required release assets are provided below.
### UniBA datasets and checkpoints
| File | Contents | Size |
| --- | --- | --- |
| [`uniba_raw_pdb.tar.zst`](https://drive.google.com/file/d/1z1xSP5U5GkCvLTmrMAnlxp8qUMspBr9y/view?usp=sharing) | 3,327 chain-selected protein complex PDBs | ~215 MB |
| [`uniba_checkpoints.tar.gz`](https://drive.google.com/file/d/1z1xSP5U5GkCvLTmrMAnlxp8qUMspBr9y/view?usp=sharing) | Five UniBA fold checkpoints | ~352 MB |

### External pre-trained models

| Model | Purpose | Link |
| --- | --- | --- |
| MINT | Paired sequence representation | Hugging Face |
| [ESM-2 (`esm2_t33_650M_UR50D`)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt) | Residue-level sequence representation | fair-esm |
| [ESM-IF1 (`esm_if1_gvp4_t16_142M_UR50`)](https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt) | Structure-aware protein representation | fair-esm |

Please place the downloaded files in the corresponding directories:


## 5. Process datasets
Before training or inference, the downloaded protein complex structures need to be processed to generate the inputs required by UniBA. The data processing pipeline includes preprocessing atomic-level PDB structures and generating MARTINI coarse-grained (CG) representations.

First, enter the data/ directory and process the protein complex structures:
```bash
cd data/
python process_data.py
```

Then, convert the processed atomic-level PDB structures into MARTINI coarse-grained models using the provided script. The MARTINI coarse-graining step requires a Python 2 environment:
```bash
conda activate python2
cd ./data/cg_input/
./martini_steps.sh
```

## 6. Generate features and construct graphs

After processing the protein complex structures, UniBA extracts sequence and structural representations and constructs multi-scale graphs required for affinity prediction.

Run the feature generation and graph construction script:

```bash
python generate_graph.py
Run the full workflow:

```bash
bash scripts/run_raw_pdb_feature_pipeline.sh \
  --env-file scripts/raw_pdb_pipeline.env
python scripts/validate_reproducibility.py --mode generated
```

The stages can be resumed independently after a failure:

```bash
# Raw pair PDB -> mapped complex, per-chain PDB, and FASTA
bash scripts/run_raw_pdb_feature_pipeline.sh \
  --env-file scripts/raw_pdb_pipeline.env --stages data

# MINT, ESM, handcrafted, PLIP, Martini/Gromacs, and energy features
bash scripts/run_raw_pdb_feature_pipeline.sh \
  --env-file scripts/raw_pdb_pipeline.env \
  --stages mint,seqstr,handnorm,plip,cgprep,energy

# Residue and coarse-grained graphs
bash scripts/run_raw_pdb_feature_pipeline.sh \
  --env-file scripts/raw_pdb_pipeline.env --stages graphs
```

Set `DATASETS=ppi`, `DATASETS=aai`, or `DATASETS=tcr-pmhc` before the command
to resume one dataset. The pipeline skips complete Martini outputs and existing
graphs. If conjugate-gradient minimization fails for a structure, the wrapper
automatically retries with steepest descent and records
`MINIMIZATION_FALLBACK.txt` in that pair's CG directory.

Full generation is much larger than the 215 MB raw archive. ESM checkpoints,
per-residue features, PLIP reports, solvated Gromacs systems, energy matrices,
and graph files can require hundreds of GB. Check disk space before a full run.
These files are intentionally excluded from Git.

## 8. Run inference or training

After all features and graphs for a requested split exist:

```bash
# Five-fold released model on data/split_data/cv/test_pair.txt
python UniAffinity.py --test --cv 5 --dataset test

# Individual external test split
python UniAffinity.py --test --cv 5 --dataset test1

# Full retraining, followed by testing the new checkpoints
python UniAffinity.py --train --no-test --cv 5 --num_epochs 100
python UniAffinity.py --test --cv 5 --dataset test
```

Predictions are written under `output/UniBA_wo_res/`; analysis tensors are
written under `analysis_graph/`. Both directories are ignored by Git.

## 9. Data contract and layout

A pair ID has the form `PDB_partner1chains_partner2chains`. For example,
`1akj_DE_AB` assigns D/E to partner 1 and A/B to partner 2. The data preparation
stage maps selected chains to A, B, C, ... in partner order for all downstream
tools.



## Reproduction status

The raw-PDB workflow was tested end to end on Linux on 2026-07-30 with one
representative from every dataset:

| Dataset | Pair | Raw-to-graph | Fold-0 prediction |
| --- | --- | --- | ---: |
| PPI | `1a22_A_B` | passed | 8.00357 |
| AAI | `1bj1_HL_W` | passed | 8.34333 |
| TCR-pMHC | `1ao7_DE_CA` | passed | 5.88164 |

The test covered chain extraction/remapping, MINT, ESM2, ESM-IF, DSSP,
GHECOM, GraphRicciCurvature, PLIP/Open Babel, Martini 2.2, Gromacs energy
minimization, energy features, residue graphs, coarse-grained graphs, checkpoint
loading, and model inference.

The empty generated-output directories visible on the server are not source
folders that need uploading; Git does not track empty directories. This
`README.md` is the repository's only Markdown documentation. See `LICENSE`,
`CITATION.cff`, and the source paper for licensing and citation information.
