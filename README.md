# flipcrow

Single-cell RNA-seq reanalysis of metastatic pancreatic ductal adenocarcinoma (PDAC),
reimplementing the core analysis of Winter et al. in Scanpy.

**Data:** [SCP1644](https://singlecell.broadinstitute.org/single_cell/study/SCP1644/microenvironment-drives-cell-state-plasticity-and-drug-response-in-pancreatic-cancer)
— matched scRNA-seq profiles and organoid models from core needle biopsies of metastatic
PDAC, profiled on Seq-Well S³.

**Paper:** Raghavan S, Winter PS, Navia AW, et al. Microenvironment drives cell state,
plasticity, and drug response in pancreatic cancer. *Cell* 184(25):6119–6137.e26 (2021).
[cell.com](https://www.cell.com/cell/fulltext/S0092-8674(21)01332-5)

The goal was to rebuild the published single-cell analysis end to end from raw digital
gene expression matrices, and to check whether the paper's tumor cell-state structure
falls out of an independent implementation.

## Installation instructions

1. Clone the repo.
2. `pip install -e .` to install needed libraries.
3. Download SCP1644 data into the `scp1644_data` folder in the repo using `curl` via the Single Cell Portal link above.
4. Download [Human MSigDB gene sets](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2026.1.Hs/msigdb_v2026.1.Hs_files_to_download_locally.zip) - you will need to register - and unzip it into the `scp1644_data` folder.
5. Download the [De Koning 2021](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.649061/full) [supplementary datasheet 1](https://public-pages-files-2025.frontiersin.org/articles/649061/file/Data_Sheet_1.xlsx/649061_supplementary-materials_datasheets_1_xlsx/1) into a `scp1644_data/markergenes` folder and rename it `DeKoning2021.xlsx`.

## What it does

**Loading and metadata reconciliation.** Reads the raw DGE matrices (1,370-cell and
23,042-cell biopsy sets) and the 70,170-cell metadata table, reconciles biosample IDs
against cell barcode tags, and handles the `PANFR0489R` resampled donor case described in
the paper's methods.

**Per-sample quality control.** QC metrics are computed per biosample before merging, with
mitochondrial, ribosomal, and hemoglobin gene sets flagged. Cells are dropped below 400
genes, below 1,000 UMIs, or above 50% mitochondrial counts. After merging, cells above
8,000 genes are dropped as probable doublets and genes detected in fewer than 50 cells are
removed.

**Doublet and low-quality cluster screening.** Scrublet doublet scores plus a
cluster-level mitochondrial screen: `marker_gene_overlap` scores each Leiden cluster
against the full MT/MTRNR gene set, and clusters above a mitochondrial fraction cutoff are
filtered alongside high doublet scores.

**Normalization and embedding.** Counts-per-10,000 normalization, log1p, highly variable
gene selection (Seurat flavor), PCA, nearest neighbors, Leiden clustering, UMAP and t-SNE.

**Cell type annotation.** Coarse cell types are called by marker-set overlap against the
published marker table (filtered to power ≥ 0.6), extended with tumor keratins and the
Moffitt basal and classical PDAC signatures, then assigned per Leiden cluster by argmax of
the overlap coefficient.

**Tumor cell-state analysis.** Tumor cells are re-embedded after removing the
neuroendocrine donor (`PANFR0580`) and donors with fewer than 25 cells. Moffitt classical
and basal signature scores are computed per cell with `score_genes`, and PCA loadings are
ranked to identify the axes carrying state structure — an EMT and fibroblast-like program
opposite classical on PC0, and basal opposite classical on PC2 — visualized as
signature-gene heatmaps with cells ordered by PC score.

**Differential expression and enrichment.** Kruskal-Wallis tests across Leiden clusters
over highly variable genes, Mann-Whitney tests of each cluster against the rest and
against zero, Scanpy's `rank_genes_groups`, and pathway enrichment through `gseapy`
prerank against Enrichr libraries.

## Layout

```
src/flipcrow/scp1644_mothership.py   loading, QC, preprocessing, annotation, DE, tumor analysis
src/flipcrow/paths.py                data path resolution
nbs/SCP1644_preprocess_*.ipynb       preprocessing and coarse cell type annotation
nbs/SCP1644_paper*.ipynb             analysis passes I–VII
nbs/SCP1644_FigS1.ipynb              supplementary figure reproduction
```

Notebooks are kept with outputs so the figures are readable without rerunning.

## Running it

Expects the SCP1644 expression matrices and metadata downloaded from the Single Cell
Portal under `data/SCP1644/`, and the paper's marker gene table at
`data/markergenes/mmc2.xlsx`.

```
pip install -e .                 # the library
pip install -e ".[notebooks]"    # plus jupyter, decoupler, pydeseq2
```

Requires Python 3.11+. Dependencies are declared in `pyproject.toml`.

## Tests

```
pip install -e ".[dev]"
pytest
```

Covers the QC filtering logic in `apply_qc`. The thresholds documented above are
asserted against synthetic `AnnData` objects, so changing one without the other fails.

## Logging

Progress output goes through the standard library logger rather than `print`, so it can
be levelled, redirected or silenced. The package attaches only a `NullHandler`, which
means **nothing is printed until you ask for it**. From a notebook:

```python
from flipcrow.scp1644_mothership import configure_logging
configure_logging()
```

## Notes

Analysis code and plotting are interleaved in the module; the notebooks are the working
record of the analysis rather than a packaged pipeline.

### AI assistance

The analysis code in this repository was written without AI assistance. On 2026-09-14
the packaging metadata, import cleanup, docstrings, and the test suite were revised with
AI assistance; no analysis logic or threshold was changed. Affected sections are marked
in the source.
