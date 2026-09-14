"""SCP1644 single-cell reanalysis: loading, QC, preprocessing, annotation and tumour analysis.

Reimplementation of the core single-cell analysis of Raghavan S, Winter PS, Navia AW,
et al. "Microenvironment drives cell state, plasticity, and drug response in pancreatic
cancer." Cell 184(25):6119-6137.e26 (2021), against the SCP1644 dataset.

Analysis code and plotting are interleaved; the notebooks under nbs/ are the working
record of the analysis rather than a packaged pipeline.

AI assistance: the analysis code in this module was written by the author without AI
assistance. On 2026-09-14 these docstrings, the import cleanup, and the trimmed
__main__ block were added with AI assistance; no analysis logic or threshold was
changed.
"""
from collections import Counter
from typing import Union, Mapping
import logging
import time
import warnings

import pandas as pd
import numpy as np
import scipy

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

import flipcrow
import flipcrow.paths

warnings.filterwarnings("ignore")

sc.set_figure_params(figsize=(10, 10))

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Send this module's progress output to stderr at the given level.

    The package attaches only a NullHandler by default, as a library should, so nothing
    is printed unless asked for. Call this once from a notebook or script to see the
    progress messages that the analysis functions emit.
    """
    logging.basicConfig(format="%(message)s", level=level)
    logger.setLevel(level)


def log(*args) -> None:
    """Progress logging with print's calling convention, routed through the logger.

    Takes the same varargs as print() so the call sites read the same way, but output is
    levelled and silenceable instead of going unconditionally to stdout.
    """
    logger.info(" ".join(str(a) for a in args))


moffitt_pdac = {
    'Basal': [
        'VGLL1', 
        'UCA1', 
        'S100A2', 
        'LY6D', 
        'SPRR3',
        'SPRR1B',
        'LEMD1',
        'KRT15',
        'CTSV', # 'CTSL2',
        'DHRS9',
        'AREG',
        'CST6',
        'SERPINB3',
        # 'KRT6C', # not present in var_names
        'KRT6A',
        'FAM83A',
        'SCEL',
        'FGFBP1',
        'KRT7',
        'KRT17',
        'GPR87',
        'TNS4',
        'SLC2A1',
        'ANXA8L2',
    ],
    'Classical': [
        'BTNL8',
        'FAM3D',
        'PRR15L', # 'ATAD4',
        'AGR3',
        'CTSE',
        # 'TMEM238L', # 'LOC400573', not present in genes
        # 'LYZ', # common in T cells
        'TFF2',
        'TFF1',
        'ANXA10',
        'LGALS4',
        'PLA2G10',
        'CEACAM6',
        'VSIG2',
        'TSPAN8',
        'ST6GALNAC1',
        'AGR2',
        'TFF3',
        'CYP3A7',
        'MYO1A',
        'CLRN3',
        'KRT20',
        'CDH17',
        'SPINK4',
        'REG4',
    ],
}

tumor_keratins = {
    'Tumor_keratins': [
        'KRT5',
        'KRT6A',
        'KRT7',
        'KRT8',
        'KRT10',
        'KRT13',
        'KRT14',
        'KRT15',
        'KRT17',
        'KRT18',
        'KRT19',
    ]
}


def load_scp1644_data(check_metadata: bool = True) -> ad.AnnData:
    """Load the SCP1644 expression matrices and attach the published cell metadata.

    Reads the 1,370-cell and 23,042-cell raw DGE matrices, concatenates them, and joins
    the 70,170-cell metadata table by cell barcode. With `check_metadata` set, prints a
    reconciliation of biosample IDs against barcode tags: the metadata carries 24 biosample
    IDs including the resampled donor PANFR0489R while the barcode tags carry 23, and this
    report makes that discrepancy visible rather than silent.

    Also normalises the 'Hepatocytes'/'Hepatocyte' label inconsistency and drops the
    'exvivo.treatment' column, whose NaNs prevent h5ad round-tripping.
    """
    scp1644_small_csv = flipcrow.paths.DATA_PATH / 'SCP1644/expression/Biopsy473_RawDGE_1370cells.csv'
    scp1644_big_csv = flipcrow.paths.DATA_PATH / 'SCP1644/expression/Biopsy_RawDGE_23042cells.csv'
    scp1644_metadata_csv = flipcrow.paths.DATA_PATH / 'SCP1644/other/complete_Metadata_70170cells_scp.csv'
    
    start_load = time.time()
    adata_small = sc.read_csv(scp1644_small_csv).T
    adata_big = sc.read_csv(scp1644_big_csv).T
    adata_metadata = pd.read_csv(scp1644_metadata_csv)
    end_load = time.time()

    log(f"Loaded expression matrices in {end_load - start_load:.1f}s")

    adata_scratch = ad.concat([adata_small, adata_big])

    # The metadata has 24 unique biosample IDs including PANFR0489R, but only 23 unique
    # cell UMI tags, which exclude it: PANFR0489 was resampled as PANFR0489R. Per the
    # paper's STAR Methods, that model failed to expand in complete organoid medium and
    # was re-established in medium without A83-01 or mNoggin, then sequenced at early
    # passage - see Raghavan et al., Cell 184(25):6119-6137.e26 (2021), STAR Methods,
    # organoid culture. The resampled data was included in the published analysis, so
    # it is kept here.

    if check_metadata:
        biosample_ids = Counter([x for x in adata_metadata.loc[:, 'biosample_id'] if 'Biopsy' in x])
        cell_tags = Counter(['_'.join(idx.split('_')[:2]) for idx in adata_scratch.obs_names])

        def mangle_biosample_ids(x):
            return '_'.join(x.split('_')[-1::-1][1:])

        log('='*40)
        log("Checking metadata!")
        log('='*40)
        log('Biosample IDs')
        log(biosample_ids)
        log('Cell tags')
        log(cell_tags)

        mangle_set = set(map(mangle_biosample_ids, biosample_ids.keys()))
        log("Biosample ID length", len(biosample_ids))
        log("Mangle set length", len(mangle_set))
        log("Cell tags length", len(cell_tags))
        
        log("Biosamples not in cell tags", mangle_set - cell_tags.keys())
        log("Cell tags not in biosamples", cell_tags.keys() - mangle_set)
        log()

    adata_metadata_fix = adata_metadata.set_index('NAME', drop=True).iloc[1:, :]

    adata_metadata_fix.index.name = None
    for col in adata_metadata_fix.columns:
        metadata = adata_metadata_fix.loc[adata_scratch.obs_names, col]
        adata_scratch.obs[col] = metadata

    # fix hepatocyte labeling
    adata_scratch.obs.loc[adata_scratch.obs.loc[:, 'Coarse_Cell_Annotations'] == 'Hepatocytes', 'Coarse_Cell_Annotations'] = 'Hepatocyte'
    
    # fix NaN in exvivo.treatment to be able to h5ad read/write
    del adata_scratch.obs['exvivo.treatment']

    log('='*40)
    log("Metadata stats")
    log('='*40)
    annot_count = Counter(adata_scratch.obs['Coarse_Cell_Annotations'])
    log("Coarse_Cell_Annotations count")
    log(annot_count)

    sex_count = Counter(adata_scratch.obs['sex'])
    log("sex_count")
    log(sex_count)
    
    biosample_count = Counter(adata_scratch.obs['biosample_id'])
    log("biosample_count")
    log(biosample_count)

    donor_count = Counter(adata_scratch.obs['donor_ID'])
    log(donor_count)
    
    log(f"Total biosamples {len(biosample_count)}")
    log(f"Total donors {len(donor_count)}")

    return adata_scratch


def apply_qc(adata: ad.AnnData, mode: str = "premerge", verbose: bool = False) -> ad.AnnData:
    """Filter cells and genes against the QC thresholds for one stage of the pipeline.

    `premerge` drops cells with fewer than 400 detected genes, fewer than 1,000 total
    counts, or more than 50% mitochondrial counts, and touches no genes. `postmerge` drops
    cells above 8,000 detected genes as probable doublets, and genes detected in fewer than
    50 cells.

    Per-criterion boolean flags are written onto `adata.obs` and `adata.var` before
    filtering, so why a given cell or gene was dropped stays inspectable. Raises ValueError
    on any other mode.
    """

    # # fewer than 400 genes - low quality cell
    # # more than 8000 genes - probable doublet
    # # fewer than 1000 UMIs / total counts / transcripts per cell
    # # fewer than 50 cells detected with gene expressed
    # # more than 50% mitochondrial counts
    if mode == "premerge":
        qc_cell = {
            'u400genes': adata.obs.n_genes_by_counts < 400,
            'u1000counts': adata.obs.total_counts < 1000,
            'o50mtpct': adata.obs.pct_counts_mt > 50,
        }
        qc_gene = {}

    elif mode == "postmerge":
        qc_cell = {
            'o8000genes': adata.obs.n_genes_by_counts > 8000,
        }
        qc_gene = {
            'low_qual_u50cells': adata.var.n_cells_by_counts < 50,
        }
    else:
        raise ValueError("mode must be premerge or postmerge")
            
    for k, v in qc_cell.items():
        if verbose:
            log(f"{k}: {sum(v)}")
        adata.obs[k] = v
    
    for k, v in qc_gene.items():
        if verbose:
            log(f"{k}: {sum(v)}")
        adata.var[k] = v

    if len(qc_cell) > 0:
        mask_cell = np.any(np.vstack([adata.obs[v] for v in qc_cell.keys()]), axis=0)
    else:
        mask_cell = np.array([False for v in adata.obs_names])

    if len(qc_gene) > 0:
        mask_gene = np.any(np.vstack([adata.var[v] for v in qc_gene.keys()]), axis=0)
    else:
        mask_gene = np.array([False for v in adata.var_names])
    
    if verbose:
        log('mask_cell', mask_cell.shape)
        log('mask_cell present?', np.any(mask_cell))
        log('mask_gene', mask_gene.shape)
        log('mask_gene present?', np.any(mask_gene))
    
    aw = adata[~mask_cell, ~mask_gene]
    return aw


def preprocess_scp1644_data(adata: ad.AnnData, scrublet: bool = True) -> ad.AnnData:
    """Run per-biosample QC and merge.

    Splits the input by `biosample_id` and computes QC metrics separately for each —
    flagging mitochondrial, ribosomal and haemoglobin gene sets — so one poor-quality
    biosample cannot drag the thresholds for the rest. Applies the premerge filter per
    chunk, concatenates, recomputes QC on the merged object, and applies the postmerge
    filter.
    """
    qc_buf = {}

    log("Per chunk QC")
    for id in list(set(adata.obs['biosample_id'])):
        adata_chunk = adata[adata.obs['biosample_id']==id, :]

        # mitochondrial genes
        adata_chunk.var['mt'] = adata_chunk.var_names.str.startswith(("MT-"))
        # ribosomal genes
        adata_chunk.var["ribo"] = adata_chunk.var_names.str.startswith(("RPS", "RPL"))
        # hemoglobin genes.
        adata_chunk.var["hb"] = adata_chunk.var_names.str.contains(("^HB[^(P)]"))
        
        # do QC calc for real
        sc.pp.calculate_qc_metrics(
            adata_chunk, 
            qc_vars=["mt", "ribo", "hb"], 
            inplace=True, 
            percent_top=[20], 
            log1p=False,
        ) 

        qc_buf[id] = apply_qc(adata_chunk)

    log("Concatenate")
    for k, v in qc_buf.items():
        log(f"{k} : {v.shape}")

    log()
    qc_join = ad.concat([x for x in qc_buf.values()])
    del qc_buf

    log("Merged QC")
    # mitochondrial genes
    qc_join.var['mt'] = qc_join.var_names.str.startswith(("MT-"))
    # ribosomal genes
    qc_join.var["ribo"] = qc_join.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    qc_join.var["hb"] = qc_join.var_names.str.contains(("^HB[^(P)]"))

    sc.pp.calculate_qc_metrics(
        qc_join, 
        qc_vars=["mt", "ribo", "hb"], 
        inplace=True, 
        percent_top=[20], 
        log1p=False,
    ) 

    # Remove cells with more than 8000 genes and genes present in fewer than 50 cells
    clean = apply_qc(qc_join, mode="postmerge")
    del qc_join

    return clean

    # if scrublet:
    #     log("Scrublet")
    #     # apply scrublet while we still have raw count data
    #     sc.pp.scrublet(clean)

    # clean.layers['trimmed_counts'] = clean.to_df().loc[clean.obs_names, :]
    # sc.pp.normalize_total(clean, target_sum=10000, inplace=True)
    # sc.pp.log1p(clean, copy=False)
    
    # log("Dimensionality reduction and clustering")
    # sc.pp.pca(clean)
    # sc.pp.neighbors(clean, n_neighbors=40, n_pcs=50)
    # sc.tl.leiden(clean)
    # sc.tl.umap(clean)
    # sc.tl.tsne(clean)

    # log("HVG + Rank genes groups")
    # sc.pp.highly_variable_genes(clean, flavor="seurat")
    # sc.tl.rank_genes_groups(clean, 'leiden')

#     return clean


def mito_doublet_screen(adata: ad.AnnData, mt_cutoff: float = 0.15, doublet_cutoff: float = 0.1, use_pct_mt_stats=False) -> ad.AnnData:
    """Remove low-quality Leiden clusters by mitochondrial content, then re-embed.

    Scores every Leiden cluster against the full MT-/MTRNR- gene set with
    `sc.tl.marker_gene_overlap`, giving a per-cluster mitochondrial fraction that catches
    dying-cell clusters a per-cell percentage threshold can miss. Cells are kept where that
    cluster-level fraction is below `mt_cutoff` and the Scrublet doublet score is below
    `doublet_cutoff`.

    The survivors are then re-embedded from scratch — PCA, neighbours, Leiden, UMAP, t-SNE,
    HVG selection and `rank_genes_groups` — since removing whole clusters changes the
    structure the original embedding was fit to. Plots before-and-after t-SNEs.

    Requires `adata.obs['doublet_score']`, populated by `sc.pp.scrublet` in the notebooks.
    """
    log("Initial tSNE")
    sc.pl.tsne(adata, color=['leiden'], legend_loc="on data", size=60)
    sc.pl.tsne(adata, color='donor_ID', size=60)
    plt.show()

    log("Calculating cluster stats")
    leiden_mt_stats = adata.obs.groupby('leiden').agg({'pct_counts_mt': ['mean', 'std', 'max']}).droplevel(level=0, axis=1).sort_values('mean')
    plt.figure()
    plt.plot(leiden_mt_stats.loc[:, 'mean'])
    plt.plot(leiden_mt_stats.loc[:, 'mean'] + 2.0*leiden_mt_stats.loc[:, 'std'])
    plt.plot(leiden_mt_stats.loc[:, 'mean'] - 2.0*leiden_mt_stats.loc[:, 'std'])
    plt.plot(leiden_mt_stats.loc[:, 'max'])
    plt.show()

    log("Running mitoscreen")
    ggb_mitoscreen = {'GGB_MITOCHONDRIAL': [x for x in adata.var_names if x.startswith('MT-') or x.startswith('MTRNR')]}
    ggb_mito_result = sc.tl.marker_gene_overlap(adata, ggb_mitoscreen, method='overlap_count', normalize='reference')
    plt.figure()
    plt.plot(ggb_mito_result.loc['GGB_MITOCHONDRIAL', :])
    log(ggb_mito_result.loc['GGB_MITOCHONDRIAL', :].sort_values(ascending=False))
    plt.show()

    for idx in ggb_mito_result.index:
        adata.obs['GGB_MITOCHONDRIAL'] = [ggb_mito_result.loc['GGB_MITOCHONDRIAL', leid_idx] for leid_idx in adata.obs['leiden']]
    
    adata_filter = adata[np.logical_and(adata.obs.GGB_MITOCHONDRIAL < mt_cutoff, adata.obs.doublet_score < doublet_cutoff), :]

    log("Preprocessing filtered dataset")
    sc.pp.pca(adata_filter)
    sc.pp.neighbors(adata_filter, n_neighbors=40, n_pcs=50)
    sc.tl.leiden(adata_filter)
    sc.tl.umap(adata_filter)
    sc.tl.tsne(adata_filter)
    sc.pp.highly_variable_genes(adata_filter, flavor="seurat")
    sc.tl.rank_genes_groups(adata_filter, 'leiden')

    log("Final tSNE")
    sc.pl.tsne(adata_filter, color=['leiden'], legend_loc="on data", size=60)
    sc.pl.tsne(adata_filter, color='donor_ID', size=60)
    plt.show()

    return adata_filter
    

def diffexpr_tests(adata: ad.AnnData): # -> Tuple[ad.AnnData, Mapping[str, KruskalType,....
    """Test highly variable genes for differential expression across Leiden clusters.

    Runs a Kruskal-Wallis test per HVG across all clusters, writing statistics and p-values
    onto `adata.var`, then Mann-Whitney tests of each cluster against all other cells and
    against zero. The second guards against genes that appear to separate clusters only
    through sparsity rather than expression level.

    Returns (adata copy, Kruskal-Wallis results by gene, Mann-Whitney p-values against other
    clusters, Mann-Whitney p-values against zero). The 1e-9 and 1e-5 cutoffs apply to the
    printed summary only; the returned frames are unfiltered.
    """
    samples = [adata[adata.obs.leiden == cluster_id, adata.var.highly_variable] for cluster_id in adata.obs.leiden.unique()]
    others = [adata[adata.obs.leiden != cluster_id, adata.var.highly_variable] for cluster_id in adata.obs.leiden.unique()]

    log("Performing KW test")
    kw_map = {}
    for idx, gene in enumerate(adata.var_names[adata.var.highly_variable]):
        
        kw_map[gene] = scipy.stats.kruskal(*[x[:, gene].X.squeeze() for x in samples])
        if idx % 100 == 0:
            log(gene, kw_map[gene])

    kw_srs = pd.Series(kw_map)
    adata.var['kw_statistic'] = np.zeros(len(adata.var_names)) 
    adata.var['kw_pval'] = np.ones(len(adata.var_names)) 

    adata.var['kw_statistic'].loc[kw_map.keys()] = [x.statistic for x in kw_map.values()]
    adata.var['kw_pval'].loc[kw_map.keys()] = [x.pvalue for x in kw_map.values()]            

    # performs pairwise significance tests
    log("Performing pairwise significance tests")
    
    buf_others = []
    buf_zeros = []

    start_gene_time = time.time()
    for idx in range(len(samples)):
        # log(idx)
        start_mean_time = time.time()
        buf_others.append(scipy.stats.mannwhitneyu(samples[idx].X, others[idx].X))
        buf_zeros.append(scipy.stats.mannwhitneyu(samples[idx].X, np.zeros(samples[idx].X.shape), alternative='greater'))
        end_zeros_time = time.time()
        log(idx, (end_zeros_time - start_mean_time))    
    end_gene_time = time.time()
    log('done', (end_gene_time - start_gene_time))
    
    mwm_others_df = pd.DataFrame([x.pvalue for x in buf_others], index=adata.obs.leiden.unique(), columns=adata.var['highly_variable'][adata.var['highly_variable']].index)
    mwm_zeros_df = pd.DataFrame([x.pvalue for x in buf_zeros], index=adata.obs.leiden.unique(), columns=adata.var['highly_variable'][adata.var['highly_variable']].index)

    mwm_others_pval_cutoff = 1.0e-9
    mwm_zeros_pval_cutoff = 1.0e-5

    mwm_mean_pass = mwm_others_df < mwm_others_pval_cutoff
    mwm_zeros_pass = mwm_zeros_df < mwm_zeros_pval_cutoff

    log(mwm_mean_pass & mwm_zeros_pass)

    return adata.copy(), kw_map, mwm_others_df, mwm_zeros_df
    

def plot_cluster_gene_distr_1d(adata, samples, others, cluster_key, gene_key, plot=True):
    """Compare one gene's expression histogram in one cluster against all other cells.

    Returns the `rank_genes_groups` frame for that cluster and the same frame joined to the
    highly-variable flag, and reports whether the gene of interest appears in it.
    """

    coi = int(cluster_key)
    samples_hist = np.histogram(samples[coi][:, gene_key].X.squeeze(), bins=20)
    others_hist = np.histogram(others[coi][:, gene_key].X.squeeze(), bins=20)
    # log(samples_hist)
    if plot:
        plt.plot(others_hist[1][:-1], others_hist[0]/others_hist[0].sum(), color='r')
        plt.plot(samples_hist[1][:-1], samples_hist[0]/samples_hist[0].sum(), color='b')
        plt.axis([0, 3.5, 0, 0.2])
    
    this_cluster = sc.get.rank_genes_groups_df(adata, str(cluster_key))
    join_cluster = this_cluster.join(adata.var.highly_variable, on='names')
    log(f"{gene_key} in {cluster_key}: {gene_key in join_cluster.names.values}")
    
    if gene_key in join_cluster.names.values:
        log(join_cluster.loc[join_cluster.names == gene_key, :])

    return this_cluster, join_cluster


# TODO:  plot co-expression by chromosomal location
def check_rank_genes_groups_df(adata: ad.AnnData, coi: Union[int, str], focus_prefix=None) -> None: 
    """Print the top ranked genes for one cluster, optionally filtered by name prefix.

    `focus_prefix` restricts the listing to genes starting with that string — used to pull
    out keratins, mitochondrial genes and other families by name.
    """

    rgg_df = sc.get.rank_genes_groups_df(adata, coi)
    log(sc.get.rank_genes_groups_df(adata, coi).iloc[:30, :])
    if focus_prefix:
        with pd.option_context('display.min_rows', 50):
            log(rgg_df.loc[[x.startswith(focus_prefix) for x in rgg_df.names.values], :])

    log('\n'.join(sc.get.rank_genes_groups_df(adata, coi).names[:100]))


def classical_basal_analysis(adata: ad.AnnData) -> None:
    """Score every cell for the Moffitt classical and basal PDAC signatures.

    Writes `classical_score`, `basal_score` and their difference onto `adata.obs` using
    `sc.tl.score_genes` against 100 control genes, and plots one against the other.
    """
    sc.tl.score_genes(adata, moffitt_pdac['Classical'], score_name='classical_score', ctrl_size=100)
    sc.tl.score_genes(adata, moffitt_pdac['Basal'], score_name='basal_score', ctrl_size=100)
    adata.obs.loc[:, 'diff_score'] = adata.obs.classical_score - adata.obs.basal_score
    plt.plot(adata.obs.classical_score, adata.obs.basal_score, 'o')


def compare_classical_basal_genes(adata: ad.AnnData, classical_gene: str, basal_gene: str) -> None:
    """Check individual marker genes against the aggregate signature scores.

    For one classical and one basal marker, plots each gene's expression against both
    signature scores and prints Pearson and Spearman correlations — a sanity check that the
    aggregate scores track the genes they are built from.
    """

    plt.plot(adata.obs.classical_score, adata.to_df().loc[:, classical_gene].values, '.')
    plt.plot(adata.obs.basal_score, adata.to_df().loc[:, classical_gene].values, '.')

    plt.figure(1)
    log(f'{classical_gene} (Classical)')
    log('Pearson Classical', scipy.stats.pearsonr(adata.obs.classical_score, adata.to_df().loc[:, classical_gene].values))
    log('Pearson Basal', scipy.stats.pearsonr(adata.obs.basal_score, adata.to_df().loc[:, classical_gene].values))

    log('Spearman Classical', scipy.stats.spearmanr(adata.obs.classical_score, adata.to_df().loc[:, classical_gene].values))
    log('Spearman Basal', scipy.stats.spearmanr(adata.obs.basal_score, adata.to_df().loc[:, classical_gene].values))

    log()
    plt.figure(2)
    plt.plot(adata.obs.classical_score, adata.to_df().loc[:, basal_gene].values, '.')
    plt.plot(adata.obs.basal_score, adata.to_df().loc[:, basal_gene].values, '.')

    log(f'{basal_gene} (Basal)')
    log('Pearson Classical', scipy.stats.pearsonr(adata.obs.classical_score, adata.to_df().loc[:, basal_gene].values))
    log('Pearson Basal', scipy.stats.pearsonr(adata.obs.basal_score, adata.to_df().loc[:, basal_gene].values))

    log('Spearman Classical', scipy.stats.spearmanr(adata.obs.classical_score, adata.to_df().loc[:, basal_gene].values))
    log('Spearman Basal', scipy.stats.spearmanr(adata.obs.basal_score, adata.to_df().loc[:, basal_gene].values))





# TODO: Train this like a little NN
def coarse_cell_type_annotation(adata: ad.AnnData) -> pd.DataFrame:
    """Assign a coarse cell type to each Leiden cluster by marker-set overlap.

    Reads the published marker table (mmc2.xlsx), keeping markers with power >= 0.6, and
    works around gene symbols Excel has silently converted to dates (MARCH1 -> 1-Mar) by
    discarding non-string entries. Extends the marker sets with tumour keratins and the
    Moffitt basal and classical signatures, then assigns each Leiden cluster the cell type
    with the highest overlap coefficient.

    Returns the cluster-by-cell-type overlap matrix.
    """

    normal_markers = pd.read_excel(flipcrow.paths.DATA_PATH / "markergenes" / "mmc2.xlsx", header=4, dtype=object)
    normal_markers = normal_markers.iloc[:, 2:]
    normal_markers = normal_markers.loc[:, ['Gene', 'myAUC', 'avg_diff', 'power', 'avg_logFC', 'pct.1', 'pct.2', 'cell.type']]
    normal_markers_dict = {}

    # filter datetime oddities found in original data probably caused by entering MARCH1, MARCH11 etc in dataset. This is (possibly?) auto converted to 11-Mar, 1-Mar, etc 
    for grpid, grp in normal_markers.groupby('cell.type'):
        normal_markers_dict[grpid] = [grp.loc[idx, 'Gene'] for idx in grp.index if type(grp.loc[idx, 'Gene']) == str and float(grp.loc[idx, 'power']) >= 0.6]

    #log(normal_markers_dict)
    markers_dict = normal_markers_dict.copy()

    # add tumor keratins
    markers_dict.update(tumor_keratins)



    # Make large leiden-based heatmap of the markers dict (Table 2)
    markers_dict.update(moffitt_pdac)
    sc.pl.heatmap(adata, markers_dict, 'leiden', figsize=(20, 20))

    # show the marker_gene_overlap verdict of the markers_dict against leiden
    # should probably rank_genes_groups first here
    verdict = sc.tl.marker_gene_overlap(adata, markers_dict, method='overlap_coef').T
    fig, ax = plt.subplots()
    im = ax.imshow(verdict.T)
    ax.set_xticks(np.arange(len(verdict.index)))
    ax.set_yticks(np.arange(len(verdict.columns)), labels=verdict.columns)
    ax.grid(None)
    plt.show()

    # count the number of cells in each leiden cluster
    for leiden_idx in sorted(adata.obs.leiden.unique().astype(int)):
        log(leiden_idx, Counter(adata.obs.loc[adata.obs.leiden == str(leiden_idx), 'Coarse_Cell_Annotations']))
    
    # identifiy the call by argmax
    # TODO: Modify this to trigger on significant amounts of tumor keratins or basal/classical
    call_dict = verdict.T.index[np.argmax(verdict.T, axis=0)]

    # relabel markers from table to file
    replace_dict = {
        'T_Cells': 'T_NK',
        'Macrophage': 'Macrophage',
        'Tumor_keratins': 'Tumor',
        'Basal': 'Tumor',
        'Classical': 'Tumor',
        'B_Cells': 'B_Cells',
        'DC': 'DC',
        'Mesenchymal': 'Mesenchymal',
        'Liver_Cell': 'Hepatocyte',
        'Plasma_cell': 'Plasma_cell',
        'T_Regs': 'T_Regs',
        'Endothelial': 'Endothelial',
        'pDC_cell': 'pDC_cell',
        'cp_DC': 'XCR1_DC',
    }

    # assign verdicts based on tumor assignment to each leiden group
    # TODO: should try this by donor_ID instead
    verdict_strings = list(verdict.T.index[np.argmax(verdict.T, axis=0)])
    verdict_buf = []
    # note nested dictionaries here
    for leid_idx in adata.obs.leiden:
        verdict_buf.append(replace_dict[verdict_strings[int(leid_idx)]])


    adata.obs.loc[:, 'local_coarse_call'] = verdict_buf

    verdict_strings = list(verdict.T.index[np.argmax(verdict.T, axis=0)])
    verdict_buf = []
    # note nested dictionaries here
    for donor in adata.obs.donor_ID:
        verdict_buf.append(replace_dict[verdict_strings[int(leid_idx)]])

    adata.obs.loc[:, 'local_coarse_call'] = verdict_buf

    return verdict

    
def tumor_study(adata: ad.AnnData, n_cells_cutoff=25) -> None:
    """Re-embed the tumour cells alone and rank PCs for cell-state structure.

    Subsets to cells annotated Tumor, drops the neuroendocrine donor PANFR0580 and any donor
    contributing fewer than `n_cells_cutoff` cells, then recomputes HVGs, PCA, neighbours,
    Leiden and t-SNE on what remains — tumour structure is otherwise dominated by donor
    identity.

    Stores per-PC gene rankings in `uns['PC_ranks']` and plots the variance spectrum and the
    Moffitt signature genes against PC scores, which is how the classical/basal and
    EMT/fibroblast axes were identified.
    """
    tumor_adata = adata[adata.obs['Coarse_Cell_Annotations'] == 'Tumor']
    
    # demonstrate PCA isolation of PANFR0580 (NET tumor)
    log("All tumors")
    sc.pp.pca(tumor_adata)
    sc.pl.pca(tumor_adata, color=['leiden', 'donor_ID'], size=50, legend_loc="on data", annotate_var_explained=True)
    plt.show()

    # Remove PANFR0580 NET + 3 low total cell counts
    donor_cell_count = Counter(adata.obs.donor_ID)
    low_cell_count = [k for k, v in donor_cell_count.items() if v < n_cells_cutoff]
    remove_list = ['PANFR0580'] + low_cell_count

    log(f"Removed {remove_list} for < {n_cells_cutoff} cells")
    clean_adata = tumor_adata[[x not in remove_list for x in tumor_adata.obs.donor_ID], :]
    sc.pp.highly_variable_genes(clean_adata, flavor="seurat")
    sc.pp.pca(clean_adata)
    sc.pl.pca(clean_adata, color=['leiden', 'donor_ID'], size=50, legend_loc="on data", annotate_var_explained=True)
    plt.show()
    sc.pp.neighbors(clean_adata, n_neighbors=35)
    sc.tl.tsne(clean_adata)
    sc.tl.leiden(clean_adata)
    log("Final tumor PCA and tSNE")
    sc.pl.pca(clean_adata, color='donor_ID', size=60, annotate_var_explained=True, dimensions=[(0, 1), (1, 2), (2, 3)], legend_loc="on data")
    sc.pl.tsne(clean_adata, color=['leiden', 'donor_ID'], size=50, legend_loc="on data")
    plt.show()

    log("Total donors", len(clean_adata.obs.donor_ID.unique()))
    log("Cell count")
    log(Counter(clean_adata.obs.donor_ID))

    # Demonstrate that as per paper
    # PC0 is EMT and characterized by FN1/VIM with -PC0 being Classical
    # PC1 is Classical 
    # PC2 is Basal/Classical
    
    # rank PCs
    pc_output = pd.DataFrame(clean_adata.varm['PCs'], index=clean_adata.var_names)
    pc_buf = []
    for pc in range(pc_output.shape[1]):
        pc_buf.append(pc_output.sort_values(pc, ascending=False).index.to_list())
    # pc_df is PCs by numeric ranks and not aligned with .varm so it goes into .uns
    pc_df = pd.DataFrame(pc_buf, index=[f"PC{x}" for x in range(len(pc_buf))]).T

    clean_adata.uns['PC_ranks'] = pc_df

    log("Standard deviation explained by PCs")
    plt.plot(np.power(clean_adata.uns['pca']['variance'][:20], 0.5), 'o')

    log("PC score and heatmaps")
    # PC 0: + Basal + Fibroblast like program of EMT tumor ; - Classical
    # PC 1: + PANC0504/HPAC CCLE ; - BxPC-3 type signature CCLE
    # PC 2: + Basal ; - Classical
    # PC 3: + TSC22D1/APCDD1 tumor suppressor, EFNB2/IFIT1 EMT indicator, CLDN4 PDAC marker https://www.gastrojournal.org/article/S0016-5085(01)54349-8/fulltext ;
    #       - Classical
    import itertools
    moffitt_labels = list(itertools.chain(*moffitt_pdac.values()))

    # PC0 and PC2 are more differentiated by Moffitt classes than PC1
    
    X_pca_df = pd.DataFrame(clean_adata.obsm['X_pca'], index=clean_adata.obs_names)

    fig, ax = plt.subplots(figsize=(10, 12))
    # important to scale this heatmap data!
    # X axis is cells sorted by PC score
    # Y axis is moffitt labels (basal on top, classical below)
    pc_idx = 3
    data = sc.pp.scale(clean_adata[X_pca_df.sort_values(pc_idx).index, moffitt_labels].X).T
    log(data.min(), data.max())
    im = ax.imshow(data, aspect='auto',  vmin=-1, vmax=3)
    ax.set_yticks(range(len(moffitt_labels)), moffitt_labels)
    ax.grid(None)

    fig, ax = plt.subplots(figsize=(10, 15))
    # the key is to scale this heatmap data!
    # X axis is cells sorted by PC pc_idx (starting from 0) value
    # Y axis is bottom n_bottom_top_genes and top n_bottom_top_genes genes by PC score
    # so checking PC internal heatmapping
    pc_idx = 2
    n_bottom_top_genes = 30
    bottom_top_genes = list(pc_output.sort_values(pc_idx).index[:n_bottom_top_genes]) + list(pc_output.sort_values(pc_idx).index[-n_bottom_top_genes:])
    data = sc.pp.scale(clean_adata[X_pca_df.sort_values(pc_idx).index, bottom_top_genes].X).T
    log(data.min(), data.max())
    im = ax.imshow(data, aspect='auto', vmin=-2, vmax=3)
    ax.set_yticks(range(len(bottom_top_genes)), bottom_top_genes)
    ax.grid(None)

    sc.pl.pca(clean_adata, color='donor_ID', size=60, annotate_var_explained=True, dimensions=[(0, 3), (1, 3), (2, 3)], legend_loc="on data")

    return clean_adata


def score_pc_by_gene_heatmap(adata: ad.AnnData, gene_sets: Mapping, pcs: list) -> None:
    """Plot scaled expression of the given gene sets with cells ordered by PC score.

    For each PC in `pcs`, orders cells along that component and draws a gene-by-cell heatmap,
    so a signature loading onto that component shows up as a gradient across the image.
    """
    log("PC score and heatmaps")
    # PC 0: + Basal + Fibroblast like program of EMT tumor ; - Classical
    # PC 1: + PANC0504/HPAC CCLE ; - BxPC-3 type signature CCLE
    # PC 2: + Basal ; - Classical
    # PC 3: + TSC22D1/APCDD1 tumor suppressor, EFNB2/IFIT1 EMT indicator, CLDN4 PDAC marker https://www.gastrojournal.org/article/S0016-5085(01)54349-8/fulltext ;
    #       - Classical
    import itertools
    gene_labels = [x for x in list(itertools.chain(*gene_sets.values())) if x in adata.var_names]

    # PC0 and PC2 are more differentiated by Moffitt classes than PC1
    
    X_pca_df = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names)

    
    # important to scale this heatmap data!
    # X axis is cells sorted by PC score
    # Y axis is labels
    for pc_idx in pcs:
        log(pc_idx)
        data = sc.pp.scale(adata[X_pca_df.sort_values(pc_idx).index, gene_labels].X).T
        # log(data.min(), data.max())
        _, ax = plt.subplots(figsize=(10, 15))    
        im = ax.imshow(data, aspect='auto',  vmin=-1, vmax=5)
        ax.set_yticks(range(len(gene_labels)), gene_labels)
        ax.grid(None)


if __name__ == "__main__":
    adata_raw = load_scp1644_data(check_metadata=True)
    adata_trim = preprocess_scp1644_data(adata_raw)

    log(adata_raw)
    log(adata_trim)
