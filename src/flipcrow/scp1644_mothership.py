from collections import Counter
from typing import Union, Mapping
import gzip
import pathlib
import time

import torch
import pandas as pd
import scipy
import numpy as np

import scanpy as sc
import anndata as ad
import gseapy

import seaborn as sns
import matplotlib.pyplot as plt

import flipcrow
import flipcrow.paths

mps_device = torch.device("mps")

# sc.settings.verbosity = 0
import warnings
warnings.filterwarnings("ignore")

from scanpy import _version

sc.set_figure_params(figsize=(10, 10))

print(sc._version.version)

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
        'KRT7',
        'KRT8', 
        'KRT10',
        'KRT13',
        'KRT14',
        'KRT17',
        'KRT18',
        'KRT19',
    ]
}


def load_scp1644_data(check_metadata: bool = True) -> ad.AnnData:
    scp1644_small_csv = flipcrow.paths.DATA_PATH / 'SCP1644/expression/Biopsy473_RawDGE_1370cells.csv'
    scp1644_big_csv = flipcrow.paths.DATA_PATH / 'SCP1644/expression/Biopsy_RawDGE_23042cells.csv'
    scp1644_metadata_csv = flipcrow.paths.DATA_PATH / 'SCP1644/other/complete_Metadata_70170cells_scp.csv'
    
    start_load = time.time()
    adata_small = sc.read_csv(scp1644_small_csv).T
    adata_big = sc.read_csv(scp1644_big_csv).T
    adata_metadata = pd.read_csv(scp1644_metadata_csv)
    end_load = time.time()

    print(end_load - start_load)

    adata_scratch = ad.concat([adata_small, adata_big])

    # In the metadata 
    # there are 24x unique biosample IDs including PANFR0489R
    # and 23x unique cell UMI tags, which exclude PANFR0489R
    # The reason is that PANFR0489 was repeated as PANFR0489R:

    # After initial processing of fresh tissue specimens, we monitored samples closely for organoid growth. We did not passage organoids at set time intervals, as there was significant variability in the time needed to establish relatively 
    # robust growth of organoids (Figure 3D). Instead, we maintained early passage organoids until they reached relative confluence, and then passaged them at low split ratios (1:1, 1:1.5, or 1:2 dilutions) in complete organoid medium to promote 
    # continued growth. In one case, PANFR0489R, cells persisted as individuals and small organoids after initiation in complete organoid medium, but did not grow and expand cell numbers significantly. Approximately 15 weeks after initiation, 
    # we switched a portion of the surviving cells to organoid medium without A83-01 or mNoggin, and observed renewed growth of organoids under these media conditions but not of those that remained in complete organoid medium. Consequently, we 
    # expanded this sample in media without A83-01 or mNoggin, including performing early passage scRNA-seq. After several additional passages, once the organoids were robustly growing, we were able to transition this model back to complete organoid
    # medium with no apparent change in growth rate, morphology, or transcriptional state. All other serially sampled organoids were maintained and assessed in complete medium except as indicated when specific media alterations or experimental 
    # perturbations were performed. The identify of organoid models was authenticated by comparison of their inferred CNV profiles with targeted genomic sequencing and CNV profiles of matched patient tissue and with inferred CNV profiles from 
    # patient tissue and earlier passage models in the case of samples serially assessed with scRNA-seq. The identify of cell line models was authenticated by short tandem repeat (STR) analysis. Cell line and organoid cultures were routinely 
    # tested for mycoplasma contamination.

    # Was the dataset using 489R combined with the other data? Yes it looks like it was analyzed.

    if check_metadata:
        biosample_ids = Counter([x for x in adata_metadata.loc[:, 'biosample_id'] if 'Biopsy' in x])
        cell_tags = Counter(['_'.join(idx.split('_')[:2]) for idx in adata_scratch.obs_names])

        def mangle_biosample_ids(x):
            return '_'.join(x.split('_')[-1::-1][1:])

        print('='*40)
        print("Checking metadata!")
        print('='*40)
        print('Biosample IDs')
        print(biosample_ids)
        print('Cell tags')
        print(cell_tags)

        mangle_set = set(map(mangle_biosample_ids, biosample_ids.keys()))
        print("Biosample ID length", len(biosample_ids))
        print("Mangle set length", len(mangle_set))
        print("Cell tags length", len(cell_tags))
        
        print("Biosamples not in cell tags", mangle_set - cell_tags.keys())
        print("Cell tags not in biosamples", cell_tags.keys() - mangle_set)
        print()

    adata_metadata_fix = adata_metadata.set_index('NAME', drop=True).iloc[1:, :]

    adata_metadata_fix.index.name = None
    for col in adata_metadata_fix.columns:
        metadata = adata_metadata_fix.loc[adata_scratch.obs_names, col]
        adata_scratch.obs[col] = metadata

    # fix hepatocyte labeling
    adata_scratch.obs.loc[adata_scratch.obs.loc[:, 'Coarse_Cell_Annotations'] == 'Hepatocytes', 'Coarse_Cell_Annotations'] = 'Hepatocyte'
    
    # fix NaN in exvivo.treatment to be able to h5ad read/write
    del adata_scratch.obs['exvivo.treatment']

    print('='*40)
    print("Metadata stats")
    print('='*40)
    annot_count = Counter(adata_scratch.obs['Coarse_Cell_Annotations'])
    print("Coarse_Cell_Annotations count")
    print(annot_count)

    sex_count = Counter(adata_scratch.obs['sex'])
    print("sex_count")
    print(sex_count)
    
    biosample_count = Counter(adata_scratch.obs['biosample_id'])
    print("biosample_count")
    print(biosample_count)

    donor_count = Counter(adata_scratch.obs['donor_ID'])
    print(donor_count)
    
    print(f"Total biosamples {len(biosample_count)}")
    print(f"Total donors {len(donor_count)}")

    return adata_scratch


def apply_qc(adata: ad.AnnData, mode: str = "premerge", verbose: bool = False) -> ad.AnnData:

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
            print(f"{k}: {sum(v)}")
        adata.obs[k] = v
    
    for k, v in qc_gene.items():
        if verbose:
            print(f"{k}: {sum(v)}")
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
        print('mask_cell', mask_cell.shape)
        print('mask_cell present?', np.any(mask_cell))
        print('mask_gene', mask_gene.shape)
        print('mask_gene present?', np.any(mask_gene))
    
    aw = adata[~mask_cell, ~mask_gene]
    return aw


def preprocess_scp1644_data(adata: ad.AnnData) -> ad.AnnData:
    qc_buf = {}

    print("Per chunk QC")
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

    print("Concatenate")
    for k, v in qc_buf.items():
        print(f"{k} : {v.shape}")

    print()
    qc_join = ad.concat([x for x in qc_buf.values()])
    del qc_buf

    print("Merged QC")
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

    # print("Scrublet")
    # apply scrublet while we still have raw count data
    # sc.pp.scrublet(clean)

    clean.layers['trimmed_counts'] = clean.to_df().loc[clean.obs_names, :]
    sc.pp.normalize_total(clean, target_sum=10000, inplace=True)
    sc.pp.log1p(clean, copy=False)
    
    # print("Dimensionality reduction and clustering")
    # sc.pp.pca(clean)
    # sc.pp.neighbors(clean, n_neighbors=40, n_pcs=50)
    # sc.tl.leiden(clean)
    # sc.tl.umap(clean)
    # sc.tl.tsne(clean)

    # print("HVG + Rank genes groups")
    # sc.pp.highly_variable_genes(clean, flavor="seurat")
    # sc.tl.rank_genes_groups(clean, 'leiden')

    return clean


def mito_doublet_screen(adata: ad.AnnData, mt_cutoff: float = 0.15, doublet_cutoff: float = 0.1, use_pct_mt_stats=False) -> ad.AnnData:
    print("Initial tSNE")
    sc.pl.tsne(adata, color=['leiden'], legend_loc="on data", size=60)
    sc.pl.tsne(adata, color='donor_ID', size=60)
    plt.show()

    print("Calculating cluster stats")
    leiden_mt_stats = adata.obs.groupby('leiden').agg({'pct_counts_mt': ['mean', 'std', 'max']}).droplevel(level=0, axis=1).sort_values('mean')
    plt.figure()
    plt.plot(leiden_mt_stats.loc[:, 'mean'])
    plt.plot(leiden_mt_stats.loc[:, 'mean'] + 2.0*leiden_mt_stats.loc[:, 'std'])
    plt.plot(leiden_mt_stats.loc[:, 'mean'] - 2.0*leiden_mt_stats.loc[:, 'std'])
    plt.plot(leiden_mt_stats.loc[:, 'max'])
    plt.show()

    print("Running mitoscreen")
    ggb_mitoscreen = {'GGB_MITOCHONDRIAL': [x for x in adata.var_names if x.startswith('MT-') or x.startswith('MTRNR')]}
    ggb_mito_result = sc.tl.marker_gene_overlap(adata, ggb_mitoscreen, method='overlap_count', normalize='reference')
    plt.figure()
    plt.plot(ggb_mito_result.loc['GGB_MITOCHONDRIAL', :])
    print(ggb_mito_result.loc['GGB_MITOCHONDRIAL', :].sort_values(ascending=False))
    plt.show()

    for idx in ggb_mito_result.index:
        adata.obs['GGB_MITOCHONDRIAL'] = [ggb_mito_result.loc['GGB_MITOCHONDRIAL', leid_idx] for leid_idx in adata.obs['leiden']]
    
    adata_filter = adata[np.logical_and(adata.obs.GGB_MITOCHONDRIAL < mt_cutoff, adata.obs.doublet_score < doublet_cutoff), :]

    print("Preprocessing filtered dataset")
    sc.pp.pca(adata_filter)
    sc.pp.neighbors(adata_filter, n_neighbors=40, n_pcs=50)
    sc.tl.leiden(adata_filter)
    sc.tl.umap(adata_filter)
    sc.tl.tsne(adata_filter)
    sc.pp.highly_variable_genes(adata_filter, flavor="seurat")
    sc.tl.rank_genes_groups(adata_filter, 'leiden')

    print("Final tSNE")
    sc.pl.tsne(adata_filter, color=['leiden'], legend_loc="on data", size=60)
    sc.pl.tsne(adata_filter, color='donor_ID', size=60)
    plt.show()

    return adata_filter
    

def diffexpr_tests(adata: ad.AnnData): # -> Tuple[ad.AnnData, Mapping[str, KruskalType,....
    samples = [adata[adata.obs.leiden == cluster_id, adata.var.highly_variable] for cluster_id in adata.obs.leiden.unique()]
    others = [adata[adata.obs.leiden != cluster_id, adata.var.highly_variable] for cluster_id in adata.obs.leiden.unique()]

    print("Performing KW test")
    kw_map = {}
    for idx, gene in enumerate(adata.var_names[adata.var.highly_variable]):
        
        kw_map[gene] = scipy.stats.kruskal(*[x[:, gene].X.squeeze() for x in samples])
        if idx % 100 == 0:
            print(gene, kw_map[gene])

    kw_srs = pd.Series(kw_map)
    adata.var['kw_statistic'] = np.zeros(len(adata.var_names)) 
    adata.var['kw_pval'] = np.ones(len(adata.var_names)) 

    adata.var['kw_statistic'].loc[kw_map.keys()] = [x.statistic for x in kw_map.values()]
    adata.var['kw_pval'].loc[kw_map.keys()] = [x.pvalue for x in kw_map.values()]            

    # performs pairwise significance tests
    print("Performing pairwise significance tests")
    
    buf_others = []
    buf_zeros = []

    start_gene_time = time.time()
    for idx in range(len(samples)):
        # print(idx)
        start_mean_time = time.time()
        buf_others.append(scipy.stats.mannwhitneyu(samples[idx].X, others[idx].X))
        buf_zeros.append(scipy.stats.mannwhitneyu(samples[idx].X, np.zeros(samples[idx].X.shape), alternative='greater'))
        end_zeros_time = time.time()
        print(idx, (end_zeros_time - start_mean_time))    
    end_gene_time = time.time()
    print('done', (end_gene_time - start_gene_time))
    
    mwm_others_df = pd.DataFrame([x.pvalue for x in buf_others], index=adata.obs.leiden.unique(), columns=adata.var['highly_variable'][adata.var['highly_variable']].index)
    mwm_zeros_df = pd.DataFrame([x.pvalue for x in buf_zeros], index=adata.obs.leiden.unique(), columns=adata.var['highly_variable'][adata.var['highly_variable']].index)

    mwm_others_pval_cutoff = 1.0e-9
    mwm_zeros_pval_cutoff = 1.0e-5

    mwm_mean_pass = mwm_others_df < mwm_others_pval_cutoff
    mwm_zeros_pass = mwm_zeros_df < mwm_zeros_pval_cutoff

    print(mwm_mean_pass & mwm_zeros_pass)

    return adata.copy(), kw_map, mwm_others_df, mwm_zeros_df
    

def plot_cluster_gene_distr_1d(adata, samples, others, cluster_key, gene_key, plot=True):

    coi = int(cluster_key)
    samples_hist = np.histogram(samples[coi][:, gene_key].X.squeeze(), bins=20)
    others_hist = np.histogram(others[coi][:, gene_key].X.squeeze(), bins=20)
    # print(samples_hist)
    if plot:
        plt.plot(others_hist[1][:-1], others_hist[0]/others_hist[0].sum(), color='r')
        plt.plot(samples_hist[1][:-1], samples_hist[0]/samples_hist[0].sum(), color='b')
        plt.axis([0, 3.5, 0, 0.2])
    
    this_cluster = sc.get.rank_genes_groups_df(adata, str(cluster_key))
    join_cluster = this_cluster.join(adata.var.highly_variable, on='names')
    print(f"{gene_key} in {cluster_key}: {gene_key in join_cluster.names.values}")
    
    if gene_key in join_cluster.names.values:
        print(join_cluster.loc[join_cluster.names == gene_key, :])

    return this_cluster, join_cluster


# TODO:  plot co-expression by chromosomal location
def check_rank_genes_groups_df(adata: ad.AnnData, coi: Union[int, str], focus_prefix=None) -> None: 

    rgg_df = sc.get.rank_genes_groups_df(adata, coi)
    print(sc.get.rank_genes_groups_df(adata, coi).iloc[:30, :])
    if focus_prefix:
        with pd.option_context('display.min_rows', 50):
            print(rgg_df.loc[[x.startswith(focus_prefix) for x in rgg_df.names.values], :])

    print('\n'.join(sc.get.rank_genes_groups_df(adata, coi).names[:100]))


def classical_basal_analysis(adata: ad.AnnData) -> None:
    sc.tl.score_genes(adata, moffitt_pdac['Classical'], score_name='classical_score', ctrl_size=100)
    sc.tl.score_genes(adata, moffitt_pdac['Basal'], score_name='basal_score', ctrl_size=100)
    adata.obs.loc[:, 'diff_score'] = adata.obs.classical_score - adata.obs.basal_score
    plt.plot(adata.obs.classical_score, adata.obs.basal_score, 'o')


def compare_classical_basal_genes(adata: ad.AnnData, classical_gene: str, basal_gene: str) -> None:

    plt.plot(adata.obs.classical_score, adata.to_df().loc[:, classical_gene].values, '.')
    plt.plot(adata.obs.basal_score, adata.to_df().loc[:, classical_gene].values, '.')

    plt.figure(1)
    print(f'{classical_gene} (Classical)')
    print('Pearson Classical', scipy.stats.pearsonr(adata.obs.classical_score, adata.to_df().loc[:, classical_gene].values))
    print('Pearson Basal', scipy.stats.pearsonr(adata.obs.basal_score, adata.to_df().loc[:, classical_gene].values))

    print('Spearman Classical', scipy.stats.spearmanr(adata.obs.classical_score, adata.to_df().loc[:, classical_gene].values))
    print('Spearman Basal', scipy.stats.spearmanr(adata.obs.basal_score, adata.to_df().loc[:, classical_gene].values))

    print()
    plt.figure(2)
    plt.plot(adata.obs.classical_score, adata.to_df().loc[:, basal_gene].values, '.')
    plt.plot(adata.obs.basal_score, adata.to_df().loc[:, basal_gene].values, '.')

    print(f'{basal_gene} (Basal)')
    print('Pearson Classical', scipy.stats.pearsonr(adata.obs.classical_score, adata.to_df().loc[:, basal_gene].values))
    print('Pearson Basal', scipy.stats.pearsonr(adata.obs.basal_score, adata.to_df().loc[:, basal_gene].values))

    print('Spearman Classical', scipy.stats.spearmanr(adata.obs.classical_score, adata.to_df().loc[:, basal_gene].values))
    print('Spearman Basal', scipy.stats.spearmanr(adata.obs.basal_score, adata.to_df().loc[:, basal_gene].values))





# TODO: Train this like a little NN
def coarse_cell_type_annotation(adata: ad.AnnData) -> pd.DataFrame:

    normal_markers = pd.read_excel(flipcrow.paths.DATA_PATH / "markergenes" / "mmc2.xlsx", header=4, dtype=object)
    normal_markers = normal_markers.iloc[:, 2:]
    normal_markers = normal_markers.loc[:, ['Gene', 'myAUC', 'avg_diff', 'power', 'avg_logFC', 'pct.1', 'pct.2', 'cell.type']]
    normal_markers_dict = {}

    # filter datetime oddities found in original data probably caused by entering MARCH1, MARCH11 etc in dataset. This is (possibly?) auto converted to 11-Mar, 1-Mar, etc 
    for grpid, grp in normal_markers.groupby('cell.type'):
        normal_markers_dict[grpid] = [grp.loc[idx, 'Gene'] for idx in grp.index if type(grp.loc[idx, 'Gene']) == str and float(grp.loc[idx, 'power']) >= 0.6]

    #print(normal_markers_dict)
    markers_dict = normal_markers_dict.copy()

    # add tumor keratins
    markers_dict['Tumor_keratins'] = [
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
            'KRT7',
            'KRT8', 
            'KRT10',
            'KRT13',
            'KRT14',
            'KRT17',
            'KRT18',
            'KRT19',
        ]
    }

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
        print(leiden_idx, Counter(adata.obs.loc[adata.obs.leiden == str(leiden_idx), 'Coarse_Cell_Annotations']))
    
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
    tumor_adata = adata[adata.obs['Coarse_Cell_Annotations'] == 'Tumor']
    
    # demonstrate PCA isolation of PANFR0580 (NET tumor)
    print("All tumors")
    sc.pp.pca(tumor_adata)
    sc.pl.pca(tumor_adata, color=['leiden', 'donor_ID'], size=50, legend_loc="on data", annotate_var_explained=True)
    plt.show()

    # Remove PANFR0580 NET + 3 low total cell counts
    donor_cell_count = Counter(adata.obs.donor_ID)
    low_cell_count = [k for k, v in donor_cell_count.items() if v < n_cells_cutoff]
    remove_list = ['PANFR0580'] + low_cell_count

    print(f"Removed {remove_list} for < {n_cells_cutoff} cells")
    clean_adata = tumor_adata[[x not in remove_list for x in tumor_adata.obs.donor_ID], :]
    sc.pp.highly_variable_genes(clean_adata, flavor="seurat")
    sc.pp.pca(clean_adata)
    sc.pl.pca(clean_adata, color=['leiden', 'donor_ID'], size=50, legend_loc="on data", annotate_var_explained=True)
    plt.show()
    sc.pp.neighbors(clean_adata, n_neighbors=35)
    sc.tl.tsne(clean_adata)
    sc.tl.leiden(clean_adata)
    print("Final tumor PCA and tSNE")
    sc.pl.pca(clean_adata, color='donor_ID', size=60, annotate_var_explained=True, dimensions=[(0, 1), (1, 2), (2, 3)], legend_loc="on data")
    sc.pl.tsne(clean_adata, color=['leiden', 'donor_ID'], size=50, legend_loc="on data")
    plt.show()

    print("Total donors", len(clean_adata.obs.donor_ID.unique()))
    print("Cell count")
    print(Counter(clean_adata.obs.donor_ID))

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

    print("Standard deviation explained by PCs")
    plt.plot(np.power(clean_adata.uns['pca']['variance'][:20], 0.5), 'o')

    print("PC score and heatmaps")
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
    print(data.min(), data.max())
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
    print(data.min(), data.max())
    im = ax.imshow(data, aspect='auto', vmin=-2, vmax=3)
    ax.set_yticks(range(len(bottom_top_genes)), bottom_top_genes)
    ax.grid(None)

    sc.pl.pca(clean_adata, color='donor_ID', size=60, annotate_var_explained=True, dimensions=[(0, 3), (1, 3), (2, 3)], legend_loc="on data")

    return clean_adata


def score_pc_by_gene_heatmap(adata: ad.AnnData, gene_sets: Mapping, pcs: list) -> None:
    print("PC score and heatmaps")
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
        print(pc_idx)
        data = sc.pp.scale(adata[X_pca_df.sort_values(pc_idx).index, gene_labels].X).T
        # print(data.min(), data.max())
        _, ax = plt.subplots(figsize=(10, 15))    
        im = ax.imshow(data, aspect='auto',  vmin=-1, vmax=5)
        ax.set_yticks(range(len(gene_labels)), gene_labels)
        ax.grid(None)


if __name__ == "__main__":
    adata_raw = load_scp1644_data(check_metadata=True)
    adata_trim = preprocess_scp1644_data(adata_raw)

    print(adata_raw)
    print(adata_trim)

    cluster_key = '10'
    gene_key = 'A1CF'

    test = plot_cluster_gene_distr_1d(cluster_key, gene_key)
    print()
    print(test)    