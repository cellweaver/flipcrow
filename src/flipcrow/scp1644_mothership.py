from collections import Counter
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

print(sc._version.version)

moffitt_pdac = {
    'BASAL': [
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
    'CLASSICAL': [
        'BTNL8',
        'FAM3D',
        'PRR15L', # 'ATAD4',
        'AGR3',
        'CTSE',
        # 'TMEM238L', # 'LOC400573', not present in genes
        'LYZ',
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



def load_scp1644_data(check_metadata: bool = True) -> ad.AnnData:
    scp1644_small_csv = flipcrow.paths.DATA_PATH / 'SCP1644/expression/Biopsy473_RawDGE_1370cells.csv'
    scp1644_big_csv = flipcrow.paths.DATA_PATH / 'SCP1644/expression/Biopsy_RawDGE_23042cells.csv'
    scp1644_metadata_csv = flipcrow.paths.DATA_PATH / 'SCP1644/other/complete_Metadata_70170cells_scp.csv'
    
    start_load = time.time()
    scp1644_small = sc.read_csv(scp1644_small_csv).T
    scp1644_big = sc.read_csv(scp1644_big_csv).T
    scp1644_metadata = pd.read_csv(scp1644_metadata_csv)
    end_load = time.time()

    print(end_load - start_load)

    scp1644_scratch = ad.concat([scp1644_small, scp1644_big])

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
        biosample_ids = Counter([x for x in scp1644_metadata.loc[:, 'biosample_id'] if 'Biopsy' in x])
        cell_tags = Counter(['_'.join(idx.split('_')[:2]) for idx in scp1644_scratch.obs_names])

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

    scp1644_metadata_fix = scp1644_metadata.set_index('NAME', drop=True).iloc[1:, :]

    scp1644_metadata_fix.index.name = None
    for col in scp1644_metadata_fix.columns:
        metadata = scp1644_metadata_fix.loc[scp1644_scratch.obs_names, col]
        scp1644_scratch.obs[col] = metadata

    # fix hepatocyte labeling
    scp1644_scratch.obs.loc[scp1644_scratch.obs.loc[:, 'Coarse_Cell_Annotations'] == 'Hepatocytes', 'Coarse_Cell_Annotations'] = 'Hepatocyte'
    
    print('='*40)
    print("Metadata stats")
    print('='*40)
    annot_count = Counter(scp1644_scratch.obs['Coarse_Cell_Annotations'])
    print("Coarse_Cell_Annotations count")
    print(annot_count)

    sex_count = Counter(scp1644_scratch.obs['sex'])
    print("sex_count")
    print(sex_count)
    
    biosample_count = Counter(scp1644_scratch.obs['biosample_id'])
    print("biosample_count")
    print(biosample_count)

    donor_count = Counter(scp1644_scratch.obs['donor_ID'])
    print(donor_count)
    
    print(f"Total biosamples {len(biosample_count)}")
    print(f"Total donors {len(donor_count)}")

    return scp1644_scratch


def apply_qc(adata: ad.AnnData, mode: str = "premerge", verbose: bool = False) -> ad.AnnData:

    # # fewer than 400 genes - low quality cell
    # # more than 8000 genes - probable doublet
    # # fewer than 1000 UMIs / total counts / transcripts per cell
    # # fewer than 50 cells detected with gene expressed
    # # more than 50% mitochondrial counts
    if mode == "premerge":
        qc_cell = {
            'u400genes': adata.obs.n_genes_by_counts < 400,
            # 'o8000genes': adata.obs.n_genes_by_counts > 8000,
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
        mask_cell = np.array([False for v in adata.obs_names]) # np.zeros(adata.obs_names.shape)

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


def trim_scp1644_data(input: ad.AnnData) -> ad.AnnData:
    adata_buf = {}
    adata_qc_buf = {}
    for id in list(set(input.obs['biosample_id'])):
        # print(id)
        adata = input[input.obs['biosample_id']==id, :]

        # mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith(("MT-", "MTRNR"))
        # ribosomal genes
        adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
        # hemoglobin genes.
        adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))
        
        # do QC calc for real
        sc.pp.calculate_qc_metrics(
            adata, 
            qc_vars=["mt", "ribo", "hb"], 
            inplace=True, 
            percent_top=[20], 
            log1p=False,
        ) 

        adata_buf[id] = adata
        adata_qc_buf[id] = apply_qc(adata)

    for k, v in adata_qc_buf.items():
        print(f"{k} : {v.shape}")

    print()
    adata_qc_join = ad.concat([x for x in adata_qc_buf.values()])

    # mitochondrial genes
    adata_qc_join.var['mt'] = adata_qc_join.var_names.str.startswith(("MT-", "MTRNR"))
    # ribosomal genes
    adata_qc_join.var["ribo"] = adata_qc_join.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    adata_qc_join.var["hb"] = adata_qc_join.var_names.str.contains(("^HB[^(P)]"))

    sc.pp.calculate_qc_metrics(
        adata_qc_join, 
        qc_vars=["mt", "ribo", "hb"], 
        inplace=True, 
        percent_top=[20], 
        log1p=False,
    ) 

    # Remove cells with more than 8000 genes and genes present in fewer than 50 cells
    adata_biopsy_prenorm = apply_qc(adata_qc_join, mode="postmerge")
    # adata_biopsy_prenorm = adata_qc_join[:, varinfo.n_cells_by_counts >= 50]

    # apply scrublet while we still have raw count data
    sc.pp.scrublet(adata_biopsy_prenorm)

    return adata_biopsy_prenorm


def norm_log_xform_scp1644_data(input: ad.AnnData) -> None:
    tmp = input.copy()
    sc.pp.normalize_total(tmp, target_sum=10000, inplace=True)
    return sc.pp.log1p(tmp, copy=True)
    

def kruskal_wallis_tests(input: ad.AnnData): # -> Tuple[ad.AnnData, Mapping[str, KruskalType,....
    samples = [input[input.obs.leiden == cluster_id, input.var.highly_variable] for cluster_id in input.obs.leiden.unique()]
    others = [input[input.obs.leiden != cluster_id, input.var.highly_variable] for cluster_id in input.obs.leiden.unique()]

    print("Performing KW test")
    kw_map = {}
    for idx, gene in enumerate(input.var_names[input.var.highly_variable]):
        
        kw_map[gene] = scipy.stats.kruskal(*[x[:, gene].X.squeeze() for x in samples])
        if idx % 100 == 0:
            print(gene, kw_map[gene])

    print(len(kw_map))
    kw_srs = pd.Series(kw_map)
    input.var['kw_statistic'] = np.zeros(len(input.var_names)) 
    input.var['kw_pval'] = np.ones(len(input.var_names)) 

    input.var['kw_statistic'].loc[kw_map.keys()] = [x.statistic for x in kw_map.values()]
    input.var['kw_pval'].loc[kw_map.keys()] = [x.pvalue for x in kw_map.values()]            

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
    
    mwm_others_df = pd.DataFrame([x.pvalue for x in buf_others], index=input.obs.leiden.unique(), columns=input.var['highly_variable'][input.var['highly_variable']].index)
    mwm_zeros_df = pd.DataFrame([x.pvalue for x in buf_zeros], index=input.obs.leiden.unique(), columns=input.var['highly_variable'][input.var['highly_variable']].index)

    mwm_others_pval_cutoff = 1.0e-9
    mwm_zeros_pval_cutoff = 1.0e-5

    mwm_mean_pass = mwm_others_df < mwm_others_pval_cutoff
    mwm_zeros_pass = mwm_zeros_df < mwm_zeros_pval_cutoff

    print(mwm_mean_pass & mwm_zeros_pass)

    return input.copy(), kw_map, mwm_others_df, mwm_zeros_df



    def plot_cluster_gene_distr_1d(samples, others, cluster_key, gene_key, plot=True):

        coi = int(cluster_key)
        samples_hist = np.histogram(samples[coi][:, gene_key].X.squeeze(), bins=20)
        others_hist = np.histogram(others[coi][:, gene_key].X.squeeze(), bins=20)
        # print(samples_hist)
        if plot:
            plt.plot(others_hist[1][:-1], others_hist[0]/others_hist[0].sum(), color='r')
            plt.plot(samples_hist[1][:-1], samples_hist[0]/samples_hist[0].sum(), color='b')
            plt.axis([0, 3.5, 0, 0.2])
        
        this_cluster = sc.get.rank_genes_groups_df(input, str(cluster_key))
        join_cluster = this_cluster.join(input.var.highly_variable, on='names')
        print(f"{gene_key} in {cluster_key}: {gene_key in join_cluster.names.values}")
        
        if gene_key in chomp.names.values:
            print(chomp.loc[chomp.names == gene_key, :])

        return this_cluster

# TODO:  plot co-expression by chromosomal location


def check_rank_genes_groups_df(input: ad.AnnData, coi: Union[int, str], focus_prefix=None) -> None: 
    coi = '34'
    rgg_df = sc.get.rank_genes_groups_df(input, coi)
    print(sc.get.rank_genes_groups_df(input, coi).iloc[:30, :])
    if focus_prefix:
        with pd.option_context('display.min_rows', 50):
            print(rgg_df.loc[[x.startswith(focus_prefix) for x in rgg_df.names.values], :])

    print('\n'.join(sc.get.rank_genes_groups_df(input, coi).names[:100]))


def basal_classical_analysis(input: ad.AnnData) -> None:
    sc.tl.score_genes(input, moffitt_pdac['CLASSICAL'], score_name='classical_score', ctrl_size=100)
    sc.tl.score_genes(input, moffitt_pdac['BASAL'], score_name='basal_score', ctrl_size=100)
    input.obs.loc[:, 'diff_score'] = input.obs.classical_score - input.obs.basal_score
    plt.plot(input.obs.classical_score, input.obs.basal_score, 'o')


def compare_classical_basal_genes(input: ad.AnnData, classical_gene: str, basal_gene: str) -> None:

    plt.plot(input.obs.classical_score, input.to_df().loc[:, classical_gene].values, '.')
    plt.plot(input.obs.basal_score, input.to_df().loc[:, classical_gene].values, '.')


    plt.figure(1)
    print(f'{classical_gene} (Classical)')
    print('Pearson Classical', scipy.stats.pearsonr(input.obs.classical_score, input.to_df().loc[:, classical_gene].values))
    print('Pearson Basal', scipy.stats.pearsonr(input.obs.basal_score, input.to_df().loc[:, classical_gene].values))

    print('Spearman Classical', scipy.stats.spearmanr(input.obs.classical_score, input.to_df().loc[:, classical_gene].values))
    print('Spearman Basal', scipy.stats.spearmanr(input.obs.basal_score, input.to_df().loc[:, classical_gene].values))

    print()
    plt.figure(2)
    plt.plot(input.obs.classical_score, input.to_df().loc[:, basal_gene].values, '.')
    plt.plot(input.obs.basal_score, input.to_df().loc[:, basal_gene].values, '.')

    print(f'{basal_gene} (Basal)')
    print('Pearson Classical', scipy.stats.pearsonr(input.obs.classical_score, input.to_df().loc[:, basal_gene].values))
    print('Pearson Basal', scipy.stats.pearsonr(input.obs.basal_score, input.to_df().loc[:, basal_gene].values))

    print('Spearman Classical', scipy.stats.spearmanr(input.obs.classical_score, input.to_df().loc[:, basal_gene].values))
    print('Spearman Basal', scipy.stats.spearmanr(input.obs.basal_score, input.to_df().loc[:, basal_gene].values))


def tumor_cleanup(input: ad.AnnData) -> None:
    sc.pl.pca(input, color=['leiden', 'donor_ID'], size=50)

    clean_input = input[[x not in ['PANFR0580', 'PANFR0588', 'PANFR0543'] for x in input.obs.donor_ID], :]
    sc.pp.highly_variable_genes(clean_input)
    sc.pp.pca(clean_input)
    sc.pl.pca(clean_input, color=['leiden', 'donor_ID'], size=50)
    sc.pp.neighbors(clean_input, n_neighbors=35)
    sc.tl.tsne(clean_input)
    sc.tl.leiden(clean_input, resolution=3.)
    sc.pl.tsne(clean_input, color=['leiden', 'donor_ID'], size=100, legend_loc="on data")

    print(len(clean_input.obs.donor_ID.unique()))
    print(Counter(clean_input.obs.donor_ID))
    with pd.option_context('display.min_rows', 50):
        print(pd.DataFrame(clean_input.varm['PCs'][:, 0:3], index=clean_input.var_names).sort_values(0))
    sc.pl.tsne(clean_input, color='pct_counts_mt')

    mttrim_input = clean_input[clean_input.obs.pct_counts_mt <= 20, :]
    sc.pp.highly_variable_genes(mttrim_input)
    sc.pp.pca(mttrim_input)
    sc.pp.neighbors(mttrim_input)
    sc.tl.tsne(mttrim_input)
    sc.tl.leiden(mttrim_input, resolution=2.)
    sc.pl.pca(mttrim_input, color=['leiden', 'donor_ID'], size=100, legend_loc='on data')
    sc.pl.tsne(mttrim_input, color=['leiden', 'donor_ID'], size=100, legend_loc='on data')

    print(Counter(mttrim_input.obs.donor_ID))
    sc.tl.umap(mttrim_input)

    # PC0 is EMT :) :)
    # PC1 is Classical :)
    # PC2 is Basal :)

    with pd.option_context('display.min_rows', 50):
        print(pd.DataFrame(mttrim_input.varm['PCs'][:, 0:3], index=mttrim_input.var_names).sort_values(2))    

    for x in range(3):
        print('\n'.join(pd.DataFrame(mttrim_input.varm['PCs'], index=mttrim_input.var_names).sort_values(x, ascending=False).index[:50]))

    sc.tl.score_genes(mttrim_input, moffitt_pdac['CLASSICAL'], score_name='classical_score', ctrl_size=100)
    sc.tl.score_genes(mttrim_input, moffitt_pdac['BASAL'], score_name='basal_score', ctrl_size=100)
    mttrim_input.obs.loc[:, 'diff_score'] = mttrim_input.obs.classical_score - mttrim_input.obs.basal_score

    sc.pl.heatmap(mttrim_input, moffitt_pdac, 'diff_score', swap_axes=True, vmin=-1.5, vmax=3)

    return 


if __name__ == "__main__":
    scp1644_raw = load_scp1644_data(check_metadata=True)
    scp1644_trim = trim_scp1644_data(scp1644_raw)
    scp1644_log = norm_log_xform_scp1644_data(scp1644_trim)

    print(scp1644_raw)
    print(scp1644_trim)
    print(scp1644_log)

    cluster_key = '10'
    gene_key = 'A1CF'

    test = plot_cluster_gene_distr_1d(cluster_key, gene_key)
    print()
    print(test)    