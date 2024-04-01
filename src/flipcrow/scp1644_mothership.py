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
        mask_gene = np.array([False for v in adata.var_names]) # np.zeros(adata.var_names.shape)
    
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

    # see if we can replace with a copy up above
    _, varinfo = sc.pp.calculate_qc_metrics(
        adata_qc_join, 
        qc_vars=["mt", "ribo", "hb"], 
        inplace=False, 
        percent_top=[20], 
        log1p=False,
    ) 

    # Remove genes present in fewer than 50 cells
    adata_biopsy_prenorm = adata_qc_join[:, varinfo.n_cells_by_counts >= 50]

    # apply scrublet while we still have raw count data
    sc.pp.scrublet(adata_biopsy_prenorm)

    return adata_biopsy_prenorm


def norm_log_xform_scp1644_data(input: ad.AnnData) -> None:
    tmp = input.copy()
    sc.pp.normalize_total(tmp, target_sum=10000, inplace=True)
    return sc.pp.log1p(tmp, copy=True)
    


if __name__ == "__main__":
    scp1644_raw = load_scp1644_data(check_metadata=True)
    scp1644_trim = trim_scp1644_data(scp1644_raw)
    scp1644_log = norm_log_xform_scp1644_data(scp1644_trim)

    print(scp1644_raw)
    print(scp1644_trim)
    print(scp1644_log)