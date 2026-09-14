"""Tests for the QC filtering logic in scp1644_mothership.

These cover `apply_qc`, which is the piece of the pipeline with behaviour that can be
checked without the SCP1644 data: given per-cell and per-gene QC metrics, it decides
which cells and genes survive. The thresholds encoded here are the ones the README
documents, so a change to either that is not reflected in the other will fail.

Written with AI assistance, 2026-09-14.
"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from flipcrow.scp1644_mothership import apply_qc


N_GENES = 100


def make_adata():
    """Six cells with hand-chosen QC metrics, and genes split into common and rare."""
    n_cells = 6
    X = np.ones((n_cells, N_GENES), dtype=np.float32)
    obs = pd.DataFrame(
        {
            # cell:                 0     1     2     3      4     5
            "n_genes_by_counts": [500,  300,  600,  700,  9000,  450],
            "total_counts":     [2000, 5000,  900, 3000, 20000, 1500],
            "pct_counts_mt":    [  10,    5,    5,   80,     2,    3],
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    # first half of genes are detected widely, second half are rare
    var = pd.DataFrame(
        {"n_cells_by_counts": [60] * (N_GENES // 2) + [10] * (N_GENES // 2)},
        index=[f"gene{i}" for i in range(N_GENES)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


class TestPremerge:
    def test_drops_each_failing_cell_and_keeps_the_rest(self):
        out = apply_qc(make_adata(), mode="premerge")
        # cell1 fails <400 genes, cell2 fails <1000 counts, cell3 fails >50% mito
        assert list(out.obs_names) == ["cell0", "cell4", "cell5"]

    def test_keeps_every_gene(self):
        """premerge defines no gene-level filter, so gene count must be untouched."""
        out = apply_qc(make_adata(), mode="premerge")
        assert out.n_vars == N_GENES

    def test_records_the_reason_flags_on_obs(self):
        adata = make_adata()
        apply_qc(adata, mode="premerge")
        for flag in ("u400genes", "u1000counts", "o50mtpct"):
            assert flag in adata.obs
        assert adata.obs["u400genes"].tolist() == [False, True, False, False, False, False]
        assert adata.obs["u1000counts"].tolist() == [False, False, True, False, False, False]
        assert adata.obs["o50mtpct"].tolist() == [False, False, False, True, False, False]

    def test_a_cell_failing_nothing_survives_at_the_boundaries(self):
        """400 genes and 1000 counts are kept; the filters are strict inequalities."""
        adata = make_adata()
        adata.obs.loc["cell0", ["n_genes_by_counts", "total_counts", "pct_counts_mt"]] = [400, 1000, 50]
        out = apply_qc(adata, mode="premerge")
        assert "cell0" in out.obs_names


class TestPostmerge:
    def test_drops_probable_doublets_only(self):
        out = apply_qc(make_adata(), mode="postmerge")
        # cell4 is the only cell above 8000 genes
        assert list(out.obs_names) == ["cell0", "cell1", "cell2", "cell3", "cell5"]

    def test_drops_genes_detected_in_fewer_than_fifty_cells(self):
        out = apply_qc(make_adata(), mode="postmerge")
        assert out.n_vars == N_GENES // 2
        assert all(name.startswith("gene") for name in out.var_names)
        assert "gene0" in out.var_names          # detected in 60 cells
        assert f"gene{N_GENES - 1}" not in out.var_names   # detected in 10


class TestMode:
    @pytest.mark.parametrize("mode", ["", "PREMERGE", "post_merge", "premerge "])
    def test_unknown_mode_raises(self, mode):
        with pytest.raises(ValueError, match="premerge or postmerge"):
            apply_qc(make_adata(), mode=mode)
