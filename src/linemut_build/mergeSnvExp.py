from anndata import AnnData

def group_meanexp_pca(
    adata: AnnData,
    group_key: str = "group",
    layer: str | None = None,
    use_raw: bool = False,
    do_normalize_log: bool = True,
    do_scale: bool = True,
    n_comps: int = 5,
):
    import numpy as np
    import scipy.sparse as sp
    import pandas as pd
    import scanpy as sc

    if group_key not in adata.obs:
        raise KeyError(f"'{group_key}' not found in adata.obs")

    if layer is not None:
        X = adata.layers[layer]
    elif use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None")
        X = adata.raw.X
    else:
        X = adata.X

    groups = adata.obs[group_key].astype("category")
    group_names = groups.cat.categories.tolist()

    mean_rows = []
    for g in group_names:
        idx = np.asarray(groups == g)
        Xg = X[idx]

        if sp.issparse(Xg):
            mu = np.asarray(Xg.mean(axis=0)).ravel()
        else:
            mu = Xg.mean(axis=0)

        mean_rows.append(mu)

    X_mean = np.vstack(mean_rows)

    if use_raw:
        var = adata.raw.var.copy()
        var_names = adata.raw.var_names.copy()
    else:
        var = adata.var.copy()
        var_names = adata.var_names.copy()

    obs = pd.DataFrame({group_key: group_names}, index=group_names)

    group_adata = AnnData(
        X=X_mean,
        obs=obs,
        var=var,
    )
    group_adata.var_names = var_names

    if do_normalize_log:
        sc.pp.normalize_total(group_adata, target_sum=1e4)
        sc.pp.log1p(group_adata)

    if do_scale:
        sc.pp.scale(group_adata, zero_center=True, max_value=10)

    sc.tl.pca(group_adata, n_comps=n_comps, svd_solver="arpack")

    return group_adata


def align_two_adatas_by_obs(adata_a, adata_b, how="inner", base="a"):
    """
    Align two AnnData objects by their `obs_names` and return the aligned copy (in the same order).
    how: "inner"` takes the intersection (recommended, convenient for joint graphs).
    base: "a" or "b"` determines which object the final order follows.
    """

    import numpy as np
    import scipy.sparse as sp
    import scanpy as sc

    if adata_a.obs_names.has_duplicates:
        raise ValueError("adata_a.obs_names has duplicate names.")
    if adata_b.obs_names.has_duplicates:
        raise ValueError("adata_b.obs_names has duplicate names.")

    a_names = adata_a.obs_names
    b_names = adata_b.obs_names

    common = a_names.intersection(b_names)
    if len(common) == 0:
        raise ValueError("The two objects do not share any obs_names.")

    if base == "a":
        order = [x for x in a_names if x in common]
    elif base == "b":
        order = [x for x in b_names if x in common]
    else:
        raise ValueError("base must be 'a' or 'b'.")

    a2 = adata_a[order].copy()
    b2 = adata_b[order].copy()

    return a2, b2


def build_joint_neighbors_graph_fusion(
    adata,
    rna_rep="X_rna_pca",
    snv_rep="X_snv_pca",
    rna_n_pcs=5,
    snv_n_pcs=4,
    n_neighbors=5,
    metric="euclidean",
    w_rna=0.5,
    adaptive_wnn_like=False,
    key_added="joint",
    store_sym_connectivities=True,
    sym_connectivities_key=None,
    zero_diagonal=True,
):
    """
    Build two neighbor graphs (RNA + SNV) from embeddings in adata.obsm, fuse them into a joint graph,
    and optionally store a symmetrized, diagonal-zeroed connectivity matrix for downstream graph metrics.

    Notes
    -----
    - The fused graphs are stored in adata.obsp:
        f"{key_added}_connectivities", f"{key_added}_distances"
    - If store_sym_connectivities=True, a symmetrized version is stored in adata.obsp under:
        sym_connectivities_key (default: f"{key_added}_connectivities_sym")
    - AnnData .layers is typically reserved for expression-like matrices (n_obs x n_vars). For graph
      adjacency matrices, adata.obsp is the standard location.
    """
    import numpy as np
    import scipy.sparse as sp
    import scanpy as sc

    if rna_rep not in adata.obsm:
        raise KeyError(f"{rna_rep} not found in adata.obsm. Available: {list(adata.obsm.keys())}")
    if snv_rep not in adata.obsm:
        raise KeyError(f"{snv_rep} not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    Xr = np.asarray(adata.obsm[rna_rep])
    Xs = np.asarray(adata.obsm[snv_rep])

    if Xr.shape[0] != adata.n_obs or Xs.shape[0] != adata.n_obs:
        raise ValueError("Both embeddings must have n_cells rows equal to adata.n_obs.")

    if rna_n_pcs <= 0 or snv_n_pcs <= 0:
        raise ValueError("rna_n_pcs and snv_n_pcs must be positive integers.")

    if rna_n_pcs > Xr.shape[1]:
        raise ValueError(f"rna_n_pcs={rna_n_pcs} exceeds {rna_rep} dims={Xr.shape[1]}")
    if snv_n_pcs > Xs.shape[1]:
        raise ValueError(f"snv_n_pcs={snv_n_pcs} exceeds {snv_rep} dims={Xs.shape[1]}")

    if not (0.0 <= w_rna <= 1.0):
        raise ValueError("w_rna must be within [0, 1].")

    rna_sub_key = f"{key_added}_rna_rep"
    snv_sub_key = f"{key_added}_snv_rep"
    adata.obsm[rna_sub_key] = Xr[:, :rna_n_pcs].copy()
    adata.obsm[snv_sub_key] = Xs[:, :snv_n_pcs].copy()

    rna_key = f"{key_added}_rna"
    snv_key = f"{key_added}_snv"

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=rna_sub_key,
        metric=metric,
        key_added=rna_key,
        method="umap",
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=snv_sub_key,
        metric=metric,
        key_added=snv_key,
        method="umap",
    )

    rna_conn = adata.obsp[f"{rna_key}_connectivities"].tocsr()
    snv_conn = adata.obsp[f"{snv_key}_connectivities"].tocsr()
    rna_dist = adata.obsp[f"{rna_key}_distances"].tocsr()
    snv_dist = adata.obsp[f"{snv_key}_distances"].tocsr()

    weights = None

    if not adaptive_wnn_like:
        joint_conn = (w_rna * rna_conn) + ((1.0 - w_rna) * snv_conn)
        joint_dist = (w_rna * rna_dist) + ((1.0 - w_rna) * snv_dist)
    else:
        def row_mean_nonzero(A: sp.csr_matrix) -> np.ndarray:
            nnz = A.getnnz(axis=1)
            s = np.asarray(A.sum(axis=1)).ravel()
            return s / np.maximum(nnz, 1)

        r_strength = row_mean_nonzero(rna_conn)
        s_strength = row_mean_nonzero(snv_conn)

        m = np.maximum(r_strength, s_strength)
        er = np.exp(r_strength - m)
        es = np.exp(s_strength - m)
        w = er / (er + es)
        weights = w

        Ww = sp.diags(w, format="csr")
        IWw = sp.diags(1.0 - w, format="csr")

        joint_conn = (Ww @ rna_conn) + (IWw @ snv_conn)
        joint_dist = (Ww @ rna_dist) + (IWw @ snv_dist)

    joint_conn = joint_conn.tocsr()
    joint_dist = joint_dist.tocsr()
    joint_conn.eliminate_zeros()
    joint_dist.eliminate_zeros()

    adata.obsp[f"{key_added}_connectivities"] = joint_conn
    adata.obsp[f"{key_added}_distances"] = joint_dist

    if store_sym_connectivities:
        if sym_connectivities_key is None:
            sym_connectivities_key = f"{key_added}_connectivities_sym"

        sym_conn = (joint_conn + joint_conn.T) * 0.5
        sym_conn = sym_conn.tocsr()
        if zero_diagonal:
            sym_conn.setdiag(0.0)
            sym_conn.eliminate_zeros()

        adata.obsp[sym_connectivities_key] = sym_conn

    adata.uns[key_added] = {
        "connectivities_key": f"{key_added}_connectivities",
        "distances_key": f"{key_added}_distances",
        "params": {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "method": "graph_fusion",
            "rna_rep": rna_rep,
            "snv_rep": snv_rep,
            "rna_n_pcs": rna_n_pcs,
            "snv_n_pcs": snv_n_pcs,
            "w_rna": w_rna,
            "adaptive_wnn_like": adaptive_wnn_like,
            "store_sym_connectivities": bool(store_sym_connectivities),
            "sym_connectivities_key": sym_connectivities_key if store_sym_connectivities else None,
            "zero_diagonal": bool(zero_diagonal),
        },
    }

    return weights
