import pandas as pd

def revise_snv_ratio(
    adata,
    layer_name="original_ratio",
    low=0.1,
    high=0.9,
    mid=0.5,
    copy=True,
    dtype="float32",
):
    """
    1) Copy adata.X into a new layer (default: "original_ratio").
    2) Discretize adata.X in-place (or on a copy) using configurable thresholds:
         - values < low   -> 0
         - values > high  -> 1
         - otherwise      -> mid
       Note: values exactly equal to low/high fall into the "otherwise" bucket.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    layer_name : str
        Name of the layer that will store the original (pre-discretization) X.
    low : float
        Lower threshold.
    high : float
        Upper threshold.
    mid : float
        Value assigned to entries within [low, high] (inclusive).
    copy : bool
        If True, operate on a copy and return it; if False, modify the input adata in-place.
    dtype : str or numpy dtype
        Output dtype for dense X and the stored original layer (dense). Sparse keeps its dtype.

    Returns
    -------
    AnnData
        AnnData containing layers[layer_name] = original X, and modified X.
    """
    import numpy as np
    from scipy import sparse

    if low > high:
        raise ValueError(f"`low` must be <= `high`. Got low={low}, high={high}.")

    ad = adata.copy() if copy else adata

    X = ad.X

    # Store the original X into a new layer
    if sparse.issparse(X):
        ad.layers[layer_name] = X.copy()
        M = X.tocsr(copy=True)
        data = M.data

        mask_low = data < low
        mask_high = data > high
        mask_mid = (~mask_low) & (~mask_high)

        data[mask_low] = 0.0
        data[mask_high] = 1.0
        data[mask_mid] = float(mid)

        M.eliminate_zeros()
        ad.X = M
    else:
        ad.layers[layer_name] = np.array(X, copy=True).astype(dtype, copy=False)

        arr = np.array(X, copy=True)
        out = np.full(arr.shape, float(mid), dtype=dtype)
        out[arr < low] = 0.0
        out[arr > high] = 1.0

        ad.X = out

    return ad


def filter_adata_by_mutation_nnz(
    adata,
    min_nnz: int = 2,
    binarize_nonzero_to_one: bool = False,
    to_csr: bool = True,
    verbose: bool = True,
):
    """
    Filter AnnData columns (mutations) by the number of non-zero entries per column,
    and optionally binarize all non-zero values to 1.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object (cells x mutations) stored in adata.X.
    min_nnz : int, default=2
        Keep columns that have at least `min_nnz` non-zero entries.
    binarize_nonzero_to_one : bool, default=False
        If True, convert any non-zero value (e.g., 0.5 or 1) to 1.
        Note: for sparse matrices, this sets ALL stored data entries to 1.0,
        matching the original behavior (including the rare case of explicitly stored zeros).
    to_csr : bool, default=True
        If True and the resulting matrix is sparse, convert it to CSR format.
    verbose : bool, default=True
        If True, print basic stats and how many columns are kept.

    Returns
    -------
    adata_filt : AnnData
        A filtered copy of the input AnnData.
    """
    import numpy as np
    import scipy.sparse as sp

    X = adata.X
    n_cells, n_muts = adata.n_obs, adata.n_vars

    # 1) Compute per-column nnz on the original matrix (no full copy)
    if sp.issparse(X):
        # Fast paths for common formats
        if sp.isspmatrix_csr(X):
            # Count occurrences of each column index in CSR indices
            nnz_per_gene = np.bincount(X.indices, minlength=n_muts)
        elif sp.isspmatrix_csc(X):
            # In CSC, indptr encodes nnz per column
            nnz_per_gene = np.diff(X.indptr)
        else:
            nnz_per_gene = np.asarray(X.getnnz(axis=0)).ravel()

        nonzero_total = int(X.nnz)
    else:
        nnz_per_gene = np.count_nonzero(X, axis=0)
        nonzero_total = int(np.count_nonzero(X))

    keep_mask = nnz_per_gene >= int(min_nnz)
    n_keep = int(keep_mask.sum())

    if verbose:
        total_entries = n_cells * n_muts
        ratio = nonzero_total / total_entries if total_entries else float("nan")
        print(f"Input shape: {n_cells} cells x {n_muts} mutations")
        print(f"Non-zero ratio: {ratio:.6g}")
        print(f"Number of kept mutations (nnz>={min_nnz}): {n_keep}")

        # Same optional diagnostics as your original implementation
        if sp.issparse(X):
            vals = X.data
            count_1 = int(np.count_nonzero(np.isclose(vals, 1.0)))
            count_05 = int(np.count_nonzero(np.isclose(vals, 0.5)))
            print(f"Original sparse data counts: 1.0 -> {count_1}, 0.5 -> {count_05}")

    # 2) Slice AnnData once (avoid double slicing X)
    adata_filt = adata[:, keep_mask].copy()

    # 3) Optional binarization on the filtered submatrix only
    if binarize_nonzero_to_one:
        if sp.issparse(adata_filt.X):
            # Match original behavior: set ALL stored data entries to 1.0
            adata_filt.X.data[:] = 1.0
        else:
            adata_filt.X = (adata_filt.X != 0).astype(np.float32)

    # 4) Convert to CSR only if requested and needed
    if to_csr and sp.issparse(adata_filt.X) and (not sp.isspmatrix_csr(adata_filt.X)):
        adata_filt.X = adata_filt.X.tocsr()

    return adata_filt


def get_pcs(adata):
    """Return (PC matrix, cell name list).

    First try adata.obsm['X_pca'] (shape: n_cells x n_pcs).
    If it does not exist, try adata.uns['pca']['X'].
    If neither is available, raise a KeyError.
    """
    import numpy as np

    names = adata.obs_names.tolist()
    if hasattr(adata, 'obsm') and 'X_pca' in adata.obsm:
        X = adata.obsm['X_pca']
        return np.asarray(X), names
    if hasattr(adata, 'uns') and 'pca' in adata.uns and isinstance(adata.uns['pca'], dict) and 'X' in adata.uns['pca']:
        X = adata.uns['pca']['X']
        return np.asarray(X), names
    raise KeyError("PCA not found, please provide adata.obsm['X_pca'] or adata.uns['pca']['X']")


def plot_snv_pca_by_group(
    adata,
    pcx=1,
    pcy=2,
    size=6,
    alpha=0.7,
    cmap="viridis",
    group_by=None,
    group_colors=None,
    legend_loc="best",
    savepath=None,       
    dpi=300,              
    bbox_inches="tight",  
):
    """
    Scatter plot of two PCs colored by a specified obs column (or uncolored if group_by=None).

    Parameters
    ----------
    adata : AnnData
    pcx, pcy : int
        1-based indices (PC1=1).
    size : float
        Marker size.
    alpha : float
        Marker alpha.
    cmap : str
        Colormap name for continuous coloring.
    group_by : str | None
        Column name in adata.obs used for coloring. If None, plot points without coloring.
    group_colors : dict | list | tuple | None
        For categorical group_by:
        - None: use matplotlib default palette.
        - dict: mapping {category: color}.
        - list/tuple: colors aligned with category order.
        For continuous group_by, this is ignored (use cmap), unless you pass group_colors,
        in which case we force categorical coloring.
    legend_loc : str
        Legend location for categorical coloring.
    savepath : str | None
        If provided, save the figure to this path (supports .png/.pdf/.svg/.jpg etc).
    dpi : int
        DPI used when saving raster formats (e.g., png/jpg).
    bbox_inches : str
        bbox_inches argument passed to fig.savefig.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    pcs, names = get_pcs(adata)
    x = pcs[:, pcx - 1]
    y = pcs[:, pcy - 1]

    fig, ax = plt.subplots(figsize=(6, 5))

    # No coloring
    if group_by is None:
        ax.scatter(x, y, s=size, alpha=alpha)
        ax.set_xlabel(f"PC{pcx}")
        ax.set_ylabel(f"PC{pcy}")
        ax.set_title("PCA projection")
        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.show()
        plt.close(fig)
        return

    if group_by not in adata.obs:
        plt.close(fig)
        raise KeyError(f"`group_by='{group_by}'` not found in `adata.obs`.")

    g = adata.obs[group_by]

    is_cat = (
        pd.api.types.is_categorical_dtype(g)
        or pd.api.types.is_object_dtype(g)
        or pd.api.types.is_string_dtype(g)
    )
    is_continuous = (not is_cat) and pd.api.types.is_numeric_dtype(g) and (group_colors is None)

    if is_continuous:
        vals = np.asarray(g, dtype=float)
        sca = ax.scatter(x, y, c=vals, s=size, alpha=alpha, cmap=cmap)
        cbar = fig.colorbar(sca, ax=ax)
        cbar.set_label(group_by)
    else:
        # stable category order
        try:
            g_cat = g.astype("category")
            cats = list(g_cat.cat.categories)
            g_str = g_cat.astype(str).values
        except Exception:
            cats = sorted(set(map(str, g)))
            g_str = np.asarray(list(map(str, g)))

        # default palette (extend if needed)
        base = plt.rcParams.get("axes.prop_cycle", None)
        base_colors = base.by_key().get("color", []) if base is not None else []
        if len(base_colors) < len(cats):
            tab = plt.get_cmap("tab20")
            extra = [tab(i % tab.N) for i in range(len(cats))]
            palette = (base_colors + extra)[: len(cats)]
        else:
            palette = base_colors[: len(cats)]

        # build default color mapping
        color_map = {c: palette[i] for i, c in enumerate(cats)}

        # apply user-provided colors
        if group_colors is not None:
            if isinstance(group_colors, dict):
                # Allow keys to be int/float/str; match via string form first, then direct
                for c in cats:
                    if c in group_colors:
                        color_map[c] = group_colors[c]
                    else:
                        cs = str(c)
                        if cs in group_colors:
                            color_map[c] = group_colors[cs]
                        else:
                            # try numeric conversion match (e.g., "0" vs 0)
                            try:
                                ci = int(float(cs))
                                if ci in group_colors:
                                    color_map[c] = group_colors[ci]
                            except Exception:
                                pass
            else:
                if len(group_colors) != len(cats):
                    plt.close(fig)
                    raise ValueError(
                        f"`group_colors` length ({len(group_colors)}) must match "
                        f"number of categories ({len(cats)}): {cats}"
                    )
                color_map = {c: group_colors[i] for i, c in enumerate(cats)}

        # plot per category to create a legend
        for c in cats:
            mask = (g_str == str(c))
            ax.scatter(
                x[mask],
                y[mask],
                s=size,
                alpha=alpha,
                color=color_map[c],
                label=str(c),
            )

        ax.legend(title=group_by, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax.set_xlabel(f"PC{pcx}")
    ax.set_ylabel(f"PC{pcy}")
    ax.set_title(f"PCA projection colored by {group_by}")
    fig.subplots_adjust(right=0.78)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)

    plt.show()
    plt.close(fig)


def combine_ratio_depth(ad_ratio, ad_depth):
    """
    Combine SNV ratio (ad_ratio) and depth (ad_depth) into a single AnnData.

    Parameters
    ----------
    ad_ratio : AnnData
        SNV ratio data. Values must be in [0, 1].
    ad_depth : AnnData
        Depth data. Must cover the same obs/var as ad_ratio.

    Returns
    -------
    AnnData
        Copy of ad_ratio with depth stored in layers["depth"].
    """
    import numpy as np
    from scipy import sparse

    X = ad_ratio.X
    if sparse.issparse(X):
        data = X.data
    else:
        data = X

    if np.nanmin(data) < 0 or np.nanmax(data) > 1:
        raise ValueError("Ratio data should be in [0,1]")

    adata = ad_ratio.copy()
    r = ad_depth[adata.obs_names, adata.var_names]
    adata.layers["depth"] = r.X

    return adata


def filter_snv_by_depth_ratio(
    adata,
    min_tot_depth=20,
    min_mut_depth=0,
    min_mut_avg_ratio=0,
    x_layer=None,
    depth_layer="depth",
):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    # ---- checks ----
    if depth_layer not in adata.layers:
        raise KeyError(f"Missing '{depth_layer}' in adata.layers.")
    depth = adata.layers[depth_layer]

    X = adata.layers[x_layer] if x_layer is not None else adata.X

    # depth could be per-cell (n_obs,) or (n_obs, 1), or per-cell-per-var (n_obs, n_vars)
    per_cell = (getattr(depth, "ndim", 2) == 1) or (hasattr(depth, "shape") and len(depth.shape) == 2 and depth.shape[1] == 1)

    # ---- total depth per SNV ----
    if per_cell:
        d = np.asarray(depth.toarray()).ravel() if sp.issparse(depth) else np.asarray(depth).ravel()
        denom_scalar = float(d.sum())
        # treat D_ij = d_i for all j (consistent with weighted_per_var per_cell branch)
        snv_depth_sums = pd.Series(np.full(adata.n_vars, denom_scalar, dtype=float), index=adata.var_names)
    else:
        if sp.issparse(depth):
            snv_depth_arr = np.asarray(depth.sum(axis=0)).ravel()
        else:
            snv_depth_arr = np.asarray(depth).sum(axis=0)

        if snv_depth_arr.shape[0] != adata.n_vars:
            raise ValueError("Depth matrix's number of columns does not match adata.var_names.")

        snv_depth_sums = pd.Series(snv_depth_arr.astype(float, copy=False), index=adata.var_names)

    # ---- mutated reads sums: sum_i X_ij * D_ij  (only compute if needed for min_mut_depth or min_mut_avg_ratio) ----
    need_num = (float(min_mut_depth) > 0) or (float(min_mut_avg_ratio) > 0)

    if need_num:
        if per_cell:
            d = np.asarray(depth.toarray()).ravel() if sp.issparse(depth) else np.asarray(depth).ravel()
            if sp.issparse(X):
                num = np.asarray(X.multiply(d[:, None]).sum(axis=0)).ravel()
            else:
                num = (np.asarray(X) * d[:, None]).sum(axis=0)
            num = num.astype(float, copy=False)
        else:
            if sp.issparse(X) and sp.issparse(depth):
                num = np.asarray(X.multiply(depth).sum(axis=0)).ravel()
            elif sp.issparse(X) and not sp.issparse(depth):
                # sparse X elementwise-multiply dense depth is supported
                num = np.asarray(X.multiply(np.asarray(depth)).sum(axis=0)).ravel()
            elif (not sp.issparse(X)) and sp.issparse(depth):
                num = np.asarray(depth.multiply(np.asarray(X)).sum(axis=0)).ravel()
            else:
                num = (np.asarray(X) * np.asarray(depth)).sum(axis=0)
            num = num.astype(float, copy=False)

        if num.shape[0] != adata.n_vars:
            raise ValueError("Mutated reads matrix's number of columns does not match adata.var_names.")

        snv_mut_sums = pd.Series(num, index=adata.var_names)
    else:
        snv_mut_sums = pd.Series(np.zeros(adata.n_vars, dtype=float), index=adata.var_names)

    # ---- weighted average ratio per SNV (same as weighted_per_var) ----
    if float(min_mut_avg_ratio) > 0:
        if per_cell:
            denom = float(snv_depth_sums.iloc[0])  # scalar
            mut_avg_ratio = np.divide(
                snv_mut_sums.to_numpy(),
                denom,
                out=np.full(adata.n_vars, np.nan, dtype=float),
                where=(denom != 0),
            )
        else:
            denom = snv_depth_sums.to_numpy()
            mut_avg_ratio = np.divide(
                snv_mut_sums.to_numpy(),
                denom,
                out=np.full(adata.n_vars, np.nan, dtype=float),
                where=(denom != 0),
            )
        mut_avg_ratio = pd.Series(mut_avg_ratio, index=adata.var_names)
    else:
        # not used in mask
        mut_avg_ratio = None

    # ---- build mask ----
    mask = (snv_depth_sums >= float(min_tot_depth))

    if float(min_mut_depth) > 0:
        mask &= (snv_mut_sums >= float(min_mut_depth))

    if float(min_mut_avg_ratio) > 0:
        mask &= (mut_avg_ratio >= float(min_mut_avg_ratio))

    snv_to_keep = snv_depth_sums.index[mask]

    print(f"SNV number before filter: {adata.n_vars}")
    print(f"SNV number after filter: {len(snv_to_keep)}")

    return adata[:, snv_to_keep].copy()


def test_all_differential_snv_by_cmb(
    adata,
    min_depth: float = 20,
    depth_layer: str = "depth",
    alternative: str = "two-sided",
    fdr_method: str = "fdr_bh",
    sort_by: str = "FDR",
    return_extra_cols: bool = True,
    fdr_threshold: float | None = 0.05,
    test: str = "fisher",
    chi2_correction: bool = True,
    ratio_diff_threshold: float = 0.2,
) -> pd.DataFrame:

    import numpy as np
    import scipy.sparse as sp
    import warnings
    from scipy.stats import fisher_exact, chi2_contingency
    from statsmodels.stats.multitest import multipletests

    test = str(test).lower()
    valid_tests = {"fisher", "chi2"}
    if test not in valid_tests:
        raise ValueError(f"Invalid test='{test}'. Must be one of {sorted(valid_tests)}.")

    if depth_layer not in adata.layers:
        raise KeyError(f"Missing layer in adata.layers: '{depth_layer}'")

    if test == "chi2" and (alternative is not None and str(alternative).lower() != "two-sided"):
        warnings.warn("Chi-square test ignores 'alternative'; it is effectively two-sided.")

    X = adata.X
    D = adata.layers[depth_layer]

    n_cmb, n_snv = adata.shape

    def _col_as_array(mat, col_index: int) -> np.ndarray:
        """Return a dense 1D numpy array for the specified column of a dense/sparse matrix."""
        if sp.issparse(mat):
            return mat[:, col_index].toarray().ravel()
        arr = np.asarray(mat)
        return arr[:, col_index].ravel()

    records = []

    for j in range(n_snv):
        snv_name = adata.var_names[j]

        depth_col = _col_as_array(D, j).astype(float)
        val_col = _col_as_array(X, j).astype(float)

        ratio = np.clip(val_col, 0.0, 1.0)
        valid = np.isfinite(ratio) & np.isfinite(depth_col) & (depth_col >= float(min_depth))

        if valid.sum() < 2:
            continue

        mut_counts_all = np.rint(ratio * depth_col).astype(int)
        mut_counts_all = np.clip(mut_counts_all, 0, depth_col.astype(int))
        ref_counts_all = depth_col.astype(int) - mut_counts_all

        total_mut = mut_counts_all[valid].sum()
        total_ref = ref_counts_all[valid].sum()

        valid_idx = np.where(valid)[0]
        for i in valid_idx:
            cmb_name = adata.obs_names[i]

            mut_i = int(mut_counts_all[i])
            ref_i = int(ref_counts_all[i])

            other_mut = int(total_mut - mut_i)
            other_ref = int(total_ref - ref_i)
            other_depth = int(other_mut + other_ref)
            other_ratio = other_mut / max(other_depth, 1) if other_depth > 0 else np.nan

            if not (np.isfinite(other_ratio) and abs(float(ratio[i]) - float(other_ratio)) >= float(ratio_diff_threshold)):
                continue

            if (mut_i + ref_i) == 0 or (other_mut + other_ref) == 0:
                p = np.nan
            else:
                table = [[mut_i, ref_i], [other_mut, other_ref]]
                try:
                    if test == "fisher":
                        _, p = fisher_exact(table, alternative=alternative)
                    else:
                        _, p, _, _ = chi2_contingency(table, correction=chi2_correction)
                except Exception:
                    p = np.nan

            rec = {
                "SNV": snv_name,
                "CMB": cmb_name,
                "depth": int(depth_col[i]) if np.isfinite(depth_col[i]) else np.nan,
                "mut_ratio": float(ratio[i]) if np.isfinite(ratio[i]) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            }

            if return_extra_cols:
                rec.update({
                    "other_depth": other_depth,
                    "other_mut_ratio": other_ratio,
                })

            records.append(rec)

    desired_order = ["SNV", "CMB", "depth", "other_depth", "mut_ratio", "other_mut_ratio", "p_value", "FDR"]
    if not records:
        if return_extra_cols:
            return pd.DataFrame(columns=desired_order)
        else:
            return pd.DataFrame(columns=[c for c in desired_order if c not in {"other_depth", "other_mut_ratio"}])

    df = pd.DataFrame.from_records(records)

    pvals = df["p_value"].values
    mask = np.isfinite(pvals)
    if mask.any():
        _, qvals, _, _ = multipletests(pvals[mask], method=fdr_method)
        df.loc[mask, "FDR"] = qvals
    else:
        df["FDR"] = np.nan

    if fdr_threshold is not None:
        df = df[df["FDR"].notna() & (df["FDR"] <= float(fdr_threshold))]

    key = "FDR" if str(sort_by).lower() == "fdr" else "p_value"
    df = df.sort_values(by=[key, "p_value"], na_position="last").reset_index(drop=True)

    present = [c for c in desired_order if c in df.columns]
    rest = [c for c in df.columns if c not in present]
    df = df[present + rest]

    return df


def test_comp_differential_snv_by_cmb(
    adata,
    CMB1: str,
    CMB2: str,
    min_depth: float = 20,
    depth_layer: str = "depth",
    alternative: str = "two-sided",
    fdr_method: str = "fdr_bh",
    sort_by: str = "FDR",
    return_extra_cols: bool = True,
    fdr_threshold: float | None = 0.1,
    test: str = "fisher",
    chi2_correction: bool = True,
    ratio_diff_threshold: float = 0.2,
) -> pd.DataFrame:

    import numpy as np
    import scipy.sparse as sp
    import warnings
    from scipy.stats import fisher_exact, chi2_contingency
    from statsmodels.stats.multitest import multipletests

    test = str(test).lower()
    valid_tests = {"fisher", "chi2"}
    if test not in valid_tests:
        raise ValueError(f"Invalid test='{test}'. Must be one of {sorted(valid_tests)}.")

    if depth_layer not in adata.layers:
        raise KeyError(f"Missing layer in adata.layers: '{depth_layer}'")

    if test == "chi2" and (alternative is not None and str(alternative).lower() != "two-sided"):
        warnings.warn("Chi-square test ignores 'alternative'; it is effectively two-sided.")

    if CMB1 == CMB2:
        raise ValueError("CMB1 and CMB2 must be different.")

    try:
        i1 = adata.obs_names.get_loc(CMB1)
    except Exception as e:
        raise KeyError(f"CMB1 '{CMB1}' not found in adata.obs_names.") from e
    try:
        i2 = adata.obs_names.get_loc(CMB2)
    except Exception as e:
        raise KeyError(f"CMB2 '{CMB2}' not found in adata.obs_names.") from e

    X = adata.X
    D = adata.layers[depth_layer]

    n_cmb, n_snv = adata.shape
    if n_cmb <= max(i1, i2):
        raise ValueError("CMB indices out of bounds for adata.X.")

    def _col_as_array(mat, col_index: int) -> np.ndarray:
        if sp.issparse(mat):
            return mat[:, col_index].toarray().ravel()
        arr = np.asarray(mat)
        return arr[:, col_index].ravel()

    records = []

    for j in range(n_snv):
        snv_name = adata.var_names[j]

        depth_col = _col_as_array(D, j).astype(float)
        val_col = _col_as_array(X, j).astype(float)

        ratio_col = np.clip(val_col, 0.0, 1.0)

        d1, d2 = depth_col[i1], depth_col[i2]
        r1, r2 = ratio_col[i1], ratio_col[i2]

        v1 = (np.isfinite(d1) and np.isfinite(r1) and d1 >= float(min_depth))
        v2 = (np.isfinite(d2) and np.isfinite(r2) and d2 >= float(min_depth))
        if not (v1 and v2):
            continue

        if abs(float(r1) - float(r2)) < float(ratio_diff_threshold):
            continue

        mut1 = int(np.clip(np.rint(r1 * d1), 0, int(d1)))
        ref1 = int(d1) - mut1
        mut2 = int(np.clip(np.rint(r2 * d2), 0, int(d2)))
        ref2 = int(d2) - mut2

        if (mut1 + ref1) == 0 or (mut2 + ref2) == 0:
            p = np.nan
        else:
            table = [[mut1, ref1], [mut2, ref2]]
            try:
                if test == "fisher":
                    _, p = fisher_exact(table, alternative=alternative)
                else:
                    _, p, _, _ = chi2_contingency(table, correction=chi2_correction)
            except Exception:
                p = np.nan

        rec = {
            "SNV": snv_name,
            "CMB1": CMB1,
            "CMB2": CMB2,
            "depth1": int(d1),
            "depth2": int(d2),
            "mut_ratio1": float(r1),
            "mut_ratio2": float(r2),
            "p_value": float(p) if np.isfinite(p) else np.nan,
        }

        if return_extra_cols:
            rec.update({
                "mut_count1": mut1,
                "ref_count1": ref1,
                "mut_count2": mut2,
                "ref_count2": ref2,
            })

        records.append(rec)

    desired_order = [
        "SNV", "CMB1", "CMB2", "depth1", "depth2",
        "mut_ratio1", "mut_ratio2", "p_value", "FDR"
    ]

    if not records:
        return pd.DataFrame(columns=desired_order)

    df = pd.DataFrame.from_records(records)

    pvals = df["p_value"].values
    mask = np.isfinite(pvals)
    if mask.any():
        _, qvals, _, _ = multipletests(pvals[mask], method=fdr_method)
        df.loc[mask, "FDR"] = qvals
    else:
        df["FDR"] = np.nan

    if fdr_threshold is not None:
        df = df[df["FDR"].notna() & (df["FDR"] <= float(fdr_threshold))]

    key = "FDR" if str(sort_by).lower() == "fdr" else "p_value"
    df = df.sort_values(by=[key, "p_value"], na_position="last").reset_index(drop=True)

    df = df[desired_order]

    return df


def add_betweenness_centrality_to_adata(
    adata_joint,
    layer="joint_connectivities_sym",
    weight_transform="inv",
    quantile_thr=0.75,
    eps=1e-12,
    add_obs_name="between_centrality",
    plot_bet_centrality_box=True,
    box_figsize=(7, 3),
    annotate_high=True,
    jitter=0.06,
    random_state=0,
    savepath=None,
    dpi=300,
    transparent=False,
    bbox_inches="tight",
):
    import numpy as np
    import scipy.sparse as sp

    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is required: pip install networkx") from e

    import matplotlib.pyplot as plt

    if not isinstance(add_obs_name, str) or len(add_obs_name.strip()) == 0:
        raise ValueError("add_obs_name must be a non-empty string.")
    add_obs_name = add_obs_name.strip()
    add_obs_name_high = f"{add_obs_name}_high"

    if layer not in adata_joint.obsp:
        raise KeyError(
            f"'{layer}' not found in adata_joint.obsp. Available keys: {list(adata_joint.obsp.keys())}"
        )

    A = adata_joint.obsp[layer]
    W = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A))

    if W.shape[0] != W.shape[1] or W.shape[0] != adata_joint.n_obs:
        raise ValueError(f"Matrix shape {W.shape} does not match adata_joint.n_obs={adata_joint.n_obs}.")

    if weight_transform not in {"inv", "neglog"}:
        raise ValueError("weight_transform must be 'inv' or 'neglog'.")

    labels = adata_joint.obs_names.astype(str).to_numpy()
    n = len(labels)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    W = W.tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()

    Wu = sp.triu(W, k=1).tocoo()
    for i, j, w in zip(Wu.row, Wu.col, Wu.data):
        w = float(w)
        if w <= 0.0:
            continue

        if weight_transform == "inv":
            length = 1.0 / (w + eps)
        else:
            ww = float(np.clip(w, eps, 1.0))
            length = -float(np.log(ww))

        G.add_edge(int(i), int(j), length=float(length), sim=float(w))

    bc = nx.betweenness_centrality(G, weight="length", normalized=True)
    bet = np.array([bc.get(i, 0.0) for i in range(n)], dtype=float)

    adata_joint.obs[add_obs_name] = bet

    thr = float(np.quantile(bet, quantile_thr)) if n > 0 else 0.0
    high = bet > thr
    adata_joint.obs[add_obs_name_high] = np.where(high, "Yes", "No")

    q1 = float(np.quantile(bet, 0.25)) if n > 0 else 0.0
    med = float(np.quantile(bet, 0.50)) if n > 0 else 0.0
    q3 = float(np.quantile(bet, 0.75)) if n > 0 else 0.0

    fig_ax = None
    if plot_bet_centrality_box:
        rng = np.random.default_rng(random_state)
        y = 1.0 + rng.uniform(-jitter, jitter, size=n)

        fig, ax = plt.subplots(figsize=box_figsize)
        ax.boxplot(bet, vert=False)
        ax.scatter(bet, y, s=18, alpha=0.7)

        if annotate_high:
            for i in np.where(high)[0]:
                ax.text(
                    float(bet[i]),
                    float(y[i]) + 0.03,
                    str(labels[i]),
                    fontsize=8,
                    rotation=45,
                    ha="left",
                    va="bottom",
                )

        ax.set_yticks([])
        ax.set_xlabel("Betweenness centrality")
        ax.set_title(f"Betweenness centrality | layer={layer} | transform={weight_transform} | obs={add_obs_name}")
        plt.tight_layout()
        fig_ax = (fig, ax)

        if savepath is not None:
            fig.savefig(
                savepath,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
            )

    return {
        "layer_used": layer,
        "add_obs_name": add_obs_name,
        "add_obs_name_high": add_obs_name_high,
        "threshold_quantile": float(quantile_thr),
        "threshold_value": thr,
        "n_high": int(high.sum()),
        "summary_three_points": {"Q1": q1, "Median": med, "Q3": q3},
        "high_labels": labels[high].tolist(),
        "plot": fig_ax,
        "savepath": savepath,
    }


def spatial_filter_conn_by_dist(
    adata,
    conn_key="joint_connectivities_sym",
    x_key="x_center",
    y_key="y_center",
    new_key="joint_connectivities_sym_filteredbyDist",
    dist_quantile=0.95,
    min_keep_per_node=2,
    symmetrize_input=False,
    zero_diagonal=True,
    return_stats=True,
):
    """
    Mask long range edges in a connectivity matrix using physical distances.

    Steps
    1) Use only existing edges (W > 0) to form the edge distance distribution.
    2) Identify long distance edges by quantile threshold: r = quantile(D_edges, dist_quantile).
    3) Set edges with D > r to zero, but enforce a minimum connectivity constraint:
       for each node, keep its min_keep_per_node nearest (by physical distance) original neighbors.
    """
    import numpy as np
    import scipy.sparse as sp

    if conn_key not in adata.obsp:
        raise KeyError(f"{conn_key!r} not found in adata.obsp. Available: {list(adata.obsp.keys())}")

    for k in (x_key, y_key):
        if k not in adata.obs.columns:
            raise KeyError(f"{k!r} not found in adata.obs.columns")

    if not (0.0 < float(dist_quantile) < 1.0):
        raise ValueError("dist_quantile must be in (0, 1). Example: 0.95")

    x = adata.obs[x_key].to_numpy()
    y = adata.obs[y_key].to_numpy()
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        bad = np.where((~np.isfinite(x)) | (~np.isfinite(y)))[0]
        raise ValueError(
            f"Found non finite {x_key}/{y_key} for {len(bad)} rows in adata.obs. "
            f"Please fill them before masking."
        )

    A = adata.obsp[conn_key]
    if sp.issparse(A):
        W = A.tocsr(copy=True)
    else:
        W = sp.csr_matrix(np.asarray(A, dtype=float))

    n = adata.n_obs
    if W.shape != (n, n):
        raise ValueError(f"{conn_key!r} has shape {W.shape}, but adata.n_obs is {n}")

    W.eliminate_zeros()

    if symmetrize_input:
        W = (W + W.T) * 0.5
        W = W.tocsr()
        W.eliminate_zeros()

    if zero_diagonal:
        W.setdiag(0.0)
        W.eliminate_zeros()

    if new_key is None:
        new_key = f"{conn_key}_spatialmasked"

    coo = W.tocoo()
    mask_upper = coo.row < coo.col
    rows = coo.row[mask_upper].astype(int, copy=False)
    cols = coo.col[mask_upper].astype(int, copy=False)
    vals = coo.data[mask_upper].astype(float, copy=False)

    n_edges = rows.size
    if n_edges == 0:
        W_new = W.copy()
        adata.obsp[new_key] = W_new
        stats = {
            "conn_key_in": conn_key,
            "conn_key_out": new_key,
            "n_nodes": int(n),
            "n_edges_input_undirected": 0,
            "dist_quantile": float(dist_quantile),
            "threshold_r": None,
            "min_keep_per_node": int(min_keep_per_node),
            "n_edges_removed": 0,
            "n_edges_output_undirected": 0,
        }
        return (W_new, stats) if return_stats else W_new

    dx = x[rows] - x[cols]
    dy = y[rows] - y[cols]
    dist = np.sqrt(dx * dx + dy * dy)

    r = float(np.quantile(dist, float(dist_quantile)))
    outlier = dist > r

    protected = set()
    m = int(min_keep_per_node)
    if m > 0:
        for i in range(n):
            row_i = W.getrow(i)
            nbrs = row_i.indices
            if nbrs.size == 0:
                continue

            dxi = x[nbrs] - x[i]
            dyi = y[nbrs] - y[i]
            di = np.sqrt(dxi * dxi + dyi * dyi)

            if nbrs.size <= m:
                keep_nbrs = nbrs
            else:
                keep_nbrs = nbrs[np.argsort(di)[:m]]

            for j in keep_nbrs:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                if a != b:
                    protected.add((a, b))

    keep = np.ones(n_edges, dtype=bool)
    if np.any(outlier):
        for k_idx in np.where(outlier)[0]:
            a = int(rows[k_idx])
            b = int(cols[k_idx])
            if (a, b) not in protected:
                keep[k_idx] = False

    rows_k = rows[keep]
    cols_k = cols[keep]
    vals_k = vals[keep]

    W_new = sp.coo_matrix((vals_k, (rows_k, cols_k)), shape=(n, n)).tocsr()
    W_new = W_new + W_new.T
    if zero_diagonal:
        W_new.setdiag(0.0)
    W_new.eliminate_zeros()

    adata.obsp[new_key] = W_new

    stats = {
        "conn_key_in": conn_key,
        "conn_key_out": new_key,
        "n_nodes": int(n),
        "n_edges_input_undirected": int(n_edges),
        "dist_quantile": float(dist_quantile),
        "threshold_r": r,
        "min_keep_per_node": int(min_keep_per_node),
        "n_outlier_edges": int(outlier.sum()),
        "n_edges_removed": int((~keep).sum()),
        "n_edges_output_undirected": int(vals_k.size),
        "dist_min": float(dist.min()) if dist.size else None,
        "dist_median": float(np.median(dist)) if dist.size else None,
        "dist_max": float(dist.max()) if dist.size else None,
    }

    return (W_new, stats) if return_stats else W_new


def infer_local_callable_directions(
    adata,
    conn_key="joint_connectivities_sym",
    ratio_layer="original_ratio",
    depth_layer="depth",
    min_callable_depth=20,
    min_mut_umi=5,
    min_mut_ratio=0.1,
    top_conn_percentile=0.5,
    spatial_max_dist=None,
    spatial_max_dist_quantile=0.5,
    x_key="x_center",
    y_key="y_center",
    min_pair_callable_sites=20,
    min_private_count_diff=3,
    min_private_rate_diff=0.01,
    founder_top_fraction=0.10,
    add_to_obs=True,
    obs_prefix="local_dir",
    verbose=True,
):
    """
    Infer local CMB direction using callable SNV sites only.

    For each retained local edge CMB_i -- CMB_j:
      1. Use only SNV sites callable in both CMBs.
      2. Callable is defined only by coverage:
             total UMI depth >= min_callable_depth.
      3. Present mutation is defined by:
             callable,
             mutated UMI >= min_mut_umi,
             mutation ratio >= min_mut_ratio.
      4. Direction is inferred from private SNV burden:
             the CMB with fewer private SNVs is treated as relatively upstream,
             and the CMB with more private SNVs is treated as relatively downstream.
      5. If the two CMBs are too similar, no arrow is assigned.

    Notes
    -----
    edge_df["source"] and edge_df["target"] are retained as graph direction terms.
    Node-level candidate upstream regions are reported as "Candidate founder-like",
    not "Candidate source", to match the manuscript terminology.

    This function only identifies candidate founder-like CMBs. It does not assign
    candidate terminal-like CMBs, because terminal-like or sink-like behavior in a
    sparse local graph can reflect local downstream connectivity rather than a
    global developmental endpoint.

    Candidate founder-like CMBs are selected as the top fraction of outgoing-biased
    CMBs ranked by net_founder_support. By default, founder_top_fraction=0.10.

    Returns
    -------
    edge_df : pandas.DataFrame
        Local edge table with direction assignment.
    node_df : pandas.DataFrame
        Node-level founder-like support table.
    """

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if conn_key not in adata.obsp:
        raise KeyError(f"{conn_key!r} not found in adata.obsp.")

    if depth_layer not in adata.layers:
        raise KeyError(f"{depth_layer!r} not found in adata.layers.")

    founder_top_fraction = float(founder_top_fraction)
    if not (0.0 < founder_top_fraction <= 1.0):
        raise ValueError("founder_top_fraction must be in (0, 1].")

    labels = pd.Index([str(x) for x in adata.obs_names])
    n_obs = adata.n_obs

    obs = adata.obs.copy()
    obs.index = labels

    conn = adata.obsp[conn_key]

    if sp.issparse(conn):
        conn = conn.toarray()
    else:
        conn = np.asarray(conn)

    if conn.shape != (n_obs, n_obs):
        raise ValueError(
            f"{conn_key!r} has shape {conn.shape}, expected {(n_obs, n_obs)}."
        )

    conn = np.asarray(conn, dtype=float)
    conn = np.nan_to_num(conn, nan=0.0, posinf=0.0, neginf=0.0)

    # Treat the graph as undirected before local edge selection.
    conn = np.maximum(conn, conn.T)
    np.fill_diagonal(conn, 0.0)

    triu = np.triu_indices(n_obs, k=1)
    weights_all = conn[triu]
    positive_weights = weights_all[weights_all > 0]

    empty_edge_columns = [
        "u",
        "v",
        "source",
        "target",
        "direction",
        "directed",
        "weight",
        "spatial_distance",
        "n_pair_callable",
        "n_shared",
        "n_private_u",
        "n_private_v",
        "private_rate_u",
        "private_rate_v",
        "private_count_diff_v_minus_u",
        "private_rate_diff_v_minus_u",
        "direction_strength",
        "reason",
    ]

    def _empty_node_df():
        df = pd.DataFrame(index=labels)
        df.index.name = "CMB"
        df["out_n"] = 0
        df["in_n"] = 0
        df["outgoing_support"] = 0.0
        df["incoming_support"] = 0.0
        df["net_founder_support"] = 0.0
        df["total_directed_edges"] = 0
        df["candidate_role"] = "Other"
        df["is_candidate_founder"] = False
        return df

    if positive_weights.size == 0:
        edge_df = pd.DataFrame(columns=empty_edge_columns)
        node_df = _empty_node_df()

        if add_to_obs:
            for col in node_df.columns:
                adata.obs[f"{obs_prefix}_{col}"] = node_df.reindex(labels)[col].to_numpy()

        return edge_df, node_df

    keep_frac = float(top_conn_percentile)
    if keep_frac <= 0 or keep_frac > 1:
        raise ValueError("top_conn_percentile must be in (0, 1].")

    conn_threshold = float(np.quantile(positive_weights, 1.0 - keep_frac))

    candidate_edges = []
    for i, j, w in zip(triu[0], triu[1], weights_all):
        if w > 0 and w >= conn_threshold:
            candidate_edges.append((int(i), int(j), float(w)))

    spatial_dist = {}
    can_spatial_filter = (
        x_key in obs.columns
        and y_key in obs.columns
        and len(candidate_edges) > 0
    )

    if can_spatial_filter:
        xy = obs.loc[labels, [x_key, y_key]].astype(float).to_numpy()
        dists = []

        for i, j, _ in candidate_edges:
            d = float(np.linalg.norm(xy[i] - xy[j]))
            spatial_dist[(i, j)] = d
            dists.append(d)

        dists = np.asarray(dists, dtype=float)

        if spatial_max_dist is None and spatial_max_dist_quantile is not None:
            spatial_max_dist = float(
                np.quantile(dists, float(spatial_max_dist_quantile))
            )

        if spatial_max_dist is not None:
            spatial_max_dist = float(spatial_max_dist)
            candidate_edges = [
                (i, j, w)
                for i, j, w in candidate_edges
                if spatial_dist.get((i, j), np.inf) <= spatial_max_dist
            ]

    elif verbose and (spatial_max_dist is not None or spatial_max_dist_quantile is not None):
        print(
            "[infer_local_callable_directions] WARNING: spatial filtering was requested, "
            f"but {x_key!r} and/or {y_key!r} were not found in adata.obs. "
            "Spatial filtering was skipped."
        )

    if ratio_layer is not None and ratio_layer in adata.layers:
        X = adata.layers[ratio_layer]
        ratio_used = ratio_layer
    else:
        X = adata.X
        ratio_used = "X"
        if ratio_layer is not None and verbose:
            print(
                f"[infer_local_callable_directions] WARNING: {ratio_layer!r} not found. "
                "Using adata.X as mutation ratio."
            )

    D = adata.layers[depth_layer]

    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    if sp.issparse(D):
        D = D.toarray()
    else:
        D = np.asarray(D)

    if X.shape != D.shape:
        raise ValueError(
            f"Ratio matrix shape {X.shape} does not match depth matrix shape {D.shape}."
        )

    X = np.asarray(X, dtype=float)
    X = np.clip(X, 0.0, 1.0)

    D = np.asarray(D, dtype=float)
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

    # Callable only means this CMB has enough total UMI coverage at this SNV site.
    callable_mtx = D >= float(min_callable_depth)

    # Approximate mutated UMI from ratio and depth.
    mut_umi = np.rint(X * D)
    mut_umi = np.clip(mut_umi, 0.0, D)

    # Present mutation means callable plus enough alternative allele support.
    present_mtx = (
        callable_mtx
        & (mut_umi >= float(min_mut_umi))
        & (X >= float(min_mut_ratio))
    )

    records = []

    for i, j, w in candidate_edges:
        name_i = labels[i]
        name_j = labels[j]

        paired_callable = callable_mtx[i, :] & callable_mtx[j, :]
        n_pair_callable = int(paired_callable.sum())

        rec = {
            "u": name_i,
            "v": name_j,
            "source": None,
            "target": None,
            "direction": 0,
            "directed": False,
            "weight": float(w),
            "spatial_distance": spatial_dist.get((i, j), np.nan),
            "n_pair_callable": n_pair_callable,
            "n_shared": 0,
            "n_private_u": 0,
            "n_private_v": 0,
            "private_rate_u": np.nan,
            "private_rate_v": np.nan,
            "private_count_diff_v_minus_u": np.nan,
            "private_rate_diff_v_minus_u": np.nan,
            "direction_strength": 0.0,
            "reason": "insufficient_pair_callable_sites",
        }

        if n_pair_callable < int(min_pair_callable_sites):
            records.append(rec)
            continue

        present_i = present_mtx[i, :] & paired_callable
        present_j = present_mtx[j, :] & paired_callable

        shared = present_i & present_j
        private_i = present_i & (~present_j)
        private_j = present_j & (~present_i)

        n_shared = int(shared.sum())
        n_private_i = int(private_i.sum())
        n_private_j = int(private_j.sum())

        private_rate_i = n_private_i / float(n_pair_callable)
        private_rate_j = n_private_j / float(n_pair_callable)

        diff_count = n_private_j - n_private_i
        diff_rate = private_rate_j - private_rate_i

        rec.update(
            {
                "n_shared": n_shared,
                "n_private_u": n_private_i,
                "n_private_v": n_private_j,
                "private_rate_u": private_rate_i,
                "private_rate_v": private_rate_j,
                "private_count_diff_v_minus_u": int(diff_count),
                "private_rate_diff_v_minus_u": float(diff_rate),
                "direction_strength": float(abs(diff_rate)),
                "reason": "ambiguous_private_snv_difference",
            }
        )

        pass_count = abs(diff_count) >= int(min_private_count_diff)
        pass_rate = abs(diff_rate) >= float(min_private_rate_diff)

        if pass_count and pass_rate:
            if diff_count > 0:
                # v has more private SNVs, so u is relatively upstream.
                rec.update(
                    {
                        "source": name_i,
                        "target": name_j,
                        "direction": 1,
                        "directed": True,
                        "reason": "v_has_more_private_snvs",
                    }
                )
            elif diff_count < 0:
                # u has more private SNVs, so v is relatively upstream.
                rec.update(
                    {
                        "source": name_j,
                        "target": name_i,
                        "direction": -1,
                        "directed": True,
                        "reason": "u_has_more_private_snvs",
                    }
                )

        records.append(rec)

    edge_df = pd.DataFrame.from_records(records, columns=empty_edge_columns)

    node_df = pd.DataFrame(index=labels)
    node_df.index.name = "CMB"

    node_df["out_n"] = 0
    node_df["in_n"] = 0
    node_df["outgoing_support"] = 0.0
    node_df["incoming_support"] = 0.0
    node_df["candidate_role"] = "Other"

    if len(edge_df) > 0:
        directed_edges = edge_df[edge_df["directed"].astype(bool)].copy()
    else:
        directed_edges = edge_df.copy()

    for _, row in directed_edges.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])

        # Direction support combines local SNV direction strength and graph connectivity.
        support = float(row["direction_strength"]) * float(row["weight"])

        if src in node_df.index:
            node_df.loc[src, "out_n"] += 1
            node_df.loc[src, "outgoing_support"] += support

        if tgt in node_df.index:
            node_df.loc[tgt, "in_n"] += 1
            node_df.loc[tgt, "incoming_support"] += support

    node_df["net_founder_support"] = (
        node_df["outgoing_support"] - node_df["incoming_support"]
    )
    node_df["total_directed_edges"] = node_df["out_n"] + node_df["in_n"]

    # Candidate founder-like CMBs are the top fraction of outgoing-biased nodes
    # in the local directed graph.
    founder_pool = node_df[
        (node_df["out_n"] > 0)
        & (node_df["net_founder_support"] > 0)
    ].copy()

    founder_pool = founder_pool.sort_values(
        by=["net_founder_support", "outgoing_support", "out_n"],
        ascending=[False, False, False],
    )

    if founder_pool.shape[0] > 0:
        n_founder_select = max(
            1,
            int(np.floor(n_obs * founder_top_fraction + 0.5))
        )
        founder_candidates = founder_pool.head(n_founder_select).index.tolist()
    else:
        n_founder_select = 0
        founder_candidates = []

    node_df.loc[founder_candidates, "candidate_role"] = "Candidate founder-like"
    node_df["is_candidate_founder"] = node_df.index.isin(founder_candidates)

    if add_to_obs:
        for col in node_df.columns:
            adata.obs[f"{obs_prefix}_{col}"] = node_df.reindex(labels)[col].to_numpy()

    edge_df.attrs["params"] = {
        "conn_key": conn_key,
        "ratio_used": ratio_used,
        "depth_layer": depth_layer,
        "min_callable_depth": min_callable_depth,
        "min_mut_umi": min_mut_umi,
        "min_mut_ratio": min_mut_ratio,
        "top_conn_percentile": top_conn_percentile,
        "spatial_max_dist": spatial_max_dist,
        "spatial_max_dist_quantile": spatial_max_dist_quantile,
        "min_pair_callable_sites": min_pair_callable_sites,
        "min_private_count_diff": min_private_count_diff,
        "min_private_rate_diff": min_private_rate_diff,
        "founder_top_fraction": founder_top_fraction,
        "n_founder_requested_by_fraction": n_founder_select,
        "n_founder_selected": len(founder_candidates),
    }

    if verbose:
        n_directed = int(edge_df["directed"].sum()) if len(edge_df) else 0
        print(f"[infer_local_callable_directions] Retained local edges: {len(edge_df)}")
        print(f"[infer_local_callable_directions] Directed local edges: {n_directed}")
        print(f"[infer_local_callable_directions] Founder top fraction: {founder_top_fraction}")
        print(f"[infer_local_callable_directions] Candidate founder-like CMBs: {founder_candidates}")

    return edge_df, node_df


def infer_local_callable_directions2(
    adata,
    conn_key="joint_connectivities_sym",
    ratio_layer="original_ratio",
    depth_layer="depth",
    min_callable_depth=20,
    min_mut_umi=5,
    min_mut_ratio=0.1,
    top_conn_percentile=0.5,
    spatial_max_dist=None,
    spatial_max_dist_quantile=0.5,
    x_key="x_center",
    y_key="y_center",
    min_pair_callable_sites=20,
    min_private_count_diff=3,
    min_private_rate_diff=0.01,
    min_snv_jaccard=0.03,
    founder_top_fraction=0.10,
    add_to_obs=True,
    obs_prefix="local_dir",
    verbose=True,
):
    """
    Parameters
    ----------
    adata : AnnData
        AnnData object in which observations are CMBs or cell type units and variables are SNV sites.
    conn_key : str
        Key in adata.obsp storing the local connectivity matrix between units.
    ratio_layer : str or None
        Layer storing mutation ratios. If unavailable, adata.X is used.
    depth_layer : str
        Layer storing total UMI depth for each unit and SNV site.
    min_callable_depth : int or float
        Minimum total depth required for an SNV site to be callable in one unit.
    min_mut_umi : int or float
        Minimum mutated UMI count required to define a mutation as present.
    min_mut_ratio : float
        Minimum mutation ratio required to define a mutation as present.
    top_conn_percentile : float
        Fraction of strongest positive connectivity edges retained before optional spatial filtering.
    spatial_max_dist : float or None
        Maximum allowed spatial distance between two connected units. If None, spatial_max_dist_quantile can be used.
    spatial_max_dist_quantile : float or None
        Quantile of candidate edge spatial distances used as the maximum spatial distance when spatial_max_dist is None.
    x_key : str
        Column name in adata.obs storing x coordinates.
    y_key : str
        Column name in adata.obs storing y coordinates.
    min_pair_callable_sites : int
        Minimum number of SNV sites callable in both units for pairwise direction inference.
    min_private_count_diff : int
        Minimum absolute difference in private SNV counts required to assign a candidate direction.
    min_private_rate_diff : float
        Minimum absolute difference in private SNV rates required to assign a candidate direction.
    min_snv_jaccard : float
        Minimum Jaccard similarity between the two units' present SNV sets required to keep a directed edge.
    founder_top_fraction : float
        Fraction of units selected as candidate founder-like units among outgoing-biased nodes.
    add_to_obs : bool
        Whether to write node-level results back to adata.obs.
    obs_prefix : str
        Prefix used for columns written to adata.obs.
    verbose : bool
        Whether to print a short summary of retained and directed edges.

    Returns
    -------
    edge_df : pandas.DataFrame
        Local edge table with direction assignment and SNV Jaccard filtering.
    node_df : pandas.DataFrame
        Node-level founder-like support table.
    """

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if conn_key not in adata.obsp:
        raise KeyError(f"{conn_key!r} not found in adata.obsp.")

    if depth_layer not in adata.layers:
        raise KeyError(f"{depth_layer!r} not found in adata.layers.")

    founder_top_fraction = float(founder_top_fraction)
    if not (0.0 < founder_top_fraction <= 1.0):
        raise ValueError("founder_top_fraction must be in (0, 1].")

    min_snv_jaccard = float(min_snv_jaccard)
    if not (0.0 <= min_snv_jaccard <= 1.0):
        raise ValueError("min_snv_jaccard must be in [0, 1].")

    labels = pd.Index([str(x) for x in adata.obs_names])
    n_obs = adata.n_obs

    obs = adata.obs.copy()
    obs.index = labels

    conn = adata.obsp[conn_key]

    if sp.issparse(conn):
        conn = conn.toarray()
    else:
        conn = np.asarray(conn)

    if conn.shape != (n_obs, n_obs):
        raise ValueError(
            f"{conn_key!r} has shape {conn.shape}, expected {(n_obs, n_obs)}."
        )

    conn = np.asarray(conn, dtype=float)
    conn = np.nan_to_num(conn, nan=0.0, posinf=0.0, neginf=0.0)

    conn = np.maximum(conn, conn.T)
    np.fill_diagonal(conn, 0.0)

    triu = np.triu_indices(n_obs, k=1)
    weights_all = conn[triu]
    positive_weights = weights_all[weights_all > 0]

    empty_edge_columns = [
        "u",
        "v",
        "source",
        "target",
        "direction",
        "directed",
        "weight",
        "spatial_distance",
        "n_pair_callable",
        "n_shared",
        "n_union_present",
        "snv_jaccard",
        "n_private_u",
        "n_private_v",
        "private_rate_u",
        "private_rate_v",
        "private_count_diff_v_minus_u",
        "private_rate_diff_v_minus_u",
        "direction_strength",
        "reason",
    ]

    def _empty_node_df():
        df = pd.DataFrame(index=labels)
        df.index.name = "CMB"
        df["out_n"] = 0
        df["in_n"] = 0
        df["outgoing_support"] = 0.0
        df["incoming_support"] = 0.0
        df["net_founder_support"] = 0.0
        df["total_directed_edges"] = 0
        df["candidate_role"] = "Other"
        df["is_candidate_founder"] = False
        return df

    if positive_weights.size == 0:
        edge_df = pd.DataFrame(columns=empty_edge_columns)
        node_df = _empty_node_df()

        if add_to_obs:
            for col in node_df.columns:
                adata.obs[f"{obs_prefix}_{col}"] = node_df.reindex(labels)[col].to_numpy()

        return edge_df, node_df

    keep_frac = float(top_conn_percentile)
    if keep_frac <= 0 or keep_frac > 1:
        raise ValueError("top_conn_percentile must be in (0, 1].")

    conn_threshold = float(np.quantile(positive_weights, 1.0 - keep_frac))

    candidate_edges = []
    for i, j, w in zip(triu[0], triu[1], weights_all):
        if w > 0 and w >= conn_threshold:
            candidate_edges.append((int(i), int(j), float(w)))

    spatial_dist = {}
    can_spatial_filter = (
        x_key in obs.columns
        and y_key in obs.columns
        and len(candidate_edges) > 0
    )

    if can_spatial_filter:
        xy = obs.loc[labels, [x_key, y_key]].astype(float).to_numpy()
        dists = []

        for i, j, _ in candidate_edges:
            d = float(np.linalg.norm(xy[i] - xy[j]))
            spatial_dist[(i, j)] = d
            dists.append(d)

        dists = np.asarray(dists, dtype=float)

        if spatial_max_dist is None and spatial_max_dist_quantile is not None:
            spatial_max_dist = float(
                np.quantile(dists, float(spatial_max_dist_quantile))
            )

        if spatial_max_dist is not None:
            spatial_max_dist = float(spatial_max_dist)
            candidate_edges = [
                (i, j, w)
                for i, j, w in candidate_edges
                if spatial_dist.get((i, j), np.inf) <= spatial_max_dist
            ]

    elif verbose and (spatial_max_dist is not None or spatial_max_dist_quantile is not None):
        print(
            "[infer_local_callable_directions2] WARNING: spatial filtering was requested, "
            f"but {x_key!r} and/or {y_key!r} were not found in adata.obs. "
            "Spatial filtering was skipped."
        )

    if ratio_layer is not None and ratio_layer in adata.layers:
        X = adata.layers[ratio_layer]
        ratio_used = ratio_layer
    else:
        X = adata.X
        ratio_used = "X"
        if ratio_layer is not None and verbose:
            print(
                f"[infer_local_callable_directions2] WARNING: {ratio_layer!r} not found. "
                "Using adata.X as mutation ratio."
            )

    D = adata.layers[depth_layer]

    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    if sp.issparse(D):
        D = D.toarray()
    else:
        D = np.asarray(D)

    if X.shape != D.shape:
        raise ValueError(
            f"Ratio matrix shape {X.shape} does not match depth matrix shape {D.shape}."
        )

    X = np.asarray(X, dtype=float)
    X = np.clip(X, 0.0, 1.0)

    D = np.asarray(D, dtype=float)
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

    callable_mtx = D >= float(min_callable_depth)

    mut_umi = np.rint(X * D)
    mut_umi = np.clip(mut_umi, 0.0, D)

    present_mtx = (
        callable_mtx
        & (mut_umi >= float(min_mut_umi))
        & (X >= float(min_mut_ratio))
    )

    records = []

    for i, j, w in candidate_edges:
        name_i = labels[i]
        name_j = labels[j]

        paired_callable = callable_mtx[i, :] & callable_mtx[j, :]
        n_pair_callable = int(paired_callable.sum())

        rec = {
            "u": name_i,
            "v": name_j,
            "source": None,
            "target": None,
            "direction": 0,
            "directed": False,
            "weight": float(w),
            "spatial_distance": spatial_dist.get((i, j), np.nan),
            "n_pair_callable": n_pair_callable,
            "n_shared": 0,
            "n_union_present": 0,
            "snv_jaccard": np.nan,
            "n_private_u": 0,
            "n_private_v": 0,
            "private_rate_u": np.nan,
            "private_rate_v": np.nan,
            "private_count_diff_v_minus_u": np.nan,
            "private_rate_diff_v_minus_u": np.nan,
            "direction_strength": 0.0,
            "reason": "insufficient_pair_callable_sites",
        }

        if n_pair_callable < int(min_pair_callable_sites):
            records.append(rec)
            continue

        present_i = present_mtx[i, :] & paired_callable
        present_j = present_mtx[j, :] & paired_callable

        shared = present_i & present_j
        private_i = present_i & (~present_j)
        private_j = present_j & (~present_i)

        n_shared = int(shared.sum())
        n_private_i = int(private_i.sum())
        n_private_j = int(private_j.sum())
        n_union_present = int(n_shared + n_private_i + n_private_j)

        if n_union_present > 0:
            snv_jaccard = n_shared / float(n_union_present)
        else:
            snv_jaccard = 0.0

        private_rate_i = n_private_i / float(n_pair_callable)
        private_rate_j = n_private_j / float(n_pair_callable)

        diff_count = n_private_j - n_private_i
        diff_rate = private_rate_j - private_rate_i

        rec.update(
            {
                "n_shared": n_shared,
                "n_union_present": n_union_present,
                "snv_jaccard": float(snv_jaccard),
                "n_private_u": n_private_i,
                "n_private_v": n_private_j,
                "private_rate_u": private_rate_i,
                "private_rate_v": private_rate_j,
                "private_count_diff_v_minus_u": int(diff_count),
                "private_rate_diff_v_minus_u": float(diff_rate),
                "direction_strength": float(abs(diff_rate)),
                "reason": "ambiguous_private_snv_difference",
            }
        )

        pass_count = abs(diff_count) >= int(min_private_count_diff)
        pass_rate = abs(diff_rate) >= float(min_private_rate_diff)
        pass_jaccard = snv_jaccard >= min_snv_jaccard

        if pass_count and pass_rate and pass_jaccard:
            if diff_count > 0:
                rec.update(
                    {
                        "source": name_i,
                        "target": name_j,
                        "direction": 1,
                        "directed": True,
                        "reason": "v_has_more_private_snvs_pass_jaccard",
                    }
                )
            elif diff_count < 0:
                rec.update(
                    {
                        "source": name_j,
                        "target": name_i,
                        "direction": -1,
                        "directed": True,
                        "reason": "u_has_more_private_snvs_pass_jaccard",
                    }
                )
        elif pass_count and pass_rate and not pass_jaccard:
            rec.update(
                {
                    "direction": 0,
                    "directed": False,
                    "reason": "failed_snv_jaccard_filter",
                }
            )

        records.append(rec)

    edge_df = pd.DataFrame.from_records(records, columns=empty_edge_columns)

    node_df = pd.DataFrame(index=labels)
    node_df.index.name = "CMB"

    node_df["out_n"] = 0
    node_df["in_n"] = 0
    node_df["outgoing_support"] = 0.0
    node_df["incoming_support"] = 0.0
    node_df["candidate_role"] = "Other"

    if len(edge_df) > 0:
        directed_edges = edge_df[edge_df["directed"].astype(bool)].copy()
    else:
        directed_edges = edge_df.copy()

    for _, row in directed_edges.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])

        support = float(row["direction_strength"]) * float(row["weight"])

        if src in node_df.index:
            node_df.loc[src, "out_n"] += 1
            node_df.loc[src, "outgoing_support"] += support

        if tgt in node_df.index:
            node_df.loc[tgt, "in_n"] += 1
            node_df.loc[tgt, "incoming_support"] += support

    node_df["net_founder_support"] = (
        node_df["outgoing_support"] - node_df["incoming_support"]
    )
    node_df["total_directed_edges"] = node_df["out_n"] + node_df["in_n"]

    founder_pool = node_df[
        (node_df["out_n"] > 0)
        & (node_df["net_founder_support"] > 0)
    ].copy()

    founder_pool = founder_pool.sort_values(
        by=["net_founder_support", "outgoing_support", "out_n"],
        ascending=[False, False, False],
    )

    if founder_pool.shape[0] > 0:
        n_founder_select = max(
            1,
            int(np.floor(n_obs * founder_top_fraction + 0.5))
        )
        founder_candidates = founder_pool.head(n_founder_select).index.tolist()
    else:
        n_founder_select = 0
        founder_candidates = []

    node_df.loc[founder_candidates, "candidate_role"] = "Candidate founder-like"
    node_df["is_candidate_founder"] = node_df.index.isin(founder_candidates)

    if add_to_obs:
        for col in node_df.columns:
            adata.obs[f"{obs_prefix}_{col}"] = node_df.reindex(labels)[col].to_numpy()

    edge_df.attrs["params"] = {
        "conn_key": conn_key,
        "ratio_used": ratio_used,
        "depth_layer": depth_layer,
        "min_callable_depth": min_callable_depth,
        "min_mut_umi": min_mut_umi,
        "min_mut_ratio": min_mut_ratio,
        "top_conn_percentile": top_conn_percentile,
        "spatial_max_dist": spatial_max_dist,
        "spatial_max_dist_quantile": spatial_max_dist_quantile,
        "min_pair_callable_sites": min_pair_callable_sites,
        "min_private_count_diff": min_private_count_diff,
        "min_private_rate_diff": min_private_rate_diff,
        "min_snv_jaccard": min_snv_jaccard,
        "founder_top_fraction": founder_top_fraction,
        "n_founder_requested_by_fraction": n_founder_select,
        "n_founder_selected": len(founder_candidates),
    }

    if verbose:
        n_directed = int(edge_df["directed"].sum()) if len(edge_df) else 0
        n_failed_jaccard = int((edge_df["reason"] == "failed_snv_jaccard_filter").sum()) if len(edge_df) else 0
        print(f"[infer_local_callable_directions2] Retained local edges: {len(edge_df)}")
        print(f"[infer_local_callable_directions2] Directed local edges: {n_directed}")
        print(f"[infer_local_callable_directions2] Edges filtered by SNV Jaccard: {n_failed_jaccard}")
        print(f"[infer_local_callable_directions2] Founder top fraction: {founder_top_fraction}")
        print(f"[infer_local_callable_directions2] Candidate founder-like CMBs: {founder_candidates}")

    return edge_df, node_df
