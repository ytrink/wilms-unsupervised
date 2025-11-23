# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:06:03 2025

@author: trinkya
"""

from plotly.offline import plot
from scipy.spatial.distance import pdist, squareform

import plotly.graph_objects as go

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime
import pdb
import pickle
import plotly.express as px
import os
from itertools import cycle


import numpy as np
import pandas as pd
from typing import List, Dict, Literal, Optional


def run_pca_on_normalized_counts(
    expr: pd.DataFrame,
    n_components: int = 3,
    arc_cols: list[str] | None = None,
    n_arcs: int | None = None,
    exclude_archetypes_from_pca: bool = True,
    center: bool = True,
    scale: bool = False,
    log1p: bool = False,
    random_state: int | None = 0,
    svd_solver: str = "auto",
):
    """
    Run PCA on an expression matrix (features x samples)

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix with genes/features in rows and samples in columns - normalized
    n_components : int
        Number of PCA components to compute.
    arc_cols : list[str] | None
        Explicit names of archetype columns (subset of expr.columns).
    n_arcs : int | None
        If arc_cols is None, take the *last* n_arcs columns as archetypes.
    exclude_archetypes_from_pca : bool
        If True, exclude archetype columns from PCA fit/transform (common).
        If False, include them in PCA like any other sample.
    center : bool
        Center features (gene-wise mean subtraction) before PCA.
    scale : bool
        Scale features to unit variance (after centering). (Standardization)
    log1p : bool
        Apply log1p transform to values before centering/scaling/PCA.
    random_state : int | None
        Random seed passed to PCA (for randomized svd if used).
    svd_solver : str
        SVD solver for sklearn PCA.

    Returns
    -------
    result : dict
        {
          "PC": pd.DataFrame of shape (n_samples_used, n_components),
          "explained_variance_ratio": np.ndarray,
          "pca": fitted sklearn PCA object,
          "archetype_cols": list[str],
          "samples_used": list[str],
          "preproc_matrix": pd.DataFrame   # the matrix actually used for PCA (features x samples_used)
        }
    """
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("expr must be a pandas DataFrame (features x samples).")

    # Identify archetype columns
    if arc_cols is None:
        if n_arcs is None:
            archetype_cols = []
        else:
            if n_arcs < 0 or n_arcs > expr.shape[1]:
                raise ValueError("n_arcs must be between 0 and number of columns.")
            archetype_cols = list(expr.columns[-n_arcs:]) if n_arcs > 0 else []
    else:
        missing = set(arc_cols) - set(expr.columns)
        if missing:
            raise ValueError(f"arc_cols not found in expr: {sorted(missing)}")
        archetype_cols = list(arc_cols)

    # Choose whether to include archetypes in PCA run
    if exclude_archetypes_from_pca and len(archetype_cols) > 0:
        samples_used = [c for c in expr.columns if c not in archetype_cols]
    else:
        samples_used = list(expr.columns)

    if len(samples_used) == 0:
        raise ValueError("No samples selected for PCA (check archetype selection/exclusion).")

    X = expr.loc[:, samples_used].copy()

    # Optional transforms
    if log1p:
        X = np.log1p(X)

    if center or scale:
        # center/scale across features (rows), so work on transposed then transpose back
        X_t = X.T  # samples x genes
        if center:
            X_t = X_t - X_t.mean(axis=0)
        if scale:
            std = X_t.std(axis=0, ddof=0).replace(0, 1.0)
            X_t = X_t / std
        X = X_t.T

    
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    PCs = pca.fit_transform(X.T) 

    PC_df = pd.DataFrame(PCs, index=samples_used, columns=[f"PC{i+1}" for i in range(n_components)])

    return {
        "PC": PC_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca": pca,
        "archetype_cols": archetype_cols,
        "samples_used": samples_used,
        "preproc_matrix": X,  
    }





def plot_pca2d(
    pca_result,
    save_to_file=False,
    folder="Desktop",
    file_end="basic",
    file_type="svg"
):
    """
    Plot PCA in 2D with archetypes highlighted.

    Parameters
    ----------
    pca_result : dict
        Output of run_pca_on_expression(). Must include keys 'PC' and 'archetype_cols'.
    save_to_file : bool
        If True, saves plot to file in `folder` with suffix `file_end.file_type`.
    folder : str
        Folder where to save the figure if save_to_file=True.
    file_end : str
        File name suffix, e.g. "basic".
    file_type : str
        Any valid Matplotlib save format (e.g. 'jpeg', 'pdf', 'svg').
    """

    PC = pca_result["PC"]
    arcs = pca_result.get("archetype_cols", [])
    n_arcs = len(arcs)

    # Split normal samples and archetypes
    if n_arcs > 0:
        score_arcs = PC.loc[arcs]
        score_normal = PC.drop(arcs, errors="ignore")
    else:
        score_arcs = None
        score_normal = PC

    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(score_normal["PC1"], score_normal["PC2"], label="Samples", alpha=0.7)
    if score_arcs is not None and len(score_arcs) > 0:
        ax.scatter(
            score_arcs["PC1"], score_arcs["PC2"],
            color="red", marker="*", s=100, label="Archetypes"
        )
        for idx, arc in enumerate(arcs):
            if arc in score_arcs.index:
                ax.text(
                    score_arcs.loc[arc, "PC1"],
                    score_arcs.loc[arc, "PC2"],
                    arc,
                    fontsize=9,
                    ha="center",
                    va="center",
                    color="darkred",
                )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA (2D)")
    ax.legend()
    plt.tight_layout()

    
    if save_to_file:
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, f"{file_end}.{file_type}")
        plt.savefig(out_path, dpi=300)
        print(f"Saved PCA plot to: {out_path}")

    return fig, ax






def plot_gene_expression2d(
    pca_result: dict,
    counts: pd.DataFrame,          
    genes: list,                   
    save_to_file: bool = False,
    folder: str = "Desktop",
    file_type: str = "svg",
    base_size: float = 5.0,        
    scale_size: float = 50.0,     
    cmap: str = "bwr"
):
    """
    Plot 2D PCA (PC1 vs PC2) where point size and color reflect gene expression.
    Used to identify archetypes/latent space

    Parameters
    ----------
    pca_result : dict
        Output from run_pca_on_expression(), must contain 'PC' and 'archetype_cols'.
    counts : pd.DataFrame
        Expression matrix with genes in rows, samples in columns.
    genes : list
        Genes to visualize (each gene gets its own figure).
    save_to_file : bool
        If True, saves each figure to `folder/<gene>.<file_type>`.
    folder : str
        Output folder if saving.
    file_type : str
        Matplotlib-supported format (e.g., 'svg', 'png', 'pdf').
    base_size : float
        Minimum marker size.
    scale_size : float
        Additional size after normalization.
    cmap : str
        Colormap for expression values.
    """
    PC = pca_result["PC"]                    
    arcs = pca_result.get("archetype_cols", [])
    n_arcs = len(arcs)

    
    if n_arcs > 0:
        score_arcs = PC.loc[arcs]
        score_reg = PC.drop(arcs, errors="ignore")
    else:
        score_arcs = PC.iloc[0:0]            # empty frame
        score_reg = PC

    # Ensure counts and PC sample order align for the regular samples
    reg_samples = score_reg.index.tolist()
    
    missing = [s for s in reg_samples if s not in counts.columns]
    if missing:
        raise KeyError(f"Samples missing in counts: {missing[:5]} ...")

    
    if save_to_file:
        os.makedirs(folder, exist_ok=True)

    for g in genes:
        if g not in counts.index:
            print(f"[warn] Gene '{g}' not found in counts. Skipping.")
            continue

        
        expr = counts.loc[g, reg_samples].astype(float)
        
        x_min, x_max = float(expr.min()), float(expr.max()) # standardize for better plots
        rng = x_max - x_min
        eps = 1e-12
        norm = (expr - x_min) / (rng + eps)   # in [0,1]
        sizes = base_size + scale_size * norm

        fig, ax = plt.subplots(figsize=(6, 5))
       
        sc = ax.scatter(
            score_reg["PC1"], score_reg["PC2"],
            s=sizes.values,
            c=norm.values,
            cmap=cmap,
            linewidths=0.7,
            edgecolors="black"
        )
       
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(f"{g} (min–max norm)")

      
        if n_arcs > 0 and len(score_arcs) > 0:
            ax.scatter(
                score_arcs["PC1"], score_arcs["PC2"],
                color="black", marker="*", s=100, label="Archetypes"
            )
            for arc in arcs:
                if arc in score_arcs.index:
                    ax.text(
                        score_arcs.loc[arc, "PC1"],
                        score_arcs.loc[arc, "PC2"],
                        arc, ha="right", va="bottom", fontsize=9
                    )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(g)
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        plt.tight_layout()

        if save_to_file:
            outfile = os.path.join(folder, f"{g}.{file_type}")
            plt.savefig(outfile, dpi=300)
            print(f"Saved: {outfile}")
            plt.close(fig)

    
    return fig, ax



def plot_categorical_pca2d(
    pca_result: dict,
    labels,                         
    save_to_file: bool = False,
    folder: str = "Desktop",
    file_end: str = "basic",
    file_type: str = "svg",
    title: str = ""
):
    """
    Plot PCA (PC1 vs PC2) coloring/marking samples by categorical labels.
    Archetypes are drawn as red stars and annotated.

    Parameters
    ----------
    pca_result : dict
        Output of run_pca_on_expression(); requires 'PC' and 'archetype_cols'.
    labels : list/np.array or pd.Series
        Category per sample. If Series, its index should be sample names.
        If list/array, it must be in the same order as pca_result["PC"].index.
    """

    PC = pca_result["PC"]                      # samples x PCs
    arcs = list(pca_result.get("archetype_cols", []))
    n_arcs = len(arcs)

   
    if n_arcs > 0:
        score_arcs = PC.loc[arcs]
        score_reg  = PC.drop(arcs, errors="ignore")
    else:
        score_arcs = PC.iloc[0:0]
        score_reg  = PC

   
    if isinstance(labels, pd.Series):
       
        lab_reg = labels.reindex(score_reg.index)
    else:
        
        labels = pd.Series(labels, index=PC.index)
        lab_reg = labels.reindex(score_reg.index)

    if lab_reg.isna().any():
        
        missing = lab_reg[lab_reg.isna()].index.tolist()
        print(f"[warn] {len(missing)} samples missing labels; they will be skipped.")

    
    order = []
    seen = set()
    for lab in lab_reg.dropna().tolist():
        if lab not in seen:
            seen.add(lab)
            order.append(lab)


    marker_cycle = cycle(['o','^','s','D','P','X','v','<','>','h','*'])
    color_cycle  = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

   
    for lab in order:
        idx = lab_reg.index[lab_reg == lab]
        if len(idx) == 0:
            continue
        mk = next(marker_cycle)
        col = next(color_cycle)
        ax.scatter(
            score_reg.loc[idx, "PC1"],
            score_reg.loc[idx, "PC2"],
            marker=mk, color=col, alpha=0.85, edgecolors='black', linewidths=0.5,
            label=str(lab)
        )

    
    if len(score_arcs) > 0:
        ax.scatter(score_arcs["PC1"], score_arcs["PC2"], color='red', marker='*', s=120, label='Archetype')
        for arc in arcs:
            if arc in score_arcs.index:
                ax.text(score_arcs.loc[arc, "PC1"], score_arcs.loc[arc, "PC2"], arc,
                        fontsize=9, ha='right', va='bottom', color='darkred')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if title:
        ax.set_title(title)
    ax.legend(title="Group", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save_to_file:
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, f"{file_end}.{file_type}")
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")

    return fig, ax



Direction = Literal["up", "down", "both"]
Mode = Literal["pairwise_all", "median_other"]

def _ensure_samples_exist(counts: pd.DataFrame, samples: List[str]) -> None:
    missing = [s for s in samples if s not in counts.columns]
    if missing:
        raise KeyError(f"Samples not found in counts: {missing}")

def lfc_genes_vectorized_one_vs_others(
    counts: pd.DataFrame,
    sample: str,
    others: List[str],
    threshold: float,
    direction: Direction = "up",
    mode: Mode = "pairwise_all",
) -> Dict[str, List[str]]:
    """
    
    compute lfc of sample vs rest (to identify diff expressed genes in archetypes Fig 1)
    
    mode:
      - "pairwise_all": require sample-others LFC > threshold for ALL others
      - "median_other": compare to the median of others and require LFC > threshold
    Returns dict with keys: {"up": [...], "down": [...]} (down empty if direction="up", etc.)
    """
    _ensure_samples_exist(counts, [sample] + others)
    x1 = counts[sample].astype(float).to_numpy()                  
    Xo = counts[others].astype(float).to_numpy()                  

    if mode == "pairwise_all":
       
        lfc_mat = x1[:, None] - Xo                               
        up_mask = (lfc_mat > threshold).all(axis=1)               
        down_mask = (lfc_mat < -threshold).all(axis=1)            
    elif mode == "median_other":
        med = np.nanmedian(Xo, axis=1)                           
        lfc = x1 - med                                            
        up_mask = lfc > threshold
        down_mask = lfc < -threshold
    else:
        raise ValueError(f"Unknown mode: {mode}")

    genes = counts.index.to_numpy()
    res = {"up": [], "down": []}
    if direction in ("up", "both"):
        res["up"] = genes[up_mask].tolist()
    if direction in ("down", "both"):
        res["down"] = genes[down_mask].tolist()
    return res

def diff_expression_one_vs_all(
    counts: pd.DataFrame,
    samples: List[str],
    threshold: float,
    direction: Direction = "up",
    mode: Mode = "pairwise_all",
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """

    """
    _ensure_samples_exist(counts, samples)

    out: Dict[str, Dict[str, List[str]]] = {}
    for i, s in enumerate(samples):
        others = samples[:i] + samples[i+1:]
        if not others:
            out[s] = {"up": [], "down": []}
            continue
        res = lfc_genes_vectorized_one_vs_others(
            counts=counts,
            sample=s,
            others=others,
            threshold=threshold,
            direction=direction,
            mode=mode,
        )
        out[s] = res

    if save_path:
      
        up_df = pd.DataFrame({k: pd.Series(v["up"]) for k, v in out.items()})
        down_df = pd.DataFrame({k: pd.Series(v["down"]) for k, v in out.items()})
        up_df.to_csv(save_path.replace(".csv", "_up.csv"), index=False)
        down_df.to_csv(save_path.replace(".csv", "_down.csv"), index=False)

    return out
























