
"""
fasttopics_analysis.py

Utility functions to analyze multinomial topic models from fastTopics.


by: Yaron Trink

"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Optional


# ============================================================
# Loading model outputs
# ============================================================

def load_topic_model(
    omega_path: str | Path,
    theta_path: str | Path,
    algo: str = "fastTopics"
) -> dict:
    """
    Load omega (L) and theta (F) matrices from CSV files.

    Returns
    -------
    model : dict
        {"omega": DataFrame, "theta": DataFrame, "k": int, "algo": str}
    """
    omega = pd.read_csv(omega_path, index_col=0)
    theta = pd.read_csv(theta_path, index_col=0)

    return {
        "omega": omega,
        "theta": theta,
        "k": omega.shape[1],
        "algo": algo
    }


# ============================================================
# Topic simplex visualizations
# ============================================================

def plot_topic_simplex_2d(model: dict, x_axis: str, y_axis: str,
                          save: bool = False, outpath: Optional[str] = None):
    omega = model["omega"]
    k = model["k"]
    algo = model["algo"]

    plt.figure()
    plt.scatter(omega[x_axis], omega[y_axis])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f"k={k} ({algo})")
    
    if save and outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()
    
    
def plot_topic_simplex_3d(model: dict, x_axis: str, y_axis: str, z_axis: str,
                          save: bool = False, outpath: Optional[str] = None):

    from mpl_toolkits.mplot3d import Axes3D  # noqa

    omega = model["omega"]
    k = model["k"]
    algo = model["algo"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(omega[x_axis], omega[y_axis], omega[z_axis])
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    ax.set_title(f"k={k} ({algo})")

    if save and outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


# ============================================================
# Posterior P(topic | gene)
# ============================================================

def compute_posterior(model: dict) -> pd.DataFrame:
    """
    Compute posterior P(topic | gene) = theta * prior / evidence.
    """
    theta = model["theta"]
    k = model["k"]

    prior = 1 / k
    post = {}

    for gene, row in theta.iterrows():
        evidence = (row * prior).sum()
        post[gene] = (row * prior) / evidence

    posterior_df = pd.DataFrame(post).T
    return posterior_df


# ============================================================
# Posterior plots
# ============================================================

def plot_posterior_2d(
    posterior: pd.DataFrame,
    topic1: str,
    topic2: str,
    selected_genes: List[str] | None = None,
    save: bool = False,
    outpath: Optional[str] = None,
    t1_label: str = "",
    t2_label: str = "",
):
    """
    2D posterior scatter plot.
    """
    plt.figure()
    plt.scatter(posterior[topic1], posterior[topic2])

    if selected_genes:
        sel = posterior.loc[selected_genes]
        plt.scatter(sel[topic1], sel[topic2], color="red")
        for g in selected_genes:
            plt.annotate(g, (posterior.loc[g, topic1], posterior.loc[g, topic2]))

    plt.xlabel(f"P(topic {topic1}) {t1_label}")
    plt.ylabel(f"P(topic {topic2}) {t2_label}")
    plt.title("Posterior P(topic | gene)")

    if save and outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def plot_posterior_3d(
    posterior: pd.DataFrame,
    topic1: str,
    topic2: str,
    topic3: str,
    selected_genes: List[str] | None = None,
):
    """
    3D posterior scatter plot.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(posterior[topic1], posterior[topic2], posterior[topic3])

    if selected_genes:
        sel = posterior.loc[selected_genes]
        ax.scatter(sel[topic1], sel[topic2], sel[topic3], color="red")
        for g in selected_genes:
            ax.text(sel.loc[g, topic1], sel.loc[g, topic2], sel.loc[g, topic3], g)

    ax.set_xlabel(f"P(topic {topic1})")
    ax.set_ylabel(f"P(topic {topic2})")
    ax.set_zlabel(f"P(topic {topic3})")
    ax.set_title("Posterior P(topic | gene)")


# ============================================================
# Posterior filtering
# ============================================================

def get_high_posterior_genes(
    posterior: pd.DataFrame,
    threshold: float,
) -> Dict[str, List[str]]:
    """
    Returns genes with posterior > threshold for each topic.
    """
    results = {}

    for topic in posterior.columns:
        high = posterior[posterior[topic] > threshold].index.tolist()
        results[topic] = high

    return results


# ============================================================
# Fit time visualization
# ============================================================

def plot_fit_times(times_csv: str, title: str = "FastTopics Runtime"):
    df = pd.read_csv(times_csv)
    plt.figure()
    plt.plot(df["k"], df["runtime_minutes"])
    plt.xlabel("k")
    plt.ylabel("minutes")
    plt.title(title)
    plt.show()


# ============================================================
# PCA + simplex triangulation plot
# ============================================================

def plot_simplex_with_pca(
    model: dict,
    score: np.ndarray,
    filepath: str,
    x_axis: str,
    y_axis: str,
    arc_names: Optional[List[str]] = None,
    filetype: str = "png",
    fix_text: bool = False
):
    """
    Creates: PCA + topic histogram + simplex position for each sample.
    """
    omega = model["omega"]
    samples = omega.index.tolist()

    Path(filepath).mkdir(parents=True, exist_ok=True)

    n_arcs = len(arc_names) if arc_names else 0
    arcs = score[-n_arcs:] if arc_names else None

    for idx, samp in enumerate(samples):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

        # 1. PCA plot
        ax1.scatter(score[:-n_arcs, 0], score[:-n_arcs, 1])
        ax1.scatter(score[idx, 0], score[idx, 1], c="red")
        circle = plt.Circle((score[idx, 0], score[idx, 1]), 4, color="black", fill=False)
        ax1.add_patch(circle)

        if arc_names:
            ax1.scatter(arcs[:, 0], arcs[:, 1], marker="P", s=40)
            for i, txt in enumerate(arc_names):
                ax1.annotate(txt, (arcs[i, 0], arcs[i, 1]), fontsize=16)

        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_aspect("equal")

        # 2. Topic bar plot
        y = omega.loc[samp]
        x = [f"{t}" for t in omega.columns]
        ax2.bar(x, y)
        ax2.tick_params(labelsize=16)
        ax2.set_aspect("auto")

        # 3. Topic simplex 2D
        ax3.scatter(omega[x_axis], omega[y_axis])
        ax3.scatter(omega.loc[samp][x_axis], omega.loc[samp][y_axis], c="red")
        c2 = plt.Circle((omega.loc[samp][x_axis], omega.loc[samp][y_axis]),
                        0.02, color="black", fill=False)
        ax3.add_patch(c2)
        ax3.set_xlabel(f"p({x_axis})")
        ax3.set_ylabel(f"p({y_axis})")
        ax3.set_aspect("equal")

        fig.tight_layout()

        outfile = Path(filepath) / f"{samp}.{filetype}"
        plt.savefig(outfile)
        plt.close()






def plot_simplex_with_pca_single(
    model: dict,
    score: np.ndarray,
    sample: str,
    x_axis: str,
    y_axis: str,
    arc_names: Optional[List[str]] = None,
    save: bool = False,
    outpath: Optional[str] = None,
    filetype: str = "png",
    fix_text: bool = False
):
    """
    Plot PCA + topic histogram + simplex position for a single sample.

    Parameters
    ----------
    model : dict
        Output of load_topic_model()
    score : np.ndarray
        PCA coordinates (samples × 2)
    sample : str
        Name of the sample to highlight
    x_axis, y_axis : str
        Topic names (e.g. "k1", "k3")
    arc_names : list of str
        Names of the archetypes (optional)
    save : bool
        Whether to save the figure
    outpath : str
        Path for saving (required if save=True)
    filetype : str
        File extension (png/svg/jpeg)
    fix_text : bool
        Whether to manually adjust arc label placement
    """

    omega = model["omega"]

    # --- Validation --- #
    if sample not in omega.index:
        raise ValueError(f"Sample '{sample}' not found in omega matrix.")

    idx = list(omega.index).index(sample)

    # Prepare arcs if provided
    n_arcs = len(arc_names) if arc_names else 0
    arcs = score[-n_arcs:] if arc_names else None

    # --- Create 3-panel figure --- #
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    # 1. PCA plot
    if n_arcs > 0:
        ax1.scatter(score[:-n_arcs, 0], score[:-n_arcs, 1])
    else:
        ax1.scatter(score[:, 0], score[:, 1])

    ax1.scatter(score[idx, 0], score[idx, 1], c="red")
    circle = plt.Circle((score[idx, 0], score[idx, 1]), 4, color="black", fill=False)
    ax1.add_patch(circle)

    if arc_names:
        ax1.scatter(arcs[:, 0], arcs[:, 1], marker="P", s=40)
        for i, txt in enumerate(arc_names):
            if fix_text:
                # Example manual offsets (adjust per your dataset)
                ax1.annotate(txt, (arcs[i, 0] + 2, arcs[i, 1] - 2), fontsize=16)
            else:
                ax1.annotate(txt, (arcs[i, 0], arcs[i, 1]), fontsize=16)

    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_aspect("equal")

    # 2. Bar plot of topic composition for this sample
    yvals = omega.loc[sample]
    xvals = omega.columns
    ax2.bar(xvals, yvals)
    ax2.tick_params(labelsize=16)
    ax2.set_title(f"Topic mixture for {sample}")

    # 3. Topic simplex scatter
    ax3.scatter(omega[x_axis], omega[y_axis])
    ax3.scatter(omega.loc[sample, x_axis], omega.loc[sample, y_axis], c="red")
    c2 = plt.Circle((omega.loc[sample, x_axis], omega.loc[sample, y_axis]),
                    0.02, color="black", fill=False)
    ax3.add_patch(c2)
    ax3.set_xlabel(f"p({x_axis})")
    ax3.set_ylabel(f"p({y_axis})")
    ax3.set_aspect("equal")

    fig.tight_layout()

    # --- Save or Show --- #
    if save:
        if outpath is None:
            raise ValueError("outpath must be provided when save=True.")
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()







