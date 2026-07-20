# postanalysis.py
"""
Minimal postanalysis utilities for SPT tracks:
    
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread 
import os
from scipy.stats import shapiro, mannwhitneyu
from matplotlib.patches import Rectangle
from cycler import cycler

# ---------- Simple config (edit in one place) ----------
CONFIG = {
    "um_per_pixel": 0.1465 * 40 / 100, 
    "delta_x": 0.005,   # interpolation dt [s]
    "tau": 0.01,
    "min_trajectory_length": 10,  # points after interpolation
    "cache_dir": "./preproc_cache", #save in the preproc_cache in the working folder
}


OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000"   # black
]


plt.rcParams['figure.figsize'] = [16, 10]  # Adjust based on your needs
plt.rcParams['axes.prop_cycle'] = cycler(color=OKABE_ITO)
plt.rcParams.update({
    'font.family': 'Arial', 
    'font.weight': 'bold', 
    'axes.labelweight': 'bold', 
    'axes.titleweight': 'bold', 
    'font.size': 40,
    'axes.titlesize': 40,
    'axes.labelsize': 40,
    'legend.fontsize': 30,
    'xtick.labelsize': 30,  
    'ytick.labelsize': 30,
}) 


# ---------- Data structure ----------
@dataclass
class PreprocessedTrack:
    track_id: int
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    flags: Dict[str, bool]
    source_file: str

    #dictionary that can be saved as json file
    def to_jsonable(self):
        d = asdict(self)
        d["t"] = self.t.tolist()
        d["x"] = self.x.tolist()
        d["y"] = self.y.tolist()
        return d

# ---------- I/O ----------
def plot_trajectories(
    trajectories_plot: List[Dict],
    contours_set1: List[np.ndarray],
    contours_set2: List[np.ndarray],
    image_norm: np.ndarray,
    title: str = "",
    save: bool = False,
    save_folder = '',
    save_name = '',
    config: Dict = CONFIG,
    grid: Optional[np.ndarray] = None,
    grid_spacing: Optional[int] = None,
    grid_success: Optional[np.ndarray] = None):  
    
    x_new = []
    y_new = []
    
    for item in trajectories_plot:
        x_new += (item["trajectory"][1]/config["um_per_pixel"]).tolist()
        y_new += (item["trajectory"][2]/config["um_per_pixel"]).tolist()
     
    #--------FILTER TRACKS IN/OUT SG/CYTOPLASM------------  
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_new, y_new, lw=0.5, s = 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.imshow(image_norm, cmap='gray')
    
    # plot SGs in green
    for c in contours_set1:
        ax.plot(c[:, 1], c[:, 0], color='green', linewidth=2.5)
    
    # plot cytoplasm in blue
    for c in contours_set2:
        ax.plot(c[:, 1], c[:, 0], color='blue', linewidth=5)
    
    # simple legend
    ax.plot([], [], color='green', label='SGs')
    ax.plot([], [], color='blue', label='Cytoplasm')
    ax.set_title(title)
    #ax.legend(loc='upper right')
    
    plt.tight_layout()
    if save: 
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/{save_name}.png', dpi=300)
        
    if grid is not None:
        grid_px = grid / config["um_per_pixel"]
        grid_px = np.rint(grid_px).astype(int)
    
        if grid_success is None:
            grid_success = np.ones(len(grid_px), dtype=bool)
    
        if grid_spacing is not None:
            cell_size_px = int(round(grid_spacing / config["um_per_pixel"]))
            half = cell_size_px // 2
    
            for (x, y), ok in zip(grid_px, grid_success):
                if not ok:
                    continue
                rect = Rectangle(
                    (x - half, y - half),
                    cell_size_px, cell_size_px,
                    facecolor='magenta',
                    edgecolor='magenta',
                    alpha=0.35,
                    linewidth=1
                )
                ax.add_patch(rect)

        ax.scatter(grid_px[grid_success, 0], grid_px[grid_success, 1], s=5, color='magenta')
    plt.show()

def plot_grid_alpha(
    contours_sg,
    contours_cyto,
    grid_points,
    local_results,
    config,
    grid_spacing,
    save=False,
    save_folder='',
    save_name=''
):
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()

    ax.set_aspect('equal')
    ax.axis('off')

    for c in contours_sg:
        ax.plot(c[:, 1], c[:, 0], color='green', linewidth=5)

    for c in contours_cyto:
        ax.plot(c[:, 1], c[:, 0], color='blue', linewidth=5)

    grid_px = grid_points / config["um_per_pixel"]
    grid_px = np.rint(grid_px).astype(int)

    alpha_vals = np.array([r.alpha if r.success else np.nan for r in local_results])
    valid = np.isfinite(alpha_vals)

    cell_size_px = int(round(grid_spacing / config["um_per_pixel"]))
    half = cell_size_px // 2

    from matplotlib import cm, colors
    norm = colors.Normalize(vmin=np.nanmin(alpha_vals), vmax=np.nanmax(alpha_vals))
    cmap = cm.get_cmap('viridis')

    for (x, y), a in zip(grid_px, alpha_vals):
        if not np.isfinite(a):
            continue

        color = cmap(norm(a))

        rect = Rectangle(
            (x - half, y - half),
            cell_size_px,
            cell_size_px,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(rect)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'$\alpha$')
    
    ax.invert_yaxis()
    plt.tight_layout()

    if save:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/{save_name}.png', dpi=600)

    plt.show()

def sanity_check(df: pd.DataFrame, preproc_tracks: List[PreprocessedTrack]):
    preproc_ids_set = {track.track_id for track in preproc_tracks}
    df_ids_set = set(df["track ID"])
    
    removed_ids = preproc_ids_set - df_ids_set     # filtered out during analysis
    kept_ids = df_ids_set & preproc_ids_set         # survived preprocessing
    
    # 4️⃣ Print summary
    print("Total raw tracks:        ", len(df_ids_set))
    print("After preprocessing:     ", len(preproc_ids_set))
    print("Tracks removed:          ", len(removed_ids))
    print("Tracks kept:             ", len(kept_ids))
    
    if df_ids_set == preproc_ids_set:
        print("Same tracks")
    else:
        print("Different tracks")
    print()
    
def boxplot_SGs_parameters(data: list, labels: list, title: str, xlabel: str = None, ylabel: str = None, round_value: int = 0, ylim: list = None, save: bool = False, save_path = '', plot_stats_text: bool = True):
    plt.rcParams.update({'figure.figsize': (18, 12)})
    
    plt.figure()
    plt.boxplot(data, 
                labels=[str(r) for r in labels],
                # widths=0.6,
                boxprops=dict(linewidth=4),
                whiskerprops=dict(linewidth=4),
                capprops=dict(linewidth=4),
                medianprops=dict(linewidth=5),
                flierprops=dict(
                    marker='o',
                    markersize=8,
                    markeredgewidth=2
                ))
    
    # compute quartiles manually
    for i, values in enumerate(data, start=1):
        
        x = np.random.normal(i, 0.05, size=len(values))  # 0.05 = jitter width
        plt.scatter(x, values, s=200)
        
        if plot_stats_text:
            q1 = np.percentile(values, 25)
            median = np.percentile(values, 50)
            q3 = np.percentile(values, 75)
            
            offset = 0.1 * (max(values) - min(values))  # 10% of data range
            # place text slightly to the right of each box
            plt.text(i + 0.25, q3 + offset, f"Q3={q3:.{round_value}f}", fontsize=30)
            plt.text(i + 0.25, median, f"Med={median:.{round_value}f}", fontsize=30)
            plt.text(i + 0.25, q1 - offset, f"Q1={q1:.{round_value}f}", fontsize=30)
    
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])    
    plt.title(title)
    if save:
        plt.savefig(f'{save_path}.tif', dpi=600)
    plt.show()
    
def plot_global_diff_params(df: pd.DataFrame, y_value: str, y_error: str, title: str, ylim: list, save: bool = False, save_path: str = ''):
    
    x = np.arange(len(df))
    labels = df['Plot_Name'].astype(str).tolist()
    
    # Plot settings
    plt.rcParams.update({'figure.figsize': (10, 6)})

    #OUTPUT: PLOT Da Vs EXPERIMENT
    plt.figure()
    plt.errorbar(x, df[y_value], yerr=df[y_error], fmt='o', linestyle='-', linewidth=1.5, markersize=6, capsize=5)
    plt.xticks(x, labels, ha='left')
    plt.ylabel(title)
    plt.title(title)
    plt.ylim(ylim[0], ylim[1])
    plt.grid(True, zorder=1)
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}.tif', dpi=600)
    plt.show()


def plot_over_cells_diff_params(
    df: pd.DataFrame,
    x_value: str,
    y_value: str,
    y_error: str,
    title: str,
    x_label: str,
    y_label: str,
    save: bool = False,
    save_path: str = ''
):
    plt.rcParams.update({'figure.figsize': (20, 6)})
    fig, ax = plt.subplots()

    for exp, group in df.groupby('Experiment'):
        

        
        # if exp == "Control":
        #     col = 'orange'
        # else:
        #     col = 'blue'

        x = pd.to_numeric(group[x_value], errors='coerce')
        y = pd.to_numeric(group[y_value], errors='coerce')
        yerr = pd.to_numeric(group[y_error], errors='coerce')

        mask = x.notna() & y.notna() & yerr.notna()
        x = x[mask].to_numpy(dtype=float)
        y = y[mask].to_numpy(dtype=float)
        yerr = yerr[mask].to_numpy(dtype=float)

        err = ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker='o',
            linestyle='none',
            capsize=10,
            markersize = 20,
            alpha=0.5,
            label=exp,
            # color = col,
        )

        color = err[0].get_color()

        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = m * x_fit + b
            ax.plot(x_fit, y_fit, color=color, linewidth=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, frameon=False)
    ax.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(f'{save_path}.tif', dpi=600)

    plt.show() 

def cliffs_delta(dist1: list, dist2: list):
    """
    Calculate Cliff's Delta effect size between two distributions (non-parametric).
    :param dist1: First distribution
    :param dist2: Second distribution
    :return: Cliff's Delta
    """
    n1 = len(dist1)
    n2 = len(dist2)
    total_pairs = n1 * n2

    greater = 0
    smaller = 0

    for x in dist1:
        for y in dist2:
            if x > y:
                greater += 1
            elif x < y:
                smaller += 1

    # Cliff's Delta calculation
    delta = (greater - smaller) / total_pairs
    return delta

def mannwhitneyu_test(df: pd.DataFrame, x_value: str, y_value: str, save: bool = False, save_path: str = ''):
    
    rows = []
    
    for exp, group in df.groupby('Experiment'):    
        vals = group[y_value].dropna().values
        if vals.size > 0:
            rows.append(pd.DataFrame({
                'experiment': exp,
                y_value: vals
            }))
        
    
    for row in rows:
    
        vals = row[y_value].dropna().values
        exp_name = row['experiment'].iloc[0]
        
        if len(vals) < 3:
            print(exp_name, f"n={len(vals)} (too small for Shapiro)")
            continue
        
        stat_sh, p_value_sh = shapiro(vals)
        print(exp_name, p_value_sh)  # p < 0.05 -> non-normal
    
    # Run permutation test for all pairs of distributions
    num_conditions = len(rows)
    results = []
    
    for i in range(num_conditions):
        for j in range(i + 1, num_conditions):
            a = rows[i][y_value].dropna().values
            b = rows[j][y_value].dropna().values
            a_name = rows[i]['experiment'].iloc[0]
            b_name = rows[j]['experiment'].iloc[0]
            
            stat, p_value = mannwhitneyu(a, b, alternative='two-sided')
            
            # Calculate effect size (rank-biserial correlation)
            n1 = len(a)
            n2 = len(b)
            rank_biserial = (2 * stat) / (n1 * n2) - 1
            delta = cliffs_delta(a, b)
            
            if save: 
                results.append({
                    'a': a_name,
                    'b': b_name,
                    'n_a': n1,
                    'n_b': n2,
                    'u_stat': stat,
                    'p_value': p_value,
                    'rank_biserial': rank_biserial,
                    'cliffs_delta': delta
                })
            else: 
                print(a_name, b_name)
                print(f"Mann-Whitney U Test statistic: {stat}")
                print(f"P-value: {p_value}")
                print(f"rank_biseral: {rank_biserial}")
                print(f'Cliff delta: {delta} ')
                print()
    if save:
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{save_path}.csv', index=False)

from itertools import combinations

def mannwhitneyu_rank_biserial(
    data: list,
    labels: list,
    compare_all_pairs: bool = False,
    alternative: str = "two-sided",
    save: bool = False,
    save_path: str = None,
):

    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")

    clean_data = []
    for values in data:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        clean_data.append(arr)

    if compare_all_pairs:
        pairs = list(combinations(range(len(clean_data)), 2))
    else:
        pairs = [(0, i) for i in range(1, len(clean_data))]

    results = []

    for i, j in pairs:
        x = clean_data[i]
        y = clean_data[j]

        u_stat, p_val = mannwhitneyu(
            x, y,
            alternative=alternative,
            method="auto"
        )

        r_rb = (2 * u_stat) / (len(x) * len(y)) - 1

        results.append({
            "group_1": labels[i],
            "group_2": labels[j],
            "n1": len(x),
            "n2": len(y),
            "U": u_stat,
            "p_value": p_val,
            "rank_biserial": r_rb,
        })

    stats_df = pd.DataFrame(results)

    if save:
        stats_df.to_csv(save_path, index=False)

    return stats_df