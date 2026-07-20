# -*- coding: utf-8 -*-
"""
===============================================================================
SPaCe-MC Visualization and Figure Generation
===============================================================================

Description
-----------
This script generates publication-quality visualizations of SPaCe-MC analysis
results. It loads Monte Carlo null distributions and observed statistics,
constructs ridge (joy) plots for comparison across experimental conditions,
and visualizes distributions of simulation parameters.

Main workflow
-------------
1. Load SPaCe-MC analysis results.
2. Extract null distributions and observed statistics.
3. Construct ridge (joy) plots of null distributions.
4. Overlay observed values and confidence intervals.
5. Visualize distributions of simulation parameters
   (Hurst exponent, diffusion coefficient, track duration, and particle count).

Outputs
-------
- Ridge (joy) plots of Monte Carlo null distributions.
- Histograms of simulation parameter distributions.
- Publication-quality figures for manuscript preparation.

Dependencies
------------
numpy
pandas
matplotlib

===============================================================================
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

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


@dataclass
class Configuration():
    
    path: str = r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure4\points_ABM10_points_BM5"
    output_path: str | None = None
    
    folders: list = field(default_factory=lambda: [
        # 'Control',
        'NaArO2',
        'RK33',
        'Combo'
    ])
    
    labels: list = field(default_factory=lambda: [
        # 'Control',
        r"NaArO2$_2$",
        "RK-33",
        "Combo"
    ])

    null_match_ABM: str = '_ABM_null_dit'
    null_match_BM: str = '_BM_null_dit'
    null_match_track_count: str = 'track_count_null'
    
    ABM_mode: bool = False
    BM_mode: bool = False
    track_count_mode: bool = False
    
    save_mode: bool = False
    
    Da: str = 'Da_value'
    a: str = 'a_value'
    Deff: str = 'Deff_value'
    D: str = 'D_value'
    track_count: str = 'track_number'
    
    Da_name: str = r'$D_{\alpha}$ ($\mu m^2/sec^{\alpha}$)'
    a_name: str = r'$\alpha$'
    Deff_name: str = r'$D_{eff}$'
    D_name: str = r'$D$ ($\mu m^2/sec$)'
    track_count_name: str = 'Tracks per 100 frames (~50 ms)'
    
    name_library = {Da: Da_name,
                    a: a_name,
                    Deff: Deff_name,
                    D: D_name,
                    track_count: track_count_name}
    
    xlim_library = {Da: [None, None],
                    a: [None, None],
                    Deff: [None, None],
                    D: [None, None],
                    track_count: [None, None]}
    
    def __post_init__(self):

        n_modes = sum([
            self.ABM_mode,
            self.BM_mode,
            self.track_count_mode
        ])
    
        if n_modes != 1:
            raise ValueError(
                "Exactly one of ABM_mode, BM_mode, or track_count_mode must be True."
            )

        if self.ABM_mode:
            self.null_match = self.null_match_ABM

        elif self.BM_mode:
            self.null_match = self.null_match_BM
        
        elif self.track_count_mode:
            self.null_match = self.null_match_track_count
            
        # if self.save_mode:
        #     self.output_path = self.path
        # else:
        #     self.output_path = None

def plot_histogram(path, mode_H=False, mode_D=False, mode_L=False, mode_P=False, save_path = None):
    
    modes = [mode_H, mode_D, mode_L, mode_P]
    if sum(modes) != 1:
        raise ValueError("Only one mode can be True.")

    df = pd.read_csv(path)
    tracks = df.groupby(['group', 'particle'])
    data_hist = []

    if mode_H:
        for _, track_df in tracks:
            data_hist.append(track_df["H"].iloc[0])
            xlabel = 'Hurst parameter'

    elif mode_D:
        for _, track_df in tracks:
            data_hist.append(track_df["D"].iloc[0]*10**12)
            xlabel = 'diffusion coefficient'

    elif mode_L:
        for _, track_df in tracks:
            data_hist.append(track_df["L"].iloc[0])
            xlabel = 'track duration'

    elif mode_P:
        data_hist = df.groupby('group')['particle'].nunique().to_list()
        xlabel = 'particle number'
        
    
    plt.figure()

    plt.hist(
        data_hist,
        bins=30,
        edgecolor='black',
        alpha=0.8
    )

    plt.xlabel(xlabel)
    plt.ylabel("Count")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + f'\ground_{xlabel}.tiff', dpi=600, bbox_inches="tight")
    plt.show()

    return data_hist
        

def build_joyplot_data(cfg: Configuration, parameter: str):
    
    joyplot_data = {}
        
    for idx, folder in enumerate(cfg.folders): 
        
        print(folder)
        
        path_to_experiment = os.path.join(cfg.path, folder)

        files = [
            f for f in os.listdir(path_to_experiment)
            if f.endswith(".csv")
        ]
        
        for file in files:
        
            if cfg.null_match in file:
                #print(file)
                file_null = file
                path_to_null = os.path.join(path_to_experiment, file_null)
            
            if f'_{parameter}' in file:
                print(file)
                file_obs = file
                path_to_obs = os.path.join(path_to_experiment, file_obs)

        df_null = pd.read_csv(path_to_null) 
        df_obs = pd.read_csv(path_to_obs)
        
        null_distribution = df_null[parameter]
        CI_low = df_obs["CI_low"].iloc[0]
        CI_high = df_obs["CI_high"].iloc[0]
        
        obs = df_obs["T_obs"].iloc[0]
        # mean = df_obs["mean_perm"].iloc[0]
        
        print(cfg.labels[idx], df_obs["p_value"])
        
        joyplot_data[cfg.labels[idx]] = {
            'distribution': null_distribution,
            # 'mean': mean,
            'CI_low': CI_low,
            'CI_high': CI_high,
            'observation': obs}
        
    return joyplot_data

def plot_joyplot(joyplot_data, title=None, save_path=None, add_to_path = None, x_min = None, x_max = None, dpi=600):
    labels = list(joyplot_data.keys())
    n = len(labels)

    # Gather all x-values to define a shared axis range
    all_values = []
    for item in joyplot_data.values():
        all_values.extend(item["distribution"].tolist())
        all_values.extend([item["CI_low"], item["CI_high"], item["observation"]])

    all_values = np.asarray(all_values, dtype=float)
    if x_min is None or x_max is None:
        x_min = np.nanmin(all_values)
        x_max = np.nanmax(all_values)

    pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0

    fig, ax = plt.subplots()

    ridge_height = 0.8
    baseline_step = 1.0
    bins = 50

    for i, label in enumerate(labels):
        item = joyplot_data[label]
        dist = np.asarray(item["distribution"], dtype=float)
        dist = dist[np.isfinite(dist)]

        baseline = (n - 1 - i) * baseline_step

        counts, edges = np.histogram(dist, bins=bins, range=(x_min - pad, x_max + pad))
        if counts.max() > 0:
            counts = counts / counts.max() * ridge_height

        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]

        ax.bar(
            centers,
            counts,
            width=width,
            bottom=baseline,
            align="center",
            alpha=0.8,
            edgecolor="black",
            # color = 'gold',
            linewidth=0.5
        )

        obs = item["observation"]
        ci_low = item["CI_low"]
        ci_high = item["CI_high"]
        # mean = item["mean"]
        mean = np.mean(item["distribution"])
        
        ax.vlines(obs, baseline, baseline + ridge_height, colors="black", linestyles="-", linewidth=8)
        if np.isnan(obs):
            ax.vlines(mean, baseline, baseline + ridge_height, colors="black", linestyles="-", linewidth=8)
        
        ax.vlines(ci_low, baseline, baseline + ridge_height, colors='black', linestyles="--", linewidth=5)
        ax.vlines(ci_high, baseline, baseline + ridge_height, colors="black", linestyles="--", linewidth=5)

        ax.text(x_min - pad * 1.5, baseline + ridge_height * 0.7, label,
                va="center", ha="center", rotation=90, fontsize = 30)

    ax.set_yticks([])
    #ax.set_xlabel("Parameter value")
    if title:
        ax.set_title(title)



    # obs_handle = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="Observed value")
    # ci_handle = plt.Line2D([0], [0], color="red", linestyle=":", linewidth=1.4, label="CI low / high")
    # ax.legend(handles=[obs_handle, ci_handle], loc="upper right")

    plt.tight_layout()
    
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(-0.2, n * baseline_step + 0.2)
    
    print(joyplot_data.keys())

    if save_path:
        plt.savefig(save_path + f'\joyplot_{add_to_path}.tiff', dpi=dpi, bbox_inches="tight")

    plt.show()
    
if __name__ == "__main__":
            
    folders = ['Control_nucmask', 'NaArO2_nucmask', 'RK33_nucmask', 'Combo_nucmask']
    
    labels = ['Control', 'NaArO$_2$', 'RK-33', 'Combo']
    
    cfg = Configuration(path = r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure3",
        output_path = r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure3\joyplots_nucmask",
        folders=folders, labels = labels,   
        ABM_mode=True, BM_mode= False, track_count_mode = False, save_mode=True)
    
    parameter = cfg.Deff
    parameter_name = cfg.name_library[parameter]
    xmin = cfg.xlim_library[parameter][0]
    xmax = cfg.xlim_library[parameter][1]
    save_path = cfg.output_path

    joyplot_data = build_joyplot_data(cfg, parameter)
    
    plot_joyplot(
        joyplot_data,
        title=f"{parameter_name}",
        add_to_path=parameter,
        x_max= xmax,
        x_min = xmin,
        save_path=save_path  # or r"D:\...\joyplot.png"
    )
    
