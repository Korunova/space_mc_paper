# -*- coding: utf-8 -*-
"""
===============================================================================
SPaCe-MC Validation and Performance Assessment
===============================================================================

Description
-----------
This script evaluates the performance of the SPaCe-MC (Spatially Constrained
Monte Carlo) framework using simulated single-particle tracking datasets with
known diffusion parameters. It compares observed diffusion measurements to
Monte Carlo null distributions, computes statistical significance, and
quantifies the sensitivity and specificity of the SPaCe-MC method.

Main workflow
-------------
1. Load simulated SPaCe-MC analysis results.
2. Compare observed diffusion parameters against null distributions.
3. Calculate empirical p-values for each simulated dataset.
4. Visualize null distributions and observed statistics.
5. Evaluate validation metrics including:
   - False positive rate (FPR)
   - Specificity
   - False negative rate (FNR)
   - Statistical power
6. Generate summary figures for method validation.

Outputs
-------
- Validation tables containing empirical p-values.
- Histograms of null distributions with observed statistics.
- Boxplots of p-values across simulated conditions.
- Performance metrics (FPR, FNR, specificity, and power).

Dependencies
------------
numpy
pandas
matplotlib
seaborn

===============================================================================
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import os
import re

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


    
def space_mc_analysis_SG_simulation(path, value,
                                    mode_ABM = True, mode_BM = False,
                                    output_folder = None):

    
    folders = [
        f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))
    ]
    
    obs_folders = [f for f in folders if "null" not in f]
    null_folder = next(f for f in folders if "null" in f)
    
    # obs_folders = [f for f in os.listdir(path) if "null" not in f]
    # null_folder = [f for f in os.listdir(path) if "null" in f][0]
    
    mode_check = sum([mode_ABM, mode_BM])
    
    if mode_check > 1:
        raise ValueError("Two modes are turned on. Select only one.")
    
    if mode_check == 0:
        raise ValueError("No mode is turned on. Select one mode.")
    
    pattern = r"ABM" if mode_ABM else r"(?<!A)BM"
    
    null_folder_path = os.path.join(path, null_folder)
    
    null_path = [
        f for f in os.listdir(null_folder_path)
        if f.endswith(".csv")
        and re.search(pattern, f)
        and "null" in f
    ][0]
    
    input_null_path = os.path.join(null_folder_path, null_path)
    
    #print(input_null_path)
    
    df_null = pd.read_csv(input_null_path)

    T_perm = df_null[value].to_numpy()
    ci_low, ci_high = np.percentile(T_perm, [2.5, 97.5])
    T_mean = np.mean(T_perm)
    T_median = np.median(T_perm)
    
    #print(T_mean)
    
    for obs_folder in obs_folders:
        
        print(obs_folder)
        
        match_H = re.search(r'_H(\d+(?:\.\d+)?)', obs_folder)
        match_D = re.search(r'_D(\d+(?:\.\d+)?)', obs_folder)

        H = float(match_H.group(1))
        D = float(match_D.group(1))
        
        print(H)
        print(D)
        # print()

        obs_folder_path = os.path.join(path, obs_folder)
        
        obs_file = [
            f for f in os.listdir(obs_folder_path)
            if f.endswith(".csv") 
            and re.search(pattern, f)
        ][0]
        
        # print(obs_file)
        
        obs_path = os.path.join(obs_folder_path, obs_file)
        df_obs = pd.read_csv(obs_path)
        # print(df_obs.columns)
        
        T_obs_list = df_obs[value].to_numpy()
        
        for T_obs in T_obs_list:
            p = (np.sum(np.abs(T_perm - np.mean(T_perm)) >= np.abs(T_obs - np.mean(T_perm))) + 1) / (len(T_perm) + 1)
            percentile = np.mean(T_perm < T_obs) * 100
    
            result = pd.DataFrame([{
                "H": H,
                "a": H*2,
                "D (um2/sec)": D,
                "obs": T_obs,
                "p_value": p,
                "percentile": percentile,
                "ci_2.5": ci_low,
                "ci_97.5": ci_high,
                "T_perm_mean": T_mean,
                "T_perm_median": T_median,
                "SG_file": obs_file,
                "permutation_file": null_path}])
            
            if output_folder is not None:
                
                output = output_folder + f'/validation_{value}.csv'
                
                result.to_csv(
                    output,
                    mode='a',                    # append
                    header=not os.path.exists(output),  # write header only once
                    index=True
                )


    
def space_mc_visualization(null_folder,
                            obs_folder,
                            value,
                            obs_idx = None,
                            x_lim = None,
                            mode_ABM = True, mode_BM = False,
                            output_folder = None):

    
    mode_check = sum([mode_ABM, mode_BM])
    
    if mode_check > 1:
        raise ValueError("Two modes are turned on. Select only one.")
    
    if mode_check == 0:
        raise ValueError("No mode is turned on. Select one mode.")
        
    pattern = r"ABM" if mode_ABM else r"(?<!A)BM"
    
    null_folder_path = os.path.join(path, null_folder)
    
    null_path = [
        f for f in os.listdir(null_folder_path)
        if f.endswith(".csv")
        and re.search(pattern, f)
        and "null" in f
    ][0]
    
    input_null_path = os.path.join(null_folder_path, null_path)
    
    #print(input_null_path)
    
    df_null = pd.read_csv(input_null_path)
    
    T_perm = df_null[value].to_numpy()
    ci_low, ci_high = np.percentile(T_perm, [2.5, 97.5])
    T_mean = np.mean(T_perm)
    T_median = np.median(T_perm)
    
    #print(T_mean)
    
    match_H = re.search(r'_H(\d+(?:\.\d+)?)', obs_folder)
    match_D = re.search(r'_D(\d+(?:\.\d+)?)', obs_folder)

    H = float(match_H.group(1))
    D = float(match_D.group(1))
    
    print(H)
    print(D)
    # print()

    obs_folder_path = os.path.join(path, obs_folder)
    
    obs_file = [
        f for f in os.listdir(obs_folder_path)
        if f.endswith(".csv") 
        and re.search(pattern, f)
    ][0]
    
    # print(obs_file)
    
    obs_path = os.path.join(obs_folder_path, obs_file)
    df_obs = pd.read_csv(obs_path)
    # print(df_obs.columns)
    
    T_obs_list = df_obs[value].to_numpy()
    
    if obs_idx is None: 
        for T_obs in T_obs_list:
            
            p = (np.sum(np.abs(T_perm - np.mean(T_perm)) >= np.abs(T_obs - np.mean(T_perm))) + 1) / (len(T_perm) + 1)
            percentile = np.mean(T_perm < T_obs) * 100
            
            plt.hist(T_perm, bins=30, density=True, alpha=0.9, color="#FFD84D")
            plt.axvline(T_obs, color = "royalblue", linewidth = 5)
            plt.axvline(ci_low, linestyle='--', color = 'black', linewidth = 5)
            plt.axvline(ci_high, linestyle='--', color = 'black', linewidth = 5)
            plt.xlabel(f"T {value}")
            plt.ylabel("Density")
            if x_lim != None: 
                plt.xlim(x_lim[0], x_lim[1])
            if output_folder is not None:
                plt.savefig(f'{output_folder}/{value}_idx[obs_idx]_sim{(H*2):.2}_calc{T_obs:.2}_p_value_{p:.3}_histogram.tif', dpi=600, format='tif')
            plt.show()
            
    else: 
            
        T_obs = T_obs_list[obs_idx]
        
        p = (np.sum(np.abs(T_perm - np.mean(T_perm)) >= np.abs(T_obs - np.mean(T_perm))) + 1) / (len(T_perm) + 1)
        percentile = np.mean(T_perm < T_obs) * 100
        
        plt.hist(T_perm, bins=30, density=True, alpha=0.9, color="#FFD84D")
        plt.axvline(T_obs, color = "royalblue", linewidth = 5)
        plt.axvline(ci_low, linestyle='--', color = 'black', linewidth = 5)
        plt.axvline(ci_high, linestyle='--', color = 'black', linewidth = 5)
        plt.xlabel(f"T {value}")
        plt.ylabel("Density")
        if x_lim != None: 
            plt.xlim(x_lim[0], x_lim[1])
        if output_folder is not None:
            plt.savefig(f'{output_folder}/{value}_idx[obs_idx]_sim{(H*2):.2}_calc{T_obs:.2}_p_value_{p:.3}_histogram.tif', dpi=600, format='tif')
        plt.show()
        


def validation(path, 
               p_treshold = 0.05,
               x_label = None,
               alpha = True,
               output_folder = None):
    
    if alpha:
        value = 'a'
        treshold = 0.7
        round_v = 2
    else:
        value = 'D (um2/sec)'
        treshold = 0.4477
        round_v  = 4
    
    # Load data
    df = pd.read_csv(path)
    
    required_cols = {value, "p_value", "obs"}
    
    # ----------------------------
    # 1) Plot p-value vs a
    # ----------------------------
    
    
    df = df.dropna(subset=[value, "p_value", "obs"]).copy()
    
    plt.figure()
    
    order = sorted(df[value].unique())
        
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(
        data=df,
        x=value,
        y="p_value",
        order=order,
        showfliers=False,
        color="white",
        linewidth=2
    )
    
    sns.stripplot(
        data=df,
        x=value,
        y="p_value",
        palette=OKABE_ITO,
        order=order,
        alpha=0.9,
        jitter=0.2,
        size=6
    )
    
    plt.axhline(p_treshold, color="red", linestyle="--", linewidth=5, zorder=10)
    
    plt.xlabel(x_label)
    plt.ylabel("p-value")
    
    # Force nice labels
    plt.xticks(
        ticks=np.arange(len(order)),
        labels=[f"{x:.2f}" for x in order]
    )
    
    plt.tight_layout()
    if output_folder is not None:
        plt.savefig(
            f"{output_folder}\{x_label}.tiff",
            dpi=600,
            bbox_inches="tight"
        )
    plt.ylim(0, 1)
    plt.show()
    
    #--------------------------
    # 2) False Positive Rate And Specifity
    #The false positive rate (FPR) is the proportion of actual negative cases that are 
    #incorrectly classified as positive by a test or model.
    #------------------------
    
    df_FPR = df[df[value].round(round_v) == treshold]
        
    fp = (df_FPR["p_value"] < p_treshold).sum()
    tn = (df_FPR["p_value"] >= p_treshold).sum()
    
    fpr = fp / (fp + tn)
    
    specifity = 1 - fpr
    
    print(f"False positives (p<{p_treshold}): {fp}")
    print(f"True negatives (p>={p_treshold}): {tn}")
    print(f"False positive rate: {fpr:.4f} ({100*fpr:.2f}%)")
    
    # ----------------------------
    # 2) False negative Rate and Power
    # Definition: 
    # p_value > 0.05 when a = 0.7 - false positive: negative cases incorrectly predicted as positive
    # p_value < 0.05 when a < 0.7 - true negative: negative cases correctly predicted as negative
    # Compute rate within each a value below 0.7
    # ----------------------------
    
    
    df_FNR = df[df[value] < treshold].copy()
    
    for obs_value, group in df_FNR.groupby(value):
    
        fn = (group["p_value"] >= p_treshold).sum()
        tp = (group["p_value"] < p_treshold).sum()
    
        fnr = fn / (fn + tp)
        power = 1 - fnr
    
        print(
            f"{value} = {obs_value:.3f}: "
            f"FNR = {fnr:.4f} ({100*fnr:.2f}%), "
            f"power = {power:.4f} ({100*power:.2f}%, "
            f"FN = {fn}, TP = {tp}"
        )
        

if __name__ == "__main__":
    
    if False:
        input_folder = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2_space_mc_NucMask\Combo_results\H"
        value = "a_value"
        output_folder = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2_space_mc_NucMask\Combo_results\H"
        
        space_mc_analysis_SG_simulation(path = input_folder,
                                        value = value,
                                        mode_ABM = True,
                                        mode_BM = False, 
                                        output_folder=output_folder)
    
    if False:
        path = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2_space_mc_NucMask\NaArO2_results\H\validation_Deff_value.csv"
        output = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2_space_mc_NucMask\NaArO2_results\space_mc\H"
        validation(path,
                    x_label='a',
                    p_treshold = 0.001,
                    alpha = True,
                    output_folder=None)
    
    if False:
        path_null = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2_space_mc_opt\space_mc_results_H\fixed_null_H0.35_pnvar_lvar_D0.44771314081722874"
        path_obs = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2_space_mc_opt\space_mc_results_H\fixed_H0.35000000000000003_pnvar_lvar_D0.44771314081722874"
        output_folder = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2_space_mc_opt\space_mc_figures"
        space_mc_visualization(path_null,
                               path_obs,
                               'a_value',
                               obs_idx=0,
                               x_lim=[0.65, 0.9],
                               output_folder = output_folder)