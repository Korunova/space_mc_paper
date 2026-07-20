# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:37:18 2026

@author: KORUNOVA
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Optional
from collections import defaultdict

import sys
sys.path.append(r"D:\manuscripts\SG_GEM paper\Code_Python_SG_GEM_manuscript\diffusion_package")
import ABM_diffusion

T = 308 #K #variable!
nw = 0.7195*10**(-3) #Pa s #variable!
Kb = 1.38*10**(-23) #J/K (Pa*m3/K)
r = 20*10**(-9) #m
Dw = Kb*T/(6*np.pi*nw*r)*10**12 #um2/sec

def Monte_Carlo_permutation_test_params(observation_df: list, trajectories_permutation: dict, experiment_name: str, idx: int, 
                                        delta_x: Optional[int] = None, tau:int = 0.1, R2:int = 0.9, Dw: Optional[int] = None, 
                                        points: int = 10,
                                        mode = 'ABM', save = False, output_dir = '', add_to_file_name = '', plot: bool = False):
    
    def extract_value_data(result, value, experiment_name = experiment_name):
        res_c = pd.DataFrame(result)
        value_list = res_c[value].astype(float).tolist()
        value_permutation = value_list
        return value_permutation
    
    def plot_null_histogram(value, value_permutation, 
                            perm_in, perm_out,
                            track_in, track_out,
                            results_pd = None, x_lim = None, exp = experiment_name):
        T_perm = np.array(value_permutation)
        
        # 95% confidence interval
        ci_low, ci_high = np.percentile(T_perm, [2.5, 97.5])
        
        print(exp)
        print(f"95% CI (null) = [{ci_low:.4f}, {ci_high:.4f}]")
        
        
        if results_pd is not None:
            T_obs = results_pd.loc[results_pd["Plot_Name"] == exp, value]
            T_obs = float(T_obs)
            # p-value (two-sided)
            p = (np.sum(np.abs(T_perm - np.mean(T_perm)) >= np.abs(T_obs - np.mean(T_perm))) + 1) / (len(T_perm) + 1)
            #percentile effect size
            percentile = np.mean(T_perm < T_obs) * 100
            
            print(f"T_obs = {T_obs:.4f}")
            print(f"p-value = {p:.4f}")
            print(f"T_obs is at {percentile:.1f} percentile of null distribution")
        else: 
            T_obs = p = percentile = 'N/A'
                                
        # Гистограмма
        plt.hist(T_perm, bins=30, density=True, alpha=0.9, color="#FFD84D")
        if results_pd is not None:
            plt.axvline(T_obs, color = "royalblue", linewidth = 5)
        plt.axvline(ci_low, linestyle='--', color = 'black', linewidth = 5)
        plt.axvline(ci_high, linestyle='--', color = 'black', linewidth = 5)
        plt.xlabel(f"T {value}")
        plt.ylabel("Density")
        if x_lim != None: 
            plt.xlim(x_lim[0], x_lim[1])
        plt.title(exp)
        if save:
            plt.savefig(f'{output_dir}/{exp}_{value}_{add_to_file_name}.tif', dpi=600, format='tif')
        plt.show()
        
        if save:
            results_perm = {
                "Experiment": exp,
                "T_obs": T_obs,
                "p_value": p,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "percentile": percentile,
                "mean_perm": np.mean(T_perm),
                "std_perm": np.std(T_perm, ddof=1),
                "N_perm": len(T_perm),
                "permutations_input": perm_in,
                "track_input": track_in,
                "permutations_output": perm_out,
                "track_output": track_out}
            
            results_perm_df = pd.DataFrame([results_perm])
            results_perm_df.to_csv(f"{output_dir}/{exp}_{value}_{add_to_file_name}.csv",index=False)
    
    if mode == 'ABM':    
        
        result_output_c = []
        real_permutation_number = 0
        
        # trajectories_permutation_ABM = defaultdict(list)
        
        permutations_output = 0
        permutations_input = 0
        track_input = 0
        track_output = 0
        print('Permutations started...')
        for key, trajectories_c in trajectories_permutation.items():
            result = ABM_diffusion.ABM_etaMSD(trajectories_c, experiment_name, key, delta_x = delta_x, tau = tau, r2 = R2, Dw = Dw, points = points, color_number = idx, loc_error=False, check_plot = plot)
            
            permutations_input += 1
            track_input += len(trajectories_c)
            if result is None:
                #print(f"Permutation {key} returned None")
                continue
            
            permutations_output += 1
            track_output += len(trajectories_c)
            
            result_output_c.append(result)
            # trajectories_permutation_ABM[key].append(trajectories_c)
        
            print(f"Real permutation number = {real_permutation_number}")
            real_permutation_number += 1
            
        print(permutations_input, permutations_output)
        
        a_permutation = extract_value_data(result_output_c, 'a_value')
        Da_permutation = extract_value_data(result_output_c, 'Da_value')
        Deff_permutation = extract_value_data(result_output_c, 'Deff_value')
        
        if save:
            null_dist_df = pd.DataFrame(result_output_c)
            null_dist_df.to_csv(f'{output_dir}/{experiment_name}_ABM_null_dit.csv', index=False)
            
        plot_null_histogram('a_value', a_permutation, permutations_input, permutations_output, track_input, track_output, observation_df, [0, 1])
        plot_null_histogram('Da_value', Da_permutation, permutations_input, permutations_output, track_input, track_output, observation_df, [0, 0.8])
        plot_null_histogram('Deff_value', Deff_permutation, permutations_input, permutations_output, track_input, track_output, observation_df, [0.01, 0.03]) #[0.01, 0.03]

    if mode == 'BM':

        result_output_c = []
        
        real_permutation_number = 0
        permutations_output = 0
        permutations_input = 0
        track_input = 0
        track_output = 0
        
        print('Permutations started...')
        for key, trajectories_c in trajectories_permutation.items():
            
            result = ABM_diffusion.BM_etaMSD(trajectories_c, experiment_name, experiment_name, delta_x, tau, R2, points = points, color_number = idx, check_plot=plot)
            
            permutations_input += 1
            track_input += len(trajectories_c)
            if result is None:
                #print(f"Permutation {key} returned None")
                continue
            
            permutations_output += 1
            track_output += len(trajectories_c)
                
            result_output_c.append(result)
            
            print(f"Real permutation number = {real_permutation_number}")
            real_permutation_number += 1
            
        print(permutations_input, permutations_output)
        if save:
            null_dist_df = pd.DataFrame(result_output_c)
            null_dist_df.to_csv(f'{output_dir}/{experiment_name}_BM_null_dit.csv', index=False)
            
             
        D_permutation = extract_value_data(result_output_c, 'D_value')
        plot_null_histogram('D_value', D_permutation, permutations_input, permutations_output, track_input, track_output, observation_df, [0, 3])
        

def Monte_Carlo_permutation_test_freq(value, value_permutation, exp, x_lim=None, save = False, output_dir = '', add_to_file_name = ''):
    """
    Permutation test and histogram plot using MEDIAN (np.nanmedian) as the test statistic.
    Зависит от внешних переменных: plot_names, idx, trajectories_frequencies_SGs, ExperDirectory.
    """
    # null distribution: медианы по каждому permutation-значению (взятые с игнорированием NaN)
    T_perm = np.array([
        np.nanmedian(value_permutation[i]) / 400 * 100
        for i in value_permutation.keys()
    ])
    
    if save:
        null_dist_df = pd.DataFrame({"track_number": T_perm})
        null_dist_df.to_csv(f'{output_dir}/{exp}_track_count_null_dit.csv', index=False)
    

    # наблюдаемая статистика = медиана наблюдаемых значений
    ci_low, ci_high = np.percentile(T_perm, [2.5, 97.5])
    
    print(exp)
    print(f"95% CI (null) = [{ci_low:.4f}, {ci_high:.4f}]")
    if value is not None:
        T_obs = float(np.nanmedian(value)) / 400 * 100
        
        # центруем по медиане null-распределения и считаем p-value (двусторонний тест)
        center = np.nanmedian(T_perm)
        p = np.mean(np.abs(T_perm - center) >= np.abs(T_obs - center))
        percentile = np.mean(T_perm < T_obs) / 400 * 100
        
        print(f"T_obs (median) = {T_obs:.4f}")
        print(f"p-value = {p:.4f}")
        # процентиль наблюдаемой статистики внутри null
        print(f"T_obs is at {percentile:.1f} percentile of null distribution")
    else: 
        T_obs = p = percentile = 'N/A'
    
    # графики
    plt.hist(T_perm, density=True, alpha=1, color='yellow', label='null (perm medians)')

    # вертикальные линии: наблюдаемая медиана, медиана null, CI
    if value is not None:
        obs_clean = np.array(value)
        obs_clean = obs_clean[np.isfinite(obs_clean)] / 400 * 100
        #plt.hist(obs_clean, bins=30, density=True, alpha=0.5, label='observed values')
        plt.axvline(T_obs, linestyle='--', color='blue', label='T_obs (median)', linewidth = 5)
        #plt.axvline(center, linestyle='-.', color='green', label='median(null)')
    plt.axvline(ci_low, linestyle=':', color='black', linewidth = 5)
    plt.axvline(ci_high, linestyle=':', color='black', linewidth = 5)

    plt.xlabel("(track frequency) \n track count per region during 100 frames")
    plt.ylabel("Density")
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
    plt.title(exp)
    #plt.legend()

    if save:
        plt.savefig(f'{output_dir}/{exp}_frequencies_median{add_to_file_name}.tif', dpi=600, format='tif')
    plt.show()

    if save:
        # результаты (с медианой вместо mean)
        results_perm = {
            "Experiment": exp,
            "T_obs": T_obs,
            "p_value": p,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "percentile": percentile,
            "median_perm": np.median(T_perm),
            "std_perm": np.std(T_perm, ddof=1),
            "N_perm": len(T_perm)
        }
    
        results_perm_df = pd.DataFrame([results_perm])
        results_perm_df.to_csv(f'{output_dir}/{add_to_file_name}_track_number.csv', index=False) 