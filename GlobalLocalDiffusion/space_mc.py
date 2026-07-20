'''
"""
===============================================================================
SPaCe-MC Analysis and Statistical Testing
===============================================================================

Description
-----------
This script performs SPaCe-MC (Spatially Constrained Monte Carlo) analysis of
single-particle tracking (SPT) datasets from either experimental measurements
or simulated trajectories. It compares diffusion properties measured within
stress granules (SGs) against null distributions generated from randomly
sampled cytoplasmic control regions.

The script also provides utilities for trajectory pooling, track statistics,
non-Gaussian displacement analysis, and benchmarking of simulated datasets.

Main workflow
-------------
1. Load experimental or simulated trajectory datasets.
2. Pool trajectories by cell or Monte Carlo permutation.
3. Calculate anomalous (ABM) and Brownian (BM) diffusion parameters.
4. Perform SPaCe-MC permutation tests to determine statistical significance.
5. Quantify track counts and trajectory durations.
6. Calculate the non-Gaussian parameter (α₂) from displacement distributions.
7. Generate summary statistics, significance tests, and publication-quality
   figures.

Analysis modules
----------------
- SPaCe-MC analysis of experimental data
- SPaCe-MC analysis of simulated datasets
- Brownian and anomalous diffusion fitting
- Track count and trajectory length statistics
- Non-Gaussian displacement analysis

Outputs
-------
- Monte Carlo null distributions
- Diffusion parameter summaries
- Track statistics
- Statistical comparisons
- Publication-quality figures and tables

Dependencies
------------
numpy
pandas
matplotlib

Custom modules:
    ABM_diffusion
    postprocessing
    space_mc_functions

===============================================================================
"""
'''


from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from dataclasses import dataclass, field
import re 

import sys
sys.path.append(r"D:\manuscripts\SG_GEM paper\Code_Python_SG_GEM_manuscript\GlobalLocalDiffusion\utils")
import postprocessing
import ABM_diffusion
import space_mc_functions



plt.rcParams['figure.figsize'] = [16, 10]  # Adjust based on your needs
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


#----------------CONFIGURATIONS-------------------

@dataclass
class MainConfig:
    # input_folder: str = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2"
    
    RUN_SPACEMC: bool = False
    RUN_SPACEMC_SIMULATIONS: bool = False
    RUN_TRACK_STATS: bool = False
    
    
    RUN_NONGAUSSIAN: bool = False
    
    
    RUN_SAVE_SIMULATED_SG_PARAMS: bool = False
    
    #PARAMETERS
    delta_x: float = 0.005 #time interval in sec for linear interpolation of collected tracks
    tau: float = 0.01 #minimum lag time for MSD calculation, interpolation
    min_trajectory_length: int = 10 #minimum trajectory length (in frames) for analyses
        
    um_per_pixel: float = (0.1465*40/100)
    
    #theoretical diffusion coefficient in water
    T: float = 308 #K #variable!
    nw: float = 0.7195*10**(-3) #Pa s #variable!
    
    Kb: float = 1.38*10**(-23) #J/K (Pa*m3/K)
    r: float = 20*10**(-9) #m
    Dw: float = Kb*T/(6*np.pi*nw*r)*10**12 #um2/sec
    


@dataclass
class SpaceMcConfig:
    
    input_folder: str =  r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2" 
    output_folder: str =  r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure3"
    
    experiments: list = field(default_factory=lambda: [
        'Control_cropped_const_roi',
        'NaArO2_cropped_const_roi',
        'RK33_cropped_const_roi',
        'Combo_cropped_const_roi'
    ])

    plot_names: list = field(default_factory=lambda: [
        'Control',
        r"NaArO2$_2$",
        "RK-33",
        "Combo"
    ])

    mode_ABM: bool = True
    mode_BM: bool = True
    mode_count: bool = True
    
    mode_observation: bool = False

    R2_ABM: float = 0.9
    R2_BM: float = 0

    points_ABM: int = 10
    points_BM: int = 5

    save_mode: bool = False
         
    # permutation_n: int = 1000
    # fixed_track_frq: int = True
    
    # track_frq: int = 250

    # tr_frq_Q1: int = 500
    # tr_frq_median: int = 1000
    # tr_frq_Q3: int = 1500
    

@dataclass
class TrackStatsConfig:
    
    input_folder: str = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2"
    output_folder: str = r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figureS2"
    
    experiments: list = field(default_factory=lambda: [
        'NaArO2_cropped',
        'RK33_cropped',
        'Combo_cropped'
    ])
    
    
    plot_names: list = field(default_factory=lambda: [
        # 'Control',
        r"NaArO2$_2$",
        "RK-33",
        "Combo"
    ])

    mode_csv: bool = False
    mode_permutations: bool = False
    save: bool = False
    
    @property
    def save_path(self):
        return f"{self.output_folder}/v2_permutations"
   
@dataclass
class NonGaussConfig: 
    
    #input_folder: str =  r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2_NucMask" 
    input_folder: str =  r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2\Parameter_Variability\simulation" 
    output_folder: str =  r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure3" #r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure4\nongaussian_figures"
    
    # experiments: list = field(default_factory=lambda: [
    #     'Control_all.csv',
    #     'NaArO2$_2$_all.csv',
    #     'RK33_all.csv',
    #     'Combo_all.csv'
    # ])
    
    experiments: list = field(default_factory=lambda: [
        'fixed_H0.25_pn1000_l75_D.csv',
        'fixed_H0.35_pn1000_l75_D.csv',
        'fixed_H0.35_pn1000_l75_Dvar.csv',
        'fixed_Hvar_pn1000_l75_D.csv'
    ])
    
           
    # plot_names: list = field(default_factory=lambda: [
    #     'Control',
    #     r"NaArO2$_2$",
    #     "RK-33",
    #     "Combo"
    # ])
    
    plot_names: list = field(default_factory=lambda: [
        r'$\alpha = 0.5$',
        r'$\alpha = 0.7$',
        "D = [0.22, 0.67]",
        r'$\alpha = [0.5, 0.7]$'
    ])
    
    
    save_mode: bool = False


#-------------------FUNCTIONS---------------------

#Filter df
def filter_df_tracks(df, area_filter_um2 = 0, track_len_filter = 0):
    
    df = df[df['roi area (um2)'] > area_filter_um2]
    
    df = (df.groupby(['group', 'particle']).filter(lambda x: len(x) >= track_len_filter))
    
    return df
        

#COLLECT ALL TRACKS IN ONE SIMULATED CSV
def collect_pooled_tracks(df, interpolate=False, dt=None):

    track_list = []

    for (group_id, particle_id), sub_df in df.groupby(['group', 'particle']):

        # Ensure time ordering
        sub_df = sub_df.sort_values('time (s)')

        t = sub_df['time (s)'].to_numpy()
        x = sub_df['x (um)'].to_numpy()
        y = sub_df['y (um)'].to_numpy()

        if interpolate and len(t) > 1:

            # Determine interpolation timestep
            interp_dt = dt if dt is not None else np.median(np.diff(t))

            # Create uniform time grid
            t_new = np.arange(t[0], t[-1] + interp_dt/2, interp_dt)

            # Interpolate coordinates
            x = np.interp(t_new, t, x)
            y = np.interp(t_new, t, y)
            t = t_new

        track = {
            "track_id": int(particle_id),
            "trajectory": [t, x, y]
        }

        track_list.append(track)

    return track_list

#COLLECT TRACKS PER GROUP IN ONE SIMULATED CSV
def collect_pooled_tracks_per_group(df, 
                                    filter_roi_id: int = None,
                                    interpolate = False, dt = 0.005):
    
    track_library = defaultdict(list)
    
    for (group_id, particle_id), sub_df in df.groupby(['group', 'particle']):
                    
        t = sub_df['time (s)'].values
        x = sub_df['x (um)'].values
        y = sub_df['y (um)'].values
        
        if 'file_name' in sub_df:
            name = sub_df['file_name'].unique()[0]
            match = re.search(r'(\d+)min', name)
            minutes = int(match.group(1))
        else:
            minutes = None
            
        if interpolate and len(t) > 1:

            # Determine interpolation timestep
            interp_dt = dt if dt is not None else np.median(np.diff(t))

            # Create uniform time grid
            t_new = np.arange(t[0], t[-1] + interp_dt/2, interp_dt)

            # Interpolate coordinates
            x = np.interp(t_new, t, x)
            y = np.interp(t_new, t, y)
            t = t_new

        track = {"track_id": int(particle_id), "trajectory": [t, x, y], 'imaging_time': minutes}
        
        track_library[group_id].append(track)
        
        #print(group_id, particle_id, len(t))
    
    return track_library

#COLLECT PERMUTATIONS (RANDOM ROIS FROM GROUP) IN ONE SIMULATED CSV
def collect_pooled_tracks_permutation(df, permutation_range: int, track_count: int, track_fixed_mode: bool = True, 
                                      Q1: int | None = None, median: int | None = None, Q3: int | None = None,
                                      interpolate = False, dt = 0.005):
    
    print(track_fixed_mode)

    track_permutation_library = defaultdict(list)
    
    grouped = list(df.groupby(['group', 'particle'])) #list of tracks if df format
    
    # if not track_fixed_mode:
    #     track_hist = []
        
    for p in range(0, permutation_range):
        
        if not track_fixed_mode:
            track_count = int(np.random.triangular(left=Q1, mode=median, right=Q3))
            # track_hist.append(track_count)
        #print(track_count)
                
        selected = np.random.choice(len(grouped), track_count, replace=False) #list of random indexes from df
        
        for i in selected:
            (group_id, particle_id), sub_df = grouped[i]
            
            t = sub_df['time (s)'].values
            x = sub_df['x (um)'].values
            y = sub_df['y (um)'].values
            
            if interpolate and len(t) > 1:

                # Determine interpolation timestep
                interp_dt = dt if dt is not None else np.median(np.diff(t))

                # Create uniform time grid
                t_new = np.arange(t[0], t[-1] + interp_dt/2, interp_dt)

                # Interpolate coordinates
                x = np.interp(t_new, t, x)
                y = np.interp(t_new, t, y)
                t = t_new
            
            track = {"track_id": int(particle_id), "trajectory": [t, x, y]}
            
            track_permutation_library[p].append(track)
        
        plt.figure()
        
    # if not track_fixed_mode:
            
    #     plt.hist(
    #         track_hist,
    #         bins=30,
    #         edgecolor='black',
    #         alpha=0.8
    #     )
    
    #     plt.xlabel('track frequency')
    #     plt.ylabel("Count")
    
    #     plt.tight_layout()
    #     save_path = "D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figureS_spacemc"
    #     plt.savefig(save_path + '\ground_track frequency.tiff', dpi=600, bbox_inches="tight")
    #     plt.show()
    
    return track_permutation_library

def collect_pooled_tracks_experiment_permutation_csv(df):

    track_permutation_library = defaultdict(list)

    group_cols = ['group', 'particle', 'permutation_index']

    if 'roi area (pix)' in df.columns:
        group_cols.append('roi area (pix)')

    for keys, sub_df in df.groupby(group_cols):

        particle_id = sub_df['particle'].iloc[0]
        permutation_index = sub_df['permutation_index'].iloc[0]

        t = sub_df['time (s)'].values
        x = sub_df['x (um)'].values
        y = sub_df['y (um)'].values

        track = {
            "track_id": int(particle_id),
            "trajectory": [t, x, y]
        }

        track_permutation_library[permutation_index].append(track)

    return track_permutation_library

#Collection of track frequencies

def collect_pooled_tracks_counts(df):
    
    traj_counts = df.groupby('group')['particle'].nunique()
    traj_counts = traj_counts.tolist()

    return traj_counts

def collect_pooled_track_counts_experiment_permutation_csv(df):
    
    count_permutation_library = defaultdict(list)

    for (group_id, permutation_index), sub_df in df.groupby(['group', 'permutation_index']):
        
        traj_count = sub_df['particle'].nunique()
        
        count_permutation_library[permutation_index].append(traj_count)
    
    return count_permutation_library

#---------------------SPACE-MC-SIMULAITON------------------------------

def space_mc_simulation(
                        input_folder,
                        output_folder,
                        area_filter = 0, 
                        track_len_filter = 0,
                        mode_observation = False,
                        delta_x = MainConfig.delta_x, 
                        tau = MainConfig.tau,
                        R2_ABM = SpaceMcConfig.R2_ABM, 
                        R2_BM = SpaceMcConfig.R2_BM, 
                        points_ABM = SpaceMcConfig.points_ABM, 
                        points_BM = SpaceMcConfig.points_BM,
                        Dw = MainConfig.Dw,
                        save_mode = False):
    
    obs_folders = [f for f in os.listdir(input_folder)]   

    for folder in obs_folders:
        
        print(folder)
        
        output = os.path.join(output_folder, folder) 
        os.makedirs(output, exist_ok=True)
        
        folder_path = os.path.join(input_folder, folder)
        
        obs_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        labels = [os.path.splitext(f)[0] for f in obs_files]

        trajectories_permutation = defaultdict(list)
        for idx, obs_file in enumerate(obs_files):
            
            print(f"track_collection, simulation {idx}")
            
            # match_H = re.search(r'_H(\d+(?:\.\d+)?)', obs_file)
            # match_D = re.search(r'_D(\d+(?:\.\d+)?)', obs_file)
        
            # if not match_H or not match_D:
            #     raise ValueError(f"Could not extract H or D from filename: {obs_file}")
        
            # H = float(match_H.group(1))
            # D = float(match_D.group(1))
            
            obs_path = os.path.join(folder_path, obs_file)
            df = pd.read_csv(obs_path)
            
    
            # COLLECT DATA FOR BOXPLOT
            pooled_cyto_tracks_per_group = collect_pooled_tracks_per_group(df, interpolate=True)
            
            for group, tracks in pooled_cyto_tracks_per_group.items():
                
                trajectories_permutation[idx].extend(tracks)
                                                
        space_mc_functions.Monte_Carlo_permutation_test_params(None, trajectories_permutation, experiment_name = folder, idx = idx, 
                                                                delta_x = delta_x, tau = tau, R2 = R2_ABM, Dw = Dw, points = points_ABM,
                                                                mode = 'ABM',
                                                                save =  save_mode, output_dir=output, add_to_file_name='',
                                                                plot = False)
        
        space_mc_functions.Monte_Carlo_permutation_test_params(None, trajectories_permutation, experiment_name = folder, idx = idx, 
                                                                delta_x = delta_x, tau = tau, R2 = R2_BM, 
                                                                points = points_BM, mode = 'BM',
                                                                save =  save_mode, output_dir=output, add_to_file_name='', 
                                                                plot = False)

#---------------------SPACE-MC-EXPERIMENTAL-DATA------------------------------
        
    
def space_mc_test(experiments, labels, input_folder,
                  area_filter = 0, track_len_filter = 0,
                  mode_ABM = False, mode_BM = False, mode_count = False, 
                  mode_observation = True,
                  delta_x = MainConfig.delta_x, 
                  tau = MainConfig.tau,
                  R2_ABM = SpaceMcConfig.R2_ABM, 
                  R2_BM = SpaceMcConfig.R2_BM, 
                  points_ABM = SpaceMcConfig.points_ABM, 
                  points_BM = SpaceMcConfig.points_BM,
                  Dw = MainConfig.Dw,
                  save_mode = False, output_dir = ''):

    for idx, experiment in enumerate(experiments):
        
        print(experiment)
        
        if mode_observation:
            SGs_file = input_folder + f"/{experiment[1]}" 
            df_SGs = pd.read_csv(SGs_file)
            #df_SGs = filter_df_tracks(df_SGs, area_filter, track_len_filter)
            pooled_SGs_tracks = collect_pooled_tracks(df_SGs)
            
            permutations_path = os.path.join(input_folder, experiment[0])
        else: 
            permutations_path = os.path.join(input_folder, experiment)
            
        permutation_files = os.listdir(permutations_path)
        
        dfs = []
        
        print('Collect permutations in df')
        for permutation_file in permutation_files:
            cyto_file = os.path.join(permutations_path, permutation_file)
        
            df_cyto = pd.read_csv(cyto_file)
            #df_cyto = filter_df_tracks(df_cyto, area_filter, track_len_filter)                
            dfs.append(df_cyto)

        big_df_cyto = pd.concat(dfs, ignore_index=True)
        
        #ABM DIFFUSION PARAMETERS
        if mode_ABM:
            
            if mode_observation:
                results_ABM_SG = []
                result_ABM_SG = ABM_diffusion.ABM_etaMSD(pooled_SGs_tracks, labels[idx], labels[idx], delta_x =delta_x, tau = tau, r2 = R2_ABM, Dw = Dw, color_number = 0, static_error=False, check_plot = False, points = points_ABM, loc_error=False)
                results_ABM_SG.append(result_ABM_SG)
                results_ABM_SG = pd.DataFrame(results_ABM_SG)
            else: 
                results_ABM_SG = None

            print('Prepare permutations for space_mc ABM')
            trajectories_permutation = collect_pooled_tracks_experiment_permutation_csv(big_df_cyto)
            print('space_mc ABM')
            space_mc_functions.Monte_Carlo_permutation_test_params(results_ABM_SG, trajectories_permutation, experiment_name = labels[idx], idx = idx, 
                                                                   delta_x = delta_x, tau = tau, R2 = R2_ABM, Dw = Dw, points = points_ABM,
                                                                   mode = 'ABM',
                                                                   save =  save_mode, output_dir=output_dir, add_to_file_name=f"{labels[idx]}",
                                                                   plot = False)
        
        #BM Diffusion Parameters
        if mode_BM:
            
            if mode_observation:
                results_BM_SG = []
                result_BM_SG = ABM_diffusion.BM_etaMSD(pooled_SGs_tracks, labels[idx], labels[idx], delta_x = delta_x, tau = tau, r2 = R2_BM, check_plot = False, points = points_BM, loc_error=False)
                results_BM_SG.append(result_BM_SG)
                results_BM_SG = pd.DataFrame(results_BM_SG)
            else: 
                results_BM_SG = None
            
            if not mode_ABM:
                print('Prepare permutations for space_mc BM')
                trajectories_permutation = collect_pooled_tracks_experiment_permutation_csv(big_df_cyto)
            print('space_mc BM')
            space_mc_functions.Monte_Carlo_permutation_test_params(results_BM_SG, trajectories_permutation, experiment_name = labels[idx], idx = idx, 
                                                                   delta_x = delta_x, tau = tau, R2 = R2_BM, 
                                                                   points = points_BM, mode = 'BM',
                                                                   save =  save_mode, output_dir=output_dir, add_to_file_name=f"{labels[idx]}",
                                                                   plot = False)
        
        #TRACK COUNTS
        if mode_count:
            
            if mode_observation:
                SG_track_counts = collect_pooled_tracks_counts(df_SGs)
            else: 
                SG_track_counts = None

            print('Prepare permutations for space_mc count')
            track_counts_permutation = collect_pooled_track_counts_experiment_permutation_csv(big_df_cyto)
        
            print('space_mc counts')
            space_mc_functions.Monte_Carlo_permutation_test_freq(SG_track_counts, track_counts_permutation, exp =  labels[idx], x_lim=[0, 20], 
                                                                 save = save_mode, output_dir = output_dir,  add_to_file_name=f"{labels[idx]}")
        
#---------------------OTHER STATISTICS------------------------------
      
def non_Gaussian_parameter(experiments, labels, input_folder, 
                           delta_x = 0.005, tau = 0.01,
                           save_mode = False, output_folder = ''):
    
    from scipy.stats import norm
    
    min_step = int(tau/delta_x)
    
    print(min_step)
    
    def non_gaussian_parameter(displacements):
        displacements = np.asarray(displacements)
        m2 = np.mean(displacements**2)
        m4 = np.mean(displacements**4)
        
        if m2 == 0:
            return np.nan
        
        return m4 / (3 * m2**2) - 1
    
    displacements_over_experiments = []
    alpha_over_experiments = []
    
    #print(output_folder)
    
    for idx, experiment in enumerate(experiments):
        
        print(experiment)
        

        tracks_file = input_folder + f"/{experiment}" 
        print(tracks_file)
        df_tracks = pd.read_csv(tracks_file)
        
        pooled_tracks_group = collect_pooled_tracks_per_group(df_tracks)
        
        #----collect_displacements-----
        displacements_group = []
        alpha2_group = []
        #img_times_group = []
        
        for group, tracks in pooled_tracks_group.items():
            
            displacements = []
            
            for track in tracks:
                
                #img_time = track['imaging_time']
                                
                t = track["trajectory"][0]
                x = track["trajectory"][1]
                y = track["trajectory"][2]
                
                for i in range(0, len(t)-min_step, min_step):
                    
                    x_displacement = x[i+min_step] - x[i]
                    y_displacement = y[i+min_step] - y[i]
                    
                    displacements.append(x_displacement)
                    displacements.append(y_displacement)
                    
                    displacements_group.append(x_displacement)
                    displacements_group.append(y_displacement)
                        
            alpha2 = non_gaussian_parameter(displacements)
            
            #img_times_group.append(img_time)
            alpha2_group.append(alpha2)
            
        displacements_over_experiments.append(displacements_group)
        alpha_over_experiments.append(alpha2_group)
        
    
    postprocessing.boxplot_SGs_parameters(alpha_over_experiments, labels, 
                                          r'$\alpha_2$', round_value=3, ylim = [0, 0.25], 
                                          save=save_mode, save_path=output_folder + '/nongaussian_alpha2',
                                          plot_stats_text = False)
    postprocessing.mannwhitneyu_rank_biserial(alpha_over_experiments, labels, save=save_mode, save_path=output_folder + '/nongaussian_alpha2.csv')
    
    for idx, d in enumerate(displacements_over_experiments):
        d = np.asarray(d, dtype=float)
    
        # 1) Fit Gaussian to the signed data
        mu, sigma = norm.fit(d)
    
        # 2) Use absolute values for the log-log plot
        d_abs = np.abs(d)
        d_abs = d_abs[d_abs > 0]   # log-log cannot show zero
    
        # Histogram of positive displacements
        hist, bins = np.histogram(d_abs, bins=50, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
    
        # Folded Gaussian corresponding to the fitted Gaussian
        x_fit = np.linspace(d_abs.min(), d_abs.max(), 500)
        fit_abs = norm.pdf(x_fit, loc=mu, scale=sigma) + norm.pdf(-x_fit, loc=mu, scale=sigma)
    
        line, = plt.loglog(centers, hist, marker='o', linestyle=None, alpha = 0.9,
                   label=f'{labels[idx]} data')
        plt.loglog(x_fit, fit_abs, linestyle='--', lw=8, alpha = 0.5,
                   color=line.get_color(),
                   label=f'{labels[idx]} Gaussian fit')
    
    plt.xlabel('Displacement magnitude')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.xlim(0.1, 1)
    if save_mode:
        plt.savefig(output_folder + '/gaussian_fit.tif', dpi=600)
    plt.show()

def track_statistics(experiments, labels, input_folder,
                     mode_csv = True, mode_permutations = False,
                     save = False, save_path = ''):
    
    def collect_track_parameters(pooled_tracks_per_group: dict):
        
        track_counts_per_cell = []
        track_lengths_per_cell = []
        
        for group, pooled_tracks in pooled_tracks_per_group.items():
            
            track_lengths = []
            
            for track in pooled_tracks:
                
                t = track['trajectory'][0]
                track_l = (t[-1] - t[0]) * 1000
                track_lengths.append(track_l)
            
            track_counts_per_cell.append(len(pooled_tracks))
            track_lengths_per_cell.append(np.median(track_lengths))
        
        return track_counts_per_cell, track_lengths_per_cell
    
    experiment_tr_counts = []
    experiment_tr_lengths = []
    print(save_path + '_counts.csv')
    for idx, experiment in enumerate(experiments):
        
        if mode_permutations and mode_csv:
            
            print('Error: Conflicting modes')
            return 
        
        print(experiment)
        
        if mode_csv:
            print('Collect tracks')
            track_file = input_folder + f"/{experiment}"
            df_tracks= pd.read_csv(track_file)
            
            pooled_tracks = collect_pooled_tracks_per_group(df_tracks)
            counts, lengths = collect_track_parameters(pooled_tracks)
            
        if mode_permutations and not mode_csv:
               
            permutations_path = os.path.join(input_folder, experiment)
            permutation_files = os.listdir(permutations_path)
            
            dfs = []
            
            print('Collect tracks')
            for permutation_file in permutation_files:
                cyto_file = os.path.join(permutations_path, permutation_file)
            
                df_cyto = pd.read_csv(cyto_file)
                dfs.append(df_cyto)
                
            df_tracks = pd.concat(dfs, ignore_index=True)
            
            # Number of tracks per group for each permutation
            tracks_per_perm = (
                df_tracks.groupby(['group', 'permutation_index'])['particle']
                  .nunique()
            )
            
            # Median number of tracks across permutations for each group
            counts = (
                tracks_per_perm
                .groupby('group')
                .median()
                .tolist()
            )
            
            # Track duration (ms) for each particle
            track_durations = (
                df_tracks.groupby(['group', 'permutation_index', 'particle'])['time (s)']
                .agg(lambda t: (t.max() - t.min()) * 1000)
            )
            
            # Median track duration for each permutation
            median_durations_per_perm = (
                track_durations
                .groupby(['group', 'permutation_index'])
                .median()
            )
            
            # Median track duration across permutations for each group
            lengths = (
                median_durations_per_perm
                .groupby('group')
                .median()
                .tolist()
            )
        
        print("count parameters")
        
        sum_tracks = sum(counts)
        
        print(f'{labels[idx]}: {len(counts)} cells ')
        print(f'amount of tracks: {sum_tracks}')
        print()
        
        experiment_tr_counts.append(counts)
        experiment_tr_lengths.append(lengths)
        
    postprocessing.boxplot_SGs_parameters(experiment_tr_counts, labels, title = 'track count per cell over 400 frames', ylabel = 'track count', ylim = [0, 2000], save=save, save_path = save_path + '_counts_sum')
    postprocessing.boxplot_SGs_parameters(experiment_tr_lengths, labels, title = 'track length per cell over 400 frames', ylabel = 'track length (ms)', ylim = [40, 140], save=save, save_path = save_path + '_lengths')
   
    postprocessing.mannwhitneyu_rank_biserial(experiment_tr_counts, labels, compare_all_pairs = True, save = save, save_path = save_path + '_counts.csv')
    postprocessing.mannwhitneyu_rank_biserial(experiment_tr_lengths, labels, compare_all_pairs = True, save = save, save_path = save_path + '_lengths.csv')

if __name__ == "__main__":
    
    #-----------------INPUT DATA--------------------------------
    
    main_cfg = MainConfig(RUN_SPACEMC=False, RUN_NONGAUSSIAN=False, RUN_TRACK_STATS=True, 
                          RUN_SPACEMC_SIMULATIONS = False, 
                          RUN_SAVE_SIMULATED_SG_PARAMS = False)
    
    # space_cfg = SpaceMcConfig(input_folder=r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2_NucMask",
    #                           output_folder=r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure3",
    #                           experiments = [['NaArO2_cropped_NucMask', 'NaArO2$_2$_SGs_cropped.csv'],
    #                                          ['RK33_cropped_NucMask', 'RK33_SGs_cropped.csv'],
    #                                          ['Combo_cropped_NucMask', 'Combo_SGs_cropped.csv']],
    #                           plot_names = [r"NaArO2$_2$", "RK-33","Combo"],
    #                           mode_ABM=True,
    #                           mode_BM=True,
    #                           mode_count=False,
    #                           mode_observation=False,
    #                           save_mode=False)
    
    space_cfg = SpaceMcConfig(input_folder=r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2_NucMask",
                              output_folder=r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure3",
                              experiments = ['NaArO2_cropped_const_roi', 'RK33_cropped_const_roi', 'Combo_cropped_const_roi'],
                              plot_names = [r"NaArO2$_2$", "RK-33","Combo"],
                              mode_ABM=True,
                              mode_BM=True,
                              mode_count=True,
                              mode_observation=False,
                              save_mode=False)
    
    stats_cfg = TrackStatsConfig(input_folder=r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2_NucMask",
                                 output_folder = r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure_1_ateDiffCoeff\track_stast",
                                 experiments=[
                                  'NaArO2_cropped_NucMask',
                                  'RK33_cropped_NucMask',
                                  'Combo_cropped_NucMask'],
                                 
                                 mode_csv=False, 
                                 mode_permutations = True, 
                                 save = True)
    
    nongaus_cfg = NonGaussConfig(save_mode=False)
    
    #-----SPACE-MC EXPERIMENTAL DATA-----------
    if main_cfg.RUN_SPACEMC:          
        
        space_mc_test(
            space_cfg.experiments,
            space_cfg.plot_names,
            input_folder=space_cfg.input_folder,
            output_dir=space_cfg.output_folder,
            mode_ABM=space_cfg.mode_ABM,
            mode_BM=space_cfg.mode_BM,
            mode_count=space_cfg.mode_count,
            mode_observation = space_cfg.mode_observation,
            R2_ABM=space_cfg.R2_ABM,
            R2_BM=space_cfg.R2_BM,
            delta_x=main_cfg.delta_x,
            tau = main_cfg.tau,
            points_ABM=space_cfg.points_ABM,
            points_BM=space_cfg.points_BM,
            save_mode=space_cfg.save_mode,
        )
        
    #-----SPACE-MC SIMULATION DATA-----------
    if main_cfg.RUN_SPACEMC_SIMULATIONS:
        
        space_mc_simulation(
                            input_folder = space_cfg.input_folder,
                            output_folder=space_cfg.output_folder,
                            R2_ABM=space_cfg.R2_ABM,
                            R2_BM=space_cfg.R2_BM,
                            delta_x=main_cfg.delta_x,
                            tau = main_cfg.tau,
                            points_ABM=space_cfg.points_ABM,
                            points_BM=space_cfg.points_BM,
                            save_mode=space_cfg.save_mode)
        
    #------------Track Statistics---------                     
    if main_cfg.RUN_TRACK_STATS:
        
        track_statistics(
            stats_cfg.experiments,
            stats_cfg.plot_names,
            input_folder=stats_cfg.input_folder,
            mode_csv=stats_cfg.mode_csv,
            mode_permutations=stats_cfg.mode_permutations,
            save=stats_cfg.save,
            save_path=stats_cfg.save_path
        )
        
    if main_cfg.RUN_NONGAUSSIAN:
        
        non_Gaussian_parameter(
                nongaus_cfg.experiments,
                nongaus_cfg.plot_names,
                tau=main_cfg.tau,
                input_folder=nongaus_cfg.input_folder,
                save_mode=nongaus_cfg.save_mode,
                output_folder=nongaus_cfg.output_folder)

