"""
===============================================================================
SPaCe-MC Permutation Analysis for Single-Particle Tracking Data
===============================================================================

Description
-----------
This script performs Spatially Constrained Monte Carlo (SPaCe-MC) sampling to
generate random cytoplasmic control regions that mimic stress granule (SG)
geometry and location within individual cells. It associates single-particle
tracking (SPT) trajectories with either native SGs or randomly generated
control regions and exports the resulting trajectory datasets for downstream
diffusion analysis.

Main workflow
-------------
1. Load cell segmentation masks (cytoplasm, stress granules, nuclei).
2. Measure stress granule properties (number, area, intensity, cytoplasmic
   occupancy).
3. Generate random control ROIs matching SG number and average size while
   remaining inside the cytoplasm.
4. Load and preprocess SPT trajectories.
5. Classify trajectories according to their spatial relationship with SGs,
   cytoplasm, or control ROIs.
6. Repeat control ROI generation for multiple Monte Carlo permutations.
7. Export trajectory datasets for downstream Brownian or anomalous diffusion
   analyses.
8. Optionally compute diffusion parameters and summarize SG morphology.

Outputs
-------
- CSV files containing trajectories assigned to SGs or control ROIs.
- Stress granule morphological measurements.
- Optional diffusion parameter summaries.
- Optional trajectory visualization and statistical plots.

Required inputs
---------------
- Segmentation masks (SG, cytoplasm, optional nuclei)
- Filtered SPT trajectory CSV files
- Raw fluorescence images (for SG intensity measurements)

Dependencies
------------
numpy
pandas
matplotlib
tifffile
scikit-image

Custom modules:
    preprocessing
    postprocessing
    ABM_diffusion

===============================================================================
"""
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread 
import os
from skimage import measure
import pandas as pd
import glob
import re
from skimage.measure import label, regionprops_table

import sys
sys.path.append(r"D:\manuscripts\SG_GEM paper\Code_Python_SG_GEM_manuscript\GlobalLocalDiffusion\utils")
import preprocessing
import postprocessing
import ABM_diffusion
#preprocessing.CONFIG["cache_dir"] = r"D:\manuscripts\SG_GEM paper\cache" #command to change cache 

#------------------------------------CODE--------------------------------------------

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

#-----------------INPUT DATA--------------------------------
ExperDirectory = r"D:\manuscripts\SG_GEM paper\SPT"
SPT_folder = 'LK_SPT_V3'
output_folder = r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figureS1\v2"

experiments = [
    f for f in os.listdir(ExperDirectory)
    if os.path.isdir(os.path.join(ExperDirectory, f)) 
    and 'tifsg' not in f.lower() 
    and 'tifspt' not in f.lower()
]

save = False
show_plots = False
analyze_over_cells = False
analyze_single_particle = False
heterogeneity_check = False
static_error = False
permutation_test_params = True
permutation_test_const_roi =True

experimental_replicas= [
    ['01292026_0_15perDMSO', '10012025_0_15perDMSO'],
    ['01222025_0_15DMSO_500uMNaAr', '10012025_0_15DMSO_500uMNaAr'],
    ['01232026_0_15perDMSO_6uMRK_02', '01222026_0_15perDMSO_6uMRK', '10012025_0_15perDMSO_6uMRK'],
    ['04302026_0_15perDMSO_6uMRK_500uMNaAr', '10012025_0_15perDMSO_6uMRK_500uMNaAr']]

plot_names = ["Control", r'NaArO2$_2$', 'RK33', 'Combo']


region_mode = 'control_any' 
crop_track = True

'''
SG_any - tracks touched SG mask in any way
SG_in - tracks strictly in

SG_touch - tracks that only touched SG mask
crop_track - opportunity to make track cropped into the SG region

cyto - tracks in cytoplasm excluding SG mask
control_in - track in control local cytoplasm regions mimicking SGs, strictly in
control_any - in control local cytoplasm in any way
all - all tracks 
'''

um_per_pixel = (0.1465*40/100)
time_filter = 30

delta_x = 0.005 #time interval in sec for linear interpolation of collected tracks
tau = 0.01 #minimum lag time for MSD calculation, interpolation
time_scale = 1 # convertion to sec
min_trajectory_length = 10 #minimum trajectory length (in frames) for analyses

#theoretical diffusion coefficient in water to calculate effective diffusion coefficient 
T = 308 #K #variable!
nw = 0.7195*10**(-3) #Pa s #variable!

Kb = 1.38*10**(-23) #J/K (Pa*m3/K)
r = 20*10**(-9) #m
Dw = Kb*T/(6*np.pi*nw*r)*10**12 #um2/sec

permutation_n = 1000

results_total = []
results_Brownian = []
results_over_cells_ABM = []
results_over_cells_BM = []
results_SPT_BM = []
df_SGs_all = pd.DataFrame() #empty dataframe for SGs params


#--------------------FUNCTIONS--------------------------
def generate_original_masks():
    
    #SEGMENTATION
    masks = preprocessing.load_segmentation_tif(segm_file)
    SG_mask = masks["SG"]
    cyto_mask = masks["cyto"]
    
    if segm_file_nuc is not None:
        mask_nuc = preprocessing.load_segmentation_tif(segm_file_nuc, nuc_mask = True)
        nuclei_mask = mask_nuc["Nuclei"]
        cyto_mask = cyto_mask & (nuclei_mask)
    
    SG_mask = SG_mask & cyto_mask
    
    # update masks dict with final versions
    masks["SG"] = SG_mask
    masks["cyto"] = cyto_mask
    
    # FIND MASKS CONTROURS
    SGs_contours = measure.find_contours(SG_mask, level=0.5)
    cytoplasm_contours = measure.find_contours(cyto_mask, level=0.5)
    
    return masks, SG_mask, cyto_mask, SGs_contours, cytoplasm_contours

def generate_control_masks(masks, n_SGs, mean_area, random_mode = "circle"):
    SG_mask = masks["SG"]
    cyto_mask = masks["cyto"]
    control_mask = preprocessing.generate_controls(SG_mask, cyto_mask, mode = random_mode, input_area = mean_area, n_particles = n_SGs)
    masks["control"] = control_mask
    # FIND MASKS CONTROURS
    control_contours = measure.find_contours(control_mask, level=0.5)
    return masks, control_contours

def generate_flagged_tracks(track_file, masks, crop = False, mode = "SG"):
    #LOAD TRACKS AND PROCESS THEM 
    df = preprocessing.load_csv_tracks(track_file)
    
    #track interpolation/filter/locate to region
    preproc_tracks_data = preprocessing.preprocess_file_from_df(
        df,
        masks=masks,
        source_file=track_file,
        crop = crop,
        mode = mode
    )
    return preproc_tracks_data
    

def filter_flagged_tracks(data, trajectories, region_mode):
    
    #unpacking track data
    preproc_tracks, summary = data
        
    for track in preproc_tracks:
        flags = track.flags
        
        if region_mode == "SG_any" and flags["touches_SG_any"]:
            if "area_pixels" in flags:
                trajectories.append({"track_id": track.track_id, "area_pixels": flags["area_pixels"],"area_um2": flags["area_um2"],"trajectory": [track.t, track.x, track.y]})
            else: 
                trajectories.append({"track_id": track.track_id, "trajectory": [track.t, track.x, track.y]})
    
        elif region_mode == "SG_in" and flags["inside_SG_all"]:
            if "area_pixels" in flags:
                trajectories.append({"track_id": track.track_id, "area_pixels": flags["area_pixels"],"area_um2": flags["area_um2"],"trajectory": [track.t, track.x, track.y]})
            else: 
                trajectories.append({"track_id": track.track_id, "trajectory": [track.t, track.x, track.y]})
    
        elif region_mode == "SG_touch" and flags["touches_SG_any"] and not flags["inside_SG_all"]:
            trajectories.append({"track_id": track.track_id, "trajectory": [track.t, track.x, track.y]})
    
        elif region_mode == "cyto" and flags["inside_cyto_all"] and not flags["touches_SG_any"]:
            trajectories.append({"track_id": track.track_id, "trajectory": [track.t, track.x, track.y]})
            
        elif region_mode == "control_any" and flags["inside_control_any"] and not flags["touches_SG_any"]:
            if "area_pixels" in flags:
                trajectories.append({"track_id": track.track_id, "area_pixels": flags["area_pixels"],"area_um2": flags["area_um2"],"trajectory": [track.t, track.x, track.y]})
            else: 
                trajectories.append({"track_id": track.track_id, "trajectory": [track.t, track.x, track.y]})
        
        elif region_mode == "control_in" and flags["inside_control_all"] and not flags["touches_SG_any"]:
            if "area_pixels" in flags:
                trajectories.append({"track_id": track.track_id, "area_pixels": flags["area_pixels"],"area_um2": flags["area_um2"],"trajectory": [track.t, track.x, track.y]})
            else: 
                trajectories.append({"track_id": track.track_id, "trajectory": [track.t, track.x, track.y]})
        
        elif region_mode == "all":
            trajectories.append({"track_id": track.track_id, "trajectory": [track.t, track.x, track.y]})
            
    return trajectories


def trajectories_to_csv(trajectories, output_file):
    rows = []

    for traj in trajectories:
        
        file_name = traj["file_name"]
        replica_name = traj["replica_name"]
        permutation_index = traj["permutation_index"]
        
        particle_id = traj["track_id"]
        group_id = traj["group"]
        t, x, y = traj["trajectory"]
        
        area_pix = traj["area_pixels"]
        area_um2 = traj["area_um2"]

        # iterate over timepoints
        for ti, xi, yi in zip(t, x, y):
            rows.append({
                "group": group_id,
                "particle": particle_id,
                "time (s)": ti,
                "x (um)": xi,
                "y (um)": yi,
                "roi area (pix)": area_pix,
                "roi area (um2)": area_um2,
                "permutation_index": permutation_index,
                "file_name": file_name,
                "replica_name": replica_name
            })

    df = pd.DataFrame(rows)
    
    file_exists = os.path.isfile(output_file)
    df.to_csv(output_file, mode='a', index=False, header=not file_exists)
    
    return df

#---------------------MAIN------------------------------

for idx, replica in enumerate(experimental_replicas):
    
    print(idx, replica)
    
    trajectories = []
    #trajectories_control_permutaion = {i: [] for i in range(0, permutation_n)}
    trajectories_frequencies_permutaion = {i: [] for i in range(0, permutation_n)}
    trajectories_frequencies_SGs = []
    
    
    replica_cell_id = 0

    for experiment in replica:
    
        print(f'{experiment}')
        
        directory_segm = f'{ExperDirectory}/tifSG_{experiment}/'
        directory_tracks = f'{ExperDirectory}/tifSPT_{experiment}/'
        #tiff_files = [f for f in os.listdir(directory_segm) if f.endswith('.tif') and "Probabilities" not in f and "LoGsegm" not in f]
        tiff_files = [f for f in os.listdir(directory_tracks) if f.endswith('.tif')]
        
        for tiff_file in tiff_files:
            print(tiff_file)
                        
            if experiment == '01222026_0_15perDMSO_6uMRK' and tiff_file == '16_U2OS_ShL5_6uMRK_68min_1ugmldox_48h_gem_02.tif':
                continue
            
            file_name = os.path.splitext(tiff_file)[0] 
            file_name_core = tiff_file.replace("_gem.tif", "")
                    
            #extract time after SPT experiment started
            match = re.search(r'(\d+)min', tiff_file)
            if match:
                imaging_time = int(match.group(1))
            else:
                imaging_time = 0
                
            if imaging_time < time_filter:
                continue
           
            print(tiff_file)
            # assuming you already have file_name_core and directory_segm defined
            #pattern = os.path.join(directory_segm, f"{file_name_core}*_LoGsegm*.tif")
            pattern = os.path.join(directory_segm, f"{file_name_core}*_LoGsegm_V4.tif")
            segm_file = glob.glob(pattern)[0]
            SG_tif = segm_file.replace("_LoGsegm_V4", "")
            
            pattern_nuc = os.path.join(directory_segm, f"{file_name_core}*_NucMask.tif")
            nuc_files = glob.glob(pattern_nuc)
            if nuc_files:
                segm_file_nuc = nuc_files[0]
            else:
                segm_file_nuc = None
            
            track_file = f"{directory_tracks}/{SPT_folder}/{file_name_core}_gem_Statistics/{file_name_core}_gem_Position_Filtered.csv"
            track_tif = f"{directory_tracks}/{file_name_core}_gem.tif"
            
            if not os.path.exists(track_file):
                print('here')
                continue  # skip to next iteration of your loop
            
    #-----------------------COLLECT TRACKS OVER CYTOPLASM AND SGs------------------------
            #--------INPUT SEGMENTATION&ORIGINAL DATA------------       
            #ORIGINAL SG FILE
            img_SGs = imread(SG_tif)
            img_SGs = img_SGs.astype('float32')  # ensure float
            img_SGs_norm = (img_SGs - np.min(img_SGs)) / (np.max(img_SGs) - np.min(img_SGs))
            
            # SEGMENTATION ORIGINAL DATA
            masks_original, SG_mask, cyto_mask, SGs_contours, cytoplasm_contours = generate_original_masks()
            
            # Areas
            cyto_area = np.count_nonzero(cyto_mask)
            total_sg_area = np.count_nonzero(SG_mask)   # or SG_label.sum() won't work; use mask or props

            #SG parameters            
            SG_label = label(SG_mask)  # creates proper integer labels
            props = regionprops_table(
                SG_label,
                intensity_image=img_SGs,
                properties=["label", "area", "mean_intensity"])
            df_SGs = pd.DataFrame(props)
            df_SGs["cell_id"] = replica_cell_id
            df_SGs["replica_id"] = plot_names[idx]
            df_SGs["cyto_area"] = cyto_area
            df_SGs["total_sg_area"] = total_sg_area
            df_SGs["SG/cyto (%)"] = total_sg_area / cyto_area * 100
            
            df_SGs_all = pd.concat([df_SGs_all, df_SGs], ignore_index=True)
            replica_cell_id +=1
            
            n_SGs = SG_label.max()
            mean_area = np.mean(props["area"])
                  
            #FOR SPaCE cytoplasm analysis
            if permutation_test_const_roi:
                masks_original['SG'] = np.zeros_like(masks_original['SG'])
                n_SGs = 10
                mean_area = 500 
            
            if "control" in region_mode:
                sg_empty = not SG_mask.any()
                if sg_empty:
                    # Empty control mask too
                    masks = masks_original.copy()
                    masks["control"] = np.zeros_like(SG_mask)
                    control_contours = []
                
                else:
                    #LOAD TRACKS AND GENERATE FLAGGED TRAJECTORIES
                    masks, control_contours = generate_control_masks(masks_original, n_SGs, mean_area, random_mode = "circle") #random_mode = "sg_template"
            else: 
                masks = masks_original
            
            if "control" in region_mode:
                mode = "control"
            elif "SG" in region_mode:
                mode = "SG"
            else:
                mode = ''
            
            preproc_tracks_data = generate_flagged_tracks(track_file, masks, crop = crop_track, mode = mode)
            trajectories = filter_flagged_tracks(preproc_tracks_data, trajectories, region_mode)
            
            trajectories_cell = []
            trajectories_cell = filter_flagged_tracks(preproc_tracks_data, trajectories_cell, region_mode)
            
            #COLLECT TRAJECTORIES TO SAVE
            if False and len(trajectories_cell) != 0:
                
                for track in trajectories_cell:
                    track["group"] = replica_cell_id 
                    track["file_name"] = file_name_core
                    track["replica_name"] = experiment
                    track["permutation_index"] = None
                
                trajectories_to_csv(trajectories_cell, rf"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2_NucMask\Control_cropped_const_roi\{plot_names[idx]}_all.csv")
                
            
            #PERMITATION TEST FOR DIFFUSION PARAMETERS
            if permutation_test_params:
                if n_SGs != 0:
                    print(n_SGs)
                    for i in range(0, permutation_n):
                                                
                        print(plot_names[idx])
                        print(experiment)
                        print(tiff_file)
                        print(f"permutation number is {i}")
                        #GENERATE CONTROL MASKS   
                        masks, control_contours = generate_control_masks(masks_original, n_SGs, mean_area) #random_mode = "sg_template" random_mode = "circle"
                        #FLAG TRACKS
                        preproc_tracks_data = generate_flagged_tracks(track_file, masks, crop = crop_track, mode = "control")
                        #FILTER TRACKS
                        #trajectories_control_permutaion[i] = filter_flagged_tracks(preproc_tracks_data, trajectories_control_permutaion[i], region_mode)
                        
                        trajectories_control = []
                        trajectories_control = filter_flagged_tracks(preproc_tracks_data, trajectories_control, region_mode)
                        
                        #postprocessing.plot_trajectories(trajectories_control, control_contours, cytoplasm_contours, img_SGs_norm, "")
                        
                        if True and len(trajectories_control) != 0:
                            for track in trajectories_control:
                                track["group"] = replica_cell_id 
                                track["file_name"] = file_name_core
                                track["replica_name"] = experiment
                                track["permutation_index"] = i
                                                                
                            trajectories_to_csv(trajectories_control, rf"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Experiment_Tracks_V2_NucMask\{plot_names[idx]}_group{replica_cell_id}_NucMask_constroi_1000permutations.csv")
                        
                else:
                    control_contours = SGs_contours 
                    trajectories_control = []
                
            #---------------------PLOT TRAJECTORIES---------------------------
            if show_plots and not "control" in region_mode and not permutation_test_params:
                output_dir_fig = r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figure_1_ateDiffCoeff\R2abm0.9_100ms_R2BM0.7_50ms_v2_nucmask"
                postprocessing.plot_trajectories(trajectories_cell, SGs_contours, cytoplasm_contours, img_SGs_norm, '', save = False, save_folder = output_dir_fig, save_name = f'{file_name_core}_None')
            elif show_plots and "control" in region_mode and not permutation_test_params:
                postprocessing.plot_trajectories(trajectories_cell, control_contours, cytoplasm_contours, img_SGs_norm, "")
            elif show_plots and permutation_test_params:
                postprocessing.plot_trajectories(trajectories_control, control_contours, cytoplasm_contours, img_SGs_norm, "")
            
            #---------------DIFFUSION OVER CELL-----------------
            if analyze_over_cells and not permutation_test_params:
                if len(trajectories_cell) > 1:
                    
                    R2 = 0.9
                    
                    result_over_cell_ABM = ABM_diffusion.ABM_etaMSD(trajectories_cell, plot_names[idx], str(imaging_time), delta_x, tau, r2 = R2, Dw = Dw, color_number = idx, plot_numbers=[idx, idx + 1], points=10, loc_error= False, check_plot = False)
                    
                    if result_over_cell_ABM != None:
                        
                        if 'SG' in region_mode:
                            area_total = np.sum(SG_mask)
                        if region_mode == "control_in":
                            area_total = np.sum(masks['control'])
                        if region_mode == "cyto" or region_mode == "all":
                            area_total = np.sum(cyto_mask)
                        result_over_cell_ABM["area"] = area_total
                        
                        
                        results_over_cells_ABM.append(result_over_cell_ABM)
                    
                    R2 = 0.7
                    
                    result_over_cell_BM = ABM_diffusion.BM_etaMSD(trajectories_cell, plot_names[idx], imaging_time, delta_x, tau, r2 = R2, points = 5, color_number = idx, static_error=static_error, loc_error= False, check_plot = False)
                    
                    if result_over_cell_BM != None:
                        results_over_cells_BM.append(result_over_cell_BM)
                    
            #---------------DIFFUSION SINGLE PARTICLE-----------------
            #trajectories_parameters_ABM = []
            trajectories_parameters_BM = []
            save_SPT = False
            if analyze_single_particle and not permutation_test_params:
                #ABM_diffusion.ABM_taMSD(trajectories_cell, trajectories_parameters_ABM, delta_x, tau, 0.8,  Dw, f'{imaging_time} min')
                ABM_diffusion.ABM_taMSD(trajectories_cell, trajectories_parameters_BM, delta_x, tau, 0.8,  Dw, f'{imaging_time} min', mode = 'BM')
                df_SPT_BM = pd.DataFrame(trajectories_parameters_BM)
                
                if save_SPT:
                    # df_SPT_ABM = pd.DataFrame(trajectories_parameters_ABM)
                    # df_SPT_ABM.to_csv(output_folder+f'/SPT_ABM_{experiment}_{time_filter}min_ROI{region_mode}.csv', index = False)
                    
                    df_SPT_BM.to_csv(output_folder+f'/SPT_BM_{experiment}_{time_filter}min_ROI{region_mode}.csv', index = False)
                
                
                median_SPT_D = df_SPT_BM['D_value'].median()
                
                result_SPT = {
                    'Plot_Name': plot_names[idx],
                    'Biol_Repeat': idx,
                    'D_value': median_SPT_D
                }
                
                results_SPT_BM.append(result_SPT)
                
        
 

#-----------------DIFFUSION PARAMETERS RESULTS OVER CELLS------------------------   
if analyze_over_cells:   
    
    save_over_cells = True
               
    res_df_over_cells_ABM = pd.DataFrame(results_over_cells_ABM)
    res_df_over_cells_ABM = res_df_over_cells_ABM.iloc[0:] 
    
    if False:
        res_df_over_cells_ABM.to_csv(output_folder+f'/taeExperiment_ABM_{time_filter}min_ROI{region_mode}.csv', index = False)
        
    postprocessing.plot_over_cells_diff_params(res_df_over_cells_ABM, 'Plot_Name', 'Da_value', 'Da_sigma', 'Da vs Time', 'Time of imaging (min)', 'Da')
    postprocessing.plot_over_cells_diff_params(res_df_over_cells_ABM, 'Plot_Name', 'a_value', 'a_sigma', 'a vs Time', 'Time of imaging (min)', 'a')
    postprocessing.plot_over_cells_diff_params(res_df_over_cells_ABM, 'Plot_Name', 'Deff_value', 'Deff_sigma', 'Deff vs Time', 'Time of imaging (min)', 'Deff')
    
    plt.figure(figsize=(20,6))

    for exp, group in res_df_over_cells_ABM.groupby('Experiment'):
        
        group = group.sort_values('Plot_Name').reset_index(drop=True)
        
        # control = first row of this experiment
        a_ctrl = group.loc[0, 'a_value']
        sigma_ctrl = group.loc[0, 'a_sigma']
        
        # normalize
        a_norm = group['a_value'] / a_ctrl
        
        sigma_norm = a_norm * np.sqrt(
            (group['a_sigma'] / group['a_value'])**2 +
            (sigma_ctrl / a_ctrl)**2
        )
        
        # plot
        plt.errorbar(group['Plot_Name'],
                     a_norm,
                     yerr=sigma_norm,
                     marker='o',
                     linestyle='-',
                     capsize=4,
                     label=exp)
    
    plt.axhline(1, linestyle='--')
    plt.xlabel('Time of imaging (min)')
    plt.ylabel('a / a_control')
    plt.title('Normalized a vs Time')
    plt.legend(loc='center left',
               bbox_to_anchor=(1.02, 0.5),
               borderaxespad=0,
               frameon=False)
    plt.ylim(0, 1.5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    #results_over_cells_BM = [x for x in results_over_cells_BM if x is not None]
    res_df_over_cells_BM = pd.DataFrame(results_over_cells_BM)
    res_df_over_cells_BM = res_df_over_cells_BM.iloc[0:] 

    experiments = (res_df_over_cells_BM["Experiment"].unique())
    
    print('D_value')
    postprocessing.mannwhitneyu_test(res_df_over_cells_BM, experiments, 'D_value', save = save_over_cells, save_path= output_folder+'\over_cells_D')
    
    data = [res_df_over_cells_BM.loc[res_df_over_cells_BM["Experiment"] == r, "D_value"].values for r in experiments]
    postprocessing.boxplot_SGs_parameters(data, experiments, "Apparent Diffusion Coefficient", None, r'$D$ ($\mu m^2/sec$)', 2, [0, 1.3], save_over_cells, output_folder+f'/over_cells_apparent_D_{time_filter}min_ROI{region_mode}_withCombo')
    
    experiments = (res_df_over_cells_ABM["Experiment"].unique())
    
    data = [res_df_over_cells_ABM.loc[res_df_over_cells_ABM["Experiment"] == r, "Da_value"].values for r in experiments]
    postprocessing.boxplot_SGs_parameters(data, experiments, "Anomalous Diffusion Coefficient", None, r'$D_{\alpha}$ ($\mu m^2/sec^{\alpha}$)', 2, [0, 0.5], save_over_cells, output_folder+f'/over_cells_Da_{time_filter}min_ROI{region_mode}_withCombo')
    
    data = [res_df_over_cells_ABM.loc[res_df_over_cells_ABM["Experiment"] == r, "a_value"].values for r in experiments]
    postprocessing.boxplot_SGs_parameters(data, experiments, "Anomalous Diffusion Exponent", None, r'$\alpha$', 2, [0.2, 1.2], save_over_cells, output_folder+f'/over_cells_a_{time_filter}min_ROI{region_mode}_withCombo')
    
    data = [res_df_over_cells_ABM.loc[res_df_over_cells_ABM["Experiment"] == r, "Deff_value"].values for r in experiments]
    postprocessing.boxplot_SGs_parameters(data, experiments, "Effective Diffusion Coefficient", None, r'$D_{eff}$', 3, [0.01, 0.035], save_over_cells, output_folder+f'/over_cells_Deff_{time_filter}min_ROI{region_mode}_withCombo')
    
    print('a_value')
    postprocessing.mannwhitneyu_test(res_df_over_cells_ABM, experiments, 'a_value', save = save_over_cells, save_path= output_folder+'\over_cells_a')
    print('Deff_value')
    postprocessing.mannwhitneyu_test(res_df_over_cells_ABM, experiments, 'Deff_value', save = save_over_cells, save_path= output_folder+'\over_cells_Deff')
    print('Da_value')
    postprocessing.mannwhitneyu_test(res_df_over_cells_ABM, experiments, 'Da_value', save = save_over_cells, save_path= output_folder+'\over_cells_Da')
    
    
if analyze_single_particle and not permutation_test_params:
    
    results_SPT_BM = pd.DataFrame(results_SPT_BM)
    
    experiments = (results_SPT_BM["Plot_Name"].unique())
    data = [results_SPT_BM.loc[results_SPT_BM["Plot_Name"] == r, "D_value"].values for r in experiments]
    postprocessing.boxplot_SGs_parameters(data, experiments, "Apparent Diffusion Coefficient", "Experiment", r"D, $um^2$/sec", 2, True, output_folder+f'/SPT_median_apparent_D_{time_filter}min_ROI{region_mode}_withCombo')
    

#-----------------------OUTPUT SGs PARAMETERS--------------------------

output = r"D:\manuscripts\SG_GEM paper\Figures_Results\draft2_figureS2"
save_SG_param = False

df_SGs_all["area_um2"] = df_SGs_all["area"]*(um_per_pixel**2)
df = df_SGs_all.copy()
replicas = (df["replica_id"].unique())
# 1) SGs per cell (per-cell counts grouped by replica)
per_cell_counts = df.groupby(["replica_id", "cell_id"]).size().reset_index(name="n_particles")
data1 = [per_cell_counts.loc[per_cell_counts["replica_id"] == r, "n_particles"].values for r in replicas]
# 2) Pooled SG areas per replica
data2 = [df.loc[df["replica_id"] == r, "area_um2"].values for r in replicas]
# 3) Pooled SG mean intensities per replica
data3 = [df.loc[df["replica_id"] == r, "mean_intensity"].values for r in replicas]

postprocessing.boxplot_SGs_parameters(data1, replicas, "SGs per cell", ylabel = "SGs number", save = save_SG_param, save_path = output + '/SG_count_per_cell_v2')
postprocessing.boxplot_SGs_parameters(data2, replicas, "Pooled SG area", ylabel = r"SG area ($\mu m^2$)", round_value=2, save = save_SG_param, save_path = output + '/SG_pooled_area_v2')
postprocessing.boxplot_SGs_parameters(data3, replicas, "Pooled SG mean intensity by replica", ylabel = "Mean intensity")

per_cell_median_area = (df.groupby(["replica_id", "cell_id"])["area_um2"].median().reset_index(name="median_area_um2"))
data4 = [per_cell_median_area.loc[per_cell_median_area["replica_id"] == r,"median_area_um2"].values for r in replicas]
postprocessing.boxplot_SGs_parameters(data4, replicas, "SG area per cell", ylabel = r"SG area ($\mu m^2$)", round_value = 2, ylim = [0,2], save = save_SG_param, save_path = output + '/SG_area_per_cell_v2')

postprocessing.mannwhitneyu_rank_biserial(data1, replicas, compare_all_pairs = True, save = save_SG_param, save_path = output + '/SG_count_per_cell.csv')
postprocessing.mannwhitneyu_rank_biserial(data4, replicas, compare_all_pairs = True, save = save_SG_param, save_path = output + '/SG_area_per_cell.csv')

# One row per cell
df_cells = df_SGs_all[["replica_id", "cell_id", "SG/cyto (%)"]].drop_duplicates()
data5 = (df_cells.groupby("replica_id")["SG/cyto (%)"].apply(list).tolist())
postprocessing.boxplot_SGs_parameters(data5, replicas, "SG/cyto (%)", ylabel = "%", round_value = 1, ylim = [0,10], save = save_SG_param, save_path = output + '/SG_percentage_over_cells_v2')
postprocessing.mannwhitneyu_rank_biserial(data5, replicas, compare_all_pairs = True, save = save_SG_param, save_path = output + '/SG_percentage_over_cells.csv')

