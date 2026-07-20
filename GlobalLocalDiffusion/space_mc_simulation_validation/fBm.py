# -*- coding: utf-8 -*-
"""
===============================================================================
Fractional Brownian Motion Trajectory Simulator
===============================================================================

Description
-----------
This script generates synthetic single-particle trajectories using fractional
Brownian motion (fBM) for benchmarking and validation of diffusion analysis.
Trajectory properties can be fixed or sampled from experimentally measured
distributions, including particle number, trajectory duration, Hurst exponent,
and diffusion coefficient.

The generated datasets reproduce the statistical characteristics of
experimentally measured trajectories within stress granules (SGs) and
SPaCe-MC control regions, enabling evaluation of diffusion parameter
estimation and Monte Carlo significance testing.

Main workflow
-------------
1. Configure simulation parameters.
2. Generate fBM trajectories using the Hosking algorithm.
3. Sample particle number and trajectory length from experimental
   distributions (optional).
4. Simulate trajectories for different Hurst exponents and diffusion
   coefficients.
5. Export simulated trajectories as CSV files.
6. Save metadata describing all simulation parameters.

Outputs
-------
- CSV files containing simulated trajectories.
- Metadata files documenting simulation settings.

Dependencies
------------
numpy
pandas
matplotlib
fbm

===============================================================================
"""


import numpy as np
from fbm import FBM
import csv
import pandas as pd
import os
from dataclasses import dataclass, field

@dataclass
class Configuration:
    save: bool = True
    output_folder: str = r"D:\manuscripts\SG_GEM paper\SPT_Simulation_Data\Simulation_V2_space_mc_NucMask"
    simulation_name: str = 'optimization'
    add_to_name: str = ''
    parameter_folder: str = 'H'
    
    fixed_H: bool = True
    fixed_particle_number: bool = True
    fixed_particle_length: bool = True
    fixed_D: bool = True

    H: float = 0.35
    D: float | None = None
    pn_Q1: int = 500
    pn_median: int = 1000
    pn_Q3: int = 2000
    length_Q1: float = 0.060
    length_median: float = 0.075
    length_Q3: float = 0.090

    group_idx: int = 30
    tau: float = 0.005

    T: int = 308
    nw: float = 0.7195*10**(-3)
    Kb: float = 1.38*10**(-23)
    r: float = 20*10**(-9)
    L: float = 1*10**(-6)
    nc: float = field(init=False)
    file_path: str = field(init=False)
    save_name: str = field(init=False)

    def __post_init__(self):
        self.nc = self.nw * 35
        if self.D is None:
            self.D = self.Kb * self.T / (6 * np.pi * self.nc * self.r)
        self.save_name = f"fixed_H{self.H}_pnvar_lvar_D{self.D*10**12}"
        self.file_path = f"{self.output_folder}/{self.simulation_name}/{self.parameter_folder}/{self.save_name}/{self.save_name}_{self.add_to_name}.csv"

def generate_tracks(configutation):

    for group in range(0, cfg.group_idx):
        # print(group)
        
        #rng = np.random.default_rng(seed=42)

        if not cfg.fixed_particle_number:
            particle_number = int(np.random.triangular(
                left=cfg.pn_Q1, mode=cfg.pn_median, right=cfg.pn_Q3
            ))
        else:
            particle_number = cfg.particle_number

        for N in range(0, particle_number):

            if not cfg.fixed_H:
                H = float(np.random.triangular(left=cfg.H_Q1, mode=cfg.M, right=cfg.H_Q3))
            else:
                H = cfg.H

            if not cfg.fixed_particle_length:
                length = float(np.random.triangular(
                    left=cfg.length_Q1, mode=cfg.length_median, right=cfg.length_Q3
                ))
            else:
                length = cfg.length

            n = int(length / cfg.tau)

            f_x = FBM(n=n, hurst=H, length=length, method='hosking')
            f_y = FBM(n=n, hurst=H, length=length, method='hosking')

            if cfg.fixed_D:
                D = cfg.D
            else:
                D = float(np.random.triangular(left=cfg.D_Q1, mode=cfg.D_median, right=cfg.D_Q3))

            fbm_path_x = f_x.fbm() * np.sqrt(2 * D) * 10**6
            fbm_path_y = f_y.fbm() * np.sqrt(2 * D) * 10**6

            time = np.linspace(0, length, n + 1)

            df_temp = pd.DataFrame({
                'group': group,
                'particle': N,
                'H': H,
                'D': D,
                'L': length,
                'time (s)': time,
                'x (um)': fbm_path_x,
                'y (um)': fbm_path_y
            })
    
            if cfg.save:
                os.makedirs(os.path.dirname(cfg.file_path), exist_ok=True)
                df_temp.to_csv(
                    cfg.file_path,
                    mode='a',
                    index=False,
                    header=not os.path.exists(cfg.file_path)
                )

    if cfg.save:
      param_file = f"{cfg.output_folder}/{cfg.simulation_name}/metadata/{cfg.parameter_folder}_{cfg.save_name}_params.csv"
      os.makedirs(os.path.dirname(param_file), exist_ok=True)
      
      with open(param_file, mode='w', newline='') as file:
          writer = csv.writer(file)
      
          writer.writerow(['parameter', 'value'])
      
          # Simulation setup
          writer.writerow(['save_name', cfg.save_name])
          writer.writerow(['groups', cfg.group_idx])
          writer.writerow(['fixed_mode (particle_number)', cfg.fixed_particle_number])
      
          if not cfg.fixed_particle_number:
              writer.writerow(['particle_number (Q1)', cfg.pn_Q1])
              writer.writerow(['particle_number (median)', cfg.pn_median])
              writer.writerow(['particle_number (Q3)', cfg.pn_Q3])
          else:
              writer.writerow(['particle_number', cfg.particle_number])
      
          writer.writerow(['tau (s)', cfg.tau])
          writer.writerow(['fixed_mode (particle length)', cfg.fixed_particle_length])
      
          if not cfg.fixed_particle_length:
              writer.writerow(['particle_length (Q1)', cfg.length_Q1])
              writer.writerow(['particle_length (median)', cfg.length_median])
              writer.writerow(['particle_length (Q3)', cfg.length_Q3])
              writer.writerow(['track_length (s)', length])
          else:
              writer.writerow(['track_length (s)', cfg.length])
      
          writer.writerow(['steps (n)', int(length / cfg.tau)])
      
          # FBM parameters
          writer.writerow(['fixed_mode (H)', cfg.fixed_H])
          writer.writerow(['hurst_exponent (H)', H])
      
          if not cfg.fixed_H:
              writer.writerow(['hurst_exponent (Q1)', cfg.H_Q1])
              writer.writerow(['hurst_exponent_mean (M)', cfg.M])
              writer.writerow(['hurst_exponent (Q3)', cfg.H_Q3])
              writer.writerow(['fbm_method', 'hosking'])
      
          writer.writerow(['fixed_mode (D)', cfg.fixed_D])
          writer.writerow(['diffusion_coefficient (D, um^2/s)', D * 10**12])
      
          if not cfg.fixed_D:
              writer.writerow(['diffusion_coefficient (Q1)', cfg.D_Q1 * 10**12])
              writer.writerow(['diffusion_coefficient_mean (median)', cfg.D_median * 10**12])
              writer.writerow(['diffusion_coefficient (Q3)', cfg.D_Q3 * 10**12])
      
          # Physical constants
          writer.writerow(['temperature (K)', cfg.T])
          writer.writerow(['water_viscosity (Pa*s)', cfg.nw])
          writer.writerow(['cytoplasm_viscosity (Pa*s)', cfg.nc])
          writer.writerow(['particle_radius (m)', cfg.r])
          writer.writerow(['boltzmann_constant', cfg.Kb])

if __name__ == "__main__":
        
    SPACE_mask = {"NaArO2": [38, 16, 36, 57, 0.065, 0.070, 0.072],
                "RK33": [23, 6, 8, 19, 0.065, 0.065, 0.067],
                "Combo": [37, 18, 27, 33, 0.067, 0.070, 0.072]}
    
    SG_mask = {"NaArO2": [38, 14, 27, 41, 0.060, 0.065, 0.070],
                "RK33": [23, 4, 7, 20, 0.061, 0.065, 0.072],
                "Combo": [37, 13, 24, 40, 0.065, 0.070, 0.075]}
    
    for key in SG_mask.keys():
        
        mask = SPACE_mask[key]
        
        print(key)
        print("null")
        
        for j in range(1000):
            
            cfg = Configuration(group_idx=mask[0], fixed_particle_number=False, pn_Q1=mask[1], pn_median=mask[2], pn_Q3 = mask[3],
                                fixed_particle_length=False, length_Q1 = mask[4], length_median=mask[5], length_Q3=mask[6],
                                H = 0.35,
                                parameter_folder = 'null_H',
                                simulation_name = key,
                                add_to_name = j)
            
            generate_tracks(cfg)
        
        h_list = np.arange(0.25, 0.375, 0.025).tolist()
        h_list = np.arange(0.25, 0.375, 0.025).tolist()
        a_list = [h*2 for h in h_list]
        
        d_list = np.arange(cfg.D*0.6, cfg.D, cfg.D*0.1).tolist()
        D_list = [d*10**12 for d in d_list]
        
        for h in h_list:
            
            print('H same')
            for j in range(1000):
                
                cfg = Configuration(group_idx=mask[0], fixed_particle_number=False, pn_Q1=mask[1], pn_median=mask[2], pn_Q3 = mask[3],
                                    fixed_particle_length=False, length_Q1 = mask[4], length_median=mask[5], length_Q3=mask[6],
                                    H = h,
                                    parameter_folder = 'H_same_as_null',
                                    simulation_name = key,
                                    add_to_name = j)
                            
                generate_tracks(cfg)
                
        for idx, d in enumerate(d_list):
            
            if idx < 3:
                continue
            
            print('D same')
            
            for j in range(169, 1000):
                                            
                cfg = Configuration(group_idx=mask[0], fixed_particle_number=False, pn_Q1=mask[1], pn_median=mask[2], pn_Q3 = mask[3],
                                    fixed_particle_length=False, length_Q1 = mask[4], length_median=mask[5], length_Q3=mask[6],
                                    D = d,
                                    parameter_folder = 'D_same_as_null',
                                    simulation_name = key,
                                    add_to_name = j)
                
                generate_tracks(cfg)
        
        
        mask = SG_mask[key]
        
        for h in h_list:
                        
            for j in range(1000):
                
                cfg = Configuration(group_idx=mask[0], fixed_particle_number=False, pn_Q1=mask[1], pn_median=mask[2], pn_Q3 = mask[3],
                                    fixed_particle_length=False, length_Q1 = mask[4], length_median=mask[5], length_Q3=mask[6],
                                    H = h,
                                    parameter_folder = 'H',
                                    simulation_name = key,
                                    add_to_name = j)
                            
                generate_tracks(cfg)
        
        for d in d_list:
            
            print('D')
            
            for j in range(1000):
                                            
                cfg = Configuration(group_idx=mask[0], fixed_particle_number=False, pn_Q1=mask[1], pn_median=mask[2], pn_Q3 = mask[3],
                                    fixed_particle_length=False, length_Q1 = mask[4], length_median=mask[5], length_Q3=mask[6],
                                    D = d,
                                    parameter_folder = 'D',
                                    simulation_name = key,
                                    add_to_name = j)
                
                generate_tracks(cfg)


