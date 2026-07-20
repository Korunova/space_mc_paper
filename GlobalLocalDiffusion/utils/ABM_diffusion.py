# myutils.py
"""Reusable utilities for SPT / MSD analysis."""

from typing import Iterable, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp
import csv

CONFIG = {
    "static_error_path": r"D:\manuscripts\SG_GEM paper\SPT\tifSPT_01232026_controls_MSD\LK_SPT_V3\MSD_lagtime10.0ms.csv"
    }



def plot(x, y, xname: Optional[str] = None, yname: Optional[str] = None,
         std: Optional[Iterable[float]] = None, plot_number: Optional[int] = None):
    """Quick plotting helper with optional errorbars."""
    if plot_number is not None:
        plt.figure(num=int(plot_number), figsize=(12, 8))
    else:
        plt.figure(figsize=(12, 8))
    plt.plot(x, y, marker='.', linestyle='-', linewidth=2, markersize=2)
    if std is not None:
        plt.errorbar(x, y, yerr=std, ecolor='green', label=f'Standard Deviation {yname}', alpha=0.5)
    if yname: plt.ylabel(yname)
    if xname: plt.xlabel(xname)
    plt.grid(True, zorder=1)


# ---------------------- fitting ---------------------------------------
def FitABMequation(name: str,
                      x: np.ndarray,
                      y: np.ndarray,
                      track_count:int,
                      track_mean_length: int,
                      track_median_length: int,
                      Dw: int,
                      experiment: Optional[str] = None,
                      label: Optional[str] = None,
                      n: Optional[int] = None,
                      points: int = 10,
                      sigma: Optional[np.ndarray] = None,
                      check_plot: bool = False,
                      plot_numbers: List[int] = [1, 2],
                      ):
    """
    Fit a linear model y = a*x + log(4*Da) to (x,y) arrays.
    Returns (name, a, a_err, Da, Da_err, r_squared)
    """
    def linear_function(x_, a, Da):
        return a * x_ + np.log(4 * Da)

    # prepare sigma if None
    sigma_local = sigma if sigma is not None else np.ones_like(y)

    params, covariance = curve_fit(linear_function, x[0:points], y[0:points], sigma=sigma_local[0:points])
    fitted_a, fitted_Da = params
    a_error, Da_error = np.sqrt(np.diag(covariance))

    predicted_msd = linear_function(x[0:points], fitted_a, fitted_Da)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)

    if check_plot:
        plt.figure(figsize=(18, 14), num=plot_numbers[0])
        plt.errorbar(x, y, yerr=sigma_local, fmt='o', color='blue', alpha=1)
        plt.plot(x[0:points], predicted_msd, marker='.', linestyle='-', linewidth=2, markersize=2,
                 color='red',
                 label= f'{round(r_squared, 2)}')
        plt.ylabel(r'log(<MSD>), $\mu$m$^2$')
        plt.xlabel('log(τ), sec')
        plt.legend(loc='lower right')
        plt.show()


    Da, a = sp.symbols('Da a')
    Deff = (Da * 2**(2 * (a - 1))) / (Dw**a) 
    partial_Da = sp.diff(Deff, Da) 
    partial_a = sp.diff(Deff, a)
    
    partial_Da_value = float(partial_Da.subs({Da: fitted_Da, a: fitted_a}))
    partial_a_value = float(partial_a.subs({Da: fitted_Da, a: fitted_a}))
    
    Deff_sigma = np.sqrt((partial_Da_value * Da_error)**2 + (partial_a_value * a_error)**2)
    Deff_value = float(Deff.subs({Da: fitted_Da, a: fitted_a}))
    
    output_result = {
        'Experiment': experiment,
        'Plot_Name': label,
        'Da_value': fitted_Da,
        'Da_sigma': Da_error,
        'a_value': fitted_a,
        'a_sigma': a_error,
        'Deff_value': Deff_value,
        'Deff_sigma': Deff_sigma,
        'R2': r_squared,
        'track_count': track_count,
        'track_mean_length': track_mean_length,
        'track_median_length': track_median_length
    }

    #return name, float(fitted_a), float(a_error), float(fitted_Da), float(Da_error), r_squared
    return output_result


# ---------------------- fitting ---------------------------------------
def FitABMequation_LocError(name: str,
                      x: np.ndarray,
                      y: np.ndarray,
                      track_count:int,
                      track_mean_length: int,
                      track_median_length: int,
                      Dw: int,
                      experiment: Optional[str] = None,
                      label: Optional[str] = None,
                      n: Optional[int] = None,
                      points: int = 10,
                      sigma: Optional[np.ndarray] = None,
                      check_plot: bool = False,
                      plot_numbers: List[int] = [1, 2],
                      ):
    """
    Fit a linear model y = a*x + log(4*Da) to (x,y) arrays.
    Returns (name, a, a_err, Da, Da_err, r_squared)
    """
    def linear_function(x_, a, Da, C):
        return 4.0 * Da * (x_ ** a) + C

    # prepare sigma if None
    sigma_local = sigma if sigma is not None else np.ones_like(y)

    params, covariance = curve_fit(linear_function, x[0:points], y[0:points], sigma=sigma_local[0:points])
    fitted_a, fitted_Da, fitted_C = params
    a_error, Da_error, C_error = np.sqrt(np.diag(covariance))

    predicted_msd = linear_function(x[0:points], fitted_a, fitted_Da, fitted_C)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)

    if check_plot:
        plt.figure(figsize=(18, 14))
        plt.errorbar(x, y, yerr=sigma_local, fmt='o', color='blue', alpha=1)
        plt.plot(x[0:points], predicted_msd, marker='.', linestyle='-', linewidth=2, markersize=2,
                 color='red',
                 label= f'{round(r_squared, 2)}')
        plt.ylabel(r'log(<MSD>), $\mu$m$^2$')
        plt.xlabel('log(τ), sec')
        plt.legend(loc='lower right')
        plt.show()


    Da, a = sp.symbols('Da a')
    Deff = (Da * 2**(2 * (a - 1))) / (Dw**a) 
    partial_Da = sp.diff(Deff, Da) 
    partial_a = sp.diff(Deff, a)
    
    partial_Da_value = float(partial_Da.subs({Da: fitted_Da, a: fitted_a}))
    partial_a_value = float(partial_a.subs({Da: fitted_Da, a: fitted_a}))
    
    Deff_sigma = np.sqrt((partial_Da_value * Da_error)**2 + (partial_a_value * a_error)**2)
    Deff_value = float(Deff.subs({Da: fitted_Da, a: fitted_a}))
    
    output_result = {
        'Experiment': experiment,
        'Plot_Name': label,
        'Da_value': fitted_Da,
        'Da_sigma': Da_error,
        'a_value': fitted_a,
        'a_sigma': a_error,
        'Deff_value': Deff_value,
        'Deff_sigma': Deff_sigma,
        'R2': r_squared,
        'track_count': track_count,
        'track_mean_length': track_mean_length,
        'track_median_length': track_median_length
    }

    #return name, float(fitted_a), float(a_error), float(fitted_Da), float(Da_error), r_squared
    return output_result



def FitBMequation_LocError(
                      x: np.ndarray,
                      y: np.ndarray,
                      track_count: int,
                      experiment: Optional[str] = None,
                      label: Optional[str] = None,
                      n: Optional[int] = None,
                      points: int = 10,
                      sigma: Optional[np.ndarray] = None,
                      check_plot: bool = False,
                      plot_numbers: List[int] = [1, 2]):
    """
    Fit a linear model y = 4*D*x to (x,y) arrays.
    Returns (name, D, D_err, r_squared)
    """
    
    def linear_function(x_, D, offset):
        return 4 * D * x_ + offset

    # prepare sigma if None
    sigma_local = sigma if sigma is not None else np.ones_like(y)

    params, covariance = curve_fit(linear_function, x[0:points], y[0:points], sigma=sigma_local[0:points])
    fitted_D = float(params[0])
    offset = float(params[1])
    D_error = float(np.sqrt(np.diag(covariance))[0])

    predicted_msd = linear_function(x[0:points], fitted_D, offset)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)
    
    if check_plot:
        plt.figure(figsize=(18, 14))
        plt.errorbar(x, y, yerr=sigma_local, fmt='o', color='blue', alpha=1)
        plt.plot(x[0:points], predicted_msd, marker='.', linestyle='-', linewidth=2, markersize=2,
                  color='red', label = f'{round(r_squared, 2)}')
        plt.ylabel(r'<MSD>, $\mu$m$^2$')
        plt.xlabel('τ, sec')
        plt.legend(loc='lower right')
        plt.show()
    
    output_result = {
        'Experiment': experiment,
        'Plot_Name': label,
        'D_value': fitted_D,
        'D_sigma': D_error,
        'R2': r_squared,
        'track_count': track_count
    }

    return output_result 

def FitBMequation(
                      x: np.ndarray,
                      y: np.ndarray,
                      track_count: int,
                      experiment: Optional[str] = None,
                      label: Optional[str] = None,
                      n: Optional[int] = None,
                      points: int = 10,
                      sigma: Optional[np.ndarray] = None,
                      check_plot: bool = False,
                      plot_numbers: List[int] = [1, 2]):
    """
    Fit a linear model y = 4*D*x to (x,y) arrays.
    Returns (name, D, D_err, r_squared)
    """
    
    def linear_function(x_, D):
        return 4 * D * x_ 

    # prepare sigma if None
    sigma_local = sigma if sigma is not None else np.ones_like(y)

    params, covariance = curve_fit(linear_function, x[0:points], y[0:points], sigma=sigma_local[0:points])
    fitted_D = float(params[0])
    D_error = float(np.sqrt(np.diag(covariance))[0])

    predicted_msd = linear_function(x[0:points], fitted_D)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)

    if check_plot:
        plt.figure(figsize=(18, 14))
        plt.errorbar(x, y, yerr=sigma_local, fmt='o', color='blue', alpha=1)
        plt.plot(x[0:points], predicted_msd, marker='.', linestyle='-', linewidth=2, markersize=2,
                  color='red', label = f'{round(r_squared, 2)}')
        plt.ylabel(r'<MSD>, $\mu$m$^2$')
        plt.xlabel('τ, sec')
        plt.legend(loc='lower right')
        plt.show()
    
    output_result = {
        'Experiment': experiment,
        'Plot_Name': label,
        'D_value': fitted_D,
        'D_sigma': D_error,
        'R2': r_squared,
        'track_count': track_count
    }

    return output_result 



# ---------------------- ABM / MSD ------------------------------------
def ABM_etaMSD(trajectories: list,
               experiment: str,
               label: str,
               delta_x: float,
               tau: float,
               r2: float,
               Dw: int,
               color_number: int = 0,
               plot_numbers: list = [1, 2],
               static_error: bool = False,
               loc_error: bool = False,
               check_plot: bool = False,
               points: int = 10):
    """
    Calculate ensemble/time averaged MSD and fit to ABM model.
    NOTE: `delta_x` and `tau` are passed explicitly (no globals).
    trajectories: list of [t_list, x_list, y_list]
    returns the FitLinearFunction result tuple
    """
    
    track_count = len(trajectories)
    
    if static_error:
        # Read MSD errors from CSV and store in a dictionary {time_interval: error_value}
        msd_errors = {}

        with open(CONFIG["static_error_path"], mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                time_interval = float(row["Time"])
                msd_static = float(row["MSD"])
                sigma_static = float(row["MSD_STD"])
                msd_errors[time_interval] = [msd_static, sigma_static]
    
    tau_list = []
    msd_list = []
    msd_std_list = []

    # collect MSD by lag time
    lagtime_data = {}
    tracks_length = [] 
    min_step = int(tau / delta_x) if delta_x > 0 else 1
    for item in trajectories:
        
        tid = item["track_id"]
        t = item["trajectory"][0]
        x = item["trajectory"][1]
        y = item["trajectory"][2]
                
        track_length = len(t)
        tracks_length.append(track_length)
        
        for delta_t in range(min_step, len(t), min_step):
            for i in range(0, len(t) - delta_t, min_step):
                time_interval = round(t[i + delta_t] - t[i], 3)
                squared_displacement = (x[i + delta_t] - x[i]) ** 2 + (y[i + delta_t] - y[i]) ** 2
                lagtime_data.setdefault(time_interval, []).append(squared_displacement)
    
    track_mean_length = np.mean(tracks_length)
    track_median_length = np.median(tracks_length)
    
    for key, values in sorted(lagtime_data.items()):
        mean_msd = np.mean(values)

        #assign Nan to sigma values if there are too little amount of data points
        if len(values) > 1:
            sigma_msd = np.std(values, ddof=1)
        else:
            continue
        
        # If static error removal is enabled, subtract the static error from the mean MSD
        if static_error and key in msd_errors:
            mean_msd -= msd_errors[key][0]
            sigma_msd = np.sqrt(sigma_msd**2 + sigma_static**2)
        
        tau_list.append(key)
        msd_list.append(mean_msd)
        msd_std_list.append(sigma_msd)

    # If no data, return None
    if not tau_list:
        return None

    time_data = np.array(tau_list)
    msd_data = np.array(msd_list)
    msd_std_data = np.array(msd_std_list)
    
    
    #filter of msd tracks to remove Inf values or negative numbers
    #valid = (msd_data > 0) & np.isfinite(msd_std_data)
    
    # valid = (
    #     np.isfinite(time_data) &
    #     np.isfinite(msd_data) &
    #     np.isfinite(msd_std_data) &
    #     (time_data > 0) &
    #     (msd_data > 0)
    # )
    
    # time_data = time_data[valid]
    # msd_data = msd_data[valid]
    # msd_std_data = msd_std_data[valid]

    # log scale
    time_data_log10 = np.log10(time_data)
    msd_data_log10 = np.log10(msd_data)
    msd_std_data_log10 = msd_std_data / (msd_data * np.log(10))
    
    #filter to remove negative or infinite sigmas after log
    msd_std_data_log10[~np.isfinite(msd_std_data_log10)] = 1e-6
    msd_std_data_log10[msd_std_data_log10 <= 0] = 1e-6
    
    #limit = max(1, round(len(msd_data_log10) / 10))
    name = ''
    

    if loc_error:
        result = FitABMequation_LocError(name, time_data[0:20], msd_data[0:20], track_count, track_mean_length, track_median_length, Dw,
                                    experiment, label, n=color_number, points=points,
                                    sigma=msd_std_data[0:20], check_plot=check_plot)
    else: 
        result = FitABMequation(name, time_data_log10[0:20], msd_data_log10[0:20], track_count, track_mean_length, track_median_length, Dw,
                                    experiment, label, n=color_number, points=points,
                                    sigma=msd_std_data_log10[0:20], check_plot=check_plot)

    if result['R2'] < r2:
        result = None
            
    return result

def ABM_taMSD(trajectories: list, output_result: list, delta_x: float, tau: float, r2: float, Dw: float, experiment: str, mode: str = "ABM"):
    for item in trajectories:
        tid = item["track_id"]
        t = item["trajectory"][0]
        x = item["trajectory"][1]
        y = item["trajectory"][2]
        
        tau_list = []
        msd_list = []
        msd_std_list = []
        msd_se_list = []
        
        min_step = int(tau/delta_x)
        
        for delta_t in range(min_step, len(t), min_step):
            time_intervals = []
           
            for i in range(0, len(t) - delta_t, min_step):
                time_interval = t[i+delta_t]-t[i] 
                #t[10] - t[0], t[20] - t[10] = 10
                #t[20] - t[0], t[30] - t[10] = 20 --- delta_t = 2
                squared_displacement = (x[i + delta_t] - x[i])**2 + (y[i + delta_t] - y[i])**2
                squared_displacements.append(squared_displacement)
            
            mean_msd = np.mean(squared_displacements)
            sigma_msd = np.std(squared_displacements, ddof=1)

            # If static error removal is enabled, subtract the static error from the mean MSD
            # if remove_static_error and key in msd_errors:
            #     mean_msd -= msd_errors[key][0]
            #     sigma_msd = np.sqrt(sigma_msd**2 + sigma_static**2)
            
            tau_list.append(time_interval)
            msd_list.append(mean_msd) #mean MSD from one time interval
            msd_std_list.append(sigma_msd) #Standard Deviation of MSD
            #msd_se_list.append(np.std(squared_displacements, ddof=1)/np.sqrt(len(squared_displacements))) #Standard error of MSD, the error from mean

        
        time_data = np.array(tau_list)
        msd_data = np.array(msd_list)
        msd_std_data = np.array(msd_std_list)
        
        if mode == 'ABM':
            #Mode of Motion Calulation
            time_data_log10 = np.log10(time_data)
            msd_data_log10 = np.log10(msd_data)
            msd_std_data_log10 = msd_std_data / (msd_data * np.log(10))  # Corrected error calculation log10x' = 1/(<x>*log10)
    
            MSD_limit = len(time_data_log10)-1
            for p in range(5, MSD_limit):
                result = FitABMequation('', time_data_log10[0:MSD_limit], msd_data_log10[0:MSD_limit], 1, Dw, experiment, tid, R2=r2, points = p, sigma = msd_std_data_log10[0:MSD_limit])
                
                if p == 5 and result['R2'] < r2:
                    continue
    
                if result != None and result['R2'] < r2:
                    result = FitABMequation('', time_data_log10[0:MSD_limit], msd_data_log10[0:MSD_limit], 1, Dw, experiment, tid, R2=r2, points = p-1, sigma = msd_std_data_log10[0:MSD_limit])
                    output_result.append(result)
                    break
    
        if mode == 'BM':
            MSD_limit = len(time_data)-1
            for p in range(5, MSD_limit):
                result = FitBMequation(time_data, msd_data, 1, experiment, tid, sigma = msd_std_data, points = 5)
                
                if p == 5 and result['R2'] < r2:
                    continue
                
                if result != None and result['R2'] < r2:
                    result = FitBMequation(time_data, msd_data, 1, experiment, tid, sigma = msd_std_data, points = p-1)
                    output_result.append(result)
                    break
                    
    
def BM_etaMSD( trajectories: list,
               experiment: str,
               label: str,
               delta_x: float,
               tau: float,
               r2: float = 0,
               points: int = 10,
               color_number: int = 0,
               plot_numbers: list = [1, 2],
               loc_error: bool = False,
               static_error: bool = False,
               check_plot: bool = False):
    """
    Calculate ensemble/time averaged MSD and fit to ABM model.
    NOTE: `delta_x` and `tau` are passed explicitly (no globals).
    trajectories: list of [t_list, x_list, y_list]
    returns the FitLinearFunction result tuple
    """
    
    track_count = len(trajectories)

    if static_error:
        # Read MSD errors from CSV and store in a dictionary {time_interval: error_value}
        msd_errors = {}

        with open(CONFIG["static_error_path"], mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                time_interval = float(row["Time"])
                msd_static = float(row["MSD"])
                sigma_static = float(row["MSD_STD"])
                msd_errors[time_interval] = [msd_static, sigma_static]
    
    tau_list = []
    msd_list = []
    msd_std_list = []

    # collect MSD by lag time
    lagtime_data = {}
    min_step = int(tau / delta_x) if delta_x > 0 else 1
    for item in trajectories:
                
        tid = item["track_id"]
        t = item["trajectory"][0]
        x = item["trajectory"][1]
        y = item["trajectory"][2]
        
        # track_mismatch =  not all(t[i] > t[i-1] for i in range(1, len(t))) #because some tracks were at too places during one permutation
        # if track_mismatch:
        #     continue
        
        for delta_t in range(min_step, len(t), min_step):
            for i in range(0, len(t) - delta_t, min_step):
                time_interval = round(t[i + delta_t] - t[i], 3)
                squared_displacement = (x[i + delta_t] - x[i]) ** 2 + (y[i + delta_t] - y[i]) ** 2
                
                lagtime_data.setdefault(time_interval, []).append(squared_displacement)

    for key, values in sorted(lagtime_data.items()):
                
        # print(key)
        
        if len(values) < 2:
            #print("time_interval,sec:", key, "n =", len(values))
            continue
        
        mean_msd = np.mean(values)
        sigma_msd = np.std(values, ddof=1)
        
        # If static error removal is enabled, subtract the static error from the mean MSD
        if static_error and key in msd_errors:
            mean_msd -= msd_errors[key][0]
            sigma_msd = np.sqrt(sigma_msd**2 + sigma_static**2)
        
        tau_list.append(key)
        msd_list.append(mean_msd)
        msd_std_list.append(sigma_msd)
        
    # If no data, return None
    if not tau_list:
        return None

    time_data = np.array(tau_list)
    msd_data = np.array(msd_list)
    msd_std_data = np.array(msd_std_list)
    
    if loc_error:
        result = FitBMequation_LocError(time_data[0:20], msd_data[0:20], track_count, experiment, label, sigma = msd_std_data[0:20], n=color_number, points = points, check_plot=check_plot)
    else: 
        result = FitBMequation(time_data[0:20], msd_data[0:20], track_count, experiment, label, sigma = msd_std_data[0:20], n=color_number, points = points, check_plot=check_plot)
    
    #LAST CHANGE - ADDITION OF R2 FILTER
    if result['R2'] < r2:
        result = None
    
    
    return result