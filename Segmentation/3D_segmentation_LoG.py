# -*- coding: utf-8 -*-
"""
===============================================================================
3D Stress Granule Segmentation and Colocalization Analysis
===============================================================================

Description
-----------
This script performs automated 3D segmentation and quantitative analysis of
stress granules and associated proteins in multichannel fluorescence microscopy
images. Objects are segmented using a Laplacian-of-Gaussian (LoG)-based
pipeline, followed by extraction of morphological, intensity, and
colocalization measurements.

The script also generates pixel-based and object-based cytofluorograms to
quantify fluorescence relationships between imaging channels.

Main workflow
-------------
1. Load multichannel 3D fluorescence microscopy images.
2. Detect nuclei for cell identification.
3. Segment stress granules and protein puncta using LoG-based segmentation.
4. Save instance-labeled segmentation masks.
5. Measure object morphology, including:
   - volume
   - equivalent diameter
   - surface area
   - sphericity
   - ellipsoid axes
   - fluorescence intensity
6. Quantify object colocalization using:
   - Manders overlap coefficients
   - intensity ratios
   - signal measurements across channels
7. Generate pixel-based and object-based cytofluorograms.
8. Export quantitative measurements as CSV files.

Outputs
-------
- Instance-labeled segmentation masks (OME-TIFF)
- Object morphology tables
- Colocalization statistics
- Manders coefficients
- Cytofluorograms
- Metadata describing imaging channels

Dependencies
------------
numpy
scipy
pandas
matplotlib
scikit-image
tifffile
napari

===============================================================================
"""

import tifffile
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log
from skimage import measure
from scipy.ndimage import gaussian_laplace, label
import napari
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.segmentation import watershed
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
import pandas as pd
from skimage.measure import regionprops_table
import os

from skimage import segmentation
from skimage.segmentation import relabel_sequential
from skimage.segmentation import find_boundaries
from skimage import morphology 
from skimage.feature import peak_local_max

# -------------------------
# 1) preprocessing function
# -------------------------
def preprocess_blur(vol, sigma=1.0):
    """
    Simple 3D Gaussian blur preprocessing.
    vol: (Z, Y, X) numpy array
    sigma: gaussian sigma in voxels (0 -> no blur)
    returns blurred volume (same dtype float32)
    """
    vol = vol.astype(np.float32)
    if sigma and sigma > 0:
        blurred = ndi.gaussian_filter(vol, sigma=sigma)
    else:
        blurred = vol.copy()
    return blurred

# -------------------------
# 2) 3D LoG detection
# -------------------------
def log3d_detect(vol,
                 min_sigma=1.0,
                 max_sigma=4.0,
                 num_sigma=6,
                 threshold=0.02):
    """
    Detect blobs with blob_log and produce a label image of spherical seeds.
    Returns: labels (Z,Y,X) int32, blobs array (n, 4) with (z,y,x,sigma)
    """
    # normalize to 0..1 to make thresholding more consistent
    v = vol.astype(np.float32)
    if v.ptp() > 0:
        v = (v - v.min()) / (v.ptp())
    blobs = blob_log(v,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     num_sigma=num_sigma,
                     threshold=threshold,
                     overlap=0.5)
    # blob_log returns (z, y, x, sigma)
    labels = np.zeros_like(v, dtype=np.int32)
    if blobs.size == 0:
        return labels, blobs

    # for i, (zc, yc, xc, sigma) in enumerate(blobs, start=1):
    #     z, y, x = map(int, np.round([zc, yc, xc]))
    #     # skip if already occupied
    #     if labels[z, y, x] == 0:
    #         labels[z, y, x] = i
            
    zz, yy, xx = np.indices(v.shape)

    for i, b in enumerate(blobs, start=1):
        zc, yc, xc, sigma = b
        r = 2
        # spherical mask around center
        mask = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2 <= r * r
        # don't overwrite existing labels (keeps seeds separate)
        labels[np.logical_and(mask, labels == 0)] = i
    
    # small cleanup / relabel
    labels = measure.label(labels > 0, connectivity=1)

    # small cleanup / relabel
    # labels, _, _ = relabel_sequential(labels)
    return labels, blobs



# -------------------------
# 3) Watershed
# -------------------------

def detect_particles_watershed_nd(channel,
                                  markers=None,
                                  peak_coords=None,
                                  mask=None,
                                  use_distance=True,
                                  smooth_sigma=1.0,
                                  min_size=8):
    """
    2D/3D watershed that accepts either:
      - markers: integer-labelled marker image (same shape as channel)
    mask should be a boolean array (True where segmentation is allowed).
    Returns: centroids (list of tuples), labels (ndarray, int), smoothed (ndarray)
    """
    # smoothing (works in nD)
    smoothed = gaussian(channel, sigma=smooth_sigma, preserve_range=True)

    mask = (mask.astype(bool))

    # Choose watershed image: distance or intensity
    if use_distance:
        # distance transform of mask (nd)
        distance = ndi.distance_transform_edt(mask)
        ws_image = distance
    else:
        # intensity-based
        ws_image = smoothed

    labels = watershed(-ws_image, markers=markers, mask=mask)

    # # cleanup: remove small objects and relabel
    # if labels.max() > 0:
    #     labels = morphology.remove_small_objects(labels, min_size=min_size)
    #     labels = measure.label(labels > 0, connectivity=1)

    # compute centroids (regionprops works for 2D/3D)
    centroids = []
    for region in measure.regionprops(labels):
        centroids.append(tuple(region.centroid))  # (z,y,x) for 3D, (y,x) for 2D

    return centroids, labels.astype(np.int32), mask, ws_image, markers



# -------------------------
# 4) Local Treshold
# -------------------------
def local_slice_mask(vol, block_size=51, offset=0.0, method='gaussian', min_size=2):
    """
    (a) Per-slice adaptive/local threshold from raw image (no LoG).
    vol: (Z,Y,X)
    returns: mask (bool, same shape) and labeled mask
    """
    mask = np.zeros_like(vol, dtype=bool)
    for z in range(vol.shape[0]):
        slice_img = vol[z].astype(np.float32)
        # ensure odd block_size and not larger than slice dims
        local_th = threshold_local(slice_img, block_size, method=method, offset=offset)
        mask[z] = (slice_img > local_th * 2.5) 
        
    # cleanup
    #mask = morphology.remove_small_objects(mask, min_size=min_size)
    # mask = morphology.binary_closing(mask, morphology.ball(1))
    labels, n = label(mask)
    return mask, labels

# -------------------------
# 5) LoG based segmentation
# -------------------------


def segmentation_LoG(channel, sigma_big = 2, sigma_small = 1.5, max_projection_trsh = 10, view = False):

    blurred = preprocess_blur(channel, sigma=blur_sigma)
    trsh, _ = local_slice_mask(blurred, 51) 
    
    log_big = gaussian_laplace(channel, sigma= sigma_big)
    particles_big = (log_big > np.percentile(log_big, 96)) & trsh
    
    log_small = gaussian_laplace(channel, sigma= sigma_small) 
    particles_small = (log_small > np.percentile(log_small, 96)) & ~particles_big & trsh
    
    particles = (particles_big | particles_small) 
    particles_filled = binary_fill_holes(particles)
    
    labels, n = label(particles_filled)
    
    labels_filtered = np.zeros_like(labels)
    
    for lab in range(1, n + 1):
        mask = labels == lab
        max_intensity = channel[mask].max()   # max over Z,Y,X for this object
    
        if max_intensity > max_projection_trsh:   # your threshold
            labels_filtered[mask] = lab
    
    if view:
        
        zmid = channel.shape[0] // 2
        plt.figure(figsize=(6, 6))
        # base grayscale image
        plt.imshow(channel[zmid], cmap='gray', vmin=0, vmax=np.percentile(channel, 99))
        # transparent overlay of label map
        plt.imshow(labels_filtered[zmid], cmap='nipy_spectral', alpha=0.4)  # alpha controls transparency
        plt.title(f'Overlay: channel + labels (Z={zmid})')
        plt.axis('off')
        plt.show()
        
        zmid = channel.shape[0] // 2
        plt.figure(figsize=(6, 6))
        plt.imshow(channel[zmid], cmap='plasma', vmin=0, vmax=np.percentile(channel, 99))  # alpha controls transparency
        plt.colorbar()
        plt.title(f'Overlay: channel + labels (Z={zmid})')
        plt.axis('off')
        plt.show()
        
        viewer = napari.Viewer()
        viewer.add_image(channel, name='channel_original', colormap='gray', contrast_limits=(0, np.percentile(channel, 99)*3))
        viewer.add_image(labels_filtered, name='markers', colormap='viridis', opacity=0.6)
        napari.run()
    
    return label(particles_filled)

# -------------------------
# 5) Save Segmentatin as Stack 
# -------------------------

def save_segmentation(labels_list, save_path):
    channels_save = [label.astype(np.int32) for label in labels_list]
    
    stack_out = np.stack(channels_save, axis=0)
    
    tifffile.imwrite(
        save_path,
        stack_out.astype(np.float32),
        ome = True,
        metadata={'axes': 'CZYX'}  
    )

# -------------------------
# 5) Save SG params in csv file 
# -------------------------

def save_SGs_param(labels_channel, labels_nuclei, channel, csv_path):
    
    # ---------- Compute particle morphology table for 'labels' (your SG labels) ----------
    # We'll compute for each labeled SG:
    #   - label id, volume (voxels), centroid (z,y,x)
    #   - mean_intensity, max_intensity in channel
    #   - bbox (minz,miny,minx,maxz,maxy,maxx)
    #   - equivalent_diameter (3D sphere diameter)
    #   - surface_area (approx via marching_cubes) and sphericity
    #   - ellipsoid semi-axes (a1,a2,a3) estimated from 3D covariance (in voxels)
    
    lbls, n_SGs = labels_channel  # your SG labels (3D int array)
    lbls_nuclei, n_nuclei = labels_nuclei  # your SG labels (3D int array)
    
    props = measure.regionprops(lbls, intensity_image=channel)
    
    records = []
    for p in props:
        lab = p.label
        vol = p.area                      # voxel count (3D)
        centroid = p.centroid             # (z, y, x)
        mean_int = p.mean_intensity
        max_int = p.max_intensity
        bbox = p.bbox                     # (minz, miny, minx, maxz, maxy, maxx)
    
        # equivalent diameter for a sphere with same volume (3D)
        equiv_diam = (6.0 * vol / np.pi) ** (1.0/3.0)
    
        # Surface area approximation via marching cubes on object mask (may be slow for many objects)
        mask_i = (lbls == lab).astype(np.uint8)
        # pad to avoid boundary effects
        mask_pad = np.pad(mask_i, pad_width=1, mode='constant', constant_values=0)
        try:
            verts, faces, normals, values = measure.marching_cubes(mask_pad, level=0.5)
            # triangle areas
            tris = verts[faces]
            v0 = tris[:,0,:]; v1 = tris[:,1,:]; v2 = tris[:,2,:]
            cross = np.cross(v1 - v0, v2 - v0)
            tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
            surface_area = tri_areas.sum()
        except Exception:
            surface_area = np.nan
    
        # sphericity: (pi^(1/3) * (6V)^(2/3)) / A
        if (not np.isnan(surface_area)) and surface_area > 0:
            sphericity = (np.pi ** (1.0/3.0)) * ((6.0 * vol) ** (2.0/3.0)) / surface_area
        else:
            sphericity = np.nan
    
        # Ellipsoid semi-axes via covariance of voxel coordinates:
        coords = np.column_stack(np.nonzero(mask_i))   # rows: (z,y,x)
        if coords.shape[0] >= 3:
            mean_coord = coords.mean(axis=0)
            cov = ((coords - mean_coord).T @ (coords - mean_coord)) / coords.shape[0]  # population covariance (3x3)
            eigvals, eigvecs = np.linalg.eigh(cov)   # eigenvalues sorted low->high
            # For a uniform solid ellipsoid, covariance eigenvalue = a^2 / 5 where a is semi-axis length
            # => semi-axes a = sqrt(5 * eigval)
            # sort descending so a1 >= a2 >= a3
            semi_axes = np.sqrt(5.0 * np.sort(eigvals)[::-1])
            a1, a2, a3 = semi_axes.tolist()
        else:
            a1 = a2 = a3 = np.nan
    
        records.append({
            'label': lab,
            'volume_voxels': int(vol),
            'centroid_z': float(centroid[0]), 'centroid_y': float(centroid[1]), 'centroid_x': float(centroid[2]),
            'mean_intensity': float(mean_int),
            'max_intensity': float(max_int),
            'bbox_minz': int(bbox[0]), 'bbox_miny': int(bbox[1]), 'bbox_minx': int(bbox[2]),
            'bbox_maxz': int(bbox[3]), 'bbox_maxy': int(bbox[4]), 'bbox_maxx': int(bbox[5]),
            'equivalent_diameter_voxels': float(equiv_diam),
            'surface_area_voxels2': float(surface_area) if not np.isnan(surface_area) else np.nan,
            'sphericity': float(sphericity) if not np.isnan(sphericity) else np.nan,
            'ellipsoid_a1': float(a1), 'ellipsoid_a2': float(a2), 'ellipsoid_a3': float(a3),
            'SGs number': n_SGs,
            'Nuclei_number': n_nuclei
        })
    
    df = pd.DataFrame.from_records(records)
    df.to_csv(csv_path, index=False)
    print("Saved morphology table to:", csv_path)


def save_SGs_coloc_param(labels_union, labels_channel2, channel1, channel2, csv_path,
                         manders_threshold_ch1=0, manders_threshold_ch2=0):
    """
    Compute per-particle intensity sums/means/ratio and Manders coefficients for
    particles defined in labels_channel1. If you intended particles defined by
    labels_channel2, change how props are built (see note below).

    Parameters:
      labels_channel1: tuple (lbls1, n1) -- labeled mask and count for channel1 particles
      labels_channel2: tuple (lbls2, n2) -- provided but not used here (see note)
      channel1, channel2: numpy arrays (same shape)
      csv_path: path to save CSV
      manders_threshold_ch1, manders_threshold_ch2: thresholds (float) applied to
        define "colocalized" pixels (default 0).
    """

    props1 = measure.regionprops(labels_channel2[0], intensity_image=channel1)
    # sample channel2 intensities using the same labels (per-particle)
    props2 = measure.regionprops(labels_channel2[0], intensity_image=channel2)

    ratio_records = []
    for p1, p2 in zip(props1, props2):
        pid = p1.label
        # p1.image is boolean mask inside the bbox; intensity_image aligns to bbox
        mask = p1.image  # boolean array
        int1 = p1.intensity_image  # channel1 intensities within the bbox
        int2 = p2.intensity_image  # channel2 intensities within the bbox

        # Flatten to 1D arrays of pixels inside the mask
        pix1 = int1[mask]
        pix2 = int2[mask]

        sum1 = float(pix1.sum())
        sum2 = float(pix2.sum())
        mean1 = float(pix1.mean()) if pix1.size > 0 else np.nan
        mean2 = float(pix2.mean()) if pix2.size > 0 else np.nan

        ratio_sum = sum1 / sum2 if sum2 != 0 else np.nan
        ratio_mean = mean1 / mean2 if mean2 != 0 else np.nan
        
        ratio_records.append({
            "particle_id": pid,
            "area_pixels": int(p1.area),
            "mean_ch1": mean1,
            "mean_ch2": mean2,
            "sum_ch1": sum1,
            "sum_ch2": sum2,
            "ratio_sum_ch1_over_ch2": ratio_sum,
            "ratio_mean_ch1_over_ch2": ratio_mean
        })
        
    df_ratios = pd.DataFrame.from_records(ratio_records)

    # Save ratios
    ratios_csv = f"{csv_path}_ratios.csv"
    if save:
        df_ratios.to_csv(ratios_csv, index=False)
# ---------- PART B: Manders for labels_union ----------
    props_u1 = measure.regionprops(labels_union[0], intensity_image=ch1)
    props_u2 = measure.regionprops(labels_union[0], intensity_image=ch2)

    manders_records = []
    for p1, p2 in zip(props_u1, props_u2):
        pid = p1.label
        mask = p1.image
        int1 = p1.intensity_image[mask].astype(float)
        int2 = p2.intensity_image[mask].astype(float)

        sum1 = float(int1.sum()) if int1.size > 0 else 0.0
        sum2 = float(int2.sum()) if int2.size > 0 else 0.0
        mean1 = float(int1.mean()) if int1.size > 0 else np.nan
        mean2 = float(int2.mean()) if int2.size > 0 else np.nan

        # Manders M1: fraction of channel1 (within this object) that overlaps channel2 presence
        if sum1 != 0:
            mask_ch2_present = int2 > manders_threshold_ch2
            M1 = int1[mask_ch2_present].sum() / sum1
        else:
            M1 = np.nan

        # Manders M2: fraction of channel2 (within this object) that overlaps channel1 presence
        if sum2 != 0:
            mask_ch1_present = int1 > manders_threshold_ch1
            M2 = int2[mask_ch1_present].sum() / sum2
        else:
            M2 = np.nan

        manders_records.append({
            "particle_id": pid,
            "area_pixels": int(p1.area),
            "mean_ch1": mean1,
            "mean_ch2": mean2,
            "sum_ch1": sum1,
            "sum_ch2": sum2,
            "manders_M1_ch1_coloc_with_ch2": M1,
            "manders_M2_ch2_coloc_with_ch1": M2
        })

    df_manders = pd.DataFrame.from_records(manders_records)

    # Save manders
    if save:
        manders_csv = f"{csv_path}_manders.csv"
        df_manders.to_csv(manders_csv, index=False)


# -------------------------
# 6) Cytofluorogram and object-based cytofluorogram 
# -------------------------

def cytofluorogram_channels(ch1_list, ch2_list, ch1_name, ch2_name, bins=300, cmap='magma', perc=1, save_path_cytofluo = False):
    """Plot one cytofluorogram from several images (same channels)."""
    
    ch1f_list = []
    ch2f_list = []
    
    for ch1, ch2 in zip(ch1_list, ch2_list):
        ch1f = ch1.astype(float, copy=True)
        ch2f = ch2.astype(float, copy=True)
        mn1, mx1 = np.nanmin(ch1f), np.nanmax(ch1f)
        mn2, mx2 = np.nanmin(ch2f), np.nanmax(ch2f)
        denom1 = (mx1 - mn1) if (mx1 - mn1) != 0 else 1.0
        denom2 = (mx2 - mn2) if (mx2 - mn2) != 0 else 1.0
        ch1f = (ch1f - mn1) / denom1
        ch2f = (ch2f - mn2) / denom2
        ch1f_list.append(ch1f)
        ch2f_list.append(ch2f)
    
    # concatenate all images
    a = np.concatenate([x.ravel() for x in ch1f_list]) #3D array into 1D array 
    b = np.concatenate([x.ravel() for x in ch2f_list])

    # simple percentile filter to remove background
    thr_a, thr_b = np.percentile(a, perc), np.percentile(b, perc)
    mask = (a > thr_a) | (b > thr_b)
    a, b = a[mask], b[mask]

    # plot
    plt.figure()
    plt.hexbin(a, b, gridsize=bins, cmap=cmap, bins='log', mincnt=1)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(f'{ch1_name} Intensity')
    plt.ylabel(f'{ch2_name} Intensity')
    plt.title(f'{plot_names[idx]}')
    #plt.colorbar(label='log10(count)')
    plt.colorbar(label='(count)')
    #plt.axis('equal')
    plt.tight_layout()
    if save_path_cytofluo != False:
        plt.savefig(f'{save_path_cytofluo}/cytofluo_{ch1_name}_{ch2_name}_{plot_names[idx]}.tiff', format='tiff', dpi=600)
    plt.show()

def cytofluorogram_labels_multi(ch1_list, ch2_list, lab1_list, lab2_list, ch1_name, ch2_name,
                                bins=300, cmap='magma', save_path_cytofluo = False):
    """
    Plot cytofluorogram based on mean intensities of labeled objects
    from multiple images (same two channels).

    ch1_list, ch2_list : list of 2D or 3D arrays
        Intensity images for channel 1 and 2.
    lab1_list, lab2_list : list of 2D or 3D label arrays
        Label maps for channel 1 and 2 (same shape as intensity images).
    """
    mean1_all, mean2_all = [], []
    
    a_ = []
    b_ = []
    for ch1, ch2, lab1, lab2 in zip(ch1_list, ch2_list, lab1_list, lab2_list):
        
        ch1f = ch1.astype(float, copy=True)
        ch2f = ch2.astype(float, copy=True)
        mn1, mx1 = np.nanmin(ch1f), np.nanmax(ch1f)
        mn2, mx2 = np.nanmin(ch2f), np.nanmax(ch2f)
        denom1 = (mx1 - mn1) if (mx1 - mn1) != 0 else 1.0
        denom2 = (mx2 - mn2) if (mx2 - mn2) != 0 else 1.0
        ch1f = (ch1f - mn1) / denom1
        ch2f = (ch2f - mn2) / denom2
        
        mask_union = (lab1[0] > 0) | (lab2[0] > 0)
        
        a_.append(ch1f[mask_union])
        b_.append(ch2f[mask_union])
        
    a = np.concatenate([x.ravel() for x in a_]) #3D array into 1D array 
    b = np.concatenate([x.ravel() for x in b_])

    # simple plot
    plt.figure()
    plt.hexbin(a, b, gridsize=bins, cmap=cmap, bins='log', mincnt=1)
    plt.xlabel(f'{ch1_name} objects channel intensity')
    plt.ylabel(f'{ch2_name} objects channel intensity')
    plt.title(f'{plot_names[idx]}')
    plt.colorbar(label='log10(count)')
    #plt.axis('equal')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    if save_path_cytofluo != False:
        plt.savefig(f'{save_path_cytofluo}/Obj_cytofluo_{ch1_name}_{ch2_name}_{plot_names[idx]}.tiff', format='tiff', dpi=600)
    plt.show()

#------Parameters---------------
blur_sigma = 1
view_napari = False
save = False

ExperDirectory = r"D:\manuscripts\SG_GEM paper\RK_SG_FXR1_DDX3_G3BP1_DAPI_Leica\cf640r_al488_mCherry_DAPI"
experiments = [f for f in os.listdir(ExperDirectory)
               if os.path.isdir(os.path.join(ExperDirectory, f)) and 'blur' not in f]

# experiments = ['G3BP1_DDX3_DMSO',
#   'G3BP1_DDX3_DMSO_NaAr',
#   'G3BP1_DDX3_RK33',
#   'G3BP1_DDX3_RK33_NaAr',
#   'G3BP1_FXR1_DMSO',
#   'G3BP1_FXR1_DMSO_NaAr',
#   'G3BP1_FXR1_RK33',
#   'G3BP1_FXR1_RK33_NaAr']

# plot_names = [
#   'Cntrl',
#   r'NaArO2$_2$',
#   'RK33',
#   'Combo',
#   'Cntrl',
#   r'NaArO2$_2$',
#   'RK33',
#   'Combo']

# ch1_name = ['DDX3', 'DDX3', 'DDX3', 'DDX3', 'FXR1', 'FXR1', 'FXR1', 'FXR1']



experiments = [
  'G3BP1_DDX3_DMSO_NaAr',
  'G3BP1_DDX3_RK33',
  'G3BP1_DDX3_RK33_NaAr',
  'G3BP1_FXR1_DMSO',
  'G3BP1_FXR1_DMSO_NaAr',
  'G3BP1_FXR1_RK33',
  'G3BP1_FXR1_RK33_NaAr']

plot_names = [
  r'NaArO2$_2$',
  'RK33',
  'Combo',
  'Cntrl',
  r'NaArO2$_2$',
  'RK33',
  'Combo']

ch1_name = ['DDX3', 'DDX3', 'DDX3', 'FXR1', 'FXR1', 'FXR1', 'FXR1']


ch2_name = 'G3BP1 Al488'
ch4_name = 'G3BP1 mCherry'

#parameters for figures
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

for idx, experiment in enumerate(experiments):
    
    print(experiment)
    
    save_folder = f'{experiment}_blursigma1_zStackFilter_2replicas'
    directory = f'{ExperDirectory}/{experiment}/'
    
    os.makedirs(f'{ExperDirectory}/{save_folder}/', exist_ok=True)  
    tiff_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    
    ch1_list = [] #cytofluorogram
    ch2_list = [] #cytofluorogram
    lbl1_list =[]
    lbl2_list = []
    cell_number = 0
    
    ch4_list = [] #cytofluorogram mCherry
    lbl4_list = []        #mCherry
    lbl2_f_list = [] #filtered list G3BP1 Al488 signal without mCherry signal
    
    for tiff_file in tiff_files:
        
        print(tiff_file)
        
        file_name = os.path.splitext(tiff_file)[0] 
        path = f'{ExperDirectory}/{experiment}/{tiff_file}' 
        save_path = f'{ExperDirectory}/{save_folder}/{file_name}_segmentation.tif' 
        
        img = tifffile.imread(path)
        
        # ch3 = img[:, 2, :, :]  # nuclei 
        # ch1 = img[:, 0, :, :]  # FXR1 or DDX3
        # ch2 = img[:, 1, :, :]  # G3BP1
        
        ch3 = img[:, 3, :, :]  # nuclei 
        ch2 = img[:, 2, :, :]  # G3BP1 Alex
        ch1 = img[:, 0, :, :]  # FXR1 or DDX3 
        
        ch4 = img[:, 1, :, :]  # G3BP1 mCherry 
        
        
        # ---- slice filtering based on ch2 percentile ----
        percentile = 90          # robust to background
        relative_thresh = 0.2    # keep slices >= 20% of brightest slice
                
        # per-slice percentile metric
        slice_metric = np.percentile(ch2, percentile, axis=(1, 2))
        
        # reference (brightest slice)
        max_metric = slice_metric.max()
        if max_metric == 0:
            max_metric = 1.0
        
        # slice mask
        keep_slices = slice_metric >= (relative_thresh * max_metric)
        
        # apply mask to all channels
        ch1 = ch1[keep_slices]
        ch2 = ch2[keep_slices]
        ch3 = ch3[keep_slices]
        
        ch4 = ch4[keep_slices]
        
        #---_END OF FILTER-----------
        
        #Nuclei Detectiom
        blurred_nuclei =  preprocess_blur(ch3, sigma=1)
        
        mask_nuclei = blurred_nuclei > threshold_otsu(blurred_nuclei)
        mask_nuclei = binary_fill_holes(mask_nuclei)
        mask_nuclei = remove_small_objects(mask_nuclei, min_size = 500)
        
        labels_nuclei, _ = ndi.label(mask_nuclei)
        cell_number += _ #cell number per experiment
        
        ch3 = ch3 * mask_nuclei
        
        if view_napari:
            viewer = napari.Viewer()
            viewer.add_image(ch3, name='ch3_original', colormap='gray')
            viewer.add_image(mask_nuclei , name='markers', colormap='viridis', opacity=0.6)
            napari.run()
        
        #SG detection 
        ch1_list.append(ch1)
        ch2_list.append(ch2)
        
        ch4_list.append(ch4)         #mCherry
        
        labels_ch1 = segmentation_LoG(ch1)
        labels_ch2 = segmentation_LoG(ch2)
        
        labels_ch4 = segmentation_LoG(ch4)
        
        lbl1_list.append(labels_ch1)
        lbl2_list.append(labels_ch2)
        
        lbl4_list.append(labels_ch4)         #mCherry
        
        save_segmentation([labels_ch1[0], labels_ch2[0], labels_nuclei], save_path)
        save_segmentation([labels_ch1[0], labels_ch4[0], labels_ch2[0], labels_nuclei], save_path)   #mCherry
        
        #---------"cf640r_al488_DAPI" COMMANDS------------------------ 
        # save_csv_ch1 = f'{ExperDirectory}/{save_folder}/{file_name}_ch1.csv'
        # save_SGs_param(labels_ch1, label(mask_nuclei), ch1, save_csv_ch1)
        
        # save_csv_ch2 = f'{ExperDirectory}/{save_folder}/{file_name}_ch2.csv'
        # save_SGs_param(labels_ch2, label(mask_nuclei), ch2, save_csv_ch2)
        
        # mask_union = (labels_ch1[0] > 0) | (labels_ch2[0] > 0)
        # lab_union = label(mask_union)[0]   # [0] is the labeled array
        
        # save_csv_ch1_col = f'{ExperDirectory}/{save_folder}/{file_name}_coloc'
        # save_SGs_coloc_param(label(mask_union), labels_ch2, ch1, ch2, save_csv_ch1_col)
        
        # save_csv_ch1 = f'{ExperDirectory}/{save_folder}/{file_name}_ch1inch2.csv'
        # save_SGs_param(labels_ch1, label(mask_nuclei), ch2, save_csv_ch1)
        
        # save_csv_ch2 = f'{ExperDirectory}/{save_folder}/{file_name}_ch2inch1.csv'
        # save_SGs_param(labels_ch2, label(mask_nuclei), ch1, save_csv_ch2)
        
        # save_path = f'{ExperDirectory}/{save_folder}/{file_name}_segmentation_union.tif' 
        # save_segmentation([lab_union, labels_nuclei], save_path)
        
        #---------"cf640r_al488_mCherry_DAPI" COMMANDS------------------------ 
        
        #save SG parameters based on antibody signal
        save_csv_ch1 = f'{ExperDirectory}/{save_folder}/{file_name}_CF640R.csv'
        save_SGs_param(labels_ch1, label(mask_nuclei), ch1, save_csv_ch1)
        
        save_csv_ch2 = f'{ExperDirectory}/{save_folder}/{file_name}_Al488.csv'
        save_SGs_param(labels_ch2, label(mask_nuclei), ch2, save_csv_ch2)
        
        save_csv_ch4 = f'{ExperDirectory}/{save_folder}/{file_name}_mCherry.csv'
        save_SGs_param(labels_ch4, label(mask_nuclei), ch4, save_csv_ch4)
        
        #Define United Masks
        mask_union = (labels_ch1[0] > 0) | (labels_ch2[0] > 0) #CF640R & AL488
        mask_union_2 = (labels_ch1[0] > 0) | (labels_ch4[0] > 0) #CF640R & mCherry
        mask_union_3 = (labels_ch2[0] > 0) | (labels_ch4[0] > 0) #AL488 & AL488
        lab_union = label(mask_union)[0]   # [0] is the labeled array "CF640R & AL488"
        
        #Manders Coefficients and Signal Metrics
        save_csv_ch1_col = f'{ExperDirectory}/{save_folder}/{file_name}_coloc_ch1CF640R_ch2Al488'
        save_SGs_coloc_param(label(mask_union), labels_ch2, ch1, ch2, save_csv_ch1_col)
        
        save_csv_ch4_col = f'{ExperDirectory}/{save_folder}/{file_name}_coloc_ch1CF640R_ch2mCherry'
        save_SGs_coloc_param(label(mask_union_2), labels_ch4, ch1, ch4, save_csv_ch4_col)
        
        save_csv_ch2ch4_col = f'{ExperDirectory}/{save_folder}/{file_name}_coloc_ch1Al488_ch2mCherry'
        save_SGs_coloc_param(label(mask_union_3), labels_ch4, ch2, ch4, save_csv_ch2ch4_col)
        
        #Signal in One Segmented Channel from Another
        save_csv_ch1 = f'{ExperDirectory}/{save_folder}/{file_name}_CF640RinAl488.csv'
        save_SGs_param(labels_ch1, label(mask_nuclei), ch2, save_csv_ch1)
        
        save_csv_ch2 = f'{ExperDirectory}/{save_folder}/{file_name}_Al488inCF640R.csv'
        save_SGs_param(labels_ch2, label(mask_nuclei), ch1, save_csv_ch2)
        
        save_csv_ch1_ch4 = f'{ExperDirectory}/{save_folder}/{file_name}_CF640RinmCherry.csv'
        save_SGs_param(labels_ch1, label(mask_nuclei), ch4, save_csv_ch1_ch4)
        
        save_csv_ch4 = f'{ExperDirectory}/{save_folder}/{file_name}_mCherryinCF640R.csv'
        save_SGs_param(labels_ch4, label(mask_nuclei), ch1, save_csv_ch4)
        save_SGs_param(labels_ch4, label(mask_nuclei), ch2, f'{ExperDirectory}/{save_folder}/{file_name}_mCherryinAl488.csv')
        save_SGs_param(labels_ch2, label(mask_nuclei), ch4, f'{ExperDirectory}/{save_folder}/{file_name}_Al488inmCherry.csv')
        
        #Save United Segmentation (CF640R & AL488)
        save_path = f'{ExperDirectory}/{save_folder}/{file_name}_segmentation_union.tif' 
        save_segmentation([lab_union, labels_nuclei], save_path)
        
        
        #Code to Filter Out Ch4 (mCherry) Objects from Ch2 (AL488)
        ch2_lbl, ch2_n = label(labels_ch2[0] > 0)
        ch4_lbl, ch4_n = label(labels_ch4[0] > 0)
        
        keep_mask = np.zeros_like(ch2_lbl, dtype=bool)
        
        for lab in range(1, ch2_n + 1):
            obj_mask = (ch2_lbl == lab)
            # if any voxel of this ch2 object overlaps ch4, skip it
            if not np.any(ch4_lbl[obj_mask] > 0):
                keep_mask[obj_mask] = True
        
        labels_ch2_noch4 = label(keep_mask)
        lbl2_f_list.append(labels_ch2_noch4)
        #corresponding SG parameters
        save_SGs_param(labels_ch2_noch4, label(mask_nuclei), ch2, f'{ExperDirectory}/{save_folder}/{file_name}_488AL_only.csv')
        
        #corresponding coloc
        mask_union = (labels_ch1[0] > 0) | (labels_ch2_noch4[0] > 0) 
        save_csv_col = f'{ExperDirectory}/{save_folder}/{file_name}_coloc_ch1_CF640R_ch2_488AL_mask_only'
        save_SGs_coloc_param(label(mask_union), labels_ch2_noch4, ch1, ch2, save_csv_col)
        
        
    #---------"cf640r_al488_DAPI" COMMANDS FOR CYTOFLUOROGRAM------------------------ 
    # save_path_cytofluo = f'{ExperDirectory}/{save_folder}'
    # cytofluorogram_channels(ch1_list, ch2_list, ch1_name[idx], ch2_name)
    # cytofluorogram_labels_multi(ch1_list, ch2_list, lbl1_list, lbl2_list, ch1_name, ch2_name)
    
    #---------"cf640r_al488_mCherry_DAPI" COMMANDS FOR CYTOFLUOROGRAM------------------------ 
    
    save_path = f'{ExperDirectory}/{save_folder}'
    cytofluorogram_channels(ch1_list, ch4_list, ch1_name[idx], ch4_name, save_path_cytofluo=save_path)
    cytofluorogram_labels_multi(ch1_list, ch4_list, lbl1_list, lbl4_list, ch1_name[idx], ch4_name)
    
    cytofluorogram_channels(ch1_list, ch2_list, ch1_name[idx], ch2_name, save_path_cytofluo=save_path)
    cytofluorogram_labels_multi(ch1_list, ch2_list, lbl4_list, lbl2_f_list, ch1_name[idx], ch2_name)
    
    cytofluorogram_channels(ch1_list, ch4_list, ch1_name[idx], ch4_name, save_path_cytofluo=save_path)
    cytofluorogram_labels_multi(ch1_list, ch4_list, lbl1_list, lbl4_list, ch1_name[idx], ch4_name)
    
    cytofluorogram_channels(ch4_list, ch2_list, ch4_name, ch2_name, save_path_cytofluo=save_path)
    cytofluorogram_labels_multi(ch4_list, ch2_list, lbl4_list, lbl2_list, ch4_name, ch2_name)

meta_data = {"ch1:": "FXR1orDDX3 cf640", 
             "ch2:": "G3BP1 al488",
             "ch3:": "DAPI",
             "ch4:": "G3BP1 mCherry"}
df = pd.DataFrame(list(meta_data.items()), columns=["Channel", "Marker"])
df.to_csv(f"{ExperDirectory}/{save_folder}/meta_data.csv", index=False)