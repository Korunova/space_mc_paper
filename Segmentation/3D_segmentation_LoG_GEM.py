# -*- coding: utf-8 -*-
"""
===============================================================================
3D Particle Segmentation and Spatial Relationship Analysis
===============================================================================

Description
-----------
This script performs automated 3D segmentation of fluorescent particles and
stress granules from volumetric microscopy images. Particles are detected using
a Laplacian-of-Gaussian (LoG)-based segmentation pipeline, followed by instance
labeling and optional watershed refinement.

The segmented objects are used to quantify the spatial organization of
particles relative to stress granules, including particle densities within
stress granule subregions and distances from stress granule boundaries.

Main workflow
-------------
1. Load multichannel 3D fluorescence microscopy images.
2. Segment particles and stress granules using LoG-based detection.
3. Generate labeled segmentation masks.
4. Quantify particle density within:
   - entire stress granules,
   - stress granule cores,
   - stress granule shells,
   - surrounding cytoplasm.
5. Measure particle distances relative to stress granule boundaries.
6. Save segmentation masks and quantitative analysis tables.

Outputs
-------
- Instance-labeled segmentation masks (OME-TIFF)
- Particle-to-stress granule relationship tables
- Radial density measurements
- Optional Napari visualization for quality control

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
# 6) Save Segmentatin as Stack 
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
# 7) Radial Shell Analysis
# -------------------------

def sg_radial_density_df(labels_particles, labels_sg, ero_iter=2, dil_iter=2, voxel_volume = 0.1137476 * 0.1137476 * 0.1001359):
    """
    labels_particles: instance labels for particles (0 = background)
    labels_sg: instance labels for SGs (0 = background)
    file_name: string to store in output dataframe
    ero_iter: number of binary erosion iterations for SG core
    dil_iter: number of binary dilation iterations for outside neighborhood
    """
    
    if labels_particles.shape != labels_sg.shape:
        raise ValueError("labels_particles and labels_sg must have the same shape")

    # Particle centroids
    particle_ids = np.unique(labels_particles)
    particle_ids = particle_ids[particle_ids != 0]

    if len(particle_ids) == 0:
        return pd.DataFrame(columns=["file", "SG_index", "density_core", "density_shell", "density_around_SGs"])

    centroids = ndi.center_of_mass(
        np.ones_like(labels_particles, dtype=np.float32),
        labels_particles,
        particle_ids
    )
    centroids = np.asarray(centroids)
    centroids_idx = np.rint(centroids).astype(int)

    # Keep only valid centroid coordinates
    valid = (
        (centroids_idx[:, 0] >= 0) & (centroids_idx[:, 0] < labels_particles.shape[0]) &
        (centroids_idx[:, 1] >= 0) & (centroids_idx[:, 1] < labels_particles.shape[1]) &
        (centroids_idx[:, 2] >= 0) & (centroids_idx[:, 2] < labels_particles.shape[2])
    )
    centroids_idx = centroids_idx[valid]

    sg_ids = np.unique(labels_sg)
    sg_ids = sg_ids[sg_ids != 0]

    rows = []

    for sg_id in sg_ids:
        sg = (labels_sg == sg_id)

        core = ndi.binary_erosion(sg, iterations=ero_iter)
        shell = sg & ~core

        dil = ndi.binary_dilation(sg, iterations=dil_iter)
        around = dil & (labels_sg == 0)   # removes voxels belonging to other SG labels
        
        sg_vol_vox = sg.sum()
        core_vol = core.sum()
        shell_vol = shell.sum()
        around_vol = around.sum()

        if centroids_idx.size == 0:
            core_count = shell_count = around_count = 0
        else:
            sg_count = sg[tuple(centroids_idx.T)].sum()
            core_count = core[tuple(centroids_idx.T)].sum()
            shell_count = shell[tuple(centroids_idx.T)].sum()
            around_count = around[tuple(centroids_idx.T)].sum()

        rows.append({
            "SG_index": int(sg_id),
            "density_SG": (sg_count / sg_vol_vox) / voxel_volume if sg_vol_vox > 0 else np.nan,
            "SG_Volume": sg_vol_vox * voxel_volume,
            "density_core": (core_count / core_vol)/voxel_volume if core_vol > 0 else np.nan,
            "density_shell": (shell_count / shell_vol)/voxel_volume if shell_vol > 0 else np.nan,
            "density_around_SGs": (around_count / around_vol)/voxel_volume if around_vol > 0 else np.nan,
        })

    return pd.DataFrame(rows)

# -------------------------
# 7) Particle-to-SG Boundary Analysis
# -------------------------


def sg_particle_relation_df(
    labels_particles,
    labels_sg,
    ero_iter=2,
    dil_iter=3,
    voxel_size_um=(0.1001359, 0.1137476, 0.1137476),
):
    """
    Build a particle-to-SG relation table using particle COMs.

    Returns columns:
    - SG_index
    - SG_Volume
    - particle_index
    - status
    - distance_from_SG_surface
    """

    if labels_particles.shape != labels_sg.shape:
        raise ValueError("labels_particles and labels_sg must have the same shape")

    voxel_volume = float(np.prod(voxel_size_um))

    # Particle COMs computed once
    particle_ids = np.unique(labels_particles)
    particle_ids = particle_ids[particle_ids != 0]

    if len(particle_ids) == 0:
        return pd.DataFrame(columns=[
            "SG_index", "SG_Volume", "particle_index",
            "status", "distance_from_SG_surface"
        ])

    coms = np.asarray(
        ndi.center_of_mass(
            np.ones_like(labels_particles, dtype=np.float32),
            labels_particles,
            particle_ids
        )
    )

    # Remove invalid / non-finite COMs
    finite = np.isfinite(coms).all(axis=1)
    particle_ids = particle_ids[finite]
    coms = coms[finite]

    coords = np.rint(coms).astype(int)

    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < labels_particles.shape[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < labels_particles.shape[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < labels_particles.shape[2])
    )
    particle_ids = particle_ids[valid]
    coords = coords[valid]

    # Fast lookup: particle id -> COM voxel index
    particle_coord = {int(pid): tuple(c) for pid, c in zip(particle_ids, coords)}

    sg_ids = np.unique(labels_sg)
    sg_ids = sg_ids[sg_ids != 0]

    rows = []

    for sg_id in sg_ids:
        sg = (labels_sg == sg_id)

        core = ndi.binary_erosion(sg, iterations=ero_iter)
        dilated = ndi.binary_dilation(sg, iterations=dil_iter)
        
        small_sg = not core.any()

        sg_vol_vox = sg.sum()
        sg_vol_um3 = sg_vol_vox * voxel_volume

        # Distance to SG boundary in microns
        inside_dist = ndi.distance_transform_edt(sg, sampling=voxel_size_um)
        outside_dist = ndi.distance_transform_edt(~sg, sampling=voxel_size_um)

        # Only particle labels present inside this SG's dilated neighborhood
        candidate_ids = np.unique(labels_particles[dilated])
        candidate_ids = candidate_ids[candidate_ids != 0]

        for pid in candidate_ids:
            c = particle_coord.get(int(pid))
            if c is None:
                continue

            # Keep only particles whose COM is actually in the dilated region
            if not dilated[c]:
                continue

            if small_sg and sg[c]:
                status = "small SG"
                dist = -inside_dist[c]
            elif core[c]:
                status = "core"
                dist = -inside_dist[c]
            elif sg[c]:
                status = "shell"
                dist = -inside_dist[c]
            else:
                status = "neighborhood"
                dist = outside_dist[c]

            rows.append({
                "SG_index": int(sg_id),
                "SG_Volume": sg_vol_um3,
                "particle_index": int(pid),
                "status": status,
                "distance_from_SG_surface": float(dist),
            })

    return pd.DataFrame(rows)


#------Parameters---------------
blur_sigma = 1
view_napari = True
save = False

ExperDirectory = r"D:\manuscripts\SG_GEM paper\RK_SG_FXR1_DDX3_G3BP1_DAPI_Leica\GEMSapphire_G3BP1StarRed_DAPI"
experiments = [f for f in os.listdir(ExperDirectory)
               if os.path.isdir(os.path.join(ExperDirectory, f)) and 'blur' not in f]

experiments = [
 'STED_zstep008_G3BP1starred_FXR1starorange_500uMNaAr',
 'STED_zstep008_G3BP1starred_FXR1starorange_6uMRK33',
 'STED_zstep008_G3BP1starred_FXR1starorange_100uMNaAr']


plot_names = [
  r'NaArO2$_2$ 500 uM',
  'RK33',
  r'NaArO2$_2$ 100 uM']


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
    
    save_folder_segmentation = f'{ExperDirectory}/{experiment}/segmentation/'
    save_folder_results = f'{ExperDirectory}/{experiment}/results/'
    directory = f'{ExperDirectory}/{experiment}/'
    
    os.makedirs(save_folder_segmentation, exist_ok=True)  
    os.makedirs(save_folder_results, exist_ok=True) 
    
    tiff_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    
    ch1_list = [] #cytofluorogram
    ch2_list = [] #cytofluorogram
    lbl1_list =[]
    lbl2_list = []
    cell_number = 0
    
    for tiff_file in tiff_files:
        
        print(tiff_file)
        
        file_name = os.path.splitext(tiff_file)[0]  
        while file_name.endswith('.tif'):
            file_name = os.path.splitext(file_name)[0]  
        
        path = f'{directory}/{tiff_file}' 
        save_path_segmentation = f'{save_folder_segmentation}/{file_name}_segmentation.tif' 
        save_path_csv = f'{save_folder_results}/{file_name}_particle_relation.csv' 
        
        img = tifffile.imread(path)
        
        #Channel isolation
        ch1 = img[:, 0, :, :]  # GEM
        ch2 = img[:, 1, :, :]  # G3BP1
                
        # Instance segmentatin
        ch1_list.append(ch1)
        ch2_list.append(ch2)
                
        labels_ch1, _ = segmentation_LoG(ch1, sigma_small = 1, max_projection_trsh = 1, view = False)
        labels_ch2, _ = segmentation_LoG(ch2)
        
        if save:
            save_segmentation([labels_ch1, labels_ch2], save_path_segmentation)
            
            # df = sg_particle_relation_df(labels_ch1, labels_ch2)
            # df.to_csv(save_path_csv, index=False)

        if view_napari:
            with napari.gui_qt():
                v = napari.Viewer()
                # show ch1 and ch2 (these are 3D stacks: (z,y,x) or (t,h,w))
                v.add_image(ch1, name='GEM')          # raw channel 1
                v.add_image(labels_ch1, name='labels GEM')          # raw channel 1
                v.add_image(ch2, name='G3BP1')          # raw channel 2
                v.add_image(labels_ch2, name='labels G3BP1')          # raw channel 2
                # add the VOI as labels (convert bool->int labels)
                
        print("Here")
        
        break
    break
        





