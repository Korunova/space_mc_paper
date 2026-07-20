# preprocessing.py
"""
Minimal preprocessing utilities for SPT tracks:
- load_files
- clean_and_convert
- interpolate_track
- interpolate_and_filter
- classify_track_using_mask
- save_preprocessed / load_preprocessed

Drop into your project and import functions:
from preprocessing import preprocess_file_from_df, save_preprocessed


Notes:
    - I checked the track opening in preprocessing and main code. Works the same
    - Preprocessing file works even more robust for particle tagging 
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import os
import json
from tifffile import imread
import math
import numpy as np
from skimage.draw import disk as draw_disk
from skimage.morphology import binary_dilation, disk as selem_disk
from skimage.util import img_as_bool
from skimage.measure import label, regionprops

# ---------- Simple config (edit in one place) ----------
CONFIG = {
    "um_per_pixel": 0.1465 * 40 / 100, 
    "delta_x": 0.005,   # interpolation dt [s]
    "tau": 0.01,
    "min_trajectory_length": 10,  # points after interpolation
    "cache_dir": "./preproc_cache", #save in the preproc_cache in the working folder
}
os.makedirs(CONFIG["cache_dir"], exist_ok=True)

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
def load_csv_tracks(path: str) -> pd.DataFrame:
    """Read CSV track file into DataFrame (expects TrackID, frame, x, y columns)."""
    df = pd.read_csv(path)
    return df

def load_segmentation_tif(path: str, nuc_mask = False) -> Dict[str, np.ndarray]:
    """
    Load segmentation tif with layers [SG_mask, cyto_mask] or similar.
    Returns dict {'SG': bool_array, 'cyto': bool_array}.
    If file missing/invalid, returns empty dict.
    """
    if not os.path.exists(path):
        return {}
    arr = imread(path)
    # if arr.ndim == 2 and not nuc_mask:
    #     # single mask -> treat as SG
    #     return {"SG": arr.astype(bool)}
    if arr.shape[0] >= 2 and not nuc_mask:
        SG = arr[0].astype(bool)
        cyto = arr[1].astype(bool)
        
        # Add 10-pixel border to cytoplasm mask
        cyto[:10, :] = False      # top
        cyto[-20:, :] = False     # bottom
        cyto[:, :10] = False      # left
        cyto[:, -10:] = False     # right
        
        return {"SG": SG, "cyto": cyto}
    if nuc_mask:
        # single mask -> treat as SG
        return {"Nuclei": arr.astype(bool)}
    return {}

#------------Control Mask Generation---------

def generate_controls(SG_mask, cyto_mask, mode:str = "circle",
                      q1_area: int=100, median_area: int=226, q3_area:int=500,
                      q1_count:int=10,  median_count: int = 16, q3_count:int=22,
                      buffer:int=3, input_area: int=None, n_particles:int=None, seed:int=None, max_attempts:int=1000):
    
    """
    Minimal, robust control particle generator.
    - If n_particles is None, sample uniformly between q1_count and q3_count (inclusive).
      (If you prefer triangular: use rng.triangular(q1_count, median_count, q3_count).)
    - Area is sampled uniformly between q1_area and q3_area (inclusive).
      (If you prefer triangular around median_area, uncomment the triangular line.)
    Returns: control_mask # particles_list can be added
    """

    """
    mode="circle":
        Place random circular controls, like your current function.
    
    mode="sg_template":
        Pick an SG-shaped connected component from SG_mask and place a shifted
        copy of that shape elsewhere in cytoplasm, avoiding the original SG mask
        and avoiding overlap with existing controls.
    """
    
    SG = img_as_bool(SG_mask)
    cyto = img_as_bool(cyto_mask)
    h, w = SG.shape
    


    # forbidden = SG dilated by buffer
    forbidden = binary_dilation(SG, selem_disk(buffer)) if buffer > 0 else SG.copy()
    placement = cyto & (~forbidden)
    if not placement.any():
        raise ValueError("No placement area (cyto minus buffered SG is empty).")

    coords = np.column_stack(np.nonzero(placement))
    control_mask = np.zeros_like(SG, dtype=bool)
    particles = []
    attempts = 0
    
    # Precompute SG templates if needed
    sg_templates = []
    if mode == "sg_template":
        labeled = label(SG)
        for reg in regionprops(labeled):
            if reg.area > 0:
                sg_templates.append(reg.coords)
    
        if len(sg_templates) == 0:
            raise ValueError("No SG components found in SG_mask for sg_template mode.")
    
    rng = np.random.default_rng(seed)
    
    if mode == "circle":
        
        if n_particles is None:
           # Or n_particles = int(rng.integers(q1_count, q3_count + 1))  # uniform discrete
           n_particles = int(round(rng.triangular(q1_count, median_count, q3_count)))
      
        while len(particles) < n_particles and attempts < max_attempts:
            attempts += 1 
            
            if input_area is None:
                area = float(rng.triangular(q1_area, median_area, q3_area))
            else: 
                area =  input_area
                
            # OR area = np.random.randint(q1_area, q3_area+1) 
            r = math.sqrt(area / math.pi)
            r_int = max(1, int(math.ceil(r)))
            margin = r_int + buffer
            
            # quick candidate pick respecting image borders
            viable = coords[
                (coords[:,0] >= margin) & (coords[:,0] < h - margin) &
                (coords[:,1] >= margin) & (coords[:,1] < w - margin)
            ]
            
            if viable.size == 0:
                continue
            
            cy, cx = viable[rng.integers(len(viable))]
            rr, cc = draw_disk((int(cy), int(cx)), r_int, shape=SG.shape)
            
            if not cyto[rr, cc].all():        # disk must be fully inside cyto
                continue
            if SG[rr, cc].any():             # must not touch SG
                continue
            
            # avoid overlapping existing controls (with buffer)
            if buffer > 0:
                rr2, cc2 = draw_disk((int(cy), int(cx)), r_int + buffer, shape=SG.shape)
                if control_mask[rr2, cc2].any():
                    continue
            
            else:
                if control_mask[rr, cc].any():
                    continue
            control_mask[rr, cc] = True
            particles.append((int(cy), int(cx), "circle" ,r, area))
    
    elif mode == "sg_template":

        for template in sg_templates:
    
            # template bounding box size
            ty = template[:, 0]
            tx = template[:, 1]
            min_y, max_y = ty.min(), ty.max()
            min_x, max_x = tx.min(), tx.max()
            th = max_y - min_y + 1
            tw = max_x - min_x + 1
            
            placed = False
            
            while placed == False and attempts < max_attempts:
                attempts += 1 
                
                # choose a random top-left placement position
                max_y0 = h - th
                max_x0 = w - tw
                if max_y0 < 0 or max_x0 < 0:
                    continue
        
                # try a few random placements for this attempt

                y0 = rng.integers(0, max_y0 + 1)
                x0 = rng.integers(0, max_x0 + 1)
    
                rr = template[:, 0] - min_y + y0
                cc = template[:, 1] - min_x + x0
    
                # must be inside cytoplasm
                if not cyto[rr, cc].all():
                    continue
    
                # avoid original SG region
                if SG[rr, cc].any():
                    continue
    
                # avoid overlap with existing controls
                if control_mask[rr, cc].any():
                    continue
    
                # optional buffer around placed object
                if buffer > 0:
                    temp_mask = np.zeros_like(SG, dtype=bool)
                    temp_mask[rr, cc] = True
                    temp_dil = binary_dilation(temp_mask, footprint=selem_disk(buffer))
                    if control_mask[temp_dil].any():
                        continue
                    if SG[temp_dil].any():
                        continue
    
                control_mask[rr, cc] = True
                
                template_area = len(rr)
                
                particles.append((int(y0), int(x0), "sg_template", template_area))
                placed = True
        
    if len(particles) < n_particles and mode == "circle":
        print(f"Warning: placed {len(particles)}/{n_particles} after {attempts} attempts.")
    return control_mask #, particles

# -------------------------
# Example usage:
# control_mask, particles = generate_controls(SG_mask, cyto_mask, seed=42, buffer=3)
# -------------------------

# ---------- Cleaning & conversion ----------
def clean_and_convert(tracks_df: pd.DataFrame, config: Dict = CONFIG) -> pd.DataFrame:
    """
    Convert x,y from pixels -> um, ensure numeric frames, sort, drop invalid rows.
    Required columns: TrackID, frame, x, y
    """
    required = ["TrackID", "frame", "x", "y"]
    if not all(c in tracks_df.columns for c in required):
        raise ValueError(f"tracks_df missing required columns: {required}")
    df = tracks_df.copy()
    df = df.dropna(subset=["x", "y", "frame"])
    df["x"] = df["x"].astype(float) 
    df["y"] = df["y"].astype(float) 
    df["frame"] = df["frame"].astype(float)
    df = df.sort_values(["TrackID", "frame"])
    return df

# ---------- Interpolation ----------
def interpolate_track(time, x, y, dt):
    if len(time) < 2:
        return np.array([]), np.array([]), np.array([])
    t0 = float(time[0])
    t_u = [t0]
    point = t0
    while point <= float(time[-1]):
        point = point + dt
        t_u.append(point)
    t_u = np.array(t_u)
    x_u = np.interp(t_u, time, x)
    y_u = np.interp(t_u, time, y)
    return t_u, x_u, y_u

# ---------- Classification ----------
def classify_track_using_mask(x_u: np.ndarray, y_u: np.ndarray, masks: Dict[str, np.ndarray], um_per_pixel: float) -> Dict[str, bool]:
    """Return simple flags: inside_SG_all, touches_SG_any, inside_cyto_all."""
    if not masks:
        return {"inside_SG_all": False, "touches_SG_any": False, "inside_cyto_all": True, "inside_control_any": False}
    # H, W = next(iter(masks.values())).shape
    # x_pix = np.clip(np.round(x_u / um_per_pixel).astype(int), 0, W - 1)
    # y_pix = np.clip(np.round(y_u / um_per_pixel).astype(int), 0, H - 1)

    x_pix = (x_u / um_per_pixel).astype(int)
    y_pix = (y_u / um_per_pixel).astype(int)
    result = {}
    if "SG" in masks:
        sg = masks["SG"]
        inside = sg[y_pix, x_pix]
        result["inside_SG_all"] = bool(inside.all()) and len(inside) > 0
        result["touches_SG_any"] = bool(inside.any()) and len(inside) > 0
    else:
        result["inside_SG_all"] = False
        result["touches_SG_any"] = False
    
    if "control" in masks:
        control = masks["control"]
        inside_control = control[y_pix, x_pix]  
        result["inside_control_all"] = bool(inside_control.all()) and len(inside_control) > 0
        result["inside_control_any"] = bool(inside_control.any()) and len(inside_control) > 0
        
    else:
        result["inside_control_all"] = False
        result["inside_control_any"] = False

    if "cyto" in masks:
        cy = masks["cyto"]
        inside_c = cy[y_pix, x_pix]  
        if not inside_c.all():
            return None
        result["inside_cyto_all"] = bool(inside_c.all()) and len(inside_c) > 0
        
    else:
        result["inside_cyto_all"] = True
        
    return result

#----------Crop Tracks in Mask---------------
def tracks_in_mask(tracks, masks: Dict[str, np.ndarray], mode: str, um_per_pixel: float, min_len=10):
    out = []
    
    template_label = label(masks[mode])

    for prop in regionprops(template_label):
        area_pixels = prop.area
        area_um2 = area_pixels * (um_per_pixel ** 2)
        
        lab = prop.label
        coords = prop.coords  # (row, col) pixels belonging to this SG

        # faster lookup for this label
        sg_pixels = set(map(tuple, coords))

        for tr in tracks:
            yi = np.round(tr.y/um_per_pixel).astype(int)
            xi = np.round(tr.x/um_per_pixel).astype(int)
            
            inside = np.array(
                [(y, x) in sg_pixels for y, x in zip(yi, xi)],
                dtype=bool
            )

            idx = np.where(inside)[0]
            if len(idx) == 0:
                continue
            
            new_flags = dict(tr.flags)
            new_flags["area_pixels"] = float(area_pixels)
            new_flags["area_um2"] = float(area_um2)
            new_flags["sg_label"] = int(lab)

            for n, seg in enumerate(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)):
                
                new_flags["segment_id"] = n
                
                if len(seg) >= min_len:
                    out.append(
                        PreprocessedTrack(
                            track_id=f"{tr.track_id}_{lab}_{n}",
                            t=tr.t[seg],
                            x=tr.x[seg],
                            y=tr.y[seg],
                            flags = new_flags,
                            #flags=tr.flags,
                            source_file=tr.source_file
                        )
                    )

    return out

#-----------Add Label Area to Track Flags--------
def add_label_area_to_tracks(
    tracks,
    masks: Dict[str, np.ndarray],
    mode: str,
    um_per_pixel: float
):
    out = []
    template_label = label(masks[mode])

    for prop in regionprops(template_label):
        area_pixels = prop.area
        area_um2 = area_pixels * (um_per_pixel ** 2)
        coords = prop.coords  # (row, col) pixels in this region
        sg_pixels = set(map(tuple, coords))

        for tr in tracks:
            yi = np.round(tr.y / um_per_pixel).astype(int)
            xi = np.round(tr.x / um_per_pixel).astype(int)

            inside = np.array(
                [(y, x) in sg_pixels for y, x in zip(yi, xi)],
                dtype=bool
            )

            if not inside.any():
                continue

            new_flags = dict(tr.flags)
            new_flags["area_pixels"] = float(area_pixels)
            new_flags["area_um2"] = float(area_um2)

            out.append(
                PreprocessedTrack(
                    track_id=f"{tr.track_id}",
                    t=tr.t,
                    x=tr.x,
                    y=tr.y,
                    flags=new_flags,
                    source_file=tr.source_file,
                )
            )

    return out

# ---------- Per-track orchestration ----------
def interpolate_and_filter(track_df: pd.DataFrame, dt: float, min_length: int,
                           masks: Dict[str, np.ndarray], source_file: str, config: Dict = CONFIG) -> Optional[PreprocessedTrack]:
    """Interpolate single track, classify and filter by min_length."""
    track_id = int(track_df["TrackID"].iloc[0])
    t = track_df["frame"].to_numpy(dtype=float)
    x = track_df["x"].to_numpy(dtype=float)
    y = track_df["y"].to_numpy(dtype=float)
    t_u, x_u, y_u = interpolate_track(t, x, y, dt)
    if t_u.size == 0 or t_u.size < min_length:
        return None
    flags = classify_track_using_mask(x, y, masks, config["um_per_pixel"])
    if flags is None:
        return None
    
    return PreprocessedTrack(track_id=track_id, t=t_u, x=x_u, y=y_u, flags=flags, source_file=source_file)

# ---------- Batch preprocessing ----------
def preprocess_file_from_df(tracks_df: pd.DataFrame, masks: Dict[str, np.ndarray] = None,
                            source_file: str = "in-memory", crop: bool = False, mode: str = "SG",
                            config: Dict = CONFIG) -> Tuple[List[PreprocessedTrack], Dict]:
    """Main orchestrator. Returns list of PreprocessedTrack and QC summary."""
    df = clean_and_convert(tracks_df, config=config)
    preproc = []
    before = int(df["TrackID"].nunique())
    for tid, g in df.groupby("TrackID"):
        pp = interpolate_and_filter(g, dt=config["delta_x"], min_length=config["min_trajectory_length"],
                                    masks=(masks or {}), source_file=source_file, config=config)
        if pp is not None:
            preproc.append(pp)
    
    if crop:
        preproc = tracks_in_mask(preproc, masks, mode, config["um_per_pixel"])
    
    after = len(preproc)
    median_len = int(np.median([p.t.size for p in preproc]) if preproc else 0)
    summary = {"n_tracks_before": before, "n_tracks_after": after, "median_length": median_len}
    return preproc, summary

# ---------- Cache helpers ----------
def save_preprocessed(preproc_list: List[PreprocessedTrack], out_npz: str):
    data = {}
    meta = []
    for tt in preproc_list:
        key = f"track_{tt.track_id}"
        data[f"{key}_t"] = tt.t
        data[f"{key}_x"] = tt.x
        data[f"{key}_y"] = tt.y
        meta.append({"track_id": tt.track_id, "flags": tt.flags, "source_file": tt.source_file})
    data["meta"] = np.array(json.dumps(meta), dtype=object)
    np.savez_compressed(out_npz, **data)

def load_preprocessed(npz_path: str) -> List[PreprocessedTrack]:
    z = np.load(npz_path, allow_pickle=True)
    meta_json = z["meta"].tolist()
    meta = json.loads(meta_json)
    preproc = []
    for m in meta:
        tid = m["track_id"]
        t = z[f"track_{tid}_t"]
        x = z[f"track_{tid}_x"]
        y = z[f"track_{tid}_y"]
        flags = m["flags"]
        preproc.append(PreprocessedTrack(track_id=tid, t=t, x=x, y=y, flags=flags, source_file=m.get("source_file", "")))
    return preproc

# ---------- Simple demo when run directly ----------
def _create_synthetic():
    t1 = np.array([0.0, 0.02, 0.05, 0.08, 0.12])
    x1 = np.array([10, 10.2, 10.5, 10.4, 10.7])
    y1 = np.array([20, 20.1, 20.4, 20.5, 20.7])
    t2 = np.array([0.01, 0.03, 0.06, 0.09])
    x2 = np.array([50, 50.2, 50.5, 50.7])
    y2 = np.array([60, 60.1, 60.6, 60.9])
    rows = []
    for i in range(len(t1)):
        rows.append({"TrackID": 1, "frame": t1[i], "x": x1[i], "y": y1[i]})
    for i in range(len(t2)):
        rows.append({"TrackID": 2, "frame": t2[i], "x": x2[i], "y": y2[i]})
    df = pd.DataFrame(rows)
    H, W = 200, 200
    sg_mask = np.zeros((H, W), dtype=bool)
    sg_mask[18:26, 8:16] = True
    cyto_mask = np.ones((H, W), dtype=bool)
    masks = {"SG": sg_mask, "cyto": cyto_mask}
    return df, masks

if __name__ == "__main__":
    # quick demonstration
    df, masks = _create_synthetic()
    preproc, summary = preprocess_file_from_df(df, masks=masks, source_file="synthetic", config=CONFIG)
    print("QC summary:", summary)
    for p in preproc:
        print(f"Track {p.track_id}: points={p.t.size}, flags={p.flags}")
    cache_path = os.path.join(CONFIG["cache_dir"], "synthetic.npz")
    save_preprocessed(preproc, cache_path)
    print("Saved cache:", cache_path)
    loaded = load_preprocessed(cache_path)
    print("Loaded tracks:", [(t.track_id, t.t.size) for t in loaded])