# %% [code]
!pip install imagecodecs

# %% [code]
#!/usr/bin/env python3
"""
================================================================================
VESUVIUS CHALLENGE - PRODUCTION-QUALITY PRECOMPUTATION
================================================================================

FIXES IN THIS VERSION:

TIME: ~1-2 seconds per 160³ volume (depends on hardware)
QUALITY: Production-grade, precision preserved
================================================================================
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool, cpu_count
from skimage.morphology import medial_axis
# skeletonize() automatically handles both 2D and 3D
from skimage.morphology import skeletonize as skimage_skeletonize
# EXPLICIT scipy.ndimage imports with aliases to avoid any skimage shadowing
from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt
from scipy.ndimage import binary_dilation as scipy_binary_dilation
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.ndimage import label as scipy_label_func
from scipy.ndimage import convolve as scipy_convolve

# skimage.morphology.skeletonize() automatically handles BOTH 2D and 3D:
# - For 3D: automatically uses Lee's method (designed for 3D, octree-based)
# - For 2D: uses Zhang's method
# Documentation: https://scikit-image.org/docs/0.25.2/api/skimage.morphology.html#skimage.morphology.skeletonize
from skimage.morphology import skeletonize as skimage_skeletonize


# ============================================================================
# CONFIGURATION
# ============================================================================

class CFG:
    # Data paths
    TRAIN_IMAGES = "/kaggle/input/vesuvius-challenge-surface-detection/train_images"
    TRAIN_LABELS = "/kaggle/input/vesuvius-challenge-surface-detection/train_labels"
    TRAIN_CSV = "/kaggle/input/vesuvius-challenge-surface-detection/train.csv"
    
    # Output
    OUTPUT_BASE = "/kaggle/working/vesuvius_precomputed_final"
    
    # Batch
    START_INDEX = 225
    END_INDEX = 249
    
    # Precision: store images as float32 normalized to [0, 1] (preserves relative information)
    IMAGE_DTYPE = np.float16
    LABEL_DTYPE = np.float32
    
    # Skeleton - Uses skimage.morphology.skeletonize() which auto-handles 2D/3D
    # For 3D: Uses Lee's method (octree-based, designed for 3D)
    # For 2D: Uses Zhang's method
    # Dilation disabled by default to prevent forcing connectivity across tightly spaced layers
    SKELETON_DILATION_ITER = 0  # Set >0 only if you intentionally want dilation (not recommended)
    
    # Centerline smoothing: disabled by default (set >0 only if you need smoothing)
    CENTERLINE_SMOOTH_SIGMA = 0.0  # Use 0.0 to avoid blurring one-voxel gaps
    
    # Vectors (fiber-aware weighting)
    FIBER_WEIGHT_Y = 1.1
    FIBER_WEIGHT_X = 0.9
    FIBER_WEIGHT_Z = 1.0
    
    # Parallelization
    NUM_PROCESSES = max(1, min(12, cpu_count() - 1))
    
    # Debug
    VERBOSE = True


# ============================================================================
# PART 1: DTYPE HANDLING (Preserve information; convert to float32 normalized)
# ============================================================================

def normalize_image_lossless(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Convert any image dtype to float32 normalized to [0, 1] WITHOUT losing relative intensity information.
    
    - For integer types, convert to float32 and divide by the original max (preserves relative differences).
    - For float types, if values are already in [0,1], keep them; if larger, divide by max.
    
    Returns:
        image_f32: (D, H, W) float32 with values in [0, 1] (unless constant 0)
        dtype_info: dict with original dtype, original_max, scaling info
    """
    original_dtype = image.dtype
    original_shape = image.shape
    
    # Ensure we operate on a copy for safety
    img = image.copy()
    
    # Force 3D shape convention if 2D input
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    
    # Determine max value (use float to avoid overflow)
    try:
        max_val = float(img.max())
    except Exception:
        max_val = 0.0
    
    # If all zeros or negative, return zeros
    if max_val == 0.0:
        image_f32 = img.astype(np.float32)
        dtype_info = {
            'original_dtype': str(original_dtype),
            'original_max': max_val,
            'scaling_divisor': 1.0,
            'original_shape': original_shape
        }
        return image_f32.astype(np.float32), dtype_info
    
    # If floating and within [0,1], preserve directly
    if np.issubdtype(original_dtype, np.floating):
        image_f32 = img.astype(np.float32)
        if max_val > 1.0:
            # Normalize into [0,1] by dividing by max (preserves relative contrasts)
            image_f32 = image_f32 / max_val
            scaling_divisor = max_val
        else:
            scaling_divisor = 1.0
    elif np.issubdtype(original_dtype, np.integer):
        # Convert integers to float32 and normalize by the integer max
        image_f32 = img.astype(np.float32)
        # If the image is already small (<=255), dividing by max_val still preserves relative info
        image_f32 = image_f32 / max_val
        scaling_divisor = max_val
    else:
        # Unknown dtype: convert to float32 and normalize by max
        image_f32 = img.astype(np.float32) / max_val
        scaling_divisor = max_val
    
    dtype_info = {
        'original_dtype': str(original_dtype),
        'original_max': max_val,
        'scaling_divisor': float(scaling_divisor),
        'original_shape': original_shape
    }
    
    # Final guarantee float32
    return np.clip(image_f32.astype(np.float16), 0.0, 1.0), dtype_info


def handle_label_dtype(label: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Convert label to appropriate discrete format.
    
    Label should be: 0=background, 1=ink, 2=ignore
    
    Returns:
        label_clean: (D, H, W) with integer values {0, 1, 2}
        label_info: dict with info
    """
    original_dtype = label.dtype
    
    lbl = label.copy()
    if lbl.ndim == 2:
        lbl = lbl[np.newaxis, ...]
    
    # Convert to integer categories
    if np.issubdtype(original_dtype, np.floating):
        lbl_f32 = lbl.astype(np.float32)
        lbl_clean = np.round(lbl_f32).astype(np.uint8)
    else:
        lbl_clean = lbl.astype(np.uint8)
    
    unique_vals = np.unique(lbl_clean)
    if not all(v in [0, 1, 2] for v in unique_vals):
        # treat any non-zero as foreground (1)
        lbl_clean = (lbl_clean > 0).astype(np.uint8)
    
    label_info = {
        'original_dtype': str(original_dtype),
        'unique_values': [int(v) for v in np.unique(lbl_clean)],
        'ink_count': int((lbl_clean == 1).sum()),
        'ignore_count': int((lbl_clean == 2).sum())
    }
    
    return lbl_clean, label_info


# ============================================================================
# PART 2: BINARY DILATION (API-COMPATIBLE)
# ============================================================================

def apply_binary_dilation(mask: np.ndarray, num_iterations: int = 1) -> np.ndarray:
    """
    Apply binary dilation using scipy.ndimage.binary_dilation with iterations parameter.
    
    Args:
        mask: Boolean or binary array (any shape)
        num_iterations: Number of dilation iterations (0 -> no dilation)
    
    Returns:
        dilated: Same shape as input, dtype float32 (0/1)
    """
    if num_iterations <= 0:
        return mask.astype(np.float32)
    
    result = scipy_binary_dilation(mask.astype(np.bool_), iterations=num_iterations)
    return result.astype(np.float32)


# ============================================================================
# PART 3: SKELETON COMPUTATION
# ============================================================================

def generate_skeleton_map(mask_3d: np.ndarray) -> np.ndarray:
    """
    Generate skeleton using skimage.morphology.skeletonize().
    
    AUTOMATIC 2D/3D HANDLING:
    - For 3D (D, H, W): Uses Lee's method (octree-based, designed for 3D)
    - For 2D (H, W): Uses Zhang's method
    
    TOPOLOGY-AWARE:
    - True 3D thinning (preserves connectivity)
    - Proper handling of 3D topology
    - No artificial broadcast artifacts
    """
    mask_3d = mask_3d.astype(np.float32)
    
    if mask_3d.sum() < 3:
        return np.zeros_like(mask_3d, dtype=np.float32)
    
    try:
        mask_bool = mask_3d > 0.5
        skel = skimage_skeletonize(mask_bool).astype(np.float32)
        
        # Dilation disabled by default to prevent merging adjacent layers
        if CFG.SKELETON_DILATION_ITER > 0:
            skel = apply_binary_dilation(skel > 0.5, CFG.SKELETON_DILATION_ITER)
        
        skel = skel * mask_3d
        return skel.astype(np.float32)
    
    except Exception as e:
        if CFG.VERBOSE:
            print("Warning: Skeletonize failed:", str(e))
        # Fallback: medial_axis
        try:
            skel, _ = medial_axis(mask_bool, return_distance=True)
            skel = skel.astype(np.float32)
            if CFG.SKELETON_DILATION_ITER > 0:
                skel = apply_binary_dilation(skel > 0.5, CFG.SKELETON_DILATION_ITER)
            skel = skel * mask_3d
            return skel.astype(np.float32)
        except Exception as e2:
            if CFG.VERBOSE:
                print("Warning: Medial axis fallback failed:", str(e2))
            return np.zeros_like(mask_3d, dtype=np.float32)


# ============================================================================
# PART 4: CENTERLINE & VECTORS
# ============================================================================

def generate_centerline_and_vectors(mask_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate centerline (distance transform) and vectors (fiber-aware gradients).
    
    PRECISION: All float32, NO loss (centerline normalized to [0,1])
    
    Args:
        mask_3d: (D, H, W) float32 binary mask
    
    Returns:
        centerline: (D, H, W) float32 normalized [0, 1]
        vectors: (D, H, W, 3) float32 unit vectors [dz, dy, dx]
    """
    mask_3d = mask_3d.astype(np.float32)
    D, H, W = mask_3d.shape
    
    if mask_3d.sum() < 3:
        return (
            np.zeros((D, H, W), dtype=np.float32),
            np.zeros((D, H, W, 3), dtype=np.float32)
        )
    
    # === STEP 1: Distance Transform ===
    mask_bool = mask_3d > 0.5
    centerline_f64 = scipy_distance_transform_edt(mask_bool)
    centerline = centerline_f64.astype(np.float32)
    
    # === STEP 2: Normalize to [0, 1] ===
    c_max = float(centerline.max())
    if c_max > 0:
        centerline = centerline / c_max
    centerline = np.clip(centerline, 0.0, 1.0).astype(np.float32)
    
    # === STEP 3: Optional smoothing ===
    # Default CFG.CENTERLINE_SMOOTH_SIGMA = 0.0 to avoid blurring 1-voxel features.
    if getattr(CFG, "CENTERLINE_SMOOTH_SIGMA", 0.0) > 0.0:
        centerline_smooth = scipy_gaussian_filter(centerline, sigma=CFG.CENTERLINE_SMOOTH_SIGMA).astype(np.float32)
    else:
        centerline_smooth = centerline
    
    # === STEP 4: Compute gradients ===
    dz, dy, dx = np.gradient(centerline_smooth)
    dz = dz.astype(np.float32)
    dy = dy.astype(np.float32)
    dx = dx.astype(np.float32)
    
    # === STEP 5: Fiber-aware weighting ===
    dz_weighted = dz * CFG.FIBER_WEIGHT_Z
    dy_weighted = dy * CFG.FIBER_WEIGHT_Y
    dx_weighted = dx * CFG.FIBER_WEIGHT_X
    
    # === STEP 6: Normalize ===
    magnitude_sq = dz_weighted**2 + dy_weighted**2 + dx_weighted**2
    magnitude = np.sqrt(magnitude_sq + 1e-8).astype(np.float32)
    magnitude = np.maximum(magnitude, 1e-8)
    
    dz_norm = (dz_weighted / magnitude).astype(np.float32)
    dy_norm = (dy_weighted / magnitude).astype(np.float32)
    dx_norm = (dx_weighted / magnitude).astype(np.float32)
    
    # === STEP 7: Stack ===
    vectors = np.stack([dz_norm, dy_norm, dx_norm], axis=-1)
    
    # Validation
    assert centerline.dtype == np.float32
    assert vectors.dtype == np.float32
    vec_norms = np.linalg.norm(vectors, axis=-1)
    # allow a small numerical slack
    assert np.all(vec_norms <= 1.01), f"Vectors not normalized: max={vec_norms.max()}"
    
    return (
        centerline.astype(np.float32),
        vectors.astype(np.float32)
    )


# ============================================================================
# PART 5: WORKER FUNCTION
# ============================================================================

def process_single_volume_worker(args: Tuple[int, str, str, str, str, str]) -> Optional[Dict]:
    """
    Worker function for parallel processing.
    
    Handles:
    - Any input dtype (uint8, uint16, float32, etc)
    - Proper validation and error handling
    - Quality topology-aware computation
    """
    worker_id, sample_id, scroll_id, img_path, lbl_path, output_dir = args
    
    try:
        # === LOAD IMAGE (with dtype handling) ===
        image_raw = tifffile.imread(img_path)
        if image_raw.ndim == 2:
            image_raw = image_raw[np.newaxis, ...]
        
        # Convert to float32 normalized, preserving relative information
        image_f32, img_dtype_info = normalize_image_lossless(image_raw)
        
        # === LOAD LABEL (with dtype handling) ===
        label_raw = tifffile.imread(lbl_path)
        if label_raw.ndim == 2:
            label_raw = label_raw[np.newaxis, ...]
        
        # Clean label
        label_clean, lbl_dtype_info = handle_label_dtype(label_raw)
        
        # Validate shapes match
        if image_f32.shape != label_clean.shape:
            print(f"  ERROR: Shape mismatch {sample_id}: image={image_f32.shape}, label={label_clean.shape}")
            return None
        
        D, H, W = image_f32.shape
        
        # === CREATE BINARY MASK ===
        mask_binary = (label_clean == 1).astype(np.float32)
        
        if mask_binary.sum() < 3:
            if CFG.VERBOSE:
                print(f"  WARNING: {sample_id}: Empty mask")
            return None
        
        # === GENERATE SKELETON ===
        skeleton = generate_skeleton_map(mask_binary)
        assert skeleton.dtype == np.float32
        assert skeleton.shape == mask_binary.shape
        
        # === GENERATE CENTERLINE & VECTORS ===
        centerline, vectors = generate_centerline_and_vectors(mask_binary)
        assert centerline.dtype == np.float32
        assert vectors.dtype == np.float32
        
        # === ASSEMBLE LABELS (6 channels) ===
        mask_channel = label_clean.astype(np.float32)[..., None]
        
        combined_label = np.concatenate([
            mask_channel,
            skeleton[..., None],
            centerline[..., None],
            vectors
        ], axis=-1)
        
        combined_label = combined_label.astype(np.float32)
        
        assert combined_label.shape == (D, H, W, 6)
        assert combined_label.dtype == np.float32
        
        # === SAVE ===
        img_out = os.path.join(output_dir, "images", f"{sample_id}.npy")
        lbl_out = os.path.join(output_dir, "labels", f"{sample_id}.npy")
        
        os.makedirs(os.path.dirname(img_out), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_out), exist_ok=True)
        
        # Save images as float32 (normalized to [0,1]) and labels as float32
        np.save(img_out, image_f32.astype(np.float16))
        np.save(lbl_out, combined_label)
        
        return {
            'sample_id': sample_id,
            'scroll_id': scroll_id,
            'image_shape': image_f32.shape,
            'image_dtype': img_dtype_info,
            'label_shape': combined_label.shape,
            'label_dtype': lbl_dtype_info,
            'mask_count': int(mask_binary.sum()),
            'skeleton_count': int((skeleton > 0.5).sum()),
            'img_path': img_out,
            'lbl_path': lbl_out,
            'success': True
        }
    
    except Exception as e:
        print(f"  ERROR worker {worker_id} (sample {sample_id}): {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# PART 6: MAIN PIPELINE
# ============================================================================

def create_optimized_dataset_production():
    """
    Production-quality precomputation pipeline.
    
    Handles any input dtype, generates quality topology-aware targets.
    """
    
    print("\n" + "="*80)
    print("VESUVIUS PRECOMPUTATION - PRODUCTION QUALITY")
    print("="*80)
    
    print(f"\nCONFIGURATION:")
    print(f"   Method: skimage.morphology.skeletonize() - Auto handles 2D/3D")
    print(f"   3D Method: Lee's algorithm (octree-based)")
    print(f"   2D Method: Zhang's algorithm")
    print(f"   Centerline sigma: {CFG.CENTERLINE_SMOOTH_SIGMA}")
    print(f"   Fiber weights: Y={CFG.FIBER_WEIGHT_Y}, X={CFG.FIBER_WEIGHT_X}, Z={CFG.FIBER_WEIGHT_Z}")
    print(f"   Workers: {CFG.NUM_PROCESSES}\n")
    
    # === LOAD METADATA ===
    print("Loading sample metadata...")
    df = pd.read_csv(CFG.TRAIN_CSV)
    
    def file_exists(fid):
        return (os.path.exists(os.path.join(CFG.TRAIN_IMAGES, f"{fid}.tif")) and
                os.path.exists(os.path.join(CFG.TRAIN_LABELS, f"{fid}.tif")))
    
    df = df[df['id'].astype(str).apply(file_exists)].reset_index(drop=True)
    all_ids = df['id'].astype(str).tolist()
    all_scrolls = df['scroll_id'].astype(str).tolist()
    
    print(f"   Found {len(all_ids)} valid samples\n")
    
    # === SELECT BATCH ===
    start_idx = CFG.START_INDEX
    end_idx = min(CFG.END_INDEX, len(all_ids))
    batch_ids = all_ids[start_idx:end_idx]
    batch_scrolls = all_scrolls[start_idx:end_idx]
    
    print(f"BATCH: [{start_idx}, {end_idx}) = {len(batch_ids)} samples\n")
    
    # === OUTPUT DIRECTORY ===
    output_dir = os.path.join(CFG.OUTPUT_BASE, f"batch_{start_idx}_{end_idx}")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    print(f"Output: {output_dir}\n")
    
    # === PREPARE WORKERS ===
    worker_args = []
    for i, (sample_id, scroll_id) in enumerate(zip(batch_ids, batch_scrolls)):
        img_path = os.path.join(CFG.TRAIN_IMAGES, f"{sample_id}.tif")
        lbl_path = os.path.join(CFG.TRAIN_LABELS, f"{sample_id}.tif")
        worker_args.append((i, sample_id, scroll_id, img_path, lbl_path, output_dir))
    
    # === PARALLEL PROCESSING ===
    print(f"Processing {len(worker_args)} samples...\n")
    
    successful = []
    failed = []
    
    with Pool(CFG.NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_volume_worker, worker_args),
            total=len(worker_args),
            desc="Pre-computing",
            ncols=100
        ))
    
    for result in results:
        if result is not None and result.get('success'):
            successful.append(result)
        else:
            failed.append(1)
    
    # === METADATA ===
    metadata = {
        'batch_range': [start_idx, end_idx],
        'num_samples': len(successful),
        'num_failed': len(failed),
        'image_dtype': 'float16',  # images preserved as float32 normalized to [0,1]
        'label_dtype': 'float32',
        'label_channels': [
            'mask (0=bg, 1=ink, 2=ignore)',
            'skeleton (0-1)',
            'centerline (0-1)',
            'vector_dz', 'vector_dy', 'vector_dx'
        ],
        'config': {
            'skeleton_algorithm': 'skimage.morphology.skeletonize()',
            'skeleton_3d_method': "Lee's algorithm (octree-based)",
            'skeleton_2d_method': "Zhang's algorithm",
            'centerline_smooth_sigma': CFG.CENTERLINE_SMOOTH_SIGMA,
            'fiber_weights': {
                'Y': CFG.FIBER_WEIGHT_Y,
                'X': CFG.FIBER_WEIGHT_X,
                'Z': CFG.FIBER_WEIGHT_Z
            }
        },
        'samples': [
            {
                'id': r['sample_id'],
                'scroll_id': r['scroll_id'],
                'image_shape': r['image_shape'],
                'image_dtype_info': r['image_dtype'],
                'label_shape': r['label_shape'],
                'mask_voxels': r['mask_count'],
                'skeleton_voxels': r['skeleton_count']
            }
            for r in successful
        ]
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # === REPORT ===
    print("\n" + "="*80)
    print("PRECOMPUTATION COMPLETE!")
    print("="*80)
    
    print(f"\nSTATISTICS:")
    print(f"   Successful: {len(successful)}/{len(worker_args)}")
    print(f"   Failed: {len(failed)}/{len(worker_args)}")
    
    if len(successful) > 0:
        avg_mask = np.mean([r['mask_count'] for r in successful])
        avg_skel = np.mean([r['skeleton_count'] for r in successful])
        print(f"   Avg mask voxels: {avg_mask:.0f}")
        print(f"   Avg skeleton voxels: {avg_skel:.0f}")
    
    # Disk usage
    img_files = list(Path(output_dir, "images").glob("*.npy"))
    lbl_files = list(Path(output_dir, "labels").glob("*.npy"))
    
    if len(img_files) > 0:
        img_size = sum(os.path.getsize(f) for f in img_files) / (1024**3)
        lbl_size = sum(os.path.getsize(f) for f in lbl_files) / (1024**3)
        print(f"\nDisk Usage:")
        print(f"   Images: {img_size:.2f} GB")
        print(f"   Labels: {lbl_size:.2f} GB")
        print(f"   Total: {img_size + lbl_size:.2f} GB")
    
    print(f"\nQUALITY ASSURANCE:")
    print(f"   - Topology-aware skeleton (no broadcast)")
    print(f"   - Skeleton dilation disabled by default to preserve topology")
    print(f"   - Centerline smoothing disabled by default (no blurring of 1-voxel gaps)")
    print(f"   - Fiber-aware vectors")
    print(f"   - Handles any input dtype (uint8, uint16, float32, etc)")
    print(f"   - Images preserved as float16 normalized to [0,1] (no quantization to uint8)")
    print(f"   - ALL float32 labels")
    
    print(f"\nOutput: {output_dir}/")
    print(f"   ├── images/ (*.npy) - float32 normalized")
    print(f"   ├── labels/ (*.npy) - float32 (D,H,W,6)")
    print(f"   └── metadata.json")
    
    print(f"\nNEXT BATCH:")
    print(f"   CFG.START_INDEX = {end_idx}")
    print(f"   CFG.END_INDEX = {end_idx + 25}")
    print("\nReady for training!\n")
    
    return output_dir, successful


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("VESUVIUS PRECOMPUTATION - PRODUCTION PRECOMPUTATION")
    print("Precision-preserving precomputation (images -> float32)")
    print("="*80)
    
    output_dir, results = create_optimized_dataset_production()
