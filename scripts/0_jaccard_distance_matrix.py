import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# We are analyzing the Polar Viz images as requested
LBL_DIR = PROJECT_ROOT / "data" / "inference_pol_cart_aligned" / "polar" / "viz"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup for Mac GPU (MPS)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Class BGR Colors from your Viz logic
CLASS_COLORS = {
    "veins_midvein": [255, 0, 0],   # Blue
    "veins_lateral": [0, 255, 0],   # Green
    "lobes_midvein": [255, 0, 255], # Magenta
    "lobes_distal": [0, 255, 255],  # Yellow
    "lobes_proximal": [255, 255, 0] # Cyan
}

def load_masks_as_tensor(uids, target_size=(128, 128)):
    """
    Loads viz images and converts them into a massive 5-channel 
    binary tensor on the GPU.
    """
    n = len(uids)
    c = len(CLASS_COLORS)
    h, w = target_size
    
    # Pre-allocate bool tensor (N, C, H*W) to save memory
    master_tensor = torch.zeros((n, c, h * w), dtype=torch.bool, device=DEVICE)
    
    print(f"📊 Stage 1: Loading & Downsampling {n} masks...")
    for i, uid in enumerate(tqdm(uids, desc="Image Processing")):
        img_path = LBL_DIR / f"{uid}.png"
        
        # In case the directory listing and filename mismatch
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        
        for j, (name, color) in enumerate(CLASS_COLORS.items()):
            # Create binary mask for this class
            mask = np.all(img == color, axis=-1)
            master_tensor[i, j, :] = torch.from_numpy(mask).flatten().to(DEVICE)
            
    return master_tensor

def calculate_jaccard_matrix(tensor):
    """
    Computes pairwise Jaccard distance matrix using GPU linear algebra.
    Distance = 1 - (Intersection / Union)
    """
    n, c, p = tensor.shape
    # Final distance matrix
    dist_matrix = torch.zeros((n, n), device=DEVICE)

    print(f"🧬 Stage 2: GPU Pairwise Calculation (5 Classes)...")
    for class_idx in tqdm(range(c), desc="Class Iteration"):
        masks = tensor[:, class_idx, :].float() # (N, Pixels)
        
        # Intersection = Dot Product (N, N)
        intersection = torch.mm(masks, masks.t()) 
        
        # Union = Sum(A) + Sum(B) - Intersection
        sum_per_leaf = masks.sum(dim=1).view(-1, 1)
        union = sum_per_leaf + sum_per_leaf.t() - intersection
        
        # Jaccard Similarity (handling 0/0 by adding epsilon)
        j_idx = intersection / (union + 1e-6)
        
        # Add Distance (1 - similarity) to total
        dist_matrix += (1.0 - j_idx)

    # Return Average Distance across all 5 classes
    return (dist_matrix / c).cpu().numpy()

def main():
    # 1. Get UIDs from filenames
    # Filters out .DS_Store and ensures we only get PNGs
    uids = sorted([f.name.replace(".png", "") for f in LBL_DIR.glob("img_*.png")])
    
    if not uids:
        print(f"❌ No files found in {LBL_DIR}")
        return

    # 2. Load and Compute
    mask_tensor = load_masks_as_tensor(uids, target_size=(128, 128))
    
    final_dist_matrix = calculate_jaccard_matrix(mask_tensor)
    
    # 3. Binary Save Phase
    print(f"💾 Stage 3: Binary Serialization...")
    
    # Save Matrix
    npy_path = OUTPUT_DIR / "jaccard_dist_matrix.npy"
    # np.save is very fast, but for 25k x 25k we still use a context for clarity
    np.save(npy_path, final_dist_matrix)
    
    # Save UIDs (so we can map rows back to leaves later)
    uid_path = OUTPUT_DIR / "jaccard_uids.txt"
    with open(uid_path, "w") as f:
        for uid in tqdm(uids, desc="Saving UIDs"):
            f.write(f"{uid}\n")

    print(f"\n✨ COMPLETE!")
    print(f"Matrix: {npy_path} (~2.5 GB)")
    print(f"UID List: {uid_path}")
    print(f"Now you can run the UMAP script in seconds!")

if __name__ == "__main__":
    main()