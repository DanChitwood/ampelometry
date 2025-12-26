import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
from pathlib import Path
import cv2
from tqdm import tqdm

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = PROJECT_ROOT / "outputs" / "jaccard_dist_matrix.npy"
UID_PATH = PROJECT_ROOT / "outputs" / "jaccard_uids.txt"
META_PATH = PROJECT_ROOT / "data" / "master_mask_datasheet.csv"
MASK_DIR = PROJECT_ROOT / "data" / "inference_pol_cart_aligned" / "polar" / "viz"
PUBLIC_DIR = PROJECT_ROOT / "public_viz"
OUTPUT_HTML = PUBLIC_DIR / "index.html"

def calculate_vein_ratio(uid):
    mask_path = MASK_DIR / f"{uid}.png"
    if not mask_path.exists(): return np.nan
    img = cv2.imread(str(mask_path)) 
    if img is None: return np.nan

    # BGR Color Ranges for Veins and Lobes
    blue_mask = cv2.inRange(img, np.array([150, 0, 0]), np.array([255, 100, 100]))
    green_mask = cv2.inRange(img, np.array([0, 150, 0]), np.array([100, 255, 100]))
    magenta_mask = cv2.inRange(img, np.array([150, 0, 150]), np.array([255, 100, 255]))
    yellow_mask = cv2.inRange(img, np.array([0, 150, 150]), np.array([100, 255, 255]))
    cyan_mask = cv2.inRange(img, np.array([150, 150, 0]), np.array([255, 255, 100]))
    
    vein_area = np.count_nonzero(blue_mask) + np.count_nonzero(green_mask)
    blade_area = np.count_nonzero(magenta_mask) + np.count_nonzero(yellow_mask) + np.count_nonzero(cyan_mask)
    
    if blade_area == 0 or vein_area == 0: return np.nan
    return -np.log(vein_area / blade_area)

def main():
    print("📂 Loading Ampelometry Data...")
    dist_matrix = np.load(MATRIX_PATH, mmap_mode='r')
    with open(UID_PATH, "r") as f:
        uids = [line.strip() for line in f if line.strip()]

    meta_df = pd.read_csv(META_PATH)
    meta_df['combined_id'] = meta_df.apply(lambda x: f"{x['image_id']}_{int(x['component_id'])}", axis=1)
    meta_df = meta_df[meta_df['combined_id'].isin(uids)].copy()

    print("🧪 Processing Vein-to-Blade Area Ratios...")
    tqdm.pandas()
    meta_df['vein_ratio'] = meta_df['combined_id'].progress_apply(calculate_vein_ratio)

    print("🚀 Computing UMAP Embedding...")
    indices = [uids.index(uid) for uid in meta_df['combined_id']]
    aligned_dist = dist_matrix[np.ix_(indices, indices)]
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(aligned_dist)
    meta_df['x'], meta_df['y'] = embedding[:, 0], embedding[:, 1]

    # Clean Metadata Tooltips
    def format_tooltip(row):
        parts = []
        for i in range(1, 5):
            val = row.get(f'metadata{i}')
            if pd.notna(val) and str(val).strip() != "":
                parts.append(f"<b>M{i}:</b> {val}")
        if pd.notna(row['vein_ratio']):
            parts.append(f"<b>Ratio:</b> {row['vein_ratio']:.4f}")
        return "<br>".join(parts)

    meta_df['hover_text'] = meta_df.apply(format_tooltip, axis=1)

    # --- BUILD PLOT ---
    fig = go.Figure()
    datasets = sorted(meta_df['dataset'].unique())
    
    # 1. Dataset Views
    for ds in datasets:
        sub = meta_df[meta_df['dataset'] == ds]
        fig.add_trace(go.Scattergl(
            x=sub['x'], y=sub['y'], name=ds, mode='markers',
            marker=dict(size=5, opacity=0.7),
            text=sub['hover_text'], 
            hovertext=sub['combined_id'],
            hovertemplate="<b>%{hovertext}</b><br>%{text}<extra>%{fullData.name}</extra>", 
            visible=True
        ))

    # 2. Vein Ratio View (Inferno)
    cont_df = meta_df.dropna(subset=['vein_ratio'])
    fig.add_trace(go.Scattergl(
        x=cont_df['x'], y=cont_df['y'], mode='markers',
        name="Morphological Ratio",
        marker=dict(
            size=6, color=cont_df['vein_ratio'], colorscale='Inferno',
            colorbar=dict(title="-ln(Vein/Blade)", thickness=20, x=1.02),
            showscale=True
        ),
        text=cont_df['hover_text'],
        hovertext=cont_df['combined_id'],
        hovertemplate="<b>%{hovertext}</b><br>%{text}<extra>Ratio View</extra>",
        visible=False
    ))

    # --- FINAL LAYOUT ---
    fig.update_layout(
        title=dict(
            text="<b>Ampelometry: Global Vitis Morphospace</b>",
            x=0.5, y=0.97, xanchor='center', font=dict(size=24)
        ),
        # BUTTON TOGGLE AT BOTTOM
        updatemenus=[dict(
            type="buttons", direction="right", x=0.5, y=-0.15,
            xanchor='center', yanchor='top',
            buttons=[
                dict(label="Color by Dataset", method="update",
                     args=[{"visible": [True]*len(datasets) + [False]}, {"showlegend": True}]),
                dict(label="Color by Vein-to-Blade Ratio", method="update",
                     args=[{"visible": [False]*len(datasets) + [True]}, {"showlegend": False}])
            ]
        )],
        template='plotly_dark',
        margin=dict(t=80, b=150, l=50, r=100)
    )

    print(f"💾 Exporting HTML to {OUTPUT_HTML}...")
    fig.write_html(str(OUTPUT_HTML), include_plotlyjs='cdn')
    print("✨ AMPELOMETRY RELEASE READY.")

if __name__ == "__main__":
    main()