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
OUTPUT_HTML = PROJECT_ROOT / "index.html"

def calculate_morphometrics(uid):
    mask_path = MASK_DIR / f"{uid}.png"
    if not mask_path.exists(): return np.nan, np.nan
    img = cv2.imread(str(mask_path)) 
    if img is None: return np.nan, np.nan
    
    # 1. Vein-to-Blade Ratio Calculation
    blue_mask = cv2.inRange(img, np.array([150, 0, 0]), np.array([255, 100, 100]))
    green_mask = cv2.inRange(img, np.array([0, 150, 0]), np.array([100, 255, 100]))
    magenta_mask = cv2.inRange(img, np.array([150, 0, 150]), np.array([255, 100, 255]))
    yellow_mask = cv2.inRange(img, np.array([0, 150, 150]), np.array([100, 255, 255]))
    cyan_mask = cv2.inRange(img, np.array([150, 150, 0]), np.array([255, 255, 100]))
    
    vein_area = np.count_nonzero(blue_mask) + np.count_nonzero(green_mask)
    blade_area = np.count_nonzero(magenta_mask) + np.count_nonzero(yellow_mask) + np.count_nonzero(cyan_mask)
    ratio = -np.log(vein_area / blade_area) if (blade_area > 0 and vein_area > 0) else np.nan

    # 2. Solidity Calculation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    leaf_area = np.count_nonzero(binary)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = leaf_area / hull_area if hull_area > 0 else np.nan
    else:
        solidity = np.nan

    return ratio, solidity

def main():
    print("📂 Loading data...")
    dist_matrix = np.load(MATRIX_PATH, mmap_mode='r')
    with open(UID_PATH, "r") as f:
        uids = [line.strip() for line in f if line.strip()]

    meta_df = pd.read_csv(META_PATH)
    meta_df['combined_id'] = meta_df.apply(lambda x: f"{x['image_id']}_{int(x['component_id'])}", axis=1)
    meta_df = meta_df[meta_df['combined_id'].isin(uids)].copy()

    print("🧪 Calculating Morphometrics (Vein Ratio & Solidity)...")
    tqdm.pandas()
    results = meta_df['combined_id'].progress_apply(calculate_morphometrics)
    meta_df['vein_ratio'], meta_df['solidity'] = zip(*results)
    meta_df['solidity_transformed'] = meta_df['solidity']**8

    def format_tooltip(row):
        parts = []
        for i in range(1, 5):
            val = row.get(f'metadata{i}')
            if pd.notna(val) and str(val).strip() != "":
                parts.append(f"<b>M{i}:</b> {val}")
        if pd.notna(row['vein_ratio']): parts.append(f"<b>Vein Ratio:</b> {row['vein_ratio']:.4f}")
        if pd.notna(row['solidity']): parts.append(f"<b>Solidity:</b> {row['solidity']:.4f}")
        return "<br>".join(parts)

    meta_df['hover_text'] = meta_df.apply(format_tooltip, axis=1)

    print("🚀 Computing UMAP...")
    indices = [uids.index(uid) for uid in meta_df['combined_id']]
    aligned_dist = dist_matrix[np.ix_(indices, indices)]
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(aligned_dist)
    meta_df['x'], meta_df['y'] = embedding[:, 0], embedding[:, 1]

    # --- BUILD PLOT ---
    fig = go.Figure()
    datasets = sorted(meta_df['dataset'].unique())
    num_datasets = len(datasets)
    
    for ds in datasets:
        sub = meta_df[meta_df['dataset'] == ds]
        fig.add_trace(go.Scattergl(
            x=sub['x'], y=sub['y'], name=ds, mode='markers',
            marker=dict(size=5, opacity=0.7),
            text=sub['hover_text'], hovertext=sub['combined_id'], customdata=sub['combined_id'],
            hovertemplate="<b>%{hovertext}</b><br>%{text}<extra>%{fullData.name}</extra>", 
            visible=True
        ))

    fig.add_trace(go.Scattergl(
        x=meta_df['x'], y=meta_df['y'], mode='markers', name="Vein Ratio View",
        marker=dict(size=6, color=meta_df['vein_ratio'], colorscale='Inferno', showscale=True, colorbar=dict(title="-ln(V/B)", x=1.02)),
        text=meta_df['hover_text'], hovertext=meta_df['combined_id'], customdata=meta_df['combined_id'],
        hovertemplate="<b>%{hovertext}</b><br>%{text}<extra>Vein View</extra>", visible=False
    ))

    fig.add_trace(go.Scattergl(
        x=meta_df['x'], y=meta_df['y'], mode='markers', name="Solidity View",
        marker=dict(size=6, color=meta_df['solidity_transformed'], colorscale='Inferno', showscale=True, colorbar=dict(title="Solidity^8", x=1.15)),
        text=meta_df['hover_text'], hovertext=meta_df['combined_id'], customdata=meta_df['combined_id'],
        hovertemplate="<b>%{hovertext}</b><br>%{text}<extra>Solidity View</extra>", visible=False
    ))

    # --- NEW TITLE CONFIGURATION ---
    fig.update_layout(
        title=dict(
            text="Grapevine leaf morphospace<br><span style='font-size: 16px; font-style: italic;'>Click on a point to open leaf in a new tab</span>",
            x=0.5, 
            y=0.95, 
            xanchor='center',
            yanchor='top',
            font=dict(size=24) # Main title size
        ),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.5, y=-0.15, xanchor='center',
            buttons=[
                dict(label="Datasets", method="update",
                     args=[{"visible": [True]*num_datasets + [False, False]}, {"showlegend": True}]),
                dict(label="Vein Ratio", method="update",
                     args=[{"visible": [False]*num_datasets + [True, False]}, {"showlegend": False}]),
                dict(label="Solidity", method="update",
                     args=[{"visible": [False]*num_datasets + [False, True]}, {"showlegend": False}])
            ]
        )],
        template='plotly_dark', margin=dict(t=100, b=150, l=50, r=150)
    )

    # JS Injection for Clicking
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    click_js = """<script>
    var plot = document.getElementsByClassName('plotly-graph-div')[0];
    plot.on('plotly_click', function(data){
        var point = data.points[0];
        if(point.customdata){
            var url = 'panels/' + point.customdata + '.jpg';
            window.open(url, '_blank');
        }
    });
    </script></body>"""
    
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_str.replace("</body>", click_js))
    print(f"✨ DONE: Website saved with updated title at {OUTPUT_HTML}")

if __name__ == "__main__":
    main()