#!/usr/bin/env python3
import argparse, os, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile as tiff

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from pathlib import Path

# Ensure local imports work when running from other directories
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from scn_utils import ensure_tiff_from_input

Image.MAX_IMAGE_PIXELS = None  # allow very large TIFFs

# ---------------------- IO ----------------------
def load_rgb_tif(path):
    """Load TIFF as uint8 RGB. Handles 16-bit by rescaling."""
    try:
        arr = tiff.imread(path)
        if arr.ndim == 3 and arr.shape[0] in (3,4) and arr.shape[-1] not in (3,4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (255.0*(arr-mn)/(mx-mn)).clip(0,255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        img = Image.fromarray(arr, mode="RGB")
    except Exception:
        img = Image.open(path).convert("RGB")
    return img

def grid_positions(length, patch, stride):
    if stride is None or stride <= 0: stride = patch
    if length <= patch: return [0]
    pos = list(range(0, length - patch + 1, stride))
    last = length - patch
    if len(pos) == 0 or pos[-1] != last:
        pos.append(last)
    return pos

# ---------------------- Model ----------------------
def build_feature_model(device):
    weights = models.ResNet50_Weights.DEFAULT
    net = models.resnet50(weights=weights)
    net.fc = nn.Identity()
    net.eval().to(device)
    # Deterministic resize to 224x224 without center-crop
    mean = weights.meta.get("mean", (0.485, 0.456, 0.406))
    std = weights.meta.get("std", (0.229, 0.224, 0.225))
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return net, preprocess

def build_simple_feature_model(device):
    """Simple histogram-based features as alternative to CNN"""
    def extract_features(tensor_batch):
        # Convert to numpy and compute color histograms
        batch_np = tensor_batch.cpu().numpy()
        features = []
        for img in batch_np:
            # Compute RGB histograms (8 bins each)
            hist_r = np.histogram(img[0], bins=8, range=(0, 1))[0]
            hist_g = np.histogram(img[1], bins=8, range=(0, 1))[0]
            hist_b = np.histogram(img[2], bins=8, range=(0, 1))[0]
            # Add texture features (simple edge detection)
            edges = np.abs(img[0] - np.roll(img[0], 1, axis=0)) + np.abs(img[0] - np.roll(img[0], 1, axis=1))
            edge_hist = np.histogram(edges, bins=8, range=(0, 2))[0]
            features.append(np.concatenate([hist_r, hist_g, hist_b, edge_hist]))
        return torch.tensor(features, dtype=torch.float32)
    
    preprocess = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),  # Smaller for simple features
        transforms.ToTensor(),
    ])
    return extract_features, preprocess

def color_palette(n):
    # Generate a distinct RGB palette using evenly spaced HSV values
    import colorsys
    n = max(1, int(n))
    hues = np.linspace(0.0, 1.0, num=n, endpoint=False)
    colors = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colors, dtype=np.uint8)

def pick_device(choice):
    if choice == "auto":
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available(): return "cuda"
        return "cpu"
    return choice

# (mclust and rpy2-based clustering removed; we use KMeans only)

# ---------------------- Utils ----------------------
def progress_bar(i, total, prefix=""):
    width = 30
    frac = 1.0 if total == 0 else i/total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix}[{bar}] {i}/{total}", end="", file=sys.stdout, flush=True)
    if i == total: print("", file=sys.stdout, flush=True)

def majority_3x3(grid, n_classes, iters=1):
    h, w = grid.shape
    for _ in range(max(0, iters)):
        newg = grid.copy()
        for y in range(h):
            y0, y1 = max(0, y-1), min(h, y+2)
            for x in range(w):
                x0, x1 = max(0, x-1), min(w, x+2)
                block = grid[y0:y1, x0:x1].ravel()
                counts = np.bincount(block, minlength=n_classes)
                newg[y, x] = np.argmax(counts)
        grid = newg
    return grid

def draw_legend_on_image(img, colors, names, margin=10, box=20):
    draw = ImageDraw.Draw(img, "RGBA")
    try: font = ImageFont.load_default()
    except Exception: font = None
    rows = len(names); panel_w = 220; panel_h = rows*(box+6)+10
    W,H = img.size; x0,y0 = margin, H-panel_h-margin
    draw.rectangle([x0,y0,x0+panel_w,y0+panel_h], fill=(0,0,0,140))
    for i,(c,name) in enumerate(zip(colors, names)):
        cy = y0+5+i*(box+6)
        draw.rectangle([x0+8,cy,x0+8+box,cy+box], fill=tuple(int(v) for v in c))
        draw.rectangle([x0+8,cy,x0+8+box,cy+box], outline=(255,255,255,200), width=1)
        draw.text((x0+8+box+8, cy+2), name, fill=(255,255,255,230), font=font)

def rgb_to_qupath_colors(r: int, g: int, b: int) -> tuple[int, int]:
    """Return (colorRGB, color) ints in the format QuPath understands.
    colorRGB: 24-bit RGB (positive)
    color: 32-bit ARGB signed int (negative for most colors)
    """
    color_rgb = (r << 16) | (g << 8) | b
    argb = (255 << 24) | color_rgb
    if argb >= 2**31:
        argb -= 2**32
    return color_rgb, argb

def create_qupath_annotations(label_grid, patch_size, stride, colors, class_names,
                              original_width, original_height, scale_factor=1.0):
    """Create QuPath-compatible GeoJSON using polygon union via shapely."""
    from shapely.geometry import box, Polygon, MultiPolygon
    from shapely.ops import unary_union

    if stride is None:
        stride = patch_size

    features = []

    num_classes = len(class_names)
    for class_id in range(num_classes):
        name_clean = class_names[class_id].split(' (')[0]
        if name_clean == 'BG':
            continue

        # Build rectangles for all patches of this class
        rects = []
        ny, nx = label_grid.shape
        for j in range(ny):
            for i in range(nx):
                if label_grid[j, i] == class_id:
                    x0 = i * stride / scale_factor
                    y0 = j * stride / scale_factor
                    x1 = x0 + patch_size / scale_factor
                    y1 = y0 + patch_size / scale_factor
                    rects.append(box(x0, y0, x1, y1))

        if not rects:
            continue

        # Union & slight smoothing (dilate then erode) to merge touching patches
        union = unary_union(rects)
        try:
            smooth = union.buffer(patch_size * 0.1, join_style=2).buffer(-patch_size * 0.1, join_style=2)
        except Exception:
            smooth = union

        geoms = []
        if isinstance(smooth, (MultiPolygon,)):
            geoms = list(smooth.geoms)
        elif isinstance(smooth, Polygon):
            geoms = [smooth]

        r, g, b = int(colors[class_id][0]), int(colors[class_id][1]), int(colors[class_id][2])
        color_rgb, color_argb = rgb_to_qupath_colors(r, g, b)

        for idx, poly in enumerate(geoms, start=1):
            exterior = list(poly.exterior.coords)
            coords = [[float(x), float(y)] for (x, y) in exterior]
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    # Name shown in the annotation list
                    "name": f"{name_clean}_Region_{idx}",
                    # Classification that appears in QuPath class list
                    "classification": {"name": name_clean, "color": color_argb, "colorRGB": color_rgb},
                    "isLocked": False,
                    "measurements": {"Num patches (approx)": int(poly.area / ((patch_size/scale_factor)*(patch_size/scale_factor)))}
                },
            }
            features.append(feature)

    return {"type": "FeatureCollection", "features": features}

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Tile + embed + KMeans cluster histology slides; accepts .tif/.tiff directly or .scn (auto-converted). Saves segmentation and overlay images.")
    ap.add_argument("--input", required=True, help="Path to the input image (.tif/.tiff) or Leica .scn (auto-converted to TIFF)")
    ap.add_argument("--patch_size", type=int, default=64, help="Tile size in pixels for patch extraction")
    ap.add_argument("--stride", type=int, default=None, help="Tile stride in pixels; defaults to patch_size (no overlap) if omitted")
    ap.add_argument("--clusters", type=int, default=3, help="Number of clusters (K) for KMeans")
    ap.add_argument("--batch", type=int, default=512, help="Batch size for feature extraction")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="Compute device selection; 'auto' prefers MPS, then CUDA, then CPU")
    ap.add_argument("--background_threshold", type=float, default=200, help="Skip tiles with mean intensity > threshold (0-255) as background")
    ap.add_argument("--smooth_iters", type=int, default=1, help="Number of 3x3 majority smoothing iterations on the tile grid")
    ap.add_argument("--feature_norm", choices=["none","zscore","l2"], default="l2",
                    help="Feature normalization across all tiles before clustering")
    ap.add_argument("--pca_dims", type=int, default=0, help="If >0, reduce feature dimensionality with PCA to this many components before clustering")
    # KMeans is the only supported clustering method in this project setup
    ap.add_argument("--overlay_max_dim", type=int, default=768, help="Max dimension for overlay/segmentation images (prevents OOM on gigapixel slides)")
    ap.add_argument("--simple_features", action="store_true", help="Use simple histogram-based features instead of CNN features")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output_dir", type=str, help="Output directory (defaults to input file's directory)")
    ap.add_argument("--output_formats", nargs="+", choices=["overlay", "segmentation", "qupath"], 
                    default=["overlay", "segmentation"], help="Output formats to generate")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    device = pick_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Accept .tif/.tiff directly, or .scn which we convert on-the-fly
    input_path = Path(args.input)
    effective_input = ensure_tiff_from_input(input_path)
    img = load_rgb_tif(str(effective_input))
    W,H = img.size
    patch = int(args.patch_size); stride = args.stride
    xs, ys = grid_positions(W, patch, stride), grid_positions(H, patch, stride)
    nx, ny = len(xs), len(ys); total_tiles = nx*ny

    if args.simple_features:
        feature_extractor, preprocess = build_simple_feature_model(device)
        model = None
    else:
        model, preprocess = build_feature_model(device)
        feature_extractor = None
    print(f"[INFO] Analysis Settings:")
    print(f"  Input: {args.input} ({W}x{H}px)")
    print(f"  Patch size: {patch}px, Stride: {stride if stride else patch}px")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Clusters: {args.clusters}")
    print(f"  Device: {device}")
    print(f"  Features: {'Simple histograms' if args.simple_features else 'CNN (ResNet50)'}")
    print(f"  Normalization: {args.feature_norm}")
    if args.pca_dims > 0:
        print(f"  PCA dimensions: {args.pca_dims}")
    print(f"  Background threshold: {args.background_threshold}")
    print(f"  Smoothing iterations: {args.smooth_iters}")
    print(f"  Output formats: {', '.join(args.output_formats)}")
    print(f"[INFO] Starting analysis...")

    # fast indexers
    x2i = {x:i for i,x in enumerate(xs)}
    y2j = {y:j for j,y in enumerate(ys)}

    # label grid (K clusters + BG as index K)
    BG = args.clusters
    label_grid = np.full((ny, nx), BG, dtype=np.int32)
    bg_color = np.array([200,200,200], dtype=np.uint8)

    # feature extraction
    coords_valid = []
    feats_chunks = []
    batch = []
    done = 0; progress_bar(done, total_tiles, prefix="Extract+Embed ")

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = img.crop((x,y,x+patch,y+patch))
                done += 1
                if args.background_threshold is not None and np.mean(tile) > args.background_threshold:
                    progress_bar(done, total_tiles, prefix="Extract+Embed ")
                    continue
                t = preprocess(tile).to(device)
                batch.append(t)
                coords_valid.append((x,y))
                if len(batch) == args.batch:
                    batch_tensor = torch.stack(batch,0)
                    if args.simple_features:
                        feats = feature_extractor(batch_tensor).cpu().numpy()
                    else:
                        feats = model(batch_tensor).cpu().numpy()
                    feats_chunks.append(feats)
                    batch = []
                progress_bar(done, total_tiles, prefix="Extract+Embed ")
        if batch:
            batch_tensor = torch.stack(batch,0)
            if args.simple_features:
                feats = feature_extractor(batch_tensor).cpu().numpy()
            else:
                feats = model(batch_tensor).cpu().numpy()
            feats_chunks.append(feats)

    if feats_chunks:
        features = np.concatenate(feats_chunks, axis=0)
    else:
        features = np.zeros((0,2048), dtype=np.float32)

    print(f"\n[INFO] feature tiles: {features.shape[0]} | background tiles: {total_tiles - len(coords_valid)}")

    # ----- feature normalization / PCA -----
    if features.shape[0] > 0:
        if args.feature_norm == "zscore":
            scaler = StandardScaler(with_mean=True, with_std=True)
            features = scaler.fit_transform(features)
        elif args.feature_norm == "l2":
            features = normalize(features, norm="l2")

        if args.pca_dims and args.pca_dims > 0 and args.pca_dims < features.shape[1]:
            pca = PCA(n_components=args.pca_dims, random_state=args.seed)
            features = pca.fit_transform(features)

        print(f"[INFO] Clustering {features.shape[0]} tiles into K={args.clusters} using KMeans")
        km = KMeans(n_clusters=args.clusters, random_state=args.seed, n_init=10)
        labels = km.fit_predict(features)
        
        # Calculate cluster statistics
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        cluster_stats = {}
        for label, count in zip(unique_labels, label_counts):
            percentage = (count / len(labels)) * 100
            cluster_stats[int(label)] = {'count': int(count), 'percentage': percentage}
        
        print(f"[INFO] Cluster distribution:")
        for cluster_id in sorted(cluster_stats.keys()):
            stats = cluster_stats[cluster_id]
            print(f"  Cluster {cluster_id}: {stats['count']} patches ({stats['percentage']:.1f}%)")
        
        for (x,y), lab in zip(coords_valid, labels):
            label_grid[y2j[y], x2i[x]] = int(lab)

    # ----- smoothing on tile grid -----
    if args.smooth_iters > 0 and total_tiles > 1:
        label_grid = majority_3x3(label_grid, args.clusters+1, iters=args.smooth_iters)

    # ----- save small tile-grid segmentation -----
    colors = color_palette(args.clusters)
    class_colors = np.vstack([colors, bg_color[None,:]])  # last is BG
    seg_small = class_colors[label_grid]  # shape (ny, nx, 3)

    # Determine output directory and base name
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base = str(output_dir / input_path.stem)
    else:
        base,_ = os.path.splitext(str(input_path))
    
    # Calculate scale for output images to prevent OOM on gigapixel images
    scale = min(args.overlay_max_dim / W, args.overlay_max_dim / H, 1.0)
    OW, OH = max(1, int(W*scale)), max(1, int(H*scale))
    
    # Calculate final cluster statistics from label_grid (after smoothing)
    final_cluster_stats = {}
    total_patches = label_grid.size
    for cluster_id in range(args.clusters + 1):  # Include background
        count = np.sum(label_grid == cluster_id)
        percentage = (count / total_patches) * 100
        final_cluster_stats[cluster_id] = {'count': int(count), 'percentage': percentage}
    
    # Create names with statistics
    names = []
    for i in range(args.clusters):
        stats = final_cluster_stats.get(i, {'count': 0, 'percentage': 0})
        names.append(f"C{i} ({stats['count']} patches, {stats['percentage']:.1f}%)")
    
    # Add background
    bg_stats = final_cluster_stats.get(args.clusters, {'count': 0, 'percentage': 0})
    names.append(f"BG ({bg_stats['count']} patches, {bg_stats['percentage']:.1f}%)")
    
    # Generate outputs based on selected formats
    model_suffix = "_simple" if args.simple_features else "_cnn"
    if "segmentation" in args.output_formats:
        seg_img = Image.fromarray(seg_small).resize((OW, OH), Image.NEAREST)
        draw_legend_on_image(seg_img, class_colors, names)
        seg_path = f"{base}_segmentation{model_suffix}.png"
        seg_img.save(seg_path)
        print(f"[OK] Saved segmentation ({OW}x{OH}): {seg_path}")

    if "overlay" in args.output_formats:
        # ----- overlay: upscale seg_small back to slide size -----
        seg_up = Image.fromarray(seg_small).resize((OW, OH), Image.NEAREST)
        img_small = img.resize((OW, OH), Image.BILINEAR)
        overlay = (0.45*np.array(img_small,dtype=np.float32)+0.55*np.array(seg_up,dtype=np.float32)).clip(0,255).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)
        draw_legend_on_image(overlay_img, class_colors, names)
        overlay_path = f"{base}_overlay{model_suffix}.png"
        overlay_img.save(overlay_path)
        print(f"[OK] Saved overlay ({OW}x{OH}): {overlay_path}")

    if "qupath" in args.output_formats:
        # Generate QuPath annotations
        import json
        qupath_annotations = create_qupath_annotations(
            label_grid, patch, stride, colors, names[:-1],  # Exclude BG from QuPath
            W, H, scale_factor=1.0
        )
        
        qupath_path = f"{base}_qupath_annotations{model_suffix}.geojson"
        with open(qupath_path, 'w') as f:
            json.dump(qupath_annotations, f, indent=2)
        print(f"[OK] Saved QuPath annotations: {qupath_path}")
        print(f"     Import this file in QuPath: Objects → Import objects → GeoJSON")

    print("[DONE]")

if __name__ == "__main__":
    main()