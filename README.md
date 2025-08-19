### Tissue Clustering – Histology Image Analysis

A compact toolkit to cluster histology whole‑slide images into tissue classes using tile embeddings and KMeans. It supports TIFF/PNG/JPG directly, and Leica SCN via on‑the‑fly conversion to a high‑resolution TIFF.

Developed by Kaveh Shahhosseini (K.Shahhosseini@tees.ac.uk)

---

## Installation

### 1) Create a Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r "necrosis project/requirements.txt"
```

Notes:
- PyTorch/TorchVision pins in `requirements.txt` are CPU/MPS compatible; adjust if you want a specific CUDA build.
- `shapely` is used to merge patch rectangles into region polygons for QuPath.

---

## Quick start

### Run the GUI (recommended)
```bash
python "necrosis project/tissue_clustering_gui.py"
```

Steps:
- Click Browse and select a `.tif/.tiff/.png/.jpg` or `.scn` file
- The panel shows Image Resolution and Estimated Tiles as you change Patch Size / Stride
- Choose Feature Model (ResNet CNN or Simple histogram features)
- Choose outputs (overlay, segmentation, QuPath)
- Click Run Analysis

Outputs are written next to the input file, or to `--output_dir` if provided in CLI mode. Filenames include the model suffix to avoid overwrites: `_cnn` or `_simple`.

### Run from the command line
```bash
python "necrosis project/tissue_clustering.py" \
  --input /path/slide.tiff \
  --patch_size 64 \
  --clusters 3 \
  --output_formats overlay segmentation qupath
```

SCN input:
```bash
python "necrosis project/tissue_clustering.py" --input /path/slide.scn
```
The script converts `.scn` to `<stem>_highres.tiff` once, then processes that TIFF.

---

## How it works

- The slide is tiled on a grid defined by Patch Size and Stride.
- Features are extracted either via:
  - ResNet50 CNN (TorchVision pretrained), or
  - Simple histogram features (fast, CPU‑only baseline).
- Optional feature normalization and PCA.
- KMeans clusters the tile embeddings into K tissue classes.
- A compact tile‑grid segmentation is generated and upscaled for visualization.
- Optional QuPath GeoJSON is produced by merging same‑class tile rectangles into smoothed polygons.

---

## Options (CLI)

```text
--input                Path to .tif/.tiff/.png/.jpg or .scn (auto‑converted)
--patch_size           Tile size in pixels (default 64)
--stride               Tile stride; default = patch_size (no overlap)
--clusters             K for KMeans (default 3)
--batch                Batch size for feature extraction (default 512)
--device               auto | cpu | cuda | mps (default auto)
--background_threshold Skip tiles with mean intensity > thr (0‑255, default 200)
--smooth_iters         3×3 majority smoothing passes on tile grid (default 1)
--feature_norm         none | zscore | l2 (default l2)
--pca_dims             If >0, reduce features to this size before clustering (default 0)
--overlay_max_dim      Max W/H for visualization images (default 768)
--simple_features      Use simple histogram features (if omitted, ResNet CNN is used)
--seed                 Random seed (default 42)
--output_dir           Directory for outputs (default: next to input)
--output_formats       One or more: overlay segmentation qupath (default overlay segmentation)
```

Notes:
- Outputs include the model suffix: `_overlay_cnn.png`, `_segmentation_simple.png`, `_qupath_annotations_cnn.geojson`, etc.
- Device “auto” prefers Apple MPS, then CUDA, then CPU.

---

## Outputs

- Overlay image: `*_overlay_<model>.png` – Original image blended with class colors.
- Segmentation map: `*_segmentation_<model>.png` – Colorized tile‑grid segmentation (with legend showing counts and percentages).
- QuPath annotations: `*_qupath_annotations_<model>.geojson` – Class regions as polygons colored per class.

### Importing GeoJSON into QuPath
- Open the slide in QuPath
- Menu: Objects → Import objects → GeoJSON
- Select `*_qupath_annotations_<model>.geojson`
- Regions appear as annotations; class colors and names are included.

---

## GUI fields

- Input File: Image file to analyze; SCN is supported (auto‑converted)
- Image Resolution / Estimated Tiles: Live info when you change Patch Size or Stride
- Patch Size (pixels): Tile size used for analysis
- Stride: Tile spacing; set to `auto` for no overlap (stride = patch size)
- Number of Clusters: K for KMeans
- Compute Device: auto/cpu/cuda/mps
- Feature Model: ResNet CNN (default) or Simple histogram features
- Output Options: Overlay, Segmentation map, QuPath annotations
- Advanced parameters: Batch Size, Background Threshold, Smoothing Iterations, Normalization, PCA, Output Max Dimension, Seed

---

## Performance tips

- GPU/Apple Silicon (MPS) speeds up the ResNet model; “Simple histogram features” is CPU‑only and fastest.
- Increase Stride (or set it to `auto`) to reduce tiles and memory.
- Lower `overlay_max_dim` to shrink visualization size.
- QuPath export unions polygons; for extremely large slides it adds some CPU time – uncheck if not needed.

---

## Troubleshooting

- Torch/TorchVision install: use versions matching your CUDA toolkit if you want GPU. The defaults are CPU/MPS friendly.
- SCN reading: we read series axes to determine X/Y; if an SCN has no standard axes, convert it externally to TIFF.
- Colors in QuPath: we write both `colorRGB` and ARGB `color` so annotations use the class color; if QuPath shows a single color, ensure you’re importing the GeoJSON as annotations (Objects → Import objects → GeoJSON).

---

## License

This project inherits the licenses of its dependencies. Consult each package’s license for details.


