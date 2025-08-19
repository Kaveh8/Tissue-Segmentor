#!/usr/bin/env python3
"""
GUI for Tissue Clustering
A user-friendly interface for the tissue clustering pipeline.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import sys
import os
from pathlib import Path
import queue
import time

class TissueClusteringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tissue Clustering - Histology Image Analysis")
        self.root.geometry("700x900")
        
        # Queue for thread communication
        self.output_queue = queue.Queue()
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.patch_size = tk.IntVar(value=64)
        self.stride = tk.StringVar(value="auto")
        self.clusters = tk.IntVar(value=3)
        self.batch_size = tk.IntVar(value=512)
        self.device = tk.StringVar(value="auto")
        self.background_threshold = tk.DoubleVar(value=200.0)
        self.smooth_iters = tk.IntVar(value=1)
        self.feature_norm = tk.StringVar(value="l2")
        self.pca_dims = tk.IntVar(value=0)
        self.overlay_max_dim = tk.IntVar(value=768)
        # Feature model selection: 'cnn' or 'simple'
        self.feature_model = tk.StringVar(value="cnn")
        self.seed = tk.IntVar(value=42)
        # Image + tiles info
        self.image_dims = None  # (W, H)
        self.image_info = tk.StringVar(value="Image Resolution: -")
        self.tiles_info = tk.StringVar(value="Estimated Tiles: -")
        
        # Output options
        self.save_overlay = tk.BooleanVar(value=True)
        self.save_segmentation = tk.BooleanVar(value=True)
        self.save_qupath_annotations = tk.BooleanVar(value=False)
        
        self.create_widgets()
        
        # React to parameter changes to update tile estimation
        self.patch_size.trace_add("write", lambda *a: self.update_tile_estimate())
        self.stride.trace_add("write", lambda *a: self.update_tile_estimate())

        # Start checking for output updates
        self.check_queue()
    
    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Title
        title_label = ttk.Label(main_frame, text="Tissue Clustering - Histology Analysis", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 5))
        row += 1
        
        # Author info (clickable email)
        import webbrowser
        author_frame = ttk.Frame(main_frame)
        author_frame.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        ttk.Label(author_frame, text="Developed by Kaveh Shahhosseini ", font=("Arial", 9), foreground="gray").pack(side=tk.LEFT)
        email_label = ttk.Label(author_frame, text="K.Shahhosseini@tees.ac.uk", font=("Arial", 9, "underline"), foreground="#1a73e8", cursor="hand2")
        def _open_email(event=None):
            webbrowser.open("mailto:K.Shahhosseini@tees.ac.uk")
        email_label.bind("<Button-1>", _open_email)
        email_label.pack(side=tk.LEFT)
        row += 1
        
        # Input file selection
        ttk.Label(main_frame, text="Input File:").grid(row=row, column=0, sticky=tk.W, pady=5)
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(input_frame, textvariable=self.input_file, width=50).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(input_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=1)
        row += 1

        # Image info + tiles estimate
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E))
        ttk.Label(info_frame, textvariable=self.image_info).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(info_frame, textvariable=self.tiles_info).pack(side=tk.LEFT)
        row += 1
        
        # Output directory selection
        ttk.Label(main_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W, pady=5)
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=1)
        ttk.Label(main_frame, text="(Leave empty to save next to input file)", font=("Arial", 8), foreground="gray").grid(row=row+1, column=1, sticky=tk.W)
        row += 2
        
        # Create notebook for organized parameters
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(row, weight=1)
        row += 1
        
        # Basic Parameters Tab
        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="Basic Parameters")
        
        basic_row = 0
        
        # Patch size
        ttk.Label(basic_frame, text="Patch Size (pixels):").grid(row=basic_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(basic_frame, from_=16, to=512, textvariable=self.patch_size, width=10).grid(row=basic_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(basic_frame, text="Size of image patches for analysis").grid(row=basic_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        basic_row += 1
        
        # Stride
        ttk.Label(basic_frame, text="Stride:").grid(row=basic_row, column=0, sticky=tk.W, pady=2)
        stride_frame = ttk.Frame(basic_frame)
        stride_frame.grid(row=basic_row, column=1, sticky=tk.W, pady=2)
        ttk.Entry(stride_frame, textvariable=self.stride, width=10).grid(row=0, column=0)
        ttk.Label(basic_frame, text="Patch overlap ('auto' = no overlap)").grid(row=basic_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        basic_row += 1
        
        # Number of clusters
        ttk.Label(basic_frame, text="Number of Clusters:").grid(row=basic_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(basic_frame, from_=2, to=20, textvariable=self.clusters, width=10).grid(row=basic_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(basic_frame, text="Number of tissue types to identify").grid(row=basic_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        basic_row += 1
        
        # Device selection
        ttk.Label(basic_frame, text="Compute Device:").grid(row=basic_row, column=0, sticky=tk.W, pady=2)
        device_combo = ttk.Combobox(basic_frame, textvariable=self.device, values=["auto", "cpu", "cuda", "mps"], 
                                   state="readonly", width=8)
        device_combo.grid(row=basic_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(basic_frame, text="Processing device (auto recommended)").grid(row=basic_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        basic_row += 1
        
        # Feature model selection (radio buttons)
        model_frame = ttk.LabelFrame(basic_frame, text="Feature Model")
        model_frame.grid(row=basic_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Radiobutton(model_frame, text="ResNet CNN (default)", value="cnn", variable=self.feature_model).grid(row=0, column=0, sticky=tk.W, padx=6, pady=2)
        ttk.Radiobutton(model_frame, text="Simple histogram features", value="simple", variable=self.feature_model).grid(row=0, column=1, sticky=tk.W, padx=6, pady=2)
        basic_row += 1
        
        # Output options section in basic tab
        ttk.Separator(basic_frame, orient='horizontal').grid(row=basic_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        basic_row += 1
        
        ttk.Label(basic_frame, text="Output Options:", font=("Arial", 9, "bold")).grid(row=basic_row, column=0, columnspan=3, sticky=tk.W, pady=2)
        basic_row += 1
        
        ttk.Checkbutton(basic_frame, text="Save overlay image", variable=self.save_overlay).grid(row=basic_row, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(basic_frame, text="Save segmentation map", variable=self.save_segmentation).grid(row=basic_row, column=1, sticky=tk.W, pady=2)
        ttk.Checkbutton(basic_frame, text="Save QuPath annotations (GeoJSON)", variable=self.save_qupath_annotations).grid(row=basic_row, column=2, sticky=tk.W, pady=2)
        basic_row += 1
        
        # Advanced Parameters Tab
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text="Advanced Parameters")
        
        adv_row = 0
        
        # Batch size
        ttk.Label(advanced_frame, text="Batch Size:").grid(row=adv_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=32, to=2048, textvariable=self.batch_size, width=10).grid(row=adv_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(advanced_frame, text="Processing batch size (higher = faster but more memory)").grid(row=adv_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        adv_row += 1
        
        # Background threshold
        ttk.Label(advanced_frame, text="Background Threshold:").grid(row=adv_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=0, to=255, textvariable=self.background_threshold, width=10, format="%.1f", increment=10.0).grid(row=adv_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(advanced_frame, text="Skip bright patches as background (0-255)").grid(row=adv_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        adv_row += 1
        
        # Smooth iterations
        ttk.Label(advanced_frame, text="Smoothing Iterations:").grid(row=adv_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=0, to=10, textvariable=self.smooth_iters, width=10).grid(row=adv_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(advanced_frame, text="Post-processing smoothing passes").grid(row=adv_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        adv_row += 1
        
        # Feature normalization
        ttk.Label(advanced_frame, text="Feature Normalization:").grid(row=adv_row, column=0, sticky=tk.W, pady=2)
        norm_combo = ttk.Combobox(advanced_frame, textvariable=self.feature_norm, values=["none", "zscore", "l2"], 
                                 state="readonly", width=8)
        norm_combo.grid(row=adv_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(advanced_frame, text="Feature preprocessing method").grid(row=adv_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        adv_row += 1
        
        # PCA dimensions
        ttk.Label(advanced_frame, text="PCA Dimensions:").grid(row=adv_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=0, to=512, textvariable=self.pca_dims, width=10).grid(row=adv_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(advanced_frame, text="Dimensionality reduction (0 = disabled)").grid(row=adv_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        adv_row += 1
        
        # Overlay max dimension
        ttk.Label(advanced_frame, text="Output Max Dimension:").grid(row=adv_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=256, to=2048, textvariable=self.overlay_max_dim, width=10).grid(row=adv_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(advanced_frame, text="Maximum size for output images").grid(row=adv_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        adv_row += 1
        
        # Random seed
        ttk.Label(advanced_frame, text="Random Seed:").grid(row=adv_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=0, to=9999, textvariable=self.seed, width=10).grid(row=adv_row, column=1, sticky=tk.W, pady=2)
        ttk.Label(advanced_frame, text="For reproducible results").grid(row=adv_row, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        adv_row += 1
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        row += 1
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to process images")
        self.status_label.grid(row=row, column=0, columnspan=3, pady=2)
        row += 1
        
        # Output text area
        ttk.Label(main_frame, text="Output Log:").grid(row=row, column=0, sticky=tk.W)
        row += 1
        
        self.output_text = scrolledtext.ScrolledText(main_frame, height=15, width=80, wrap=tk.WORD)
        self.output_text.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(row, weight=2)  # Give more weight to expand the log area
        
        # Running process reference
        self.process = None
    
    def browse_input_file(self):
        """Open file dialog to select input image."""
        filetypes = [
            ("All supported", "*.tif *.tiff *.scn *.png *.jpg *.jpeg"),
            ("TIFF files", "*.tif *.tiff"),
            ("SCN files", "*.scn"),
            ("Image files", "*.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=filetypes
        )
        
        if filename:
            self.input_file.set(filename)
            # Auto-set output directory to input file's directory if not set
            if not self.output_dir.get():
                input_path = Path(filename)
                self.output_dir.set(str(input_path.parent))
            # Read image dimensions and update tiles
            self.image_dims = self.get_image_dimensions(Path(filename))
            if self.image_dims:
                W, H = self.image_dims
                self.image_info.set(f"Image Resolution: {W} x {H} px")
            else:
                self.image_info.set("Image Resolution: unknown size")
            self.update_tile_estimate()
    
    def browse_output_dir(self):
        """Open folder dialog to select output directory."""
        folder = filedialog.askdirectory(
            title="Select Output Directory"
        )
        
        if folder:
            self.output_dir.set(folder)
    
    def reset_defaults(self):
        """Reset all parameters to default values."""
        self.patch_size.set(64)
        self.stride.set("auto")
        self.clusters.set(3)
        self.batch_size.set(512)
        self.device.set("auto")
        self.background_threshold.set(200.0)
        self.smooth_iters.set(1)
        self.feature_norm.set("l2")
        self.pca_dims.set(0)
        self.overlay_max_dim.set(768)
        self.feature_model.set("cnn")
        self.seed.set(42)
        self.save_overlay.set(True)
        self.save_segmentation.set(True)
        self.save_qupath_annotations.set(False)
        self.update_tile_estimate()

    def get_image_dimensions(self, path: Path):
        """Return (W, H) for supported image types without loading pixels."""
        try:
            suf = path.suffix.lower()
            if suf in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                from PIL import Image
                with Image.open(path) as im:
                    return im.size  # (W, H)
            if suf == ".scn":
                import tifffile
                with tifffile.TiffFile(str(path)) as tif:
                    best = None
                    best_pixels = -1
                    for series in tif.series:
                        axes = getattr(series, 'axes', '')
                        shape = series.shape
                        if not axes or not shape:
                            continue
                        axes = str(axes)
                        try:
                            idx_y = axes.index('Y')
                            idx_x = axes.index('X')
                        except ValueError:
                            continue
                        H = int(shape[idx_y])
                        W = int(shape[idx_x])
                        pixels = W * H
                        if pixels > best_pixels:
                            best_pixels = pixels
                            best = (W, H)
                    if best:
                        return best
        except Exception:
            return None
        return None

    def update_tile_estimate(self):
        """Update the estimated number of tiles based on current params and image size."""
        if not self.image_dims:
            self.tiles_info.set("Estimated Tiles: -")
            return
        W, H = self.image_dims
        try:
            patch = int(self.patch_size.get())
        except Exception:
            patch = 64
        stride_val = None
        try:
            stride_val = int(self.stride.get())
        except Exception:
            stride_val = None
        if not stride_val or stride_val <= 0:
            stride_val = patch

        def count_positions(length, patch_sz, stride_sz):
            if length <= patch_sz:
                return 1
            positions = list(range(0, max(0, length - patch_sz + 1), stride_sz))
            last = length - patch_sz
            if len(positions) == 0 or positions[-1] != last:
                positions.append(last)
            return len(positions)

        nx = count_positions(W, patch, stride_val)
        ny = count_positions(H, patch, stride_val)
        self.tiles_info.set(f"Estimated Tiles: {nx * ny} ({nx} × {ny})")
    
    def build_command(self):
        """Build the command line arguments for the tissue clustering script."""
        if not self.input_file.get():
            raise ValueError("Please select an input file")
        
        # Get the script directory
        script_dir = Path(__file__).parent
        script_path = script_dir / "tissue_clustering.py"
        
        cmd = [sys.executable, str(script_path)]
        cmd.extend(["--input", self.input_file.get()])
        cmd.extend(["--patch_size", str(self.patch_size.get())])
        
        if self.stride.get() != "auto":
            try:
                stride_val = int(self.stride.get())
                cmd.extend(["--stride", str(stride_val)])
            except ValueError:
                pass  # Use default (auto)
        
        cmd.extend(["--clusters", str(self.clusters.get())])
        cmd.extend(["--batch", str(self.batch_size.get())])
        cmd.extend(["--device", self.device.get()])
        cmd.extend(["--background_threshold", str(self.background_threshold.get())])
        cmd.extend(["--smooth_iters", str(self.smooth_iters.get())])
        cmd.extend(["--feature_norm", self.feature_norm.get()])
        cmd.extend(["--pca_dims", str(self.pca_dims.get())])
        cmd.extend(["--overlay_max_dim", str(self.overlay_max_dim.get())])
        cmd.extend(["--seed", str(self.seed.get())])
        
        if self.feature_model.get() == "simple":
            cmd.append("--simple_features")
        
        # Add output directory if specified
        if self.output_dir.get():
            cmd.extend(["--output_dir", self.output_dir.get()])
        
        # Add output format options
        output_formats = []
        if self.save_overlay.get():
            output_formats.append("overlay")
        if self.save_segmentation.get():
            output_formats.append("segmentation")
        if self.save_qupath_annotations.get():
            output_formats.append("qupath")
        
        if output_formats:
            cmd.extend(["--output_formats"] + output_formats)
        
        return cmd
    
    def run_analysis(self):
        """Run the tissue clustering analysis in a separate thread."""
        try:
            cmd = self.build_command()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        # Clear output text
        self.output_text.delete(1.0, tk.END)
        
        # Update UI state
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self.status_label.config(text="Running analysis...")
        
        # Start the process in a separate thread
        self.analysis_thread = threading.Thread(target=self._run_process, args=(cmd,))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def _run_process(self, cmd):
        """Run the subprocess and capture output."""
        import datetime
        
        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Track last progress message to avoid duplicates
            last_progress_line = ""
            last_log_line_index = None
            
            # Read output line by line
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")
                    
                    # Detect progress lines (contain progress bar patterns like [###---] or percentages)
                    is_progress = ('[' in line and ']' in line and ('#' in line or '-' in line)) or \
                                 ('Extract+Embed' in line) or \
                                 (line.count('/') == 1 and any(c.isdigit() for c in line))
                    
                    if is_progress and line != last_progress_line:
                        # This is a progress update - replace the last progress line
                        self.output_queue.put(('progress_update', f"[{current_time}] {line}"))
                        last_progress_line = line
                    elif not is_progress:
                        # Regular output line
                        self.output_queue.put(('output', f"[{current_time}] {line}"))
                        last_progress_line = ""
            
            # Wait for process to complete
            self.process.wait()
            
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            if self.process.returncode == 0:
                self.output_queue.put(('status', 'Analysis completed successfully!'))
                self.output_queue.put(('output', f"[{current_time}] Analysis completed successfully!"))
                self.output_queue.put(('success', True))
            else:
                self.output_queue.put(('status', f'Analysis failed with exit code {self.process.returncode}'))
                self.output_queue.put(('output', f"[{current_time}] Analysis failed with exit code {self.process.returncode}"))
                self.output_queue.put(('success', False))
                
        except Exception as e:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.output_queue.put(('status', f'Error: {str(e)}'))
            self.output_queue.put(('output', f"[{current_time}] Error: {str(e)}"))
            self.output_queue.put(('success', False))
        finally:
            self.output_queue.put(('finished', None))
    
    def stop_analysis(self):
        """Stop the running analysis."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.output_queue.put(('status', 'Analysis stopped by user'))
            self.output_queue.put(('finished', None))
    
    def check_queue(self):
        """Check for messages from the analysis thread."""
        try:
            while True:
                msg_type, msg_data = self.output_queue.get_nowait()
                
                if msg_type == 'output':
                    self.output_text.insert(tk.END, msg_data + '\n')
                    self.output_text.see(tk.END)
                elif msg_type == 'progress_update':
                    # Replace the last line if it was a progress line
                    current_pos = self.output_text.index(tk.END)
                    lines = self.output_text.get("1.0", tk.END).strip().split('\n')
                    
                    # Check if last line was a progress line
                    if lines and any(pattern in lines[-1] for pattern in ['Extract+Embed', '[', '#', '-']):
                        # Delete the last line and replace with new progress
                        self.output_text.delete(f"{len(lines)}.0", tk.END)
                        self.output_text.insert(tk.END, msg_data + '\n')
                    else:
                        # Just add the new progress line
                        self.output_text.insert(tk.END, msg_data + '\n')
                    self.output_text.see(tk.END)
                elif msg_type == 'status':
                    self.status_label.config(text=msg_data)
                elif msg_type == 'success':
                    if msg_data:
                        # Show completion message and offer to open output folder
                        if self.output_dir.get():
                            output_dir = Path(self.output_dir.get())
                        else:
                            input_path = Path(self.input_file.get())
                            output_dir = input_path.parent
                        
                        result = messagebox.askyesno(
                            "Analysis Complete", 
                            f"Analysis completed successfully!\n\nOutput files saved to:\n{output_dir}\n\nWould you like to open the output folder?"
                        )
                        
                        if result:
                            self.open_output_folder(output_dir)
                elif msg_type == 'finished':
                    # Reset UI state
                    self.run_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.progress.stop()
                    self.process = None
                    break
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def open_output_folder(self, folder_path):
        """Open the output folder in the system file manager."""
        import platform
        import subprocess
        
        try:
            if platform.system() == "Windows":
                os.startfile(folder_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", folder_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", folder_path])
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not open folder: {e}")

def main():
    root = tk.Tk()
    
    # Set application icon (optional)
    try:
        # You can add an icon file here if you have one
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    app = TissueClusteringGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
