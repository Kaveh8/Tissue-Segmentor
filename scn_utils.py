#!/usr/bin/env python3
"""
Lightweight SCN utilities for converting Leica .scn files to high-resolution TIFF
compatible with the tissue_clustering pipeline, without extra heavy deps.
"""

from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import tifffile


def _find_highest_resolution_page(scn_path: Path) -> int:
    """Return the index of the SCN page with the most pixels."""
    with tifffile.TiffFile(scn_path) as tif:
        max_pixels = -1
        best_idx = 0
        for i, page in enumerate(tif.pages):
            if hasattr(page, 'shape') and page.shape is not None and len(page.shape) >= 2:
                pixels = int(page.shape[0]) * int(page.shape[1])
                if pixels > max_pixels:
                    max_pixels = pixels
                    best_idx = i
        return best_idx


def convert_scn_to_tiff(scn_path: Path, output_path: Optional[Path] = None, compression: str = 'lzw') -> Path:
    """
    Convert a Leica .scn file to a TIFF by extracting the highest-resolution page.

    Args:
        scn_path: Path to the .scn file
        output_path: Optional path to write the .tiff; defaults to <stem>_highres.tiff next to the .scn
        compression: TIFF compression method ('lzw', 'jpeg', 'none')

    Returns:
        Path to the created TIFF
    """
    if output_path is None:
        output_path = scn_path.with_name(f"{scn_path.stem}_highres.tiff")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with tifffile.TiffFile(scn_path) as tif:
            page_index = _find_highest_resolution_page(scn_path)
            page = tif.pages[page_index]
            image_data = page.asarray()

    tifffile.imwrite(
        output_path,
        image_data,
        compression=compression,
        metadata={'source_file': str(scn_path), 'page_index': page_index},
    )

    return output_path


def ensure_tiff_from_input(input_path: Path) -> Path:
    """
    If input is .scn, convert to TIFF (if not already converted) and return the TIFF path.
    Otherwise, return the original input path.
    """
    suffix = input_path.suffix.lower()
    if suffix == '.scn':
        # Reuse existing output if present
        desired_tiff = input_path.with_name(f"{input_path.stem}_highres.tiff")
        if desired_tiff.exists():
            return desired_tiff
        return convert_scn_to_tiff(input_path, desired_tiff)
    return input_path


