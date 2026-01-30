#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    # Try absolute imports first (for when the package is installed)
    from kkg_area_detection.fine_tuning.foundation.generator import (
        generate_foundation_plan, main)
    from kkg_area_detection.fine_tuning.foundation.grid_foundation import (
        create_foundation_with_width, create_grid_lines,
        generate_foundation_path)
    from kkg_area_detection.fine_tuning.foundation.symbols import (
        _compute_interior_mask, _place_symbols)
    from kkg_area_detection.fine_tuning.foundation.visual_effects import \
        add_noise
except ImportError:
    # Fall back to relative imports (for direct script execution)
    from .generator import generate_foundation_plan, main
    from .grid_foundation import (create_foundation_with_width,
                                  create_grid_lines, generate_foundation_path)
    from .symbols import _compute_interior_mask, _place_symbols
    from .visual_effects import add_noise

__all__ = [
    'generate_foundation_plan',
    'main',
    'create_grid_lines',
    'generate_foundation_path',
    'create_foundation_with_width',
    '_compute_interior_mask',
    '_place_symbols',
    'add_noise',
]
