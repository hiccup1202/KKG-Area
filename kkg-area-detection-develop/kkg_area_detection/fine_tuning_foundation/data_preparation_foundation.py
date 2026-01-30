#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module is a wrapper around the foundation package functionality.
It maintains backward compatibility with existing code that imports from this module.
All functions are re-exported to maintain the same API.
"""

# Import all functions from the foundation package
try:
    # Try absolute import first (for when the package is installed)
    from kkg_area_detection.fine_tuning.foundation import (
        _compute_interior_mask, _place_symbols, add_noise,
        create_foundation_with_width, create_grid_lines,
        generate_foundation_path, generate_foundation_plan, main)
except ImportError:
    # Fall back to relative import (for direct script execution)
    from foundation import (_compute_interior_mask, _place_symbols, add_noise,
                            create_foundation_with_width, create_grid_lines,
                            generate_foundation_path, generate_foundation_plan,
                            main)

# Re-export all imported functions
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

# For backward compatibility
if __name__ == '__main__':
    main()
