"""
Core functionality for the KKG Area Detection package.

This module contains the core algorithms and utilities for area detection.
"""

try:
    from kkg_area_detection.core.inference import (get_segmentation_result,
                                                   initialize_model)
except ImportError as e:
    print(f'Warning: Could not import inference module: {e}')

    def initialize_model(*args, **kwargs):
        raise ImportError(
            'Inference functions are not available. Please ensure all dependencies are installed.'
        )

    def get_segmentation_result(*args, **kwargs):
        raise ImportError(
            'Inference functions are not available. Please ensure all dependencies are installed.'
        )


try:
    from kkg_area_detection.core.contours import (
        get_approx_contours_and_vertices, visualize_contours)
except ImportError as e:
    print(f'Warning: Could not import contours module: {e}')

    def get_approx_contours_and_vertices(*args, **kwargs):
        raise ImportError(
            'Inference functions are not available. Please ensure all dependencies are installed.'
        )

    def visualize_contours(*args, **kwargs):
        raise ImportError(
            'Visualization functions are not available. Please ensure all dependencies are installed.'
        )
