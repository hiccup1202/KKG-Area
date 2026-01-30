"""
KKG Area Detection Package

A package for detecting and visualizing areas in images using Mask2Former models.
"""

__version__ = '0.1.0'

try:
    from kkg_area_detection.core.inference import (get_segmentation_result,
                                                   initialize_model)
    from kkg_area_detection.core.room_detection import (
        get_regions_with_room_names, get_room_names)
except ImportError as e:
    print(f'Warning: Could not import core inference functions: {e}')

    def get_region_coordinates(*args, **kwargs):
        raise ImportError(
            'Core inference functions are not available. Please ensure all dependencies are installed.'
        )

    def get_segmentation_result(*args, **kwargs):
        raise ImportError(
            'Core inference functions are not available. Please ensure all dependencies are installed.'
        )

    def initialize_model(*args, **kwargs):
        raise ImportError(
            'Core inference functions are not available. Please ensure all dependencies are installed.'
        )

    def get_regions_with_room_names(*args, **kwargs):
        raise ImportError(
            'Room detection functions are not available. Please ensure all dependencies are installed.'
        )

    def get_room_names(*args, **kwargs):
        raise ImportError(
            'Room detection functions are not available. Please ensure all dependencies are installed.'
        )


try:
    from kkg_area_detection.core.contours import (
        get_approx_contours_and_vertices, visualize_contours, WallCenterlineExtractor)
except ImportError as e:
    print(f'Warning: Could not import contours module: {e}')

    def get_approx_contours_and_vertices(*args, **kwargs):
        raise ImportError(
            'Core inference functions are not available. Please ensure all dependencies are installed.'
        )

    def visualize_contours(*args, **kwargs):
        raise ImportError(
            'Core inference functions are not available. Please ensure all dependencies are installed.'
        )

    class WallCenterlineExtractor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'WallCenterlineExtractor is not available. Please ensure all dependencies are installed.'
            )


try:
    from kkg_area_detection.visualization import (
        create_color_mask, create_color_mask_from_segmentation,
        overlay_mask_on_image, visualize_regions)
except ImportError as e:
    print(f'Warning: Could not import visualization functions: {e}')

    def create_color_mask(*args, **kwargs):
        raise ImportError(
            'Visualization functions are not available. Please ensure all dependencies are installed.'
        )

    def create_color_mask_from_segmentation(*args, **kwargs):
        raise ImportError(
            'Visualization functions are not available. Please ensure all dependencies are installed.'
        )

    def overlay_mask_on_image(*args, **kwargs):
        raise ImportError(
            'Visualization functions are not available. Please ensure all dependencies are installed.'
        )

    def visualize_regions(*args, **kwargs):
        raise ImportError(
            'Visualization functions are not available. Please ensure all dependencies are installed.'
        )


try:
    from kkg_area_detection.core.polygon_alignment import (
        detect_lines_pylsd, visualize_lines_and_polygons)
except ImportError as e:
    print(f'Warning: Could not import polygon alignment module: {e}')

    def detect_lines_pylsd(*args, **kwargs):
        raise ImportError(
            'Polygon alignment functions are not available. Please ensure all dependencies are installed.'
        )

    def visualize_lines_and_polygons(*args, **kwargs):
        raise ImportError(
            'Polygon alignment functions are not available. Please ensure all dependencies are installed.'
        )


try:
    from kkg_area_detection.core.wall_line_extraction import (
        WallSkeletonToCAD, extract_wall_lines_from_segment)
except ImportError as e:
    print(f'Warning: Could not import wall line extraction module: {e}')

    class WallSkeletonToCAD:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'WallSkeletonToCAD is not available. Please ensure all dependencies are installed.'
            )

    def extract_wall_lines_from_segment(*args, **kwargs):
        raise ImportError(
            'Wall line extraction functions are not available. Please ensure all dependencies are installed.'
        )
