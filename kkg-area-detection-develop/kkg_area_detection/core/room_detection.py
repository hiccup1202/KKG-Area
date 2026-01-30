"""
Room detection module for KKG Area Detection.

This module provides functionality to detect room names in images using OCR.
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None
    print('Warning: PIL could not be imported. Room detection will be limited.')

from ..service.ocr import find_room_name_in_region, handle_ocr


def get_room_names(
    image_path: str,
    contours_list: List[Dict[str, Any]],
    azure_endpoint: str,
    azure_key: str,
    room_name_keywords: Optional[List[str]] = None,
    default_name: str = 'Room 1',
) -> Dict[int, str]:
    """
    Get room names for detected regions in an image.

    Args:
        image_path: Path to the image file.
        contours_list: List of contour information as returned by get_approx_contours_and_vertices.
        azure_endpoint: Azure Form Recognizer endpoint.
        azure_key: Azure Form Recognizer API key.
        room_name_keywords: List of keywords that indicate a room name.
        default_name: Default name to return if no room name is found.

    Returns:
        Dictionary mapping region IDs to room names.
    """
    ocr_results = handle_ocr(image_path, azure_endpoint, azure_key, enable_cache=False)
    room_names = {}
    for i, contour_info in enumerate(contours_list):
        region_id = contour_info['id']
        vertices = contour_info['vertices']
        room_name = find_room_name_in_region(
            vertices,
            ocr_results,
            room_name_keywords,
            f'Room {i+1}' if default_name == 'Room 1' else default_name,
        )
        room_names[region_id] = room_name
    return room_names


def get_regions_with_room_names(
    image: Image.Image,
    image_path: str,
    azure_endpoint: str,
    azure_key: str,
    room_name_keywords: Optional[List[str]] = None,
    default_name: str = 'Room 1',
    epsilon: float = 0.015,
    use_smoothing: bool = False,
    smoothing_kernel_size: int = 5,
    smoothing_iterations: int = 1,
    use_angle_filter: bool = False,
    min_angle: float = 30.0,
    wall_filter: bool = False,
    target_label_ids: List[int] = [1, 2],
    edge_margin: float = 0.05,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """
    Get regions with room names for an image.

    This function combines segmentation, contour extraction, and room name detection.

    Args:
        image: PIL Image object.
        image_path: Path to the image file.
        azure_endpoint: Azure Form Recognizer endpoint.
        azure_key: Azure Form Recognizer API key.
        room_name_keywords: List of keywords that indicate a room name.
        default_name: Default name to return if no room name is found.
        epsilon: Approximation accuracy parameter for contour extraction.
        use_smoothing: Whether to apply smoothing to the segment mask.
        smoothing_kernel_size: Size of the Gaussian kernel for smoothing.
        smoothing_iterations: Number of smoothing iterations.
        use_angle_filter: Whether to filter out sharp angles.
        min_angle: Minimum angle in degrees to keep a vertex.
        wall_filter: Whether to apply wall filtering.
        target_label_ids: Target label IDs for wall filtering.
        edge_margin: Edge margin ratio for wall filtering.

    Returns:
        Tuple of (contours_list, room_names) where:
        - contours_list is a list of contour information
        - room_names is a dictionary mapping region IDs to room names
    """
    from ..core.contours import get_approx_contours_and_vertices
    from ..core.inference import get_segmentation_result

    try:
        segmentation_result = get_segmentation_result(image)
    except Exception as e:
        print(f'Error getting segmentation result: {e}')
        raise

    segmentation_map = segmentation_result['segmentation'].cpu().numpy()
    segments_info = segmentation_result['segments_info'].cpu().numpy()

    contours_list = get_approx_contours_and_vertices(
        segmentation_map,
        epsilon=epsilon,
        use_smoothing=use_smoothing,
        smoothing_kernel_size=smoothing_kernel_size,
        smoothing_iterations=smoothing_iterations,
        use_angle_filter=use_angle_filter,
        min_angle=min_angle,
        wall_filter=wall_filter,
        segments_info=segments_info,
        target_label_ids=target_label_ids,
        edge_margin=edge_margin,
        align_to_lines=False,  # room detection ではアライメント不要
    )

    # Get room names
    room_names = get_room_names(
        image_path,
        contours_list,
        azure_endpoint,
        azure_key,
        room_name_keywords,
        default_name,
    )

    return contours_list, room_names
