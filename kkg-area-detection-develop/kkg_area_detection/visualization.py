"""
Visualization module for area detection results.

This module provides functions to visualize detected areas on images.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def create_color_mask_from_segmentation(
    segmentation_map: np.ndarray,
    segments_info: List[Dict[str, Any]],
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    use_random_colors: bool = True,
    fixed_class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    visible_classes: Optional[List[int]] = None,
) -> Image.Image:
    """
    Create a color mask image from a segmentation map.

    This function matches the behavior in app.py, creating a color mask directly
    from the segmentation map without contour processing.

    Args:
        segmentation_map: A 2D numpy array containing segment IDs.
        segments_info: A list of dictionaries with segment information.
        color_map: Optional dictionary mapping label_ids to RGB color tuples.
                  If not provided, random colors will be generated for each segment.
        use_random_colors: Whether to use random colors for segments not in color_map.
                  If False and color_map is None, all segments will use the same color.
        fixed_class_colors: Optional dictionary mapping specific label_ids to fixed RGB color tuples.
                  This overrides both color_map and random colors for the specified label_ids.
                  For example, {1: (255, 0, 0), 2: (0, 0, 255)} will make all class 1 segments red
                  and all class 2 segments blue, while other classes follow normal coloring rules.
        visible_classes: Optional list of label_ids to display. If provided, only segments with
                  these label_ids will be colored. All other segments will be transparent.
                  If None, all segments will be displayed.

    Returns:
        A PIL Image with colored regions.
    """
    # Default fixed class colors if not provided
    if fixed_class_colors is None:
        fixed_class_colors = {
            1: (255, 0, 0),    # Red for class 1
            2: (0, 0, 255)     # Blue for class 2
        }
    # Get dimensions from segmentation map
    height, width = segmentation_map.shape

    # Create empty RGB array
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate a random color map if not provided
    if color_map is None and use_random_colors:
        # Create a segment_id to color mapping instead of label_id
        # This ensures each segment gets a different color even if they have the same label_id
        segment_colors = {}
        for segment in segments_info:
            seg_id = segment['id']
            # Generate a random color (bright for visibility)
            segment_colors[seg_id] = (
                np.random.randint(100, 256),  # R: 100-255
                np.random.randint(100, 256),  # G: 100-255
                np.random.randint(100, 256),  # B: 100-255
            )

    # Default color if not using random colors and no color_map provided
    default_color = np.array([0, 255, 255], dtype=np.uint8)  # Cyan

    # Process each segment
    for segment in segments_info:
        seg_id = segment['id']
        label_id = segment['label_id']
        # Skip this segment if visible_classes is specified and this label_id is not in it
        if visible_classes is not None and label_id not in visible_classes:
            continue

        # First check if this label_id has a fixed class color
        if fixed_class_colors and label_id in fixed_class_colors:
            # Use the fixed color for this class
            color = np.array(fixed_class_colors[label_id], dtype=np.uint8)
        # Otherwise follow the normal coloring rules
        elif color_map and label_id in color_map:
            # Use color from provided color_map if available
            color = np.array(color_map[label_id], dtype=np.uint8)
        elif (
            use_random_colors
            and 'segment_colors' in locals()
            and seg_id in segment_colors
        ):
            # Use pre-generated random color for this segment
            color = np.array(segment_colors[seg_id], dtype=np.uint8)
        elif use_random_colors:
            # Generate a new random color if needed
            color = np.array(
                [
                    np.random.randint(100, 256),  # R: 100-255
                    np.random.randint(100, 256),  # G: 100-255
                    np.random.randint(100, 256),  # B: 100-255
                ],
                dtype=np.uint8,
            )
        else:
            # Use default color
            color = default_color

        # Apply color to the segment
        color_mask[segmentation_map == seg_id] = color

    return Image.fromarray(color_mask)


def create_color_mask(
    image_size: Tuple[int, int],
    regions: List[Dict[str, Any]],
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    fixed_class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    visible_classes: Optional[List[int]] = None,
) -> Image.Image:
    """
    Create a color mask image where each detected region is filled with a color.

    This is a compatibility function that works with the output of get_region_coordinates.
    For direct segmentation map processing, use create_color_mask_from_segmentation.

    Args:
        image_size: Tuple of (width, height) for the output mask.
        regions: List of region dictionaries as returned by get_region_coordinates.
                Each region should have 'coordinates' which can be either:
                - List of lists: [[x1, y1], [x2, y2], ...]
                - List of dicts: [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}, ...]
        color_map: Optional dictionary mapping label_ids to RGB color tuples.
                  If not provided, random colors will be generated.
        fixed_class_colors: Optional dictionary mapping specific label_ids to fixed RGB color tuples.
        visible_classes: Optional list of label_ids to display.

    Returns:
        A PIL Image with colored regions.
    """
    # Default fixed class colors if not provided
    if fixed_class_colors is None:
        fixed_class_colors = {
            1: (255, 0, 0),    # Red for class 1
            2: (0, 0, 255)     # Blue for class 2
        }

    # Create empty RGB array
    height, width = image_size[1], image_size[0]
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Separate walls and other regions for independent processing
    wall_regions = []
    other_regions = []

    for region in regions:
        if region.get('label_id') in [1, 2]:  # Wall classes
            wall_regions.append(region)
        else:
            other_regions.append(region)

    # Process non-wall regions first
    if other_regions and other_regions[0].get('coordinates') and len(other_regions[0]['coordinates']) > 0:
        for i, region in enumerate(other_regions):
            label_id = region['label_id']
            segment_id = region.get('segment_id', label_id)

            # Skip this region if visible_classes is specified and this label_id is not in it
            if visible_classes is not None and label_id not in visible_classes:
                continue

            # Determine color for this region - each region gets a unique color
            # First check if this label_id has a fixed class color
            if fixed_class_colors and label_id in fixed_class_colors:
                color = np.array(fixed_class_colors[label_id], dtype=np.uint8)
            elif color_map and label_id in color_map:
                color = np.array(color_map[label_id], dtype=np.uint8)
            else:
                # Generate a unique color for each region using its index
                np.random.seed(i + int(segment_id) * 1000 + 42)  # Use region index + segment_id for uniqueness
                color = np.array(
                    [
                        np.random.randint(100, 256),  # R: 100-255
                        np.random.randint(100, 256),  # G: 100-255
                        np.random.randint(100, 256),  # B: 100-255
                    ],
                    dtype=np.uint8,
                )

            # Draw filled contours for this region
            coordinates = region.get('coordinates', [])
            if not coordinates:
                continue

            try:
                # Check if coordinates is a list of dictionaries
                if isinstance(coordinates, list) and len(coordinates) > 0:
                    if isinstance(coordinates[0], dict):
                        # Convert [{'x': x, 'y': y}, ...] to [[x, y], ...]
                        contour_array = np.array([[point['x'], point['y']] for point in coordinates], dtype=np.int32)
                    elif isinstance(coordinates[0], list):
                        # Already in [[x, y], ...] format
                        contour_array = np.array(coordinates, dtype=np.int32)
                    else:
                        print(f"Warning: Unexpected coordinate format: {type(coordinates[0])}")
                        continue

                    # Ensure we have at least 3 points for a valid polygon
                    if len(contour_array) >= 3:
                        # For non-wall regions, use simple filling
                        cv2.fillPoly(color_mask, [contour_array], color.tolist())
                    else:
                        print(f"Warning: Contour has less than 3 points: {len(contour_array)}")
                else:
                    print(f"Warning: Invalid coordinates format: {type(coordinates)}")

            except (IndexError, KeyError, TypeError, ValueError) as e:
                # Skip this contour if there's an error processing it
                print(f"Warning: Skipping invalid contour: {e}")
                continue

    # Process wall regions separately with hierarchical filling
    if wall_regions:
        for i, region in enumerate(wall_regions):
            label_id = region['label_id']
            segment_id = region.get('segment_id', label_id)

            # Skip this region if visible_classes is specified and this label_id is not in it
            if visible_classes is not None and label_id not in visible_classes:
                continue

            # Determine color for this region
            if fixed_class_colors and label_id in fixed_class_colors:
                color = np.array(fixed_class_colors[label_id], dtype=np.uint8)
            elif color_map and label_id in color_map:
                color = np.array(color_map[label_id], dtype=np.uint8)
            else:
                # Generate a unique color for each region
                np.random.seed(i + int(segment_id) * 1000 + 42)
                color = np.array(
                    [
                        np.random.randint(100, 256),
                        np.random.randint(100, 256),
                        np.random.randint(100, 256),
                    ],
                    dtype=np.uint8,
                )

            # Process wall contours with holes
            coordinates = region.get('coordinates', [])
            if not coordinates:
                continue

            try:
                # Convert coordinates
                if isinstance(coordinates, list) and len(coordinates) > 0:
                    if isinstance(coordinates[0], dict):
                        contour_array = np.array([[point['x'], point['y']] for point in coordinates], dtype=np.int32)
                    elif isinstance(coordinates[0], list):
                        contour_array = np.array(coordinates, dtype=np.int32)
                    else:
                        continue

                    # Ensure we have at least 3 points
                    if len(contour_array) >= 3:
                        # Create a temporary mask for this wall
                        temp_mask = np.zeros((height, width, 3), dtype=np.uint8)

                        # Fill the outer polygon
                        cv2.fillPoly(temp_mask, [contour_array], color.tolist())

                        # Cut out holes
                        holes = region.get('holes', [])
                        if holes:
                            for hole in holes:
                                if isinstance(hole, list) and len(hole) > 0:
                                    if isinstance(hole[0], dict):
                                        hole_array = np.array([[point['x'], point['y']] for point in hole], dtype=np.int32)
                                    else:
                                        hole_array = np.array(hole, dtype=np.int32)

                                    if len(hole_array) >= 3:
                                        cv2.fillPoly(temp_mask, [hole_array], (0, 0, 0))

                        # Add wall mask to the color mask where it's non-black
                        wall_indices = np.any(temp_mask != 0, axis=2)
                        color_mask[wall_indices] = temp_mask[wall_indices]

            except (IndexError, KeyError, TypeError, ValueError) as e:
                print(f"Warning: Skipping invalid wall contour: {e}")
                continue

    return Image.fromarray(color_mask)


def overlay_mask_on_image(
    image: Image.Image, mask: Image.Image, alpha: float = 0.5
) -> Image.Image:
    """
    Overlay a color mask on an image with specified transparency.

    Args:
        image: Original PIL Image.
        mask: Color mask PIL Image (same size as original).
        alpha: Transparency level (0.0 to 1.0) where 1.0 is opaque.

    Returns:
        A PIL Image with the mask overlaid on the original image.
    """
    # Ensure image is RGB
    image = image.convert("RGB")

    # Convert to numpy arrays
    image_array = np.array(image, dtype=np.uint8)
    mask_array = np.array(mask, dtype=np.uint8)

    # Ensure same dimensions
    if image_array.shape[:2] != mask_array.shape[:2]:
        raise ValueError(
            f"Image dimensions {image_array.shape[:2]} don't match mask dimensions {mask_array.shape[:2]}"
        )

    # Create a mask of non-zero pixels in the mask
    mask_indices = np.any(mask_array != 0, axis=2)

    # Create output array (copy of original image)
    blended = image_array.copy()

    # Apply blending only where mask has non-zero values
    blended[mask_indices] = (
        (1 - alpha) * image_array[mask_indices] + alpha * mask_array[mask_indices]
    ).astype(np.uint8)

    return Image.fromarray(blended)


def visualize_regions(
    image: Image.Image,
    regions: List[Dict[str, Any]],
    alpha: float = 0.5,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    segmentation_result: Optional[Dict[str, Any]] = None,
    fixed_class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    visible_classes: Optional[List[int]] = None,
    use_contours_for_visualization: bool = True,
) -> Image.Image:
    """
    Visualize detected regions on an image.

    Args:
        image: Original PIL Image.
        regions: List of region dictionaries as returned by get_region_coordinates.
        alpha: Transparency level (0.0 to 1.0) where 1.0 is opaque.
        color_map: Optional dictionary mapping label_ids to RGB color tuples.
                  If not provided, random colors will be generated.
        segmentation_result: Optional raw segmentation result from get_segmentation_result.
                  If provided, this will be used directly for visualization.
        fixed_class_colors: Optional dictionary mapping specific label_ids to fixed RGB color tuples.
                  This overrides both color_map and random colors for the specified label_ids.
                  Default is {1: (255, 0, 0), 2: (0, 0, 255)} if not provided.
        visible_classes: Optional list of label_ids to display. If provided, only segments with
                  these label_ids will be colored. All other segments will be transparent.
                  For example, [2] will only show class 2 segments and hide all others.
        use_contours_for_visualization: If True and segmentation_result is provided, extract contours
                  with hole information for proper visualization of donut-shaped regions like walls.

    Returns:
        A PIL Image with detected regions visualized.
    """
    # Default fixed class colors if not provided
    if fixed_class_colors is None:
        fixed_class_colors = {
            1: (255, 0, 0),    # Red for class 1
            2: (0, 0, 255)     # Blue for class 2
        }
    # If segmentation_result is provided, use it directly
    if segmentation_result is not None and use_contours_for_visualization:
        # Import get_approx_contours_and_vertices for extracting contours with holes
        from kkg_area_detection.core.contours import get_approx_contours_and_vertices

        # Convert segmentation tensor to numpy array
        segmentation_map = segmentation_result['segmentation'].cpu().numpy()
        segments_info = segmentation_result['segments_info']

        # Handle negative values in segmentation map
        if np.any(segmentation_map < 0):
            print("Converting negative values in segmentation map...")
            # Create a copy to avoid modifying the original
            segmentation_map = segmentation_map.copy()
            # Replace negative values with a value that won't match any segment ID
            segmentation_map[segmentation_map < 0] = -9999

        # Extract contours with holes
        contours = get_approx_contours_and_vertices(
            segmentation_map,
            epsilon=0.015,
            use_shapely=True,  # Important: use shapely to get holes
            segments_info=segments_info,  # Pass segments_info for wall detection
            align_to_lines=False,  # 可視化目的ではアライメント不要
        )

        # Convert contours to regions format
        regions_with_holes = []
        for contour in contours:
            segment_id = int(contour['id'])
            # Find the corresponding segment info
            segment_info = next((s for s in segments_info if s['id'] == segment_id), None)
            if segment_info:
                region = {
                    'label_id': segment_info.get('label_id', segment_info.get('category_id', 0)),
                    'segment_id': segment_id,
                    'coordinates': contour['vertices'],
                    'holes': contour.get('holes', [])
                }
                regions_with_holes.append(region)

        color_mask = create_color_mask(image.size, regions_with_holes, color_map, fixed_class_colors, visible_classes)
    elif segmentation_result is not None:
        # Original implementation without contour extraction
        # Convert segmentation tensor to numpy array
        segmentation_map = segmentation_result['segmentation'].cpu().numpy()

        # Handle negative values in segmentation map
        if np.any(segmentation_map < 0):
            print("Converting negative values in segmentation map...")
            # Create a copy to avoid modifying the original
            segmentation_map = segmentation_map.copy()
            # Replace negative values with a value that won't match any segment ID
            segmentation_map[segmentation_map < 0] = -9999

        # Convert to uint8 for visualization
        segmentation_map = segmentation_map.astype(np.uint8)
        segments_info = segmentation_result['segments_info']

        color_mask = create_color_mask_from_segmentation(
            segmentation_map,
            segments_info,
            color_map,
            use_random_colors=True,  # Explicitly use random colors
            fixed_class_colors=fixed_class_colors,  # Use fixed colors for specific classes
            visible_classes=visible_classes,  # Only show specified classes
        )
    # Otherwise, use the regions to create a color mask
    else:
        color_mask = create_color_mask(image.size, regions, color_map, fixed_class_colors, visible_classes)

    # Overlay mask on original image
    return overlay_mask_on_image(image, color_mask, alpha)
