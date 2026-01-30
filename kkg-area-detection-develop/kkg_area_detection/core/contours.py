import math
from typing import List, Tuple, Dict, Any, Optional, Union

from ..types.contours import ContourInfo, ContourVertex

import numpy as np

import cv2
from PIL import Image, ImageDraw, ImageFont


def angle_between(
    p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]
) -> float:
    """
    Calculate the angle between three points.

    Args:
        p1: First point (x, y)
        p2: Center point (x, y)
        p3: Third point (x, y)

    Returns:
        float: Angle in degrees
    """

    def vector(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        return (b[0] - a[0], b[1] - a[1])

    v1 = vector(p2, p1)
    v2 = vector(p2, p3)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    cos_angle = dot / (norm1 * norm2 + 1e-6)
    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(angle)


def filter_sharp_vertices(
    vertices: List[ContourVertex], min_angle: float = 30.0
) -> List[ContourVertex]:
    """
    Filter out vertices that form sharp angles.

    Args:
        vertices: List of vertices to filter
        min_angle: Minimum angle in degrees to keep a vertex (default: 30.0)

    Returns:
        List[ContourVertex]: Filtered list of vertices
    """
    if len(vertices) < 3:
        return vertices

    filtered = []
    for i in range(len(vertices)):
        p1 = vertices[i - 1]
        p2 = vertices[i]
        p3 = vertices[(i + 1) % len(vertices)]
        angle = angle_between(
            (p1['x'], p1['y']), (p2['x'], p2['y']), (p3['x'], p3['y'])
        )
        if angle > min_angle:
            filtered.append(p2)
    return filtered


def smooth_segment_mask(
    mask: np.ndarray, kernel_size: int = 5, iterations: int = 1
) -> np.ndarray:
    """
    Smooth a binary mask using Gaussian blur and morphological operations.

    Args:
        mask: Binary mask to smooth
        kernel_size: Size of the Gaussian kernel (default: 5)
        iterations: Number of smoothing iterations (default: 1)

    Returns:
        np.ndarray: Smoothed mask
    """
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    opened = cv2.morphologyEx(
        blurred, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=iterations
    )
    return opened


def filter_edge_contours_by_wall_reference(
    contours: List[ContourInfo],
    segment_array: np.ndarray,
    segments_info: List[Dict[str, Any]],
    target_label_ids: List[int],
    margin_ratio: float = 0.05
) -> List[ContourInfo]:
    """
    Filter and remove contours that are further to the edges than the outermost points of label IDs 1 and 2.

    Args:
        contours: List of contour information
        segment_array: Segmentation array
        segments_info: List of segment information
        target_label_ids: List of reference label IDs (IDs representing walls)
        margin_ratio: Margin for edge detection (ratio to image size)

    Returns:
        List[ContourInfo]: Filtered list of contour information
    """
    if not contours:
        return contours

    # Collect segment IDs of target labels
    wall_segment_ids = []
    for segment in segments_info:
        # Get label_id or category_id
        label_id = segment.get('label_id', segment.get('category_id', 0))
        if label_id in target_label_ids:
            wall_segment_ids.append(segment['id'])

    if not wall_segment_ids:
        print(f"Warning: No segments found with label IDs {target_label_ids}")
        return contours

    # Calculate bounds of wall segments
    wall_bounds = []
    for segment_id in wall_segment_ids:
        segment_mask = (segment_array == segment_id)
        if not np.any(segment_mask):
            continue

        y_coords, x_coords = np.where(segment_mask)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        wall_bounds.append({
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y
        })

    if not wall_bounds:
        return contours

    # Calculate the outermost bounds of all wall segments
    wall_leftmost = min(bound['min_x'] for bound in wall_bounds)
    wall_rightmost = max(bound['max_x'] for bound in wall_bounds)
    wall_topmost = min(bound['min_y'] for bound in wall_bounds)
    wall_bottommost = max(bound['max_y'] for bound in wall_bounds)

    # Calculate margin
    height, width = segment_array.shape
    margin = int(min(height, width) * margin_ratio)

    # Calculate boundaries with applied margin (inside the walls)
    left_boundary = wall_leftmost + margin
    right_boundary = wall_rightmost - margin
    top_boundary = wall_topmost + margin
    bottom_boundary = wall_bottommost - margin

    # Keep only contours within the boundaries
    filtered_contours = []
    for i, contour in enumerate(contours):
        # Always keep the wall segments themselves
        if int(contour['id']) in wall_segment_ids:
            filtered_contours.append(contour)
            continue

        vertices = contour['vertices']
        if not vertices:
            continue

        x_coords = [v['x'] for v in vertices]
        y_coords = [v['y'] for v in vertices]

        # Calculate contour's bounding box
        min_cx = min(x_coords)
        max_cx = max(x_coords)
        min_cy = min(y_coords)
        max_cy = max(y_coords)

        # Check if the entire contour's bounding box is within the boundaries
        within_bounds = (left_boundary <= min_cx and max_cx <= right_boundary and
                         top_boundary <= min_cy and max_cy <= bottom_boundary)

        if within_bounds:
            filtered_contours.append(contour)

    return filtered_contours


def get_approx_contours_and_vertices(
    segment_array: np.ndarray,
    epsilon: float = 0.015,
    use_smoothing: bool = False,
    smoothing_kernel_size: int = 5,
    smoothing_iterations: int = 1,
    use_angle_filter: bool = False,
    min_angle: float = 30.0,
    wall_filter: bool = False,
    segments_info: List[Dict[str, Any]] = None,
    target_label_ids: List[int] = [2, 3],
    edge_margin: float = 0.05,
    use_shapely: bool = True,
    # Line detection and alignment control
    align_to_lines: bool = True,
    return_lines: bool = False,
    # Line detection and alignment parameters (optional)
    original_image: Optional[np.ndarray] = None,
    # PyLSD parameters
    scale: float = 0.8,
    sigma_scale: float = 0.6,
    quant: float = 2.0,
    ang_th: float = 22.5,
    density_th: float = 0.7,
    n_bins: int = 1024,
    line_angle_tolerance: float = 5.0,
    max_snap_distance: float = 20.0,
    snap_angle_tolerance: float = 15.0,
    min_edge_length: float = 10.0,
    min_line_length_ratio: float = 0.3,
    straighten_first: bool = True,
    straightening_angle_tolerance: float = 10.0,
    fill_holes: bool = True,
    ensure_complete_perimeter_flag: bool = True,
    extend_to_outermost: bool = True,
    parallel_search_distance: float = 5.0,
    # Wall centerline extraction parameters
    extract_wall_centerlines: bool = False,
    centerline_rdp_epsilon: float = 3.0,
    centerline_min_length: float = 20.0,
    centerline_hough_min_length: int = 80,
    centerline_hough_max_gap: int = 30,
    centerline_hough_threshold: int = 40,
    centerline_enable_refinement: bool = True,
    # Wall line extraction parameters (new skeletonization method)
    extract_wall_lines: bool = False,
    wall_line_class_ids: List[int] = None,  # Default to [2, 3] (inwall, door) if None
    wall_line_rdp_epsilon: float = 3.0,
    wall_line_min_segment_length: float = 20,
    wall_line_enable_refinement: bool = True,
    wall_line_extend_threshold: float = 40,
    wall_line_snap_tolerance: float = 10,
    wall_line_manhattan_angle_tolerance: float = 10,
    wall_line_coordinate_snap_tolerance: float = 10,
) -> Union[List[ContourInfo], Tuple[List[ContourInfo], List[Any]]]:
    """
    Extract contours and vertices from a segment array with optional smoothing and angle filtering.

    This function combines the functionality of get_contours_with_smoothing and the original
    get_approx_contours_and_vertices, providing more control over the contour extraction process.

    Processing Steps:
    1. Smoothing (Optional)
       - Applies Gaussian smoothing to the segment mask
       - Helps reduce noise and small irregularities in the contour
       - Parameters:
         - smoothing_kernel_size: Size of the Gaussian kernel (default: 5)
         - smoothing_iterations: Number of smoothing passes (default: 1)

    2. Contour Extraction (Optional: depends on epsilon)
       - Extracts contours using cv2.findContours
       - Approximates contours using cv2.approxPolyDP
       - epsilon parameter controls the approximation accuracy:
         - epsilon = 0: No approximation, original contour
         - epsilon > 0: Approximated contour with specified accuracy
         - Higher epsilon values result in fewer vertices

    3. Angle Filtering (Optional)
       - Filters out sharp angles in the contour
       - Helps remove unwanted vertices in straight lines
       - min_angle parameter controls the minimum angle to keep (default: 30 degrees)

    4. Wall Filtering (Optional)
       - Filters and removes contours that are further to the edges than the outermost points of walls with label ID 2 or 3 during post-processing
       - Parameters:
         - target_label_ids: List of label IDs to reference as walls (default: [2, 3])
         - edge_margin: Margin for edge detection (ratio to image size) (default: 0.05)

    5. Shapely Integration (Optional)
       - When use_shapely=True (default), uses Shapely for proper handling of polygons with holes
       - Automatically detects and preserves donut-shaped structures for all segments
       - Uses cv2.RETR_TREE to capture hierarchical contour relationships
       - Maintains topology during polygon simplification for all object types

    Example:
        image = Image.open("/path/to/image")
        segmentation_result = kkg_area_detection.get_segmentation_result(image)
        segmentation_map = segmentation_result["segmentation"].cpu().numpy()
        segments_info = segmentation_result["segments_info"]

        original_contours = kkg_area_detection.get_approx_contours_and_vertices(
            segmentation_map,
            epsilon=0
        )
        original_image = kkg_area_detection.visualize_contours(
            image=image,
            contours_list=original_contours
        )

        smoothed_contours = kkg_area_detection.get_approx_contours_and_vertices(
            segmentation_map,
            epsilon=0.015,
            use_smoothing=True,
            smoothing_kernel_size=5,
            smoothing_iterations=2
        )
        smoothed_image = kkg_area_detection.visualize_contours(
            image=image,
            contours_list=smoothed_contours
        )

        filtered_contours = kkg_area_detection.get_approx_contours_and_vertices(
            segmentation_map,
            epsilon=0.015,
            use_angle_filter=True,
            min_angle=30.0
        )
        filtered_image = kkg_area_detection.visualize_contours(
            image=image,
            contours_list=filtered_contours
        )

        wall_contours = kkg_area_detection.get_approx_contours_and_vertices(
            segmentation_map,
            epsilon=0.015,
            wall_filter=True,
            segments_info=segments_info,
            target_label_ids=[2, 3],
            edge_margin=0.05
        )
        wall_image = kkg_area_detection.visualize_contours(
            image=image,
            contours_list=wall_contours
        )

    Args:
        segment_array (np.ndarray): 2D numpy array of shape (Height, Width)
        epsilon (float): Approximation accuracy parameter (0.0 to 1.0, default: 0.015)
        use_smoothing (bool): Whether to apply smoothing to the segment mask (default: False)
        smoothing_kernel_size (int): Size of the Gaussian kernel for smoothing (default: 5)
        smoothing_iterations (int): Number of smoothing iterations (default: 1)
        use_angle_filter (bool): Whether to filter out sharp angles (default: False)
        min_angle (float): Minimum angle in degrees to keep a vertex (default: 30.0)
        wall_filter (bool): Whether to apply wall filtering (default: False)
        segments_info (List[Dict[str, Any]]): Segments info for wall filtering (required if wall_filter=True)
        target_label_ids (List[int]): Target label IDs for wall filtering (default: [2, 3])
        edge_margin (float): Edge margin ratio for wall filtering (default: 0.05)
        use_shapely (bool): Whether to use Shapely for proper polygon with holes handling (default: True)

        # Line detection and alignment control:
        align_to_lines (bool): Whether to perform line detection and alignment (default: True)
        return_lines (bool): Whether to return detected lines along with contours (default: False)

        # Line detection and alignment parameters (only used when align_to_lines is True):
        original_image (Optional[np.ndarray]): Original image for line detection and alignment (default: None)
        # PyLSD parameters:
        scale (float): Image scaling factor (default: 0.8)
        sigma_scale (float): Gaussian kernel sigma value scale (default: 0.6)
        quant (float): Gradient quantization level (default: 2.0)
        ang_th (float): Angle tolerance threshold in degrees (default: 22.5)
        density_th (float): Density threshold (default: 0.7)
        n_bins (int): Number of histogram bins (default: 1024)
        line_angle_tolerance (float): Tolerance for horizontal/vertical lines in degrees (default: 5.0)
        # Alignment parameters:
        max_snap_distance (float): Maximum distance to snap vertices to lines (default: 20.0)
        snap_angle_tolerance (float): Maximum angle difference for line matching in degrees (default: 15.0)
        min_edge_length (float): Minimum edge length to keep after alignment (default: 10.0)
        min_line_length_ratio (float): Minimum ratio of line length to edge length (default: 0.3)
        straighten_first (bool): Whether to straighten edges to horizontal/vertical first (default: True)
        straightening_angle_tolerance (float): Angle tolerance for straightening in degrees (default: 10.0)
        fill_holes (bool): Whether to fill holes in polygons (default: True)
        ensure_complete_perimeter_flag (bool): Whether to ensure complete perimeter edges (default: True)
        extend_to_outermost (bool): Whether to extend polygon edges to outermost parallel lines (default: True)
        parallel_search_distance (float): Search distance for finding parallel lines (default: 5.0)

        # Wall centerline extraction parameters:
        extract_wall_centerlines (bool): Whether to extract wall centerlines instead of contours (default: False)
        centerline_rdp_epsilon (float): RDP simplification tolerance for centerlines (default: 3.0)
        centerline_min_length (float): Minimum length for centerline segments (default: 20.0)
        centerline_hough_min_length (int): Minimum line length for Hough detection on centerlines (default: 80)
        centerline_hough_max_gap (int): Maximum gap for Hough line merging on centerlines (default: 30)
        centerline_hough_threshold (int): Threshold for Hough line detection on centerlines (default: 40)
        centerline_enable_refinement (bool): Whether to apply CAD refinement to centerlines (default: True)

        # Wall line extraction parameters (new skeletonization method):
        extract_wall_lines (bool): Whether to extract wall lines using skeletonization for specified class IDs (default: False)
        wall_line_class_ids (List[int]): Class IDs to extract as lines (default: [2, 3] for inwall, door)
        wall_line_rdp_epsilon (float): RDP simplification tolerance for wall lines (default: 3.0)
        wall_line_min_segment_length (float): Minimum length for wall line segments (default: 20)
        wall_line_enable_refinement (bool): Whether to apply CAD refinement to wall lines (default: True)
        wall_line_extend_threshold (float): Maximum distance for line extension (default: 40)
        wall_line_snap_tolerance (float): Tolerance for endpoint snapping (default: 10)
        wall_line_manhattan_angle_tolerance (float): Angle tolerance for Manhattan alignment (default: 10)
        wall_line_coordinate_snap_tolerance (float): Distance tolerance for coordinate snapping (default: 10)

    Returns:
        Union[List[ContourInfo], Tuple[List[ContourInfo], List['Line']]]:
            - If return_lines is False: List of dictionaries containing segment IDs and their approximated contour vertices
            - If return_lines is True: Tuple of (aligned_contours, detected_lines)

    Raises:
        ValueError: If required dependencies (OpenCV, NumPy, SciPy) are not available
        RuntimeError: If contour detection fails
    """
    if cv2 is None or np is None:
        raise ValueError('Required dependencies (OpenCV, NumPy) not available')

    # Handle wall centerline extraction
    if extract_wall_centerlines:
        try:
            extractor = WallCenterlineExtractor(segment_array, wall_label_ids=[2, 3])
            centerline_contours = extractor.extract_centerlines(
                rdp_epsilon=centerline_rdp_epsilon,
                min_segment_length=centerline_min_length,
                hough_min_length=centerline_hough_min_length,
                hough_max_gap=centerline_hough_max_gap,
                hough_threshold=centerline_hough_threshold,
                enable_refinement=centerline_enable_refinement,
            )

            if return_lines:
                # Return empty lines list for centerlines since they are already line segments
                return centerline_contours, []
            else:
                return centerline_contours

        except Exception as e:
            print(f"Error in wall centerline extraction: {e}")
            print("Falling back to standard contour extraction")

    # Handle wall line extraction with skeletonization
    if extract_wall_lines:
        # Default to class IDs 2, 3 (inwall, door) if not specified
        if wall_line_class_ids is None:
            wall_line_class_ids = [2, 3]

        # Import the wall line extraction function
        from .wall_line_extraction import extract_wall_lines_from_segment

        # Process each segment
        all_contours = []

        for segment_id in np.unique(segment_array):
            if segment_id < 0:
                continue

            # Check if this segment should be processed as lines
            segment_label_id = None
            if segments_info:
                for seg_info in segments_info:
                    if seg_info['id'] == segment_id:
                        segment_label_id = seg_info.get('label_id', seg_info.get('category_id'))
                        break

            # Extract segment mask
            segment_mask = (segment_array == segment_id).astype(np.uint8)

            # If this is a wall segment that should be extracted as lines
            if segment_label_id in wall_line_class_ids:
                try:
                    # Extract lines using skeletonization
                    wall_lines = extract_wall_lines_from_segment(
                        segment_mask=segment_mask,
                        rdp_epsilon=wall_line_rdp_epsilon,
                        min_segment_length=wall_line_min_segment_length,
                        enable_refinement=wall_line_enable_refinement,
                        extend_threshold=wall_line_extend_threshold,
                        snap_tolerance=wall_line_snap_tolerance,
                        manhattan_angle_tolerance=wall_line_manhattan_angle_tolerance,
                        coordinate_snap_tolerance=wall_line_coordinate_snap_tolerance,
                    )

                    # Convert lines to contour format
                    for line in wall_lines:
                        coords = list(line.coords)
                        if len(coords) >= 2:
                            # Create a line segment as a contour
                            line_vertices = [{'x': int(x), 'y': int(y)} for x, y in coords]
                            contour_info = {
                                'id': segment_id,
                                'vertices': line_vertices,
                                'is_line': True,  # Mark as line segment
                            }
                            all_contours.append(contour_info)

                except Exception as e:
                    print(f"Error extracting wall lines for segment {segment_id}: {e}")
                    # Fallback to regular contour extraction for this segment
                    # Process as regular contour
                    if use_smoothing:
                        segment_mask = smooth_segment_mask(
                            segment_mask,
                            kernel_size=smoothing_kernel_size,
                            iterations=smoothing_iterations,
                        )

                    contours_found, _ = cv2.findContours(
                        segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours_found:
                        contour = max(contours_found, key=cv2.contourArea)
                        perimeter = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)

                        vertices = approx.squeeze().tolist()
                        if not isinstance(vertices[0], list):
                            vertices = [vertices]

                        contour_vertices = [
                            {'x': int(x), 'y': int(y)} for x, y in vertices
                        ]

                        if use_angle_filter:
                            contour_vertices = filter_sharp_vertices(contour_vertices, min_angle)

                        contour_info = {
                            'id': segment_id,
                            'vertices': contour_vertices,
                        }
                        all_contours.append(contour_info)
            else:
                # Process as regular polygon
                # Apply the standard contour extraction process
                if use_smoothing:
                    segment_mask = smooth_segment_mask(
                        segment_mask,
                        kernel_size=smoothing_kernel_size,
                        iterations=smoothing_iterations,
                    )

                # Use the same contour extraction logic as below
                if use_shapely:
                    # Use RETR_TREE to get hierarchy information for detecting holes
                    contours_found, hierarchy = cv2.findContours(
                        segment_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if not contours_found or hierarchy is None:
                        continue

                    hierarchy = hierarchy[0]

                    # Process contours with hierarchy
                    for i in range(len(contours_found)):
                        next_idx, prev_idx, first_child, parent = hierarchy[i]

                        # Only process top-level contours
                        if parent == -1:
                            # Skip very small contours
                            if cv2.contourArea(contours_found[i]) < 50:
                                continue

                            # Approximate the outer contour
                            perimeter = cv2.arcLength(contours_found[i], True)
                            approx_outer = cv2.approxPolyDP(
                                contours_found[i], epsilon * perimeter, True
                            )

                            # Create vertices
                            outer_vertices = approx_outer.squeeze().tolist()
                            if not isinstance(outer_vertices[0], list):
                                outer_vertices = [outer_vertices]

                            contour_vertices = [
                                {'x': int(x), 'y': int(y)} for x, y in outer_vertices
                            ]

                            if use_angle_filter:
                                contour_vertices = filter_sharp_vertices(contour_vertices, min_angle)

                            contour_info = {
                                'id': segment_id,
                                'vertices': contour_vertices,
                            }

                            # Check for holes
                            if first_child != -1:
                                holes = []
                                child_idx = first_child
                                while child_idx != -1:
                                    child_contour = contours_found[child_idx]
                                    child_perimeter = cv2.arcLength(child_contour, True)
                                    approx_hole = cv2.approxPolyDP(
                                        child_contour, epsilon * child_perimeter, True
                                    )
                                    holes.append(approx_hole.squeeze().tolist())

                                    # Move to next sibling
                                    child_idx = hierarchy[child_idx][0]

                                if holes:
                                    # Process hole vertices
                                    processed_holes = []
                                    for hole in holes:
                                        if not isinstance(hole[0], list):
                                            hole = [hole]
                                        hole_verts = [{'x': int(x), 'y': int(y)} for x, y in hole]
                                        if use_angle_filter:
                                            hole_verts = filter_sharp_vertices(hole_verts, min_angle)
                                        processed_holes.append(hole_verts)
                                    contour_info['holes'] = processed_holes

                            all_contours.append(contour_info)
                else:
                    # Original implementation without Shapely
                    contours_found, _ = cv2.findContours(
                        segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours_found:
                        # Process significant contours
                        if len(contours_found) > 1:
                            # Calculate areas for all contours
                            contour_areas = [(contour, cv2.contourArea(contour)) for contour in contours_found]
                            # Find the largest area
                            largest_area = max(area for _, area in contour_areas)
                            # Keep contours with area >= 20% of the largest
                            area_threshold = largest_area * 0.2
                            contours_to_process = [contour for contour, area in contour_areas if area >= area_threshold]
                        else:
                            contours_to_process = contours_found

                        for contour in contours_to_process:
                            perimeter = cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)

                            vertices = approx.squeeze().tolist()
                            if not isinstance(vertices[0], list):
                                vertices = [vertices]

                            contour_vertices = [
                                {'x': int(x), 'y': int(y)} for x, y in vertices
                            ]

                            if use_angle_filter:
                                contour_vertices = filter_sharp_vertices(contour_vertices, min_angle)

                            contour_info = {
                                'id': segment_id,
                                'vertices': contour_vertices,
                            }
                            all_contours.append(contour_info)

        # Apply wall filter and alignment if needed
        contours = all_contours

        # Apply wall filter to remove edge contours if enabled
        if wall_filter:
            contours = filter_edge_contours_by_wall_reference(
                contours, segment_array, segments_info, target_label_ids, edge_margin
            )

        # If align_to_lines is True, perform line detection and alignment (but skip line segments)
        if align_to_lines:
            if original_image is None:
                raise ValueError("original_image is required when align_to_lines is True")

            from .polygon_alignment import (
                detect_lines_pylsd,
                align_polygon_to_lines
            )

            # Detect lines in the original image
            detected_lines = detect_lines_pylsd(
                image=original_image,
                scale=scale,
                sigma_scale=sigma_scale,
                quant=quant,
                ang_th=ang_th,
                density_th=density_th,
                n_bins=n_bins,
                angle_tolerance=line_angle_tolerance,
            )

            # Align each contour to the detected lines (skip line segments)
            aligned_contours = []
            for contour in contours:
                # Skip if this is a line segment
                if contour.get('is_line', False):
                    aligned_contours.append(contour)
                    continue

                # Fill holes if requested
                processed_contour = contour
                if fill_holes:
                    from .polygon_alignment import fill_polygon_holes
                    processed_contour = fill_polygon_holes(contour)

                # Extract vertices
                vertices = processed_contour['vertices']

                # Align vertices to detected lines
                aligned_vertices = align_polygon_to_lines(
                    vertices=vertices,
                    lines=detected_lines,
                    max_distance=max_snap_distance,
                    angle_tolerance=snap_angle_tolerance,
                    min_edge_length=min_edge_length,
                    min_line_length_ratio=min_line_length_ratio,
                    straighten_first=straighten_first,
                    straightening_angle_tolerance=straightening_angle_tolerance,
                    fill_holes=fill_holes,
                    ensure_complete_perimeter_flag=ensure_complete_perimeter_flag,
                    extend_to_outermost=extend_to_outermost,
                    parallel_search_distance=parallel_search_distance,
                )

                # Create new contour with aligned vertices
                aligned_contour = processed_contour.copy()
                aligned_contour['vertices'] = aligned_vertices

                # If the contour has holes and we're not filling them, align them as well
                if not fill_holes and 'holes' in aligned_contour:
                    aligned_holes = []
                    for hole in aligned_contour['holes']:
                        aligned_hole = align_polygon_to_lines(
                            vertices=hole,
                            lines=detected_lines,
                            max_distance=max_snap_distance,
                            angle_tolerance=snap_angle_tolerance,
                            min_edge_length=min_edge_length,
                            min_line_length_ratio=min_line_length_ratio,
                            straighten_first=straighten_first,
                            straightening_angle_tolerance=straightening_angle_tolerance,
                            fill_holes=fill_holes,
                            ensure_complete_perimeter_flag=ensure_complete_perimeter_flag,
                            extend_to_outermost=extend_to_outermost,
                            parallel_search_distance=parallel_search_distance,
                        )
                        if len(aligned_hole) >= 3:  # Keep only valid holes
                            aligned_holes.append(aligned_hole)
                    aligned_contour['holes'] = aligned_holes

                aligned_contours.append(aligned_contour)

            # Return based on return_lines flag
            if return_lines:
                return aligned_contours, detected_lines
            else:
                return aligned_contours

        return contours

    # Standard contour extraction (when not using wall line extraction)
    contours: List[ContourInfo] = []

    for segment_id in np.unique(segment_array):
        if segment_id < 0:
            continue

        segment_mask = (segment_array == segment_id).astype(np.uint8)

        if use_smoothing:
            segment_mask = smooth_segment_mask(
                segment_mask,
                kernel_size=smoothing_kernel_size,
                iterations=smoothing_iterations,
            )

        if use_shapely:
            # Use RETR_TREE to get hierarchy information for detecting holes
            contours_found, hierarchy = cv2.findContours(
                segment_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours_found or hierarchy is None:
                continue

            hierarchy = hierarchy[0]

            # Check if there are any holes in the shape by looking at hierarchy
            has_holes = False
            for i in range(len(contours_found)):
                next_idx, prev_idx, first_child, parent = hierarchy[i]
                if parent != -1:  # This contour has a parent, so it's a hole
                    has_holes = True
                    break

            if has_holes:
                # Use hierarchical processing to properly handle shapes with holes
                # Process each top-level contour (no parent)
                processed_polygons = []

                for i in range(len(contours_found)):
                    next_idx, prev_idx, first_child, parent = hierarchy[i]

                    # Only process top-level contours
                    if parent == -1:
                        # Skip very small contours
                        if cv2.contourArea(contours_found[i]) < 50:
                            continue

                        # Approximate the outer contour
                        perimeter = cv2.arcLength(contours_found[i], True)
                        approx_outer = cv2.approxPolyDP(
                            contours_found[i], epsilon * perimeter, True
                        )

                        # Find holes (direct children)
                        holes = []
                        child_idx = first_child
                        while child_idx != -1:
                            child_contour = contours_found[child_idx]
                            child_perimeter = cv2.arcLength(child_contour, True)
                            approx_hole = cv2.approxPolyDP(
                                child_contour, epsilon * child_perimeter, True
                            )
                            holes.append(approx_hole.squeeze().tolist())

                            # Move to next sibling
                            child_idx = hierarchy[child_idx][0]

                        # Create vertices
                        outer_vertices = approx_outer.squeeze().tolist()
                        if not isinstance(outer_vertices[0], list):
                            outer_vertices = [outer_vertices]

                        contour_vertices: List[ContourVertex] = [
                            {'x': int(x), 'y': int(y)} for x, y in outer_vertices
                        ]

                        if use_angle_filter:
                            contour_vertices = filter_sharp_vertices(contour_vertices, min_angle)

                        # Create polygon info
                        polygon_info = {
                            'vertices': contour_vertices,
                            'holes': []
                        }

                        if holes:
                            # Process hole vertices
                            for hole in holes:
                                if not isinstance(hole[0], list):
                                    hole = [hole]
                                hole_verts = [{'x': int(x), 'y': int(y)} for x, y in hole]
                                if use_angle_filter:
                                    hole_verts = filter_sharp_vertices(hole_verts, min_angle)
                                polygon_info['holes'].append(hole_verts)

                        processed_polygons.append(polygon_info)

                # If we have polygons, add each as a separate contour entry
                if processed_polygons:

                    # Add each polygon as a separate contour entry
                    for polygon_info in processed_polygons:
                        contour_info = {
                            'id': segment_id,
                            'vertices': polygon_info['vertices'],
                        }

                        # Add holes if present
                        if polygon_info['holes']:
                            contour_info['holes'] = polygon_info['holes']

                        contours.append(contour_info)
            else:
                # No holes, use simple processing
                # Process all significant contours
                if len(contours_found) > 1:
                    # Calculate areas for all contours
                    contour_areas = [(contour, cv2.contourArea(contour)) for contour in contours_found]
                    # Find the largest area
                    largest_area = max(area for _, area in contour_areas)
                    # Keep contours with area >= 10% of the largest
                    area_threshold = largest_area * 0.1
                    significant_contours = [contour for contour, area in contour_areas if area >= area_threshold]
                else:
                    significant_contours = contours_found

                # Process each significant contour
                for contour in significant_contours:
                    # Skip very small contours
                    if cv2.contourArea(contour) < 50:
                        continue

                    # Approximate the contour
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)

                    # Create vertices
                    vertices = approx.squeeze().tolist()
                    if not isinstance(vertices[0], list):
                        vertices = [vertices]

                    contour_vertices: List[ContourVertex] = [
                        {'x': int(x), 'y': int(y)} for x, y in vertices
                    ]

                    if use_angle_filter:
                        contour_vertices = filter_sharp_vertices(contour_vertices, min_angle)

                    contour_info = {
                        'id': segment_id,
                        'vertices': contour_vertices,
                    }
                    contours.append(contour_info)
        else:
            # Original implementation without Shapely
            contours_found, _ = cv2.findContours(
                segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # If multiple contours are found, keep the largest and those with area >= 20% of largest
            if contours_found:
                if len(contours_found) > 1:
                    # Calculate areas for all contours
                    contour_areas = [(contour, cv2.contourArea(contour)) for contour in contours_found]
                    # Find the largest area
                    largest_area = max(area for _, area in contour_areas)
                    # Keep contours with area >= 20% of the largest
                    area_threshold = largest_area * 0.2
                    contours_to_process = [contour for contour, area in contour_areas if area >= area_threshold]
                else:
                    contours_to_process = contours_found

                for contour in contours_to_process:
                    approx_contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

                    raw_vertices = approx_contour.squeeze().tolist()
                    if not isinstance(raw_vertices[0], list):
                        raw_vertices = [raw_vertices]

                    contour_vertices: List[ContourVertex] = [
                        {'x': int(x), 'y': int(y)} for x, y in raw_vertices
                    ]

                    if use_angle_filter:
                        contour_vertices = filter_sharp_vertices(contour_vertices, min_angle)

                    contours.append(
                        {
                            'id': segment_id,
                            'vertices': contour_vertices,
                        }
                    )

    # Apply wall filter to remove edge contours if enabled
    if wall_filter:
        contours = filter_edge_contours_by_wall_reference(
            contours, segment_array, segments_info, target_label_ids, edge_margin
        )

    # If align_to_lines is True, perform line detection and alignment
    if align_to_lines:
        if original_image is None:
            raise ValueError("original_image is required when align_to_lines is True")

        from .polygon_alignment import (
            detect_lines_pylsd,
            align_polygon_to_lines
        )

        # Detect lines in the original image
        detected_lines = detect_lines_pylsd(
            image=original_image,
            scale=scale,
            sigma_scale=sigma_scale,
            quant=quant,
            ang_th=ang_th,
            density_th=density_th,
            n_bins=n_bins,
            angle_tolerance=line_angle_tolerance,
        )

        # Align each contour to the detected lines
        aligned_contours = []
        for contour in contours:
            # Fill holes if requested
            processed_contour = contour
            if fill_holes:
                from .polygon_alignment import fill_polygon_holes
                processed_contour = fill_polygon_holes(contour)

            # Extract vertices
            vertices = processed_contour['vertices']

            # Align vertices to detected lines
            aligned_vertices = align_polygon_to_lines(
                vertices=vertices,
                lines=detected_lines,
                max_distance=max_snap_distance,
                angle_tolerance=snap_angle_tolerance,
                min_edge_length=min_edge_length,
                min_line_length_ratio=min_line_length_ratio,
                straighten_first=straighten_first,
                straightening_angle_tolerance=straightening_angle_tolerance,
                fill_holes=fill_holes,
                ensure_complete_perimeter_flag=ensure_complete_perimeter_flag,
                extend_to_outermost=extend_to_outermost,
                parallel_search_distance=parallel_search_distance,
            )

            # Create new contour with aligned vertices
            aligned_contour = processed_contour.copy()
            aligned_contour['vertices'] = aligned_vertices

            # If the contour has holes and we're not filling them, align them as well
            if not fill_holes and 'holes' in aligned_contour:
                aligned_holes = []
                for hole in aligned_contour['holes']:
                    aligned_hole = align_polygon_to_lines(
                        vertices=hole,
                        lines=detected_lines,
                        max_distance=max_snap_distance,
                        angle_tolerance=snap_angle_tolerance,
                        min_edge_length=min_edge_length,
                        min_line_length_ratio=min_line_length_ratio,
                        straighten_first=straighten_first,
                        straightening_angle_tolerance=straightening_angle_tolerance,
                        fill_holes=fill_holes,
                        ensure_complete_perimeter_flag=ensure_complete_perimeter_flag,
                        extend_to_outermost=extend_to_outermost,
                        parallel_search_distance=parallel_search_distance,
                    )
                    if len(aligned_hole) >= 3:  # Keep only valid holes
                        aligned_holes.append(aligned_hole)
                aligned_contour['holes'] = aligned_holes

            aligned_contours.append(aligned_contour)

        # Return based on return_lines flag
        if return_lines:
            return aligned_contours, detected_lines
        else:
            return aligned_contours

    return contours


def visualize_contours(
    image: Image.Image,
    contours_list: List[ContourInfo],
    vertex_size: int = 3,
    font_size: int = 20,
    show_vertex_count: bool = True,
    show_vertices: bool = True,
) -> Image.Image:
    """
    Visualize contours on an image with vertex counts at the center of each contour.

    Args:
        image (Image.Image): Input image.
        contours_list (List[ContourInfo]): List of contour information.
        vertex_size (int, optional): Vertex size. Defaults to 3.
        font_size (int, optional): Font size for vertex count text. Defaults to 20.
        show_vertex_count (bool, optional): Whether to show vertex count. Defaults to True.
        show_vertices (bool, optional): Whether to show vertices. Defaults to True.

    Returns:
        Image.Image: Visualized image.
    """
    vis_array = np.array(image)

    vis_array = vis_array.copy()

    for contour_info in contours_list:
        vertices = contour_info['vertices']
        num_vertices = len(vertices)

        contour_np = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)

        center_x = int(np.mean(contour_np[:, 0]))
        center_y = int(np.mean(contour_np[:, 1]))

        cv2.polylines(
            vis_array, [contour_np], isClosed=True, color=(0, 255, 0), thickness=2
        )
        if show_vertices:
            for pt in contour_np:
                cv2.circle(vis_array, tuple(pt), vertex_size, (0, 0, 255), -1)

    vis_image = Image.fromarray(vis_array)
    draw = ImageDraw.Draw(vis_image)

    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except OSError:
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', font_size)
        except OSError:
            font = ImageFont.load_default()

    if show_vertex_count:
        for contour_info in contours_list:
            vertices = contour_info['vertices']
            num_vertices = len(vertices)
            contour_np = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
            center_x = int(np.mean(contour_np[:, 0]))
            center_y = int(np.mean(contour_np[:, 1]))

            text = str(num_vertices)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            padding = 2
            draw.rectangle(
                [
                    center_x - text_width / 2 - padding,
                    center_y - text_height / 2 - padding,
                    center_x + text_width / 2 + padding,
                    center_y + text_height / 2 + padding,
                ],
                fill=(255, 255, 255),  # White background
            )

            draw.text(
                (center_x - text_width / 2, center_y - text_height / 2),
                text,
                fill=(255, 0, 0),  # Red color
                font=font,
            )

    return vis_image


class WallCenterlineExtractor:
    """壁マスクから中心線を抽出してCAD用線分に変換"""

    def __init__(self, segment_array: np.ndarray, wall_label_ids: List[int] = [2, 3]):
        """
        初期化

        Args:
            segment_array: セグメンテーション配列
            wall_label_ids: 壁として扱うラベルIDのリスト
        """
        # 壁マスクを作成
        self.wall_mask = np.zeros_like(segment_array, dtype=bool)
        for label_id in wall_label_ids:
            self.wall_mask |= (segment_array == label_id)

        self.H, self.W = self.wall_mask.shape
        self.skeleton = None
        self.graph = None
        self.paths = []
        self.simplified_paths = []
        self.final_segments = []

    def extract_centerlines(self,
                           rdp_epsilon: float = 3.0,
                           min_segment_length: float = 20.0,
                           hough_min_length: int = 80,
                           hough_max_gap: int = 30,
                           hough_threshold: int = 40,
                           enable_refinement: bool = True) -> List[Dict[str, Any]]:
        """
        中心線を抽出して線ポリゴン形式で返す

        Args:
            rdp_epsilon: RDP簡略化の閾値
            min_segment_length: 最小セグメント長
            hough_min_length: Hough変換の最小線長
            hough_max_gap: Hough変換の最大ギャップ
            hough_threshold: Hough変換の閾値
            enable_refinement: CAD精細化を有効にするか

        Returns:
            線ポリゴンのリスト（ContourInfo形式）
        """
        try:
            # スケルトン化
            self._skeletonize()

            # グラフ構築
            self._build_skeleton_graph()

            # パス抽出
            self._extract_paths()

            # RDP簡略化
            self._simplify_paths(rdp_epsilon)

            # Hough変換で長い線を検出
            hough_lines = self._detect_long_lines_hough(
                hough_min_length, hough_max_gap, hough_threshold
            )

            # セグメントをマージ
            self._merge_segments(hough_lines, min_segment_length)

            # CAD精細化（オプション）
            if enable_refinement:
                self._refine_segments()

            # 線ポリゴン形式に変換
            return self._convert_to_contour_format()

        except Exception as e:
            print(f"Error in centerline extraction: {e}")
            return []

    def _skeletonize(self):
        """スケルトン化処理"""
        try:
            from skimage.morphology import skeletonize, medial_axis

            # スケルトン化
            self.skeleton = skeletonize(self.wall_mask).astype(np.uint8) * 255

            # medial axisも計算（壁の幅情報用）
            self.medial, self.distance = medial_axis(self.wall_mask, return_distance=True)

        except ImportError:
            raise ImportError("scikit-image is required for skeletonization")

    def _build_skeleton_graph(self):
        """スケルトンからグラフを構築"""
        try:
            import sknw
            self.graph = sknw.build_sknw(self.skeleton, multi=True)
        except ImportError:
            raise ImportError("sknw is required for skeleton graph construction")

    def _extract_paths(self):
        """グラフからパスを抽出"""
        self.paths = []

        for s, e in self.graph.edges():
            for idx in self.graph[s][e]:
                path = self.graph[s][e][idx]["pts"]
                if len(path) > 1:
                    # [y, x] から [x, y] に変換
                    path_xy = np.array([(p[1], p[0]) for p in path])
                    self.paths.append(path_xy)

    def _simplify_paths(self, epsilon: float):
        """RDPアルゴリズムでパスを簡略化"""
        try:
            from rdp import rdp

            self.simplified_paths = []

            for path in self.paths:
                if len(path) > 2:
                    simplified = rdp(path, epsilon=epsilon)
                    self.simplified_paths.append(simplified)
                else:
                    self.simplified_paths.append(path)

        except ImportError:
            raise ImportError("rdp is required for path simplification")

    def _detect_long_lines_hough(self, min_length: int, max_gap: int, threshold: int):
        """Hough変換で長い線を検出"""
        if self.skeleton is None:
            return []

        # スケルトンをバイナリ画像に変換
        binary_image = (self.skeleton > 0).astype(np.uint8) * 255

        # Hough変換
        lines = cv2.HoughLinesP(
            binary_image,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )

        hough_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                from shapely.geometry import LineString
                line_segment = LineString([(x1, y1), (x2, y2)])
                hough_lines.append(line_segment)

        return hough_lines

    def _merge_segments(self, hough_lines, min_length: float):
        """セグメントをマージして最終セグメントを作成"""
        from shapely.geometry import LineString

        # RDPパスをLineStringに変換
        rdp_segments = []
        for path in self.simplified_paths:
            if len(path) > 1:
                for i in range(len(path) - 1):
                    line = LineString([path[i], path[i + 1]])
                    if line.length >= min_length:
                        rdp_segments.append(line)

        # Houghセグメントと結合
        all_segments = rdp_segments + hough_lines

        # 基本的な重複除去（近接する同じ角度のセグメントをマージ）
        self.final_segments = self._basic_merge_overlapping(all_segments, min_length)

    def _basic_merge_overlapping(self, segments, min_length: float):
        """基本的な重複セグメントのマージ"""
        if not segments:
            return []

        # 長さでフィルタリング
        valid_segments = [seg for seg in segments if seg.length >= min_length]

        # 簡単な距離ベースのマージ
        merged = []
        used = set()

        for i, seg1 in enumerate(valid_segments):
            if i in used:
                continue

            # 近接するセグメントを探す
            to_merge = [seg1]
            used.add(i)

            for j, seg2 in enumerate(valid_segments):
                if i == j or j in used:
                    continue

                # 距離と角度をチェック
                if seg1.distance(seg2) < 10.0:
                    # 角度の差を計算
                    angle1 = self._get_line_angle(seg1)
                    angle2 = self._get_line_angle(seg2)
                    angle_diff = abs(angle1 - angle2)
                    angle_diff = min(angle_diff, 180 - angle_diff)

                    if angle_diff < 15.0:  # 15度以内
                        to_merge.append(seg2)
                        used.add(j)

            # セグメントをマージ
            if len(to_merge) == 1:
                merged.append(to_merge[0])
            else:
                merged_segment = self._merge_line_segments(to_merge)
                if merged_segment and merged_segment.length >= min_length:
                    merged.append(merged_segment)

        return merged

    def _get_line_angle(self, line):
        """線分の角度を取得（度）"""
        coords = list(line.coords)
        start, end = coords[0], coords[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return np.degrees(np.arctan2(dy, dx)) % 180

    def _merge_line_segments(self, segments):
        """複数の線分を1つにマージ"""
        from shapely.geometry import LineString

        if len(segments) == 1:
            return segments[0]

        # 全ての点を収集
        all_points = []
        for seg in segments:
            coords = list(seg.coords)
            all_points.extend(coords)

        # 重複点を除去
        unique_points = []
        for point in all_points:
            is_duplicate = any(
                abs(point[0] - up[0]) < 0.1 and abs(point[1] - up[1]) < 0.1
                for up in unique_points
            )
            if not is_duplicate:
                unique_points.append(point)

        if len(unique_points) < 2:
            return None

        # 最遠点ペアを見つける
        max_distance = 0
        start_point, end_point = unique_points[0], unique_points[1]

        for i, p1 in enumerate(unique_points):
            for j, p2 in enumerate(unique_points):
                if i >= j:
                    continue
                distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
                if distance > max_distance:
                    max_distance = distance
                    start_point, end_point = p1, p2

        return LineString([start_point, end_point])

    def _refine_segments(self):
        """CAD精細化処理（簡易版）"""
        if not self.final_segments:
            return

        # マンハッタン配置（水平・垂直線の強制）
        refined_segments = []

        for seg in self.final_segments:
            coords = list(seg.coords)
            start, end = coords[0], coords[-1]

            dx = abs(end[0] - start[0])
            dy = abs(end[1] - start[1])

            # 水平・垂直判定
            angle_tolerance = 10.0
            angle = self._get_line_angle(seg)

            is_horizontal = (abs(angle) < angle_tolerance or
                           abs(angle - 180) < angle_tolerance)
            is_vertical = abs(angle - 90) < angle_tolerance

            if is_horizontal:
                # 水平線に強制
                avg_y = (start[1] + end[1]) / 2
                x1, x2 = sorted([start[0], end[0]])
                refined_seg = seg.__class__([(x1, avg_y), (x2, avg_y)])
            elif is_vertical:
                # 垂直線に強制
                avg_x = (start[0] + end[0]) / 2
                y1, y2 = sorted([start[1], end[1]])
                refined_seg = seg.__class__([(avg_x, y1), (avg_x, y2)])
            else:
                # 斜線はそのまま
                refined_seg = seg

            refined_segments.append(refined_seg)

        self.final_segments = refined_segments

    def _convert_to_contour_format(self) -> List[Dict[str, Any]]:
        """線ポリゴン形式（ContourInfo）に変換"""
        contours = []

        for i, segment in enumerate(self.final_segments):
            coords = list(segment.coords)
            if len(coords) >= 2:
                # LineStringを頂点のリストに変換
                vertices = [{'x': int(round(x)), 'y': int(round(y))} for x, y in coords]

                contour_info = {
                    'id': f"wall_centerline_{i}",  # 仮のID
                    'vertices': vertices,
                }
                contours.append(contour_info)

        return contours


