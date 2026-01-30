"""
Wall line extraction module using skeletonization and CAD refinement techniques.
Converts wall segments to clean line representations instead of polygons.
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import List

import sknw
from rdp import rdp
from shapely.geometry import LineString, Point
from skimage.morphology import skeletonize, medial_axis
import matplotlib.pyplot as plt

# Optional import for DXF export
try:
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False


class WallSkeletonToCAD:
    """Process wall masks to extract CAD-ready line segments"""

    def __init__(self, mask_bool, params=None):
        self.mask_bool = mask_bool
        self.H, self.W = mask_bool.shape
        self.skeleton = None
        self.graph = None
        self.paths = []
        self.simplified_paths = []
        self.manhattan_segments = []
        self.merged_segments = []
        self.final_segments = []
        self.junction_points = []  # Store junction coordinates
        self.endpoints = []  # Initialize endpoints
        self.junctions = []  # Initialize junctions

        # Parameters
        self.params = params or {}
        self.rdp_epsilon = self.params.get("rdp_epsilon", 3.0)
        self.is_debug = False  # Debug flag for detailed logging

    def process(
        self,
        rdp_epsilon=None,
        consolidation_angle_threshold=9.0,
        min_segment_length=20,
        max_consolidation_iterations=3,
        hough_min_length=80,
        hough_max_gap=30,
        hough_threshold=40,
        enable_refinement=True,
        extend_threshold=40,
        snap_tolerance=10,
        manhattan_angle_tolerance=10,
        coordinate_snap_tolerance=10,
        extend_diagonal_lines=False,
        is_debug=False,
    ):
        """
        Complete processing pipeline

        Args:
            rdp_epsilon: RDP simplification tolerance (default: 2.0 for new, 0.5 for legacy)
            consolidation_angle_threshold: Angle threshold for consolidating segments (degrees)
            min_segment_length: Minimum length for final segments (pixels)
            max_consolidation_iterations: Maximum number of consolidation iterations
            hough_min_length: Minimum line length for Hough detection (pixels)
            hough_max_gap: Maximum gap for Hough line merging (pixels)
            hough_threshold: Threshold for Hough line detection
            enable_refinement: Whether to apply CAD refinement steps (cleanup, extend, snap)
            extend_threshold: Maximum distance for line extension (pixels)
            snap_tolerance: Tolerance for endpoint snapping (pixels)
            manhattan_angle_tolerance: Angle tolerance for Manhattan alignment (degrees)
            coordinate_snap_tolerance: Distance tolerance for coordinate snapping (pixels)
            extend_diagonal_lines: Whether to extend diagonal lines (default: False)
            is_debug: Enable detailed debug logging (default: False)

        Returns:
            list: Final line segments
        """

        print("=== Wall Skeleton to CAD: Improved RDP + Hough Transform + Overlap Merging + Refinement ===")

        # Set debug flag
        self.is_debug = is_debug

        # Use provided epsilon or default
        if rdp_epsilon is not None:
            self.rdp_epsilon = rdp_epsilon

        # Update min_seg_len parameter for consistency
        self.min_seg_len = min_segment_length

        # Stage 1: Basic RDP Processing (No Consolidation)
        print("\n--- Stage 1: Basic RDP Processing ---")

        # Skeletonization
        self.skeletonize()

        # Build graph from skeleton
        self.build_skeleton_graph()

        # Extract paths
        self.extract_paths()

        # Simplify paths with RDP
        self.simplify_paths()

        # Convert to LineString segments
        self.final_segments = []
        for path in self.simplified_paths:
            if len(path) > 1:
                for i in range(len(path) - 1):
                    line = LineString([path[i], path[i + 1]])
                    if line.length >= min_segment_length:
                        self.final_segments.append(line)

        rdp_segments = self.final_segments.copy()
        print(f"Stage 1 result: {len(rdp_segments)} basic segments")

        # Stage 2: Hough Transform Long Line Detection
        print("\n--- Stage 2: Hough Transform Long Line Detection ---")

        hough_lines = self.detect_long_lines_hough(
            min_length=hough_min_length,
            max_line_gap=hough_max_gap,
            line_threshold=hough_threshold,
        )

        print(f"Stage 2 result: {len(hough_lines)} long lines detected")

        # Stage 3: Combined Fusion + Overlap Merging
        print("\n--- Stage 3: Combined Fusion + Overlap Merging ---")

        # Combine all segments first
        all_segments = list(hough_lines) + rdp_segments
        print(f"Combined {len(hough_lines)} Hough + {len(rdp_segments)} RDP = {len(all_segments)} total segments")

        # Do overlap merging on combined segments (this will automatically handle fusion)
        # Multi-pass merging to catch overlaps that become visible after first pass
        current_segments = all_segments
        max_passes = 3
        for pass_num in range(max_passes):
            print(f"\n--- Merging Pass {pass_num + 1}/{max_passes} ---")

            merged_segments = self._merge_overlapping_segments(
                current_segments,
                distance_tolerance=8.0,
                angle_tolerance=14.0,
                overlap_threshold=5.0,
            )

            # Check if any merging happened
            if len(merged_segments) == len(current_segments):
                print(f"No more merging possible after pass {pass_num + 1}")
                break

            print(f"Pass {pass_num + 1}: {len(current_segments)} → {len(merged_segments)} segments")
            current_segments = merged_segments

        self.final_segments = current_segments
        print(f"After merging: {len(self.final_segments)} segments")

        # Stage 4: CAD Refinement (if enabled)
        if enable_refinement:
            print("\n--- Stage 4: CAD Refinement ---")

            # Step 4a: Clean up short segments
            print("Step 4a: Cleaning up short segments...")
            cleaned_segments = self.filter_segments_by_length(self.final_segments)
            self.final_segments = cleaned_segments
            print(f"After cleanup: {len(cleaned_segments)} segments")

            # Step 4a.5: Manhattan alignment and coordinate snapping
            print("Step 4a.5: Manhattan alignment and coordinate snapping...")
            aligned_segments = self._align_to_manhattan_and_snap_coordinates(
                cleaned_segments, manhattan_angle_tolerance, coordinate_snap_tolerance
            )
            print(f"After Manhattan alignment: {len(aligned_segments)} segments")

            # Step 4b: Extend lines to create proper intersections
            print("Step 4b: Extending lines to intersections...")
            extended_segments = self._extend_lines_to_intersect(
                aligned_segments, extend_threshold, extend_diagonal_lines
            )
            print(f"After extension: {len(extended_segments)} segments")

            # Step 4c: Snap nearby endpoints together
            print("Step 4c: Snapping nearby endpoints...")
            self.final_segments = self._snap_endpoints(extended_segments, snap_tolerance)
            print(f"After snapping: {len(self.final_segments)} segments")

        print(f"Final result: {len(self.final_segments)} segments")

        return self.final_segments

    def skeletonize(self):
        """Apply skeletonization using scikit-image"""
        print("\nStep 1: Skeletonization...")

        # Use scikit-image skeletonize
        self.skeleton = skeletonize(self.mask_bool).astype(np.uint8) * 255

        # Get medial axis for wall width information
        self.medial, self.distance = medial_axis(self.mask_bool, return_distance=True)

        print(f"Skeleton pixels: {np.sum(self.skeleton > 0)}")

    def build_skeleton_graph(self):
        """Build graph from skeleton using sknw"""
        print("\nStep 2: Building skeleton graph...")

        # Use sknw to build graph
        self.graph = sknw.build_sknw(self.skeleton, multi=True)

        print(f"Nodes: {self.graph.number_of_nodes()}")
        print(f"Edges: {self.graph.number_of_edges()}")

        # Analyze node degrees
        degrees = dict(self.graph.degree())
        self.endpoints = [n for n, d in degrees.items() if d == 1]
        self.junctions = [n for n, d in degrees.items() if d > 2]

        # Store junction coordinates in [x, y] format
        self.junction_points = []
        for node in self.junctions:
            y, x = self.graph.nodes[node]["o"]
            self.junction_points.append([x, y])

        print(f"Endpoints: {len(self.endpoints)}, Junctions: {len(self.junctions)}")

    def extract_paths(self):
        """Extract paths from skeleton graph"""
        print("\nStep 3: Extracting paths...")

        self.paths = []

        # Extract paths from graph edges
        for s, e in self.graph.edges():
            for idx in self.graph[s][e]:
                path = self.graph[s][e][idx]["pts"]
                if len(path) > 1:
                    # Convert from [y, x] to [x, y] if needed
                    # sknw stores points as [row, col] = [y, x]
                    path_xy = np.array([(p[1], p[0]) for p in path])
                    self.paths.append(path_xy)

        print(f"Extracted paths: {len(self.paths)}")

    def simplify_paths(self):
        """Simplify paths using RDP algorithm"""
        print("\nStep 4: Simplifying with RDP...")
        print(f"RDP epsilon: {self.rdp_epsilon}")

        self.simplified_paths = []
        total_points_before = sum(len(p) for p in self.paths)

        for path in self.paths:
            if len(path) > 2:
                # Apply RDP simplification
                simplified = rdp(path, epsilon=self.rdp_epsilon)
                self.simplified_paths.append(simplified)
            else:
                self.simplified_paths.append(path)

        total_points_after = sum(len(p) for p in self.simplified_paths)
        print(f"Points reduced: {total_points_before} → {total_points_after}")

    def filter_segments_by_length(self, segments, min_length=None):
        """Filter segments by minimum length

        Args:
            segments: List of segments to filter
            min_length: Minimum length threshold (uses self.min_seg_len if None)

        Returns:
            list: Filtered segments
        """
        if min_length is None:
            min_length = self.min_seg_len

        filtered = []
        for seg in segments:
            if isinstance(seg, LineString):
                if seg.length > min_length:
                    filtered.append(seg)
            else:
                # Convert to LineString if needed
                if len(seg) >= 2:
                    line = LineString(seg)
                    if line.length > min_length:
                        filtered.append(line)

        return filtered

    def detect_long_lines_hough(self, min_length=100, max_line_gap=20, line_threshold=50):
        """
        Hough Transform Line Detection Algorithm
        Detects lines using OpenCV's Hough Line Transform

        Args:
            min_length: Minimum line length to consider (pixels)
            max_line_gap: Maximum gap between line segments to connect them
            line_threshold: Threshold for line detection in Hough space

        Returns:
            list: List of detected lines as LineString objects
        """
        print("\n=== Hough Transform Line Detection ===")
        print(f"Min length: {min_length}px, Max gap: {max_line_gap}px, Threshold: {line_threshold}")

        if not hasattr(self, 'skeleton') or self.skeleton is None:
            print("No skeleton available. Run skeletonization first.")
            return []

        # Convert skeleton to binary image for Hough transform
        binary_image = (self.skeleton > 0).astype(np.uint8) * 255

        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(
            binary_image,
            rho=1,                     # Distance resolution in pixels
            theta=np.pi / 180,         # Angle resolution in radians
            threshold=line_threshold,  # Minimum votes to consider a line
            minLineLength=min_length,  # Minimum line length
            maxLineGap=max_line_gap,   # Maximum gap between line segments
        )

        if lines is None:
            print("No lines detected by Hough transform")
            return []

        print(f"Hough transform detected {len(lines)} line segments")

        # Convert to LineString objects - no intermediate merging
        detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segment = LineString([(x1, y1), (x2, y2)])
            detected_lines.append(line_segment)

        print(f"Raw Hough lines: {len(detected_lines)} (merging will be handled in final stage)")
        return detected_lines

    def _align_to_manhattan_and_snap_coordinates(self, segments, angle_tolerance=10, coordinate_tolerance=5):
        """
        Align near-horizontal/vertical lines to strict Manhattan geometry and snap coordinates

        Args:
            segments: List of line segments to align
            angle_tolerance: Maximum angle deviation from horizontal/vertical (degrees)
            coordinate_tolerance: Maximum distance for coordinate snapping (pixels)

        Returns:
            list: Aligned and coordinate-snapped line segments
        """
        if not segments:
            return []

        print(f"  Manhattan alignment: angle_tolerance={angle_tolerance}°, coord_tolerance={coordinate_tolerance}px")

        aligned_segments = []
        horizontal_lines = []
        vertical_lines = []
        diagonal_lines = []

        # Step 1: Classify lines and force Manhattan alignment
        for seg in segments:
            coords = list(seg.coords)
            if len(coords) < 2:
                continue

            start, end = coords[0], coords[-1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]

            # Calculate angle
            angle = np.degrees(np.arctan2(dy, dx)) % 180

            # Check if line is near horizontal or vertical
            is_near_horizontal = abs(angle) <= angle_tolerance or abs(angle - 180) <= angle_tolerance
            is_near_vertical = abs(angle - 90) <= angle_tolerance

            if is_near_horizontal:
                # Force to horizontal by averaging y-coordinates
                avg_y = (start[1] + end[1]) / 2
                # Keep original x-coordinates but ensure proper ordering
                x1, x2 = sorted([start[0], end[0]])
                horizontal_line = {
                    'segment': LineString([(x1, avg_y), (x2, avg_y)]),
                    'y_coord': avg_y,
                    'x_range': (x1, x2),
                    'length': abs(x2 - x1),
                    'original': seg
                }
                horizontal_lines.append(horizontal_line)

            elif is_near_vertical:
                # Force to vertical by averaging x-coordinates
                avg_x = (start[0] + end[0]) / 2
                # Keep original y-coordinates but ensure proper ordering
                y1, y2 = sorted([start[1], end[1]])
                vertical_line = {
                    'segment': LineString([(avg_x, y1), (avg_x, y2)]),
                    'x_coord': avg_x,
                    'y_range': (y1, y2),
                    'length': abs(y2 - y1),
                    'original': seg
                }
                vertical_lines.append(vertical_line)

            else:
                # Keep diagonal lines as-is
                diagonal_lines.append(seg)

        print(f"  Classified: {len(horizontal_lines)} horizontal, "
              f"{len(vertical_lines)} vertical, {len(diagonal_lines)} diagonal")

        # Step 2: Snap horizontal lines by y-coordinate
        if horizontal_lines:
            snapped_horizontal = self._snap_coordinate_groups(
                horizontal_lines, 'y_coord', coordinate_tolerance, is_horizontal=True
            )
            aligned_segments.extend(snapped_horizontal)
            print(f"  Snapped horizontal lines: {len(horizontal_lines)} → {len(snapped_horizontal)}")

        # Step 3: Snap vertical lines by x-coordinate
        if vertical_lines:
            snapped_vertical = self._snap_coordinate_groups(
                vertical_lines, 'x_coord', coordinate_tolerance, is_horizontal=False
            )
            aligned_segments.extend(snapped_vertical)
            print(f"  Snapped vertical lines: {len(vertical_lines)} → {len(snapped_vertical)}")

        # Step 4: Add diagonal lines unchanged
        aligned_segments.extend(diagonal_lines)

        return aligned_segments

    def _snap_coordinate_groups(self, lines, coord_key, tolerance, is_horizontal=True):
        """
        Group lines by coordinate and snap nearby coordinates using weighted average

        Args:
            lines: List of line dictionaries with coordinate info
            coord_key: Key for the coordinate to group by ('x_coord' or 'y_coord')
            tolerance: Maximum distance for coordinate snapping
            is_horizontal: True for horizontal lines (group by y), False for vertical (group by x)

        Returns:
            list: Line segments with snapped coordinates
        """
        if not lines:
            return []

        # Sort lines by the coordinate we're grouping by
        sorted_lines = sorted(lines, key=lambda x: x[coord_key])

        # Group nearby coordinates
        coordinate_groups = []
        current_group = [sorted_lines[0]]

        for i in range(1, len(sorted_lines)):
            current_coord = sorted_lines[i][coord_key]
            prev_coord = current_group[-1][coord_key]

            if abs(current_coord - prev_coord) <= tolerance:
                # Add to current group
                current_group.append(sorted_lines[i])
            else:
                # Start new group
                coordinate_groups.append(current_group)
                current_group = [sorted_lines[i]]

        # Add the last group
        if current_group:
            coordinate_groups.append(current_group)

        # Process each group
        snapped_segments = []

        for group in coordinate_groups:
            if len(group) == 1:
                # Single line, keep as-is
                snapped_segments.append(group[0]['segment'])
            else:
                # Multiple lines, calculate weighted average coordinate
                total_length = sum(line['length'] for line in group)
                if total_length > 0:
                    weighted_coord = sum(line[coord_key] * line['length'] for line in group) / total_length
                else:
                    # Fallback to simple average if all lengths are zero
                    weighted_coord = sum(line[coord_key] for line in group) / len(group)

                # Create snapped segments
                for line in group:
                    if is_horizontal:
                        # Update y-coordinate for horizontal lines
                        x1, x2 = line['x_range']
                        snapped_segment = LineString([(x1, weighted_coord), (x2, weighted_coord)])
                    else:
                        # Update x-coordinate for vertical lines
                        y1, y2 = line['y_range']
                        snapped_segment = LineString([(weighted_coord, y1), (weighted_coord, y2)])

                    snapped_segments.append(snapped_segment)

        return snapped_segments

    def _extend_lines_to_intersect(self, segments, extend_threshold=50, extend_diagonal_lines=False):
        """
        Line extension - works with any line orientations, not just Manhattan

        Args:
            segments: List of line segments to extend
            extend_threshold: Maximum extension distance (pixels)
            extend_diagonal_lines: Whether to extend diagonal lines (default: False)

        Returns:
            list: Extended line segments
        """
        extended_segments = []
        segment_lines = []

        # Convert all segments to LineString objects
        for seg in segments:
            if isinstance(seg, LineString):
                segment_lines.append(seg)
            else:
                segment_lines.append(LineString(seg))

        # Process each segment
        for i, line1 in enumerate(segment_lines):
            coords1 = list(line1.coords)
            x1, y1 = coords1[0]
            x2, y2 = coords1[-1]

            # Calculate line1's angle and direction
            dx1 = x2 - x1
            dy1 = y2 - y1
            angle1 = np.degrees(np.arctan2(dy1, dx1)) % 180

            # Determine if line1 is primarily horizontal, vertical, or diagonal
            is_horizontal1 = abs(angle1) < 10 or abs(angle1 - 180) < 10
            is_vertical1 = abs(angle1 - 90) < 10
            is_diagonal1 = not (is_horizontal1 or is_vertical1)

            # Skip diagonal line extension if disabled
            if is_diagonal1 and not extend_diagonal_lines:
                extended_segments.append(line1)
                continue

            # Find potential intersections with other lines
            extended_start = False
            extended_end = False
            new_start = (x1, y1)
            new_end = (x2, y2)

            for j, line2 in enumerate(segment_lines):
                if i == j:
                    continue

                coords2 = list(line2.coords)
                x3, y3 = coords2[0]
                x4, y4 = coords2[-1]

                # Calculate line2's angle
                dx2 = x4 - x3
                dy2 = y4 - y3
                angle2 = np.degrees(np.arctan2(dy2, dx2)) % 180

                is_horizontal2 = abs(angle2) < 10 or abs(angle2 - 180) < 10
                is_vertical2 = abs(angle2 - 90) < 10
                is_diagonal2 = not (is_horizontal2 or is_vertical2)

                # Skip intersection with diagonal lines if diagonal extension is disabled
                if is_diagonal2 and not extend_diagonal_lines:
                    continue

                # Calculate potential intersection point
                intersection_point = None

                if is_horizontal1 and is_vertical2:
                    # line1 is horizontal, line2 is vertical
                    intersection_point = (x3, y1)
                elif is_vertical1 and is_horizontal2:
                    # line1 is vertical, line2 is horizontal
                    intersection_point = (x1, y3)
                elif extend_diagonal_lines and (is_diagonal1 or is_diagonal2):
                    # Both or one is diagonal and diagonal extension is enabled
                    intersection_point = self._calculate_line_intersection(
                        (x1, y1), (x2, y2), (x3, y3), (x4, y4)
                    )

                if intersection_point is None:
                    continue

                ix, iy = intersection_point

                # Check start point extension
                if not extended_start:
                    dist_start = Point(x1, y1).distance(Point(ix, iy))
                    if dist_start < extend_threshold:
                        # Check if extension makes sense geometrically
                        if self._should_extend_endpoint((x1, y1), (x2, y2), (ix, iy), is_start=True):
                            new_start = (ix, iy)
                            extended_start = True

                # Check end point extension
                if not extended_end:
                    dist_end = Point(x2, y2).distance(Point(ix, iy))
                    if dist_end < extend_threshold:
                        # Check if extension makes sense geometrically
                        if self._should_extend_endpoint((x1, y1), (x2, y2), (ix, iy), is_start=False):
                            new_end = (ix, iy)
                            extended_end = True

            # Create the extended segment
            extended_segments.append(LineString([new_start, new_end]))

        return extended_segments

    def _calculate_line_intersection(self, p1, p2, p3, p4):
        """
        Calculate intersection point between two line segments

        Args:
            p1, p2: Points defining first line segment
            p3, p4: Points defining second line segment

        Returns:
            tuple: Intersection point (x, y) or None if no intersection
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # Calculate the denominator
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-10:  # Lines are parallel
            return None

        # Calculate intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Calculate intersection coordinates
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)

        return (ix, iy)

    def _should_extend_endpoint(self, start, end, intersection, is_start=True):
        """
        Check if extending an endpoint to an intersection makes geometric sense

        Args:
            start: Start point of line
            end: End point of line
            intersection: Potential intersection point
            is_start: Whether we're extending the start point (True) or end point (False)

        Returns:
            bool: Whether extension should be performed
        """
        if is_start:
            # For start point, intersection should be in the direction opposite to the line
            line_dir = (end[0] - start[0], end[1] - start[1])
            ext_dir = (intersection[0] - start[0], intersection[1] - start[1])
        else:
            # For end point, intersection should be in the direction of the line
            line_dir = (end[0] - start[0], end[1] - start[1])
            ext_dir = (intersection[0] - end[0], intersection[1] - end[1])

        # Calculate dot product to check direction alignment
        dot_product = line_dir[0] * ext_dir[0] + line_dir[1] * ext_dir[1]

        # For start point: dot product should be negative (opposite direction)
        # For end point: dot product should be positive (same direction)
        if is_start:
            return dot_product < 0
        else:
            return dot_product > 0

    def _snap_endpoints(self, segments, snap_tolerance=10):
        """
        Endpoint snapping - works with any line orientations

        Args:
            segments: List of line segments
            snap_tolerance: Distance tolerance for snapping

        Returns:
            list: Segments with snapped endpoints
        """
        refined = []

        # Build a list of all endpoints
        all_endpoints = []
        for seg in segments:
            coords = list(seg.coords)
            all_endpoints.append(coords[0])
            all_endpoints.append(coords[-1])

        # Find clusters of nearby points
        endpoint_clusters = []
        used = set()

        for i, p1 in enumerate(all_endpoints):
            if i in used:
                continue

            cluster = [p1]
            used.add(i)

            for j, p2 in enumerate(all_endpoints):
                if j in used or j == i:
                    continue

                if Point(p1).distance(Point(p2)) < snap_tolerance:
                    cluster.append(p2)
                    used.add(j)

            if cluster:
                # Calculate cluster center
                x_avg = sum(p[0] for p in cluster) / len(cluster)
                y_avg = sum(p[1] for p in cluster) / len(cluster)
                endpoint_clusters.append((x_avg, y_avg))

        # Rebuild segments with snapped endpoints
        for seg in segments:
            coords = list(seg.coords)
            new_start = coords[0]
            new_end = coords[-1]

            # Find closest cluster for start point
            for cluster_point in endpoint_clusters:
                if Point(coords[0]).distance(Point(cluster_point)) < snap_tolerance:
                    new_start = cluster_point
                    break

            # Find closest cluster for end point
            for cluster_point in endpoint_clusters:
                if Point(coords[-1]).distance(Point(cluster_point)) < snap_tolerance:
                    new_end = cluster_point
                    break

            refined.append(LineString([new_start, new_end]))

        return refined

    def _merge_overlapping_segments(
        self, segments, distance_tolerance=10.0, 
        angle_tolerance=10.0, overlap_threshold=10.0
    ):
        """
        Optimized merge of overlapping and collinear line segments

        Args:
            segments: List of LineString objects
            distance_tolerance: Maximum perpendicular distance to consider lines as parallel (pixels)
            angle_tolerance: Maximum angle difference to consider lines as parallel (degrees)
            overlap_threshold: Minimum overlap length in pixels to trigger merging

        Returns:
            list: Merged line segments
        """
        if not segments:
            return []

        print("\nMerging overlapping segments...")
        print(f"Input segments: {len(segments)}")
        print(f"Distance tolerance: {distance_tolerance}px, Angle tolerance: {angle_tolerance}°")

        # Pre-filter valid segments and extract data in one pass
        valid_segments = []
        for seg in segments:
            coords = list(seg.coords)
            if len(coords) < 2:
                continue

            start, end = coords[0], coords[-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = (dx**2 + dy**2)**0.5

            if length < 1e-6:
                continue

            # Normalize angle to 0-180 and quantize for efficient grouping
            angle = np.degrees(np.arctan2(dy, dx)) % 180
            angle_bucket = round(angle / angle_tolerance) * angle_tolerance

            valid_segments.append({
                'segment': seg,
                'start': start, 'end': end, 'length': length,
                'angle': angle, 'angle_bucket': angle_bucket,
                'dx': dx, 'dy': dy,
            })

        if not valid_segments:
            return []

        # Efficient angle grouping using quantized buckets
        angle_groups = defaultdict(list)
        for seg_data in valid_segments:
            angle_groups[seg_data['angle_bucket']].append(seg_data)

        # Merge angle groups that are within angle_tolerance of each other (handles 0/180 boundary)
        self._merge_nearby_angle_groups(angle_groups, angle_tolerance)

        print(f"Grouped into {len(angle_groups)} angle groups")

        merged_segments = []
        total_merged = 0

        # Process each angle group
        for group_segments in angle_groups.values():
            if len(group_segments) <= 1:
                merged_segments.extend([seg['segment'] for seg in group_segments])
                continue

            if self.is_debug:
                print(f"\n🔍 DEBUG Angle GROUP: Processing {len(group_segments)} segments "
                      f"at angle {group_segments[0]['angle']:.1f}°")
                for i, seg in enumerate(group_segments):
                    start, end = seg['start'], seg['end']
                    length = seg['length']
                    print(f"  Seg {i}: ({start[0]:.1f},{start[1]:.1f}) → "
                          f"({end[0]:.1f},{end[1]:.1f}), len={length:.1f}")

                # Calculate line segment distances for all the pairs of segments
                for i in range(len(group_segments)):
                    for j in range(i + 1, len(group_segments)):
                        seg1 = group_segments[i]
                        seg2 = group_segments[j]
                        distance = seg1['segment'].distance(seg2['segment'])
                        if distance < 10.0:
                            print(f"  Distance between seg {i} and seg {j}: {distance:.1f}")

            # Optimized merging within group
            group_merged = self._merge_segments_in_group_optimized(
                group_segments, distance_tolerance, overlap_threshold, debug=self.is_debug
            )
            merged_segments.extend(group_merged)

            segments_before = len(group_segments)
            segments_after = len(group_merged)
            if segments_before > segments_after:
                total_merged += (segments_before - segments_after)
                print(f"  Angle {group_segments[0]['angle']:.1f}°: {segments_before} → {segments_after} segments")

            if self.is_debug:
                print(f"🔍 DEBUG Angle GROUP: Result = {segments_after} segments after merging")
                for i, seg in enumerate(group_merged):
                    coords = list(seg.coords)
                    start, end = coords[0], coords[-1]
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    print(f"  Result {i}: ({start[0]:.1f},{start[1]:.1f}) → "
                          f"({end[0]:.1f},{end[1]:.1f}), len={length:.1f}")

        print(f"Merged {total_merged} overlapping segments")
        print(f"Final segments: {len(merged_segments)}")

        return merged_segments

    def _merge_nearby_angle_groups(self, angle_groups, angle_tolerance):
        """Merge angle groups that are within angle_tolerance of each other"""
        if len(angle_groups) <= 1:
            return

        # Get all bucket angles and sort them
        buckets = sorted(angle_groups.keys())

        # Find groups to merge by checking if their representative angles are within tolerance
        merged_groups = {}  # bucket -> target_bucket mapping

        for i in range(len(buckets)):
            bucket_i = buckets[i]
            if bucket_i in merged_groups:
                continue  # Already merged

            # Get representative angle for this bucket (use average of actual angles)
            angles_i = [seg['angle'] for seg in angle_groups[bucket_i]]
            repr_angle_i = sum(angles_i) / len(angles_i)

            # Check if this bucket should be merged with any later buckets
            for j in range(i + 1, len(buckets)):
                bucket_j = buckets[j]
                if bucket_j in merged_groups:
                    continue  # Already merged

                # Get representative angle for bucket j
                angles_j = [seg['angle'] for seg in angle_groups[bucket_j]]
                repr_angle_j = sum(angles_j) / len(angles_j)

                # Calculate angle difference (handle 0/180 boundary)
                angle_diff = abs(repr_angle_i - repr_angle_j)
                angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wraparound

                # If within tolerance, mark for merging
                if angle_diff <= angle_tolerance:
                    merged_groups[bucket_j] = bucket_i

        # Apply the merges
        for source_bucket, target_bucket in merged_groups.items():
            angle_groups[target_bucket].extend(angle_groups[source_bucket])
            del angle_groups[source_bucket]

    def _merge_segments_in_group_optimized(self, group_segments, distance_tolerance, overlap_threshold, debug=False):
        """Simple double-loop merging - if segments are touching (distance ≤ distance_tolerance), merge them"""
        if len(group_segments) <= 1:
            return [seg['segment'] for seg in group_segments]

        # Step 1: Remove segments that are completely contained within other segments
        filtered_segments = self._remove_contained_segments(group_segments, debug)

        if len(filtered_segments) <= 1:
            return [seg['segment'] for seg in filtered_segments]

        # Step 2: Build adjacency list for connected segments (distance ≤ distance_tolerance)
        n = len(filtered_segments)
        connected = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                distance = filtered_segments[i]['segment'].distance(filtered_segments[j]['segment'])
                if distance <= distance_tolerance:
                    connected[i].append(j)
                    connected[j].append(i)
                    if debug:
                        print(f"    Connected: seg {i} ↔ seg {j} (distance: {distance:.1f})")

        # Step 3: Find connected components using DFS
        visited = [False] * n
        merged_segments = []

        for i in range(n):
            if visited[i]:
                continue

            # DFS to find all connected segments
            component = []
            stack = [i]

            while stack:
                current = stack.pop()
                if visited[current]:
                    continue

                visited[current] = True
                component.append(current)

                # Add all unvisited neighbors
                for neighbor in connected[current]:
                    if not visited[neighbor]:
                        stack.append(neighbor)

            # Create merged segment from this component
            if len(component) == 1:
                # Single segment - keep as is
                merged_segments.append(filtered_segments[component[0]]['segment'])
                if debug:
                    print(f"    Component {len(component)} segment: kept as single")
            else:
                # Multiple segments - validate mask before merging
                segments_data = [filtered_segments[idx] for idx in component]
                is_horizontal = abs(filtered_segments[0]['dx']) > abs(filtered_segments[0]['dy'])

                if debug:
                    print(f"    Component {len(component)} segments: merging...")
                    for idx in component:
                        seg = filtered_segments[idx]
                        print(f"      Seg {idx}: ({seg['start'][0]:.1f},{seg['start'][1]:.1f}) → "
                              f"({seg['end'][0]:.1f},{seg['end'][1]:.1f})")

                # Create merged segment and trim it to mask boundaries
                merged_segment = self._create_merged_segment(segments_data, is_horizontal)
                trimmed_segments = self._trim_segment_to_mask(merged_segment, debug)

                if trimmed_segments:
                    merged_segments.extend(trimmed_segments)
                    if debug:
                        print(f"      → Merged into {len(trimmed_segments)} trimmed segments")
                else:
                    # Fallback: keep original segments if trimming fails completely
                    for idx in component:
                        merged_segments.append(filtered_segments[idx]['segment'])
                    if debug:
                        print(f"      → Trimming failed, kept {len(component)} original segments")

        return merged_segments

    def _remove_contained_segments(self, group_segments, debug=False):
        """Remove segments that are completely contained within other segments"""
        if len(group_segments) <= 1:
            return group_segments

        # Mark segments for removal
        to_remove = set()

        group_segments = sorted(group_segments, key=lambda x: x['length'])

        for i in range(len(group_segments)):
            if i in to_remove:
                continue

            seg1 = group_segments[i]

            for j in range(len(group_segments)):
                if i == j or j in to_remove:
                    continue

                seg2 = group_segments[j]

                # Check if seg1 is completely contained within seg2
                if self._is_segment_contained_in_segment(seg1, seg2):
                    to_remove.add(i)
                    if debug:
                        print(f"  Removing contained segment {i}: len={seg1['length']:.1f} "
                              f"(contained in segment {j}: len={seg2['length']:.1f})")
                    break

        # Return segments that are not marked for removal
        filtered_segments = [seg for i, seg in enumerate(group_segments) if i not in to_remove]

        if debug and to_remove:
            print(f"  Removed {len(to_remove)} contained segments, {len(filtered_segments)} remaining")

        return filtered_segments

    def _is_segment_contained_in_segment(self, seg1, seg2):
        """Check if seg1 is completely contained within seg2"""
        # Since segments are in same angle group, they have similar angles

        # First check if seg2 is longer than seg1 (only remove shorter segments)
        if seg2['length'] < seg1['length']:
            return False

        # Check if segments are close enough
        distance = seg1['segment'].distance(seg2['segment'])
        if distance > 10.0:  # If not close, can't be contained
            return False

        # Get coordinates
        start1, end1 = seg1['start'], seg1['end']
        start2, end2 = seg2['start'], seg2['end']

        # Simple range check as you suggested:
        # L2.start_x <= L1.start_x and L1.end_x <= L2.end_x (and same for y)

        seg1_x_min, seg1_x_max = min(start1[0], end1[0]), max(start1[0], end1[0])
        seg1_y_min, seg1_y_max = min(start1[1], end1[1]), max(start1[1], end1[1])

        seg2_x_min, seg2_x_max = min(start2[0], end2[0]), max(start2[0], end2[0])
        seg2_y_min, seg2_y_max = min(start2[1], end2[1]), max(start2[1], end2[1])

        # Check if seg1 is contained within seg2's ranges
        x_contained = seg2_x_min <= seg1_x_min and seg1_x_max <= seg2_x_max
        y_contained = seg2_y_min <= seg1_y_min and seg1_y_max <= seg2_y_max

        return x_contained and y_contained

    def _trim_segment_to_mask(self, segment, debug=False):
        """
        Trim a segment to only the parts that are within the mask

        Args:
            segment: LineString segment to trim
            debug: Whether to print debug information

        Returns:
            List of LineString segments that are within the mask
        """
        if self.mask_bool is None:
            if debug:
                print("        No mask available, returning original segment")
            return [segment]

        from shapely.geometry import LineString

        coords = list(segment.coords)
        if len(coords) < 2:
            return []

        # Sample points along the line to check mask coverage
        start_point = coords[0]
        end_point = coords[-1]

        # Create dense sampling of points along the line
        num_samples = max(10, int(segment.length / 2))  # Sample every ~2 pixels
        sample_points = []

        for i in range(num_samples + 1):
            t = i / num_samples
            x = start_point[0] + t * (end_point[0] - start_point[0])
            y = start_point[1] + t * (end_point[1] - start_point[1])
            sample_points.append((x, y, t))

        # Check which points are within the mask
        valid_points = []
        for x, y, t in sample_points:
            if (0 <= int(x) < self.mask_bool.shape[1]
                and 0 <= int(y) < self.mask_bool.shape[0]
                    and self.mask_bool[int(y), int(x)]):
                valid_points.append((x, y, t))

        if not valid_points:
            if debug:
                print("        No points within mask, discarding segment")
            return []

        # Group consecutive valid points into segments
        segments = []
        current_segment_points = []

        for i, (x, y, t) in enumerate(valid_points):
            if not current_segment_points:
                # Start new segment
                current_segment_points = [(x, y)]
            else:
                # Check if this point is consecutive with the previous
                prev_t = valid_points[i - 1][2] if i > 0 else 0
                if abs(t - prev_t) <= 2.0 / num_samples:  # Allow small gaps
                    current_segment_points.append((x, y))
                else:
                    # Gap detected, finish current segment and start new one
                    if len(current_segment_points) >= 2:
                        segments.append(LineString(current_segment_points))
                    current_segment_points = [(x, y)]

        # Add final segment
        if len(current_segment_points) >= 2:
            segments.append(LineString(current_segment_points))

        # Filter out very short segments
        min_length = 5.0  # Minimum 5 pixels
        valid_segments = [seg for seg in segments if seg.length >= min_length]

        if debug:
            print(f"        Original segment: {start_point} → {end_point} (len={segment.length:.1f})")
            print(f"        Trimmed to {len(valid_segments)} segments")
            for i, seg in enumerate(valid_segments):
                seg_coords = list(seg.coords)
                print(f"          Segment {i + 1}: {seg_coords[0]} → {seg_coords[-1]} (len={seg.length:.1f})")

        return valid_segments

    def _create_merged_segment(self, segments_data, is_horizontal):
        """Create a single merged segment from multiple segment data"""
        if len(segments_data) == 1:
            return segments_data[0]['segment']

        # Collect all points from all segments
        all_points = []
        for seg_data in segments_data:
            all_points.extend([seg_data['start'], seg_data['end']])

        # Remove duplicate points
        unique_points = []
        for point in all_points:
            if not any(abs(point[0] - up[0]) < 0.1 and abs(point[1] - up[1]) < 0.1 for up in unique_points):
                unique_points.append(point)

        if len(unique_points) < 2:
            # Fallback to original method if we don't have enough unique points
            return segments_data[0]['segment']

        # Calculate the angle of the original segments to preserve it
        first_seg = segments_data[0]
        dx = first_seg['end'][0] - first_seg['start'][0]
        dy = first_seg['end'][1] - first_seg['start'][1]
        original_angle = np.degrees(np.arctan2(dy, dx)) % 180

        # Check if this is a truly horizontal (0°/180°) or vertical (90°) line
        is_truly_horizontal = abs(original_angle) < 5 or abs(original_angle - 180) < 5
        is_truly_vertical = abs(original_angle - 90) < 5

        if is_truly_horizontal:
            # Pure horizontal line - use x-sorting and y-averaging
            unique_points.sort(key=lambda p: p[0])
            y_coords = [p[1] for p in unique_points]
            avg_y = sum(y_coords) / len(y_coords)
            start_point = (unique_points[0][0], avg_y)
            end_point = (unique_points[-1][0], avg_y)
        elif is_truly_vertical:
            # Pure vertical line - use y-sorting and x-averaging
            unique_points.sort(key=lambda p: p[1])
            x_coords = [p[0] for p in unique_points]
            avg_x = sum(x_coords) / len(x_coords)
            start_point = (avg_x, unique_points[0][1])
            end_point = (avg_x, unique_points[-1][1])
        else:
            # Diagonal line - use linear regression to preserve angle
            x_coords = [p[0] for p in unique_points]
            y_coords = [p[1] for p in unique_points]

            # Use linear regression to find best-fit line
            from sklearn.linear_model import LinearRegression

            X = np.array(x_coords).reshape(-1, 1)
            y = np.array(y_coords)

            reg = LinearRegression().fit(X, y)

            # Find the extreme points along the fitted line
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Determine which dimension has larger range to use as primary axis
            x_range = max_x - min_x
            y_range = max_y - min_y

            if x_range >= y_range:
                # Use x as primary axis
                start_point = (min_x, reg.predict([[min_x]])[0])
                end_point = (max_x, reg.predict([[max_x]])[0])
            else:
                # Use y as primary axis - need inverse regression
                X_inv = np.array(y_coords).reshape(-1, 1)
                y_inv = np.array(x_coords)
                reg_inv = LinearRegression().fit(X_inv, y_inv)

                start_point = (reg_inv.predict([[min_y]])[0], min_y)
                end_point = (reg_inv.predict([[max_y]])[0], max_y)

        return LineString([start_point, end_point])

    def get_statistics(self):
        """Get processing statistics"""
        stats = {
            "skeleton_pixels": (
                np.sum(self.skeleton > 0) if self.skeleton is not None else 0
            ),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
            "endpoints": len(self.endpoints),
            "junctions": len(self.junctions),
            "extracted_paths": len(self.paths),
            "rdp_paths": len(self.simplified_paths),
            "rdp_total_points": sum(len(p) for p in self.simplified_paths),
            "final_segments": len(self.final_segments),
            "total_length": sum(line.length for line in self.final_segments),
        }

        return stats

    def visualize_result(self):
        """
        Visualize the result
        """
        if not hasattr(self, 'final_segments'):
            print("No results to visualize. Run process first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Panel 1: Original skeleton
        axes[0].imshow(self.skeleton, cmap='gray', alpha=0.3)
        axes[0].set_title("Original Skeleton")
        axes[0].set_xlim(0, self.W)
        axes[0].set_ylim(self.H, 0)

        # Panel 2: RDP consolidated segments (if available)
        axes[1].imshow(self.skeleton, cmap='gray', alpha=0.2)
        if hasattr(self, 'rdp_segments'):
            for segment in getattr(self, 'rdp_segments', []):
                coords = list(segment.coords)
                if len(coords) >= 2:
                    seg_array = np.array(coords)
                    axes[1].plot(seg_array[:, 0], seg_array[:, 1], 'b-', linewidth=2, alpha=0.7)
        axes[1].set_title("RDP Consolidated Segments")
        axes[1].set_xlim(0, self.W)
        axes[1].set_ylim(self.H, 0)

        # Panel 3: Hough long lines (if available)
        axes[2].imshow(self.skeleton, cmap='gray', alpha=0.2)
        if hasattr(self, 'hough_lines'):
            colors = plt.cm.tab10(np.linspace(0, 1, len(getattr(self, 'hough_lines', []))))
            for i, line in enumerate(getattr(self, 'hough_lines', [])):
                coords = list(line.coords)
                if len(coords) >= 2:
                    line_array = np.array(coords)
                    axes[2].plot(line_array[:, 0], line_array[:, 1],
                                 color=colors[i], linewidth=3, alpha=0.8)
        axes[2].set_title("Hough Long Lines")
        axes[2].set_xlim(0, self.W)
        axes[2].set_ylim(self.H, 0)

        # Panel 4: Final result
        axes[3].imshow(self.skeleton, cmap='gray', alpha=0.2)
        for segment in self.final_segments:
            coords = list(segment.coords)
            if len(coords) >= 2:
                seg_array = np.array(coords)
                axes[3].plot(seg_array[:, 0], seg_array[:, 1], 'r-', linewidth=2, alpha=0.8)

        # Highlight junctions
        for junction in self.junction_points:
            axes[3].plot(junction[0], junction[1], 'go', markersize=6, alpha=0.8)

        axes[3].set_title(f"Result ({len(self.final_segments)} segments)")
        axes[3].set_xlim(0, self.W)
        axes[3].set_ylim(self.H, 0)

        plt.tight_layout()

        return fig


def extract_wall_lines_from_segment(
    segment_mask: np.ndarray,
    rdp_epsilon: float = 3.0,
    min_segment_length: float = 20,
    enable_refinement: bool = True,
    extend_threshold: float = 40,
    snap_tolerance: float = 10,
    manhattan_angle_tolerance: float = 10,
    coordinate_snap_tolerance: float = 10,
    **kwargs
) -> List[LineString]:
    """
    Extract wall lines from a binary segment mask using skeletonization.

    Args:
        segment_mask: Binary mask for the wall segment
        rdp_epsilon: RDP simplification tolerance
        min_segment_length: Minimum length for final segments (pixels)
        enable_refinement: Whether to apply CAD refinement steps
        extend_threshold: Maximum distance for line extension (pixels)
        snap_tolerance: Tolerance for endpoint snapping (pixels)
        manhattan_angle_tolerance: Angle tolerance for Manhattan alignment (degrees)
        coordinate_snap_tolerance: Distance tolerance for coordinate snapping (pixels)
        **kwargs: Additional parameters passed to WallSkeletonToCAD.process()

    Returns:
        List of LineString objects representing wall lines
    """
    # Ensure mask is boolean
    mask_bool = segment_mask.astype(bool)

    # Create processor
    processor = WallSkeletonToCAD(mask_bool)

    # Process and return lines
    lines = processor.process(
        rdp_epsilon=rdp_epsilon,
        min_segment_length=min_segment_length,
        enable_refinement=enable_refinement,
        extend_threshold=extend_threshold,
        snap_tolerance=snap_tolerance,
        manhattan_angle_tolerance=manhattan_angle_tolerance,
        coordinate_snap_tolerance=coordinate_snap_tolerance,
        **kwargs
    )

    return lines
