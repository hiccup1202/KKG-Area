#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from typing import List, Tuple

import cv2
import numpy as np


def _candidate_points(
    current: Tuple[int, int],
    horiz: List[int],
    vert: List[int],
    allow_diag: bool = True,
) -> dict:
    """Calculate possible next points, separating HV and diagonal candidates."""
    cx, cy = current
    candidates = {'hv': [], 'diag': []}
    # Horizontal/Vertical moves
    if cy in horiz:
        candidates['hv'] += [(vx, cy) for vx in vert if vx != cx]
    if cx in vert:
        candidates['hv'] += [(cx, hy) for hy in horiz if hy != cy]

    # Diagonal moves (>= 30 degrees, grid intersection to intersection)
    if allow_diag:
        sqrt3 = math.sqrt(3)
        for vx in vert:
            for hy in horiz:
                if (vx, hy) == current:
                    continue
                dx = abs(vx - cx)
                dy = abs(hy - cy)
                if dx == 0 or dy == 0:  # Already handled by H/V moves
                    continue
                if dy * sqrt3 >= dx and dx * sqrt3 >= dy:
                    candidates['diag'].append((vx, hy))

    return candidates


def calculate_grid_positions_only(
    image_size: Tuple[int, int],
    min_horizontal: int,
    max_horizontal: int,
    min_vertical: int,
    max_vertical: int,
) -> Tuple[List[int], List[int]]:
    """Calculates precise grid line positions without drawing them.

    Args:
        image_size: Size of the image (width, height)
        min_horizontal: Minimum number of horizontal grid lines
        max_horizontal: Maximum number of horizontal grid lines
        min_vertical: Minimum number of vertical grid lines
        max_vertical: Maximum number of vertical grid lines

    Returns:
        Tuple of (horizontal positions, vertical positions)
    """
    width, height = image_size
    num_horizontal = random.randint(min_horizontal, max_horizontal)
    num_vertical = random.randint(min_vertical, max_vertical)

    horizontal_spacing = height // (num_horizontal + 1) if num_horizontal > 0 else height
    vertical_spacing = width // (num_vertical + 1) if num_vertical > 0 else width

    horizontal_positions = []
    for i in range(1, num_horizontal + 1):
        y_precise = i * horizontal_spacing
        horizontal_positions.append(y_precise)

    vertical_positions = []
    for i in range(1, num_vertical + 1):
        x_precise = i * vertical_spacing
        vertical_positions.append(x_precise)

    return horizontal_positions, vertical_positions


def create_grid_lines(
    image_size: Tuple[int, int],
    min_horizontal: int,
    max_horizontal: int,
    min_vertical: int,
    max_vertical: int,
    line_thickness: int,
    line_color: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Create precise grid lines (通り芯). Draws dotted lines on the image
    and solid lines on the mask.

    Args:
        image_size: Size of the image (width, height)
        min_horizontal: Minimum number of horizontal grid lines
        max_horizontal: Maximum number of horizontal grid lines
        min_vertical: Minimum number of vertical grid lines
        max_vertical: Maximum number of vertical grid lines
        line_thickness: Thickness of the grid lines
        line_color: Color of the grid lines

    Returns:
        Tuple of (image with grid lines, mask of grid lines, horizontal positions, vertical positions)
    """
    width, height = image_size

    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    mask = np.zeros((height, width), dtype=np.uint8)

    horizontal_positions, vertical_positions = calculate_grid_positions_only(
        image_size,
        min_horizontal,
        max_horizontal,
        min_vertical,
        max_vertical
    )

    # Draw horizontal grid lines (precise)
    for y_precise in horizontal_positions:
        # Draw precise dotted line on image
        for x_start in range(0, width, 10):
            if x_start % 20 < 10:
                cv2.line(
                    image,
                    (x_start, y_precise),
                    (x_start + 8, y_precise),
                    line_color,
                    line_thickness,
                )

        # Draw precise solid line on mask
        cv2.line(mask, (0, y_precise), (width - 1, y_precise), 1, line_thickness)

    # Draw vertical grid lines (precise)
    for x_precise in vertical_positions:
        # Draw precise dotted line on image
        for y_start in range(0, height, 10):
            if y_start % 20 < 10:
                cv2.line(
                    image,
                    (x_precise, y_start),
                    (x_precise, y_start + 8),
                    line_color,
                    line_thickness,
                )

        # Draw precise solid line on mask
        cv2.line(mask, (x_precise, 0), (x_precise, height - 1), 1, line_thickness)

    return image, mask, horizontal_positions, vertical_positions


def _offset_foundation_segments(
    segments: List[np.ndarray],
    max_offset: int,
) -> List[np.ndarray]:
    """Apply a uniform perpendicular offset to each continuous straight line.

    Horizontal lines: all segments sharing the same y are shifted by the same dy.
    Vertical   lines: all segments sharing the same x are shifted by the same dx.
    Diagonal   lines: each segment shifted by one random (dx,dy).
    """
    if max_offset == 0:
        return segments

    # Maps to store chosen offsets so the same straight line shares one offset
    horiz_off: dict = {}
    vert_off: dict = {}
    new_segments: List[np.ndarray] = []

    for seg in segments:
        p1, p2 = seg.astype(float)
        # Detect orientation
        if abs(p1[1] - p2[1]) < 1e-3:  # Horizontal
            key = round(p1[1])  # use y as key
            dy = horiz_off.setdefault(key, random.uniform(-max_offset, max_offset))
            p1[1] += dy
            p2[1] += dy
        elif abs(p1[0] - p2[0]) < 1e-3:  # Vertical
            key = round(p1[0])
            dx = vert_off.setdefault(key, random.uniform(-max_offset, max_offset))
            p1[0] += dx
            p2[0] += dx
        else:  # Diagonal – treat individually
            dx = random.uniform(-max_offset, max_offset)
            dy = random.uniform(-max_offset, max_offset)
            p1 += np.array([dx, dy])
            p2 += np.array([dx, dy])
        new_segments.append(np.array([p1, p2]))

    return new_segments


def generate_foundation_path(
    horizontal_positions: List[int],
    vertical_positions: List[int],
    image_size: Tuple[int, int],
    min_segments: int,
    max_segments: int,
    foundation_jitter: int = 8,
    diag_prob: float = 0.1,
) -> Tuple[List[np.ndarray], np.ndarray, List[Tuple[int, int]]]:
    """
    Generate a random foundation path, biasing towards HV moves, then applies jitter.

    Args:
        horizontal_positions: List of y-coordinates for horizontal grid lines
        vertical_positions: List of x-coordinates for vertical grid lines
        image_size: Size of the image (width, height)
        min_segments: Minimum number of segments in the foundation path (supplied by caller)
        max_segments: Maximum number of segments in the foundation path (supplied by caller)
        foundation_jitter: Maximum pixels to jitter foundation endpoints (0 for no jitter)
        diag_prob: Probability (0.0 to 1.0) of choosing diagonal when HV is possible.

    Returns:
        Tuple of (list of jittered foundation path segments, foundation mask (unused), list of intersection points)
    """
    width, height = image_size
    foundation_mask_dummy = np.zeros((height, width), dtype=np.uint8)
    foundation_segments = []
    intersection_points = []

    # --- Starting point selection (remains the same) ---
    start_on_horizontal = random.choice([True, False])
    if start_on_horizontal:
        start_y = random.choice(horizontal_positions)
        start_x = random.choice([0] + vertical_positions + [width - 1])
    else:
        start_x = random.choice(vertical_positions)
        start_y = random.choice([0] + horizontal_positions + [height - 1])
    current_point = (start_x, start_y)
    # --- End Starting point selection ---

    num_segments = random.randint(min_segments, max_segments)
    visited_points = set([current_point])

    for _ in range(num_segments):
        candidates = _candidate_points(
            current_point,
            horizontal_positions,
            vertical_positions,
            allow_diag=True,
        )

        # Filter candidates: remove visited and out-of-bounds points
        hv_candidates = [
            p
            for p in candidates['hv']
            if p not in visited_points and 0 <= p[0] < width and 0 <= p[1] < height
        ]
        diag_candidates = [
            p
            for p in candidates['diag']
            if p not in visited_points and 0 <= p[0] < width and 0 <= p[1] < height
        ]

        chosen_list = None

        # Biased selection logic
        if hv_candidates and diag_candidates:
            if random.random() < diag_prob:
                chosen_list = diag_candidates  # Choose diagonal with diag_prob
            else:
                chosen_list = hv_candidates  # Choose HV otherwise
        elif hv_candidates:
            chosen_list = hv_candidates  # Only HV possible
        elif diag_candidates:
            chosen_list = diag_candidates  # Only Diagonal possible
        else:
            break  # No valid moves left

        if not chosen_list:
            break  # Should not happen if break above works, but safety check

        next_point = random.choice(chosen_list)

        foundation_segments.append(
            np.array([current_point, next_point], dtype=np.int32)
        )

        if len(foundation_segments) > 1:
            intersection_points.append(current_point)

        visited_points.add(next_point)
        current_point = next_point

    # --- Offset application per straight line ---
    jittered_foundation_segments = _offset_foundation_segments(
        foundation_segments, foundation_jitter
    )

    return jittered_foundation_segments, foundation_mask_dummy, intersection_points


def create_foundation_with_width(
    foundation_segments: List[np.ndarray],
    image_size: Tuple[int, int],
    foundation_width: int = 20,
    line_color: Tuple[int, int, int] = (0, 0, 0),
    line_thickness: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create foundation with constant width based on foundation path segments,
    extending segments at the ends to ensure proper corner connections in the mask.

    Args:
        foundation_segments: List of foundation path segments
        image_size: Size of the image (width, height)
        foundation_width: Width of the foundation
        line_color: Color of the foundation lines
        line_thickness: Thickness of the foundation outline

    Returns:
        Tuple of (image with foundation, mask of foundation)
    """
    width, height = image_size

    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    mask = np.zeros((height, width), dtype=np.uint8)

    half_width = foundation_width / 2.0  # Use float division

    for segment in foundation_segments:
        p1, p2 = segment

        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])

        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            continue
        udx = dx / length
        udy = dy / length

        perpendicular_x = -udy
        perpendicular_y = udx

        # Precise extended endpoints (using float)
        p1_ext = (float(p1[0]) - udx * half_width, float(p1[1]) - udy * half_width)
        p2_ext = (float(p2[0]) + udx * half_width, float(p2[1]) + udy * half_width)

        # Calculate the four corners of the precise extended rectangle
        points = np.array(
            [
                [
                    p1_ext[0] + perpendicular_x * half_width,
                    p1_ext[1] + perpendicular_y * half_width,
                ],
                [
                    p2_ext[0] + perpendicular_x * half_width,
                    p2_ext[1] + perpendicular_y * half_width,
                ],
                [
                    p2_ext[0] - perpendicular_x * half_width,
                    p2_ext[1] - perpendicular_y * half_width,
                ],
                [
                    p1_ext[0] - perpendicular_x * half_width,
                    p1_ext[1] - perpendicular_y * half_width,
                ],
            ],
            dtype=np.float32,
        )  # Keep as float initially

        # Fill mask using the precise extended rectangle (convert to int for cv2 function)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, line_color, line_thickness)

    return image, mask
