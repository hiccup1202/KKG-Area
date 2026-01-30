#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import List, Tuple

import cv2
import numpy as np


def _compute_interior_mask(foundation_mask: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8) where pixels that are *inside* closed foundation
    outlines are set to 1.  The algorithm flood-fills the background from the image
    border after inverting the foundation mask.  Any remaining pixels that are not
    reached by the flood-fill are by definition enclosed by the foundation walls.

    This will work reliably only when the foundation path forms a closed loop, which
    is assumed when generating the synthetic dataset.
    """
    # `foundation_mask` is 0/1, where 1 indicates the foundation wall
    h, w = foundation_mask.shape
    foundation_uint8 = (foundation_mask > 0).astype(np.uint8) * 255  # 0/255 image

    # Pixels that are not part of the wall are set to 255 so that we can flood-fill
    # from the border.  The walls (foundations) remain 0 and will stop the flood.
    inv = cv2.bitwise_not(foundation_uint8)
    flood = inv.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    # Flood-fill from the top-left corner (0,0) which is guaranteed to be outside
    cv2.floodFill(flood, mask_ff, (0, 0), 0)

    # After flood-fill, background/outside pixels are 0, enclosed pixels remain 255.
    interior_mask = (flood == 255).astype(np.uint8)
    return interior_mask


def _draw_symbol(img: np.ndarray, center: Tuple[int, int], symbol_type: str,
                 radius: int, color: Tuple[int, int, int], thickness: int) -> None:
    """Draw a circle (\"circle\") or cross (\"cross\") symbol at *center* on *img*."""
    x, y = center
    if symbol_type == 'circle':
        cv2.circle(img, (x, y), radius, color, thickness, cv2.LINE_AA)
    elif symbol_type == 'double_circle':
        cv2.circle(img, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius - 3, color, thickness, cv2.LINE_AA)
    elif symbol_type == 'filled_circle':
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
    elif symbol_type == 'circle_cross':
        cv2.circle(img, (x, y), radius, color, thickness, cv2.LINE_AA)
        size = int(radius * 0.8)
        cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)
    elif symbol_type == 'circle_plus':
        cv2.circle(img, (x, y), radius, color, thickness, cv2.LINE_AA)
        size = int(radius * 0.8)
        cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    elif symbol_type == 'square':
        cv2.rectangle(img, (x - radius, y - radius), (x + radius, y + radius), color, thickness, cv2.LINE_AA)
    elif symbol_type == 'double_square':
        cv2.rectangle(img, (x - radius, y - radius), (x + radius, y + radius), color, thickness, cv2.LINE_AA)
        cv2.rectangle(
            img, (x - radius + 3, y - radius + 3), (x + radius - 3, y + radius - 3), color, thickness, cv2.LINE_AA)
    elif symbol_type == 'filled_square':
        cv2.rectangle(img, (x - radius, y - radius), (x + radius, y + radius), color, -1, cv2.LINE_AA)
    elif symbol_type == 'square_cross':
        cv2.rectangle(img, (x - radius, y - radius), (x + radius, y + radius), color, thickness, cv2.LINE_AA)
        size = int(radius * 0.8)
        cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)
    elif symbol_type == 'square_plus':
        cv2.rectangle(img, (x - radius, y - radius), (x + radius, y + radius), color, thickness, cv2.LINE_AA)
        size = int(radius * 0.8)
        cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    elif symbol_type == 'cross':
        size = radius
        cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)
    elif symbol_type == 'plus':
        size = radius
        cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    else:
        raise ValueError(f'Unsupported symbol type: {symbol_type}')


def _place_symbols(
    base_image: np.ndarray,
    horizontal_positions: List[int],
    vertical_positions: List[int],
    foundation_line_mask: np.ndarray,
    interior_mask: np.ndarray,
    symbol_color: Tuple[int, int, int],
    symbol_thickness: int,
    max_floor_posts: int,
    num_wall_symbols: int,
    num_random_symbols: int,
    min_symbol_radius: int,
    max_symbol_radius: int,
    foundation_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    foundation_width: int = 0,
    floor_post_placement_prob: float = 1.0,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Draw symbols (◯/×) on the *base_image* and return the resulting image together
    with a list of bounding boxes (x_min, y_min, x_max, y_max) for the symbols that
    qualify as floor posts.

    A symbol qualifies as a floor post when it satisfies *both*:
      1. It lies on an intersection of the grid lines (通り芯).
      2. The centre is located inside a region fully enclosed by the foundation
         lines.

    Other symbols are still drawn but are *not* included in the returned list of
    bounding boxes, thereby acting as negative examples.
    """
    symbol_types = [
        'circle', 'double_circle', 'filled_circle', 'circle_cross', 'circle_plus',
        'square', 'double_square', 'filled_square', 'square_cross', 'square_plus',
        'cross', 'plus'
    ]
    wall_symbol_types = ['square', 'filled_square', 'filled_circle']

    symbol_type = random.choice(symbol_types)

    h, w, _ = base_image.shape
    img = base_image.copy()

    # ------------------------------------------------------------------
    # 1. Symbols to be considered as floor posts (grid intersections *inside* foundation)
    # ------------------------------------------------------------------
    intersection_coords = [(vx, hy) for vx in vertical_positions for hy in horizontal_positions]
    interior_intersections = [(x, y) for (x, y) in intersection_coords if interior_mask[y, x] > 0]

    random.shuffle(interior_intersections)
    selected_floor_posts = interior_intersections[:max_floor_posts]

    floor_post_bboxes: List[Tuple[int, int, int, int]] = []

    symbol_radius = random.randint(min_symbol_radius, max_symbol_radius)
    # The actual number of floor posts will be <= max_floor_posts AND also affected by placement probability
    placed_floor_post_count = 0
    for (x, y) in selected_floor_posts:
        if placed_floor_post_count >= max_floor_posts:
            break  # Stop if we have already placed the maximum allowed due to probability

        if random.random() < floor_post_placement_prob:
            _draw_symbol(img, (x, y), symbol_type, radius=symbol_radius, color=symbol_color, thickness=symbol_thickness)
            floor_post_bboxes.append((x - symbol_radius, y - symbol_radius,
                                      x + symbol_radius, y + symbol_radius))
            placed_floor_post_count += 1

    # ------------------------------------------------------------------
    # 2. Symbols at the unique endpoints of foundation segments (negative samples)
    # ------------------------------------------------------------------
    wall_symbol_centers = []
    if foundation_segments is not None:
        # Collect all endpoints
        endpoints = []
        for seg in foundation_segments:
            p1, p2 = tuple(seg[0]), tuple(seg[1])  # Ensure points are hashable tuples
            endpoints.append(p1)
            endpoints.append(p2)
        # Deduplicate endpoints (use set for uniqueness)
        unique_endpoints = list(set((int(round(x)), int(round(y))) for (x, y) in endpoints))

        wall_symbol = random.choice(wall_symbol_types)
        # symbol_radius for wall symbols is randomized once, then potentially clamped per symbol
        base_wall_symbol_radius = random.randint(min_symbol_radius, max_symbol_radius)

        for (mx, my) in unique_endpoints:
            # Calculate max_radius to ensure symbol fits *within* the foundation width
            # A symbol of radius R spans 2R+1 pixels. Foundation width is F pixels.
            # So, 2R+1 <= F  => R <= (F-1)/2
            symbol_radius_to_draw = base_wall_symbol_radius
            if foundation_width > 0:
                max_allowed_radius = (foundation_width - 1) // 2
                if max_allowed_radius < 0:  # Should not happen if foundation_width is reasonable
                    max_allowed_radius = 0
                if symbol_radius_to_draw > max_allowed_radius:
                    symbol_radius_to_draw = max_allowed_radius
            else:  # handle case where foundation_width might be 0 or not provided
                symbol_radius_to_draw = min(symbol_radius_to_draw, 3)  # a small default

            # Ensure radius is not negative if foundation_width is very small (e.g., 1)
            if symbol_radius_to_draw < 0:
                symbol_radius_to_draw = 0

            _draw_symbol(img, (mx, my), wall_symbol, radius=symbol_radius_to_draw,
                         color=symbol_color, thickness=symbol_thickness)
            wall_symbol_centers.append((mx, my))

    # ------------------------------------------------------------------
    # 3. Random symbols anywhere, avoiding foundation and other key symbol locations
    # ------------------------------------------------------------------
    occupied_points = list(selected_floor_posts)  # Centers of floor posts
    occupied_points.extend(wall_symbol_centers)  # Centers of wall symbols

    random_symbol_types_list = [s for s in symbol_types if s != symbol_type]  # Exclude floor post type
    if not random_symbol_types_list:  # Ensure list is not empty if symbol_type was the only one
        random_symbol_types_list = symbol_types

    max_retries_per_symbol = 100
    # Use max_symbol_radius for bounds checking to ensure any symbol fits
    max_possible_radius = max_symbol_radius

    for _ in range(num_random_symbols):
        retries = 0
        while retries < max_retries_per_symbol:
            # Generate candidate position, ensuring max_possible_radius doesn't go out of bounds
            if w <= 2 * max_possible_radius or h <= 2 * max_possible_radius:
                break  # Image too small to place symbols with this logic
            x_rand = random.randint(max_possible_radius, w - 1 - max_possible_radius)
            y_rand = random.randint(max_possible_radius, h - 1 - max_possible_radius)

            # Check 1: Not on foundation mask
            if foundation_line_mask[y_rand, x_rand] > 0:
                retries += 1
                continue

            # Check 2: Not too close to any existing key symbols
            # Proximity threshold: sum of max radii, squared for efficiency
            proximity_threshold_sq = (max_possible_radius * 2) ** 2
            is_too_close = False
            for ox, oy in occupied_points:
                dist_sq = (x_rand - ox) ** 2 + (y_rand - oy) ** 2
                if dist_sq < proximity_threshold_sq:
                    is_too_close = True
                    break
            if is_too_close:
                retries += 1
                continue

            # If all checks pass, place the symbol
            actual_rand_symbol_type = random.choice(random_symbol_types_list)
            actual_rand_symbol_radius = random.randint(min_symbol_radius, max_symbol_radius)
            _draw_symbol(img, (x_rand, y_rand), actual_rand_symbol_type,
                         radius=actual_rand_symbol_radius, color=symbol_color,
                         thickness=symbol_thickness)
            break  # Placed symbol, exit retry loop

    return img, floor_post_bboxes
