#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from typing import Tuple

import cv2
import numpy as np


def add_noise(image: np.ndarray,
              foundation_mask: np.ndarray,
              grid_mask: np.ndarray,  # grid_mask is needed for dimension lines
              hatch_direction: str,
              max_dim_lines: int,
              max_rand_lines: int,
              noise_color: Tuple[int, int, int],
              hatch_color: Tuple[int, int, int],
              hatch_step: int,
              min_dim_line_offset: int,
              max_dim_line_offset: int,
              min_rand_line_length: int,
              max_rand_line_length: int,
              min_rand_circle_radius: int,
              max_rand_circle_radius: int,
              max_hatch: int = 4,  # Retained with default as it's currently unused
              ) -> np.ndarray:
    """Adds visual noise (dimension lines, hatching, random lines/circles) to the image.

    Args:
        image: The base image to add noise to
        foundation_mask: Binary mask of the foundation
        grid_mask: Binary mask of the grid lines
        hatch_direction: The direction for hatching ('diag_down_right' or 'diag_down_left')
        max_dim_lines: Maximum number of dimension lines to add
        max_rand_lines: Maximum number of random lines to add
        noise_color: Color for general noise elements like random lines/dimension lines.
        hatch_color: Color for hatching patterns.
        hatch_step: Step size for hatching lines.
        min_dim_line_offset: Min offset for dimension lines from grid lines.
        max_dim_line_offset: Max offset for dimension lines from grid lines.
        min_rand_line_length: Min length for random noise lines.
        max_rand_line_length: Max length for random noise lines.
        min_rand_circle_radius: Min radius for random noise circles.
        max_rand_circle_radius: Max radius for random noise circles.
        max_hatch: Maximum number of hatching patterns to add (currently unused)

    Returns:
        Image with added visual noise
    """
    h, w, _ = image.shape
    img = image.copy()
    # noise_color is now a parameter
    # hatch_color is now a parameter

    # 3-1 Dimension Lines (parallel to grid lines outside foundation)
    grid_y_coords = np.where(grid_mask.sum(axis=1) > 0)[0]
    if len(grid_y_coords) > 0:
        for _ in range(random.randint(1, max_dim_lines)):
            y = random.choice(grid_y_coords)  # Choose a grid line y
            # Offset above or below
            off = random.randint(min_dim_line_offset, max_dim_line_offset) * random.choice([-1, 1])
            if 0 <= y + off < h:
                cv2.line(img, (0, y + off), (w - 1, y + off), noise_color, 1, cv2.LINE_AA)
                # Arrows/Ticks (simple vertical lines)
                cv2.line(img, (10, y + off - 3), (10, y + off + 3), noise_color, 1)
                cv2.line(img, (w - 11, y + off - 3), (w - 11, y + off + 3), noise_color, 1)

    # Similar logic for vertical dimension lines
    grid_x_coords = np.where(grid_mask.sum(axis=0) > 0)[0]
    if len(grid_x_coords) > 0:
        for _ in range(random.randint(1, max_dim_lines)):
            x = random.choice(grid_x_coords)
            off = random.randint(min_dim_line_offset, max_dim_line_offset) * random.choice([-1, 1])
            if 0 <= x + off < w:
                cv2.line(img, (x + off, 0), (x + off, h - 1), noise_color, 1, cv2.LINE_AA)
                cv2.line(img, (x + off - 3, 10), (x + off + 3, 10), noise_color, 1)
                cv2.line(img, (x + off - 3, h - 11), (x + off + 3, h - 11), noise_color, 1)

    # 3-2 Hatching (inside some foundation contours)
    contours, _ = cv2.findContours(foundation_mask, cv2.RETR_LIST,  # Use RETR_LIST to get inner contours too
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area to avoid hatching tiny fragments
    min_hatch_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_hatch_area]

    if len(valid_contours) > 0:
        for cnt in valid_contours:
            if random.random() > 0.5:  # 50% chance to hatch a contour
                continue
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 1, -1)

            step = hatch_step  # Use parameterized hatch_step
            # Use the passed hatch_direction
            if hatch_direction == 'diag_down_right':
                # Original: from (x+k, y) to (x+k+h_rect, y+h_rect)
                for k_offset in range(-h_rect, w_rect, step):
                    if random.random() > 0.5:  # 50% chance per line
                        continue
                    p1 = (x_rect + k_offset, y_rect)
                    p2 = (x_rect + k_offset + h_rect, y_rect + h_rect)
                    temp_line_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.line(temp_line_mask, p1, p2, 1, 1)
                    img[(contour_mask > 0) & (temp_line_mask > 0)] = hatch_color
            elif hatch_direction == 'diag_down_left':  # Ensure this condition explicitly matches the other option
                # New: from (x+k, y+h_rect) to (x+k+h_rect, y)
                # or from (x+w_rect-k, y) to (x+w_rect-(k+h_rect), y+h_rect) (adjusting k range)
                for k_offset in range(-h_rect, w_rect, step):
                    if random.random() > 0.5:  # 50% chance per line
                        continue
                    p1 = (x_rect + k_offset, y_rect + h_rect)  # Start from bottom-left-ish
                    p2 = (x_rect + k_offset + h_rect, y_rect)  # Go to top-right-ish
                    temp_line_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.line(temp_line_mask, p1, p2, 1, 1)
                    img[(contour_mask > 0) & (temp_line_mask > 0)] = hatch_color
            else:
                raise ValueError(f"Unknown hatch_direction: {hatch_direction}. "
                                 "Expected 'diag_down_right' or 'diag_down_left'.")

    # 3-3 Random Short Lines / Circles
    for _ in range(random.randint(max_rand_lines // 2, max_rand_lines)):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(min_rand_line_length, max_rand_line_length)
        x2 = int(x1 + length * math.cos(angle))
        y2 = int(y1 + length * math.sin(angle))
        # Ensure endpoints are within bounds
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.line(img, (x1, y1), (x2, y2), noise_color, 1, cv2.LINE_AA)
        if random.random() < 0.3:  # Occasionally add a circle at the start point
            cv2.circle(img, (x1, y1), random.randint(min_rand_circle_radius, max_rand_circle_radius), noise_color, 1)

    return img
