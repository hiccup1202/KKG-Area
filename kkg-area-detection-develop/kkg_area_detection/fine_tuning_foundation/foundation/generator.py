#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import shutil
from typing import Tuple

import cv2
import numpy as np

try:
    # Try absolute imports first (for when the package is installed)
    from kkg_area_detection.fine_tuning.foundation.grid_foundation import (
        calculate_grid_positions_only, create_foundation_with_width,
        create_grid_lines, generate_foundation_path)
    from kkg_area_detection.fine_tuning.foundation.symbols import (
        _compute_interior_mask, _place_symbols)
    from kkg_area_detection.fine_tuning.foundation.visual_effects import \
        add_noise
except ImportError:
    # Fall back to relative imports (for direct script execution)
    from .grid_foundation import (calculate_grid_positions_only,
                                  create_foundation_with_width,
                                  create_grid_lines, generate_foundation_path)
    from .symbols import _compute_interior_mask, _place_symbols
    from .visual_effects import add_noise


def generate_foundation_plan(
    output_dir: str,
    num_images: int,
    grid_color: Tuple[int, int, int],
    foundation_color: Tuple[int, int, int],
    foundation_jitter: int,
    diag_prob: float,
    min_foundation_segments: int,
    max_foundation_segments: int,
    grid_line_thickness: int,
    max_foundation_outline_thickness: int,
    min_horizontal_grid_lines: int,
    max_horizontal_grid_lines: int,
    min_vertical_grid_lines: int,
    max_vertical_grid_lines: int,
    symbol_color: Tuple[int, int, int],
    symbol_thickness: int,
    max_floor_posts: int,
    num_wall_symbols: int,
    num_random_symbols: int,
    min_symbol_radius: int,
    max_symbol_radius: int,
    max_dimension_lines: int,
    max_random_lines: int,
    noise_color: Tuple[int, int, int],
    hatch_color: Tuple[int, int, int],
    hatch_step: int,
    min_dim_line_offset: int,
    max_dim_line_offset: int,
    min_rand_line_length: int,
    max_rand_line_length: int,
    min_rand_circle_radius: int,
    max_rand_circle_radius: int,
    min_canvas_padding: int,
    max_canvas_padding: int,
    no_grid_prob: float,
    floor_post_placement_prob: float,
) -> None:
    """
    Generate foundation plan images with grid lines and foundation annotations.

    Args:
        output_dir: Directory to save the generated images
        num_images: Number of images to generate
        grid_color: Color of the grid lines
        foundation_color: Color of the foundation
        foundation_jitter: Max pixels to jitter foundation endpoints (default: 8)
        diag_prob: Probability of choosing diagonal moves (default: 0.1)
        min_foundation_segments: Minimum number of segments in the foundation path
        max_foundation_segments: Maximum number of segments in the foundation path
        grid_line_thickness: Thickness for the grid lines
        max_foundation_outline_thickness: Max thickness for the outline of the foundation
        min_horizontal_grid_lines: Min number of horizontal grid lines
        max_horizontal_grid_lines: Max number of horizontal grid lines
        min_vertical_grid_lines: Min number of vertical grid lines
        max_vertical_grid_lines: Max number of vertical grid lines
        symbol_color: BGR color for symbols
        symbol_thickness: Thickness for symbol lines
        max_floor_posts: Maximum number of floor post symbols
        num_wall_symbols: Number of symbols on foundation wall endpoints
        num_random_symbols: Number of randomly placed symbols
        min_symbol_radius: Min radius for symbols
        max_symbol_radius: Max radius for symbols
        max_dimension_lines: Max number of dimension lines for noise
        max_random_lines: Max number of random lines for noise
        noise_color: BGR color for noise elements
        hatch_color: BGR color for hatching
        hatch_step: Step size for hatching lines
        min_dim_line_offset: Min offset for dimension lines
        max_dim_line_offset: Max offset for dimension lines
        min_rand_line_length: Min length for random noise lines
        max_rand_line_length: Max length for random noise lines
        min_rand_circle_radius: Min radius for random noise circles
        max_rand_circle_radius: Max radius for random noise circles
        min_canvas_padding: Minimum padding to add around the generated content on the final canvas
        max_canvas_padding: Maximum padding to add around the generated content on the final canvas
        no_grid_prob: Probability of generating an image with no visible grid lines
        floor_post_placement_prob: Probability of placing a floor post symbol at a valid location
    """
    images_dir = os.path.join(output_dir, 'images')
    annotations_dir = os.path.join(output_dir, 'annotations')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    hatch_options = ['diag_down_right', 'diag_down_left']  # Define options once

    for i in range(num_images):
        # Randomize image size (rectangular or square)
        content_width = np.random.randint(700, 1201)
        content_height = np.random.randint(700, 1201)
        content_image_size = (content_width, content_height)

        foundation_width = np.random.randint(10, 20)

        foundation_outline_thickness = np.random.randint(1, max_foundation_outline_thickness + 1)

        # Create precise grid lines (or skip drawing them)
        if random.random() < no_grid_prob:
            # No visible grid: calculate positions but don't draw lines
            horizontal_positions, vertical_positions = calculate_grid_positions_only(
                image_size=content_image_size,
                min_horizontal=min_horizontal_grid_lines,
                max_horizontal=max_horizontal_grid_lines,
                min_vertical=min_vertical_grid_lines,
                max_vertical=max_vertical_grid_lines,
            )
            grid_image = np.ones((content_height, content_width, 3), dtype=np.uint8) * 255
            grid_mask = np.zeros((content_height, content_width), dtype=np.uint8)
        else:
            # Visible grid: use existing function
            grid_image, grid_mask, horizontal_positions, vertical_positions = create_grid_lines(
                image_size=content_image_size,
                line_color=grid_color,
                line_thickness=grid_line_thickness,
                min_horizontal=min_horizontal_grid_lines,
                max_horizontal=max_horizontal_grid_lines,
                min_vertical=min_vertical_grid_lines,
                max_vertical=max_vertical_grid_lines,
            )

        # Generate foundation path (applying jitter to endpoints, biased selection)
        foundation_segments, _, _ = generate_foundation_path(
            horizontal_positions=horizontal_positions,
            vertical_positions=vertical_positions,
            image_size=content_image_size,
            min_segments=min_foundation_segments,
            max_segments=max_foundation_segments,
            foundation_jitter=foundation_jitter,
            diag_prob=diag_prob
        )

        # Create foundation with constant width using jittered segments
        foundation_image, foundation_mask = create_foundation_with_width(
            foundation_segments=foundation_segments,
            image_size=content_image_size,
            foundation_width=foundation_width,
            line_color=foundation_color,
            line_thickness=foundation_outline_thickness)

        # Combine grid and foundation images
        combined_image = grid_image.copy()
        foundation_line_mask = np.all(foundation_image == foundation_color, axis=-1)
        combined_image[foundation_line_mask] = foundation_color

        # Determine hatch direction for this image
        current_hatch_direction = random.choice(hatch_options)

        # Add noise to the combined image
        combined_image_noised = add_noise(combined_image,
                                          foundation_mask,
                                          grid_mask,
                                          hatch_direction=current_hatch_direction,
                                          max_dim_lines=max_dimension_lines,
                                          max_rand_lines=max_random_lines,
                                          noise_color=noise_color,
                                          hatch_color=hatch_color,
                                          hatch_step=hatch_step,
                                          min_dim_line_offset=min_dim_line_offset,
                                          max_dim_line_offset=max_dim_line_offset,
                                          min_rand_line_length=min_rand_line_length,
                                          max_rand_line_length=max_rand_line_length,
                                          min_rand_circle_radius=min_rand_circle_radius,
                                          max_rand_circle_radius=max_rand_circle_radius)

        # --------------------------------------------------------------
        # Compute interior mask and place symbols (◯/×)
        # --------------------------------------------------------------
        interior_mask = _compute_interior_mask(foundation_mask)

        combined_with_symbols, floor_post_bboxes_orig = _place_symbols(
            base_image=combined_image_noised,
            horizontal_positions=horizontal_positions,
            vertical_positions=vertical_positions,
            foundation_line_mask=foundation_line_mask,
            interior_mask=interior_mask,
            foundation_segments=foundation_segments,
            foundation_width=foundation_width,
            symbol_color=symbol_color,
            symbol_thickness=symbol_thickness,
            max_floor_posts=max_floor_posts,
            num_wall_symbols=num_wall_symbols,
            num_random_symbols=num_random_symbols,
            min_symbol_radius=min_symbol_radius,
            max_symbol_radius=max_symbol_radius,
            floor_post_placement_prob=floor_post_placement_prob,
        )

        # --- Apply canvas padding and random offset ---
        current_padding = random.randint(min_canvas_padding, max_canvas_padding)

        final_width = content_width + 2 * current_padding
        final_height = content_height + 2 * current_padding

        offset_x = random.randint(0, 2 * current_padding) if current_padding > 0 else 0
        offset_y = random.randint(0, 2 * current_padding) if current_padding > 0 else 0

        # Ensure content fits even with max offset from one side
        # This effectively means the offset can be at most current_padding towards any one direction from center
        # So, if content is at (0,0) on a canvas of content_width + 2*current_padding,
        # offset_x can range from 0 to 2*current_padding
        # The actual placement will be such that the top-left of content is at (offset_x, offset_y)
        # The effective padding for top/left will be offset_x, offset_y
        # The effective padding for right/bottom will be (final_width - content_width - offset_x) etc.

        final_combined_image = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255
        final_combined_image[
            offset_y:offset_y + content_height,
            offset_x:offset_x + content_width] = combined_with_symbols

        final_grid_mask = np.zeros((final_height, final_width), dtype=np.uint8)
        final_grid_mask[
            offset_y:offset_y + content_height,
            offset_x:offset_x + content_width] = grid_mask

        final_foundation_mask = np.zeros((final_height, final_width), dtype=np.uint8)
        final_foundation_mask[
            offset_y:offset_y + content_height,
            offset_x:offset_x + content_width] = foundation_mask

        # Adjust bounding boxes
        floor_post_bboxes = []
        for (x1, y1, x2, y2) in floor_post_bboxes_orig:
            floor_post_bboxes.append((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y))

        # --------------------------------------------------------------
        # Create color mask for visualisation (grid=blue, foundation=green) on the final canvas
        # --------------------------------------------------------------
        final_color_mask = np.zeros((final_height, final_width, 3), dtype=np.uint8)
        final_color_mask[final_grid_mask > 0] = [255, 0, 0]  # Blue in BGR
        final_color_mask[final_foundation_mask > 0] = [0, 255, 0]  # Green

        # --------------------------------------------------------------
        # Save floor-post annotation and visualisation
        # --------------------------------------------------------------
        bbox_json_path = os.path.join(annotations_dir, f'foundation_plan_{i}_floor_post.json')
        with open(bbox_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image': f'foundation_plan_{i}.png',
                'bboxes': [{'bbox': bbox, 'category': 'floor_post'} for bbox in floor_post_bboxes]
            }, f, ensure_ascii=False, indent=2)

        # Visualise bounding boxes on the colour mask
        bbox_vis = final_color_mask.copy()
        for (x1, y1, x2, y2) in floor_post_bboxes:
            cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red boxes
        cv2.imwrite(os.path.join(annotations_dir, f'foundation_plan_{i}_bbox_vis.png'), bbox_vis)

        # Save colour mask for downstream use/testing
        cv2.imwrite(os.path.join(annotations_dir, f'foundation_plan_{i}_color.png'), final_color_mask)

        # Save single-channel segmentation mask for semantic segmentation (0=background, 1=foundation)
        final_seg_mask = (final_foundation_mask > 0).astype(np.uint8)  # 0/1 mask
        cv2.imwrite(os.path.join(annotations_dir, f'foundation_plan_{i}_mask.png'), final_seg_mask * 255)

        # --------------------------------------------------------------
        # Finally, save the combined image with symbols *after* all additions
        # --------------------------------------------------------------
        cv2.imwrite(os.path.join(images_dir, f'foundation_plan_{i}.png'), final_combined_image)


def create_argument_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description='Generate foundation plan images with grid lines and foundation annotations.')
    parser.add_argument('--output_dir', type=str, default='foundation_data', help='Output directory')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to generate')
    parser.add_argument('--foundation_jitter', type=int, default=8, help='Max pixels to jitter foundation endpoints')
    parser.add_argument('--diag_prob', type=float, default=0.1,
                        help='Probability of choosing diagonal foundation segments')
    parser.add_argument('--min_foundation_segments', type=int, default=15,
                        help='Minimum number of segments in the foundation path')
    parser.add_argument('--max_foundation_segments', type=int, default=30,
                        help='Maximum number of segments in the foundation path')
    parser.add_argument('--grid_color', type=int, nargs=3, default=[100, 100, 100],
                        help='BGR color of the grid lines (e.g., 100 100 100)')
    parser.add_argument('--foundation_color', type=int, nargs=3, default=[0, 0, 0],
                        help='BGR color of the foundation lines (e.g., 0 0 0)')

    # Parameters from grid_foundation.py
    parser.add_argument('--grid_line_thickness', type=int, default=1, help='Thickness for the grid lines')
    parser.add_argument('--max_foundation_outline_thickness',
                        type=int, default=2,
                        help='Max thickness for the outline of the foundation')
    parser.add_argument('--min_horizontal_grid_lines', type=int, default=5, help='Min number of horizontal grid lines')
    parser.add_argument('--max_horizontal_grid_lines', type=int, default=10, help='Max number of horizontal grid lines')
    parser.add_argument('--min_vertical_grid_lines', type=int, default=5, help='Min number of vertical grid lines')
    parser.add_argument('--max_vertical_grid_lines', type=int, default=10, help='Max number of vertical grid lines')

    # Parameters from symbols.py
    parser.add_argument('--symbol_color', type=int, nargs=3, default=[0, 0, 0], help='BGR color for symbols')
    parser.add_argument('--symbol_thickness', type=int, default=1, help='Thickness for symbol lines')
    parser.add_argument('--max_floor_posts', type=int, default=15, help='Maximum number of floor post symbols')
    parser.add_argument('--num_wall_symbols',
                        type=int, default=20,
                        help='Number of symbols on foundation wall endpoints')
    parser.add_argument('--num_random_symbols', type=int, default=20, help='Number of randomly placed symbols')
    parser.add_argument('--min_symbol_radius', type=int, default=6, help='Min radius for symbols')
    parser.add_argument('--max_symbol_radius', type=int, default=9, help='Max radius for symbols')

    # Parameters from visual_effects.py
    parser.add_argument('--max_dimension_lines', type=int, default=6, help='Max number of dimension lines for noise')
    parser.add_argument('--max_random_lines', type=int, default=20, help='Max number of random lines for noise')
    parser.add_argument('--noise_color', type=int, nargs=3, default=[0, 0, 0], help='BGR color for noise elements')
    parser.add_argument('--hatch_color', type=int, nargs=3, default=[80, 80, 80], help='BGR color for hatching')
    parser.add_argument('--hatch_step', type=int, default=15, help='Step size for hatching lines')
    parser.add_argument('--min_dim_line_offset', type=int, default=25, help='Min offset for dimension lines')
    parser.add_argument('--max_dim_line_offset', type=int, default=60, help='Max offset for dimension lines')
    parser.add_argument('--min_rand_line_length', type=int, default=20, help='Min length for random noise lines')
    parser.add_argument('--max_rand_line_length', type=int, default=60, help='Max length for random noise lines')
    parser.add_argument('--min_rand_circle_radius', type=int, default=2, help='Min radius for random noise circles')
    parser.add_argument('--max_rand_circle_radius', type=int, default=5, help='Max radius for random noise circles')

    # Canvas padding arguments
    parser.add_argument('--min_canvas_padding', type=int, default=0,
                        help='Minimum padding (pixels) to add around the content on the final canvas')
    parser.add_argument('--max_canvas_padding', type=int, default=50,
                        help='Maximum padding (pixels) to add around the content on the final canvas')
    parser.add_argument('--no_grid_prob', type=float, default=0.3,
                        help='Probability (0.0 to 1.0) of generating an image with no visible grid lines')
    parser.add_argument('--floor_post_placement_prob', type=float, default=1.0,
                        # Default to 1.0 for backward compatibility
                        help='Probability (0.0 to 1.0) of placing a floor post symbol at a valid location')
    return parser


def main(args):
    """Main function to generate foundation plan images."""

    # Remove output_dir if it exists, then create it
    if os.path.exists(args.output_dir):
        print(f"Removing existing directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    generate_foundation_plan(
        output_dir=args.output_dir,
        num_images=args.num_images,
        grid_color=tuple(args.grid_color),
        foundation_color=tuple(args.foundation_color),
        foundation_jitter=args.foundation_jitter,
        diag_prob=args.diag_prob,
        min_foundation_segments=args.min_foundation_segments,
        max_foundation_segments=args.max_foundation_segments,
        grid_line_thickness=args.grid_line_thickness,
        max_foundation_outline_thickness=args.max_foundation_outline_thickness,
        min_horizontal_grid_lines=args.min_horizontal_grid_lines,
        max_horizontal_grid_lines=args.max_horizontal_grid_lines,
        min_vertical_grid_lines=args.min_vertical_grid_lines,
        max_vertical_grid_lines=args.max_vertical_grid_lines,
        symbol_color=tuple(args.symbol_color),
        symbol_thickness=args.symbol_thickness,
        max_floor_posts=args.max_floor_posts,
        num_wall_symbols=args.num_wall_symbols,
        num_random_symbols=args.num_random_symbols,
        min_symbol_radius=args.min_symbol_radius,
        max_symbol_radius=args.max_symbol_radius,
        max_dimension_lines=args.max_dimension_lines,
        max_random_lines=args.max_random_lines,
        noise_color=tuple(args.noise_color),
        hatch_color=tuple(args.hatch_color),
        hatch_step=args.hatch_step,
        min_dim_line_offset=args.min_dim_line_offset,
        max_dim_line_offset=args.max_dim_line_offset,
        min_rand_line_length=args.min_rand_line_length,
        max_rand_line_length=args.max_rand_line_length,
        min_rand_circle_radius=args.min_rand_circle_radius,
        max_rand_circle_radius=args.max_rand_circle_radius,
        min_canvas_padding=args.min_canvas_padding,
        max_canvas_padding=args.max_canvas_padding,
        no_grid_prob=args.no_grid_prob,
        floor_post_placement_prob=args.floor_post_placement_prob)

    print(f'Generated {args.num_images} foundation plan images in {args.output_dir}')


if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
