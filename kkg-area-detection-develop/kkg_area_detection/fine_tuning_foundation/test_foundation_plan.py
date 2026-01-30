#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import cv2
import matplotlib.pyplot as plt
from foundation.generator import create_argument_parser
from foundation.generator import main as generator_main


def display_images(output_dir, index=0):
    """
    Display the generated foundation plan and color mask images.

    Args:
        output_dir: Directory containing the generated images
        index: Index of the image to display
    """
    # Load images
    image_path = os.path.join(output_dir, 'images', f'foundation_plan_{index}.png')
    color_path = os.path.join(
        output_dir, 'annotations', f'foundation_plan_{index}_color.png'
    )
    bbox_vis_path = os.path.join(
        output_dir, 'annotations', f'foundation_plan_{index}_bbox_vis.png'
    )

    image = cv2.imread(image_path)
    color = cv2.imread(color_path)
    bbox_vis = cv2.imread(bbox_vis_path)

    # Check if images loaded correctly
    if image is None:
        print(f'Error loading image: {image_path}')
        return
    if color is None:
        print(f'Error loading color mask: {color_path}')
        return
    if bbox_vis is None:
        print(f'Error loading bbox visualisation: {bbox_vis_path}')
        return

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    bbox_vis_rgb = cv2.cvtColor(bbox_vis, cv2.COLOR_BGR2RGB)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display images
    axes[0].imshow(image_rgb)
    axes[0].set_title('Foundation Plan')
    axes[0].axis('off')

    axes[1].imshow(color_rgb)
    axes[1].set_title('Color Mask')
    axes[1].axis('off')

    axes[2].imshow(bbox_vis_rgb)
    axes[2].set_title('BBoxes')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'visualization_{index}.png'))


def main():
    """Generate foundation plan images and display them."""

    parser = create_argument_parser()
    args = parser.parse_args()

    print('Generating foundation plan images...')
    generator_main(args)

    # Check if the images were generated
    if os.path.exists(os.path.join(args.output_dir, 'images')):
        print(f'Images directory exists: {os.path.join(args.output_dir, "images")})')
        print(f'Image files: {len(os.listdir(os.path.join(args.output_dir, "images")))}')
    else:
        print(f'Images directory does not exist: {os.path.join(args.output_dir, "images")})')

    # Display the first image
    print('Displaying images...')
    try:
        display_images(args.output_dir, index=0)
    except Exception as e:
        print(f'Error: {e}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
