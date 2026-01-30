#!/usr/bin/env python3
"""
Advanced Wall Line Extraction Example

This example demonstrates advanced usage of wall line extraction including:
- Custom parameter tuning for different wall types
- Direct usage of WallSkeletonToCAD class
- Visualization of intermediate processing steps
- Export to DXF format
"""

import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt

import kkg_area_detection
from kkg_area_detection import WallSkeletonToCAD


def extract_and_visualize_wall_lines(
    image_path: str,
    model_path: str = None,
    target_class_id: int = 3,
    show_intermediate_steps: bool = True,
    export_dxf: bool = False,
    output_dir: str = "./output",
) -> None:
    """
    Extract wall lines with visualization of intermediate steps.

    Args:
        image_path: Path to input image
        model_path: Path to model directory (optional)
        target_class_id: Class ID to extract as lines (default: 3)
        show_intermediate_steps: Whether to show intermediate processing steps
        export_dxf: Whether to export results to DXF format
        output_dir: Directory to save output files
    """
    # Load image
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path)

    # Initialize model
    print("Initializing model...")
    try:
        if model_path:
            kkg_area_detection.initialize_model(model_path=model_path)
            print(f"Using model from path: {model_path}")
        else:
            kkg_area_detection.initialize_model()
            print("Using default model")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Get segmentation result
    print("Running segmentation...")
    try:
        segmentation_result = kkg_area_detection.get_segmentation_result(image)
    except Exception as e:
        print(f"Error getting segmentation result: {e}")
        return

    segmentation_map = segmentation_result["segmentation"].cpu().numpy()
    segments_info = segmentation_result["segments_info"]

    # Find wall segments with target class ID
    wall_segment_ids = []
    for seg_info in segments_info:
        label_id = seg_info.get('label_id', seg_info.get('category_id'))
        if label_id == target_class_id:
            wall_segment_ids.append(seg_info['id'])

    print(f"\nFound {len(wall_segment_ids)} wall segments with class ID {target_class_id}")

    if not wall_segment_ids:
        print("No wall segments found with the specified class ID")
        return

    # Process each wall segment
    for i, segment_id in enumerate(wall_segment_ids):
        print(f"\n--- Processing wall segment {i+1}/{len(wall_segment_ids)} (ID: {segment_id}) ---")

        # Extract segment mask
        segment_mask = (segmentation_map == segment_id).astype(bool)

        # Process with different parameter sets
        parameter_sets = [
            {
                "name": "Default",
                "rdp_epsilon": 3.0,
                "min_segment_length": 20,
                "enable_refinement": True,
            },
            {
                "name": "High Detail",
                "rdp_epsilon": 1.0,
                "min_segment_length": 10,
                "enable_refinement": True,
            },
            {
                "name": "Simplified",
                "rdp_epsilon": 5.0,
                "min_segment_length": 30,
                "enable_refinement": False,
            },
        ]

        results = []
        for params in parameter_sets:
            print(f"\nProcessing with {params['name']} parameters...")
            processor_copy = WallSkeletonToCAD(segment_mask)
            lines = processor_copy.process(
                rdp_epsilon=params['rdp_epsilon'],
                min_segment_length=params['min_segment_length'],
                enable_refinement=params['enable_refinement'],
            )
            results.append({
                'name': params['name'],
                'lines': lines,
                'processor': processor_copy,
                'params': params,
            })
            print(f"  Result: {len(lines)} line segments")

        # Visualize results
        if show_intermediate_steps:
            # Create visualization of processing steps
            fig = plt.figure(figsize=(20, 12))

            # Original mask
            ax1 = plt.subplot(2, 4, 1)
            ax1.imshow(segment_mask, cmap='gray')
            ax1.set_title("Original Mask")
            ax1.axis('off')

            # Skeleton
            ax2 = plt.subplot(2, 4, 2)
            if hasattr(results[0]['processor'], 'skeleton'):
                ax2.imshow(results[0]['processor'].skeleton, cmap='gray')
            ax2.set_title("Skeleton")
            ax2.axis('off')

            # Graph structure
            ax3 = plt.subplot(2, 4, 3)
            ax3.imshow(segment_mask, cmap='gray', alpha=0.3)
            if hasattr(results[0]['processor'], 'junction_points'):
                for junction in results[0]['processor'].junction_points:
                    ax3.plot(junction[0], junction[1], 'ro', markersize=8)
            ax3.set_title(f"Junctions ({len(results[0]['processor'].junction_points)} points)")
            ax3.axis('off')

            # RDP paths
            ax4 = plt.subplot(2, 4, 4)
            ax4.imshow(segment_mask, cmap='gray', alpha=0.3)
            if hasattr(results[0]['processor'], 'simplified_paths'):
                for path in results[0]['processor'].simplified_paths:
                    if len(path) > 1:
                        ax4.plot(path[:, 0], path[:, 1], 'g-', linewidth=2)
                        ax4.plot(path[:, 0], path[:, 1], 'go', markersize=4)
            ax4.set_title("RDP Simplified Paths")
            ax4.axis('off')

            # Compare different parameter results
            for idx, result in enumerate(results):
                ax = plt.subplot(2, 4, 5 + idx)
                ax.imshow(segment_mask, cmap='gray', alpha=0.3)

                # Draw lines
                for line in result['lines']:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        x_coords = [c[0] for c in coords]
                        y_coords = [c[1] for c in coords]
                        ax.plot(x_coords, y_coords, 'r-', linewidth=3)
                        ax.plot(x_coords, y_coords, 'bo', markersize=5)

                ax.set_title(f"{result['name']} ({len(result['lines'])} lines)")
                ax.axis('off')

            # Statistics
            ax8 = plt.subplot(2, 4, 8)
            ax8.axis('off')
            stats_text = "Processing Statistics:\n\n"
            for result in results:
                stats = result['processor'].get_statistics()
                stats_text += f"{result['name']}:\n"
                stats_text += f"  Lines: {stats['final_segments']}\n"
                stats_text += f"  Total length: {stats['total_length']:.1f} px\n"
                stats_text += f"  RDP epsilon: {result['params']['rdp_epsilon']}\n\n"
            ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

            plt.tight_layout()

            # Save figure
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                fig_path = os.path.join(output_dir, f"wall_segment_{segment_id}_processing.png")
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                print(f"Saved processing visualization to: {fig_path}")

            plt.show()

        # Export to DXF if requested
        if export_dxf and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for result in results:
                dxf_path = os.path.join(
                    output_dir,
                    f"wall_segment_{segment_id}_{result['name'].lower().replace(' ', '_')}.dxf"
                )
                # Note: The export_dxf method is part of WallSkeletonToCAD class
                # but not implemented in this example. You would need to add it.
                print(f"DXF export would save to: {dxf_path}")

    # Create combined visualization
    print("\n--- Creating combined visualization ---")

    # Extract all wall lines
    all_lines = kkg_area_detection.get_approx_contours_and_vertices(
        segment_array=segmentation_map,
        segments_info=segments_info,
        extract_wall_lines=True,
        wall_line_class_ids=[target_class_id],
        wall_line_rdp_epsilon=3.0,
        wall_line_enable_refinement=True,
        align_to_lines=False,  # Disable line alignment since wall line extraction handles this
    )

    # Create final visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image with overlay
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Wall lines overlay
    axes[1].imshow(image, alpha=0.5)

    # Draw all lines
    num_lines = 0
    for contour in all_lines:
        if contour.get('is_line', False):
            vertices = contour['vertices']
            if len(vertices) >= 2:
                x_coords = [v['x'] for v in vertices]
                y_coords = [v['y'] for v in vertices]
                axes[1].plot(x_coords, y_coords, 'r-', linewidth=3)
                axes[1].plot(x_coords, y_coords, 'bo', markersize=6)
                num_lines += 1

    axes[1].set_title(f"Extracted Wall Lines ({num_lines} total)")
    axes[1].axis('off')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, "wall_lines_final.png")
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        print(f"Saved final visualization to: {final_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Advanced wall line extraction with intermediate visualizations"
    )

    parser.add_argument(
        "-i", "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "-p", "--model-path",
        type=str,
        help="Path to the model directory (optional)"
    )

    parser.add_argument(
        "--target-class-id",
        type=int,
        default=3,
        help="Target class ID to extract as lines (default: 3)"
    )

    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Show intermediate processing steps"
    )

    parser.add_argument(
        "--export-dxf",
        action="store_true",
        help="Export results to DXF format"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./output",
        help="Output directory for saved files"
    )

    args = parser.parse_args()

    # Run extraction and visualization
    extract_and_visualize_wall_lines(
        image_path=args.image,
        model_path=args.model_path,
        target_class_id=args.target_class_id,
        show_intermediate_steps=args.show_steps,
        export_dxf=args.export_dxf,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
