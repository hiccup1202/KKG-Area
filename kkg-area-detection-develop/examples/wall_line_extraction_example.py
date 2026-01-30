#!/usr/bin/env python3
"""
Wall Line Extraction Example

This example demonstrates how to extract wall segments as lines instead of polygons
using the skeletonization-based approach. Wall lines are extracted for specific
class IDs (e.g., exwall/外壁) while other segments remain as polygons.
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import kkg_area_detection


def visualize_wall_lines(
    image: Image.Image,
    contours_list: List[Dict[str, Any]],
    line_color: tuple = (255, 0, 0),  # Red for lines
    polygon_color: tuple = (0, 255, 0),  # Green for polygons
    line_thickness: int = 3,
    polygon_thickness: int = 2,
    show_endpoints: bool = True,
    endpoint_size: int = 5,
) -> Image.Image:
    """
    Visualize wall lines and polygons on an image.

    Args:
        image: Input image
        contours_list: List of contour information (may contain lines or polygons)
        line_color: Color for line segments (BGR)
        polygon_color: Color for polygons (BGR)
        line_thickness: Thickness for drawing lines
        polygon_thickness: Thickness for drawing polygons
        show_endpoints: Whether to show line endpoints
        endpoint_size: Size of endpoint markers

    Returns:
        Visualized image
    """
    vis_array = np.array(image).copy()

    for contour_info in contours_list:
        vertices = contour_info['vertices']
        is_line = contour_info.get('is_line', False)

        if is_line:
            # Draw as line segment
            if len(vertices) >= 2:
                # Draw the line
                cv2.line(
                    vis_array,
                    (vertices[0]['x'], vertices[0]['y']),
                    (vertices[-1]['x'], vertices[-1]['y']),
                    line_color,
                    line_thickness
                )

                # Draw endpoints if requested
                if show_endpoints:
                    cv2.circle(
                        vis_array,
                        (vertices[0]['x'], vertices[0]['y']),
                        endpoint_size,
                        (0, 0, 255),  # Blue for endpoints
                        -1
                    )
                    cv2.circle(
                        vis_array,
                        (vertices[-1]['x'], vertices[-1]['y']),
                        endpoint_size,
                        (0, 0, 255),  # Blue for endpoints
                        -1
                    )
        else:
            # Draw as polygon
            contour_np = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
            cv2.polylines(
                vis_array,
                [contour_np],
                isClosed=True,
                color=polygon_color,
                thickness=polygon_thickness
            )

    return Image.fromarray(vis_array)


def compare_methods(
    image_path: str,
    model_path: str = None,
    wall_class_ids: List[int] = None,
    save_comparison: bool = False,
    output_dir: str = "./output",
) -> None:
    """
    Compare standard polygon extraction vs wall line extraction.

    Args:
        image_path: Path to input image
        model_path: Path to model directory (optional)
        wall_class_ids: Class IDs to extract as lines (default: [2, 3] for inwall, door)
        save_comparison: Whether to save comparison images
        output_dir: Directory to save output images
    """
    # Default to class IDs 2, 3 (inwall, door) if not specified
    if wall_class_ids is None:
        wall_class_ids = [2, 3]

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


    # Method 1: Standard polygon extraction
    print("\nExtracting standard polygons...")
    polygon_contours = kkg_area_detection.get_approx_contours_and_vertices(
        segment_array=segmentation_map,
        segments_info=segments_info,
        epsilon=0.015,
        use_smoothing=True,
        align_to_lines=False,  # Disable line alignment for comparison
    )

    # Method 2: Wall line extraction for specified class IDs
    print(f"\nExtracting wall lines for class IDs: {wall_class_ids}...")
    line_contours = kkg_area_detection.get_approx_contours_and_vertices(
        segment_array=segmentation_map,
        segments_info=segments_info,
        epsilon=0.015,
        extract_wall_lines=True,
        wall_line_class_ids=wall_class_ids,
        wall_line_rdp_epsilon=3.0,
        wall_line_min_segment_length=20,
        wall_line_enable_refinement=True,
        wall_line_extend_threshold=40,
        wall_line_snap_tolerance=10,
        wall_line_manhattan_angle_tolerance=10,
        wall_line_coordinate_snap_tolerance=10,
        align_to_lines=False,  # Disable line alignment since wall line extraction handles this
    )

    # Count lines vs polygons
    num_lines = sum(1 for c in line_contours if c.get('is_line', False))
    num_polygons = sum(1 for c in line_contours if not c.get('is_line', False))

    print(f"\nResults:")
    print(f"- Standard method: {len(polygon_contours)} polygons")
    print(f"- Wall line method: {num_lines} lines, {num_polygons} polygons")

    # Print which segments were converted to lines
    print("\nSegments converted to lines:")
    for contour in line_contours:
        if contour.get('is_line', False):
            segment_id = contour['id']
            # Find segment info
            for seg_info in segments_info:
                if seg_info['id'] == segment_id:
                    label_id = seg_info.get('label_id', seg_info.get('category_id'))
                    print(f"  - Segment {segment_id}: class ID {label_id}")
                    break

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Standard polygon method
    polygon_vis = visualize_wall_lines(
        image,
        polygon_contours,
        polygon_color=(0, 255, 0),
        show_endpoints=False
    )
    axes[1].imshow(polygon_vis)
    axes[1].set_title(f"Standard Polygons ({len(polygon_contours)} total)")
    axes[1].axis('off')

    # Wall line method
    line_vis = visualize_wall_lines(
        image,
        line_contours,
        line_color=(255, 0, 0),
        polygon_color=(0, 255, 0),
        show_endpoints=True
    )
    axes[2].imshow(line_vis)
    axes[2].set_title(f"Wall Lines ({num_lines} lines, {num_polygons} polygons)")
    axes[2].axis('off')

    plt.tight_layout()

    # Save comparison if requested
    if save_comparison:
        os.makedirs(output_dir, exist_ok=True)

        # Save individual visualizations
        polygon_vis.save(os.path.join(output_dir, "polygons.png"))
        line_vis.save(os.path.join(output_dir, "wall_lines.png"))

        # Save comparison plot
        plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison images to: {output_dir}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Extract wall segments as lines using skeletonization"
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
        "--wall-class-ids",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Class IDs to extract as lines (default: 2, 3 for inwall, door)"
    )

    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save comparison images"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./output",
        help="Output directory for saved images"
    )

    # Wall line extraction parameters
    parser.add_argument(
        "--rdp-epsilon",
        type=float,
        default=3.0,
        help="RDP simplification tolerance for wall lines"
    )

    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=20,
        help="Minimum length for wall line segments"
    )

    parser.add_argument(
        "--extend-threshold",
        type=float,
        default=40,
        help="Maximum distance for line extension"
    )

    parser.add_argument(
        "--snap-tolerance",
        type=float,
        default=10,
        help="Tolerance for endpoint snapping"
    )

    parser.add_argument(
        "--manhattan-angle-tolerance",
        type=float,
        default=10,
        help="Angle tolerance for Manhattan alignment"
    )

    parser.add_argument(
        "--coordinate-snap-tolerance",
        type=float,
        default=10,
        help="Distance tolerance for coordinate snapping"
    )

    parser.add_argument(
        "--disable-refinement",
        action="store_true",
        help="Disable CAD refinement steps"
    )

    args = parser.parse_args()

    # Run comparison
    compare_methods(
        image_path=args.image,
        model_path=args.model_path,
        wall_class_ids=args.wall_class_ids,
        save_comparison=args.save_comparison,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
