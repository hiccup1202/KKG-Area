"""
Basic usage example for the kkg_area_detection package.

This script demonstrates how to use the kkg_area_detection package to detect
and visualize areas in an image.
"""

import argparse
import os

import numpy as np
from dotenv import load_dotenv
from PIL import Image

import kkg_area_detection

load_dotenv(override=True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect and visualize areas in an image using kkg_area_detection.'
    )

    parser.add_argument(
        '-i', '--image',
        help='Path to the input image',
        default='sandbox/sample_image.jpg'
    )

    # モデル指定のグループ (model_pathとmodel_nameのどちらかを指定可能)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '-p', '--model-path',
        help='Path to a local model directory',
    )
    model_group.add_argument(
        '-n', '--model-name',
        help='Name of the model to use (for S3/cached models)',
    )

    parser.add_argument(
        '-d', '--device',
        choices=['cuda', 'cpu'],
        help='Device to run the model on (cuda or cpu)',
    )

    parser.add_argument(
        '-o', '--output',
        default='detected_areas_output.jpg',
        help='Path to save the output visualization image'
    )

    parser.add_argument(
        '-a', '--alpha',
        type=float,
        default=0.6,
        help='Alpha (transparency) value for visualization (0.0-1.0)'
    )

    parser.add_argument(
        '-e', '--epsilon',
        type=float,
        default=0.015,
        help='Epsilon value for contour approximation'
    )

    parser.add_argument(
        '-w', '--wall-filter',
        action='store_true',
        help='Apply wall filter to contours'
    )

    parser.add_argument(
        '-t', '--target-label-ids',
        type=int,
        nargs='+',
        default=[2, 3],
        help='Target label IDs for wall filter'
    )

    parser.add_argument(
        '--smoothing',
        action='store_true',
        help='Apply smoothing to contours'
    )

    parser.add_argument(
        '--angle-filter',
        action='store_true',
        help='Apply angle filtering to contours'
    )

    # Add new visualization options
    parser.add_argument(
        '--visible-classes',
        type=int,
        nargs='+',
        help='List of class IDs to display (e.g., --visible-classes 2 will only show class 2)'
    )

    parser.add_argument(
        '--class-colors',
        type=str,
        nargs='+',
        help='Fixed colors for specific classes in format "class_id:r,g,b"' +
             ' (e.g., --class-colors "1:255,0,0" "2:0,0,255")'
    )

    return parser.parse_args()


def main():
    """Run a basic example of area detection and visualization."""
    args = parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f'Error: Image file not found at {image_path}')
        print('Please provide a valid image path with --image option')
        return

    print(f'Processing image: {image_path}')

    try:
        image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    except Exception as e:
        print(f'Error loading image: {e}')
        return

    try:
        print('Initializing model...')
        # 新しいAPIに従って、model_pathとmodel_nameを適切に渡す
        init_kwargs = {}

        if args.model_path:
            init_kwargs['model_path'] = args.model_path
            print(f'Using model from path: {args.model_path}')
        elif args.model_name:
            init_kwargs['model_name'] = args.model_name
            print(f'Using model by name: {args.model_name}')

        if args.device:
            init_kwargs['device'] = args.device
            print(f'Using device: {args.device}')

        kkg_area_detection.initialize_model(**init_kwargs)
    except Exception as e:
        print(f'Error initializing model: {e}')
        return

    try:
        print('Getting segmentation result...')
        segmentation_result = kkg_area_detection.get_segmentation_result(image)
    except Exception as e:
        print(f'Error getting segmentation result: {e}')
        return

    segments_info = segmentation_result.get('segments_info', [])
    segmentation_map_tensor = segmentation_result.get('segmentation')

    if segmentation_map_tensor is None:
        print('Error: Segmentation map not found in the result.')
        return

    print(f'Detected {len(segments_info)} segments initially.')

    # Count segments by label_id
    label_counts = {}
    for seg in segments_info:
        label_id = seg.get('label_id', seg.get('category_id', -1))
        label_counts[label_id] = label_counts.get(label_id, 0) + 1

    print('Segments by label ID:')
    for label_id, count in sorted(label_counts.items()):
        print(f'  Label {label_id}: {count} segments')

    regions_with_coords = []
    print('\nExtracting contours...')
    segmentation_map_numpy = segmentation_map_tensor.cpu().numpy()
    contours_data = kkg_area_detection.get_approx_contours_and_vertices(
        segmentation_map_numpy,
        epsilon=args.epsilon,
        use_smoothing=args.smoothing,
        use_angle_filter=args.angle_filter,
        wall_filter=args.wall_filter,
        segments_info=segments_info,
        target_label_ids=args.target_label_ids,
        edge_margin=0.001,
        align_to_lines=False,  # 基本的な使用例ではアライメント不要
    )
    print(f'get_approx_contours_and_vertices returned {len(contours_data)} contour entries')

    # Count unique segment IDs in contours_data
    unique_segments = set(c['id'] for c in contours_data)
    print(f'Unique segment IDs in contours: {len(unique_segments)}')

    segment_details_map = {info['id']: info for info in segments_info}
    for contour_info in contours_data:
        segment_id = contour_info['id']
        if segment_id in segment_details_map:
            details = segment_details_map[segment_id]
            region_data = {
                'score': details['score'],
                'label_id': details['label_id'],
                'segment_id': segment_id,
                'coordinates': contour_info['vertices'],
            }
            # Add holes if present
            if 'holes' in contour_info:
                region_data['holes'] = contour_info['holes']
            regions_with_coords.append(region_data)
    print(f'Successfully extracted coordinates for {len(regions_with_coords)} regions.')

    # Count contours (some segments might have multiple contours)
    total_contours = 0
    for region in regions_with_coords:
        total_contours += 1  # Main contour
        if 'additional_polygons' in region:
            total_contours += len(region['additional_polygons'])

    print(f'Total polygons after approximation: {total_contours}')

    # Calculate total image area
    image_width, image_height = image.size
    total_image_area = image_width * image_height

    # Calculate area for each region and check for overlaps
    total_region_area = 0
    total_mask_area = 0
    area_threshold = total_image_area * 0.001  # 0.1% of image area

    # First pass: calculate areas and filter regions
    filtered_regions = []

    for i, region in enumerate(regions_with_coords):
        segment_id = region['segment_id']

        # Calculate original mask area
        mask_area = np.sum(segmentation_map_numpy == segment_id)
        mask_percentage = (mask_area / total_image_area) * 100
        total_mask_area += mask_area

        # Calculate area using the Shoelace formula
        vertices = region['coordinates']
        if len(vertices) >= 3:
            # Convert vertices to numpy array for easier calculation
            points = np.array([[v['x'], v['y']] for v in vertices])
            # Calculate area using cross product (Shoelace formula)
            area = 0.5 * abs(
                sum(
                    points[i][0] * points[(i + 1) % len(points)][1] -
                    points[(i + 1) % len(points)][0] * points[i][1]
                    for i in range(len(points))))

            # Subtract hole areas if present
            if 'holes' in region and region['holes']:
                for hole in region['holes']:
                    if len(hole) >= 3:
                        hole_points = np.array([[v['x'], v['y']] for v in hole])
                        hole_area = 0.5 * abs(
                            sum(
                                hole_points[i][0] * hole_points[(i + 1) % len(hole_points)][1] -
                                hole_points[(i + 1) % len(hole_points)][0] * hole_points[i][1]
                                for i in range(len(hole_points))))
                        area -= hole_area
        else:
            area = 0

        percentage = (area / total_image_area) * 100
        total_region_area += area

        # Filter out small regions
        if area >= area_threshold:
            filtered_regions.append(region)
            print(f"Region {i + 1} (Segment ID: {region['segment_id']}):")
            print(f"  Score: {region['score']:.4f}")
            print(f"  Label ID: {region['label_id']}")
            print(f"  Vertices: {len(region['coordinates'])}")
            print(f"  Original mask area: {mask_area} pixels ({mask_percentage:.2f}%)")
            print(f"  Contour area: {area:.0f} pixels ({percentage:.2f}%)")
            print(f"  Area difference: {abs(area - mask_area):.0f} pixels ({abs(percentage - mask_percentage):.2f}%)")
        else:
            print(f"Region {i + 1} (Segment ID: {region['segment_id']}): FILTERED (area {percentage:.3f}% < 0.1%)")

    # Print summary
    print("\nSummary:")
    print(f"Original segments detected: {len(segments_info)}")
    print(f"Segments after contour extraction: {len(regions_with_coords)}")
    print(
        f"Regions after area filtering: {len(filtered_regions)} "
        f"(filtered out: {len(regions_with_coords) - len(filtered_regions)})"
    )

    # Count final polygons
    final_polygon_count = 0
    for region in filtered_regions:
        final_polygon_count += 1  # Main polygon
        if 'additional_polygons' in region:
            final_polygon_count += len(region['additional_polygons'])
    print(f"Total polygons to display: {final_polygon_count}")
    print(f"Total image area: {total_image_area} pixels")
    print(f"Total original mask area: {total_mask_area} pixels ({(total_mask_area / total_image_area) * 100:.2f}%)")
    print(f"Total contour area: {total_region_area:.0f} pixels ({(total_region_area / total_image_area) * 100:.2f}%)")
    print(
        f"Difference: {abs(total_region_area - total_mask_area):.0f} pixels "
        f"({abs((total_region_area - total_mask_area) / total_image_area) * 100:.2f}%)",
    )

    # Check for potential overlaps
    if total_mask_area > total_image_area:
        print(
            f"WARNING: Total mask area exceeds image area by "
            f"{((total_mask_area / total_image_area) - 1) * 100:.2f}%, "
            f"indicating overlaps in segmentation!",
        )
    elif total_region_area > total_image_area:
        print(
            f"WARNING: Total contour area exceeds image area by "
            f"{((total_region_area / total_image_area) - 1) * 100:.2f}%, "
            f"indicating overlaps in contours!",
        )
    else:
        print(f"No overlaps detected (masks cover {(total_mask_area / total_image_area) * 100:.2f}% of image)")

    try:
        print('Visualizing regions...')

        # Process fixed class colors if provided
        fixed_class_colors = None
        if args.class_colors:
            fixed_class_colors = {}
            for color_spec in args.class_colors:
                try:
                    class_id_str, color_str = color_spec.split(':')
                    class_id = int(class_id_str)
                    r, g, b = map(int, color_str.split(','))
                    fixed_class_colors[class_id] = (r, g, b)
                    print(f"Using fixed color ({r},{g},{b}) for class {class_id}")
                except (ValueError, IndexError):
                    print(f"Warning: Invalid color specification '{color_spec}'. Expected format 'class_id:r,g,b'")

        # Process visible classes if provided
        visible_classes = args.visible_classes
        if visible_classes:
            print(f"Only showing classes: {visible_classes}")

        visualized_image = kkg_area_detection.visualize_regions(
            image,
            filtered_regions,     # Pass the filtered list (excluding small regions)
            alpha=args.alpha,     # Use alpha from command-line args
            color_map=None,       # Use default random colors
            fixed_class_colors=fixed_class_colors,  # Pass fixed colors for specific classes
            visible_classes=visible_classes,        # Pass list of classes to display
        )

        output_path = args.output
        visualized_image.save(output_path)

        visualized_raw_image = kkg_area_detection.visualize_regions(
            image,
            regions_with_coords,
            alpha=args.alpha,
            color_map=None,
            segmentation_result=segmentation_result,
            fixed_class_colors=fixed_class_colors,
            use_contours_for_visualization=False,  # Use raw mask, not contours
        )

        raw_output_path = args.output.replace('.png', '_raw.png')
        visualized_raw_image.save(raw_output_path)
        print(f'Result saved to {output_path}')

    except AttributeError:
        print('Error: `visualize_regions` function not found in kkg_area_detection.')
    except Exception as e:
        print(f'Error during visualization or saving: {e}')


if __name__ == '__main__':
    main()
