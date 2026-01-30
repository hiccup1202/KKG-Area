#!/usr/bin/env python3
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.colors import hsv_to_rgb


def fill_holes(mask):
    """
    Fill any holes (internal regions with value 0) in the binary mask.

    Args:
        mask: Binary mask image

    Returns:
        A new binary mask with holes filled
    """
    # Make a copy of the mask
    mask_copy = mask.copy()

    # Get the dimensions of the mask
    height, width = mask.shape

    # Create a slightly larger mask to ensure border is filled
    flood_mask = np.zeros((height+2, width+2), np.uint8)

    # Fill from the border (this will fill the background, leaving the holes)
    cv2.floodFill(mask_copy, flood_mask, (0, 0), 1)

    # Invert the result to get the holes
    holes = np.where(mask_copy == 0, 1, 0).astype(np.uint8)

    # Add the holes to the original mask
    filled_mask = np.logical_or(mask, holes).astype(np.uint8)

    return filled_mask

def get_mask_centroid(mask):
    """
    Calculate the centroid of a binary mask.
    """
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # Fallback to center of bounding box if moments fail
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        cx = x + w // 2
        cy = y + h // 2
    return (cx, cy)

def calculate_overlap_ratio(mask1, mask2):
    """
    Calculate the overlap ratio between two masks.

    Args:
        mask1, mask2: Binary mask images

    Returns:
        A tuple of (overlap_ratio_1, overlap_ratio_2):
        - overlap_ratio_1: The ratio of the overlap area to mask1's area
        - overlap_ratio_2: The ratio of the overlap area to mask2's area
    """
    # Calculate the intersection area
    intersection = np.logical_and(mask1, mask2).sum()

    # Calculate the areas of each mask
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)

    # If either mask is empty, return 0 for both ratios
    if area1 == 0 or area2 == 0:
        return 0, 0

    # Calculate the overlap ratios
    overlap_ratio_1 = intersection / area1
    overlap_ratio_2 = intersection / area2

    return overlap_ratio_1, overlap_ratio_2

def approximate_with_orthogonal_lines(mask, epsilon_factor=0.02):
    """
    Approximate the contour of a mask with strictly horizontal and vertical lines only.
    Better handling for L-shaped and complex room layouts.

    Args:
        mask: Binary mask image
        epsilon_factor: Factor to control the approximation precision

    Returns:
        A orthogonal mask with horizontal and vertical lines only
    """
    # Find contours in the mask - handle different OpenCV versions
    contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract contours based on the length of the result
    if len(contour_result) == 3:
        # OpenCV 3.x
        _, contours, _ = contour_result
    else:
        # OpenCV 4.x
        contours, _ = contour_result

    if not contours:
        return mask

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a new mask for the orthogonal approximation
    orthogonal_mask = np.zeros_like(mask)

    # Step 1: Approximate the contour with Douglas-Peucker algorithm
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If we have too few points, just use a bounding rectangle
    if len(approx_polygon) < 4:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(orthogonal_mask, (x, y), (x+w, y+h), 1, -1)
        return orthogonal_mask

    # Get the points from the approximated polygon
    points = [p[0] for p in approx_polygon]
    n_points = len(points)

    # For each pair of consecutive points, create a rectilinear path
    rectilinear_polygon = []
    for i in range(n_points):
        p1 = points[i]
        p2 = points[(i + 1) % n_points]

        # Add the current point
        rectilinear_polygon.append([p1[0], p1[1]])

        # Create a "stair" between p1 and p2 (horizontal then vertical)
        rectilinear_polygon.append([p2[0], p1[1]])  # Horizontal segment

    # Convert to numpy array
    rectilinear_polygon = np.array(rectilinear_polygon).reshape((-1, 1, 2)).astype(np.int32)

    # Draw the rectilinear polygon on the mask
    cv2.drawContours(orthogonal_mask, [rectilinear_polygon], 0, 1, -1)

    # Fill any holes in the mask
    orthogonal_mask = fill_holes(orthogonal_mask)

    return orthogonal_mask

def filter_overlapping_masks(masks, overlap_threshold=3):
    """
    Filter out smaller masks that overlap significantly with larger masks.

    Args:
        masks: List of (mask_file, binary_mask, area, centroid) tuples
        overlap_threshold: Threshold for considering significant overlap (default: 0.5)

    Returns:
        Filtered list of masks
    """
    # Sort masks by area in descending order (largest first)
    sorted_masks = sorted(masks, key=lambda x: x[2], reverse=True)

    # Initialize list of masks to keep
    filtered_masks = []

    # Process each mask
    for i, (mask_file, mask, area, centroid) in enumerate(sorted_masks):
        # Assume we keep this mask by default
        keep_mask = True

        # Check if this mask overlaps significantly with any already-kept masks
        for _, kept_mask, _, _ in filtered_masks:
            overlap_ratio_1, overlap_ratio_2 = calculate_overlap_ratio(mask, kept_mask)

            # If this mask overlaps significantly with a kept mask, discard it
            if overlap_ratio_1 > overlap_threshold:
                keep_mask = False
                print(f"  Discarding {mask_file.name} - Overlaps with larger mask by {overlap_ratio_1:.2f}")
                break

        # If we decided to keep this mask, add it to the filtered list
        if keep_mask:
            filtered_masks.append((mask_file, mask, area, centroid))

    return filtered_masks

def main():
    # Directory containing mask images
    mask_dir = Path("mask_images")

    # Check if directory exists
    if not mask_dir.exists():
        print(f"Error: Directory '{mask_dir}' does not exist.")
        return

    # Get all image files in the directory
    mask_files = sorted([f for f in mask_dir.glob("*.png") or mask_dir.glob("*.jpg")])

    if not mask_files:
        print(f"No image files found in '{mask_dir}'.")
        return

    print(f"Found {len(mask_files)} mask images.")

    # Read the first mask to get dimensions
    first_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    height, width = first_mask.shape

    # Load the original room image
    room_image_path = "room.png"
    if not Path(room_image_path).exists():
        print(f"Warning: Original room image '{room_image_path}' not found. Creating white background.")
        # Create a white RGB image for the combined visualization
        background_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        # Read the original room image
        background_image = cv2.imread(room_image_path)
        # Resize if dimensions don't match
        if background_image.shape[:2] != (height, width):
            print(f"Resizing room image from {background_image.shape[:2]} to {(height, width)}")
            background_image = cv2.resize(background_image, (width, height))
        # Convert from BGR to RGB
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

    # Create a copy of the background image for the combined visualization
    combined_image = background_image.copy()

    # Define minimum area threshold (in pixels)
    min_area_threshold = 0.01 * height * width  # 1% of the image size

    # Filter masks and generate colors
    valid_masks = []

    # Process each mask to check area
    for mask_file in tqdm.tqdm(mask_files):
        # Read mask (grayscale)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        # Ensure mask is binary (0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)

        # Find connected components in the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # Skip the first component (label 0) which is the background
        if num_labels > 1:
            # Get areas of all components (excluding background)
            component_areas = stats[1:, cv2.CC_STAT_AREA]

            # Find the largest component
            largest_component_idx = np.argmax(component_areas) + 1

            # Create a new mask with only the largest component
            binary_mask = (labels == largest_component_idx).astype(np.uint8)

            if num_labels > 2:  # If there were multiple components (more than just background + 1)
                print(f"  {mask_file.name}: Kept largest of {num_labels-1} components (area: {component_areas[largest_component_idx-1]} pixels)")

        # Fill any holes in the mask
        binary_mask = fill_holes(binary_mask)

        # Approximate the mask with orthogonal lines
        orthogonal_mask = approximate_with_orthogonal_lines(binary_mask)

        # Calculate area
        area = np.sum(orthogonal_mask)

        if area >= min_area_threshold:
            # Get centroid for room number placement
            centroid = get_mask_centroid(orthogonal_mask)
            valid_masks.append((mask_file, orthogonal_mask, area, centroid))
            print(f"Including {mask_file.name} - Area: {area} pixels")
        else:
            print(f"Ignoring {mask_file.name} - Area too small: {area} pixels (threshold: {min_area_threshold})")

    # Filter out overlapping masks (keep larger ones)
    valid_masks = filter_overlapping_masks(valid_masks, overlap_threshold=0.3)

    print(f"Using {len(valid_masks)} masks after filtering by area and overlap.")

    # Sort masks by y-coordinate (top to bottom) and then x-coordinate (left to right)
    valid_masks.sort(key=lambda x: (x[3][1], x[3][0]))  # Sort by centroid coordinates

    # Generate distinct colors using HSV color space
    num_masks = len(valid_masks)
    colors = [hsv_to_rgb((i/num_masks, 0.9, 0.9)) for i in range(num_masks)]

    # Process each valid mask
    for i, (mask_file, binary_mask, area, centroid) in enumerate(valid_masks):
        print(f"Processing {mask_file.name}...")

        # Get color for this mask
        color = colors[i]
        color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

        # Create a colored mask image
        colored_mask = np.zeros_like(combined_image)
        for c in range(3):
            colored_mask[:, :, c] = binary_mask * int(color[c] * 255)

        # Apply the colored mask with alpha blending
        alpha = 0.5
        mask_area = (binary_mask > 0)
        for c in range(3):
            combined_image[:, :, c] = np.where(
                mask_area,
                combined_image[:, :, c] * (1 - alpha) + colored_mask[:, :, c] * alpha,
                combined_image[:, :, c]
            )

        # Add room number
        # Use a larger font size for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        room_number = str(i + 1)

        # Get text size to center it properly
        (text_width, text_height), baseline = cv2.getTextSize(room_number, font, font_scale, thickness)
        text_x = centroid[0] - text_width // 2
        text_y = centroid[1] + text_height // 2

        # Draw text with white background for better visibility
        cv2.putText(combined_image, room_number, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(combined_image, room_number, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # Clip values to valid range
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    # Save the combined image
    output_path = "combined_masks_with_room.png"
    cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    print(f"Combined visualization saved to {output_path}")

    # Display the combined image
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_image)
    plt.title("Room with Detected Areas")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("combined_masks_matplotlib.png")

if __name__ == "__main__":
    main()
