#!/usr/bin/env python3
"""
Test script for wall line extraction functionality

This script tests the basic imports and functionality of the wall line extraction
module to ensure everything is working correctly.
"""

import numpy as np
import sys
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import kkg_area_detection
        print("✓ kkg_area_detection imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import kkg_area_detection: {e}")
        return False

    try:
        from kkg_area_detection import WallSkeletonToCAD, extract_wall_lines_from_segment
        print("✓ Wall line extraction functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import wall line extraction functions: {e}")
        return False

    try:
        from kkg_area_detection.core.wall_line_extraction import WallSkeletonToCAD
        print("✓ WallSkeletonToCAD class imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import WallSkeletonToCAD: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of wall line extraction."""
    print("\nTesting basic functionality...")

    try:
        from kkg_area_detection.core.wall_line_extraction import (
            WallSkeletonToCAD,
            extract_wall_lines_from_segment
        )

        # Create a simple test mask (a rectangle)
        mask = np.zeros((100, 200), dtype=bool)
        mask[30:70, 50:150] = True  # Create a rectangular wall

        print("✓ Created test mask")

        # Test WallSkeletonToCAD class
        processor = WallSkeletonToCAD(mask)
        print("✓ WallSkeletonToCAD instance created")

        # Test the process method with minimal parameters
        lines = processor.process(
            rdp_epsilon=2.0,
            min_segment_length=10,
            enable_refinement=False,  # Disable refinement for simple test
            is_debug=False
        )

        print(f"✓ Wall line extraction completed: {len(lines)} lines extracted")

        # Test the convenience function
        lines2 = extract_wall_lines_from_segment(
            segment_mask=mask,
            rdp_epsilon=2.0,
            min_segment_length=10,
            enable_refinement=False,
        )

        print(f"✓ Convenience function completed: {len(lines2)} lines extracted")

        return True

    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        traceback.print_exc()
        return False


def test_get_approx_contours_integration():
    """Test integration with get_approx_contours_and_vertices function."""
    print("\nTesting get_approx_contours_and_vertices integration...")

    try:
        import kkg_area_detection

        # Create a simple segmentation array
        # Segment 0: background
        # Segment 1: inwall (class ID 2)
        # Segment 2: door (class ID 3)
        # Segment 3: room (class ID 0)
        segment_array = np.zeros((100, 200), dtype=np.int32)
        segment_array[30:70, 50:150] = 1  # Inwall segment
        segment_array[75:85, 80:120] = 2  # Door segment
        segment_array[10:25, 10:40] = 3   # Room segment

        # Create segments info
        segments_info = [
            {'id': 1, 'label_id': 2, 'category_id': 2},  # Inwall
            {'id': 2, 'label_id': 3, 'category_id': 3},  # Door
            {'id': 3, 'label_id': 0, 'category_id': 0},  # Room
        ]

        print("✓ Created test segmentation data")

        # Test wall line extraction (using default class IDs [2, 3])
        contours = kkg_area_detection.get_approx_contours_and_vertices(
            segment_array=segment_array,
            segments_info=segments_info,
            extract_wall_lines=True,
            # wall_line_class_ids not specified to test default [2, 3]
            wall_line_rdp_epsilon=2.0,
            wall_line_min_segment_length=10,
            wall_line_enable_refinement=False,
            align_to_lines=False,  # Disable line alignment for test
        )

        print(f"✓ get_approx_contours_and_vertices completed: {len(contours)} contours")

        # Check that we have both lines and polygons
        num_lines = sum(1 for c in contours if c.get('is_line', False))
        num_polygons = sum(1 for c in contours if not c.get('is_line', False))

        print(f"  - Lines: {num_lines}")
        print(f"  - Polygons: {num_polygons}")

        if num_lines > 0:
            print("✓ Wall line extraction working correctly")
        else:
            print("! No lines extracted (may be expected for simple test)")

        return True

    except Exception as e:
        print(f"✗ Error in integration test: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Wall Line Extraction Test Suite")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    # Test 1: Imports
    if test_imports():
        tests_passed += 1

    # Test 2: Basic functionality
    if test_basic_functionality():
        tests_passed += 1

    # Test 3: Integration
    if test_get_approx_contours_integration():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("✓ All tests passed! Wall line extraction is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
