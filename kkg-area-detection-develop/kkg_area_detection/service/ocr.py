"""
OCR service for KKG Area Detection.

This module provides OCR functionality using Azure Form Recognizer.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..types.contours import ContourVertex

try:
    import numpy as np
except ImportError:
    np = None
    print('Warning: NumPy could not be imported. OCR functionality will be limited.')

try:
    from PIL import Image
except ImportError:
    Image = None
    print('Warning: PIL could not be imported. OCR functionality will be limited.')

try:
    from azure.ai.formrecognizer import (AnalyzeResult,
                                         DocumentAnalysisApiVersion,
                                         DocumentAnalysisClient)
    from azure.core.credentials import AzureKeyCredential
    from azure.core.polling import LROPoller
except ImportError:
    DocumentAnalysisClient = None
    AzureKeyCredential = None
    print('Warning: Azure Form Recognizer could not be imported. OCR functionality will not be available.')


class BoundingBox:
    """
    Represents a bounding box with text content.
    """

    def __init__(self, polygon: List[List[float]], content: str):
        """
        Initialize a BoundingBox.

        Args:
            polygon: List of [x, y] coordinates defining the bounding box.
            content: Text content within the bounding box.
        """
        self.polygon = polygon
        self.content = content

    def get_center(self) -> Tuple[float, float]:
        """
        Get the center point of the bounding box.

        Returns:
            Tuple of (x, y) coordinates of the center.
        """
        x_coords = [point.x for point in self.polygon]
        y_coords = [point.y for point in self.polygon]
        return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)

    def is_inside_region(self, region_vertices: List[ContourVertex]) -> bool:
        """
        Check if the center of this bounding box is inside a region.

        Args:
            region_vertices: List of vertices defining the region.

        Returns:
            True if the center is inside the region, False otherwise.
        """
        if not np:
            raise ImportError('NumPy is required for this functionality.')

        region_points = np.array([[v['x'], v['y']] for v in region_vertices])
        center_x, center_y = self.get_center()
        return self._point_in_polygon(center_x, center_y, region_points)

    def _point_in_polygon(self, x: float, y: float, polygon: np.ndarray) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.

        Args:
            x: X-coordinate of the point.
            y: Y-coordinate of the point.
            polygon: Numpy array of polygon vertices.

        Returns:
            True if the point is inside the polygon, False otherwise.
        """
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


class AzureVisionClient:
    """
    Client for Azure Form Recognizer API.
    """

    def __init__(self, endpoint: str, key: str):
        """
        Initialize the Azure Vision client.

        Args:
            endpoint: Azure Form Recognizer endpoint.
            key: Azure Form Recognizer API key.
        """
        if DocumentAnalysisClient is None or AzureKeyCredential is None:
            raise ImportError(
                'Azure Form Recognizer is required for this functionality.'
            )
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
            api_version=DocumentAnalysisApiVersion.V2023_07_31,
        )

    def text_detection(self, image_content: bytes) -> 'AnalyzeResult':
        """
        Detect text in an image.

        Args:
            image_content: Image content as bytes.

        Returns:
            Result of text detection.
        """
        poller = self.client.begin_analyze_document(
            'prebuilt-layout',
            document=image_content,
        )
        return poller.result()


def get_image(filename: str) -> bytes:
    """
    Read image file as bytes.

    Args:
        filename: Path to image file.

    Returns:
        Image content as bytes.
    """
    with open(filename, 'rb') as f:
        return f.read()


def azure_result_to_content(res: AnalyzeResult) -> List[BoundingBox]:
    """
    Convert Azure OCR result to a list of BoundingBox objects.

    Args:
        res: Azure OCR result.

    Returns:
        List of BoundingBox objects.
    """
    content = []
    for line in res.pages[0].lines:
        content.append(BoundingBox(line.polygon, line.content))
    return content


class OCRService:
    """
    Service for OCR using Azure Form Recognizer.
    """

    def __init__(self, azure_endpoint: str, azure_key: str):
        """
        Initialize the OCR service.

        Args:
            azure_endpoint: Azure Form Recognizer endpoint.
            azure_key: Azure Form Recognizer API key.
        """
        if not azure_endpoint or not azure_key:
            raise ValueError(
                'Azure credentials (azure_endpoint, azure_key) are required'
            )
        self.client = AzureVisionClient(azure_endpoint, azure_key)

    def get_content(self, image_path: str) -> List[BoundingBox]:
        """
        Get text content from an image.

        Args:
            image_path: Path to image file.

        Returns:
            List of BoundingBox objects.
        """
        try:
            image_content = get_image(image_path)
            res: AnalyzeResult = self.client.text_detection(image_content)
            return azure_result_to_content(res)
        except Exception as e:
            print(f'Failed to get content from {image_path}: {e}')
            return []


OCR_CACHE_FILE = Path(__file__).parent.parent / 'cache' / 'ocr_cache.pkl'
ROOM_NAMES_CONFIG_FILE = Path(__file__).parent.parent / 'config' / 'room_names.json'


def calculate_image_hash(image_path: str) -> str:
    """
    Calculate MD5 hash of an image.

    Args:
        image_path: Path to image file.

    Returns:
        MD5 hash of the image.
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        return hashlib.md5(image_bytes).hexdigest()


def _load_ocr_cache() -> Dict[str, List[BoundingBox]]:
    """
    Load OCR cache from file.

    Returns:
        Dictionary mapping image hashes to OCR results.
    """
    if OCR_CACHE_FILE.exists():
        with open(OCR_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


def handle_ocr(
    path: str, azure_endpoint: str, azure_key: str, enable_cache: bool = False
) -> List[BoundingBox]:
    """
    Handle OCR with optional caching.

    Args:
        path: Path to image file.
        azure_endpoint: Azure Form Recognizer endpoint.
        azure_key: Azure Form Recognizer API key.
        enable_cache: Whether to enable caching.

    Returns:
        List of BoundingBox objects.
    """
    ocr_cache = {}
    image_hash = None
    content = None

    if enable_cache:
        ocr_cache = _load_ocr_cache()
        image_hash = calculate_image_hash(path)
        if image_hash in ocr_cache:
            print(f'OCR cache found for {path}')
            content = ocr_cache[image_hash]

    if content is None:
        print(f'OCR cache not found for {path} or cache disabled, applying OCR')
        ocr_service = OCRService(azure_endpoint, azure_key)
        content = ocr_service.get_content(path)

        if enable_cache and image_hash:
            ocr_cache[image_hash] = content  # Add/update the cache dict
            OCR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OCR_CACHE_FILE, 'wb') as f:
                pickle.dump(ocr_cache, f)

    return content


def load_room_names() -> List[str]:
    """
    Load room names from the configuration file.

    Returns:
        List of room names.
    """
    try:
        with open(ROOM_NAMES_CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('room_names', [])
    except FileNotFoundError:
        print(f'Room names config file not found: {ROOM_NAMES_CONFIG_FILE}')
        return []
    except json.JSONDecodeError as e:
        print(f'Error decoding room names config: {e}')
        return []


def is_room_name(text: str, room_name_keywords: Optional[List[str]] = None) -> bool:
    """
    Check if a text is likely to be a room name.

    Args:
        text: Text to check.
        room_name_keywords: List of keywords that indicate a room name.

    Returns:
        True if the text is likely to be a room name, False otherwise.
    """
    keywords = room_name_keywords or load_room_names()
    if not keywords:
        # Fallback to a minimal set of keywords if config file is not available
        keywords = ['LDK', 'リビング', '寝室', 'キッチン', '浴室', 'トイレ', '洗面所', '玄関', '廊下']

    for keyword in keywords:
        if keyword == text:
            return True
    # Check if text contains any of the keywords
    for keyword in keywords:
        if keyword in text:
            return True
    return False


def find_room_name_in_region(
    region_vertices: List[ContourVertex],
    ocr_results: List[BoundingBox],
    room_name_keywords: Optional[List[str]] = None,
    default_name: str = 'Room 1',
) -> str:
    """
    Find the most likely room name in a region.

    Args:
        region_vertices: List of vertices defining the region.
        ocr_results: List of OCR results.
        room_name_keywords: List of keywords that indicate a room name.
        default_name: Default name to return if no room name is found.

    Returns:
        Most likely room name in the region.
    """
    region_texts = []
    for box in ocr_results:
        if box.is_inside_region(region_vertices):
            region_texts.append(box)
    if not region_texts:
        return default_name
    room_name_candidates = []
    for box in region_texts:
        if is_room_name(box.content, room_name_keywords):
            room_name_candidates.append(box.content)
    if not room_name_candidates:
        return default_name
    return room_name_candidates[0]
