"""
Inference module for area detection using Mask2Former models.

This module provides functionality to detect regions in images using
pre-trained or custom Mask2Former models.
"""

import logging
import os
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:
    torch = None
    print('Warning: PyTorch could not be imported. Model inference will not be available.')

try:
    from PIL import Image
except ImportError:
    Image = None
    print('Warning: PIL could not be imported. Image processing will not be available.')

try:
    from transformers import (Mask2FormerForUniversalSegmentation,
                              MaskFormerImageProcessor)
except ImportError:
    Mask2FormerForUniversalSegmentation = None
    MaskFormerImageProcessor = None
    print('Warning: transformers could not be imported. Model inference will not be available.')

try:
    import cv2
except ImportError:
    cv2 = None
    print('Warning: OpenCV could not be imported. Contour detection will not be available.')

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    boto3 = None
    ClientError = Exception
    S3_AVAILABLE = False
    print('Warning: boto3 could not be imported. S3 model download will not be available.')

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LOCAL_MODEL_PATH = 'model/large_20'  # Default path to locally saved weights
DEFAULT_MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_cache')


_model = None
_processor = None
_device = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def download_model_from_s3(
    bucket_name: str,
    s3_key: str,
    local_path: str,
) -> bool:
    """
    Download a model from S3 to a local path.
    If s3_key ends with '/' it's treated as a directory and all contents are downloaded.

    Args:
        bucket_name: S3 bucket name
        s3_key: S3 object key (path to the model in S3)
        local_path: Local path to save the downloaded model

    Returns:
        bool: True if download was successful, False otherwise
    """
    if not S3_AVAILABLE:
        logger.warning("boto3 is not available. Cannot download model from S3.")
        return False

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'ap-northeast-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        )

        # Check if s3_key is a directory (ends with /)
        is_directory = s3_key.endswith('/')

        if is_directory:
            logger.info(f"Downloading directory from S3: s3://{bucket_name}/{s3_key} to {local_path}")

            # Make sure the local directory exists
            os.makedirs(local_path, exist_ok=True)

            # List all objects with the given prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_key)

            download_count = 0
            for page in pages:
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    # Get the relative path from the prefix
                    relative_path = obj['Key'][len(s3_key):]
                    if not relative_path:  # Skip the directory itself
                        continue

                    # Create target file path
                    target_file = os.path.join(local_path, relative_path)

                    # Create directories if needed
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)

                    # Download the file
                    logger.debug(f"Downloading {obj['Key']} to {target_file}")
                    s3_client.download_file(bucket_name, obj['Key'], target_file)
                    download_count += 1

            logger.info(f"Successfully downloaded {download_count} files from S3 directory")
        else:
            # Download a single file
            logger.info(f"Downloading model from S3: s3://{bucket_name}/{s3_key} to {local_path}")
            s3_client.download_file(bucket_name, s3_key, local_path)
            logger.info("Successfully downloaded model from S3")

        return True
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading from S3: {e}")
        return False


def initialize_model(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """
    Initialize the Mask2Former model and processor.

    Args:
        model_path: Path to a custom model. Takes highest priority if specified.
        device: Device to load the model on ('cuda', 'cpu', or None for auto-detection).
        model_name: Name to use for caching the model. Used when model_path is None.
                   Will check if model already exists in cache before downloading from S3.

    Raises:
        RuntimeError: If the model or processor cannot be initialized or no model is available.
    """
    global _model, _processor

    try:
        _processor = MaskFormerImageProcessor(
            reduce_labels=True,
            do_rescale=True,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    except Exception as e:
        raise RuntimeError(f'Error initializing MaskFormerImageProcessor: {e}')

    # 1. If model_path is specified, use it directly (highest priority)
    if model_path is not None:
        try:
            _model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_path,
                local_files_only=True
            )
            logger.info(f'Loaded model from specified path: {model_path}')
            _setup_device(device)
            return
        except Exception as e:
            logger.error(f'Error loading model from specified path: {e}')
            raise RuntimeError(f'Error loading model from {model_path}: {e}')

    # 2. If model_name is specified but model_path is not, use model_name
    if model_name:
        cache_dir = os.getenv('MODEL_CACHE_DIR', DEFAULT_MODEL_CACHE_DIR)
        local_model_path = os.path.join(cache_dir, model_name)

        logger.info(f'Using model cache name: {model_name}')
        logger.info(f'Local model path: {local_model_path}')

        # Check if the model is already cached
        model_is_cached = os.path.exists(local_model_path)

        # For directories, check if it's not empty
        if model_is_cached and os.path.isdir(local_model_path):
            model_is_cached = len(os.listdir(local_model_path)) > 0

        # If cached, load from cache
        if model_is_cached:
            try:
                logger.info(f'Loading model from cache: {local_model_path}')
                _model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    local_model_path,
                    local_files_only=True
                )
                logger.info('Successfully loaded model from cache')
                _setup_device(device)
                return
            except Exception as e:
                logger.error(f'Error loading model from cache: {e}')
                # Continue to try downloading if loading from cache failed

        # Not cached or failed to load from cache, try downloading from S3
        s3_bucket = os.getenv('S3_BUCKET_NAME')
        s3_model_key = os.getenv('S3_MODEL_KEY')

        if not (S3_AVAILABLE and s3_bucket and s3_model_key):
            raise RuntimeError(
                f'Model {model_name} not found in cache and S3 configuration is incomplete. '
                'Please set S3_BUCKET_NAME and S3_MODEL_KEY environment variables.'
            )

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Download the model
        if not download_model_from_s3(s3_bucket, s3_model_key, local_model_path):
            raise RuntimeError(
                'Failed to download model from S3. '
                'Check S3 configuration and network connection.'
            )

        # Try to load the downloaded model
        try:
            _model = Mask2FormerForUniversalSegmentation.from_pretrained(
                local_model_path,
                local_files_only=True
            )
            logger.info(f'Loaded model from S3 (cached at: {local_model_path})')
            _setup_device(device)
            return
        except Exception as e:
            logger.error(f'Error loading downloaded model: {e}')
            raise RuntimeError(f'Downloaded model from S3 but failed to load it: {e}')

    # 3. If neither model_path nor model_name is specified, raise an error
    raise RuntimeError(
        'No model could be loaded. Please provide either model_path or model_name. '
        'If using model_name, ensure S3 environment variables are set correctly.'
    )


def _setup_device(device: Optional[str] = None) -> None:
    """
    Set up the device for the model.

    Args:
        device: Device to load the model on ('cuda', 'cpu', or None for auto-detection).
    """
    global _device

    _model.eval()

    if device:
        _device = torch.device(device)
    else:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _model.to(_device)
    logger.info(f'Model loaded on device: {_device}')


def get_segmentation_result(image: Image.Image) -> Dict[str, Any]:
    """
    Applies Mask2Former to an image and returns the raw segmentation result.

    This function matches the behavior in app.py, returning the raw segmentation
    result without any contour processing.

    Args:
        image: A PIL Image object.

    Returns:
        A dictionary containing the segmentation result with:
        - 'segmentation': The segmentation map as a tensor.
        - 'segments_info': A list of dictionaries with segment information.

    Raises:
        ValueError: If the model or processor was not loaded correctly.
        RuntimeError: If an error occurs during inference or processing.
    """
    if not _model or not _processor:
        raise ValueError(
            'Model or processor not initialized correctly. Call initialize_model() first.'
        )
    if not _device:
        raise ValueError(
            'Device (CPU/GPU) not determined due to model loading failure.'
        )

    try:
        image = image.convert('RGB')

        inputs = _processor(
            images=image,
            return_tensors='pt',
        ).to(_device)

        with torch.no_grad():
            outputs = _model(**inputs)

        results = _processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
        )[0]

        return results

    except Exception as e:
        logger.error(f'Error during image processing or inference: {e}')
        raise RuntimeError(f'Failed to get segmentation result: {e}') from e
