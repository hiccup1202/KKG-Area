"""
Example script for room name detection using KKG Area Detection.

This script demonstrates how to use the room name detection functionality
to identify room names in floor plan images.
"""

import os

from PIL import Image, ImageDraw, ImageFont

import kkg_area_detection
from kkg_area_detection.core.room_detection import get_regions_with_room_names

AZURE_ENDPOINT = os.environ.get('AZURE_LAYOUT_ENDPOINT', 'your_azure_endpoint')
AZURE_KEY = os.environ.get('AZURE_API_KEY', 'your_azure_key')

ROOM_NAME_KEYWORDS = [
    'LDK', 'リビング', '寝室', 'キッチン', '浴室', 'トイレ', '洗面所', '玄関', '廊下',
    '和室', '洋室', '客間', '書斎', '子供部屋', 'クローゼット', '納戸', '物置',
    'ダイニング', 'バルコニー', 'テラス', '庭', 'ガレージ', '駐車場',
    '階段', 'エレベーター', 'ホール', 'エントランス',
]


def visualize_room_names(
    image_path: str,
    output_path: str,
    azure_endpoint: str,
    azure_key: str,
    room_name_keywords: list = None,
) -> None:
    """
    Visualize room names on an image.

    Args:
        image_path: Path to the input image.
        output_path: Path to save the output image.
        azure_endpoint: Azure Form Recognizer endpoint.
        azure_key: Azure Form Recognizer API key.
        room_name_keywords: List of keywords that indicate a room name.
    """
    image = Image.open(image_path)

    kkg_area_detection.initialize_model()

    contours_list, room_names = get_regions_with_room_names(
        image=image,
        image_path=image_path,
        azure_endpoint=azure_endpoint,
        azure_key=azure_key,
        room_name_keywords=room_name_keywords,
        default_name='Room 1',
        epsilon=0.015,
        use_smoothing=True,
        use_angle_filter=True,
        wall_filter=True,
        target_label_ids=[1, 2],
        edge_margin=0.05,
    )
    vis_image = kkg_area_detection.visualize_contours(
        image=image,
        contours_list=contours_list,
        show_vertex_count=False,
    )

    draw = ImageDraw.Draw(vis_image)

    try:
        font = ImageFont.truetype('arial.ttf', 20)
    except OSError:
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 20)
        except OSError:
            font = ImageFont.load_default()

    for contour_info in contours_list:
        region_id = contour_info['id']
        vertices = contour_info['vertices']

        x_coords = [v['x'] for v in vertices]
        y_coords = [v['y'] for v in vertices]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        room_name = room_names.get(region_id, 'Unknown')

        text_bbox = draw.textbbox((0, 0), room_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding = 2
        draw.rectangle(
            [
                center_x - text_width / 2 - padding,
                center_y - text_height / 2 - padding,
                center_x + text_width / 2 + padding,
                center_y + text_height / 2 + padding,
            ],
            fill=(255, 255, 255),  # White background
        )

        draw.text(
            (center_x - text_width / 2, center_y - text_height / 2),
            room_name,
            fill=(0, 0, 255),  # Blue color
            font=font,
        )
    vis_image.save(output_path)
    print(f'Saved visualization to {output_path}')
    print(f'Room names: {room_names}')


if __name__ == '__main__':
    input_image = 'path/to/your/floorplan.jpg'  # Replace with your image path
    output_image = 'output_room_names.jpg'
    visualize_room_names(
        image_path=input_image,
        output_path=output_image,
        azure_endpoint=AZURE_ENDPOINT,
        azure_key=AZURE_KEY,
        room_name_keywords=ROOM_NAME_KEYWORDS,
    )
