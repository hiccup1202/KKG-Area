import cv2
import numpy as np
from symbols import _draw_symbol

symbol_types = [
    'circle', 'double_circle', 'filled_circle', 'circle_cross', 'circle_plus',
    'square', 'double_square', 'filled_square', 'square_cross', 'square_plus',
    'cross', 'plus'
]

for symbol_type in symbol_types:
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255
    _draw_symbol(img, (32, 32), symbol_type, radius=20, color=(0, 0, 0), thickness=2)
    cv2.imwrite(f'test_symbol_{symbol_type}.png', img)
    print(f"Saved: test_symbol_{symbol_type}.png")
