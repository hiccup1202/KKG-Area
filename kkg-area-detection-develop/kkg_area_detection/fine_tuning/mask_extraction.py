#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET

import cv2
import numpy as np
from svg.path import parse_path

from .svg_utils import parse_transform, transform_point


def extract_object_masks(svg_file, output_mask_file, output_color_mask_file=None):
    """
    SVGファイルから全ての要素（path、polygon、rectなど）を対象にマスクを抽出する

    Args:
        svg_file: SVGファイルのパス
        output_mask_file: オブジェクト部分が255、その他は0の2値マスク
        output_color_mask_file: オブジェクトのインスタンスごとに異なる色を付けたカラー版マスク（オプション）

    Returns:
        tuple: (object_mask, object_instance_mask)
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # SVGのサイズを取得。属性がなければスキップ
    if 'width' not in root.attrib or 'height' not in root.attrib:
        print(f'SVGのサイズが指定されていないため、スキップ: {svg_file}')
        return None, None

    width = int(float(root.attrib['width']))
    height = int(float(root.attrib['height']))

    # マスクの初期化
    object_mask = np.zeros((height, width), dtype=np.uint8)
    object_instance_mask = np.zeros((height, width), dtype=np.uint16)
    instance_id = 1

    def process_element(element, parent_class, parent_style, current_transform):
        """グループ以外の要素を処理してマスクを描画する"""
        nonlocal instance_id

        # display:none を持つ要素を無視したい場合は下記を参照
        # style_str = (element.attrib.get("style", "") + parent_style).replace(" ", "")
        # if "display:none" in style_str:
        #    return

        tag = element.tag.split('}')[-1]

        if tag == 'path':
            d = element.attrib.get('d', '')
            if d:
                try:
                    path_obj = parse_path(d)
                    pts = []
                    for segment in path_obj:
                        seg_length = segment.length(error=1e-5)
                        num_samples = max(10, int(seg_length))
                        for t in np.linspace(0, 1, num_samples, endpoint=False):
                            pt = segment.point(t)
                            pts.append((float(pt.real), float(pt.imag)))

                    if pts:
                        # パスがZで閉じられている場合に最初の頂点を最後にも追加
                        if d.strip()[-1].upper() == 'Z':
                            pts.append(pts[0])
                        else:
                            pt = path_obj[-1].point(1.0)
                            pts.append((float(pt.real), float(pt.imag)))

                        # 変換行列を適用
                        pts_transformed = [
                            transform_point(x, y, current_transform) for (x, y) in pts
                        ]
                        pts_array = np.array(pts_transformed, dtype=np.int32)

                        # fill属性がnoneでない場合のみ塗りつぶす
                        fill_attr = element.attrib.get('fill', '')
                        if not (fill_attr and fill_attr.lower() == 'none'):
                            cv2.fillPoly(object_mask, [pts_array], 255)
                            cv2.fillPoly(object_instance_mask, [pts_array], instance_id)
                            instance_id += 1
                except Exception as e:
                    print(f'パスの処理中にエラー: {e}')

        elif tag == 'polygon':
            points_str = element.attrib.get('points', '')
            try:
                point_pairs = points_str.strip().split()
                pts = []
                for pair in point_pairs:
                    coords = pair.strip().split(',')
                    if len(coords) >= 2:
                        pts.append((float(coords[0]), float(coords[1])))
                if pts:
                    pts_transformed = [
                        transform_point(x, y, current_transform) for (x, y) in pts
                    ]
                    pts_array = np.array(pts_transformed, dtype=np.int32)
                    cv2.fillPoly(object_mask, [pts_array], 255)
                    cv2.fillPoly(object_instance_mask, [pts_array], instance_id)
                    instance_id += 1
            except Exception as e:
                print(f'ポリゴンの処理中にエラー: {e}')

        elif tag == 'rect':
            try:
                x = float(element.attrib.get('x', '0'))
                y = float(element.attrib.get('y', '0'))
                w = float(element.attrib.get('width', '0'))
                h = float(element.attrib.get('height', '0'))
                rect_pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                pts_transformed = [
                    transform_point(px, py, current_transform) for (px, py) in rect_pts
                ]
                pts_array = np.array(pts_transformed, dtype=np.int32)
                cv2.fillPoly(object_mask, [pts_array], 255)
                cv2.fillPoly(object_instance_mask, [pts_array], instance_id)
                instance_id += 1
            except Exception as e:
                print(f'矩形の処理中にエラー: {e}')

    def process_group(group, current_transform, parent_style):
        """グループ要素を再帰的に処理する"""
        style = group.attrib.get('style', '') + parent_style
        group_class = group.attrib.get('class', '')

        if 'transform' in group.attrib:
            T = parse_transform(group.attrib['transform'])
            current_transform = current_transform @ T

        for child in group:
            tag = child.tag.split('}')[-1]
            if tag == 'g':
                process_group(child, current_transform, style)
            else:
                process_element(child, group_class, style, current_transform)

    # ルート要素から処理開始
    identity = np.eye(3)
    for child in root:
        tag = child.tag.split('}')[-1]
        if tag == 'g':
            process_group(child, identity, '')
        else:
            process_element(child, '', '', identity)

    # 2値マスクを保存
    cv2.imwrite(output_mask_file, object_mask)

    # カラーマスクを作成（オプション）
    if output_color_mask_file:
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        unique_ids = np.unique(object_instance_mask)
        # 0は背景なので除外
        unique_ids = unique_ids[unique_ids != 0]

        np.random.seed(42)
        for uid in unique_ids:
            # 黒色系を避けるランダム色を生成
            while True:
                color = tuple(map(int, np.random.randint(0, 256, 3)))
                # 黒色系を避ける (全ての値が50未満)
                if not (color[0] < 50 and color[1] < 50 and color[2] < 50):
                    break
            mask = object_instance_mask == uid
            color_mask[mask] = color

        # カラーマスクを保存
        cv2.imwrite(output_color_mask_file, color_mask)

    return object_mask, object_instance_mask


def extract_room_wall_door_window_masks(
    svg_file, output_color_file, output_annotation_file=None
):
    """
    SVGファイルからカラーマスクを作成する:
    - 壁は同一のインスタンスID
    - 部屋はそれぞれ固有のインスタンスID
    - ドアと窓はそれぞれ固有のクラスIDとインスタンスID=0

    Args:
        svg_file: SVGファイルのパス
        output_color_file: カラーマスクの保存先パス
        output_annotation_file: アノテーションマスクの保存先パス（オプション）

    Returns:
        tuple: (color_mask, instance_mask, class_mask)
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # SVGのサイズを取得。属性がなければスキップ
    if 'width' not in root.attrib or 'height' not in root.attrib:
        print(f'SVGのサイズが指定されていないため、スキップ: {svg_file}')
        return None, None, None

    width = int(float(root.attrib['width']))
    height = int(float(root.attrib['height']))

    # インスタンスID管理用のマスク（16bitにしておくと余裕がある）
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    # クラスID管理用のマスク
    class_mask = np.zeros((height, width), dtype=np.uint8)

    # 各要素の固定ID
    WALL_ID = 0  # 壁のインスタンスID
    DOOR_ID = 0  # ドアのインスタンスID
    WINDOW_ID = 0  # 窓のインスタンスID
    # 部屋のインスタンスID開始番号
    current_room_id = 0  # 0からスタート

    # クラスIDの定義
    ROOM_CLASS = 0
    DOOR_CLASS = 1
    WINDOW_CLASS = 2
    WALL_CLASS = 3

    # 要素カウント
    wall_count = 0
    door_count = 0
    window_count = 0

    # クラス名の判定用
    wall_keywords = {'Wall', 'Walls', 'Wall External'}
    room_keywords = {'Room', 'LivingRoom', 'Space', 'SpaceDimensionsLabel'}
    door_keywords = {'Door'}
    window_keywords = {'Window', 'Windows'}

    def fill_polygon(pts, instance_id, class_id):
        """頂点配列 pts をinstance_idとclass_idで塗りつぶす"""
        pts_array = np.array(pts, dtype=np.int32)
        cv2.fillPoly(instance_mask, [pts_array], instance_id)
        cv2.fillPoly(class_mask, [pts_array], class_id)

    def process_element(element, inherited_classes, parent_style, current_transform):
        nonlocal current_room_id
        nonlocal wall_count, door_count, window_count
        element_class_str = element.attrib.get('class', '')
        element_classes = set(element_class_str.split()) if element_class_str else set()
        all_classes = inherited_classes | element_classes

        # Debug: Print element info
        #  Element ---")
        # print(f"  Tag: {element.tag.split('}')[-1]}, Attrib: {element.attrib}")
        # print(f"  Combined Classes: {all_classes}")

        # 要素の種類を判定
        is_wall = any(token in all_classes for token in wall_keywords) and not (
            any(token in all_classes for token in door_keywords)
            or any(token in all_classes for token in window_keywords)
        )
        is_room = any(token in all_classes for token in room_keywords)
        is_door = any(token in all_classes for token in door_keywords)
        is_window = any(token in all_classes for token in window_keywords)

        # Refine room classification: Not a room if it's also a wall/door/window
        is_room = is_room and not (is_wall or is_door or is_window)

        # Debug: Print classification results
        # print(f"  Is Wall: {is_wall}, Is Room: {is_room}, Is Door: {is_door}, Is Window: {is_window}")

        # 対象の要素でない場合はスキップ
        if not (is_wall or is_room or is_door or is_window):
            # print(f"  Skipping element (not a target type).")
            return

        # インスタンスIDとクラスIDを設定
        if is_wall:
            instance_id = WALL_ID
            class_id = WALL_CLASS
            wall_count += 1
        elif is_door:
            instance_id = DOOR_ID
            class_id = DOOR_CLASS
            door_count += 1
        elif is_window:
            instance_id = WINDOW_ID
            class_id = WINDOW_CLASS
            window_count += 1
        else:  # is_room
            instance_id = current_room_id
            class_id = ROOM_CLASS

        # Debug: Print assigned IDs
        # print(f"  Assigned Instance ID: {instance_id}, Class ID: {class_id}")

        tag = element.tag.split('}')[-1]
        d = None

        if tag == 'path':
            d = element.attrib.get('d', '')
            if d:
                try:
                    path_obj = parse_path(d)
                    pts = []
                    for segment in path_obj:
                        seg_length = segment.length(error=1e-5)
                        num_samples = max(10, int(seg_length))
                        for t in np.linspace(0, 1, num_samples, endpoint=False):
                            pt = segment.point(t)
                            pts.append((float(pt.real), float(pt.imag)))

                    if pts:
                        # パスがZで閉じられている場合に最初の頂点を最後にも追加
                        if d.strip()[-1].upper() == 'Z':
                            pts.append(pts[0])
                        else:
                            pt = path_obj[-1].point(1.0)
                            pts.append((float(pt.real), float(pt.imag)))

                        # 変換行列を適用
                        pts_transformed = [
                            transform_point(x, y, current_transform) for (x, y) in pts
                        ]
                        fill_polygon(pts_transformed, instance_id, class_id)
                        # Debug: Confirm fill_polygon call
                        # print(f"  Called fill_polygon for path (Class ID: {class_id}, Instance ID: {instance_id})")
                        # if is_window:
                        #     print("    >> Filled WINDOW Path")

                except Exception as e:
                    print(f'Error processing path: {e}')

        elif tag == 'polygon':
            points_str = element.attrib.get('points', '')
            try:
                point_pairs = points_str.strip().split()
                pts = []
                for pair in point_pairs:
                    coords = pair.strip().split(',')
                    if len(coords) >= 2:
                        pts.append((float(coords[0]), float(coords[1])))
                if pts:
                    pts_transformed = [
                        transform_point(x, y, current_transform) for (x, y) in pts
                    ]
                    fill_polygon(pts_transformed, instance_id, class_id)
                    # Debug: Confirm fill_polygon call
                    # print(f"  Called fill_polygon for polygon (Class ID: {class_id}, Instance ID: {instance_id})")
                    # if is_window:
                    #     print("    >> Filled WINDOW Polygon")
            except Exception as e:
                print(f'Error processing polygon: {e}')

        elif tag == 'rect':
            try:
                x = float(element.attrib.get('x', '0'))
                y = float(element.attrib.get('y', '0'))
                w = float(element.attrib.get('width', '0'))
                h = float(element.attrib.get('height', '0'))
                rect_pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                pts_transformed = [
                    transform_point(px, py, current_transform) for (px, py) in rect_pts
                ]
                fill_polygon(pts_transformed, instance_id, class_id)
                # Debug: Confirm fill_polygon call
                # print(f"  Called fill_polygon for rect (Class ID: {class_id}, Instance ID: {instance_id})")
                # if is_window:
                #     print("    >> Filled WINDOW Rect")
            except Exception as e:
                print(f'Error processing rect: {e}')

        # 部屋なら塗り終わった後にIDをインクリメント
        if is_room:
            current_room_id += 1

    def process_group(group, current_transform, parent_style, inherited_classes):
        if 'transform' in group.attrib:
            T = parse_transform(group.attrib['transform'])
            current_transform = current_transform @ T

        group_class_str = group.attrib.get('class', '')
        current_classes = (
            inherited_classes | set(group_class_str.split())
            if group_class_str
            else inherited_classes
        )
        style = group.attrib.get('style', '') + parent_style

        for child in group:
            tag = child.tag.split('}')[-1]
            if tag == 'g':
                process_group(child, current_transform, style, current_classes)
            else:
                process_element(child, current_classes, style, current_transform)

    # ルート要素から処理開始
    identity = np.eye(3)
    for child in root:
        tag = child.tag.split('}')[-1]
        if tag == 'g':
            process_group(child, identity, '', set())
        else:
            process_element(child, set(), '', identity)

    # カラーマスクを作成
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # 1ピクセルしかない色を修正する処理を追加
    def fix_single_pixel_colors(mask_image):
        """1ピクセルしかない色を周囲の色の多数決で置き換える"""
        height, width, _ = mask_image.shape

        # 各色のピクセル数をカウント
        reshaped_colors = mask_image.reshape(-1, 3)
        unique_colors, counts = np.unique(reshaped_colors, axis=0, return_counts=True)

        # 1ピクセルしかない色を特定
        single_pixel_colors = []
        for i, count in enumerate(counts):
            if count == 1:
                single_pixel_colors.append(tuple(unique_colors[i]))

        # 1ピクセルしかない色がなければ終了
        if not single_pixel_colors:
            return mask_image

        # 元の画像のコピーを作成
        fixed_mask = mask_image.copy()

        # 1ピクセルしかない色を四方の色の多数決で置き換える
        for i in range(height):
            for j in range(width):
                pixel_color = tuple(mask_image[i, j])
                if pixel_color in single_pixel_colors:
                    # 四方の色を取得
                    neighbors = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbors.append(tuple(mask_image[ni, nj]))

                    # 最も頻度の高い色を選択
                    if neighbors:
                        neighbor_counts = {}
                        for color in neighbors:
                            if color not in single_pixel_colors:  # 1ピクセル色は除外
                                neighbor_counts[color] = (
                                    neighbor_counts.get(color, 0) + 1
                                )

                        if neighbor_counts:
                            most_common_color = max(
                                neighbor_counts.items(), key=lambda x: x[1]
                            )[0]
                            fixed_mask[i, j] = most_common_color

        return fixed_mask

    # 各クラスの固定色
    WALL_COLOR = (255, 255, 255)  # 壁は白
    DOOR_COLOR = (255, 0, 0)  # ドアは青 (BGR)
    WINDOW_COLOR = (0, 165, 255)  # 窓はオレンジ (BGR)

    # ランダム色生成
    np.random.seed(42)

    def random_color_avoid(avoid_colors):
        """指定された色と被らないランダム色を返す"""
        while True:
            c = tuple(map(int, np.random.randint(0, 256, 3)))
            # 青色系（ドア）とオレンジ色系（窓）と黒色系を避ける
            # 青色系: B値が高く、R値とG値が低い
            # オレンジ色系: R値とG値が高く、B値が低い
            # 黒色系: 全ての値が低い
            if c not in avoid_colors and not (
                # 青色系を避ける (B > 200, R < 100, G < 100)
                (c[0] > 200 and c[1] < 100 and c[2] < 100)
                or
                # オレンジ色系を避ける (R > 200, G > 100, B < 100)
                (c[0] < 100 and c[1] > 100 and c[2] > 200)
                or
                # 黒色系を避ける (全ての値が50未満)
                (c[0] < 50 and c[1] < 50 and c[2] < 50)
            ):
                return c

    # 固定色のリスト
    fixed_colors = [WALL_COLOR, DOOR_COLOR, WINDOW_COLOR]

    # カラーマスク生成順序: 部屋 -> ドア -> 窓 -> 壁
    # 1. まず部屋にランダム色を適用
    processed_room_ids = set()
    for room_id in range(current_room_id):
        if room_id == 0:
            continue
        room_mask_pixels = (instance_mask == room_id) & (class_mask == ROOM_CLASS)
        if np.any(room_mask_pixels):
            c = random_color_avoid(fixed_colors)
            color_mask[room_mask_pixels] = c
            fixed_colors.append(c)
            processed_room_ids.add(room_id)

    # 2. ドア、窓、壁に固定色を適用
    color_mask[class_mask == DOOR_CLASS] = DOOR_COLOR
    color_mask[class_mask == WINDOW_CLASS] = WINDOW_COLOR

    # まだ色が付いていない壁ピクセルを見つける
    wall_pixels_mask = class_mask == WALL_CLASS
    black_pixels_mask = np.all(color_mask == [0, 0, 0], axis=2)
    target_wall_pixels = wall_pixels_mask & black_pixels_mask

    # 壁の色を適用
    color_mask[target_wall_pixels] = WALL_COLOR

    # 1ピクセルしかない色を修正
    color_mask = fix_single_pixel_colors(color_mask)

    # カラーマスクを保存
    cv2.imwrite(output_color_file, color_mask)

    # アノテーションマスクを作成（オプション）
    if output_annotation_file:
        annotation_mask = np.zeros((height, width, 3), dtype=np.uint8)

        # 各ピクセルに対して[0, instance_id, class_id]を設定
        for y in range(height):
            for x in range(width):
                if instance_mask[y, x] > 0 or class_mask[y, x] > 0:
                    annotation_mask[y, x, 0] = 0
                    annotation_mask[y, x, 1] = instance_mask[y, x]
                    annotation_mask[y, x, 2] = class_mask[y, x]

        # 1ピクセルしかない色を修正（アノテーションマスクにも適用）
        annotation_mask = fix_single_pixel_colors(annotation_mask)

        # アノテーションマスクを保存
        cv2.imwrite(output_annotation_file, annotation_mask)

    return color_mask, instance_mask, class_mask
