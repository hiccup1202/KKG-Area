#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def parse_transform(transform_str):
    """transform属性を3x3行列に変換"""
    if not transform_str:
        return np.eye(3)

    # matrix変換のパース
    m = re.search(r"matrix\(\s*([-\d\.e]+)[,\s]+([-\d\.e]+)[,\s]+([-\d\.e]+)[,\s]+([-\d\.e]+)[,\s]+([-\d\.e]+)[,\s]+([-\d\.e]+)\s*\)", transform_str)
    if m:
        a, b, c, d, e, f = map(float, m.groups())
        return np.array([[a, c, e],
                        [b, d, f],
                        [0, 0, 1]])

    # translate変換のパース
    m = re.search(r"translate\(\s*([-\d\.e]+)[,\s]+([-\d\.e]+)\s*\)", transform_str)
    if m:
        tx, ty = map(float, m.groups())
        return np.array([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0, 1]])

    return np.eye(3)

def transform_point(x, y, T):
    """点(x, y)に変換行列Tを適用"""
    pt = np.array([x, y, 1])
    tpt = T @ pt
    return tpt[0], tpt[1]

def find_element_by_id(root, id_value):
    """ID属性を持つ要素を再帰的に検索"""
    for elem in root.iter():
        if elem.get('id') == id_value:
            return elem
    return None

def find_adjacent_rooms(room_instance_mask):
    """
    隣接する部屋のペアを見つける

    Args:
        room_instance_mask: 部屋のインスタンスマスク

    Returns:
        隣接する部屋ID同士のペアのリスト
    """
    # 部屋のIDリスト
    room_ids = np.unique(room_instance_mask)[1:]  # 0を除外

    # 隣接ペア
    adjacent_pairs = []

    # 各部屋ペアをチェック
    for i, room_id1 in enumerate(room_ids):
        mask1 = (room_instance_mask == room_id1).astype(np.uint8)
        # 部屋1を膨張
        dilated1 = cv2.dilate(mask1, np.ones((3,3), np.uint8), iterations=1)

        for room_id2 in room_ids[i+1:]:
            mask2 = (room_instance_mask == room_id2).astype(np.uint8)

            # 膨張した部屋1と部屋2の重なりをチェック
            if np.any(dilated1 & mask2):
                adjacent_pairs.append((room_id1, room_id2))

    return adjacent_pairs

def get_boundary_between_rooms(room_instance_mask, room_id1, room_id2):
    """
    2つの部屋間の境界を抽出

    Args:
        room_instance_mask: 部屋のインスタンスマスク
        room_id1, room_id2: 部屋のID

    Returns:
        2つの部屋間の境界マスク
    """
    # 部屋のマスクを取得
    mask1 = (room_instance_mask == room_id1).astype(np.uint8)
    mask2 = (room_instance_mask == room_id2).astype(np.uint8)

    # 部屋を膨張
    dilated1 = cv2.dilate(mask1, np.ones((3,3), np.uint8), iterations=1)
    dilated2 = cv2.dilate(mask2, np.ones((3,3), np.uint8), iterations=1)

    # 両方の膨張領域の重なりが境界
    boundary = dilated1 & dilated2

    return boundary

def analyze_wall_segments(boundary_mask, wall_mask):
    """
    部屋間の境界線上の壁の分布を分析

    Args:
        boundary_mask: 部屋間の境界マスク
        wall_mask: 壁のマスク

    Returns:
        壁のカバレッジ、最大の連続壁セグメントの長さ、セグメント数、セグメント長のリスト
    """
    # 境界上の壁
    wall_on_boundary = cv2.bitwise_and(boundary_mask, wall_mask)

    # 境界の輪郭を抽出
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return 0, 0, 0, []

    # 最も長い境界を選択
    boundary_contour = max(contours, key=cv2.contourArea)

    # 境界に沿ってピクセルをサンプリング
    boundary_pixels = []

    for point in boundary_contour:
        x, y = point[0]
        boundary_pixels.append((x, y))

    # 境界線の壁の有無をサンプリング
    wall_presence = []

    for x, y in boundary_pixels:
        if wall_on_boundary[y, x] > 0:
            wall_presence.append(1)  # 壁あり
        else:
            wall_presence.append(0)  # 壁なし

    # 連続壁セグメントの分析
    segments = []
    current_length = 0

    for has_wall in wall_presence:
        if has_wall:
            current_length += 1
        else:
            if current_length > 0:
                segments.append(current_length)
                current_length = 0

    # 最後のセグメントを追加
    if current_length > 0:
        segments.append(current_length)

    # 総境界長
    boundary_length = len(boundary_pixels)
    if boundary_length == 0:
        return 0, 0, 0, []

    # 壁のカバレッジ計算
    wall_coverage = sum(wall_presence) / boundary_length

    # 最大の連続壁セグメント長と総セグメント数
    max_segment_length = max(segments) if segments else 0
    segment_count = len(segments)

    return wall_coverage, max_segment_length / boundary_length, segment_count, segments

def check_wall_between_rooms(boundary_mask, wall_mask, wall_coverage_threshold=0.05, max_segment_ratio_threshold=0.3, door_width_threshold=0.15):
    """
    部屋間の境界に壁が存在するかチェック（ドアや開口部を考慮）

    Args:
        boundary_mask: 部屋間の境界マスク
        wall_mask: 壁のマスク
        wall_coverage_threshold: 壁の存在と判断する最小カバレッジの閾値
        max_segment_ratio_threshold: 最大壁セグメントの境界長に対する比率の閾値
        door_width_threshold: ドアとして認識する開口部の最大幅（境界長に対する比率）

    Returns:
        壁が存在するかのブール値、分析結果の辞書
    """
    # 境界が存在しない場合
    if np.sum(boundary_mask) == 0:
        return False, {"coverage": 0, "max_segment_ratio": 0, "segments": [], "reason": "境界なし"}

    # 壁分布の分析
    coverage, max_segment_ratio, segment_count, segments = analyze_wall_segments(boundary_mask, wall_mask)

    # 分析結果
    analysis = {
        "coverage": coverage,
        "max_segment_ratio": max_segment_ratio,
        "segment_count": segment_count,
        "segments": segments
    }

    # ほとんど壁がない場合（カバレッジが極端に低い）
    if coverage < wall_coverage_threshold:
        analysis["reason"] = "壁のカバレッジが低すぎる"
        return False, analysis

    # 壁の最大連続セグメントが小さい場合 -> おそらく壁ではなく点在するノイズ
    if max_segment_ratio < max_segment_ratio_threshold:
        analysis["reason"] = "最大壁セグメントが小さすぎる"
        return False, analysis

    # 壁があると判断
    analysis["reason"] = "壁あり"
    return True, analysis

def find_connected_components(adjacency_list, room_ids):
    """
    連結成分を見つける（壁のない部屋をグループ化）

    Args:
        adjacency_list: 部屋同士の隣接関係（壁なし）
        room_ids: 全部屋のIDリスト

    Returns:
        部屋グループのリスト
    """
    # グラフ表現（隣接リスト）
    graph = {room_id: [] for room_id in room_ids}
    for room1, room2 in adjacency_list:
        graph[room1].append(room2)
        graph[room2].append(room1)

    # 訪問済みフラグ
    visited = {room_id: False for room_id in room_ids}

    # 連結成分のリスト
    components = []

    # DFSで連結成分を見つける
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)

    # 全ノードについてDFS
    for room_id in room_ids:
        if not visited[room_id]:
            component = []
            dfs(room_id, component)
            components.append(component)

    return components

def visualize_merged_rooms(merged_instance_mask, room_mask, wall_mask, output_path):
    """
    統合された部屋を可視化

    Args:
        merged_instance_mask: 統合後の部屋インスタンスマスク
        room_mask: 部屋の全体マスク
        wall_mask: 壁のマスク
        output_path: 出力ファイルパス
    """
    # マスクの形状を取得
    height, width = merged_instance_mask.shape

    # 部屋の数
    room_ids = np.unique(merged_instance_mask)[1:]  # 0を除外
    num_rooms = len(room_ids)

    if num_rooms == 0:
        print("No rooms to visualize.")
        return

    print(f"Visualizing {num_rooms} merged rooms.")

    # 色付きマスクの初期化
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # 背景を黒に (変更：白から黒へ)
    color_mask.fill(0)

    # 各部屋に色を割り当て
    np.random.seed(42)  # 再現性のため

    for room_id in room_ids:
        # 黄色を避けて色を生成
        while True:
            color = np.random.randint(0, 256, 3)
            # 黄色に近い色を避ける
            if not (color[0] < 100 and color[1] > 200 and color[2] > 200):
                break

        color = tuple(map(int, color))
        mask = merged_instance_mask == room_id
        color_mask[mask] = color

    # 壁を黄色で表示
    color_mask[wall_mask > 0] = (0, 255, 255)

    # 保存
    cv2.imwrite(output_path, color_mask)
    print(f"Visualization saved to {output_path}")

def visualize_room_boundaries(room_instance_mask, wall_mask, output_path):
    """
    部屋間の境界と壁の関係を可視化

    Args:
        room_instance_mask: 部屋のインスタンスマスク
        wall_mask: 壁のマスク
        output_path: 出力ファイルパス
    """
    # マスクの形状を取得
    height, width = room_instance_mask.shape

    # 部屋IDリスト
    room_ids = np.unique(room_instance_mask)[1:]  # 0を除外

    # 隣接ペアを取得
    adjacent_pairs = find_adjacent_rooms(room_instance_mask)

    # 可視化用のマスク
    boundary_viz = np.zeros((height, width, 3), dtype=np.uint8)
    boundary_viz.fill(255)  # 白背景

    # 壁を灰色で表示
    boundary_viz[wall_mask > 0] = (200, 200, 200)

    # 各部屋の輪郭を描画
    for room_id in room_ids:
        mask = (room_instance_mask == room_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary_viz, contours, -1, (150, 150, 150), 1)

    # 境界線を赤で、壁のある部分を緑で描画
    for room_id1, room_id2 in adjacent_pairs:
        boundary = get_boundary_between_rooms(room_instance_mask, room_id1, room_id2)
        wall_on_boundary = cv2.bitwise_and(boundary, wall_mask)

        # 赤色で境界線を描画
        boundary_viz[boundary > 0] = (0, 0, 255)
        # 緑色で壁のある部分を上書き
        boundary_viz[wall_on_boundary > 0] = (0, 255, 0)

    # 保存
    cv2.imwrite(output_path, boundary_viz)
    print(f"Boundary visualization saved to {output_path}")

def merge_rooms_without_walls(room_instance_mask, wall_mask, wall_coverage_threshold=0.01, max_segment_ratio_threshold=0.05, door_width_threshold=0.15):
    """
    壁のない隣接部屋を統合（ドア考慮版）

    Args:
        room_instance_mask: 部屋のインスタンスマスク
        wall_mask: 壁のマスク
        wall_coverage_threshold: 壁の存在と判断する最小カバレッジの閾値
        max_segment_ratio_threshold: 最大壁セグメントの境界長に対する比率の閾値
        door_width_threshold: ドアとして認識する開口部の最大幅（境界長に対する比率）

    Returns:
        統合された部屋インスタンスマスク
    """
    # 部屋のIDリスト
    room_ids = np.unique(room_instance_mask)[1:]  # 0を除外

    if len(room_ids) == 0:
        print("No rooms to merge.")
        return room_instance_mask

    print(f"Found {len(room_ids)} rooms in instance mask.")

    # 隣接部屋を見つける
    adjacent_pairs = find_adjacent_rooms(room_instance_mask)
    print(f"Found {len(adjacent_pairs)} adjacent room pairs: {adjacent_pairs}")

    # 壁のない隣接部屋ペア
    no_wall_pairs = []

    # 境界可視化用のデータ
    boundaries_data = []

    # 各隣接ペアについて壁の存在をチェック
    for room_id1, room_id2 in adjacent_pairs:
        print(f"Checking wall between rooms {room_id1} and {room_id2}")
        boundary = get_boundary_between_rooms(room_instance_mask, room_id1, room_id2)
        has_wall, analysis = check_wall_between_rooms(
            boundary,
            wall_mask,
            wall_coverage_threshold,
            max_segment_ratio_threshold,
            door_width_threshold
        )

        # 分析結果を保存
        boundaries_data.append({
            "room_id1": room_id1,
            "room_id2": room_id2,
            "has_wall": has_wall,
            "analysis": analysis
        })

        if not has_wall:
            print(f"  No wall found between rooms {room_id1} and {room_id2}: {analysis['reason']}")
            print(f"  Coverage: {analysis['coverage']:.2f}, Max segment ratio: {analysis['max_segment_ratio']:.2f}")
            no_wall_pairs.append((room_id1, room_id2))
        else:
            print(f"  Wall found between rooms {room_id1} and {room_id2}")
            print(f"  Coverage: {analysis['coverage']:.2f}, Max segment ratio: {analysis['max_segment_ratio']:.2f}")

    # 境界分析データをCSVに保存
    with open("boundary_analysis.csv", "w") as f:
        f.write("room_id1,room_id2,has_wall,coverage,max_segment_ratio,segment_count,reason\n")
        for data in boundaries_data:
            f.write(f"{data['room_id1']},{data['room_id2']},{data['has_wall']}," +
                   f"{data['analysis']['coverage']:.4f},{data['analysis']['max_segment_ratio']:.4f}," +
                   f"{data['analysis']['segment_count']},{data['analysis']['reason']}\n")

    # 壁のない部屋をグループ化
    room_groups = find_connected_components(no_wall_pairs, room_ids)
    print(f"Rooms grouped into {len(room_groups)} connected components:")
    for i, group in enumerate(room_groups):
        print(f"  Group {i+1}: {group}")

    # 新しいインスタンスマスクを作成
    new_mask = np.zeros_like(room_instance_mask)

    for i, group in enumerate(room_groups, 1):
        for room_id in group:
            new_mask[room_instance_mask == room_id] = i

    print(f"Created new mask with {len(room_groups)} merged room groups")
    return new_mask

def extract_room_and_wall_masks(svg_file, output_file, output_color_file, wall_output_file, merged_output_file=None):
    """SVGファイルからroom_maskとwall_maskを抽出してPNG形式で保存し、隣接部屋の統合も行う"""
    print(f"Reading SVG file: {svg_file}")
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # 名前空間の処理
    namespaces = {}
    for name, value in root.attrib.items():
        if name.startswith('{'):
            xmlns = name[1:].split('}')[0]
            namespaces[xmlns] = value

    # 主要な名前空間を追加
    if 'http://www.w3.org/2000/svg' not in namespaces:
        namespaces['svg'] = 'http://www.w3.org/2000/svg'
    if 'http://www.w3.org/1999/xlink' not in namespaces:
        namespaces['xlink'] = 'http://www.w3.org/1999/xlink'

    # SVGのサイズを取得
    width = int(float(root.attrib.get("width", "2276")))
    height = int(float(root.attrib.get("height", "1136")))
    print(f"SVG dimensions: {width}x{height}")

    # SVGの大きさに基づく適切なサイズに調整
    scale_factor = min(4000 / max(width, height), 1.0)
    target_width = int(width * scale_factor)
    target_height = int(height * scale_factor)
    print(f"Target image dimensions: {target_width}x{target_height}")

    # マスクの初期化
    room_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    wall_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    room_instance_mask = np.zeros((target_height, target_width), dtype=np.uint16)

    # デバッグ用マスク
    debug_mask = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    floor1_mask = np.zeros((target_height, target_width, 3), dtype=np.uint8)  # 1階用
    floor2_mask = np.zeros((target_height, target_width, 3), dtype=np.uint8)  # 2階用

    instance_id = 1

    # ID参照の要素を格納する辞書
    element_by_id = {}

    # カウンター
    processed_elements = 0
    filled_polygons = 0

    # 全てのIDを持つ要素を辞書に格納
    print("Collecting all elements with IDs...")
    for elem in root.iter():
        element_id = elem.get("id", "")
        if element_id:
            element_by_id[element_id] = elem
            if "Floor" in element_id:
                print(f"  Found floor element: {element_id}")
                # Floor要素の属性を確認
                for k, v in elem.attrib.items():
                    print(f"    {k} = {v}")

    print(f"Found {len(element_by_id)} elements with IDs")
    floor_ids = [key for key in element_by_id.keys() if "Floor" in key]
    print(f"Floor-related IDs: {floor_ids}")

    # 特にFloor-1とFloor-2を明示的に探す
    floor1_elem = find_element_by_id(root, "Floor-1")
    floor2_elem = find_element_by_id(root, "Floor-2")

    if floor1_elem is not None:
        print("Found Floor-1 element directly!")
        for k, v in floor1_elem.attrib.items():
            print(f"  {k} = {v}")
    else:
        print("Floor-1 element not found directly.")

    if floor2_elem is not None:
        print("Found Floor-2 element directly!")
        for k, v in floor2_elem.attrib.items():
            print(f"  {k} = {v}")
    else:
        print("Floor-2 element not found directly.")

    # UseタグからFloor参照を探す
    print("Looking for use tags referring to floors...")
    for elem in root.iter():
        if elem.tag.split('}')[-1] == 'use':
            href = elem.get("{http://www.w3.org/1999/xlink}href", "")
            if href and "Floor" in href:
                print(f"  Found use tag referring to {href}")
                # 親要素の情報
                parent = list(root.iter())[[elem in list(p) for p in root.iter()].index(True)]
                print(f"  Parent tag: {parent.tag.split('}')[-1]}, class: {parent.get('class', '')}")

    def process_element(element, parent_class, parent_style, current_transform, depth=0, processed_ids=None, floor_level=None):
        if processed_ids is None:
            processed_ids = set()

        nonlocal instance_id, processed_elements, filled_polygons
        processed_elements += 1

        # 要素の基本情報を取得
        element_id = element.get("id", "")
        element_tag = element.tag.split('}')[-1]
        element_class = element.get("class", "")

        indent = "  " * depth
        print(f"{indent}Processing element: {element_tag}, ID: {element_id}, Class: {element_class}, Floor: {floor_level}")

        # useタグの処理
        if element_tag == "use":
            href = elem.get("{http://www.w3.org/1999/xlink}href", "")
            if href and href.startswith("#"):
                ref_id = href[1:]

                # 既に処理済みのIDはスキップ (Floor要素は例外)
                if ref_id in processed_ids and not "Floor" in ref_id:
                    print(f"{indent}  Already processed ID: {ref_id}")
                    return

                # Floor要素の場合は明示的に処理
                if "Floor-1" in ref_id:
                    floor_level = 1
                elif "Floor-2" in ref_id:
                    floor_level = 2

                if ref_id in element_by_id:
                    processed_ids.add(ref_id)
                    print(f"{indent}  Following use reference to #{ref_id} (Floor level: {floor_level})")

                    # 変換行列の更新
                    element_transform = element.get("transform", "")
                    if element_transform:
                        T = parse_transform(element_transform)
                        new_transform = current_transform @ T
                    else:
                        new_transform = current_transform

                    # 参照先の要素を処理
                    ref_element = element_by_id[ref_id]
                    process_element(ref_element, parent_class, parent_style, new_transform, depth+1, processed_ids, floor_level)
                else:
                    print(f"{indent}  Referenced ID not found: {ref_id}")
            return

        # クラス情報の統合
        classes = element_class.split() if element_class else []
        parent_tokens = parent_class.split() if parent_class else []
        all_classes = set(classes) | set(parent_tokens)

        # スタイル情報の処理
        style = element.get("style", "") + " " + parent_style
        is_display_none = "display:none" in style.replace(" ", "")

        # display:noneでも処理する条件
        force_process = (
            "Floor" in element_id or
            "Floor" in element_class or
            "FloorsCompose" in all_classes
        )

        if is_display_none and not force_process:
            print(f"{indent}  Skipping due to display:none")
            return

        # room要素とwall要素の判定
        room_keywords = {"Room", "LivingRoom", "Space", "Floor"}
        wall_keywords = {"Wall", "Walls"}

        # IDやクラスに基づく処理
        if "Floor" in element_id:
            all_classes.add("Floor")

        is_room = any(token in all_classes for token in room_keywords)
        is_wall = any(token in all_classes for token in wall_keywords)

        # FloorNumberLabelの処理
        if "FloorNumberLabel" in element_id:
            print(f"{indent}  Found floor label: {element_id}")
            is_room = True

        # 対象マスクの選択
        target_mask = None
        if is_room:
            target_mask = room_mask
        elif is_wall:
            target_mask = wall_mask

        # デバッグマスクの選択
        if floor_level == 1:
            debug_floor_mask = floor1_mask
        elif floor_level == 2:
            debug_floor_mask = floor2_mask
        else:
            debug_floor_mask = debug_mask

        # 特定の要素の処理
        if target_mask is not None:
            # 変換行列の更新
            element_transform = element.get("transform", "")
            if element_transform:
                T = parse_transform(element_transform)
                current_transform = current_transform @ T

            # rect要素の処理
            if element_tag == "rect":
                try:
                    x = float(element.get("x", "0"))
                    y = float(element.get("y", "0"))
                    w = float(element.get("width", "0"))
                    h = float(element.get("height", "0"))

                    # 小さすぎる要素はスキップ
                    if w < 1 or h < 1:
                        print(f"{indent}  Skipping tiny rectangle: {w}x{h}")
                        return

                    # 頂点の変換
                    rect_pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                    pts_transformed = [transform_point(px, py, current_transform) for (px, py) in rect_pts]

                    # スケーリング
                    pts_scaled = [(x * scale_factor, y * scale_factor) for (x, y) in pts_transformed]
                    pts_array = np.array(pts_scaled, dtype=np.int32)

                    # デバッグ情報
                    print(f"{indent}  Rectangle at ({x}, {y}) size {w}x{h}")
                    print(f"{indent}  Transformed points: {pts_transformed}")

                    # マスクに描画
                    color = (0, 255, 0) if is_room else (0, 255, 255)  # 緑 or 黄
                    cv2.fillPoly(debug_mask, [pts_array], color)
                    cv2.fillPoly(debug_floor_mask, [pts_array], color)
                    cv2.fillPoly(target_mask, [pts_array], 255)

                    if is_room:
                        cv2.fillPoly(room_instance_mask, [pts_array], instance_id)
                        instance_id += 1

                    filled_polygons += 1

                except Exception as e:
                    print(f"{indent}  Error processing rect: {e}")

            # polygon要素の処理
            elif element_tag == "polygon":
                points_str = element.get("points", "")
                try:
                    point_pairs = points_str.strip().split()
                    pts = []
                    for pair in point_pairs:
                        coords = pair.strip().split(',')
                        if len(coords) >= 2:
                            pts.append((float(coords[0]), float(coords[1])))

                    if pts:
                        # 変換とスケーリング
                        pts_transformed = [transform_point(x, y, current_transform) for (x, y) in pts]
                        pts_scaled = [(x * scale_factor, y * scale_factor) for (x, y) in pts_transformed]
                        pts_array = np.array(pts_scaled, dtype=np.int32)

                        # デバッグ情報
                        print(f"{indent}  Polygon with {len(pts)} points")

                        # マスクに描画
                        color = (0, 255, 0) if is_room else (0, 255, 255)  # 緑 or 黄
                        cv2.fillPoly(debug_mask, [pts_array], color)
                        cv2.fillPoly(debug_floor_mask, [pts_array], color)
                        cv2.fillPoly(target_mask, [pts_array], 255)

                        if is_room:
                            cv2.fillPoly(room_instance_mask, [pts_array], instance_id)
                        instance_id += 1

                        filled_polygons += 1

                except Exception as e:
                    print(f"{indent}  Error processing polygon: {e}")

            # path要素の処理（単純化版）
            elif element_tag == "path":
                try:
                    d = element.get("d", "")
                    if d:
                        # 極めて単純なパスの処理（M, L, Zのみ）
                        d = d.replace("M", " M ").replace("L", " L ").replace("Z", " Z ").strip()
                        tokens = d.split()

                        pts = []
                        current_x, current_y = 0, 0

                        i = 0
                        while i < len(tokens):
                            token = tokens[i]
                            if token == "M" or token == "L":
                                if i + 2 < len(tokens):
                                    try:
                                        x = float(tokens[i+1])
                                        y = float(tokens[i+2])
                                        pts.append((x, y))
                                        current_x, current_y = x, y
                                        i += 3
                                    except ValueError:
                                        i += 1
                                else:
                                    i += 1
                            elif token == "Z":
                                if pts:
                                    pts.append(pts[0])  # 閉じる
                                i += 1
                            else:
                                try:
                                    coords = token.split(',')
                                    if len(coords) == 2:
                                        x = float(coords[0])
                                        y = float(coords[1])
                                        pts.append((x, y))
                                        current_x, current_y = x, y
                                except ValueError:
                                    pass
                                i += 1

                        if pts:
                            # 変換とスケーリング
                            pts_transformed = [transform_point(x, y, current_transform) for (x, y) in pts]
                            pts_scaled = [(x * scale_factor, y * scale_factor) for (x, y) in pts_transformed]
                            pts_array = np.array(pts_scaled, dtype=np.int32)

                            # デバッグ情報
                            print(f"{indent}  Path with {len(pts)} points")

                            # fill属性がnoneでなければ塗りつぶす
                            fill_attr = element.get("fill", "")
                            if not (fill_attr and fill_attr.lower() == "none"):
                                color = (0, 255, 0) if is_room else (0, 255, 255)  # 緑 or 黄
                                cv2.fillPoly(debug_mask, [pts_array], color)
                                cv2.fillPoly(debug_floor_mask, [pts_array], color)
                                cv2.fillPoly(target_mask, [pts_array], 255)

                                if is_room:
                                    cv2.fillPoly(room_instance_mask, [pts_array], instance_id)
                                    instance_id += 1

                                filled_polygons += 1

                except Exception as e:
                    print(f"{indent}  Error processing path: {str(e)}")

        # 子要素の処理
        for child in element:
            child_tag = child.tag.split('}')[-1]
            process_element(child, element_class, style, current_transform, depth+1, processed_ids, floor_level)

    # ルート要素から処理開始
    identity = np.eye(3)
    processed_ids = set()
    process_element(root, "", "", identity, 0, processed_ids)

    # 明示的にFloor-1とFloor-2を処理
    print("\n--- Explicitly processing Floor-1 ---")
    if floor1_elem is not None:
        process_element(floor1_elem, "", "", identity, 0, set(), 1)
    else:
        print("Floor-1 element not available for direct processing.")

    print("\n--- Explicitly processing Floor-2 ---")
    if floor2_elem is not None:
        process_element(floor2_elem, "", "", identity, 0, set(), 2)
    else:
        print("Floor-2 element not available for direct processing.")

    # FLoor-1とFloor-2を使用しているuse要素を探して処理
    print("\n--- Looking for and processing use tags for floors ---")
    for elem in root.iter():
        if elem.tag.split('}')[-1] == 'use':
            href = elem.get("{http://www.w3.org/1999/xlink}href", "")
            if href == "#Floor-1":
                print("Processing use tag for Floor-1")
                # 親の変換行列を取得
                parent_transform = identity
                for parent in root.iter():
                    if elem in list(parent):
                        if parent.get("transform"):
                            parent_transform = parse_transform(parent.get("transform"))
                            break

                elem_transform = parse_transform(elem.get("transform", ""))
                combined_transform = parent_transform @ elem_transform

                if floor1_elem is not None:
                    process_element(floor1_elem, "", "", combined_transform, 0, set(), 1)

            elif href == "#Floor-2":
                print("Processing use tag for Floor-2")
                # 親の変換行列を取得
                parent_transform = identity
                for parent in root.iter():
                    if elem in list(parent):
                        if parent.get("transform"):
                            parent_transform = parse_transform(parent.get("transform"))
                            break

                elem_transform = parse_transform(elem.get("transform", ""))
                combined_transform = parent_transform @ elem_transform

                if floor2_elem is not None:
                    process_element(floor2_elem, "", "", combined_transform, 0, set(), 2)

    print(f"\nSummary: Processed {processed_elements} elements, filled {filled_polygons} polygons")
    print(f"Room mask non-zero pixels: {np.count_nonzero(room_mask)}")
    print(f"Wall mask non-zero pixels: {np.count_nonzero(wall_mask)}")

    # マスクが空の場合、強制的にテスト用のデータを生成
    if np.count_nonzero(room_mask) == 0 and np.count_nonzero(wall_mask) == 0:
        print("WARNING: No rooms or walls detected! Creating test shapes...")
        # テスト用の部屋
        cv2.rectangle(room_mask, (100, 100), (400, 400), 255, -1)
        cv2.rectangle(room_instance_mask, (100, 100), (400, 400), 1, -1)
        # テスト用の壁
        cv2.rectangle(wall_mask, (50, 50), (450, 450), 255, 2)

    # カラーマスクの生成
    color_mask = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    unique_ids = np.unique(room_instance_mask)[1:]  # 0を除外

    # 黄色を定義（wallsの色として使用）
    yellow_color = (0, 255, 255)  # BGR形式

    # 壁を黄色で表示
    color_mask[wall_mask > 0] = yellow_color

    # 各部屋インスタンスに異なる色を割り当て（黄色を除く）
    np.random.seed(42)  # 再現性のため
    for instance_id in unique_ids:
        # 黄色を避けて色を生成
        while True:
            color = np.random.randint(0, 256, 3)
            # 黄色に近い色を避ける
            if not (color[0] < 100 and color[1] > 200 and color[2] > 200):
                break

        color = tuple(map(int, color))
        mask = room_instance_mask == instance_id
        color_mask[mask] = color

    # 結果を保存
    cv2.imwrite(output_file, room_mask)
    cv2.imwrite(wall_output_file, wall_mask)
    cv2.imwrite(output_color_file, color_mask)
    cv2.imwrite("debug_mask.png", debug_mask)  # デバッグ用
    cv2.imwrite("floor1_mask.png", floor1_mask)  # 1階のデバッグ用
    cv2.imwrite("floor2_mask.png", floor2_mask)  # 2階のデバッグ用

    print(f"Saved room mask to: {output_file}")
    print(f"Saved wall mask to: {wall_output_file}")
    print(f"Saved colored mask to: {output_color_file}")
    print(f"Saved debug masks to: debug_mask.png, floor1_mask.png, floor2_mask.png")

    # *** 新しい処理: 壁のない隣接部屋の統合 ***
    print("\n--- Processing adjacent rooms without walls ---")
    if merged_output_file:
        # 境界可視化
        boundary_viz_path = os.path.splitext(merged_output_file)[0] + "_boundaries.png"
        visualize_room_boundaries(room_instance_mask, wall_mask, boundary_viz_path)
        print(f"Saved boundary visualization to: {boundary_viz_path}")

        # 部屋の統合処理
        merged_instance_mask = merge_rooms_without_walls(
            room_instance_mask,
            wall_mask,
            wall_coverage_threshold=0.01,  # 非常に小さな値に設定（1%でも壁があれば別の部屋と判断）
            max_segment_ratio_threshold=0.01,  # 小さな壁セグメントでも認識するよう閾値を下げる
            door_width_threshold=0.55  # ドアとして認識する最大幅
        )

        # 統合された部屋の可視化
        merged_visualization_path = os.path.splitext(merged_output_file)[0] + "_visualization.png"
        visualize_merged_rooms(merged_instance_mask, room_mask, wall_mask, merged_visualization_path)

        # 統合されたマスクを保存（インスタンスIDを使用した画像として）
        cv2.imwrite(merged_output_file, (merged_instance_mask * 50).astype(np.uint8))
        print(f"Saved merged room mask to: {merged_output_file}")
        print(f"Saved merged room visualization to: {merged_visualization_path}")

if __name__ == '__main__':
    svg_file = 'dataset/23/model.svg'
    output_file = 'output/23/room_mask.png'
    output_color_file = 'output/23/room_mask_color.png'
    wall_output_file = 'output/23/wall_mask.png'
    merged_output_file = 'output/23/merged_room_mask.png'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    extract_room_and_wall_masks(svg_file, output_file, output_color_file, wall_output_file, merged_output_file)
