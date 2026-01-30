"""
ポリゴンを実際の壁の線に沿わせるためのアライメント機能
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
from pylsd.lsd import lsd


@dataclass
class Line:
    """検出された直線を表すクラス"""
    rho: float  # 原点からの距離
    theta: float  # 角度（ラジアン）
    x1: int  # 始点のx座標
    y1: int  # 始点のy座標
    x2: int  # 終点のx座標
    y2: int  # 終点のy座標

    @property
    def angle_degrees(self) -> float:
        """角度を度数法で返す"""
        return math.degrees(self.theta)

    @property
    def length(self) -> float:
        """線分の長さを返す"""
        return math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)


def detect_lines_pylsd(
    image: np.ndarray,
    scale: float = 0.8,
    sigma_scale: float = 0.6,
    quant: float = 2.0,
    ang_th: float = 22.5,
    density_th: float = 0.7,
    n_bins: int = 1024,
    angle_tolerance: float = 5.0,
) -> List[Line]:
    """
    PyLSD (Line Segment Detector) を使用して画像から水平・垂直に近い直線のみを検出する

    Args:
        image: 入力画像（BGR or グレースケール）
        scale: 画像のスケーリング係数（デフォルト: 0.8）
        sigma_scale: ガウシアンカーネルのシグマ値のスケール（デフォルト: 0.6）
        quant: 勾配の量子化レベル（デフォルト: 2.0）
        ang_th: 角度の許容閾値（度）（デフォルト: 22.5）
        density_th: 密度閾値（デフォルト: 0.7）
        n_bins: ヒストグラムのビン数（デフォルト: 1024）
        angle_tolerance: 垂直・水平線と見なす角度の許容範囲（度）（デフォルト: 5.0）

    Returns:
        検出された水平・垂直に近い直線のリスト（斜めの線は除外）
    """
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # PyLSDで線分を検出
    # PyLSDは (x1, y1, x2, y2, width) の形式で線分を返す
    lines = lsd(
        gray.astype(np.float64),
        scale=scale,
        sigma_scale=sigma_scale,
        quant=quant,
        ang_th=ang_th,
        density_th=density_th,
        n_bins=n_bins
    )

    detected_lines = []

    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])

            # 線分の角度を計算
            dx = x2 - x1
            dy = y2 - y1
            angle = math.atan2(dy, dx)

            # 垂直・水平線に近い角度かをチェック
            angle_deg = math.degrees(angle)

            # 角度を -180 から 180 の範囲に正規化
            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg <= -180:
                angle_deg += 360

            # 水平線の判定：0度付近または ±180度付近
            is_horizontal = (abs(angle_deg) < angle_tolerance or
                           abs(abs(angle_deg) - 180) < angle_tolerance)

            # 垂直線の判定：±90度付近
            is_vertical = abs(abs(angle_deg) - 90) < angle_tolerance

            if is_horizontal:
                angle = 0
                # y座標を平均化して完全に水平にする
                y_avg = (y1 + y2) // 2
                y1 = y2 = y_avg
            elif is_vertical:
                angle = math.pi / 2
                # x座標を平均化して完全に垂直にする
                x_avg = (x1 + x2) // 2
                x1 = x2 = x_avg
            else:
                # 水平・垂直に近くない線は採用しない
                continue

            # rhoを計算（原点から直線への垂直距離）
            rho = abs(x1 * math.sin(angle) - y1 * math.cos(angle))

            detected_lines.append(Line(
                rho=rho,
                theta=angle,
                x1=x1, y1=y1,
                x2=x2, y2=y2
            ))

    return detected_lines


def find_nearest_line(
    point1: Tuple[int, int],
    point2: Tuple[int, int],
    lines: List[Line],
    max_distance: float = 20.0,
    angle_tolerance: float = 15.0,
    min_line_length_ratio: float = 0.8,
) -> Optional[Line]:
    """
    2点を結ぶ線分に最も近い検出線を見つける（線分の長さの30%以上の長さを持つ線のみ採用）

    Args:
        point1: 線分の始点 (x, y)
        point2: 線分の終点 (x, y)
        lines: 検出された直線のリスト
        max_distance: 最大許容距離
        angle_tolerance: 角度の最大許容差（度）
        min_line_length_ratio: 線分に対する検出線の最小長さ比率（デフォルト: 0.3）

    Returns:
        最も近い直線、見つからない場合はNone
    """
    x1, y1 = point1
    x2, y2 = point2

    # 線分の長さを計算
    segment_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    min_required_line_length = segment_length * min_line_length_ratio

    # 線分の角度を計算
    segment_angle = math.atan2(y2 - y1, x2 - x1)
    segment_angle_deg = math.degrees(segment_angle)

    # 線分の中点
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    best_line = None
    min_distance = max_distance


    candidates_checked = 0
    candidates_rejected_length = 0
    candidates_rejected_angle = 0
    candidates_rejected_distance = 0

    for line in lines:
        candidates_checked += 1

        # 検出線の長さをチェック
        if line.length < min_required_line_length:
            candidates_rejected_length += 1
            continue

        # 角度の差を計算（-180～180度の範囲に正規化）
        angle_diff = abs(segment_angle_deg - line.angle_degrees)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # 角度が許容範囲外ならスキップ
        if angle_diff > angle_tolerance:
            candidates_rejected_angle += 1
            continue

        # 中点から直線への距離を計算
        distance = point_to_line_distance(
            (mid_x, mid_y),
            line.rho,
            line.theta
        )

        if distance >= min_distance:
            candidates_rejected_distance += 1
            continue

        min_distance = distance
        best_line = line


    return best_line


def point_to_line_distance(
    point: Tuple[float, float],
    rho: float,
    theta: float
) -> float:
    """
    点から直線（rho, theta形式）への距離を計算

    Args:
        point: 点の座標 (x, y)
        rho: 直線のrho値
        theta: 直線のtheta値（ラジアン）

    Returns:
        点から直線への距離
    """
    x, y = point
    return abs(x * math.sin(theta) - y * math.cos(theta) - rho)


def project_point_to_line(
    point: Tuple[float, float],
    rho: float,
    theta: float
) -> Tuple[float, float]:
    """
    点を直線に投影

    Args:
        point: 投影する点 (x, y)
        rho: 直線のrho値
        theta: 直線のtheta値（ラジアン）

    Returns:
        投影された点の座標 (x, y)
    """
    x, y = point

    # 直線の法線ベクトル
    nx = math.sin(theta)
    ny = -math.cos(theta)

    # 点から直線への距離（符号付き）
    distance = x * nx + y * ny - rho

    # 投影点
    proj_x = x - distance * nx
    proj_y = y - distance * ny

    return (proj_x, proj_y)


def find_parallel_lines_in_direction(
    reference_line: Line,
    all_lines: List[Line],
    polygon_centroid: Tuple[float, float],
    search_distance: float = 50.0,
    angle_tolerance: float = 5.0,
) -> List[Line]:
    """
    基準線と平行で、ポリゴン重心から見て外側方向にある線を検出

    Args:
        reference_line: 基準となる線
        all_lines: 全ての検出された線
        polygon_centroid: ポリゴンの重心座標 (x, y)
        search_distance: 平行線検索の距離範囲
        angle_tolerance: 平行線判定の角度許容範囲（度）

    Returns:
        外側方向にある平行線のリスト
    """
    parallel_lines = []

    # 基準線の角度
    ref_angle = reference_line.angle_degrees

    # 基準線が水平か垂直かを判定
    is_horizontal = abs(ref_angle % 180) < angle_tolerance or abs(abs(ref_angle % 180) - 180) < angle_tolerance
    is_vertical = abs(abs(ref_angle % 180) - 90) < angle_tolerance

    if not (is_horizontal or is_vertical):
        return parallel_lines

    for line in all_lines:
        if line == reference_line:
            continue

        # 角度の差をチェック
        angle_diff = abs(ref_angle - line.angle_degrees)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff > angle_tolerance:
            continue

        # 距離をチェック
        if is_horizontal:
            # 水平線の場合：Y軸方向の距離をチェック
            ref_y = (reference_line.y1 + reference_line.y2) / 2
            line_y = (line.y1 + line.y2) / 2
            distance = abs(line_y - ref_y)

            if distance <= search_distance:
                # 外側方向かチェック（ポリゴン重心から遠い方向）
                centroid_y = polygon_centroid[1]
                if (line_y > ref_y and ref_y > centroid_y) or (line_y < ref_y and ref_y < centroid_y):
                    parallel_lines.append(line)

        elif is_vertical:
            # 垂直線の場合：X軸方向の距離をチェック
            ref_x = (reference_line.x1 + reference_line.x2) / 2
            line_x = (line.x1 + line.x2) / 2
            distance = abs(line_x - ref_x)

            if distance <= search_distance:
                # 外側方向かチェック（ポリゴン重心から遠い方向）
                centroid_x = polygon_centroid[0]
                if (line_x > ref_x and ref_x > centroid_x) or (line_x < ref_x and ref_x < centroid_x):
                    parallel_lines.append(line)

    return parallel_lines


def get_outermost_line_position(
    reference_line: Line,
    parallel_lines: List[Line],
    polygon_centroid: Tuple[float, float],
) -> Optional[float]:
    """
    ポリゴン重心から見て最も外側にある平行線の位置を取得

    Args:
        reference_line: 基準となる線
        parallel_lines: 平行線のリスト
        polygon_centroid: ポリゴンの重心座標

    Returns:
        最外側線の座標（水平線ならY座標、垂直線ならX座標）
    """
    if not parallel_lines:
        return None

    ref_angle = reference_line.angle_degrees
    is_horizontal = abs(ref_angle % 180) < 5.0 or abs(abs(ref_angle % 180) - 180) < 5.0

    if is_horizontal:
        # 水平線の場合：Y座標で比較
        centroid_y = polygon_centroid[1]
        ref_y = (reference_line.y1 + reference_line.y2) / 2

        # 基準線がポリゴンの上下どちらにあるかで判定方向を決める
        if ref_y > centroid_y:
            # 基準線がポリゴンより上にある場合、最も上の線を探す
            max_y = max((line.y1 + line.y2) / 2 for line in parallel_lines + [reference_line])
            return max_y
        else:
            # 基準線がポリゴンより下にある場合、最も下の線を探す
            min_y = min((line.y1 + line.y2) / 2 for line in parallel_lines + [reference_line])
            return min_y
    else:
        # 垂直線の場合：X座標で比較
        centroid_x = polygon_centroid[0]
        ref_x = (reference_line.x1 + reference_line.x2) / 2

        if ref_x > centroid_x:
            # 基準線がポリゴンより右にある場合、最も右の線を探す
            max_x = max((line.x1 + line.x2) / 2 for line in parallel_lines + [reference_line])
            return max_x
        else:
            # 基準線がポリゴンより左にある場合、最も左の線を探す
            min_x = min((line.x1 + line.x2) / 2 for line in parallel_lines + [reference_line])
            return min_x


def calculate_polygon_centroid(vertices: List[Dict[str, int]]) -> Tuple[float, float]:
    """
    ポリゴンの重心を計算
    """
    if not vertices:
        return (0.0, 0.0)

    x_sum = sum(v['x'] for v in vertices)
    y_sum = sum(v['y'] for v in vertices)

    return (x_sum / len(vertices), y_sum / len(vertices))


def extend_edge_to_outermost_parallel(
    edge_start: Dict[str, int],
    edge_end: Dict[str, int],
    reference_line: Line,
    outermost_position: float,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    辺を最も外側の平行線まで拡張

    Args:
        edge_start: 辺の開始点
        edge_end: 辺の終了点
        reference_line: スナップした基準線
        outermost_position: 最外側線の位置

    Returns:
        拡張された辺の開始点と終了点
    """
    ref_angle = reference_line.angle_degrees
    is_horizontal = abs(ref_angle % 180) < 5.0 or abs(abs(ref_angle % 180) - 180) < 5.0

    new_start = edge_start.copy()
    new_end = edge_end.copy()

    if is_horizontal:
        # 水平線の場合：Y座標を調整
        new_start['y'] = int(round(outermost_position))
        new_end['y'] = int(round(outermost_position))
    else:
        # 垂直線の場合：X座標を調整
        new_start['x'] = int(round(outermost_position))
        new_end['x'] = int(round(outermost_position))

    return new_start, new_end


def align_polygon_to_lines(
    vertices: List[Dict[str, int]],
    lines: List[Line],
    max_distance: float = 20.0,
    angle_tolerance: float = 15.0,
    min_edge_length: float = 10.0,
    min_line_length_ratio: float = 0.8,
    straighten_first: bool = True,
    straightening_angle_tolerance: float = 10.0,
    fill_holes: bool = True,
    ensure_complete_perimeter_flag: bool = True,
    extend_to_outermost: bool = True,
    parallel_search_distance: float = 5.0,
) -> List[Dict[str, int]]:
    """
    ポリゴンの頂点を検出された直線に沿うように調整

    Args:
        vertices: ポリゴンの頂点リスト [{'x': x, 'y': y}, ...]
        lines: 検出された直線のリスト
        max_distance: 直線にスナップする最大距離
        angle_tolerance: 角度の最大許容差（度）
        min_edge_length: 最小エッジ長（これより短いエッジは削除）
        min_line_length_ratio: 辺の長さに対する検出線の最小長さ比率
        straighten_first: まず辺を垂直・水平に修正するかどうか
        straightening_angle_tolerance: 垂直・水平修正時の角度許容範囲（度、デフォルト: 10.0）
        fill_holes: ポリゴンの穴を埋めるかどうか（デフォルト: True）
        ensure_complete_perimeter_flag: 外周辺の完全性を保証するかどうか（デフォルト: True）
        extend_to_outermost: 最外側の平行線まで拡張するかどうか（デフォルト: True）
        parallel_search_distance: 平行線検索の距離範囲（デフォルト: 5.0）

    Returns:
        調整された頂点のリスト
    """
    if len(vertices) < 3:
        return vertices

    working_vertices = [v.copy() for v in vertices]

    # Step 1: まず辺を垂直・水平に修正（オプション）
    if straighten_first:
        working_vertices = straighten_polygon_edges(
            working_vertices,
            straightening_angle_tolerance
        )

    # Step 1.5: 外周辺の完全性を保証
    if ensure_complete_perimeter_flag:
        working_vertices = ensure_complete_perimeter(working_vertices)

    # Step 2: 垂直・水平化後の辺の長さで検出線にスナップ
    n = len(working_vertices)
    aligned_vertices = [v.copy() for v in working_vertices]
    edge_line_mapping = []  # 辺とスナップした線のマッピング


    # ポリゴンの重心を計算（最外側拡張で使用）
    centroid = calculate_polygon_centroid(working_vertices) if extend_to_outermost else None

    for i in range(n):
        v_curr = working_vertices[i]
        v_next = working_vertices[(i + 1) % n]


        # 現在の辺に最も近い直線を見つける（垂直・水平化後の辺の長さで30%条件を適用）
        nearest_line = find_nearest_line(
            (v_curr['x'], v_curr['y']),
            (v_next['x'], v_next['y']),
            lines,
            max_distance,
            angle_tolerance,
            min_line_length_ratio
        )

        if nearest_line:
            edge_line_mapping.append((i, nearest_line))

            # 両端点を直線に投影
            proj_curr_x, proj_curr_y = project_point_to_line(
                (v_curr['x'], v_curr['y']),
                nearest_line.rho,
                nearest_line.theta
            )
            proj_next_x, proj_next_y = project_point_to_line(
                (v_next['x'], v_next['y']),
                nearest_line.rho,
                nearest_line.theta
            )


            # 頂点を投影点に更新
            aligned_vertices[i] = {
                'x': int(round(proj_curr_x)),
                'y': int(round(proj_curr_y))
            }
            aligned_vertices[(i + 1) % n] = {
                'x': int(round(proj_next_x)),
                'y': int(round(proj_next_y))
            }
        else:
            edge_line_mapping.append((i, None))

    # アライメント結果サマリー
    successful_alignments = sum(1 for _, line in edge_line_mapping if line is not None)

    # Step 2.5: 最外側平行線への拡張（オプション）
    if extend_to_outermost and centroid:

        extensions_applied = 0
        for edge_idx, snapped_line in edge_line_mapping:
            if snapped_line is None:
                continue


            # この辺にスナップした線と平行な外側の線を検索
            parallel_lines = find_parallel_lines_in_direction(
                reference_line=snapped_line,
                all_lines=lines,
                polygon_centroid=centroid,
                search_distance=parallel_search_distance,
                angle_tolerance=5.0
            )


            if parallel_lines:
                # 最外側の線の位置を取得
                outermost_position = get_outermost_line_position(
                    reference_line=snapped_line,
                    parallel_lines=parallel_lines,
                    polygon_centroid=centroid
                )

                if outermost_position is not None:
                    # 辺を最外側の線まで拡張
                    next_idx = (edge_idx + 1) % n
                    new_start, new_end = extend_edge_to_outermost_parallel(
                        edge_start=aligned_vertices[edge_idx],
                        edge_end=aligned_vertices[next_idx],
                        reference_line=snapped_line,
                        outermost_position=outermost_position
                    )

                    aligned_vertices[edge_idx] = new_start
                    aligned_vertices[next_idx] = new_end
                    extensions_applied += 1


    # aligned_verticesから最終的な頂点リストを作成

    # Step 3: 重複する頂点や短いエッジを削除
    cleaned_vertices = []
    for i in range(len(aligned_vertices)):
        v_curr = aligned_vertices[i]
        v_next = aligned_vertices[(i + 1) % len(aligned_vertices)]

        # エッジの長さを計算
        edge_length = math.sqrt(
            (v_next['x'] - v_curr['x'])**2 +
            (v_next['y'] - v_curr['y'])**2
        )

        # 十分な長さがあれば頂点を保持
        if edge_length >= min_edge_length:
            cleaned_vertices.append(v_curr)

    # 最低3頂点は必要
    if len(cleaned_vertices) < 3:
        return vertices

    return cleaned_vertices


def straighten_polygon_edges(
    vertices: List[Dict[str, int]],
    angle_tolerance: float = 10.0
) -> List[Dict[str, int]]:
    """
    ポリゴンの辺を厳密に垂直・水平に修正する

    Args:
        vertices: ポリゴンの頂点リスト
        angle_tolerance: 垂直・水平修正時の角度許容範囲（度、デフォルト: 10.0）

    Returns:
        修正された頂点のリスト
    """
    if len(vertices) < 3:
        return vertices

    n = len(vertices)
    straightened_vertices = [v.copy() for v in vertices]

    # Step 1: 各辺を解析して水平・垂直に近い辺を特定
    edge_directions = []
    for i in range(n):
        v_curr = vertices[i]
        v_next = vertices[(i + 1) % n]

        dx = v_next['x'] - v_curr['x']
        dy = v_next['y'] - v_curr['y']

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            edge_directions.append('skip')
            continue

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # 角度を -180 から 180 の範囲に正規化
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360

        # 水平線の判定：0度付近または ±180度付近
        is_horizontal = (abs(angle_deg) < angle_tolerance or
                       abs(abs(angle_deg) - 180) < angle_tolerance)

        # 垂直線の判定：±90度付近
        is_vertical = abs(abs(angle_deg) - 90) < angle_tolerance

        if is_horizontal:
            edge_directions.append('horizontal')
        elif is_vertical:
            edge_directions.append('vertical')
        else:
            edge_directions.append('diagonal')

    # Step 2: 水平・垂直な辺を厳密に修正
    for i in range(n):
        current_direction = edge_directions[i]
        prev_direction = edge_directions[(i - 1) % n]

        if current_direction == 'horizontal':
            # この辺を水平にする
            v_curr = straightened_vertices[i]
            v_next = straightened_vertices[(i + 1) % n]

            # y座標を平均化
            y_avg = (v_curr['y'] + v_next['y']) // 2

            # 前の辺も考慮して調整
            if prev_direction == 'vertical':
                # 前の辺が垂直の場合、現在の頂点のy座標のみ調整
                straightened_vertices[i]['y'] = y_avg
                straightened_vertices[(i + 1) % n]['y'] = y_avg
            else:
                # 両端点を平均y座標に移動
                straightened_vertices[i]['y'] = y_avg
                straightened_vertices[(i + 1) % n]['y'] = y_avg

        elif current_direction == 'vertical':
            # この辺を垂直にする
            v_curr = straightened_vertices[i]
            v_next = straightened_vertices[(i + 1) % n]

            # x座標を平均化
            x_avg = (v_curr['x'] + v_next['x']) // 2

            # 前の辺も考慮して調整
            if prev_direction == 'horizontal':
                # 前の辺が水平の場合、現在の頂点のx座標のみ調整
                straightened_vertices[i]['x'] = x_avg
                straightened_vertices[(i + 1) % n]['x'] = x_avg
            else:
                # 両端点を平均x座標に移動
                straightened_vertices[i]['x'] = x_avg
                straightened_vertices[(i + 1) % n]['x'] = x_avg

    # Step 3: さらに厳密に水平・垂直にスナップ
    for i in range(n):
        current_direction = edge_directions[i]

        if current_direction == 'horizontal':
            # 完全に水平にする - 次の頂点のy座標を現在の頂点に合わせる
            next_idx = (i + 1) % n
            straightened_vertices[next_idx]['y'] = straightened_vertices[i]['y']

        elif current_direction == 'vertical':
            # 完全に垂直にする - 次の頂点のx座標を現在の頂点に合わせる
            next_idx = (i + 1) % n
            straightened_vertices[next_idx]['x'] = straightened_vertices[i]['x']

    # Step 4: 隣接する辺の整合性を最終調整
    for iteration in range(2):  # 2回繰り返して収束させる
        for i in range(n):
            current_direction = edge_directions[i]
            next_direction = edge_directions[(i + 1) % n]

            if current_direction == 'horizontal' and next_direction == 'vertical':
                # 水平→垂直の接続点で整合性を保つ
                next_idx = (i + 1) % n
                next_next_idx = (i + 2) % n
                # 次の辺の始点を調整
                straightened_vertices[next_idx]['y'] = straightened_vertices[i]['y']
                straightened_vertices[next_next_idx]['x'] = straightened_vertices[next_idx]['x']

            elif current_direction == 'vertical' and next_direction == 'horizontal':
                # 垂直→水平の接続点で整合性を保つ
                next_idx = (i + 1) % n
                next_next_idx = (i + 2) % n
                # 次の辺の始点を調整
                straightened_vertices[next_idx]['x'] = straightened_vertices[i]['x']
                straightened_vertices[next_next_idx]['y'] = straightened_vertices[next_idx]['y']

    return straightened_vertices


def fill_polygon_holes(contour_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    ポリゴンの穴を埋める

    Args:
        contour_info: 輪郭情報

    Returns:
        穴が埋められた輪郭情報
    """
    filled_contour = contour_info.copy()

    # 穴の情報を削除
    if 'holes' in filled_contour:
        del filled_contour['holes']

    return filled_contour


def ensure_complete_perimeter(
    vertices: List[Dict[str, int]],
    min_gap_threshold: float = 5.0,
    max_gap_threshold: float = 50.0,
) -> List[Dict[str, int]]:
    """
    ポリゴンの外周辺が完全になるように補完する

    Args:
        vertices: ポリゴンの頂点リスト
        min_gap_threshold: 最小ギャップ閾値（これより小さいギャップは無視）
        max_gap_threshold: 最大ギャップ閾値（これより大きいギャップは分割）

    Returns:
        外周が完全になった頂点のリスト
    """
    if len(vertices) < 3:
        return vertices

    completed_vertices = []
    n = len(vertices)

    for i in range(n):
        current = vertices[i]
        next_vertex = vertices[(i + 1) % n]

        # 現在の頂点を追加
        completed_vertices.append(current.copy())

        # 次の頂点との距離を計算
        dx = next_vertex['x'] - current['x']
        dy = next_vertex['y'] - current['y']
        distance = math.sqrt(dx * dx + dy * dy)

        # ギャップが大きい場合は中間点を補完
        if distance > max_gap_threshold:
            # 直線で接続する中間点を追加
            num_segments = max(2, int(distance / max_gap_threshold))

            for j in range(1, num_segments):
                ratio = j / num_segments
                intermediate_x = current['x'] + dx * ratio
                intermediate_y = current['y'] + dy * ratio

                completed_vertices.append({
                    'x': int(round(intermediate_x)),
                    'y': int(round(intermediate_y))
                })

        # 小さすぎるギャップの場合は次の頂点をスキップ（重複除去）
        elif distance < min_gap_threshold:
            # 次の頂点をスキップ（現在の頂点のみ追加済み）
            continue

    # 重複する連続した頂点を削除
    cleaned_vertices = []
    for i, vertex in enumerate(completed_vertices):
        if i == 0 or (vertex['x'] != completed_vertices[i-1]['x'] or
                      vertex['y'] != completed_vertices[i-1]['y']):
            cleaned_vertices.append(vertex)

    # 最初と最後の頂点が重複している場合は最後を削除
    if (len(cleaned_vertices) > 2 and
        cleaned_vertices[0]['x'] == cleaned_vertices[-1]['x'] and
        cleaned_vertices[0]['y'] == cleaned_vertices[-1]['y']):
        cleaned_vertices.pop()

    return cleaned_vertices if len(cleaned_vertices) >= 3 else vertices




def find_matching_wall_lines(
    polygons: List[List[Dict[str, int]]],
    detected_lines: List[Line],
    max_distance: float = 20.0,
    angle_tolerance: float = 15.0,
    min_line_length_ratio: float = 0.8,
) -> List[List[Tuple[Dict[str, int], Dict[str, int], Optional[Line]]]]:
    """
    各ポリゴンの辺について、対応する壁の線を見つける

    Args:
        polygons: ポリゴンの頂点リストのリスト
        detected_lines: 検出された直線のリスト
        max_distance: 最大許容距離
        angle_tolerance: 角度の最大許容差（度）
        min_line_length_ratio: 辺の長さに対する検出線の最小長さ比率

    Returns:
        各ポリゴンについて、(始点, 終点, 対応する壁線) のリストのリスト
    """
    polygon_edge_matches = []

    for polygon_vertices in polygons:
        edge_matches = []
        n = len(polygon_vertices)

        for i in range(n):
            v_curr = polygon_vertices[i]
            v_next = polygon_vertices[(i + 1) % n]

            # この辺に対応する壁の線を見つける
            matching_line = find_nearest_line(
                (v_curr['x'], v_curr['y']),
                (v_next['x'], v_next['y']),
                detected_lines,
                max_distance,
                angle_tolerance,
                min_line_length_ratio
            )

            edge_matches.append((v_curr, v_next, matching_line))

        polygon_edge_matches.append(edge_matches)

    return polygon_edge_matches


def visualize_polygon_edge_matching(
    image: np.ndarray,
    polygon_edge_matches: List[List[Tuple[Dict[str, int], Dict[str, int], Optional[Line]]]],
    polygon_color: Tuple[int, int, int] = (0, 0, 255),  # 赤: ポリゴンの辺
    matched_line_color: Tuple[int, int, int] = (0, 255, 0),  # 緑: マッチした壁の線
    unmatched_edge_color: Tuple[int, int, int] = (128, 128, 128),  # グレー: マッチしなかった辺
    thickness: int = 2,
) -> np.ndarray:
    """
    ポリゴンの辺と対応する壁の線を可視化

    Args:
        image: 背景画像（BGR）
        polygon_edge_matches: ポリゴンの辺と対応する壁線の情報
        polygon_color: ポリゴンの辺の色（BGR）
        matched_line_color: マッチした壁線の色（BGR）
        unmatched_edge_color: マッチしなかった辺の色（BGR）
        thickness: 線の太さ

    Returns:
        可視化された画像（BGR）
    """
    result_image = image.copy()

    for edge_matches in polygon_edge_matches:
        for v_start, v_end, matching_line in edge_matches:
            # ポリゴンの辺を描画
            start_point = (int(v_start['x']), int(v_start['y']))
            end_point = (int(v_end['x']), int(v_end['y']))

            if matching_line is not None:
                # マッチした辺は通常色で描画
                cv2.line(result_image, start_point, end_point, polygon_color, thickness)

                # 対応する壁の線を描画
                wall_start = (int(matching_line.x1), int(matching_line.y1))
                wall_end = (int(matching_line.x2), int(matching_line.y2))
                cv2.line(result_image, wall_start, wall_end, matched_line_color, thickness)

                # マッチングを示す薄い線で接続（オプション）
                # cv2.line(result_image, start_point, wall_start, (255, 255, 0), 1)
                # cv2.line(result_image, end_point, wall_end, (255, 255, 0), 1)
            else:
                # マッチしなかった辺はグレーで描画
                cv2.line(result_image, start_point, end_point, unmatched_edge_color, thickness)

    return result_image


def visualize_lines_and_polygons(
    image: np.ndarray,
    lines: List[Line],
    original_polygons: List[List[Dict[str, int]]],
    aligned_polygons: List[List[Dict[str, int]]],
    line_color: Tuple[int, int, int] = (0, 255, 0),
    original_color: Tuple[int, int, int] = (255, 0, 0),
    aligned_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    検出された直線とポリゴン（調整前後）を可視化

    Args:
        image: 背景画像
        lines: 検出された直線のリスト
        original_polygons: 元のポリゴンのリスト
        aligned_polygons: 調整後のポリゴンのリスト
        line_color: 直線の色 (BGR)
        original_color: 元のポリゴンの色 (BGR)
        aligned_color: 調整後のポリゴンの色 (BGR)
        thickness: 線の太さ

    Returns:
        可視化された画像
    """
    result = image.copy()

    # 検出された直線を描画
    for line in lines:
        cv2.line(result, (line.x1, line.y1), (line.x2, line.y2),
                line_color, thickness)

    # 元のポリゴンを描画
    for polygon in original_polygons:
        if len(polygon) >= 3:
            points = np.array([[v['x'], v['y']] for v in polygon], np.int32)
            cv2.polylines(result, [points], True, original_color, thickness)

    # 調整後のポリゴンを描画
    for polygon in aligned_polygons:
        if len(polygon) >= 3:
            points = np.array([[v['x'], v['y']] for v in polygon], np.int32)
            cv2.polylines(result, [points], True, aligned_color, thickness)

    return result
