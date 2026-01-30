import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. カラーで画像を読み込む
img = cv2.imread('/Users/kohei/kkg/kkg-sandbox/drawing-tracing/_input/mitsui_home/page_1.png', cv2.IMREAD_COLOR)
# グレースケール版も必要なので、別途作成
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 必要に応じてガウシアンぼかしなどを行いノイズ除去
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
# 適応的二値化または固定閾値による二値化
th = cv2.threshold(img_blur, 128, 255, cv2.THRESH_BINARY_INV)[1]

# 2. 線分検出 (HoughやLSDを使用)
# ここではHough LinePを例とする(ただし実線・破線両方検出するため、後で点線か判定が必要)
lines = cv2.HoughLinesP(th, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# 3. 一点鎖線であるかどうかを判定
#   - 各検出線について、その線上のピクセル値をサンプリングし、黒(=線)と白(=背景)が交互に繰り返されるパターンを探索
#   - 以下は概念例
def is_dotted_line(img, line, sample_step=1):
    x1, y1, x2, y2 = line
    length = int(np.hypot(x2-x1, y2-y1))
    # 線上の画素サンプリング
    xs = np.linspace(x1, x2, length, dtype=int)
    ys = np.linspace(y1, y2, length, dtype=int)
    values = img[ys, xs]

    # 点線判定ロジック（簡易例）:
    # values中の黒(255)と白(0)の繰り返しパターンを調べ、一定以上の交互パターンがあれば点線と見なす。
    # 下は大まかな例で、実際には詳細なパターン解析が必要
    # ここでは黒と白が何度も切り替わるかを見る
    transitions = 0
    prev_val = values[0]
    for v in values[1:]:
        if v != prev_val:
            transitions += 1
            prev_val = v

    # 繰り返し回数が一定以上なら点線と仮定
    return transitions > (length/20)  # この閾値は要調整


vertical_lines = [
    [1380, 508, 1380, 3016],
    [1552, 508, 1552, 3016],
    [1720, 508, 1720, 3016],
    [1900, 508, 1900, 3016],
    [2072, 508, 2072, 3016],
    [2244, 508, 2244, 3016],
    [2420, 508, 2420, 3016],
    [2592, 508, 2592, 3016],
    [2940, 508, 2940, 3016],
    [3112, 508, 3112, 3016],
    [3288, 508, 3288, 3016],
]

horizontal_lines = [
    [1108, 632, 3424, 632],
    [1108, 804, 3424, 804],
    [1108, 980, 3424, 980],
    [1108, 1156, 3424, 1156],
    [1108, 1324, 3424, 1324],
    [1108, 1500, 3424, 1500],
    [1108, 1676, 3424, 1676],
    [1108, 1848, 3424, 1848],
    [1108, 2024, 3424, 2024],
    [1108, 2196, 3424, 2196],
    [1108, 2368, 3424, 2368],
    [1108, 2540, 3424, 2540],
    [1108, 2716, 3424, 2716],
    [1108, 2888, 3424, 2888],
]
# マスク生成関数
def create_line_mask(line, width, shape, is_vertical=True):
    x1, y1, x2, y2 = line
    # 法線方向ベクトル
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return None

    if is_vertical:
        nx = -dy / length
        ny = dx / length
    else:
        nx = dy / length
        ny = -dx / length

    # 帯領域の多角形（四角形）を定義
    # 中心線に対して上下に幅width/2ずつ確保
    poly_pts = np.array([
        [x1 + nx*width, y1 + ny*width],
        [x1 - nx*width, y1 - ny*width],
        [x2 - nx*width, y2 - ny*width],
        [x2 + nx*width, y2 + ny*width]
    ], dtype=np.int32)

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 255)
    return mask


# 垂直線と水平線それぞれのマスクを生成
vertical_masks = []
horizontal_masks = []
width = 2

for line in vertical_lines:
    mask = create_line_mask(line, width, th.shape, is_vertical=True)
    if mask is not None:
        vertical_masks.append(mask)

for line in horizontal_lines:
    mask = create_line_mask(line, width, th.shape, is_vertical=False)
    if mask is not None:
        horizontal_masks.append(mask)

# 垂直方向のスキャン（既存のコード）
cell_height = 12
cell_width = 36
vertical_densities = []
y_positions = []

# 水平方向のスキャン（新規追加）
horizontal_cell_width = 12
horizontal_cell_height = 30
horizontal_densities = []
x_positions = []


# 垂直方向と水平方向の密度分布をそれぞれプロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(vertical_densities, y_positions)
ax1.set_title('Vertical Density Distribution')
ax1.set_xlabel('Density')
ax1.set_ylabel('Y Position')
ax1.grid(True)
ax1.invert_yaxis()

ax2.plot(x_positions, horizontal_densities)
ax2.set_title('Horizontal Density Distribution')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Density')
ax2.grid(True)

plt.tight_layout()
plt.savefig('density_distributions_new_sample_1.png')
plt.close()

# ---------------------------------------
# 既存コードの最後あたりに以下を追加
# 垂直方向・水平方向スキャン時に描画した矩形をリストとして保持するようにする
# 変更点：
#   rectangle描画時にrectangles_v, rectangles_hというリストに(x, y, w, h)形式で保持

rectangles_v = []
rectangles_h = []

# 斜め線検出用の関数を追加
def has_diagonal_lines(cell, threshold_angle=30):
    """
    セル内の斜め線の有無を検出
    Args:
        cell: 検査対象のセル画像
        threshold_angle: 水平・垂直とみなさない最小角度（度）
    Returns:
        bool: 斜め線が存在するかどうか
    """
    # エッジ検出（既に2値化済みなのでスキップ可能）
    edges = cell

    # Hough変換で線分検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                           threshold=15,  # 点の数の閾値
                           minLineLength=10,  # 最小線分長
                           maxLineGap=3)  # 許容ギャップ

    if lines is None:
        return False

    # 検出された線分の角度をチェック
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 線分の角度を計算（ラジアンから度に変換）
        angle = np.abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        # 水平(0°,180°)や垂直(90°)から一定以上離れている場合を斜め線とみなす
        is_diagonal = (threshold_angle < angle < (90 - threshold_angle)) or \
                     ((90 + threshold_angle) < angle < (180 - threshold_angle))
        if is_diagonal:
            return True

    return False

for mask in vertical_masks:
    band_region = cv2.bitwise_and(th, mask)
    h, w = band_region.shape

    mask_positions = np.where(mask == 255)
    if len(mask_positions[1]) > 0:
        center_x = int(np.mean(mask_positions[1]))
    else:
        continue

    for y in range(0, h - cell_height, cell_height // 2):
        x = center_x - (cell_width // 2)
        cell = band_region[y:y+cell_height, x:x+cell_width]
        mask_cell = mask[y:y+cell_height, x:x+cell_width]
        valid_pixels = (mask_cell == 255)

        if np.sum(valid_pixels) > 0:
            black_ratio = np.sum(cell == 255) / np.sum(valid_pixels)
            has_diagonal = has_diagonal_lines(cell)
            vertical_densities.append(black_ratio)
            y_positions.append(y)

            # 黒い部分の割合が一定以上 かつ 斜め線が存在する場合に記録
            # if black_ratio > 0.3 and has_diagonal:
            if black_ratio > 0.3:
                rectangles_v.append((x, y, cell_width, cell_height))
                cv2.rectangle(img, (x, y), (x+cell_width, y+cell_height), (0, 0, 255), thickness=1)


for mask in horizontal_masks:
    band_region = cv2.bitwise_and(th, mask)
    h, w = band_region.shape

    mask_positions = np.where(mask == 255)
    if len(mask_positions[0]) > 0:
        center_y = int(np.mean(mask_positions[0]))
    else:
        continue

    for x in range(0, w - horizontal_cell_width, horizontal_cell_width // 2):
        y = center_y - (horizontal_cell_height // 2)
        cell = band_region[y:y+horizontal_cell_height, x:x+horizontal_cell_width]
        mask_cell = mask[y:y+horizontal_cell_height, x:x+horizontal_cell_width]
        valid_pixels = (mask_cell == 255)

        if np.sum(valid_pixels) > 0:
            black_ratio = np.sum(cell == 255) / np.sum(valid_pixels)
            horizontal_densities.append(black_ratio)
            x_positions.append(x)
            has_diagonal = has_diagonal_lines(cell)
            # 黒い部分の割合が一定以上 かつ 斜め線が存在する場合に記録
            if black_ratio > 0.3:
                rectangles_h.append((x, y, horizontal_cell_width, horizontal_cell_height))
                cv2.rectangle(img, (x, y), (x+horizontal_cell_width, y+horizontal_cell_height), (0, 255, 0), thickness=1)


# 密度分布プロット保存処理
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(vertical_densities, y_positions)
ax1.set_title('Vertical Density Distribution')
ax1.set_xlabel('Density')
ax1.set_ylabel('Y Position')
ax1.grid(True)
ax1.invert_yaxis()

ax2.plot(x_positions, horizontal_densities)
ax2.set_title('Horizontal Density Distribution')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Density')
ax2.grid(True)

plt.tight_layout()
plt.savefig('density_distributions.png')
plt.close()


# ---------------------------------------
# 矩形統合処理 ここから追加

# 全ての検出矩形を一つのリストに統合
all_rects = rectangles_v + rectangles_h

# 矩形がオーバーラップしているかどうかを判定する関数
def rects_overlap(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    # 矩形範囲
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    # オーバーラップ条件:
    # 横方向で重なり:  x1 <= x2_max && x2 <= x1_max
    # 縦方向で重なり:  y1 <= y2_max && y2 <= y1_max
    return not (x1_max < x2 or x2_max < x1 or y1_max < y2 or y2_max < y1)

# 矩形群をクラスタリングして結合
def merge_overlapping_rects(rectangles):
    # 連結成分分解用
    used = [False]*len(rectangles)
    clusters = []

    def dfs(idx, group):
        used[idx] = True
        group.append(idx)
        for j in range(len(rectangles)):
            if not used[j]:
                if rects_overlap(rectangles[idx], rectangles[j]):
                    dfs(j, group)

    for i in range(len(rectangles)):
        if not used[i]:
            group = []
            dfs(i, group)
            clusters.append(group)

    # クラスタごとに最小外接矩形を作る
    merged = []
    for grp in clusters:
        # grp内の矩形をまとめる
        xs = []
        ys = []
        xs_max = []
        ys_max = []
        for idx in grp:
            x, y, w, h = rectangles[idx]
            xs.append(x)
            ys.append(y)
            xs_max.append(x+w)
            ys_max.append(y+h)
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs_max)
        y_max = max(ys_max)
        merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return merged

merged_rects = merge_overlapping_rects(all_rects)

# 結果を描画
for (x, y, w, h) in merged_rects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


# JSON出力
# JSONフォーマットは{x,y,w,h}のリストを出力する例
output_data = []
for (x, y, w, h) in merged_rects:
    output_data.append({
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h)
    })

with open('merged_bboxes.json', 'w') as f:
    json.dump(output_data, f, indent=4)

# 結果を保存
cv2.imwrite('detected_marks_bboxes_mitsui_new_sample_1.png', img)

import os

# クロップ画像保存用のディレクトリを作成
output_dir = 'cropped_regions_mitsui_new_sample_1'
os.makedirs(output_dir, exist_ok=True)

# 元画像を使用してクロップ（imgは描画済みなので、最初に読み込んだ画像を再度読み込む）
original_img = cv2.imread('/Users/kohei/kkg/kkg-sandbox/drawing-tracing/_input/mitsui_home/page_1.png', cv2.IMREAD_COLOR)

# 各検出領域をクロップして拡大保存
for i, (x, y, w, h) in enumerate(merged_rects):
    # 画像の範囲内に収まるようにクロップ座標を調整
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(original_img.shape[1], x + w)
    y2 = min(original_img.shape[0], y + h)

    # 領域をクロップ
    cropped = original_img[y1:y2, x1:x2]

    # 画像を8倍に拡大 (interpolation=cv2.INTER_CUBIC でより滑らかな拡大が可能)
    scale = 8.0
    enlarged = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 拡大した画像を保存
    output_path = os.path.join(output_dir, f'region_{i}.png')
    cv2.imwrite(output_path, enlarged)
