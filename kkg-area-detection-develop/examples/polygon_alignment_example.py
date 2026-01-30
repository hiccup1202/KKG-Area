"""
ポリゴンを実際の壁の線に沿わせるサンプルスクリプト

このスクリプトは、検出された部屋のポリゴンを、
オリジナル画像の壁の線に沿うように調整します。
"""
import argparse
import os
from PIL import Image
import kkg_area_detection
import numpy as np
import cv2




def main():
    parser = argparse.ArgumentParser(
        description='部屋のポリゴンを壁の線に沿うように調整します'
    )
    parser.add_argument(
        '-i', '--image',
        required=True,
        help='入力画像のパス'
    )
    parser.add_argument(
        '-o', '--output',
        default='aligned_polygons_output.jpg',
        help='出力画像のパス（デフォルト: aligned_polygons_output.jpg）'
    )
    parser.add_argument(
        '-d', '--device',
        choices=['cuda', 'cpu'],
        help='処理に使用するデバイス（デフォルト: 自動選択）'
    )
    parser.add_argument(
        '-p', '--model-path',
        help='カスタムモデルのパス（オプション）'
    )
    parser.add_argument(
        '-n', '--model-name',
        help='モデル名（S3/キャッシュから読み込む場合）'
    )

    # 輪郭抽出パラメータ
    parser.add_argument(
        '-e', '--epsilon',
        type=float,
        default=0.015,
        help='輪郭近似の精度パラメータ（0.0〜1.0、デフォルト: 0.015）'
    )
    parser.add_argument(
        '--smoothing',
        action='store_true',
        help='輪郭のスムージングを適用'
    )
    parser.add_argument(
        '--angle-filter',
        action='store_true',
        help='角度フィルタリングを適用'
    )
    parser.add_argument(
        '-w', '--wall-filter',
        action='store_true',
        help='輪郭に壁フィルタを適用'
    )
    parser.add_argument(
        '-t', '--target-label-ids',
        type=int,
        nargs='+',
        default=[2, 3],
        help='壁フィルタのターゲットラベルID（デフォルト: [2, 3]）'
    )

    # 直線検出パラメータ（PyLSD）
    parser.add_argument(
        '--scale',
        type=float,
        default=0.8,
        help='PyLSD 画像スケーリング係数（デフォルト: 0.8）'
    )
    parser.add_argument(
        '--sigma-scale',
        type=float,
        default=0.6,
        help='PyLSD ガウシアンカーネルのシグマ値スケール（デフォルト: 0.6）'
    )
    parser.add_argument(
        '--quant',
        type=float,
        default=2.0,
        help='PyLSD 勾配の量子化レベル（デフォルト: 2.0）'
    )
    parser.add_argument(
        '--ang-th',
        type=float,
        default=22.5,
        help='PyLSD 角度の許容閾値（度）（デフォルト: 22.5）'
    )
    parser.add_argument(
        '--density-th',
        type=float,
        default=0.7,
        help='PyLSD 密度閾値（デフォルト: 0.7）'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=1024,
        help='PyLSD ヒストグラムビン数（デフォルト: 1024）'
    )


    # アライメントパラメータ
    parser.add_argument(
        '--snap-distance',
        type=float,
        default=20.0,
        help='頂点を直線にスナップする最大距離（デフォルト: 20.0）'
    )
    parser.add_argument(
        '--snap-angle',
        type=float,
        default=15.0,
        help='線のマッチングの最大角度差（度、デフォルト: 15.0）'
    )
    parser.add_argument(
        '--parallel-search-distance',
        type=float,
        default=5.0,
        help='最外側平行線検索の最大距離（デフォルト: 5.0）'
    )
    parser.add_argument(
        '--straightening-angle-tolerance',
        type=float,
        default=10.0,
        help='垂直・水平修正時の角度許容範囲（度、デフォルト: 10.0）'
    )

    # 可視化オプション
    parser.add_argument(
        '-a', '--alpha',
        type=float,
        default=0.6,
        help='可視化の透明度（0.0〜1.0、デフォルト: 0.6）'
    )
    parser.add_argument(
        '--show-original',
        action='store_true',
        help='元のポリゴンも表示'
    )
    parser.add_argument(
        '--save-comparison',
        action='store_true',
        help='比較画像を保存（_comparison.jpg）'
    )
    parser.add_argument(
        '--show-lines',
        action='store_true',
        help='検出された直線を最終結果に表示'
    )
    parser.add_argument(
        '--original-only',
        action='store_true',
        help='元のポリゴンのみを出力（壁線への調整なし）'
    )
    parser.add_argument(
        '--visible-classes',
        type=int,
        nargs='+',
        help='表示するクラスIDのリスト（例: --visible-classes 2 はクラス2のみ表示）'
    )
    parser.add_argument(
        '--class-colors',
        type=str,
        nargs='+',
        help='特定クラスの固定色を "class_id:r,g,b" 形式で指定 ' +
             '（例: --class-colors "1:255,0,0" "2:0,0,255"）'
    )

    args = parser.parse_args()

    # 画像が存在するか確認
    if not os.path.exists(args.image):
        print(f"エラー: 画像ファイル '{args.image}' が見つかりません。")
        return

    # モデルの初期化
    print("モデルを初期化しています...")
    init_kwargs = {}
    if args.model_path:
        init_kwargs['model_path'] = args.model_path
    elif args.model_name:
        init_kwargs['model_name'] = args.model_name
    if args.device:
        init_kwargs['device'] = args.device

    kkg_area_detection.initialize_model(**init_kwargs)

    # 画像を読み込む
    print(f"画像を読み込んでいます: {args.image}")
    image = Image.open(args.image)

    # numpy配列に変換（OpenCV処理用）
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        # グレースケールの場合はBGRに変換
        image_np = np.stack([image_np] * 3, axis=-1)
    else:
        # RGBからBGRに変換
        image_np = image_np[:, :, ::-1]

    # セグメンテーション実行
    print("セグメンテーションを実行しています...")
    segmentation_result = kkg_area_detection.get_segmentation_result(image)
    segmentation_map = segmentation_result['segmentation'].cpu().numpy()
    segments_info = segmentation_result.get('segments_info', [])

    # 元のポリゴンを抽出（比較用、original-onlyオプション用）
    if args.show_original or args.save_comparison or args.original_only:
        print("元のポリゴンを抽出しています...")
        original_contours = kkg_area_detection.get_approx_contours_and_vertices(
            segmentation_map,
            epsilon=args.epsilon,
            use_smoothing=args.smoothing,
            use_angle_filter=args.angle_filter,
            wall_filter=args.wall_filter,
            segments_info=segments_info,
            target_label_ids=args.target_label_ids,
            edge_margin=0.001,
            align_to_lines=False,  # 元のポリゴンはアライメント不要
        )

    # 直線検出が必要な場合（show-linesまたはアライメントを行う場合）
    if args.show_lines or not args.original_only:
        # ポリゴンをアライメント（または直線のみ検出）
        if not args.original_only:
            print("ポリゴンを壁の線に沿うように調整しています...")
        else:
            print("直線を検出しています...")

        result = kkg_area_detection.get_approx_contours_and_vertices(
            segment_array=segmentation_map,
            epsilon=args.epsilon,
            use_smoothing=args.smoothing,
            use_angle_filter=args.angle_filter,
            wall_filter=args.wall_filter,
            segments_info=segments_info,
            target_label_ids=args.target_label_ids,
            edge_margin=0.001,
            # 直線検出とアライメントのために original_image を追加
            original_image=image_np,
            # 線を返すように指定
            return_lines=True,
            # 直線検出パラメータ（PyLSD）
            scale=args.scale,
            sigma_scale=args.sigma_scale,
            quant=args.quant,
            ang_th=args.ang_th,
            density_th=args.density_th,
            n_bins=args.n_bins,
            # アライメントパラメータ
            max_snap_distance=args.snap_distance,
            snap_angle_tolerance=args.snap_angle,
            # 最外側拡張パラメータ
            parallel_search_distance=args.parallel_search_distance,
            # 垂直・水平修正パラメータ
            straightening_angle_tolerance=args.straightening_angle_tolerance,
        )

        # タプルを展開
        aligned_contours_tmp, detected_lines = result

        print(f"検出された直線数: {len(detected_lines)}")

        if not args.original_only:
            aligned_contours = aligned_contours_tmp
            print(f"調整されたポリゴン数: {len(aligned_contours)}")
        else:
            # original-onlyモードの場合は元のコンターを使用
            aligned_contours = original_contours
            print("元のポリゴンを使用します（壁線への調整なし）")
    else:
        # 直線検出もアライメントも不要な場合
        aligned_contours = original_contours
        detected_lines = []
        print("元のポリゴンのみを使用します")

    # セグメント情報とマージ
    segment_details_map = {info['id']: info for info in segments_info}

    # 面積フィルタリング（小さすぎる領域を除去）
    min_area_ratio = 0.001  # 画像全体の0.1%
    image_area = image.width * image.height
    min_area = image_area * min_area_ratio

    aligned_regions = []
    for contour_info in aligned_contours:
        segment_id = contour_info['id']
        vertices = contour_info['vertices']

        if len(vertices) < 3:
            continue

        # 面積計算
        area = 0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i]['x'] * vertices[j]['y']
            area -= vertices[j]['x'] * vertices[i]['y']
        area = abs(area) / 2.0

        if area < min_area:
            continue

        matched_info = segment_details_map.get(segment_id)
        if matched_info:
            region_data = {
                'score': matched_info['score'],
                'label_id': matched_info['label_id'],
                'segment_id': segment_id,
                'coordinates': vertices,
                'area': area
            }
            # holesがある場合は追加
            if 'holes' in contour_info:
                region_data['holes'] = contour_info['holes']
            aligned_regions.append(region_data)

    print(f"フィルタリング後のポリゴン数: {len(aligned_regions)}")

    # visible_classesが指定されている場合はフィルタリング
    regions_to_visualize = aligned_regions
    if args.visible_classes:
        regions_to_visualize = [
            region for region in aligned_regions
            if region['label_id'] in args.visible_classes
        ]
        print(f"表示するクラス: {args.visible_classes}")
        print(f"フィルタリング後の表示ポリゴン数: {len(regions_to_visualize)}")

    # class_colorsの処理
    fixed_class_colors = None
    if args.class_colors:
        fixed_class_colors = {}
        for color_spec in args.class_colors:
            try:
                class_id_str, color_str = color_spec.split(':')
                class_id = int(class_id_str)
                r, g, b = map(int, color_str.split(','))
                fixed_class_colors[class_id] = (r, g, b)
                print(f"クラス{class_id}に固定色 ({r},{g},{b}) を使用")
            except (ValueError, IndexError):
                print(f"警告: 無効な色指定 '{color_spec}'。 'class_id:r,g,b' 形式で指定してください")

    # save-comparisonオプションで個別画像を保存
    if args.save_comparison:
        print("比較用画像を生成しています...")

        # ベースファイル名を取得
        if '.' in args.output:
            base_name = args.output.rsplit('.', 1)[0]
            ext = args.output.rsplit('.', 1)[1]
        else:
            base_name = args.output
            ext = 'jpg'

        print(f"{len(aligned_regions)}個のポリゴンを塗りつぶし＋輪郭線で表示します")

        # 1. 元のポリゴンのみ（単色塗りつぶし＋輪郭線）
        if 'original_contours' in locals():
            # セグメント情報を構築
            original_regions = []
            for contour_info in original_contours:
                segment_id = contour_info['id']
                matched_info = segment_details_map.get(segment_id)
                if matched_info:
                    region_data = {
                        'score': matched_info['score'],
                        'label_id': matched_info['label_id'],
                        'segment_id': segment_id,
                        'coordinates': contour_info['vertices'],
                    }
                    if 'holes' in contour_info:
                        region_data['holes'] = contour_info['holes']
                    original_regions.append(region_data)

            # visible_classesでフィルタリング
            if args.visible_classes:
                original_regions = [r for r in original_regions if r['label_id'] in args.visible_classes]

            # まず塗りつぶしで可視化（単色）
            original_image = kkg_area_detection.visualize_regions(
                image,
                original_regions,
                alpha=args.alpha,
                fixed_class_colors={0: (220, 220, 220), 1: (200, 200, 220), 2: (180, 200, 180), 3: (220, 200, 200), 4: (200, 220, 220)},  # わずかに異なる色
                visible_classes=args.visible_classes,
            )

            # 輪郭線を追加
            vis_array = np.array(original_image)
            for contour_info in original_contours:
                segment_id = contour_info['id']
                matched_info = segment_details_map.get(segment_id)
                if matched_info and (not args.visible_classes or matched_info['label_id'] in args.visible_classes):
                    vertices = contour_info['vertices']
                    if len(vertices) >= 3:
                        contour_np = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
                        cv2.polylines(vis_array, [contour_np], isClosed=True, color=(0, 0, 0), thickness=2)  # 黒い輪郭線

            original_image = Image.fromarray(vis_array)
            original_path = f"{base_name}_original.{ext}"
            if original_path.lower().endswith(('.jpg', '.jpeg')) and original_image.mode == 'RGBA':
                original_image.convert('RGB').save(original_path)
            else:
                original_image.save(original_path)
            print(f"元のポリゴン画像（塗りつぶし＋輪郭線）を保存しました: {original_path}")

        # 2. 調整後のポリゴンのみ（単色塗りつぶし＋輪郭線）
        # まず塗りつぶしで可視化（単色）
        aligned_image = kkg_area_detection.visualize_regions(
            image,
            regions_to_visualize,
            alpha=args.alpha,
            fixed_class_colors={0: (200, 200, 200), 1: (200, 200, 200), 2: (200, 200, 200), 3: (200, 200, 200), 4: (200, 200, 200)},  # 全て同じグレー
            visible_classes=args.visible_classes,
        )

        # 輪郭線を追加
        vis_array = np.array(aligned_image)
        for region in regions_to_visualize:
            vertices = region['coordinates']
            if len(vertices) >= 3:
                contour_np = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
                cv2.polylines(vis_array, [contour_np], isClosed=True, color=(0, 0, 0), thickness=2)  # 黒い輪郭線

        aligned_image = Image.fromarray(vis_array)
        aligned_path = f"{base_name}_aligned.{ext}"
        if aligned_path.lower().endswith(('.jpg', '.jpeg')) and aligned_image.mode == 'RGBA':
            aligned_image.convert('RGB').save(aligned_path)
        else:
            aligned_image.save(aligned_path)
        print(f"調整後のポリゴン画像（塗りつぶし＋輪郭線）を保存しました: {aligned_path}")

        # 3. 検出した線のみ
        if detected_lines:
            # 元画像のコピーに線を描画
            lines_image_np = image_np.copy()
            for line in detected_lines:
                cv2.line(lines_image_np, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (0, 255, 0), 2)
            # BGRからRGBに変換
            lines_image_rgb = lines_image_np[:, :, ::-1]
            lines_image = Image.fromarray(lines_image_rgb)
            lines_path = f"{base_name}_lines.{ext}"
            if lines_path.lower().endswith(('.jpg', '.jpeg')) and lines_image.mode == 'RGBA':
                lines_image.convert('RGB').save(lines_path)
            else:
                lines_image.save(lines_path)
            print(f"検出線画像を保存しました: {lines_path}")

        # 4. ポリゴンの辺と対応する壁の線のマッチング
        if detected_lines and regions_to_visualize:
            from kkg_area_detection.core.polygon_alignment import find_matching_wall_lines, visualize_polygon_edge_matching

            # 調整後のポリゴンの座標リストを作成
            polygon_vertices_list = [region['coordinates'] for region in regions_to_visualize]

            # 各ポリゴンの辺と対応する壁の線を見つける
            polygon_edge_matches = find_matching_wall_lines(
                polygons=polygon_vertices_list,
                detected_lines=detected_lines,
                max_distance=args.snap_distance,
                angle_tolerance=args.snap_angle,
                min_line_length_ratio=0.8,
            )

            # 可視化
            edge_matching_image_np = visualize_polygon_edge_matching(
                image=image_np.copy(),
                polygon_edge_matches=polygon_edge_matches,
                polygon_color=(0, 0, 255),  # 赤: ポリゴンの辺
                matched_line_color=(0, 255, 0),  # 緑: マッチした壁の線
                unmatched_edge_color=(128, 128, 128),  # グレー: マッチしなかった辺
            )

            # BGRからRGBに変換
            edge_matching_image_rgb = edge_matching_image_np[:, :, ::-1]
            edge_matching_image = Image.fromarray(edge_matching_image_rgb)

            matching_path = f"{base_name}_edge_matching.{ext}"
            if matching_path.lower().endswith(('.jpg', '.jpeg')) and edge_matching_image.mode == 'RGBA':
                edge_matching_image.convert('RGB').save(matching_path)
            else:
                edge_matching_image.save(matching_path)
            print(f"辺と壁線のマッチング画像を保存しました: {matching_path}")

            # マッチング統計を表示
            total_edges = 0
            matched_edges = 0
            for edge_matches in polygon_edge_matches:
                for _, _, matching_line in edge_matches:
                    total_edges += 1
                    if matching_line is not None:
                        matched_edges += 1

            match_percentage = (matched_edges / total_edges * 100) if total_edges > 0 else 0
            print(f"マッチング統計: {matched_edges}/{total_edges} ({match_percentage:.1f}%) の辺が壁線とマッチしました")

    # 結果を可視化（show_originalオプション用）
    if args.show_original and not args.save_comparison:
        # 元のポリゴンも含めて可視化
        original_vertices_list = [[v['vertices'] for v in original_contours]]
        aligned_vertices_list = [[r['coordinates'] for r in aligned_regions]]

        visualized = kkg_area_detection.visualize_lines_and_polygons(
            image=image_np,
            lines=detected_lines if args.show_lines else [],
            original_polygons=original_vertices_list[0],
            aligned_polygons=aligned_vertices_list[0],
            line_color=(0, 255, 0),      # 緑: 検出された直線
            original_color=(255, 0, 0),   # 赤: 元のポリゴン
            aligned_color=(0, 0, 255),    # 青: 調整後のポリゴン
        )

        # BGRからRGBに変換してPIL画像に
        visualized_rgb = visualized[:, :, ::-1]
        result_image = Image.fromarray(visualized_rgb)

        # 最終結果として保存される画像を更新
        # この場合、result_imageは上記で作成された重ね合わせ画像になる

    # 調整後のポリゴンのみを可視化（または元のポリゴンのみ）
    if 'result_image' not in locals():  # show_originalで作成されていない場合
        print(f"{len(regions_to_visualize)}個のポリゴンを塗りつぶし＋輪郭線で表示します")

        # まず塗りつぶしで可視化（単色）
        result_image = kkg_area_detection.visualize_regions(
            image,
            regions_to_visualize,
            alpha=args.alpha,
            fixed_class_colors={0: (200, 200, 200), 1: (200, 200, 200), 2: (200, 200, 200), 3: (200, 200, 200), 4: (200, 200, 200)},  # 全て同じグレー
            visible_classes=args.visible_classes,
        )

        # 輪郭線を追加
        result_np = np.array(result_image)
        for region in regions_to_visualize:
            vertices = region['coordinates']
            if len(vertices) >= 3:
                contour_np = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
                cv2.polylines(result_np, [contour_np], isClosed=True, color=(0, 0, 0), thickness=2)  # 黒い輪郭線

        if args.show_lines and detected_lines:
            # 検出された直線も描画
            for line in detected_lines:
                cv2.line(result_np, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (0, 255, 0), 2)  # 緑色で描画
            print(f"検出された{len(detected_lines)}本の直線を表示しました")

        # PIL画像に変換
        result_image = Image.fromarray(result_np)

    # 結果を保存
    # JPEG保存時はRGBAをRGBに変換
    if args.output.lower().endswith(('.jpg', '.jpeg')) and result_image.mode == 'RGBA':
        rgb_image = result_image.convert('RGB')
        rgb_image.save(args.output)
    else:
        result_image.save(args.output)
    print(f"結果を保存しました: {args.output}")

    # 結果の詳細を表示
    polygon_type = "元のポリゴン" if args.original_only else "調整後のポリゴン"
    for i, region in enumerate(aligned_regions):
        print(f"\n{polygon_type} {i+1}:")
        print(f"  ラベルID: {region['label_id']}")
        print(f"  信頼度: {region['score']:.4f}")
        print(f"  頂点数: {len(region['coordinates'])}")
        print(f"  面積: {region['area']:.2f} ピクセル")


if __name__ == '__main__':
    main()

