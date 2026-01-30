#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil

from tqdm import tqdm

# Try relative import first, then fall back to absolute import
try:
    from .image_processing import align_resize_and_crop, whiten_background
    from .mask_extraction import (extract_object_masks,
                                  extract_room_wall_door_window_masks)
except ImportError:
    from kkg_area_detection.fine_tuning.image_processing import (
        align_resize_and_crop, whiten_background)
    from kkg_area_detection.fine_tuning.mask_extraction import (
        extract_object_masks, extract_room_wall_door_window_masks)


def process_cubicasa_dataset(base_dirs, output_dir, process_all_objects=True):
    """
    CubiCasa5kデータセットを処理して部屋、壁、ドア、窓のマスクを抽出する

    Args:
        base_dirs: CubiCasa5kデータを含むベースディレクトリのリスト
        output_dir: 処理済みデータの保存先ディレクトリ
        process_all_objects: 背景白色化のために全オブジェクトを処理するかどうか
    """
    # 画像とアノテーション用のディレクトリを作成
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    if process_all_objects:
        os.makedirs(os.path.join(output_dir, "allobj"), exist_ok=True)

    n = 0

    for base_dir in base_dirs:
        print(f"ディレクトリを処理中: {base_dir}")
        if not os.path.isdir(base_dir):
            print(f"警告: ディレクトリが見つかりません - {base_dir}")
            continue

        for item_name in tqdm(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, item_name)

            # ディレクトリでなければスキップ
            if not os.path.isdir(folder_path):
                continue

            # model.svgの確認
            svg_file = os.path.join(folder_path, "model.svg")
            if not os.path.exists(svg_file):
                continue

            # 元画像の確認
            ori_file = os.path.join(folder_path, "F1_scaled.png")
            if not os.path.exists(ori_file):
                continue

            n += 1
            base_filename = f"floorplan_{n}"  # Consistent base name

            # 画像を images ディレクトリに保存
            dst_img = os.path.join(images_dir, f"{base_filename}.png")
            try:
                shutil.copy(ori_file, dst_img)
            except Exception as e:
                print(f"コピー中にエラー {ori_file} -> {dst_img}: {e}")
                continue

            # 出力パス（アノテーションを annotations ディレクトリに保存）
            # Use the same base filename
            output_color_file = os.path.join(
                annotations_dir, f"{base_filename}_color.png"
            )
            output_annotation_file = os.path.join(
                annotations_dir, f"{base_filename}_annotation.png"
            )

            # マスクの抽出
            try:
                extract_room_wall_door_window_masks(
                    svg_file, output_color_file, output_annotation_file
                )
            except Exception as e:
                print(f"処理中にエラー {svg_file}: {e}")
                continue

            # 全オブジェクト処理（オプション）
            if process_all_objects:
                # 全オブジェクトのマスクを抽出
                allobj_dir = os.path.join(output_dir, "allobj")
                # Use the same base filename
                output_obj_mask_file = os.path.join(
                    allobj_dir, f"{base_filename}_object_mask.png"
                )
                output_obj_color_file = os.path.join(
                    allobj_dir, f"{base_filename}_object_color.png"
                )
                try:
                    extract_object_masks(
                        svg_file, output_obj_mask_file, output_obj_color_file
                    )
                except Exception as e:
                    print(f"全オブジェクト処理中にエラー {svg_file}: {e}")

    print(f"\n処理した画像の総数: {n}")

    # リサイズ用のディレクトリを作成
    resize_dir = f"{output_dir}_resize"
    resize_images_dir = os.path.join(resize_dir, "images")
    resize_annotations_dir = os.path.join(resize_dir, "annotations")

    os.makedirs(resize_images_dir, exist_ok=True)
    os.makedirs(resize_annotations_dir, exist_ok=True)

    # 画像のアライメントとリサイズ
    if process_all_objects:
        allobj_dir = os.path.join(output_dir, "allobj")
        resize_allobj_dir = os.path.join(resize_dir, "allobj")
        os.makedirs(resize_allobj_dir, exist_ok=True)

        align_resize_and_crop(
            images_dir,
            annotations_dir,
            resize_images_dir,
            resize_annotations_dir,
            allobj_dir,
            resize_allobj_dir,
        )
    else:
        align_resize_and_crop(
            images_dir, annotations_dir, resize_images_dir, resize_annotations_dir
        )

    # 全オブジェクト処理が有効なら背景を白色化
    if process_all_objects:
        # 白色化用のディレクトリを作成
        white_dir = f"{output_dir}_resize_white"
        white_images_dir = os.path.join(white_dir, "images")
        white_annotations_dir = os.path.join(white_dir, "annotations")

        os.makedirs(white_images_dir, exist_ok=True)
        os.makedirs(white_annotations_dir, exist_ok=True)

        whiten_background(
            resize_images_dir,
            white_images_dir,
            resize_allobj_dir,
            resize_annotations_dir,
            white_annotations_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description='CubiCasa5kデータセットを処理して部屋、壁、ドア、窓のセグメンテーション用データを作成'
    )
    parser.add_argument('--cubicasa_dir', type=str, required=True,
                        help='CubiCasa5kデータセットのディレクトリパス')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='処理済みデータの出力ディレクトリ')
    parser.add_argument('--process_all_objects', action='store_true',
                        help='背景白色化のために全オブジェクトを処理する')

    args = parser.parse_args()

    # ベースディレクトリの定義
    base_dirs = [
        os.path.join(args.cubicasa_dir, "high_quality_architectural"),
        os.path.join(args.cubicasa_dir, "colorful"),
        os.path.join(args.cubicasa_dir, "high_quality"),
    ]

    process_cubicasa_dataset(base_dirs, args.output_dir, args.process_all_objects)


if __name__ == "__main__":
    main()
