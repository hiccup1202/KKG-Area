#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import cv2
from tqdm import tqdm


def align_resize_and_crop(
    input_images_dir,
    input_annotations_dir,
    output_images_dir,
    output_annotations_dir,
    input_allobj_dir=None,
    output_allobj_dir=None,
):
    """
    フロアプラン画像とマスク画像を処理する：
    1. 両方の画像を小さい方の幅と高さにクロップする
    2. 全オブジェクトマスクが存在する場合は、それもクロップする

    Args:
        input_images_dir: 元画像を含むディレクトリ
        input_annotations_dir: 元アノテーションを含むディレクトリ
        output_images_dir: 処理済み画像の保存先ディレクトリ
        output_annotations_dir: 処理済みアノテーションの保存先ディレクトリ
        input_allobj_dir: 全オブジェクトマスクを含むディレクトリ（オプション）
        output_allobj_dir: 処理済み全オブジェクトマスクの保存先ディレクトリ（オプション）
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    # 全オブジェクトマスク用のディレクトリが指定されている場合
    if (
        input_allobj_dir is not None
        and os.path.exists(input_allobj_dir)
        and output_allobj_dir is not None
    ):
        os.makedirs(output_allobj_dir, exist_ok=True)

    # 入力ディレクトリ内のすべての画像ファイルを取得
    image_files = [f for f in os.listdir(input_images_dir) if f.endswith('.png')]

    for image_file in tqdm(image_files):
        # ベースファイル名を取得（拡張子を除く）
        base_filename = os.path.splitext(image_file)[0]

        # 対応するマスクファイル名を構築
        color_mask_file = f"{base_filename}_color.png"
        annotation_file = f"{base_filename}_annotation.png"

        # ファイルパスを構築
        floorplan_path = os.path.join(input_images_dir, image_file)
        color_mask_path = os.path.join(input_annotations_dir, color_mask_file)

        # 必要なファイルが存在するか確認
        if not (os.path.exists(floorplan_path) and os.path.exists(color_mask_path)):
            continue

        # 画像を読み込む
        floorplan = cv2.imread(floorplan_path)
        mask_img = cv2.imread(color_mask_path)

        if floorplan is None or mask_img is None:
            continue

        # 各画像の幅・高さを取得
        h_f, w_f = floorplan.shape[:2]
        h_m, w_m = mask_img.shape[:2]

        # 小さい方の幅を目標とする
        target_width = min(w_f, w_m)

        # 小さい方の高さを目標とする
        target_height = min(h_f, h_m)

        # 左上を基準にクロップ
        floorplan_cropped = floorplan[0:target_height, 0:target_width]
        mask_cropped = mask_img[0:target_height, 0:target_width]

        # 出力ファイルパス
        out_floorplan = os.path.join(output_images_dir, image_file)
        out_mask = os.path.join(output_annotations_dir, color_mask_file)

        cv2.imwrite(out_floorplan, floorplan_cropped)
        cv2.imwrite(out_mask, mask_cropped)

        # アノテーションマスクも処理（存在する場合）
        annotation_path = os.path.join(input_annotations_dir, annotation_file)
        if os.path.exists(annotation_path):
            annotation_img = cv2.imread(annotation_path)
            if annotation_img is not None:
                annotation_cropped = annotation_img[0:target_height, 0:target_width]
                out_annotation = os.path.join(output_annotations_dir, annotation_file)
                cv2.imwrite(out_annotation, annotation_cropped)

        # 全オブジェクトマスクも処理（存在する場合）
        if input_allobj_dir is not None and output_allobj_dir is not None:
            obj_mask_file = f"{base_filename}_object_mask.png"
            obj_color_file = f"{base_filename}_object_color.png"

            obj_mask_path = os.path.join(input_allobj_dir, obj_mask_file)
            if os.path.exists(obj_mask_path):
                obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
                if obj_mask is not None:
                    # 同じサイズにクロップ
                    obj_mask_cropped = obj_mask[0:target_height, 0:target_width]
                    out_obj_mask = os.path.join(output_allobj_dir, obj_mask_file)
                    cv2.imwrite(out_obj_mask, obj_mask_cropped)

                    # カラーマスクも処理（存在する場合）
                    obj_color_path = os.path.join(input_allobj_dir, obj_color_file)
                    if os.path.exists(obj_color_path):
                        obj_color = cv2.imread(obj_color_path)
                        if obj_color is not None:
                            obj_color_cropped = obj_color[
                                0:target_height, 0:target_width
                            ]
                            out_obj_color = os.path.join(
                                output_allobj_dir, obj_color_file
                            )
                            cv2.imwrite(out_obj_color, obj_color_cropped)


def whiten_background(
    input_images_dir,
    output_images_dir,
    input_allobj_dir,
    input_annotations_dir=None,
    output_annotations_dir=None,
):
    """
    フロアプラン画像のオブジェクト外の領域を白色化する

    Args:
        input_images_dir: 元画像を含むディレクトリ
        output_images_dir: 処理済み画像の保存先ディレクトリ
        input_allobj_dir: オブジェクトマスクを含むディレクトリ
        input_annotations_dir: 元アノテーションを含むディレクトリ（オプション）
        output_annotations_dir: 処理済みアノテーションの保存先ディレクトリ（オプション）
    """
    os.makedirs(output_images_dir, exist_ok=True)

    if output_annotations_dir is not None:
        os.makedirs(output_annotations_dir, exist_ok=True)

    # 入力ディレクトリ内のすべての画像ファイルを取得
    image_files = [f for f in os.listdir(input_images_dir) if f.endswith('.png')]

    for image_file in tqdm(image_files):
        # ベースファイル名を取得（拡張子を除く）
        base_filename = os.path.splitext(image_file)[0]

        # 対応するオブジェクトマスクファイル名を構築
        obj_mask_file = f"{base_filename}_object_mask.png"

        # ファイルパスを構築
        floorplan_path = os.path.join(input_images_dir, image_file)
        mask_path = os.path.join(input_allobj_dir, obj_mask_file)

        # 必要なファイルが存在するか確認
        if not (os.path.exists(floorplan_path) and os.path.exists(mask_path)):
            continue

        # 画像を読み込む
        floorplan = cv2.imread(floorplan_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if floorplan is None or mask is None:
            continue

        # マスクが0の領域（オブジェクト外）を白色化
        white_area = mask == 0
        floorplan[white_area] = [255, 255, 255]  # BGR白

        # 白色化した画像を保存
        white_image_file = f"white_{image_file}"
        output_file = os.path.join(output_images_dir, white_image_file)
        cv2.imwrite(output_file, floorplan)

        # アノテーションをコピー（指定されている場合）
        if input_annotations_dir is not None and output_annotations_dir is not None:
            # 対応するアノテーションファイル名を構築
            color_mask_file = f"{base_filename}_color.png"
            annotation_file = f"{base_filename}_annotation.png"

            # カラーマスク
            color_mask_path = os.path.join(input_annotations_dir, color_mask_file)
            if os.path.exists(color_mask_path):
                color_mask = cv2.imread(color_mask_path)
                if color_mask is not None:
                    out_color_mask = os.path.join(
                        output_annotations_dir, color_mask_file
                    )
                    cv2.imwrite(out_color_mask, color_mask)

            # アノテーションマスク
            annotation_path = os.path.join(input_annotations_dir, annotation_file)
            if os.path.exists(annotation_path):
                annotation = cv2.imread(annotation_path)
                if annotation is not None:
                    out_annotation = os.path.join(
                        output_annotations_dir, annotation_file
                    )
                    cv2.imwrite(out_annotation, annotation)
