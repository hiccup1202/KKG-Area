# KKG Area Detection

画像内のエリアを自動検出し, 可視化するためのPythonパッケージです. Meta AI研究所が開発したMask2Formerモデルを使用して, 高精度なエリア検出を実現しています.

## 機能

- 画像内のエリア（領域）の自動検出
- 検出エリアの座標情報の取得
- 検出エリアの可視化（カスタマイズ可能な色とアルファ値）
- 簡単に使えるPythonインターフェース

## インストール

### 必要条件

- Python 3.11以上
- pip または uv

### インストール方法

```bash
# uvを使用する場合
uv pip install git+ssh://git@github.com/KK-Generation/kkg-area-detection.git

# pipを使用する場合
pip install git+ssh://git@github.com/KK-Generation/kkg-area-detection.git
```


<details>
<summary>開発環境の構築手順</summary>

#### 前提条件
- Python 3.11以上
- Visual C++ Build Tools（Windowsの場合）

#### 1. 仮想環境の作成と有効化
```bash
# 仮想環境の作成
uv venv --python 3.11

# Linux/Macでの有効化
source .venv/bin/activate

# Windowsでの有効化
# .\.venv\Scripts\activate

# 依存関係のインストール
uv sync
```

### 2. パッケージの開発モードでのインストール
```bash
# 開発モードでインストール
pip install -e .
```

### 3. Jupyter Notebookのセットアップ（オプション）
```bash
# Jupyter関連パッケージのインストール
pip install -e "[.notebook]"

# 必要に応じて関連パッケージをインストール
pip install matplotlib

# カーネルの登録
python -m ipykernel install --user --name=kkg_area_detection --display-name="KKG Area Detection"
```

#### その他のエラーが発生する場合
- 仮想環境が正しく有効化されているか確認
- Pythonのバージョンが3.11以上であることを確認
- 必要なビルドツールがインストールされていることを確認

</details>

## モデルの利用方法

KKG Area Detectionは、Mask2Formerモデルを使用してエリア検出を行います。モデルの準備と読み込みには複数のオプションがあります。

### モデルの読み込み方法

```python
import kkg_area_detection

# 方法1: 明示的なローカルパスからモデルを読み込む
kkg_area_detection.initialize_model(
    model_path="/path/to/your/local/model",
    device="cuda"  # または "cpu"
)

# 方法2: モデル名を指定して、キャッシュまたはS3から読み込む
kkg_area_detection.initialize_model(
    model_name="large_model",
    device="cuda"  # または "cpu"
)
```

### モデルの読み込み優先順位

1. `model_path`が指定された場合、そのパスからモデルを直接読み込みます
2. `model_path`が指定されず、`model_name`が指定された場合:
   - まずローカルキャッシュをチェックし、存在すればそこから読み込みます
   - キャッシュになければ、S3からダウンロードを試みます
3. どちらも指定されていない場合はエラーが発生します

### S3からのモデルダウンロード設定

S3からモデルをダウンロードする場合、以下の環境変数を設定する必要があります：

```bash
# S3バケット名 (必須)
export S3_BUCKET_NAME=your-bucket-name

# S3のモデルキー (必須)
export S3_MODEL_KEY=path/to/model/

# AWSリージョン (オプション、デフォルトはap-northeast-1)
export AWS_REGION=ap-northeast-1

# AWS認証情報 (必須)
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# モデルキャッシュディレクトリ (オプション)
export MODEL_CACHE_DIR=/path/to/cache/dir
```

これらの環境変数は`.env`ファイルに保存することもできます。

### ディレクトリ構造のモデル

`S3_MODEL_KEY`が`/`で終わる場合、その内容はディレクトリとして扱われ、すべてのファイルがダウンロードされます。この機能は、transformersのモデル形式のように複数のファイルを含むモデルに便利です。

```bash
# ディレクトリモデルの例
export S3_MODEL_KEY=models/mask2former/large/
```

### モデルのキャッシュ

一度ダウンロードされたモデルは、ローカルにキャッシュされます。同じ`model_name`が指定された場合、すでにキャッシュされていれば再ダウンロードは行われません。

キャッシュディレクトリはデフォルトで`<パッケージディレクトリ>/model_cache/`に設定されていますが、`MODEL_CACHE_DIR`環境変数でカスタマイズできます。

### その他のオプション

- **デバイス選択**: `device`パラメータで処理を行うデバイスを指定できます（"cuda"または"cpu"）
- **カスタムモデル**: 追加の学習を施したカスタムモデルも、同様の方法で読み込めます

使用例：

```python
import kkg_area_detection
from PIL import Image

# 環境変数でS3設定を指定している場合
kkg_area_detection.initialize_model(model_name="large")

# 画像を読み込む
image = Image.open("example.jpg")

# 以降の処理は通常通り
result = kkg_area_detection.get_segmentation_result(image)
```

## 使用方法

### 基本的な使い方

```python
from PIL import Image
import kkg_area_detection
import numpy as np
import torch
# kkg_area_detectionの初期化
# 必要に応じてモデルパスやデバイスを指定できます
# kkg_area_detection.initialize_model(model_path="path/to/custom/model", device="cuda")
kkg_area_detection.initialize_model()

# 画像を読み込む
image_path = "path/to/your/image.jpg" # ここを実際の画像パスに置き換えてください
image = Image.open(image_path)

# セグメンテーション結果を取得
segmentation_result = kkg_area_detection.get_segmentation_result(image)

# セグメンテーションマップをNumPy配列に変換
# segmentation_result['segmentation'] は PyTorch テンソルです
segmentation_map_tensor = segmentation_result.get("segmentation")
# CPUに転送してからNumPy配列に変換
segmentation_map_numpy = segmentation_map_tensor.cpu().numpy()

# 各セグメントの情報を取得
segments_info = segmentation_result.get("segments_info", [])

# セグメンテーションマップから輪郭座標を抽出
# オプションでスムージングや角度フィルタリングも利用可能
contours_data = kkg_area_detection.get_approx_contours_and_vertices(
    segmentation_map_numpy,
    epsilon=0.015,  # 近似の精度 (0で近似なし、値を大きくすると頂点数が減る)
    use_smoothing=False, # 必要に応じてTrueに (ノイズが多い場合に有効)
    # smoothing_kernel_size=5,
    # smoothing_iterations=1,
    use_angle_filter=False, # 必要に応じてTrueに (直線部分の不要な頂点を削除)
    # min_angle=30.0,
    wall_filter=False, # 必要に応じてTrueに （壁の外にある領域を除去）
    # segments_info=segments_info,
    # target_label_ids=[1, 2],
    # edge_margin=0.05
)

# segments_info (スコア、ラベルID) と contours_data (座標) をマージ
# segment_id をキーとして結合
segment_details_map = {info['id']: info for info in segments_info}

regions_with_coords = []
for contour_info in contours_data:
    # contours.py は segment_id を 1から始めるが、segments_info は 0 から始まる場合があるため注意
    # Mask2Formerの実装に依存するが、多くの場合 segment_id はラベルIDとは別管理
    segment_id_in_contour = contour_info['id'] # contours.py が返すID (通常1始まり)

    matched_info = segment_details_map.get(segment_id_in_contour)

    if matched_info:
        regions_with_coords.append({
            'score': matched_info['score'],
            'label_id': matched_info['label_id'],
            'segment_id': segment_id_in_contour, # contours.pyから取得したID
            'coordinates': contour_info['vertices'] # 抽出した頂点リスト [{'x': x, 'y': y}, ...]
        })

# 検出結果を表示
print(f"検出されたエリア数 (座標あり): {len(regions_with_coords)}")
for i, region in enumerate(regions_with_coords):
    print(f"エリア {i+1}:")
    print(f"  信頼度: {region['score']:.4f}")
    print(f"  ラベルID: {region['label_id']}")
    print(f"  セグメントID: {region['segment_id']}")
    # region['coordinates'] は {'x': x座標, 'y': y座標} の辞書のリスト
    print(f"  頂点数: {len(region['coordinates'])}")
    if region['coordinates']:
        print(f"    最初の頂点: {region['coordinates'][0]}")


# 検出エリアを可視化
output_path = "detected_areas_with_contours.jpg"
visualized_image = kkg_area_detection.visualize_regions(
    image,
   regions_with_coords, # 座標情報を含むリストを渡す
   # color_map = {...}, # オプションで色指定
   # alpha=0.5         # オプションで透明度指定
)
# 結果を保存
visualized_image.save(output_path)
print(f"可視化結果を {output_path} に保存しました。")
```

### カスタムモデルの使用

独自の学習済みモデルを使用する場合：

```python
import kkg_area_detection

# カスタムモデルを初期化
kkg_area_detection.initialize_model(
    model_path="path/to/your/custom/model",
    device="cuda"  # GPUを使用（"cpu"も指定可能）
)

# 以降は通常通り使用可能
```

### 可視化のカスタマイズ

#### 全クラスの色を統一して指定する場合

```python
# シアン色のカラーマップを作成
cyan_color = (0, 255, 255)  # RGB形式
color_map = {region['label_id']: cyan_color for region in regions}

# カスタムカラーマップで可視化
visualized_image = kkg_area_detection.visualize_regions(
    image,
    regions,
    alpha=0.5,  # 透明度
    color_map=color_map  # カスタムカラーマップ
)
```

#### 特定のクラスのみに色を指定する場合

特定のクラスだけに固定色を指定し、その他のクラスにはランダムな色を割り当てることができます：

```python
# クラス1（exwall）には赤色、クラス2（inwall）には青色を指定
fixed_class_colors = {
    1: (255, 0, 0),   # exwall: 赤
    2: (0, 0, 255)    # inwall: 青
}

# 特定のクラスのみ色を固定して可視化
visualized_image = kkg_area_detection.visualize_regions(
    image,
    regions,
    alpha=0.6,
    segmentation_result=segmentation_result,
    fixed_class_colors=fixed_class_colors  # 特定クラスのみ色指定
)
```

#### 特定のクラスのみを表示する場合

表示したいクラスのIDを指定して、特定のクラスのみを可視化できます：

```python
# クラス2（inwall）のみを表示
visible_classes = [2]

visualized_image = kkg_area_detection.visualize_regions(
    image,
    regions,
    alpha=0.6,
    segmentation_result=segmentation_result,
    visible_classes=visible_classes  # 表示するクラスIDのリスト
)
```

#### label_idとクラス名の対応について

各label_idが何を表すかは、使用しているモデルの`config.json`ファイルに記載されています。例えば：

```json
"id2label": {
    "0": "room",    // 部屋
    "1": "exwall",  // 外壁
    "2": "inwall"   // 内壁
}
```

この対応関係を確認することで、どのlabel_idにどの色を割り当てるべきか判断できます。

## 例

`examples/` ディレクトリに使用例があります：

### 基本的な使用例

```bash
# 基本的な使用例を実行
python examples/basic_usage.py -i path/to/your/image.jpg

# カスタムモデルを使用
python examples/basic_usage.py -i image.jpg -p path/to/your/model

# S3/キャッシュされたモデルを名前で指定
python examples/basic_usage.py -i image.jpg -n large_wall

# 特定のクラスに色を指定（クラス1を赤、クラス2を青に）
python examples/basic_usage.py -i image.jpg --class-colors "1:255,0,0" "2:0,0,255"

# クラス2（inwall）のみを表示
python examples/basic_usage.py -i image.jpg --visible-classes 2

# 複数のオプションを組み合わせる例
python examples/basic_usage.py -i image.jpg \
    --visible-classes 1 2 \
    --class-colors "1:255,0,0" "2:0,0,255" \
    --alpha 0.7 \
    --output my_result.jpg
```

### コマンドライン引数の詳細

#### 基本オプション

- `-i, --image PATH`: 入力画像のパス（必須）
- `-o, --output PATH`: 出力画像の保存パス（デフォルト: `detected_areas_output.jpg`）
- `-d, --device {cuda,cpu}`: 処理に使用するデバイス（デフォルト: 自動選択）

#### モデル指定オプション（いずれか1つを指定）

- `-p, --model-path PATH`: ローカルモデルディレクトリのパス
- `-n, --model-name NAME`: キャッシュまたはS3からロードするモデル名

#### 輪郭抽出オプション

- `-e, --epsilon FLOAT`: 輪郭近似の精度パラメータ（デフォルト: 0.015）
  - 0.0: 近似なし（元の輪郭をそのまま使用）
  - 値を大きくすると頂点数が減り、より単純な多角形になります

- `-w, --wall-filter`: 壁フィルターを適用
  - 壁（label_id 1, 2）の外側にある領域を除去します
  
- `-t, --target-label-ids ID1 ID2...`: 壁フィルターで参照するラベルID（デフォルト: 1 2）

- `--smoothing`: 輪郭のスムージングを適用
  - ノイズが多い画像で有効です

- `--angle-filter`: 角度フィルタリングを適用
  - 直線部分の不要な頂点を削除します

#### 可視化オプション

- `-a, --alpha FLOAT`: 透明度（0.0〜1.0、デフォルト: 0.6）
  - 0.0: 完全に透明
  - 1.0: 完全に不透明

- `--visible-classes ID1 ID2...`: 表示するクラスIDのリスト
  - 例: `--visible-classes 2` はクラス2のみを表示

- `--class-colors "ID:R,G,B"...`: 特定クラスの色を指定
  - 例: `--class-colors "1:255,0,0" "2:0,0,255"`
  - クラス1を赤（255,0,0）、クラス2を青（0,0,255）に設定

### 高度な使用例

#### 壁フィルターを使用した例

```bash
# 壁の内側の領域のみを検出
python examples/basic_usage.py -i floor_plan.jpg -w --epsilon 0.02

# 特定のラベルIDを壁として扱う
python examples/basic_usage.py -i floor_plan.jpg -w --target-label-ids 1 2 3
```

#### 輪郭処理の最適化

```bash
# ノイズの多い画像に対してスムージングと角度フィルタを適用
python examples/basic_usage.py -i noisy_image.jpg \
    --smoothing \
    --angle-filter \
    --epsilon 0.03
```

#### 出力のカスタマイズ

```bash
# 特定のクラスのみを半透明で表示し、固定色を使用
python examples/basic_usage.py -i building_plan.jpg \
    --visible-classes 1 2 \
    --class-colors "1:255,100,100" "2:100,100,255" \
    --alpha 0.5 \
    --output result_walls_only.png
```

### 面積フィルタリング

basic_usage.pyは自動的に以下の処理を行います：

1. **小さな領域の除去**: 画像全体の0.1%未満の面積を持つ領域は自動的にフィルタリングされます
2. **複数輪郭の処理**: 1つのセグメントに複数の輪郭がある場合、最大の輪郭とその20%以上の面積を持つ輪郭のみが保持されます
3. **面積情報の表示**: 各領域について、元のマスク面積と輪郭面積の両方が表示されます

## 依存パッケージ

- torch: PyTorchディープラーニングフレームワーク
- torchvision: PyTorch用の画像処理ライブラリ
- transformers: Hugging Face Transformersライブラリ
- pillow: 画像処理ライブラリ
- numpy: 数値計算ライブラリ
- opencv-contrib-python: OpenCVコンピュータビジョンライブラリ

## 注意事項

- GPUを使用する場合は、`initialize_model(device="cuda")` を呼び出してください
