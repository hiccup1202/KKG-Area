# 部屋・壁・ドア・窓検出のためのデータ準備

このディレクトリには、CubiCasa5kデータセットから部屋、壁、ドア、窓検出のためのトレーニングデータを準備するためのツールが含まれています。

## ドアと窓検出のためのデータ準備

`data_preparation_door_window.py`スクリプトは、CubiCasa5kデータセットを処理して部屋、壁、ドア、窓のマスクを抽出します。これらのマスクはセグメンテーションモデルのトレーニングに使用できます。

### 前提条件

スクリプトを実行する前に、必要な依存関係をインストールする必要があります：

```bash
pip install numpy opencv-python svg.path tqdm
```

また、CubiCasa5kデータセットをダウンロードする必要があります。[Kaggle](https://www.kaggle.com/datasets/qmarva/cubicasa5k)または[公式ウェブサイト](https://github.com/CubiCasa/CubiCasa5k)からダウンロードできます。

### 使用方法

CubiCasa5kデータセットを処理するには：

```bash
python data_preparation_door_window.py --cubicasa_dir /path/to/cubicasa5k --output_dir processed_data
```

#### 引数

- `--cubicasa_dir`: CubiCasa5kデータセットのディレクトリパス（必須）
- `--output_dir`: 処理済みデータの出力ディレクトリ（デフォルト: 'processed_data'）
- `--process_all_objects`: 背景白色化のために全オブジェクトを処理するフラグ

### 出力

スクリプトは以下の出力を生成します：

1. 元のフロアプラン画像: `original_floorplan_{i}.png`
2. 部屋、壁、ドア、窓のカラーマスク: `room_wall_door_window_mask_color_{i}.png`
3. インスタンスIDとクラスIDを持つアノテーションマスク: `room_wall_door_window_annotation_{i}.png`
4. リサイズおよびアライメントされた画像: `{output_dir}_resize/`
5. `--process_all_objects`が指定されている場合、背景が白色化された画像: `{output_dir}_resize_white/`

### カラーコーディング

カラーマスクは以下の色スキームを使用します：

- 壁: 白 (255, 255, 255)
- ドア: 青 (255, 0, 0) BGRフォーマット
- 窓: オレンジ (0, 165, 255) BGRフォーマット
- 部屋: ランダムな色（各部屋は固有の色を持つ）

### アノテーションフォーマット

アノテーションマスクは以下の形式で情報を格納します：

- 青チャネル（インデックス0）: 常に0
- 緑チャネル（インデックス1）: インスタンスID
- 赤チャネル（インデックス2）: クラスID
  - 0: 部屋
  - 1: ドア
  - 2: 窓
  - 3: 壁

## 処理パイプライン

スクリプトは以下の手順を実行します：

1. CubiCasa5kデータセットからSVGデータを抽出
2. クラス属性に基づいて部屋、壁、ドア、窓を識別
3. カラーマスクとアノテーションマスクを作成
4. 一貫した寸法を確保するために画像をアライメントおよびリサイズ
5. オプションでオブジェクト外の背景を白色化

## コード構成

このパッケージは以下のモジュールで構成されています：

- `svg_utils.py`: SVG関連のユーティリティ関数
- `mask_extraction.py`: マスク抽出関連の関数
- `image_processing.py`: 画像処理関連の関数
- `data_preparation_door_window.py`: メインスクリプト

## 使用例

```bash
# 基本的な使用法
python data_preparation_door_window.py --cubicasa_dir ./cubicasa5k

# 背景白色化あり
python data_preparation_door_window.py --cubicasa_dir ./cubicasa5k --process_all_objects

# カスタム出力ディレクトリ
python data_preparation_door_window.py --cubicasa_dir ./cubicasa5k --output_dir ./my_processed_data
```
