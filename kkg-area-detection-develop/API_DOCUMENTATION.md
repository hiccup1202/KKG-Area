# kkg_area_detection API ドキュメント

`kkg_area_detection`は、建築図面から領域（部屋、壁、ドア、窓など）を検出するためのPythonパッケージです。Mask2Formerベースのセグメンテーションモデルを使用し、OCRによる部屋名検出機能も提供します。

## インストール

```bash
pip install kkg-area-detection
```

## 基本的な使い方

```python
from PIL import Image
import kkg_area_detection

# モデルの初期化
kkg_area_detection.initialize_model()

# 画像の読み込みと領域検出
image = Image.open("floor_plan.png")
segmentation_result = kkg_area_detection.get_segmentation_result(image)

# 輪郭と頂点の抽出
contours = kkg_area_detection.get_approx_contours_and_vertices(
    segmentation_result['segmentation'].numpy()
)

# 結果の可視化
visualized = kkg_area_detection.visualize_contours(image, contours)
visualized.show()
```

## 主要な関数

### 1. initialize_model

Mask2Formerモデルとプロセッサーを初期化します。

```python
def initialize_model(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None
```

**引数:**
- `model_path` (str, optional): カスタムモデルのパス。指定された場合、最優先で使用されます
- `device` (str, optional): モデルをロードするデバイス（'cuda'、'cpu'、またはNoneで自動検出）
- `model_name` (str, optional): S3からダウンロードする前にキャッシュをチェックするモデル名

**使用例:**
```python
# デフォルトモデルを使用
kkg_area_detection.initialize_model()

# カスタムモデルを使用
kkg_area_detection.initialize_model(model_path="/path/to/model")

# GPUを明示的に指定
kkg_area_detection.initialize_model(device="cuda")
```

### 2. get_segmentation_result

画像にMask2Formerを適用し、セグメンテーション結果を取得します。

```python
def get_segmentation_result(image: Image.Image) -> Dict[str, Any]
```

**引数:**
- `image` (PIL.Image.Image): 入力画像

**戻り値:**
- `dict`: セグメンテーション結果を含む辞書
  - `'segmentation'`: セグメンテーションマップ（テンソル）
  - `'segments_info'`: セグメント情報のリスト

**使用例:**
```python
image = Image.open("floor_plan.png")
result = kkg_area_detection.get_segmentation_result(image)
segmentation_map = result['segmentation'].numpy()
segments_info = result['segments_info']
```

### 3. get_approx_contours_and_vertices

セグメンテーション配列から輪郭と頂点を抽出します。

```python
def get_approx_contours_and_vertices(
    segment_array: np.ndarray,
    epsilon: float = 0.015,
    use_smoothing: bool = False,
    smoothing_kernel_size: int = 5,
    smoothing_iterations: int = 1,
    use_angle_filter: bool = False,
    min_angle: float = 30.0,
    wall_filter: bool = False,
    segments_info: List[Dict[str, Any]] = None,
    target_label_ids: List[int] = [1, 2],
    edge_margin: float = 0.05,
    use_shapely: bool = True,
) -> List[ContourInfo]
```

**引数:**
- `segment_array` (np.ndarray): 2次元のセグメンテーション配列
- `epsilon` (float): 輪郭近似の精度パラメータ（0.0〜1.0、デフォルト: 0.015）
- `use_smoothing` (bool): スムージングを適用するか（デフォルト: False）
- `smoothing_kernel_size` (int): ガウシアンカーネルのサイズ（デフォルト: 5）
- `smoothing_iterations` (int): スムージングの反復回数（デフォルト: 1）
- `use_angle_filter` (bool): 鋭角をフィルタリングするか（デフォルト: False）
- `min_angle` (float): 保持する最小角度（度）（デフォルト: 30.0）
- `wall_filter` (bool): 壁フィルタリングを適用するか（デフォルト: False）
- `segments_info` (list): 壁フィルタリング用のセグメント情報
- `target_label_ids` (list): 壁フィルタリング用のターゲットラベルID（デフォルト: [1, 2]）
- `edge_margin` (float): 壁フィルタリング用のエッジマージン比率（デフォルト: 0.05）
- `use_shapely` (bool): Shapelyを使用して穴のあるポリゴンを処理（デフォルト: True）

**戻り値:**
- `List[ContourInfo]`: 輪郭情報のリスト。各要素は以下の形式:
  ```python
  {
      'id': int,  # セグメントID
      'vertices': List[{'x': int, 'y': int}],  # 頂点のリスト
      'holes': List[List[{'x': int, 'y': int}]]  # 穴の頂点リスト（optional）
  }
  ```

**使用例:**
```python
# 基本的な使用
contours = kkg_area_detection.get_approx_contours_and_vertices(segmentation_map)

# スムージングと角度フィルタリングを有効化
contours = kkg_area_detection.get_approx_contours_and_vertices(
    segmentation_map,
    epsilon=0.02,
    use_smoothing=True,
    use_angle_filter=True,
    min_angle=45.0
)

# 壁フィルタリングを適用
contours = kkg_area_detection.get_approx_contours_and_vertices(
    segmentation_map,
    wall_filter=True,
    segments_info=result['segments_info'],
    target_label_ids=[1, 2]  # 1: 壁, 2: ドア
)
```

### 4. visualize_contours

輪郭を画像上に可視化します。

```python
def visualize_contours(
    image: Image.Image,
    contours_list: List[ContourInfo],
    vertex_size: int = 3,
    font_size: int = 20,
    show_vertex_count: bool = True,
    show_vertices: bool = True,
) -> Image.Image
```

**引数:**
- `image` (PIL.Image.Image): 入力画像
- `contours_list` (list): 輪郭情報のリスト
- `vertex_size` (int): 頂点のサイズ（デフォルト: 3）
- `font_size` (int): 頂点数テキストのフォントサイズ（デフォルト: 20）
- `show_vertex_count` (bool): 頂点数を表示するか（デフォルト: True）
- `show_vertices` (bool): 頂点を表示するか（デフォルト: True）

**戻り値:**
- `PIL.Image.Image`: 可視化された画像

**使用例:**
```python
# 基本的な可視化
visualized = kkg_area_detection.visualize_contours(image, contours)

# 頂点を非表示にして、頂点数のみ表示
visualized = kkg_area_detection.visualize_contours(
    image, 
    contours,
    show_vertices=False,
    font_size=30
)
```

### 5. get_regions_with_room_names

セグメンテーション、輪郭抽出、部屋名検出を統合した関数です。

```python
def get_regions_with_room_names(
    image: Image.Image,
    image_path: str,
    azure_endpoint: str,
    azure_key: str,
    room_name_keywords: Optional[List[str]] = None,
    default_name: str = 'Room 1',
    epsilon: float = 0.015,
    use_smoothing: bool = False,
    smoothing_kernel_size: int = 5,
    smoothing_iterations: int = 1,
    use_angle_filter: bool = False,
    min_angle: float = 30.0,
    wall_filter: bool = False,
    target_label_ids: List[int] = [1, 2],
    edge_margin: float = 0.05,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]
```

**引数:**
- `image` (PIL.Image.Image): 入力画像
- `image_path` (str): 画像ファイルのパス（OCR用）
- `azure_endpoint` (str): Azure Form Recognizerエンドポイント
- `azure_key` (str): Azure Form Recognizer APIキー
- `room_name_keywords` (list, optional): 部屋名を示すキーワードのリスト
- `default_name` (str): 部屋名が見つからない場合のデフォルト名（デフォルト: 'Room 1'）
- その他の引数は`get_approx_contours_and_vertices`と同じ

**戻り値:**
- `tuple`: (輪郭情報のリスト, 領域IDから部屋名へのマッピング辞書)

**使用例:**
```python
# Azure資格情報の設定
azure_endpoint = "https://your-resource.cognitiveservices.azure.com/"
azure_key = "your-api-key"

# 部屋名キーワードの定義
room_keywords = ["リビング", "キッチン", "寝室", "浴室", "トイレ", "廊下"]

# 領域と部屋名の検出
regions, room_names = kkg_area_detection.get_regions_with_room_names(
    image,
    "floor_plan.png",
    azure_endpoint,
    azure_key,
    room_name_keywords=room_keywords,
    default_name="部屋"
)

# 結果の確認
for region_id, room_name in room_names.items():
    print(f"領域 {region_id}: {room_name}")
```

### 6. get_room_names

検出された領域の部屋名をOCRで取得します。

```python
def get_room_names(
    image_path: str,
    contours_list: List[Dict[str, Any]],
    azure_endpoint: str,
    azure_key: str,
    room_name_keywords: Optional[List[str]] = None,
    default_name: str = 'Room 1',
) -> Dict[int, str]
```

**引数:**
- `image_path` (str): 画像ファイルのパス
- `contours_list` (list): 輪郭情報のリスト
- `azure_endpoint` (str): Azure Form Recognizerエンドポイント
- `azure_key` (str): Azure Form Recognizer APIキー
- `room_name_keywords` (list, optional): 部屋名を示すキーワードのリスト
- `default_name` (str): 部屋名が見つからない場合のデフォルト名

**戻り値:**
- `dict`: 領域IDから部屋名へのマッピング辞書

**使用例:**
```python
# 先に輪郭を抽出
contours = kkg_area_detection.get_approx_contours_and_vertices(segmentation_map)

# 部屋名を取得
room_names = kkg_area_detection.get_room_names(
    "floor_plan.png",
    contours,
    azure_endpoint,
    azure_key,
    room_name_keywords=["リビング", "キッチン", "寝室"]
)
```

### 7. create_color_mask

検出された領域のカラーマスクを作成します。

```python
def create_color_mask(
    image_size: Tuple[int, int],
    regions: List[Dict[str, Any]],
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    fixed_class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    visible_classes: Optional[List[int]] = None,
) -> Image.Image
```

**引数:**
- `image_size` (tuple): 出力マスクのサイズ (width, height)
- `regions` (list): 領域辞書のリスト
- `color_map` (dict, optional): label_idからRGBカラーへのマッピング
- `fixed_class_colors` (dict, optional): 特定のlabel_idに固定色を設定
- `visible_classes` (list, optional): 表示するlabel_idのリスト

**戻り値:**
- `PIL.Image.Image`: カラーマスク画像

**使用例:**
```python
# カスタムカラーマップを定義
color_map = {
    1: (255, 0, 0),    # 壁: 赤
    2: (0, 0, 255),    # ドア: 青
    3: (0, 255, 0),    # 窓: 緑
    4: (255, 255, 0),  # 部屋: 黄色
}

# カラーマスクを作成
mask = kkg_area_detection.create_color_mask(
    (image.width, image.height),
    regions,
    color_map=color_map
)

# 特定のクラスのみ表示
mask = kkg_area_detection.create_color_mask(
    (image.width, image.height),
    regions,
    visible_classes=[1, 4]  # 壁と部屋のみ表示
)
```

### 8. create_color_mask_from_segmentation

セグメンテーションマップから直接カラーマスクを作成します。

```python
def create_color_mask_from_segmentation(
    segmentation_map: np.ndarray,
    segments_info: List[Dict[str, Any]],
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    use_random_colors: bool = True,
    fixed_class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    visible_classes: Optional[List[int]] = None,
) -> Image.Image
```

**引数:**
- `segmentation_map` (np.ndarray): セグメンテーションマップ
- `segments_info` (list): セグメント情報のリスト
- `color_map` (dict, optional): label_idからRGBカラーへのマッピング
- `use_random_colors` (bool): 未定義のセグメントにランダム色を使用（デフォルト: True）
- `fixed_class_colors` (dict, optional): 特定のlabel_idに固定色を設定
- `visible_classes` (list, optional): 表示するlabel_idのリスト

**戻り値:**
- `PIL.Image.Image`: カラーマスク画像

**使用例:**
```python
# セグメンテーション結果から直接マスクを作成
result = kkg_area_detection.get_segmentation_result(image)
mask = kkg_area_detection.create_color_mask_from_segmentation(
    result['segmentation'].numpy(),
    result['segments_info'],
    fixed_class_colors={
        1: (255, 0, 0),   # 壁: 赤
        2: (0, 0, 255)    # ドア: 青
    }
)
```

### 9. overlay_mask_on_image

カラーマスクを元画像に重ねます。

```python
def overlay_mask_on_image(
    image: Image.Image, 
    mask: Image.Image, 
    alpha: float = 0.5
) -> Image.Image
```

**引数:**
- `image` (PIL.Image.Image): 元画像
- `mask` (PIL.Image.Image): カラーマスク画像
- `alpha` (float): 透明度（0.0〜1.0、デフォルト: 0.5）

**戻り値:**
- `PIL.Image.Image`: マスクを重ねた画像

**使用例:**
```python
# マスクを半透明で重ねる
overlaid = kkg_area_detection.overlay_mask_on_image(image, mask, alpha=0.5)

# より不透明にする
overlaid = kkg_area_detection.overlay_mask_on_image(image, mask, alpha=0.8)
```

### 10. visualize_regions

検出された領域を画像上に可視化する統合関数です。

```python
def visualize_regions(
    image: Image.Image,
    regions: List[Dict[str, Any]],
    alpha: float = 0.5,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    segmentation_result: Optional[Dict[str, Any]] = None,
    fixed_class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    visible_classes: Optional[List[int]] = None,
    use_contours_for_visualization: bool = True,
) -> Image.Image
```

**引数:**
- `image` (PIL.Image.Image): 元画像
- `regions` (list): 領域辞書のリスト
- `alpha` (float): 透明度（デフォルト: 0.5）
- `color_map` (dict, optional): label_idからRGBカラーへのマッピング
- `segmentation_result` (dict, optional): `get_segmentation_result`からの結果
- `fixed_class_colors` (dict, optional): 特定のlabel_idに固定色を設定
- `visible_classes` (list, optional): 表示するlabel_idのリスト
- `use_contours_for_visualization` (bool): 輪郭を使用して可視化するか、生のマスクを使用するか（デフォルト: True）

**戻り値:**
- `PIL.Image.Image`: 領域を可視化した画像

**使用例:**
```python
# 基本的な可視化
visualized = kkg_area_detection.visualize_regions(image, regions)

# セグメンテーション結果を直接使用
result = kkg_area_detection.get_segmentation_result(image)
visualized = kkg_area_detection.visualize_regions(
    image,
    regions=[],  # 空のリストでOK
    segmentation_result=result,
    alpha=0.7
)

# 特定のクラスのみ可視化
visualized = kkg_area_detection.visualize_regions(
    image,
    regions,
    visible_classes=[1, 4],  # 壁と部屋のみ
    fixed_class_colors={
        1: (255, 0, 0),   # 壁: 赤
        4: (0, 255, 0)    # 部屋: 緑
    }
)
```

## 実践的な使用例

### 例1: 基本的な領域検出と可視化

```python
from PIL import Image
import kkg_area_detection

# 1. モデルの初期化
kkg_area_detection.initialize_model()

# 2. 画像の読み込み
image = Image.open("floor_plan.png")

# 3. セグメンテーション実行
result = kkg_area_detection.get_segmentation_result(image)

# 4. 輪郭抽出（スムージング付き）
contours = kkg_area_detection.get_approx_contours_and_vertices(
    result['segmentation'].numpy(),
    use_smoothing=True,
    epsilon=0.02
)

# 5. 結果の可視化
visualized = kkg_area_detection.visualize_contours(image, contours)
visualized.save("result_contours.png")

# 6. カラーマスクの作成と重ね合わせ
mask = kkg_area_detection.create_color_mask_from_segmentation(
    result['segmentation'].numpy(),
    result['segments_info']
)
overlaid = kkg_area_detection.overlay_mask_on_image(image, mask, alpha=0.6)
overlaid.save("result_overlay.png")
```

### 例2: 部屋名検出を含む完全な処理

```python
from PIL import Image
import kkg_area_detection
import os

# Azure認証情報
azure_endpoint = os.environ["AZURE_ENDPOINT"]
azure_key = os.environ["AZURE_API_KEY"]

# 部屋名キーワード
room_keywords = [
    "リビング", "LDK", "キッチン", "ダイニング",
    "寝室", "洋室", "和室", "子供部屋",
    "浴室", "バス", "トイレ", "WC",
    "廊下", "玄関", "クローゼット", "収納"
]

# モデルの初期化
kkg_area_detection.initialize_model()

# 画像の読み込み
image = Image.open("floor_plan.png")

# 領域と部屋名の検出（統合関数を使用）
regions, room_names = kkg_area_detection.get_regions_with_room_names(
    image,
    "floor_plan.png",
    azure_endpoint,
    azure_key,
    room_name_keywords=room_keywords,
    default_name="不明",
    use_smoothing=True,
    use_angle_filter=True,
    min_angle=45.0
)

# 結果の表示
print("検出された部屋:")
for region_id, room_name in room_names.items():
    region = next(r for r in regions if r['id'] == region_id)
    vertex_count = len(region['vertices'])
    print(f"  {room_name} (ID: {region_id}, 頂点数: {vertex_count})")

# 可視化（部屋ごとに異なる色で表示）
color_map = {}
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
          (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]

for i, region_id in enumerate(room_names.keys()):
    color_map[region_id] = colors[i % len(colors)]

visualized = kkg_area_detection.visualize_regions(
    image,
    regions,
    alpha=0.5,
    color_map=color_map
)
visualized.save("result_with_room_names.png")
```

### 例3: 特定のクラスのみを検出・表示

```python
# 壁とドアのみを検出して表示
result = kkg_area_detection.get_segmentation_result(image)

# 壁（label_id=1）とドア（label_id=2）のみの輪郭を抽出
contours = kkg_area_detection.get_approx_contours_and_vertices(
    result['segmentation'].numpy(),
    wall_filter=True,
    segments_info=result['segments_info'],
    target_label_ids=[1, 2]
)

# 可視化（壁は赤、ドアは青で固定）
visualized = kkg_area_detection.visualize_regions(
    image,
    [],  # regionsは空でOK
    segmentation_result=result,
    visible_classes=[1, 2],
    fixed_class_colors={
        1: (255, 0, 0),  # 壁: 赤
        2: (0, 0, 255)   # ドア: 青
    },
    alpha=0.7
)
visualized.save("walls_and_doors_only.png")
```

## 注意事項

1. **モデルの初期化**: 必ず最初に`initialize_model()`を呼び出してください
2. **Azure OCR**: 部屋名検出機能を使用する場合は、Azure Form RecognizerのAPIキーが必要です
3. **画像形式**: 入力画像はPIL Image形式である必要があります
4. **メモリ使用量**: 大きな画像を処理する場合は、十分なメモリが必要です
5. **GPU使用**: CUDAが利用可能な環境では自動的にGPUが使用されます
6. **階層的輪郭処理**: `use_shapely=True`の場合、すべてのセグメントで穴の検出が行われ、穴を含む多角形も正確に表現されます

## トラブルシューティング

### モデルのダウンロードエラー
- インターネット接続を確認してください
- S3へのアクセス権限を確認してください
- `model_path`パラメータでローカルモデルを指定することも可能です

### OCRエラー
- Azure APIキーとエンドポイントが正しいか確認してください
- 画像ファイルが存在し、読み取り可能か確認してください

### メモリエラー
- 画像サイズを小さくするか、バッチサイズを調整してください
- GPUメモリが不足している場合は、`device='cpu'`を指定してCPUで実行してください