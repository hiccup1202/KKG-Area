# S3モデルダウンロード機能

このドキュメントではkkg-area-detectionにおけるS3モデルダウンロード機能の使用方法について説明します。

## 概要

kkg-area-detectionライブラリは、Amazon S3からモデルを自動的にダウンロードする機能をサポートしています。この機能により、以下のことが可能になります：

1. 会社のメンバーのみがアクセス可能な中央のS3バケットにモデルを保存
2. ライブラリの初期化時に自動的にモデルをダウンロード
3. ダウンロードしたモデルをローカルにキャッシュして繰り返しのダウンロードを回避
4. ディレクトリ構造全体のダウンロードに対応（transformersモデル形式など）

## 設定

S3モデルダウンロード機能を使用するには、以下の環境変数を設定する必要があります：

```
AWS_ACCESS_KEY_ID=あなたのアクセスキー
AWS_SECRET_ACCESS_KEY=あなたのシークレットキー
AWS_REGION=ap-northeast-1  # デフォルトはap-northeast-1
S3_BUCKET_NAME=あなたのバケット名
S3_MODEL_KEY=S3内のモデルへのパス
MODEL_CACHE_DIR=/ローカルキャッシュへのパス  # オプション、デフォルトはパッケージディレクトリ内の'model_cache'
```

これらの変数は`.env`ファイルに設定し、ライブラリは自動的に`python-dotenv`を使用して読み込みます。

## モデル読み込みの優先順位

モデル読み込みシステムは以下の優先順位に従います：

1. `model_path`パラメータが指定された場合、そのパスからモデルを直接読み込みます
2. `model_path`が指定されず、`model_name`が指定された場合:
   - まずローカルキャッシュをチェックし、存在すればそこから読み込みます
   - キャッシュになければ、S3からダウンロードを試みます（環境変数が適切に設定されている場合）
3. どちらも指定されていない場合はエラーが発生します

## モデル名とキャッシング

`model_name`パラメータを使用すると、一度ダウンロードしたモデルはローカルにキャッシュされ、次回以降は自動的にキャッシュから読み込まれます：

```python
# 初回実行時: S3からダウンロードしてキャッシュ
kkg_area_detection.initialize_model(model_name="large", device="cuda")

# 2回目以降: 自動的にキャッシュから読み込む
kkg_area_detection.initialize_model(model_name="large", device="cuda")
```

キャッシュディレクトリはデフォルトで`<パッケージディレクトリ>/model_cache/`に設定されていますが、`MODEL_CACHE_DIR`環境変数でカスタマイズできます。

## ディレクトリ構造のダウンロード

`S3_MODEL_KEY`が`/`で終わる場合、その内容はディレクトリとして扱われ、すべてのファイルがダウンロードされます。これは以下のような場合に便利です：

- transformersのモデル形式（複数のファイルで構成）
- 設定ファイルと重みファイルのセット
- カスタムレイヤーや追加ファイルを含むモデル

```
S3_MODEL_KEY=models/mask2former/large/  # ディレクトリ全体をダウンロード
```

## 使用例

### 基本的な使用法

```python
import os
from dotenv import load_dotenv
import kkg_area_detection
from PIL import Image

# .envファイルから環境変数を読み込む (ライブラリ内でも自動的に読み込まれます)
load_dotenv()

# モデル名を指定して初期化（キャッシュまたはS3からダウンロード）
kkg_area_detection.initialize_model(model_name="large", device="cuda")

# 画像を処理
image = Image.open("your_image.jpg")
result = kkg_area_detection.get_segmentation_result(image)
```

### 明示的なローカルパスを使用

```python
kkg_area_detection.initialize_model(
    model_path="/path/to/local/model",
    device="cuda"
)
```

### コマンドラインツールでの使用

```bash
# S3からモデルをダウンロードして使用
python examples/basic_usage.py -i input.jpg -n large -d cuda

# ローカルモデルを使用
python examples/basic_usage.py -i input.jpg -p /path/to/model -d cuda
```

## セキュリティに関する考慮事項

1. **IAMアクセス権限**: 最小限の権限（特定のS3バケット/パスへの読み取り専用アクセス）を持つ専用のIAMユーザーを作成
2. **キーローテーション**: AWSアクセスキーを定期的にローテーション
3. **環境変数**: 実際の認証情報を含む`.env`ファイルをバージョン管理にコミットしない
4. **本番環境**: 本番環境ではアクセスキーの代わりにIAMロールの使用を検討

## トラブルシューティング

S3モデルダウンロードに問題が発生した場合：

1. boto3がインストールされていることを確認（`pip install boto3`）
2. AWS認証情報が正しいことを確認
3. S3バケットとオブジェクトが存在しアクセス可能であることを確認
4. 詳細なエラーメッセージについてログを確認
5. キャッシュの問題が疑われる場合は、`MODEL_CACHE_DIR`ディレクトリを削除して再試行
