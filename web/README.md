# KKG 壁面積計算ツール（Next.js）

これはコーディング課題向けのシンプルな Web アプリです。部屋／壁／ドアの輪郭（contour）を記述した JSON をアップロードし、輪郭を可視化（平面図にオーバーレイ）し、**部屋ごとの壁面積**を計算します。

## 要件（概要）

- JSON ファイルをアップロードできる（下の「入力 JSON 形式」を参照）
- 可視化を表示できる（平面図画像に輪郭を重ねる）
- 部屋ごとの壁面積のテーブルを表示できる
- 壁面積の計算条件：
  - スケール：`96.7 px = 1.365 m`
  - 天井高：`2.4 m`
  - ドアは高さ `2.0 m` 分を壁面積から差し引く

## 実行方法

```bash
npm install
npm run dev
```

その後、`http://localhost:3000` を開きます。

## 入力 JSON 形式（想定）

- `contours[]`：`vertices[]` と `label_id` を持つ領域／線のリスト
- `image_data.base64` と `image_data.mime_type`（任意。プレビュー表示用）

課題仕様（ラベル）：
- `label_id = 1`：room（部屋）
- `label_id = 2`：wall（壁）
- `label_id = 3`：door（ドア）

## 壁面積の計算方法

このアプリは、**部屋ポリゴンの境界に沿った壁の長さ**を（ピクセル単位で）求め、メートルに変換した上で天井高を掛け、さらにドア開口分を差し引くことで **部屋ごとの壁面積**を計算します。

### 入力 → 出力

**入力**
- JSON に含まれる輪郭（contour）集合：
  - `label_id = 1`：部屋ポリゴン（閉曲線）
  - `label_id = 2`：壁ポリライン（開曲線）
  - `label_id = 3`：ドアポリライン（開曲線）

**出力（部屋ごと）**
- `wall_length_m`：境界上で「壁に覆われている」合計長（m）
- `door_length_m`：境界上で「ドアに覆われている」合計長（m）
- `wall_area_m2`：壁面積（m²）＝（壁長 × 天井高）−（ドア長 × ドア高）

### 定数（課題仕様）

- **スケール**：`96.7 px = 1.365 m`  → \(m/px = 1.365 / 96.7\)
- **天井高**：`2.4 m`
- **ドア高**：`2.0 m`（壁面積から差し引く）

### 幾何モデル（ピクセル座標系）

- **部屋**（`label_id=1`）：頂点 `v0..v(n-1)` を持つポリゴン。境界エッジは
  - \(e_i = (v_i, v_{(i+1)\bmod n})\)
- **壁／ドア**（`label_id=2/3`）：ポリラインから得られる線分集合
  - \(s_j = (p_j, p_{j+1})\)（\(j=0..m-2\)）

計算はまず **ピクセル空間**で行い、最後にメートルへ変換します。

### 基本アイデア

壁（またはドア）の線分は、**その部屋のポリゴン境界と重なっている部分だけ**が、その部屋に寄与します。したがって問題は次の形になります：

- 各部屋の境界エッジ \(e\) について、壁線分／ドア線分と **重なっている区間**を求める
- 重なり区間をマージして **二重カウントを防ぐ**
- 各エッジの重なり長を合計し、部屋単位の壁長／ドア長を得る

### 詳細アルゴリズム

1) **輪郭を線分に前処理**
- `label_id=2` の全ポリラインから `wallSegments[]` を生成する
- `label_id=3` の全ポリラインから `doorSegments[]` を生成する

2) **部屋ごとに集計**
- `wall_overlap_px = 0`、`door_overlap_px = 0` を初期化
- 部屋ポリゴンの各境界エッジ \(e=(a,b)\) について：
  - `wall_overlap_px += overlapOnEdgePx(e, wallSegments)`
  - `door_overlap_px += overlapOnEdgePx(e, doorSegments)`

3) **`overlapOnEdgePx` の中身（ほぼ共線 + 近さ + 射影）**

部屋の境界エッジ \(e=(a,b)\) を固定し、次を定義します：
- エッジベクトル：\(\vec{u} = b-a\)、長さ \(L = |\vec{u}|\)
- 単位方向：\(\hat{u} = \vec{u}/L\)
- エッジ上のパラメータ表示：\(x(t) = a + t\vec{u}\)、\(t \in [0,1]\)

候補線分 \(s=(p,q)\) ごとに：

- **退化ケースの除外**：\(L \approx 0\) または \(|q-p|\approx 0\) の場合は無視

- **方向（平行）チェック**：線分方向がエッジ方向と十分近いこと
  - \(\vec{v} = q-p\)、\(\hat{v} = \vec{v}/|\vec{v}|\)
  - \(|\hat{u} \times \hat{v}| < \varepsilon_{\text{angle}}\)（2D クロス積の大きさ）

- **近さチェック**：線分がエッジと同一直線上にある（あるいは十分近い）こと
  - 点 \(r\) の直線（点 \(a\) と方向 \(\hat{u}\)）への距離：
    - \(d(r, \text{line}(a,\hat{u})) = |(r-a) \times \hat{u}|\)
  - \(d(p, \cdot)\) と \(d(q, \cdot)\)（または平均）が \(< \varepsilon_{\text{dist}}\)

- **エッジパラメータ \(t\) への射影（1D 区間化）**
  - \(t_p = \frac{(p-a)\cdot \vec{u}}{\vec{u}\cdot \vec{u}}\)
  - \(t_q = \frac{(q-a)\cdot \vec{u}}{\vec{u}\cdot \vec{u}}\)
  - エッジ上の区間：\([t_1,t_2] = [\min(t_p,t_q), \max(t_p,t_q)]\)
  - エッジの範囲にクランプ：\([t_1,t_2] \leftarrow [\max(0,t_1), \min(1,t_2)]\)
  - \(t_2 \le t_1\) なら重なりなし
  - それ以外は重なり区間 \([t_1,t_2]\) を記録

4) **区間マージ（エッジ内の二重カウント防止）**
- 区間を開始点 \(t_1\) でソートし、重なり／隣接（\(\varepsilon_{\text{merge}}\) 以内）をマージ
- マージ後の被覆量をピクセル長へ変換：
  - \(\text{overlap\_px}(e) = \sum_k (t_{2,k}-t_{1,k}) \cdot L\)

5) **メートルに変換して面積を計算**

- ピクセル→メートル変換：
  - \(m/px = 1.365 / 96.7\)
  - `wall_length_m = wall_overlap_px * (m/px)`
  - `door_length_m = door_overlap_px * (m/px)`

- 壁面積（部屋ごと）：
  - `wall_area_m2 = wall_length_m * 2.4 - door_length_m * 2.0`
  - 実装上は必要に応じて 0 未満を丸める（例：`max(0, wall_area_m2)`）

### 擬似コード（部屋の1エッジあたり）

```text
function overlapOnEdgePx(edge(a,b), segments):
  u = b - a
  L = |u|
  if L ~= 0: return 0

  intervals = []
  for each segment(p,q) in segments:
    if not parallel(u, q-p): continue
    if distanceToLine(p, a, u) > epsDist: continue
    if distanceToLine(q, a, u) > epsDist: continue

    tp = dot(p-a, u) / dot(u,u)
    tq = dot(q-a, u) / dot(u,u)
    t1 = clamp(min(tp,tq), 0, 1)
    t2 = clamp(max(tp,tq), 0, 1)
    if t2 > t1: intervals.push([t1,t2])

  merged = mergeIntervals(intervals, epsMerge)
  return sum((t2-t1) * L for [t1,t2] in merged)
```

### 前提／注意点

- **境界ベースの割り当て**：壁／ドアは「部屋ポリゴン境界との重なり」で部屋に紐づくため、2部屋で共有される壁は両方の部屋に（それぞれの境界として）寄与します。
- **許容誤差（トレランス）が必要**：アノテーションの微小なズレを吸収するため、\(\varepsilon_{\text{angle}}\)、\(\varepsilon_{\text{dist}}\)、\(\varepsilon_{\text{merge}}\) のような許容誤差が前提になります（値は実装詳細）。
- **ドアは高さで差し引く**：ドア長は境界に沿って測り、ドア高（2.0 m）を掛けて壁面積から差し引きます。
- **斜めの壁にも対応**：射影と区間処理により、軸に平行でない境界エッジも扱えます。
