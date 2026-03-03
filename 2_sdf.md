以下は **SDF層（`wafergeo/sdf/*`）**についての **設計図レベルの詳細仕様**です。
全体設計（Artifact駆動、Observerで2Dへ統一、SDF/meshサロゲート両立、multi-material≤5） は前回の通りで、ここでは **SDF層だけをアーキテクト観点で徹底的に具体化**します。

---

# 1. SDF層の責務と非責務（この層が「どこまで」やるか）

## 1.1 SDF層の役割（アーキテクト視点）

SDF層は、**LabelVolume（multi-material label）を、学習・同化・変換（mesh/observer）に適した連続表現（TSDF等）へ変換し、追加の境界特徴を生成する**層です。

この層でやること（責務）：

1. **材料ごとの符号付き距離場（SDF）を生成**

   * 基本は EDT（Euclidean Distance Transform）
   * multi-material ≤5 なので **材料別にループ**して安定運用

2. **TSDF化（truncation + 正規化）**

   * `TSDF = clip(SDF, -μ, +μ)/μ` で `[-1,1]`
   * 学習・同化・メッシュ化（0等値面）に使いやすい形式に揃える

3. **境界補助特徴（軽量・効きが良い）**

   * `d_boundary = min_m |phi_m|`
   * `pair_code = encode(closest_material, second_closest_material)`（界面種別）
   * ※材料数が小さいほど安定して有効

4. **narrow-band（band）生成・管理**

   * band内（`|phi| < w`）だけ精密計算/保存する仕組み（大規模対応）

5. **再初期化（reinit）**

   * SDF性（距離性）を回復するための再SDF化（標準はEDTベースで堅牢に）

6. **SDF/TSDFのQA（品質保証）**

   * TSDF値域、符号整合、band内勾配統計などを出力（Artifactへ保存）

7. **TSDF→Labelの復元（任意だが重要）**

   * SDF/meshサロゲートの推論出力（TSDF stack）を、observer/topdown等で使うためのラベル化が必要になるため

---

SDF層でやらないこと（非責務）：

* `.vti` 読み込み（I/O）はしない（`io/`で完結）
* ラベルの意味付け（材料定義の整備）は `label/` が責務
* mesh化・点群化は `mesh/` が責務（SDF層は0等値面抽出に必要な場を供給）
* SEM輪郭→2D TSDFは `observe/` が責務（ただし2D TSDF生成ユーティリティは再利用可）

---

# 2. コード構成（SDF層のファイル責務とレビューしやすさ）

前回の構成のまま、SDF層の各ファイルの責務を“具体化”します。

```
wafergeo/sdf/
  edt.py               # SDFエンジン（EDTバックエンド）+ phi生成
  tsdf.py              # TSDF変換・復元・label化・基本ops
  band.py              # narrow-band/ROI/marginなど領域制御
  boundary_features.py # d_boundary / pair_code 等の境界特徴
  reinit.py            # 再初期化（EDT再計算等）
  qa.py                # SDF/TSDF品質保証（統計・閾値判定）
```

## 2.1 SDF層の設計原則（第三者が改修しやすい）

* **Functional core, imperative shell**

  * SDF層はI/Oを持たず、入力→出力が決まる **純粋関数/純粋クラス**中心
* **アルゴリズム切替（バックエンド）とパイプラインを分離**

  * EDTの実装（SciPy/CuPy/他）は `edt.py` に閉じ込める
  * TSDF化・band・特徴生成は別ファイルに分離
* **“返すもの”を固定（契約）**

  * 下流（mesh/observe/surrogate/assimilation）が前提にできるように
    `TSDFVolume(tsdf[M,Z,Y,X], mu_nm, material ordering, meta)` を固定

---

# 3. SDF層の主要データ型（I/O契約）

SDF層の入力は `LabelVolume`、出力は `TSDFVolume` が基本です（前回の型のまま）。

### 入力

* `LabelVolume`

  * `material_id`：shape `(Z,Y,X)`、dtype `uint8/uint16`
  * `GridSpec`：spacing/origin、axis_order=`ZYX`、sample_location=`cell_center`
  * `MaterialSpec`：ids（≤5）、void_id など

### 出力

* `TSDFVolume`

  * `tsdf`：shape `(M,Z,Y,X)`、dtype `float16（保存）/float32（計算）`、値域 `[-1,1]`
  * `mu_nm`：TSDFのtruncation幅（nm）
  * optional：

    * `d_boundary`：shape `(Z,Y,X)` float16（0..1正規化を推奨）
    * `pair_code`：shape `(Z,Y,X)` uint8（界面種別、band外は255など）
    * `present_mask`：shape `(M,)` bool（その材料が存在するか）

---

# 4. SDF層の設定（Config）— 運用でブレないためのパラメータ固定

SDF層は「学習・同化・mesh化に影響が大きいパラメータ」を持つので、**Profile/YAMLで固定**し、Metaに埋め込みます。

## 4.1 SDFBuildConfig（設計案）

```python
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

EDTBackend = Literal["scipy", "cupy"]  # optionalで増やせる

@dataclass(frozen=True)
class SDFBuildConfig:
    mu_nm: float                         # TSDF truncation
    backend: EDTBackend = "scipy"
    compute_tsdf_per_material: bool = True
    include_void_channel: bool = True    # voidもTSDFスタックに含めるか
    compute_binary_tsdf: bool = False    # solid/voidの二値TSDFも作るか
    boundary_features: bool = True       # d_boundary / pair_code
    pair_code_outside_band: int = 255    # band外のpair_code
    band_only_pair_code: bool = True     # pair_codeをband内だけ意味ある値にする
    # ROI最適化（大規模対応）
    roi_zyx: Optional[tuple[slice, slice, slice]] = None
    roi_margin_nm: Optional[float] = None  # ROI計算の余白（通常mu以上）
    # 数値型（保存用）
    tsdf_store_dtype: str = "float16"
```

---

# 5. SDF生成の標準パイプライン（処理順を固定してレビューしやすく）

SDF層の基本フローは「**材料別SDF（phi）→ TSDF化 → 境界特徴**」です。
multi-material ≤5 を活かして、**シンプルで堅牢**にします。

---

## Step 1：材料ごとのマスク作成（Label→mask）

材料 `m` について

* `mask_m = (label.material_id == m_id)`
  ※m_idは canonical id（0..4想定）

ここで重要なのは **“maskはセル中心である”**こと（ingestで保証済み）。

---

## Step 2：EDTで signed distance（phi）生成（コア）

### EDTの基本式（inside negative / outside positive）

材料mの SDF `phi_m` を、2つのEDTで作ります：

* `dist_in  = EDT(mask_m)`

  * inside（mask=True）要素について「最近傍のoutside（False）までの距離」
* `dist_out = EDT(~mask_m)`

  * outside（mask=False）要素について「最近傍のinside（True）までの距離」

そして

* `phi_m = dist_out - dist_in`

**性質**

* insideでは `dist_out=0`, `dist_in>0` → `phi_m < 0`
* outsideでは `dist_in=0`, `dist_out>0` → `phi_m > 0`
* 境界付近でゼロ交差が生じる

### 実装上の重要点（spacing）

* EDTは **物理単位（nm）で計算**する必要があるため、
  `sampling = grid.spacing (sz,sy,sx)` を必ず渡す（Z=1の2D扱いは後述）

---

## Step 3：TSDF化（truncation + 正規化）

TSDFは学習・同化・mesh化に適した “標準形” です。

* `tsdf_m = clip(phi_m, -mu_nm, +mu_nm) / mu_nm`
* 値域は `[-1,1]`

**保存はfloat16推奨**

* TSDFは[-1,1]に収まるので float16 と相性が良い
* 計算は float32 を使う（勾配統計や差分の安定性のため）

---

## Step 4：境界補助特徴（d_boundary / pair_code）生成

材料数が小さい（≤5）場合、下記は **“軽量なのに効く”**代表特徴です。
SDF層で標準生成します。

### 4-1) d_boundary（境界までの距離）

* `d_boundary_nm = min_m |phi_m|`
* TSDFと同じく `clip(d_boundary_nm, 0, mu_nm)/mu_nm` に正規化して保存推奨（0..1）

### 4-2) pair_code（界面種別）

各voxelで、`|phi_m|` が小さい材料を2つ取り

* `m1 = argmin |phi|`
* `m2 = second_argmin |phi|`

を `pair_code = encode(m1,m2)` として保存します（m1<m2で正規化する等）。

**重要：band外のpair_codeは意味が薄い**
遠方では `|phi|` がクリップされ、順序が不安定になり得ます。
なので推奨は：

* `if min|phi| >= mu_nm: pair_code = 255`（unknown）

---

# 6. `edt.py` の設計（EDTバックエンドを閉じ込める）

## 6.1 edt.py の役割

* “距離変換の実装差” を `edt.py` に隔離し、上位（tsdf/boundary_features）が backend を意識しない設計にします。

## 6.2 主要関数（設計案）

```python
# wafergeo/sdf/edt.py
import numpy as np

def edt_distance(mask: np.ndarray, sampling_zyx: tuple[float,...], backend: str) -> np.ndarray:
    """
    Return EDT distance in physical units (nm) for non-zero elements to nearest zero.
    mask: bool array (True=inside)
    """
    ...

def signed_distance_from_mask(mask: np.ndarray, sampling_zyx: tuple[float,...], backend: str) -> np.ndarray:
    """
    phi = EDT(~mask) - EDT(mask)
    Returns float32 phi in nm.
    """
    ...

def compute_phi_per_material(label_zyx: np.ndarray,
                             material_ids: list[int],
                             sampling_zyx: tuple[float,...],
                             backend: str,
                             include_void: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - phi_stack: (M,Z,Y,X) float32 [nm]   (optionalで返さない構成でもOK)
      - present: (M,) bool
    """
    ...
```

## 6.3 2D扱い（Z==1の高速化）

vtiが “2D相当（Z=1）” の場合、EDTは2Dで回す方が速く・メモリも軽いです。

* 入力 shape `(1,Y,X)` のとき

  * `mask2d = mask[0,:,:]`
  * `sampling2d = (sy,sx)`
  * 結果を `(1,Y,X)` に戻す

このロジックは `edt.py` 内で統一すると、上位層は気にしなくて済みます。

---

# 7. `tsdf.py` の設計（TSDF変換・復元・TSDF→Label）

SDF層は “出すもの（TSDFVolume）” を強く固定します。
TSDF変換、復元、TSDF→labelは `tsdf.py` に集約します。

## 7.1 TSDF変換

```python
def to_tsdf(phi_nm: np.ndarray, mu_nm: float, out_dtype=np.float16) -> np.ndarray:
    """
    phi_nm: float32 SDF in nm
    return: TSDF normalized [-1,1]
    """
    tsdf = np.clip(phi_nm, -mu_nm, mu_nm) / mu_nm
    return tsdf.astype(out_dtype, copy=False)
```

## 7.2 TSDF→（クリップされた）phi復元

```python
def from_tsdf(tsdf: np.ndarray, mu_nm: float) -> np.ndarray:
    """Returns clipped phi_nm = tsdf * mu_nm"""
    return tsdf.astype(np.float32) * mu_nm
```

## 7.3 TSDF stack → Label（重要）

SDF/meshサロゲート推論後に「dominant material」を決めたくなる場面が必ずあります（topdown露出材料判定など）。

**基本ルール（推奨）**

* voxelごとに “最も内側” を選ぶ：`argmin(tsdf_m)`

  * tsdfが最も負＝最も材料内部とみなす
* 全チャンネルが正（どれにも属さない）場合は voidへ

tie-break：

* 同値は `MaterialSpec.priority` で決める（deterministic）

```python
def label_from_tsdf(tsdf_stack: np.ndarray,
                    material: "MaterialSpec",
                    void_index: int = 0,
                    tie_break: str = "priority") -> np.ndarray:
    """
    tsdf_stack: (M,Z,Y,X)
    returns: label_index (Z,Y,X) uint8 (0..M-1)
    """
    ...
```

> これを SDF層に置く理由：
> “TSDF→ラベル化の規約” が散ると、observerやmesh属性付与で結果が変わり、再現性が壊れます。

---

# 8. `boundary_features.py` の設計（d_boundary / pair_codeを“メモリ効率良く”）

ここは大規模3Dでメモリが問題になりやすいので、**phi_stack全保持不要**で計算できるように設計します。

## 8.1 省メモリ戦略（推奨）

材料数Mが小さいので、phi_mを1材料ずつ計算しながら

* `best_abs`（最小|phi|）
* `second_abs`（2番目の|phi|）
* `best_idx`
* `second_idx`

を逐次更新することで、**phi_stackをメモリに持たずに d_boundary/pair_code を生成**できます。

## 8.2 主要関数（設計案）

```python
def update_top2(abs_phi: np.ndarray, mat_idx: int,
                best_abs: np.ndarray, best_idx: np.ndarray,
                second_abs: np.ndarray, second_idx: np.ndarray) -> None:
    """in-place update of top-2 smallest distances"""

def encode_pair(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """encode pair of material indices into uint8. requires M<=16 etc."""

def compute_boundary_features_from_phi_generator(phi_iter, mu_nm: float, band_only: bool, outside_code: int):
    """
    phi_iter yields (mat_idx, phi_nm) sequentially
    returns d_boundary_norm (Z,Y,X), pair_code (Z,Y,X)
    """
```

---

# 9. `band.py` の設計（narrow-band / ROI最適化）

SDF計算は全域に対して重い場合があります。
SEM同化では “視野（ROI）” が明確なことが多いので、SDF層に **ROI + margin** の設計を入れると運用が強くなります。

## 9.1 ROI + margin の考え方

TSDFは `|phi|<mu` 近傍だけが重要です。
ROI内のTSDFを正しくするには、ROI外からの最短距離が mu以内に影響する可能性があるため、

* ROI計算領域 = ROI + margin（通常 mu_nm 以上）

として EDTを計算し、**ROI外は±1で埋める**（圧縮が効く）という設計が可能です。

## 9.2 主要関数（設計案）

```python
def roi_with_margin(roi_zyx: tuple[slice,slice,slice],
                    spacing_zyx: tuple[float,float,float],
                    margin_nm: float,
                    shape_zyx: tuple[int,int,int]) -> tuple[slice,slice,slice]:
    """Compute expanded ROI slices with margin in voxels."""

def fill_outside_roi(tsdf: np.ndarray, roi_zyx, fill_value_outside: float = 1.0) -> np.ndarray:
    """Outside ROI becomes constant -> compressible"""
```

> 注意：EDTの性質上「完全な局所計算」はできないので、
> ROI最適化は “marginを確保した領域だけEDTする” という作りになります。

---

# 10. `reinit.py` の設計（再初期化：距離性回復）

EDTで生成したSDFは基本的に距離性が良いですが、

* TSDF同士の演算（min/max合成）
* surrogateが出したTSDF
* 形状更新（同化で形状を更新するケース）
  などで **距離性が崩れる**ことがあります。

そこで SDF層に再初期化を持たせます。

## 10.1 最も堅牢な標準 reinit（EDT from mask）

**方針**

* TSDFからmask（inside）を復元し、EDTでSDFを再生成してTSDF化する
* “距離性”は回復しやすい（境界位置は離散化誤差の範囲で変動し得る）

```python
def reinit_tsdf_from_tsdf(tsdf: np.ndarray, mu_nm: float, spacing_zyx, backend="scipy") -> np.ndarray:
    """
    For each channel:
      mask = tsdf < 0
      phi = EDT(~mask) - EDT(mask)
      tsdf = clip(phi)/mu
    """
```

## 10.2 multi-materialの整合（重要）

multi-channel TSDFには「同じvoxelが複数材料のinside」になり得ます。
reinit前に **label_from_tsdf** で一意な label を作り直し、

* label→mask_m→EDT→TSDF

に戻すのが最も安定です。
（これを `reinit_tsdf_from_label` として提供）

---

# 11. `qa.py` の設計（SDF品質保証：Artifactに必ず保存する）

SDF層のQAは、下流（mesh/observer/同化）の “結果の信頼性” を担保します。
SDF/TSDFは見た目だけだと壊れていても気づきにくいので、数値QAが必須です。

## 11.1 SDFQA（例）

```python
@dataclass(frozen=True)
class SDFQA:
    tsdf_min: float
    tsdf_max: float
    nan_count: int
    inf_count: int
    present_materials: dict[int, bool]     # material_id -> present
    sign_consistency_error_rate: float     # labelとtsdf符号が矛盾する率
    grad_mag_stats: dict[str, float] | None  # band内 |∇phi| の平均・分散など（任意）
    notes: list[str]
```

## 11.2 QA項目（最低限）

1. TSDF値域が `[-1,1]` に収まる
2. NaN/Infがない
3. labelとの符号整合（簡易）

   * ランダムサンプル点で

     * `label==m` の点は `tsdf_m <= 0` が多いはず
     * `label!=m` の点は `tsdf_m >= 0` が多いはず
   * 矛盾率が一定以上ならfail
4. （推奨）band内の |∇phi| 統計

   * phiは`from_tsdf(tsdf)*mu`で近似できる（クリップされるがband内なら概ねOK）
   * |∇phi|≈1 から大きく外れる割合を指標化

---

# 12. SDF層の “Builder” 具体案（処理の組み立てを固定）

SDF層は、`edt.py`/`tsdf.py`/`boundary_features.py` を束ねる “組み立て” が必要です。
この Builder は I/O を持たず、上位（pipeline）から呼ばれます。

## 12.1 `SDFBuilder.build()` の責務

* label→TSDFVolumeを構築し、QAを返す
* 途中でROIやband、境界特徴生成を適用
* backendやdtype、mu等の設定をMetaに記録できるように情報を揃える

## 12.2 擬似コード（レビューしやすい粒度）

```python
def build_tsdf_volume(label: LabelVolume, cfg: SDFBuildConfig) -> tuple[TSDFVolume, SDFQA]:
    grid = label.grid
    mats = label.material

    # ROI最適化があるなら切り出し範囲を決める（band.py）
    roi = compute_roi(...) if cfg.roi_zyx else full_slices(...)
    sub_label = label.material_id[roi]

    # 出力配列
    M = len(mats.ids) if cfg.include_void_channel else len(mats.ids)-1
    tsdf = empty((M,)+sub_label.shape, dtype=float32 or float16)

    # 境界特徴用（省メモリ更新）
    if cfg.boundary_features:
        best_abs, second_abs, best_idx, second_idx = init_top2_arrays(sub_label.shape)

    present = np.zeros((M,), dtype=bool)

    for mi, mat_id in enumerate(selected_material_ids):
        mask = (sub_label == mat_id)

        if mask.any(): present[mi] = True

        if mask.all():
            tsdf[mi] = -1.0
            if cfg.boundary_features: update_top2_with_constant(...)
            continue
        if not mask.any():
            tsdf[mi] = +1.0
            if cfg.boundary_features: update_top2_with_constant(...)
            continue

        phi_nm = signed_distance_from_mask(mask, spacing_zyx_effective, cfg.backend)  # edt.py
        tsdf[mi] = to_tsdf(phi_nm, cfg.mu_nm, out_dtype=float32)  # tsdf.py

        if cfg.boundary_features:
            abs_phi = np.abs(phi_nm)
            update_top2(abs_phi, mi, ...)  # boundary_features.py

    # 後処理
    tsdf_store = tsdf.astype(cfg.tsdf_store_dtype)
    d_boundary, pair_code = None, None
    if cfg.boundary_features:
        d_boundary = np.clip(best_abs, 0, cfg.mu_nm)/cfg.mu_nm
        if cfg.band_only_pair_code:
            pair_code = encode_pair(best_idx, second_idx)
            pair_code[best_abs >= cfg.mu_nm] = cfg.pair_code_outside_band
        else:
            pair_code = encode_pair(best_idx, second_idx)

    # ROI外を埋めたいなら（band.py）
    tsdf_full = place_back_to_full_volume(tsdf_store, roi, fill=+1 or -1)
    d_boundary_full = place_back_to_full_volume(d_boundary, roi, fill=1.0) if d_boundary else None
    pair_code_full = place_back_to_full_volume(pair_code, roi, fill=255) if pair_code else None

    # QA（qa.py）
    qa = compute_sdf_qa(tsdf_full, label, cfg, present, ...)

    # Meta組み立て（上位がやっても良いが、必要情報はここで揃える）
    meta = make_meta(...)

    return TSDFVolume(grid, mats, cfg.mu_nm, tsdf_full, d_boundary_full, pair_code_full, meta), qa
```

---

# 13. エラー設計（SDF層で起きる異常を明確に分類する）

SDF層の典型エラーは以下です：

* `InvalidSpacingError`：spacingが0/NaN/負
* `EDTBackendUnavailableError`：cupy未導入など
* `EDTComputationError`：EDT内部例外（入力が異常、メモリ不足等）
* `InvalidMuError`：mu<=0
* `ShapeMismatchError`：ROI戻しなどでshape矛盾

**重要**：SDF層はI/Oしないので、例外メッセージに

* material_id
* shape
* spacing
* backend
* ROI
  など “再現に必要な情報” を必ず含めると保守が楽です。

---

# 14. テスト設計（SDF層は回帰が出やすいので厚めに）

## 14.1 ユニットテスト（純粋関数）

* `to_tsdf/from_tsdf`（値域・型）
* `signed_distance_from_mask`（簡単形状：球/箱/円で符号・距離の妥当性）
* `update_top2/encode_pair`（境界特徴の正しさ）
* ROI+marginの切り出し（端境界）

## 14.2 統合テスト（ゴールデン形状）

* 2材料界面（平面境界）
* 3材料積層（薄膜）
* Z=1の2D相当ケース
* 欠損材料（ある材料が全く存在しないケース）
  → TSDFが全+1になる等、規約通りか確認

**回帰基準**

* 0等値面の位置（概ね）
* TSDFの統計（mean/min/max）
* d_boundary/pair_codeの分布（band内のみで比較）

---

# 15. 下流（mesh/observe/surrogate/assimilation）との契約

SDF層の出力が下流でどう使われるかを、契約として明文化します。

## 15.1 mesh層への契約

* `TSDFVolume.tsdf[m]` の 0等値面が “材料mの境界” として扱える
* `pair_code` を使えば「どの界面ペアか」を推定しやすい
* TSDFが[-1,1]でも 0近傍の情報が残るので mesh抽出に十分

## 15.2 observe層への契約

* topdown/slice観測は `TSDFVolume` から生成可能
* topdown露出材料判定が必要なら `label_from_tsdf` でラベル化できる（規約固定）

## 15.3 surrogate層への契約

* SDFサロゲートの教師データは `TSDFVolume.tsdf` が標準
* meshサロゲートでも、学習・評価で2D観測損失を入れる場合に `TSDFVolume` が基準となる

## 15.4 同化への契約

* 同化損失は Obs2D（2D TSDF/輪郭）で比較するが、sim側予測をObs2Dに落とす入口として `TSDFVolume` が使われる
* mu、符号規約、材料順が Meta で完全に追跡できること

---

# 16. まとめ（SDF層の設計要点）

* multi-material ≤5 なので **材料別TSDFスタックを標準生成**しても運用が成立しやすい
* その上で

  * `d_boundary`
  * `pair_code`
    を **省メモリ方式**で同時生成すると、界面情報が増えて学習・同化が強くなる
* ROI+margin、band外を定数埋めにすると、**大規模3Dでも圧縮が効き、運用コストが下がる**
* TSDF→Label規約をSDF層に固定し、observer/mesh/同化で結果がブレないようにする
* QA（値域、符号整合、勾配統計）をArtifactに保存し、後で原因追跡できるようにする

