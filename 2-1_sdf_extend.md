了解です。ここでは「**様々なSDF手法を“利用できる”だけでなく、“公平に評価できる”**」ことを主目的に、**先ほどのEDT/TSDF/Top-layer等以外**で本目的（ウェハ形状サロゲート＋SEM同化）に有用な手法を追加しつつ、**後から編集・追加しても運用が壊れない設計**を具体化します。

---

# 1. 追加で有用なSDF手法の体系（本目的に効く順に）

## A. “Exact EDT”を複数実装で持つ（基準器＋互換性の要）

EDT（Exact Euclidean Distance Transform）は、あなたの用途（サロゲート教師／SEM同化の観測場／mesh化）で **基準になる**ので、実装を複数持っておくと運用が強くなります。

### A-1) Meijster系（線形時間EDT）

* 2パス系の線形時間距離変換で、古典的なExact EDT系。([fab.cba.mit.edu][1])
* **用途**：CPUで確実に正しいSDFを作る基準（Reference）、研究検証

### A-2) Felzenszwalb & Huttenlocher（距離変換 of sampled functions）

* 正確なEDTを線形時間で計算できる枠組みとして有名。([cs.cornell.edu][2])
* **用途**：実装しやすい・高速なExact EDTバックエンドの候補
  （binaryだけでなく“サンプル関数”に拡張できるので、後述の「ソフト観測」にもつながる）

### A-3) Maurer（任意次元のExact EDT）

* 任意次元で線形時間のExact EDT。([csd.uwo.ca][3])
* ITKの `SignedMaurerDistanceMapImageFilter` がこれを提供（実用上かなり強い）。([ITK][4])
* **用途**：SciPyと別系統の正しさ・境界挙動の比較、運用での“実装差事故”の回避

> **設計上の狙い**：Exact EDTを複数持つのは「速度」ではなく、
> **同じ入力でも結果が違う時に原因切り分けできる**（実装差・spacing扱い・origin解釈）という運用上の価値が大きいです。

---

## B. “近似DT”を用意（高速プレビュー／多数回評価の探索に効く）

サロゲートの学習データ生成はExactが望ましい一方で、同化や最適化で **候補評価を大量に回す**場合は近似でも効果があります（まず粗く探索→最後にExactで仕上げ、が可能）。

### B-1) Chamfer（重み付き近傍による近似距離）

* 小近傍の重みで距離を近似する古典手法。([people.cmm.minesparis.psl.eu][5])
* **利点**：非常に速い・実装が単純・GPU/並列化しやすい
* **欠点**：Exactではない（微小形状の精度に影響）

### B-2) Danielsson系（ベクトル伝播／ほぼ正確）

* “距離写像”の古典で、ベクトル伝播系のDTに繋がる。([サイエンスダイレクト][6])
* **用途**：近似だが誤差が小さくしやすい実装がある（GPU実装例もある）。([Department of Computer Science][7])

### B-3) Jump Flooding Algorithm（JFA：GPU近似DT）

* GPUで距離変換（Voronoi/DT近似）を高速に計算できる枠組み。([NUS Computing][8])
* **用途**：同化の探索（CMA-ES等）で **1反復に多数の候補評価**が必要なとき、近似DTを“高速評価器”として使う
* **注意**：近似なので、最終ステップはExact EDTに差し替える運用が必須

---

## C. Eikonal/Level-set系（工程物理に寄せる：プロセス指向SDF）

“距離場”としてのSDFだけでなく、SDFを **物理的なフロント伝播**で更新したい場合があります（エッチや成膜をLevel Setで扱うなど）。このとき重要になるのが Eikonal 方程式（|∇T|=1/Fなど）の数値解法です。

### C-1) Fast Marching Method（FMM）

* Eikonal方程式を高速に解く代表的手法。([SIAM E-Books][9])
* Sethian は3Dフォトリソグラフィ等への応用も述べています。([バークレー大学数学科][10])
* **用途**：

  * “速度場F(x)”を入れた **異方性・位置依存の距離**（=工程依存の進行）を表現したい
  * 「単なる幾何距離」ではなく「プロセス距離（到達時間）」で特徴量化したい（サロゲートが工程差を学びやすくなる）

### C-2) Fast Sweeping Method（FSM）

* Eikonalを反復掃引で解く手法。([アメリカ数学協会][11])
* **用途**：

  * グリッド上のEikonalを高速に（実装しやすく）解きたい
  * FMMより相性が良いケース（特徴線に沿った掃引など）がある

> これらは「SDF生成」だけでなく、将来的に **SDFを更新して工程変化を近似**する方向（簡易プロセスモデル）にも繋がります。

---

## D. Sparse / Narrow-band “産業級”SDF（巨大3Dを現実的に扱う）

3Dウェハ領域が大きい場合、dense SDFはすぐにメモリとI/Oが支配します。ここで有効なのが“界面近傍だけ持つ”設計と、それを支えるデータ構造です。

### D-1) OpenVDB（疎なレベルセットSDF）

* 高解像度の疎ボリューム（レベルセットSDF）を効率よく扱う代表例。([museth.org][12])
* **用途**：

  * 3Dの界面近傍だけを保持し、SDF演算・平滑化・リサンプリング・mesh化を高速に
  * SEM同化ではROIが小さい場合が多いので、ROI＋narrow-band運用と相性が良い

> ※あなたの設計（TSDF/narrow-band）と整合的です。OpenVDBはその“実装バックエンド”候補になります。

---

## E. Mesh→SDF / Sign推定（メッシュ学習ルートやサブボクセル精度に効く）

SDF→meshだけでなく、meshからSDFへ戻す（またはサブボクセル精度のSDFを作る）手法も、メッシュ系サロゲートを運用するなら重要になります。

### E-1) Generalized Winding Numbers（メッシュの内外判定を頑健に）

* 穴・自己交差などがあっても内外判定（符号）を安定化する考え方として有名。([ユーザーCS Utah][13])
* **用途**：

  * mesh出力を“符号付き距離”へ戻す（同化の共通ドメインへ戻す）
  * 不完全なmeshでも符号判定を壊しにくい

### E-2) “mesh-to-SDF（近傍三角形距離）”を疎に構築

* OpenVDBの文脈では、界面セルを見つけて三角形距離計算を最小化しつつ局所SDFを構築する流れが説明されています。([museth.org][12])
* **用途**：サブボクセル精度が必要な時に、voxel EDTだけより滑らかなSDFを得る

---

## F. 2D輪郭（SEM contour）専用の“連続”SDF（ラスタ化EDT以外）

あなたのSEM同化は最終的に2D輪郭比較ですが、輪郭→SDFは必ずしも「ラスタ化→EDT」である必要はありません。

### F-1) “ポリライン距離＋符号（winding）”の解析的SDF

* 2Dなら、輪郭（polyline）への最短距離は線分距離で計算でき、符号は2D winding（多角形内外判定）で決まります（幾何として自然）。
* **用途**：

  * **ラスタ化によるエイリアシング**が問題の時（CDが数ピクセル級、微小差を同化したい）
  * 観測格子（SEM pixel）と解析格子（sim grid）の違いを吸収しやすい（連続表現→任意格子へサンプル）

> 実装はBVH/グリッドで線分検索を加速して、denseな2D TSDFも生成できます。
> これを “SEM専用エンジン” として持つ価値は高いです。

---

# 2. 「本目的」での使い分け（何を標準にして、何を評価対象にするか）

## 2.1 推奨：最低限の“標準（Canonical）”と“評価候補”を分ける

* **Canonical（基準）**：Exact EDT（SciPy/ITKなど）
* **Production（大規模運用）**：OpenVDB（疎SDF）やROI narrow-band
* **Search/Preview（多数回評価）**：Chamfer / Danielsson / JFA
* **Process-aware（工程物理寄り）**：Fast Marching / Fast Sweeping
* **SEM専用（高精度輪郭）**：Polyline解析SDF

これを“最初から”分離しておくと、後で手法が増えても運用が壊れません。

---

# 3. “後から編集・追加しても運用できる”SDFパッケージ設計（具体案）

ここから設計の核心です。ポイントは **「SDF計算アルゴリズム（Engine）」と「共通後処理（TSDF/band/QA）」を厳密に分離**することです。

---

## 3.1 SDF層を「Engine」「Transform」「Feature」「QA」「Bench」に分割する

既存の `wafergeo/sdf/edt.py, tsdf.py, band.py, ...` を活かしつつ、追加に耐える構造にします。

推奨ディレクトリ（SDF層だけ抜粋）：

```text
wafergeo/sdf/
  engines/
    edt_scipy.py              # Exact EDT (SciPy)
    edt_itk_maurer.py         # Exact EDT (ITK SignedMaurer)
    edt_felzenszwalb.py       # Exact EDT (Felzenszwalb-Huttenlocher)
    dt_chamfer.py             # Approx DT (Chamfer)
    dt_danielsson.py          # Approx DT (vector propagation)
    dt_gpu_jfa.py             # Approx DT (JFA)
    eikonal_fast_marching.py  # FMM
    eikonal_fast_sweeping.py  # FSM
    mesh_openvdb.py           # mesh->sdf sparse
    mesh_winding.py           # sign via winding numbers
    polyline_exact_2d.py      # contour->sdf continuous 2D
  transforms/
    to_tsdf.py
    to_band.py
    reinit.py                 # EDT reinit / PDE reinit
  features/
    boundary_features.py      # d_boundary / pair_code / neighbor_id
    gradients.py              # (optional) normal/curvature support
  qa/
    sdf_qa.py
    cross_engine_check.py
  bench/
    benchmark_runner.py
    metrics.py
    reports.py
```

> 既存の `edt.py/tsdf.py/band.py` は「transforms」や「engines/edt_*」へ移してもいいですが、
> 移行時は“互換ラッパー”を残して破壊的変更を避けるのが運用上安全です。

---

## 3.2 Engineの共通インターフェース（追加が壊れない契約）

SDF手法は入力も多様です（label、binary mask、mesh、polyline）。
なので Engine は「入力タイプ」を明示し、**能力（capabilities）**を返すようにします。

### SDFRequest / SDFResult（概念設計）

* `SDFRequest`

  * `input`: `BinaryMask | LabelVolume | MeshGeom | Polyline2D`
  * `grid_spec`: spacing/origin/units（必須）
  * `roi`: optional（ROI＋margin対応）
  * `materials`: optional（multi-material時）
  * `params`: engine固有（chamfer weights、eikonal speed、JFA steps等）
* `SDFResult`

  * `phi_nm`: （必要なら）SDF（物理単位）
  * `tsdf`: （必要なら）TSDF
  * `band_mask`: （必要なら）
  * `engine_meta`: engine名・version・依存ライブラリ
  * `qa`: 基本統計（min/max/NaN等）

### EngineCapabilities（運用で効く）

例：

* 対応入力：`label/binary/mesh/polyline`
* 次元：`2D/3D`
* `exact=True/False`
* `supports_anisotropic_spacing`
* `supports_roi_margin`
* `deterministic`
* `gpu_accelerated`
* `output_phi` / `output_tsdf`（どこまで出すか）

> **これがあると**、CLIやパイプラインが
> 「このengineはmesh入力できない」「2Dしかできない」を実行前に弾けます。

---

## 3.3 “共通後処理”は Engine 外へ固定する（重要）

どのSDF手法でも、下流（mesh/observe/同化）が期待するのは “TSDF / band / boundary features / QA” です。

そこで、

* TSDF化（μ、符号規約、float16保存）
* band抽出
* d_boundary/pair_code
* reinit
* QA

は **Engineの外（transforms/features/qa）**で共通に提供します。

これにより、新しいEngineが増えても

* 「TSDFの正規化が違う」
* 「bandの定義が違う」
* 「符号が逆」
  が起きづらくなります（運用安定）。

---

# 4. “評価できるパッケージ”として必要な仕組み（ベンチ設計）

手法が増えるほど「どれが良いか」が曖昧になり、運用が壊れます。
そこで評価は **パッケージ機能として内蔵**します。

## 4.1 評価の2本立て

### (1) Field-level（SDF自体の品質）

* 参照（ReferenceEngine：Exact EDT）との差

  * band内L1/L2/L∞誤差
  * 0等値面のズレ（近傍での符号反転率）
* 距離性（band内 |∇φ| ≈ 1 の逸脱率）

※参照EDTの候補：Meijster/Felzenszwalb/Maurer系のどれかを基準にできます。([fab.cba.mit.edu][1])

### (2) Task-level（本目的に対する有効性）

あなたの目的は “最終的にSEM輪郭比較” なので、必ず

* `Observer（topdown/slice）→ 2D TSDF → contour` へ落とし、
* contour距離（Chamfer/Hausdorff）とCD誤差

を評価指標に入れます（＝SDFが違っても最終指標で比較できる）。

---

## 4.2 ベンチの出力（運用が回る形）

* JSON/CSV：engineごとのスコア（精度/時間/メモリ）
* “MethodCard”（engineの説明・依存・制約・推奨用途）を一緒に保存
* グラフ（任意）：`speed vs error`、`CD error distribution`

---

# 5. SDF→メッシュ変換も「手法評価対象」に入れる（本プロジェクトなら重要）

あなたは「SDF学習」と「SDF→mesh学習」を両方扱うので、SDF手法の違いが mesh品質にどう影響するかも評価対象です。

## 5.1 例：等値面抽出アルゴリズム（Marching Cubes vs Flying Edges等）

VTKには大規模向けに高速化された Flying Edges 実装があります。([vtk.org][14])
SDF手法だけでなく、mesh抽出手法も “プラグイン評価対象”にすると一貫します。

---

# 6. 追加・編集を前提にした運用設計（壊れないための実務ルール）

## 6.1 “Engine signature” をArtifactメタに必ず刻む

Artifactに必須で入れる：

* `engine_name`
* `engine_version`
* `dependencies`（scipy/itk/openvdb/cuda等）
* `params`（chamfer weights、JFA steps、eikonal speed…）
* `grid_spec`（spacing/origin/units）
* `mu_nm`（TSDFの場合）

こうしておくと、将来 engine を増やしても

* 過去データの再現
* ベンチ比較
* “いつから結果が変わったか”
  が追跡できます。

## 6.2 optional依存（ITK/OpenVDB/GPU）を“Extras”で分離

* `core`：SciPy Exact EDT（最低限ここで完結）
* `itk`：SignedMaurer（別系統Exact）([ITK][4])
* `openvdb`：疎SDF（大規模）([museth.org][12])
* `cuda`：JFA/GPU DT ([NUS Computing][8])

→ 依存が無い環境でも core が動く。運用現場で強いです。

## 6.3 新Engine追加の“必須条件”を規約化

第三者が追加する時の最小要件を固定します。

* `capabilities()` 実装（入力種・次元・exact性など）
* `compute(request)` 実装
* `method_card.json`（用途・制約・参考文献）
* `golden test`（球/箱/薄膜などで参照との差をテスト）
* `cross_engine_check`（符号規約、μ、単位、axis order）

これをCIで必須にすると、手法追加しても壊れません。

---

# 7. “本目的”に対して追加すると強い内容（まとめ）

最後に、先ほどのEDT/TSDF中心設計に追加して、特に効く拡張を要点でまとめます。

## 7.1 追加すべきSDF手法（優先度順）

1. **Exact EDTの複数実装**（SciPy + ITK SignedMaurer + もう1系統）([ITK][4])
2. **近似DT（Chamfer/Danielsson）**（探索高速化の選択肢）([people.cmm.minesparis.psl.eu][5])
3. **GPU近似（JFA）**（多数回評価向け）([NUS Computing][8])
4. **Sparse SDF（OpenVDB）**（巨大3D運用）([museth.org][12])
5. **Eikonal系（FMM/FSM）**（工程物理寄り特徴の導入）([SIAM E-Books][9])
6. **mesh sign（Generalized Winding）**（meshルートの頑健化）([ユーザーCS Utah][13])
7. **SEM輪郭の解析的SDF（polyline）**（ラスタ化誤差を減らす）

## 7.2 設計として必須な拡張

* Engineと共通Transform/QA/Benchの分離
* capabilities / method card / golden test の規約化
* Artifactに engine signature を刻む
* ベンチは「SDF品質」と「2D輪郭タスク品質」の両方

---

必要なら次に、上の設計をそのままコードに落とすために、

* `SDFEngine` / `SDFRequest` / `EngineCapabilities` の**具体API案（型定義）**
* `method_card.json` のテンプレ
* `benchmark_runner` の入出力仕様（JSONスキーマ）
* Golden shapes（薄膜・トレンチ・多材料界面）の定義

まで「設計書→実装仕様」レベルで提示します。

[1]: https://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf?utm_source=chatgpt.com "A GENERAL ALGORITHM FOR COMPUTING DISTANCE ..."
[2]: https://www.cs.cornell.edu/dph/papers/dt.pdf?utm_source=chatgpt.com "Distance Transforms of Sampled Functions - Computer Science"
[3]: https://www.csd.uwo.ca/~oveksler/Courses/Fall2009/9840/Chosen/linearExactEucl.pdf?utm_source=chatgpt.com "A linear time algorithm for computing exact Euclidean ..."
[4]: https://docs.itk.org/projects/doxygen/en/v5.3.0/classitk_1_1SignedMaurerDistanceMapImageFilter.html?utm_source=chatgpt.com "SignedMaurerDistanceMapImag..."
[5]: https://people.cmm.minesparis.psl.eu/users/marcoteg/cv/publi_pdf/MM_refs/1986_Borgefors_distance.pdf?utm_source=chatgpt.com "Distance Transformations in Digital Images"
[6]: https://www.sciencedirect.com/science/article/pii/0146664X80900544/pdf?md5=626f9c14e0a0a8ad4cfb80779596c79c&pid=1-s2.0-0146664X80900544-main.pdf&utm_source=chatgpt.com "Euclidean Distance Mapping"
[7]: https://www.cs.cit.tum.de/fileadmin/w00cfj/cg/Research/Publications/2009/GPU-Based_Real-Time_Discrete_Euclidean_Distance/visapp09.pdf?utm_source=chatgpt.com "GPU-Based Real-Time Discrete Euclidean Distance Transforms"
[8]: https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf?utm_source=chatgpt.com "Jump Flooding in GPU with Applications to Voronoi ..."
[9]: https://epubs.siam.org/doi/10.1137/S0036144598347059?utm_source=chatgpt.com "Fast Marching Methods | SIAM Review"
[10]: https://math.berkeley.edu/~sethian/2006/Papers/sethian.spie.fastmarching.1996.pdf?utm_source=chatgpt.com "Fast Marching Level Set Methods for Three-Dimensional ..."
[11]: https://www.ams.org/mcom/2005-74-250/S0025-5718-04-01678-3/S0025-5718-04-01678-3.pdf?utm_source=chatgpt.com "Fast sweeping method for eikonal equations"
[12]: https://www.museth.org/Ken/Publications_files/Museth_TOG13.pdf?utm_source=chatgpt.com "VDB: High-resolution sparse volumes with dynamic topology"
[13]: https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.pdf?utm_source=chatgpt.com "Robust Inside-Outside Segmentation using Generalized ..."
[14]: https://vtk.org/doc/nightly/html/classvtkFlyingEdges3D.html?utm_source=chatgpt.com "vtkFlyingEdges3D Class Reference"
