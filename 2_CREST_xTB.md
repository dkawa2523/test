以下では、ご指定の全体ワークフロー（RDKit → CREST+xTB → PES探索 → DFT → 熱化学/速度論 →（将来ClearML））を維持したまま、**「配座/複合体（CREST + xTB）」フェーズ**を、アーキテクト/計算化学/MIの観点で“実装しやすい粒度”まで落として詳細化します。
対象は現状「アミン系 × HF」の気相ですが、**将来は任意分子×任意分子（SDF 1分子/ファイル）**へ拡張できる設計にします。

---

## 1. このフェーズの役割（なぜ重要か）

DFTで活性化エネルギー（ΔE‡/ΔG‡）を出す以前に、**入口構造（反応物の配座＋遭遇複合体）の取り方**で、

* 見つかるTSが変わる（別経路・別TSに落ちる）
* 同一経路でも「初期複合体のエネルギー基準」が変わる（相対ΔE‡がぶれる）
* そもそもTS探索が収束しない/誤収束する

が頻発します。ガスプロセス（気相）では、溶媒で「それっぽい」配置に誘導されにくいので、**遭遇複合体（encounter complex）を多様に列挙→下流に渡す**のが特に効きます。

このフェーズのアウトプットは下流（PES探索）に対して：

* **“反応物側の初期集合”**（Conformer ensemble）
* **“反応開始点候補”**（Encounter complex ensemble）
* **“反応に関係する原子インデックスのマッピング”**（後でNEB/GSM/TSで距離拘束やIRCを組むため）

を、**機械可読な形で**提供することです。

---

## 2. 外部ツール（CREST/xTB）の前提とバージョン要件

### 2.1 CREST（配座探索、NCI探索）

* 通常の配座探索は `crest struc.xyz --gfn2 -T 4` のように実行します。([Crest Lab][1])
* CRESTは出力として最終の配座集合を `crest_conformers.xyz`（conformers）、`crest_rotamers.xyz`（conformers+rotamers）に書き出します。([Crest Lab][1])
* NCI（非共有結合複合体）探索は `crest complex.xyz --nci` で実行可能です。([Crest Lab][2])

  * NCIでは、MTDで複合体が解離しないように**楕円体の壁ポテンシャル**が自動で導入され、`--wscal` でサイズを調整できます。([Crest Lab][2])
* CRESTのコマンドラインキーワード（`--gfn2`, `--gfnff`, `--chrg`, `--uhf`, `-T`, `-xnam` 等）は公式のkeyword docを参照できます。([Crest Lab][3])
* CREST 3系では **TOML入力ファイル**（`--input input.toml`）が使え、`preopt` 等も制御できます。([Crest Lab][4])

### 2.2 xTB（事前最適化、Docking aISS）

* xTBの幾何最適化は `xtb coord --opt tight --cycles 50 --charge -1` のように実行できます。([Xtb Docs][5])
* xTBでは、電荷・不対電子数は `.CHRG` / `.UHF` もしくは `--chrg/--uhf` で指定できます（コマンドライン指定が優先）。([Xtb Docs][6])
* **遭遇複合体列挙の主役として強いのが xTB docking submodule (aISS)**です。

  * これは **xTB 6.6 以降**にある機能です。([Xtb Docs][7])
  * 実行は `xtb dock [options] <geometry1> <geometry2>` 形式です。([Xtb Docs][7])
  * まず剛体の xTB-IFF でスクリーニングし、その後GFNで最適化します（スクリーニング中は剛体で、分子内の変形は最終最適化で反映）。([Xtb Docs][7])
  * `--ensemble` を使うと、負の相互作用エネルギーを持つ構造をすべて最適化し、重複除去してNCI ensembleを得られます。([Xtb Docs][7])
  * さらに `xcontrol(7)`形式の入力で **directed docking**（分子1の特定部位へ誘導）も可能で、例として `$directed ... atoms: 1-5 elements: N` のように設定できます。([Xtb Docs][7])

---

## 3. アーキテクト観点：このフェーズを「2タスク＋共通データモデル」で分離する

このフェーズは、パイプラインとしては最低でも次の2タスクに分けるのが保守性/再利用性/拡張性に強いです。

1. **ConformerSearchTask**（単分子の配座探索）
2. **ComplexEnumerationTask**（2分子の遭遇複合体列挙）

どちらも **「入力→外部実行→出力解析→標準フォーマットで保存」**の形に統一します。

---

## 4. データモデル（I/O標準化）：下流（PES探索/DFT）につなぐための最重要ポイント

### 4.1 “生ファイル”を残しつつ、“機械可読メタデータ”を正にする

外部ツールはログやXYZを吐きますが、下流自動化では **JSON（or JSONL）**が軸です。

推奨する保存単位：

* `runs/<run_id>/species/<species_id>/conformers/`

  * `input.xyz`（このタスクが使った入力）
  * `crest.out`（標準出力ログ）
  * `crest_conformers.xyz`（公式出力）([Crest Lab][1])
  * `crest_rotamers.xyz`（必要なら）([Crest Lab][1])
  * `crest.energies`（相対エネルギー一覧）([Crest Lab][8])
  * `conformer_set.json`（あなたのコードが生成する正規メタデータ）
* `runs/<run_id>/pairs/<pair_id>/complexes/`

  * `dock.inp`（directed dockingなどの入力、任意）
  * `xtb_dock.out`
  * `dock_ensemble.xyz`（抽出した複合体群）
  * `complex_set.json`

### 4.2 原子インデックスのマッピングを必ず保存（これが後で効く）

* RDKitのSDF → XYZ → CREST出力 → 結合した複合体XYZ
  ここで原子順が崩れると、TS探索で「N原子に拘束したいのに別原子を拘束してた」が起きます。

したがって、各構造に以下を持たせます：

* `atom_order`: `[0,1,2,...]`（SDF→XYZ変換時の原子順）
* `origin_map`: `{"sdf_atom_idx": ..., "xyz_atom_idx": ...}`（必要なら）
* 複合体はさらに

  * `fragment_slices`: `{"A": [0, nA-1], "B": [nA, nA+nB-1]}`
  * `fragment_atom_map`: A/Bそれぞれ元の単分子インデックスへの写像

> CRESTが扱うXYZ ensembleはコメント行にエネルギー（Hartree）を入れるとCREGEN等で処理しやすい、とドキュメントされています。([Crest Lab][9])
> あなたの出力XYZにも「コメント行にエネルギー/ID/出典」を埋め込む設計が後で効きます。

---

## 5. ConformerSearchTask 詳細（CREST iMTD-GC + xTB preopt）

### 5.1 入力（Upstreamから受け取るもの）

* `std.xyz`（構造処理フェーズで作った3D構造、H付与済み）
* `charge`（整数）
* `uhf`（Δn = Nα−Nβ の整数。閉殻は0）

  * CRESTでも `--chrg` / `--uhf` が指定できます。([Crest Lab][3])
* `species_meta.json`（分子名/SMILES/InChIKey/反応部位候補原子など）

### 5.2 実行戦略：**“必ずpreopt → CREST”**にする

CREST docsでも、入力構造は同一理論レベルでxTB preoptしておくのが賢い、と明示されています（トポロジーチェックの参照に使うため）。([Crest Lab][1])

#### (A) preopt（xTB）

例：

* `xtb input.xyz --opt tight --cycles 200 --gfn 2 --chrg <q> --uhf <u> -P <nthreads> > xtb_preopt.out`

`--opt tight` の例は公式ドキュメントにあります。([Xtb Docs][5])

失敗時：

* xTBは最適化が収束しないと `NOT_CONVERGED` ファイルを作ることがあり、bulk jobで検出に使えます。([Xtb Docs][10])
* フォールバックとして

  * まずGFN0で直してからGFN2へ、というガイドもあります。([Xtb Docs][10])

#### (B) CREST（標準：iMTD-GC）

基本コマンド例：
`crest struc.xyz --gfn2 -T 8` ([Crest Lab][1])

* 生成された最終アンサンブルは `crest_conformers.xyz`（conformers）、`crest_rotamers.xyz`（conformers+rotamers）へ出力されます。([Crest Lab][1])

**推奨：CREST 3系ならTOML入力生成を採用**
理由：

* コマンドラインが肥大化しない
* 設定の機械可読性が上がる
* 将来ClearMLで「設定差分」追跡が容易

TOML入力は `--input input.toml` で読み込めます。([Crest Lab][4])

たとえば（概念）：

```toml
# CREST 3 input file (generated)
input = "preopt.xyz"
runtype = "imtd-gc"
threads = 8
preopt = false   # 既にxTBでpreopt済みならfalse

[[calculation.level]]
method = "gfn2"
chrg = 0
uhf  = 0

[cregen]
ewin = 10.0
rthr = 0.125
ethr = 0.05
bthr = 0.01
```

※ `ewin/rthr/ethr/bthr` はCREGENの設定で、CLIでも指定可能です（デフォルトなどはman/ドキュメント参照）。([mankier.com][11])

### 5.3 どの理論レベルを使うべきか（GFN2 / GFN-FF / composite）

CRESTキーワードには：

* `--gfn2`, `--gfn1`, `--gfnff`
* そして `--gfn2//gfnff`（サンプリングと最適化をGFN-FF、最後にGFN2 singlepoint付与）
  が公式に記載されています。([Crest Lab][3])

**実務推奨（半導体ガス装置の探索スケールを意識）**：

* 小分子〜中分子（~50原子程度、柔軟性そこまで）：`GFN2-xTB` を基本
* 大きい/柔軟（>80原子、回転自由度多）：

  * サンプリングは `GFN-FF`（速い）
  * ランキングや後段投入は `GFN2 singlepoint` を付ける（`gfn2//gfnff` 的な思想）

→ **configで切り替え可能**にしておく（分子数が増えると必須）。

### 5.4 出力解析（CREST→ConformerSet）

CREST出力の核：

* `crest_conformers.xyz` ([Crest Lab][1])
* `crest.energies`（相対エネルギー一覧が保存される例が公式ページに明記）([Crest Lab][8])

#### 解析して `conformer_set.json` に正規化する項目例

* `tool`: `{ "crest_version": "...", "xtb_version": "...", "cmd": "...", "settings": {...}}`
* `thermo_reference`: `{ "T": 298.15, "units": "kcal/mol" }`
* `conformers`: 配列

  * `conf_id`
  * `energy_abs_hartree`（あるなら）
  * `energy_rel_kcalmol`
  * `geom_path`（抽出して個別xyzにしても良い）
  * `source`（crest_conformers.xyzの何番目か）
  * `hash`（座標のハッシュ）
  * `site_features`（後述：反応部位周りの記述子）
* `selection`:

  * `kept_by`: `"ewin"` or `"topN"` or `"cumulative_weight"`
  * `N_total`, `N_kept`

#### “何個残すか”の現実解（MI観点）

下流のPES探索/DFTは重いので、ここで**落としすぎず、増やしすぎず**が重要です。

おすすめの2段階：

1. CRESTはやや広め（例：`ewin=10 kcal/mol`）で回す
2. 下流投入は

   * `topN`（例：20）＋
   * 多様性（RMSDクラスタ）＋
   * 反応部位露出スコア
     で 10–30 程度に絞る

---

## 6. ComplexEnumerationTask 詳細（遭遇複合体列挙）

遭遇複合体は **“配座×配座×相対配置”**なので爆発します。ここは **目的（TS探索に効く初期構造を作る）**に絞って設計します。

### 6.1 入力

* `ConformerSet(A)`（候補分子）
* `ConformerSet(B)`（反応相手：HF など）
* `pair_config`（どのサイトに寄せるか、何個残すか、温度など）

### 6.2 推奨戦略：xTB dock（aISS）を主、CREST NCIをオプションで補強

#### 6.2.1 xTB dock を主にする理由

* aISSは「相互作用部位スクリーニング＋遺伝的最適化＋GFN最適化」の流れで、遭遇複合体の列挙に向く。([Xtb Docs][7])
* `--ensemble` でNCI ensembleを直接得られる。([Xtb Docs][7])
* directed dockingで「分子1の特定領域（例：N周り）に寄せる」ができる。([Xtb Docs][7])
* ただしスクリーニングは剛体xTB-IFFで、分子内変形は最終最適化で反映される点は理解しておく。([Xtb Docs][7])

#### 6.2.2 CREST NCI を使う場面

* “ドッキング→最適化”だけだと、**内部回転＋相対配置の相互作用**が十分に探索されないことがある
* その補強として、上位数件に `crest complex.xyz --nci` をかけるのは有効

  * NCIモードは壁ポテンシャルで解離を防ぎつつ、複合体の配座を探索する目的。([Crest Lab][2])

---

### 6.3 xTB dock による遭遇複合体生成（実務フロー）

#### ステップ0：どの配座を使うか（組合せ爆発対策）

* Aの上位 `NA`（例：5〜10）
* Bの上位 `NB`（HFなら1でOK、将来の拡張で2〜5）
* 組合せは `NA × NB` に制限

#### ステップ1：directed docking の入力（アミン×HFの例）

xTBドッキングは `xtb dock <mol1> <mol2>` で実行します。([Xtb Docs][7])

directed dockingは `--input dock.inp` で `xcontrol(7)` を渡し、`$directed` ブロックで設定できます。([Xtb Docs][7])

例（概念）：

```text
$directed
   attractive
   scaling factor=0.8
   elements: N
$end
```

* `elements: N` により、分子1（A）のN近傍に分子2（HF）が来やすくなる、という設計思想です（公式doc例にもNを指定した例がある）。([Xtb Docs][7])
* 反応テンプレ由来で「反応部位原子ID（例：amine N）」が既知なら `atoms: 12` のようにピンポイント指定するのがより堅牢です。

#### ステップ2：電荷・スピンの扱い（重要）

xTB dock では `.CHRG` / `.UHF` を読む場合、**3行（全体、分子1、分子2）**を入れる必要がある、と明記されています。([Xtb Docs][7])
一方、通常のxTB計算では `.CHRG`/`.UHF` は1行でよい（分子全体）説明もあります。([Xtb Docs][6])

なので、実装では：

* **dock用の作業ディレクトリ**に、dock仕様の `.CHRG/.UHF` を生成（3行）
* それ以外の用途（単分子preopt等）は通常仕様（1行 or CLI指定）でよい

を分けます（ディレクトリ分離が安全）。

また、dockでは `--chrg1/--chrg2` などで直接指定も可能です。([Xtb Docs][7])

#### ステップ3：実行コマンド（推奨）

* “アンサンブルで欲しい”ので基本は `--ensemble`
* “生成数”は初期は多め→後で絞る

例（概念）：

```bash
xtb dock A_conf01.xyz HF_conf01.xyz \
  --input dock.inp \
  --ensemble \
  --fast \
  --nfinal 30 \
  --gfn2 \
  > xtb_dock.out
```

`--ensemble` の意味（負の相互作用エネルギー構造を最適化し、重複除去してNCI ensemble化）が公式に書かれています。([Xtb Docs][7])
またこの機能は xTB 6.6+ にある点も明記されています。([Xtb Docs][7])

#### ステップ4：出力解析→ComplexSet

xTB dock ログには最終の相互作用エネルギーが一覧表示される例があります（Interaction energies）。([Xtb Docs][7])
実装ではログをパースして、

* `complex_id`
* `E_int_kcalmol`（dockが出している値）
* `geom_path`（抽出したXYZ）
* `source`（A_conf/B_conf、dock順位など）
* `geometry_checks`（最小距離、HF結合長が壊れてないか等）

を `complex_set.json` にまとめます。

> ここで「相互作用エネルギーの定義」を統一しておくと、後段で ΔE‡ の基準がブレにくいです。
> 例：`E_complex - (E_monomerA + E_monomerB)` を同一レベル（GFN2等）で再計算して揃える（dockのE_intは便利だが将来の一貫性のために再計算も選択肢）。

---

### 6.4 CREST NCI による複合体サンプリング（オプション補強）

上位の複合体（例：上位5件）に対して、

`crest complex.xyz --nci` ([Crest Lab][2])

を回すと、複合体の配座（相対配置＋内部回転）をさらに揺らして拾えます。

* NCIモードは楕円体壁ポテンシャルで解離を抑える、という説明があります。([Crest Lab][2])
* サイズ調整は `--wscal`。([Crest Lab][2])

**注意（設計に織り込むべき）**

* NCIは「複合体が解離する/潰れる」両方の失敗があり得るので、

  * 最小原子間距離チェック
  * 断片間距離が大きすぎる（解離）チェック
    を必須にします。

---

## 7. QC（品質保証）と失敗対策：自動化で必ず問題になる点と実装上の対策

### 7.1 よくある失敗と対策（ConformerSearch）

1. **xTB最適化が収束しない**

   * `NOT_CONVERGED` の生成を検知してリトライ/フォールバック ([Xtb Docs][10])
   * `--cycles` 増やす、`--opt normal` に落とす、GFN0→GFN2 の順にする ([Xtb Docs][10])

2. **CRESTのトポロジーチェックで落ちる/変な削除**

   * preoptを同一理論レベルで実施（CREST公式Tip）([Crest Lab][1])
   * 入力構造の結合異常（PubChem由来SDFの水素・価数）を上流で徹底修正

3. **アンサンブルが巨大（数百〜数千）**

   * `ewin` を調整（例：6→10）しつつ、下流投入数は別途絞る
   * `--quick`/`--squick`/`--mquick` のような軽量モードも検討（ただし精度/網羅性トレードオフ、利用するならconfigで明示）

### 7.2 よくある失敗と対策（ComplexEnumeration）

1. **ドッキングで原子が“融合”して最適化が失敗**

   * directed dockingのポテンシャルが強すぎる場合があるので `scaling factor` を下げる（0〜1推奨、例に0.9がある）。([Xtb Docs][7])
   * 最小原子間距離（例：0.7Å未満）で弾く

2. **複合体が解離してしまう**

   * xTB dock `--ensemble` は相互作用が負の構造を集めるので比較的マシ ([Xtb Docs][7])
   * CREST NCIを使うなら壁ポテンシャル＋`--wscal` 調整 ([Crest Lab][2])

3. **電荷指定の事故**

   * dock用 `.CHRG` は3行仕様、通常xTBは1行/CLI、をディレクトリ分離で実装に埋め込む ([Xtb Docs][7])

---

## 8. コード構成（例）：第三者が追加・改良しやすい形

### 8.1 モジュール構造（このフェーズ周辺のみ抜粋）

```
src/
  core/
    runners/
      subprocess_runner.py     # 外部コマンド実行（timeout, log, env, retry）
      tool_detect.py           # crest/xtb version取得、機能フラグ
    geom/
      xyz_io.py                # multi-xyz parse/write, energy comment handling
      rmsd.py                  # 近似RMSD/重複判定（簡易でも可）
      checks.py                # 物理チェック（最小距離、結合長レンジ等）
    hashing/
      fingerprint.py           # 入力ハッシュ（SDF/XYZ+設定）
  domain/
    models/
      species.py               # charge/uhf/reactive_sites
      ensemble.py              # Conformer, ConformerSet, Complex, ComplexSet (pydantic)
  plugins/
    conformer_generators/
      base.py
      crest_imtdgc.py          # CREST実装（CLI/TOML両対応）
    complex_builders/
      base.py
      xtb_dock_aiss.py         # xTB dock実装（directed/ensemble対応）
      crest_nci.py             # CREST --nci 実装
      hybrid.py                # dock→nci refinement
  tasks/
    conformers_generate.py     # ConformerSearchTask
    complexes_generate.py      # ComplexEnumerationTask
  cli/
    main.py                    # `pipeline run conformers ...` など
```

### 8.2 pluginインターフェース例（概念）

* `ConformerGenerator.generate(species: Species, input_xyz: Path, config) -> ConformerSet`
* `ComplexBuilder.generate(pair: PairSpec, confA: ConformerSet, confB: ConformerSet, config) -> ComplexSet`

> こうしておけば、将来「CREST→別手法」「xTB dock→別の配置生成」に差し替えても、タスクI/Oが壊れません。

---

## 9. MI観点の効率化（このフェーズでやると効く“実務的最適化”）

### 9.1 “計算資源の使い方”最適化

* **2段階の粗密**：

  * まず `GFN-FF` or 軽め設定で広く
  * 上位だけ `GFN2`、さらに上位だけ `CREST NCI` など
* **キャッシュ**：

  * `species_hash + config_hash + tool_version` をキーに `conformer_set.json` を再利用
* **並列**：

  * 分子単位・ペア単位で並列化（CRESTの `-T` は内部並列なので、外側並列と競合しないように“外側プロセス数×内側threads”を管理）
  * CRESTは `-T <threads>` を指定し、内部で最適並列化する旨が記載されています。([Crest Lab][3])

### 9.2 “下流TS探索の成功率を上げる”ための特徴量をここで付与

ComplexSetの各複合体に、次を計算してJSONに入れておくと、PES探索が劇的に安定します。

* 反応部位（例：アミンN）とHFのH/Fの距離
* N–H–F角
* 最小原子間距離（衝突検出）
* “プロトン移動っぽい”指標（N–H距離が短い、H–Fが伸びてる 等）

これを使って

* TS探索に渡す候補を「良さげな初期幾何」中心に絞る
* 反応テンプレ別に候補をクラスタリングする

が可能になります。

---

## 10. まとめ：このフェーズの推奨デフォルト（アミン×HFの気相）

* **単分子配座**：

  1. xTB preopt（GFN2、tight、失敗時フォールバック）([Xtb Docs][5])
  2. CREST iMTD-GC（GFN2 or GFN2//GFNFF、`crest_conformers.xyz`＋`crest.energies` を正にする）([Crest Lab][1])
* **遭遇複合体**：

  1. xTB dock aISS `--ensemble` を主にして列挙（directed dockingでNへ誘導）([Xtb Docs][7])
  2. 上位だけ CREST `--nci` で補強（必要時）([Crest Lab][2])
* **出力は必ず** `ConformerSet/ComplexSet`（JSON）＋生ファイルを保存し、原子インデックス写像を保持する。

---

もし次に進めるなら、あなたの上流「構造処理（RDKit）」で作っている **“反応部位候補（site定義）”** を前提にして、

* アミンNの自動同定（SMARTS/原子タイプ）
* HFのH/F識別
* directed dockingの `atoms:` を「原子番号指定」で生成
* 複合体の“反応座標スコアリング”の具体式（距離/角度/しきい値）

まで落として、`ComplexEnumerationTask` の仕様書（入出力JSONスキーマ含む）をさらに具体化できます。

[1]: https://crest-lab.github.io/crest-docs/page/examples/example_1.html "Conformational Sampling | CREST Docs"
[2]: https://crest-lab.github.io/crest-docs/page/examples/example_3.html "NCI Conformational Sampling | CREST Docs"
[3]: https://crest-lab.github.io/crest-docs/page/documentation/keywords.html "Command Line Keyword Documentation | CREST Docs"
[4]: https://crest-lab.github.io/crest-docs/page/documentation/inputfiles.html "Input File Documentation | CREST Docs"
[5]: https://xtb-docs.readthedocs.io/en/latest/optimization.html?utm_source=chatgpt.com "Geometry Optimization — xtb doc 2023 documentation"
[6]: https://xtb-docs.readthedocs.io/en/latest/sp.html?utm_source=chatgpt.com "Singlepoint Calculations — xtb doc 2023 documentation"
[7]: https://xtb-docs.readthedocs.io/en/latest/xtb_docking.html "Docking Submodule (aISS) — xtb doc 2023 documentation"
[8]: https://crest-lab.github.io/crest-docs/page/examples/publication/example_2.html "Metal-organic Systems | CREST Docs"
[9]: https://crest-lab.github.io/crest-docs/page/documentation/coords.html?utm_source=chatgpt.com "File Formats | CREST Docs"
[10]: https://xtb-docs.readthedocs.io/en/latest/basics.html?utm_source=chatgpt.com "Quickstart into Production — xtb doc 2023 documentation"
[11]: https://www.mankier.com/1/crest?utm_source=chatgpt.com "crest: Conformer-Rotamer Ensemble Sampling Tool"
