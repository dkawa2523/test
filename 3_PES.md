以下では **PES探索（pysisyphus）フェーズ**を、あなたがこれまでに固めてきた全体ワークフロー（RDKit→CREST+xTB→pysisyphus→DFT→熱化学/速度論→将来ClearML）を崩さずに、**アーキテクト観点（タスク分割・I/O・拡張性・第三者が追える設計）**と **MI観点（スクリーニング効率・失敗時のリカバリ・計算資源の使い方）**で、できるだけ具体的に落とし込みます。

---

## 1. PES探索フェーズのゴールと「前提条件」

### ゴール（この層で確実に出したい成果物）

* **反応経路（MEPの近似）**：NEB/GSM（Growing String）などで「反応が起きる道筋」を作る
* **TS候補（鞍点）**：TS最適化（rsirfo/rsprfo/trim あるいは dimer）で **1次の鞍点** を得る
* **IRCで接続検証**：TSが「想定した反応物⇄生成物」を本当に結ぶかを確認し、端点を最適化する
* 上記から **スクリーニング用の活性化障壁（最低限 ΔE‡、できれば後段で熱補正して ΔG‡）** を出す

pysisyphusは、**COS（NEB/GSM）→ TS最適化 → IRC → 端点最適化**までを YAML で一括実行できる設計になっており、この目的に合っています。([Pysisyphus][1])

### 前提条件（ここを満たさないと壊れる）

* **反応物側と生成物側で “原子順序が一致” していることが必須**（同一原子集合で経路を張るため）。pysisyphusのWorked Exampleでも「consistent atom ordering is mandatory」と明記されています。([Pysisyphus][1])

  * 今回（アミン + HF → プロトン移動/付加/解離など）は原子集合が同じなので成立しやすい
  * 将来、反応物/生成物の分子数が変わる・原子が出入りするケースは「同一原子集合の超分子表現」に落とす設計が必要（＝あなたの“遭遇複合体”設計が非常に重要）

---

## 2. pysisyphus の実運用前提（pip・外部バイナリ・rc管理）

### インストール（pipのみ）

pysisyphus は PyPI から pip で入れられます。([Pysisyphus][2])

```bash
python -m pip install pysisyphus
```

### 外部コード（xTBなど）の見つけ方：.pysisyphusrc

pysisyphusは外部QCコードの場所を **`$HOME/.pysisyphusrc`** に登録する想定です。xTBは例として以下のように `cmd=xtb` を書く形式です。([Pysisyphus][2])

```ini
[xtb]
cmd=xtb
```

### 実行形態：YAML入力を `pysis` で実行

Worked Exampleでも、YAMLを用意して `pysis <yaml> | tee pysis.log` として STDOUT をログ化する流れが示されています。([Pysisyphus][1])

---

## 3. アーキテクト観点：PES層の「タスク分割」と責務

あなたの要求（**各計算・評価は個別タスクとして実行可能**／それをパイプラインでも直列実行できる）に合わせて、PES層を次の **最小DAG** に分割するのが扱いやすいです。

### 推奨タスク分割（最小で破綻しない粒度）

1. **`pes_prepare_endpoints`**

   * 入力：ReactantComplex（遭遇複合体） geometry、ProductGuess geometry
   * 出力：pysisyphusに渡す **原子順序一致済み endpoints（xyz）**、charge/mult、マッピングメタ
   * 役割：**PESに入れる幾何の整合性担保**（ここが最重要）

2. **`pes_cos`**（Chain-of-States：GSM/NEB）

   * 入力：endpoints、COS設定
   * 出力：最終パス（例：`final_geometries.trj`）と HEI（例：`splined_hei.xyz`）
   * 役割：**TS探索の初期構造（HEI）**を作る
   * “過収束しない”設計にして計算量を抑える（後述）([Pysisyphus][3])

3. **`pes_tsopt`**（TS最適化）

   * 入力：TS guess（HEI）、TS最適化設定
   * 出力：TS最適化構造（例：`ts_opt.xyz`）、（可能なら）Hessian/imag mode
   * 役割：**鞍点を確定**（不正な極小に落ちたら失敗扱い）

4. **`pes_irc`**（IRC＋端点最適化）

   * 入力：TS構造、IRC設定
   * 出力：IRCデータ（例：`irc_data.h5`）＋前後端点（例：`forward_end_opt.xyz` / `backward_end_opt.xyz`）
   * 役割：**TSが正しく反応物/生成物を結ぶか検証**。弱結合系は `fragments` で分割最適化が効く。([Pysisyphus][4])

5. **`pes_validate_and_summarize`**

   * 入力：COS/TS/IRCの成果物
   * 出力：`pes_result.json`（障壁、QCフラグ、採否、次段DFTに渡すTS構造パス、等）

> 実務的には、pysisyphusは 1つのYAMLで COS→TSopt→IRC→endopt を連結できます（Worked Example）。([Pysisyphus][1])
> ただしあなたの要件は「個別タスク」なので、**内部では同じrunnerを使いつつ“YAMLテンプレを段階ごとに切る”**のが設計上きれいです（将来ClearMLで各段を別Experimentにもできる）。

---

## 4. コード構成案（将来のASE+Sella+geomeTRIC差し替え前提）

### ディレクトリ/モジュール（例）

```text
src/gasrxn/
  core/
    models/                 # Pydantic/dataclass: Geometry, ReactionPair, PESJob, PESResult...
    io/                     # xyz/trj/sdf/json/h5 の読み書き（共通）
    hash/                   # 入力・設定のcontent hash（キャッシュキー）
    runners/
      subprocess_runner.py  # 外部コマンド実行（stdout/stderr収集、タイムアウト等）
  backends/
    pes/
      base.py               # IPESBackend interface
      pysisyphus/
        builder.py          # YAML生成（COS/TS/IRC/SCANテンプレ）
        runner.py           # pysis呼び出し、環境(.pysisyphusrc)準備
        parser.py           # 出力ファイル探索/抽出/要約（ログから拾う/ファイルglob）
  tasks/
    pes/
      prepare_endpoints.py
      cos.py
      tsopt.py
      irc.py
      validate.py
  configs/
    pes_defaults.yaml
    pes_profiles.yaml       # reaction type別（proton_transfer等）プロファイル
  cli.py                    # typer/click で task単体実行 + pipeline実行
```

### バックエンド差し替えのためのインターフェース

将来、ASE+Sella+geomeTRICへ移行してもタスク層を変えないため、PES層は「バックエンド注入」にします。

```python
class IPESBackend(Protocol):
    def run_scan(self, job: PESJob) -> PESStageArtifacts: ...
    def run_cos(self, job: PESJob) -> PESStageArtifacts: ...
    def run_tsopt(self, job: PESJob) -> PESStageArtifacts: ...
    def run_irc(self, job: PESJob) -> PESStageArtifacts: ...
```

* **pysisyphus backend**：YAMLを書いて `pysis` 実行 ([Pysisyphus][1])
* **将来のASE backend**：Python APIでNEB/TS/optを回す（ただしI/Oの契約は同じ）

---

## 5. 具体処理：`pes_cos`（NEB/GSM）設計

### 5.1 COSメソッドの選び方（ガス相・遭遇複合体向けの現実解）

pysisyphusは NEB（＋Adaptive/Free-End等）と String系（GSM, SZTS）を提供し、**GSMは内部座標でも利用可能**とされています。([Pysisyphus][3])

* **GSM（Growing String）推奨ケース**

  * 反応物/生成物が「それなりに近い構造」（同じ複合体の中で、結合が入れ替わる/プロトンが移る等）
  * すでにCRESTで“遭遇複合体”が妥当な配置を持っており、生成物側もそれなりに作れている
  * TS guess を効率よく作りたい（MIスクリーニング向き）

* **NEB推奨ケース**

  * 生成物guessが怪しい／端点が遠い／経路が複雑でGSMが不安定
  * 画像数を増やして経路を丁寧に追いたい
  * ガス相で「会合・解離」が絡む場合は、NEBのcartesianで `align` を使って全体の並進・回転を取りたい（後述）([Pysisyphus][3])

### 5.2 「計算を軽くする」ためのCOS運用の原則

pysisyphusのCOS一般助言として、

* **過収束しない**（COSをゆるく収束→HEIをTS探索に渡すほうが良い）([Pysisyphus][3])
* **climbing image** を可能なら使う（`climb: True`）([Pysisyphus][3])
* **cartesianでCOSする場合は align: True**（並進・回転除去）ただし **DLCでは使わない** ([Pysisyphus][3])
  が明確に書かれています。

この方針は **候補分子の材料探索スクリーニング**にとても効きます。COSで数百サイクル回しても、結局TSopt/IRCで詰まるなら無駄が多いからです。

### 5.3 pysisyphus COSのYAMLテンプレ（GSM）

chain-of-statesドキュメントにGSM例があり、主要パラメータ（max_nodes, climb, reparam系, stop_in_when_full等）が示されています。([Pysisyphus][3])

```yaml
geom:
  type: dlc
  fn: [reactant.xyz, product.xyz]

calc:
  type: xtb
  pal: 4
  charge: 0
  mult: 1

preopt: {}     # 端点を同一理論で軽く整える（推奨）:contentReference[oaicite:15]{index=15}

cos:
  type: gs
  max_nodes: 9
  climb: true
  climb_rms: 0.005
  reparam_every: 2
  reparam_every_full: 3

opt:
  type: string
  align: false          # DLCではalign禁止 :contentReference[oaicite:16]{index=16}
  stop_in_when_full: 2  # fully grown後、2cycleで止める（スクリーニング用）
  max_cycles: 30
```

**設計上のポイント**

* `stop_in_when_full` を積極的に使って「fully grownになったら早く止める」—MIスクリーニングに効く ([Pysisyphus][3])
* 端点が弱結合（分子が離れやすい）なら `preopt.max_cycles` を小さく制限（Worked Exampleでも非共有結合系への注意あり）([Pysisyphus][1])

### 5.4 NEBのYAMLテンプレ（補間・align・TS/IRCまで連結）

chain-of-statesページにNEB例があり、`interpol.type`（redund|idpp|lst|linear）や `between`、`align` などが具体例として出ています。([Pysisyphus][3])

```yaml
geom:
  type: cart
  fn: reactant_product_pair.trj

calc:
  type: xtb
  charge: 0
  mult: 1
  pal: 4

preopt:
  max_cycles: 5

interpol:
  type: idpp
  between: 10

cos:
  type: neb
  climb: false
  align_fixed: true

opt:
  type: lbfgs
  align: true
  align_factor: 0.9
  rms_force: 0.01
  max_step: 0.04
```

**ガス相での注意点（会合/解離）**

* 会合反応は「分子同士が離れていく自由度」を持つので、NEBをcartesianで回す場合は `align: True` が重要になりがちです（並進・回転が混ざると経路が崩れる）。([Pysisyphus][3])
* ただし DLC/内部座標で回すなら align は使わない（明記されています）。([Pysisyphus][3])

### 5.5 COSを並列化するか？（Dask）

pysisyphusは Dask.distributed による「複数画像の並列評価」をサポートし、`cos.cluster: True` で簡単に有効化できる、と書かれています。([Pysisyphus][3])

MIスクリーニング的には、

* **候補が多い**：まずは「反応候補（ジョブ）単位の並列化」で十分
* **1ジョブが重い（画像が多い/DFTを使う）**：画像並列も効く

の順で使うのが安全です（CPUの二重取りを避けるため）。

---

## 6. 具体処理：`pes_tsopt` 設計（TS探索の安定化が肝）

### 6.1 TS最適化は「内部座標 + 反応座標の明示」が効く

TS optimizationドキュメントに以下が明確に書かれています：

* TS最適化は internal coordinates（`redund|dlc`）が望ましい
* **`pysistrj [xyz] --internals` で座標が揃っているか確認し、足りないなら `add_prims` で追加** ([Pysisyphus][5])
* `rx_modes` や `prim_coord` で「登るべきモード」を指定できる（プロトン移動などで特に重要）([Pysisyphus][5])

これは HF + アミンのような **プロトン移動**で非常に効きます（誤ったモードを登って別のTSへ行きがち）。

### 6.2 COS→TSopt連結のメリット（pysisyphusの強み）

chain-of-statesの一般remarksとして、

* COSの初期/最終画像の情報から **TS guessの内部座標セットをより完全に構築**でき、重要座標が抜けにくい
* 初期の虚モードは **HEI tangentとのoverlapが最大のものを選ぶ**
  と書かれています。([Pysisyphus][3])

つまり、**COSでまともな経路が作れればTS探索は相当安定**します。

### 6.3 TSoptのYAMLテンプレ（Hessianベース）

TS optimizationページのYAML例（rsirfo/rsprfo/trim）に沿って、スクリーニング用に寄せた例です。([Pysisyphus][5])

```yaml
tsopt:
  type: rsirfo
  do_hess: true          # 最終Hessian → 虚振動1本か確認 :contentReference[oaicite:27]{index=27}
  hessian_recalc: 3      # コストと安定性のトレードオフ
  thresh: gau_loose

  # プロトン移動なら、例えば「H-F結合」や「N-H結合」を prim_coord で示す
  prim_coord: [BOND, i_H, i_F]
  # あるいは rx_modes でより一般化したモード指定
  # rx_modes: [[[[BOND, i_H, i_F], 1]]]
calc:
  type: xtb
  pal: 4
  charge: 0
  mult: 1
geom:
  type: redund
  fn: splined_hei.xyz
  add_prims:
    - [i_H, i_F]          # 必要なら明示追加 :contentReference[oaicite:28]{index=28}
    - [i_H, i_N]
```

### 6.4 TSoptの成果物とパーサ設計（ファイル名に依存しすぎない）

Worked Exampleログでは、典型的に

* COSの最終パス：`final_geometries.trj`
* HEI：`splined_hei.xyz`
* TS最終：`ts_opt.xyz` / `ts_final_geometry.xyz`
* Hessian：`final_hessian.h5`
* 虚モード：`imaginary_mode_000.trj`
  などが出ています。([Pysisyphus][1])

**ただしファイル名はバージョン/設定で変わり得る**ので、設計では

* **run_dirを丸ごとartifactとして保持**
* その上で parser は

  * `*.h5`（Hessian/IRC）
  * `*ts*_opt*.xyz` / `*final_geometry*.xyz`
  * `*hei*.xyz`
  * `*.trj`
    をグロブで探して、見つかったものを `pes_result.json` にリンクする
    という **耐久性重視**が良いです。

---

## 7. 具体処理：`pes_irc` 設計（接続検証と弱結合系の扱い）

IRCページの仕様として：

* デフォルトは **EulerPC**（推奨・標準）
* 初期変位は「一定のエネルギー低下量 dE を満たすステップ長」方式（デフォルト dE=0.0005 au）
* 端点は `endopt` でさらに最適化でき、弱結合なら `fragments: True|total` が使える
* IRCの経路データは `irc_data.h5` に周期ダンプされ、`pysisplot --irc` で描画可能 ([Pysisyphus][4])

### IRCテンプレ

```yaml
geom:
  fn: ts_opt.xyz
calc:
  type: xtb
  pal: 4
  charge: 0
  mult: 1
irc:
  type: eulerpc
  rms_grad_thresh: 0.001
  # displ: energy
  # displ_energy: 0.001
endopt:
  fragments: total   # ガス相で分子が離れる/弱結合なら有効 :contentReference[oaicite:31]{index=31}
  do_hess: false
```

### `pes_validate_and_summarize`でやるべきQC（最低限）

* **TSが一次鞍点か**：虚振動が1本（pysisyphusが `do_hess: True` でHessianを出せる）([Pysisyphus][1])
* **IRC forward/backward の端点が、想定の反応物/生成物に対応するか**

  * 端点最適化後のRMSDや構造比較で判断（Worked ExampleでもRMSDが接続判定に使えると説明）([Pysisyphus][1])
  * 弱結合でバラける場合は fragments最適化で安定に比較する ([Pysisyphus][4])

---

## 8. “ガスプロセス/遭遇複合体”に特有の問題点と、PES層での対策

### 問題1：会合/解離・並進回転が経路を壊す

* 対策：

  * NEBをcartesianで回す場合は `align: True` を徹底（ただしDLCでは禁止）([Pysisyphus][3])
  * **端点固定（fix_ends）** を基本にする（COS助言にあり）([Pysisyphus][3])
  * それでも崩れる場合：**拘束（restrain）** を導入して「COM距離」等を暴れにくくする

    * pysisyphusの calculators には ExternalPotential/Restraint の入力例があり、BOND等の原始内部座標拘束が書けます。([Pysisyphus][6])
    * 実装設計としては「PESJobに optional restraints を持たせる」→「YAMLの calc: を ext にして wrap」などの拡張点を用意

### 問題2：プロトン移動はTSが複数・モード選択が難しい

* 対策：

  * COS→TSopt 連結で **HEI tangent overlap** に基づくモード選択を活用（pysisyphusが標準でやる）([Pysisyphus][3])
  * それでも誤る場合は `prim_coord` / `rx_modes` を **反応中心（H-F / H-N）**に寄せる ([Pysisyphus][5])
  * `add_prims` で内部座標に反応座標を強制追加（TSoptドキュメントの推奨）([Pysisyphus][5])

### 問題3：候補数が多い（材料探索）→ 計算爆発

* 対策（MI流の“予算段階”）

  1. **短いCOS**（`stop_in_when_full` を使う / サイクル上限低め）→ TS guess を作る ([Pysisyphus][3])
  2. TSoptでダメなら「その複合体は捨てる」ではなく、**別の遭遇複合体**（CRESTが出した別配置）へ
  3. 上位だけIRC（検証はコストがかかるので最後）
  4. DFTはさらに上位だけ

  * COSを“過収束しない”のはpysisyphus自身が推奨しており、スクリーニング設計と一致します。([Pysisyphus][3])

---

## 9. タスクI/O（JSON）設計例：第三者が追いやすく、将来ClearMLに載せやすい形

### `PESJob`（入力）

```json
{
  "job_id": "rxn_000123__cx_04__site_N1",
  "reactant_xyz": "artifacts/endpoints/reactant.xyz",
  "product_xyz": "artifacts/endpoints/product.xyz",
  "charge": 0,
  "mult": 1,
  "profile": "proton_transfer",
  "backend": "pysisyphus",
  "settings": {
    "calc": {"type": "xtb", "pal": 4},
    "cos": {"method": "gs", "max_nodes": 9, "climb": true, "stop_in_when_full": 2},
    "tsopt": {"type": "rsirfo", "do_hess": true, "hessian_recalc": 3},
    "irc": {"type": "eulerpc", "rms_grad_thresh": 0.001, "endopt_fragments": "total"}
  }
}
```

### `PESResult`（出力）

```json
{
  "job_id": "rxn_000123__cx_04__site_N1",
  "status": "success|failed|needs_retry",
  "barrier": {
    "deltaE_kjmol": 12.3,
    "reference": "reactant_min"
  },
  "artifacts": {
    "cos_path_trj": "runs/.../final_geometries.trj",
    "ts_xyz": "runs/.../ts_opt.xyz",
    "hessian_h5": "runs/.../final_hessian.h5",
    "irc_h5": "runs/.../irc_data.h5",
    "irc_end_forward": "runs/.../forward_end_opt.xyz",
    "irc_end_backward": "runs/.../backward_end_opt.xyz",
    "log": "runs/.../pysis.log",
    "input_yaml": "runs/.../stage.yaml"
  },
  "qc": {
    "n_imag_freq": 1,
    "irc_connected": true,
    "notes": ["..."]
  },
  "provenance": {
    "backend_version": "pysisyphus ...",
    "cmd": "pysis stage.yaml",
    "hash": "contenthash..."
  }
}
```

---

## 10. 実装の要点（pysisyphus runner を“堅牢”にする）

### 10.1 `.pysisyphusrc` を runner が自己完結で用意する

ユーザ環境の `$HOME/.pysisyphusrc` に依存すると再現性が落ちます。
設計としては runner が

* `run_dir/.pysisyphusrc` を生成（最低限 `[xtb] cmd=xtb`）([Pysisyphus][2])
* `subprocess env` の `HOME=run_dir` にして `pysis` を叩く

にすると、**ジョブごとに完全に自己完結**します（ClearML化しても壊れにくい）。

### 10.2 pysisyphusのSTDOUTは必ず捕まえる

Worked Exampleにも「pysisyphus prints to STDOUTなので capture が必要」と明記があります。([Pysisyphus][1])
runner は `stdout` をファイルにteeするか、Python側でリダイレクトして保存してください。

---

## 11. ここまでの設計で「将来ASE+Sella+geomeTRIC」に移行できる理由

* いま定義した **PESJob/PESResult と stage artifacts** を契約にしておけば、バックエンドが

  * pysisyphus（YAML+CLI）でも
  * ASE+Sella+geomeTRIC（Python API）でも
    生成する成果物（TS xyz / IRC端点 / barrier / qc flags）を同じ構造で返せます。

* 特に重要なのは **「原子順序一致 endpoints」を作る `pes_prepare_endpoints`** をバックエンド非依存にすること

  * これを守れば、差し替えコストは “経路探索・TS探索の実装部分” に限定できます。

---

もし次に進めるなら、PES層の次に「**pysisyphusの出力をDFT層（NWChem/将来Psi4/PySCF）へ渡すためのフォーマット**（TS xyz → 入力テンプレ、原子ラベル・電荷・スピン・周波数/IRCの扱い）」を、**“DFT計算タスク設計”**として同じ粒度で具体化できます。

[1]: https://pysisyphus.readthedocs.io/en/latest/worked_example.html "https://pysisyphus.readthedocs.io/en/latest/worked_example.html"
[2]: https://pysisyphus.readthedocs.io/en/latest/installation.html "https://pysisyphus.readthedocs.io/en/latest/installation.html"
[3]: https://pysisyphus.readthedocs.io/en/latest/chainofstates.html "https://pysisyphus.readthedocs.io/en/latest/chainofstates.html"
[4]: https://pysisyphus.readthedocs.io/en/master/irc.html "https://pysisyphus.readthedocs.io/en/master/irc.html"
[5]: https://pysisyphus.readthedocs.io/en/latest/tsoptimization.html "https://pysisyphus.readthedocs.io/en/latest/tsoptimization.html"
[6]: https://pysisyphus.readthedocs.io/en/latest/calculators.html "https://pysisyphus.readthedocs.io/en/latest/calculators.html"
