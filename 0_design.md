以下は、あなたが指定したスタック（RDKit / CREST+xTB / pysisyphus / NWChem / GoodVibes+Arkane+Cantera / 将来ClearML）を**維持**しつつ、先ほどの問題点（barrierless、会合エントロピー、圧力依存、入力SDF品質、ライセンス/差し替え）への対策を織り込み、**「個別タスクとして実行可能」かつ「パイプラインとして一連実行可能」**、さらに**第三者が拡張・修正しやすい**ことを主眼にした、アーキテクト観点の **全体構成＋コード設計の具体案**です。

また、**conda不可**の条件に合わせて、Python側は **pip前提**、xTB/CREST/NWChem等は **外部バイナリ（PATH）前提**で設計します（インストール方法も後述）。

---

# 1. 目標と設計原則

## 1.1 ゴール

* 入力：

  * `candidates/*.sdf`（1ファイル=1分子）
  * `reactants/*.sdf` もしくは `reactants.yaml`（HFや将来の他ガス）
  * `pipeline.yaml`（手法・条件・予算・並列度など）
* 出力：

  * 反応候補ごとの **ΔE‡ / ΔG‡（錯体基準・分離基準の両方）**
  * TS/IRC/QCフラグ
  * （オプション）GoodVibes/Arkane/Cantera **入力フォーマット**（このコード内で計算する/しないを切替可能）

## 1.2 コード設計の原則（第三者が理解しやすい）

1. **タスク粒度を固定**し、I/O（入力・出力・副産物）を明文化
2. **データモデルを統一**（JSONスキーマ/Typedモデル）して、ツール差を吸収
3. **実行は “Task＝サブプロセス単位”** に寄せる（ClearMLパイプラインのローカル実行に合わせる）

   * ClearMLはローカルモードで「ステップがサブプロセス実行」されることが明記されています ([ClearML][1])
4. **プラグイン方式**（反応テンプレ・PES探索エンジン・DFTエンジン等を差し替え可能）
5. **キャッシュ／再開（resume）／失敗の資産化**を最初から入れる（MI運用必須）

---

# 2. 全体アーキテクチャ（コンポーネントと責務）

## 2.1 レイヤ構造（維持＋強化）

* **Layer A: 構造処理（RDKit）**
  SDF正規化、フラグメント処理、H付与、ID付与、入力品質フラグ
* **Layer B: 配座/複合体（CREST + xTB）**
  配座列挙、遭遇複合体列挙（HF×候補）
* **Layer C: PES探索（pysisyphus）**
  NEB/GSM/TS/IRC、barrierless判定、QCフラグ

  * 将来：ASE+Sella+geomeTRICへ差し替え可能にする
* **Layer D: DFT（NWChem）**
  TS/周波数/IRC確認（必要時）、出力パース

  * 将来：Psi4、PySCF(+GPU)追加可能
* **Layer E: 熱化学/速度論（GoodVibes / Arkane / Cantera）**
  このコードでは **「入力フォーマット変換」中心**

  * GoodVibesは pip で入れられ、出力の整形に有用 ([PyPI][2])
  * Arkaneは RMG-Py由来で、公式にはDockerが推奨されているため、当面は「入力生成＋外部実行」想定が現実的 ([Reaction Mechanism Generator][3])
  * Canteraは維持（pipでも入る） ([Cantera][4])
* **Layer F: 実行管理（将来ClearML）**
  今はローカルで「ClearMLに移しやすい形」に寄せる

---

# 3. リポジトリ構成（理解しやすい “分割” を最優先）

以下のように「**core（共通基盤）**」「**interfaces（抽象）**」「**impl（実装）**」「**tasks（実行単位）**」を分けます。

```text
gasrxn-pipeline/
  pyproject.toml
  README.md
  docs/
    architecture.md
    task_catalog.md
    data_model.md
    troubleshooting.md
  configs/
    pipeline_hf_amine.yaml
    tools.yaml
    reactants.yaml
  src/gasrxn/
    __init__.py
    cli.py                       # gasrxn コマンド
    core/
      models/                    # Pydanticで統一データモデル
        molecule.py
        conformer.py
        complex.py
        reaction.py
        calc_result.py
        qc_flags.py
      io/
        sdf.py
        xyz.py
        json.py
        hashing.py               # content hash
      runtime/
        context.py               # 実行コンテキスト（run_dir, tool paths, etc）
        tracker.py               # LocalTracker（将来ClearMLTracker差し替え）
        cache.py                 # キャッシュ層
        subprocess.py            # 外部コマンド呼び出し共通
        logging.py
      plugins/
        registry.py              # プラグイン登録・発見
        base.py                  # 反応/エンジンのABC
    interfaces/
      standardize.py             # 分子標準化
      conformer.py               # 配座生成
      complex.py                 # 複合体生成
      reaction_template.py       # 反応テンプレ
      pes.py                     # PES探索
      dft.py                     # DFT実行
      export_thermo.py           # GoodVibes入力生成
      export_arkane.py           # Arkane入力生成
      export_cantera.py          # Cantera入力生成
    impl/
      standardize_rdkit.py
      conformer_crest_xtb.py
      complex_xtb_docking.py
      reaction_hf_proton_transfer.py
      pes_pysisyphus_xtb.py
      dft_nwchem.py
      export_goodvibes.py
      export_arkane.py
      export_cantera.py
      # 将来追加:
      # pes_ase_sella.py
      # dft_psi4.py
      # dft_pyscf.py
    tasks/
      base.py                    # Task ABC
      ingest_dataset.py
      standardize_dataset.py
      conformers_generate.py
      complexes_generate.py
      reactions_enumerate.py
      pes_search.py
      dft_refine.py
      thermo_export.py
      arkane_export.py
      cantera_export.py
      summarize_rank.py
    pipeline/
      dag.py                     # 依存関係解決（DAG）
      runner.py                  # ローカルパイプライン実行（サブプロセス）
  tests/
    test_models.py
    test_hashing.py
    test_task_io.py
```

---

# 4. “タスク”の基本仕様（個別実行できるように）

## 4.1 タスクの共通I/O（必須）

各タスクは必ず以下を守ります：

* **入力**：

  * `inputs.json`（参照する上流成果物のパス一覧＋メタ）
  * `params.json`（このタスクのパラメータ）
* **出力**：

  * `outputs.json`（成果物パス一覧＋要約メタ）
  * `metrics.json`（時間、件数、失敗数など）
  * `qc_flags.jsonl`（個体ごとのQCフラグ）
  * `logs/`（stdout/stderr、ツールログ）

これを “規約化” すると、第三者が途中成果を見て理解しやすいです。

## 4.2 タスクの実行単位（MI効率）

「SDFが多数」の前提なので、タスクは **dataset単位**で実行し、内部で **molecule単位並列**をかけられる設計が効率的です。

ただし「個別に実行したい」要求があるので、どのタスクも

* `--scope dataset`（全候補を処理）
* `--scope molecule --id <molecule_id>`（1分子だけ処理）

の両方を提供します。

---

# 5. ローカル実行パイプライン設計（ClearML移行しやすい）

## 5.1 “サブプロセス実行”をデフォルトにする

ClearMLのローカルモードは「ステップがサブプロセス実行」されると説明されています ([ClearML][1])。
したがって、今のローカル実装でも：

* パイプラインコントローラ（親プロセス）がDAG順に
* `gasrxn task run <task_name> ...` を **subprocess** で起動

する方式に寄せると、将来のClearMLパイプライン化が非常にスムーズです。

## 5.2 CLI設計（第三者が使いやすい）

例：

```bash
# データセット登録＋標準化（全件）
gasrxn task run ingest_dataset   --config pipeline.yaml
gasrxn task run standardize      --config pipeline.yaml

# 配座生成（特定分子だけ）
gasrxn task run conformers_generate --config pipeline.yaml --scope molecule --id MOL_000123

# パイプライン一括
gasrxn pipeline run --config pipeline.yaml --resume
```

---

# 6. データモデル（JSONで統一して “ツール差” を吸収）

ここはMI的に最重要です。
「RDKit→xTB→pysisyphus→NWChem」と段階が進むほど、ファイル形式がバラけます。
そこで**内部表現は必ず同じ**（Pydanticで型付け）にしておく。

## 6.1 Molecule（例）

* `molecule_id`: `MOL_<hash>`
* `source`: 元SDFパス、CID等
* `charge`, `multiplicity`
* `canonical_smiles`, `inchikey`（可能なら）
* `fragments_detected`: true/false
* `stereo_undefined_count`: int
* `geometry_ref`: `xyz`への参照

## 6.2 ConformerSet（例）

* `parent_molecule_id`
* `method`: `crest_gfn2xtb`
* `conformers[]`:

  * `conf_id`
  * `xyz_path`
  * `energy_hartree`（xTB）
  * `rank`
  * `qc_flags`

## 6.3 Complex（遭遇複合体）

* `complex_id`
* `a_molecule_id`, `b_molecule_id`
* `build_method`: `xtb_docking_aiss_directed`
* `xyz_path`
* `binding_energy_est`（低レベル指標）
* `orientation_tags`（反応テンプレに重要）

## 6.4 ReactionCandidate

* `reaction_id`
* `reaction_class`: `proton_transfer`
* `reactant_complex_id`
* `product_guess_id`
* `sites`: 原子マッピング（donor/acceptor等）
* `barrierless_candidate`: true/false（早期判定フラグ）

---

# 7. タスク設計（あなたの指定スタックに沿って具体化）

以下では「**処理内容**」「**入出力**」「**主要パラメータ**」「**拡張点**」「**効率化ポイント**」を、タスクごとにまとめます。

---

## Task 01: `ingest_dataset`（SDF取り込み・ID採番）

### 目的

* SDF群（1分子/ファイル）を読み、内部データモデルに登録
* “SDF品質問題”を検出してフラグ化（塩/多フラグメント、H欠落、立体未定義など）

### 実装

* RDKitで読み込み（pipでRDKitは導入可能） ([PyPI][5])
* `molecule_id = hash(canonical_smiles + charge + multiplicity + standardize_version)`

### 出力

* `registry/molecules.jsonl`
* `artifacts/molecules/<molecule_id>/raw.sdf`
* `artifacts/molecules/<molecule_id>/mol.json`（内部表現）

### 効率化

* ここで**重複排除**（同一分子が別名で混入してもOK）
* “多フラグメント”は後段爆死の原因なので、早期フラグが重要

---

## Task 02: `standardize`（RDKit正規化、H付与、3D下準備）

### 目的

* PubChem由来SDFの揺れを吸収し、計算可能な状態へ

### 処理

* 最大フラグメント抽出（デフォルト）
* 明示H付与
* 3D座標が無ければ埋め込み（RDKit）→後段CRESTの初期構造にする

### 出力

* `artifacts/molecules/<id>/std.sdf`
* `artifacts/molecules/<id>/std.xyz`

### 拡張点

* “反応クラス”に応じて「中和する/しない」「プロトン化状態列挙する/しない」を切り替え

---

## Task 03: `conformers_generate`（CREST + xTB）

### 目的

* 候補分子（単体）の配座を列挙し、**上位N**を後段へ

### 外部依存

* CREST は “リリースバイナリを展開してPATHに入れる” 形式での導入が想定されています ([Crest Lab][6])
* xTB も “GitHubのprecompiled binaries” が案内されています ([Xtb Docs][7])

### 入力

* `std.xyz`

### 出力

* `conformers/<molecule_id>/confs/*.xyz`
* `conformers/<molecule_id>/conformer_set.json`

### 主要パラメータ（pipeline.yamlで制御）

* `max_conformers_keep`（例：10〜30）
* `energy_window_kcal`（例：5〜10 kcal/mol）
* `crest_level`（gfn2 / gfnff 等）
* `n_jobs`（分子単位並列）

### 効率化ポイント（MI観点）

* **必ず“予算”を入れる**：配座爆発は最初のボトルネック
* 後段の複合体生成・PES探索は配座数に比例して爆増するため、上位Nで切る

---

## Task 04: `complexes_generate`（遭遇複合体：xTB docking / aISS / directed）

### 目的

* 候補分子×HF（将来は他ガス）で “遭遇複合体（pre-reactive complex）” を複数列挙

### 実装

* xTB docking（aISS）を外部実行
* 反応クラスが “donor/acceptor原子” を指定できる場合は directed docking を優先

### 入力

* `conformer_set.json`（上位N配座）
* `reactants.yaml`（HF等）
* `reaction_template` が指定する “注目原子集合”

### 出力

* `complexes/<candidate_id>/<reactant_id>/*.xyz`
* `complexes_index.jsonl`

### 主要パラメータ

* `max_complexes_per_pair`（例：10〜20）
* `directed_docking: on/off`
* `keep_nci_ensemble: on/off`

### 効率化ポイント

* barrierless系が多い（HF×塩基）ほど、複合体の向きが結果を支配 → **複合体列挙をケチるとランキングが崩れる**
* ただし無限に増やせないので、

  * 反応中心に近いものを優先
  * binding energy推定で上位を残す
    のようなフィルタを入れる

---

## Task 05: `reactions_enumerate`（反応テンプレで生成物推定）

### 目的

* 現状：HF×アミンのプロトン移動
* 将来：HF以外/アミン以外も増やせるように、**ReactionTemplateをプラグイン化**

### 実装（例：ProtonTransferテンプレ）

* donor（HF）のHをacceptor（N等）へ移した生成物構造を作る
* charge/multiplicity ルールもテンプレに持たせる（将来重要）

### 出力

* `reactions/<reaction_id>/reaction.json`
* `reactions/<reaction_id>/reactant.xyz`
* `reactions/<reaction_id>/product_guess.xyz`

### 拡張点

* `interfaces/reaction_template.py` を実装するだけで追加できる
* SMARTSでサイト検出を差し替え可能

---

## Task 06: `pes_search`（pysisyphusで NEB/GSM/TS/IRC）

### 目的

* 反応候補ごとに、TS候補探索・IRC検証・barrierless判定をする

### 実装

* pysisyphus は PyPIにあり、GPLv3であることが明記されています ([PyPI][8])
* ここでは **xTBをエネルギー/勾配エンジン**として使い、低忠実度で多数探索

### 重要：barrierless分岐（先ほどの対策を反映）

* TSが見つからない/単調減少 → `barrierless_candidate = true`
* 後段で「ΔG‡ではなく ΔG_assoc / 捕獲律速」側で扱う

### 出力

* `pes/<reaction_id>/path.xyz`（反応座標）
* `pes/<reaction_id>/ts_guess.xyz`
* `pes/<reaction_id>/irc_endpoints/`
* `pes/<reaction_id>/pes_result.json`（QCフラグ込み）

### 将来差し替え（ASE+Sella+geomeTRIC）

* **今の段階で設計として差し替え可能にしておく**（実装は後）

  * ASEはpipで入る（LGPL-2.1-or-later） ([PyPI][9])
  * geomeTRICはpipで入る ([GeomeTRIC Documentation][10])
  * SellaもPyPIにある ([PyPI][11])

---

## Task 07: `dft_refine`（NWChemで最終値を確定）

### 目的

* 上位候補（予算内）だけをDFTで精密化し、活性化自由エネルギー評価に耐える出力を得る

### 重要：このタスクは “prepare/run” を分ける

* `mode: prepare`：NWChem入力生成まで（外部投入は別）
* `mode: run`：ローカル実行（小規模なら）＋パース

NWChemはソースがGitHub releasesから取得でき、コンパイル手順ページも用意されています ([NWChem][12])
（conda不可でも「自前ビルド＋PATH」方式は成立）

### 入力

* `ts_guess.xyz`（pysisyphus）
* DFT条件（functional/basis/dispersion/grid 等）

### 出力

* `dft/<reaction_id>/nwchem.nw`（入力）
* `dft/<reaction_id>/nwchem.out`（出力）
* `dft/<reaction_id>/dft_result.json`（エネルギー/周波数/QC）

### 効率化（MI観点）

* DFTは最も重いので、**予算制御タスク（次項）**で絞る
* 低忠実度で “見込みが薄い候補” を落とし、DFT投入数を固定上限にする

---

## Task 07b: `budget_select`（DFTに回す候補選別：MI効率の中核）

### 目的

* xTB/PES結果をもとに、DFT精密化候補を決める

### 典型ルール（例）

* TS系：`xTB ΔE‡` が小さい順に上位K
* barrierless系：`ΔG_assoc_est`（会合が強い）や “反応熱” を重視
* 失敗系：再探索回数を超えたら除外（ログを残す）

### 出力

* `selection/dft_targets.jsonl`

---

## Task 08: `thermo_export`（GoodVibes用入力生成＋必要なら実行）

### 目的

* **このコードの主眼：フォーマット変換**
* GoodVibesはpipで導入でき、温度・濃度（=圧力換算）指定も可能 ([PyPI][2])

### モード

* `mode: prepare`：GoodVibes実行コマンド、入力ファイル一覧、PES YAML等を生成
* `mode: run`：`python -m goodvibes ...` を実行して出力を集約

### 出力（例）

* `thermo/goodvibes/input_files.txt`
* `thermo/goodvibes/pes.yaml`
* `thermo/goodvibes/goodvibes.tsv`（集約した結果）

---

## Task 09: `arkane_export`（Arkane入力生成：当面 “実行しない” 前提が現実的）

### 背景

ArkaneはRMG-Pyに含まれ、インストールはRMG-Pyに依存します ([Reaction Mechanism Generator][13])。
RMG-Pyの推奨導入はDockerで、conda無しでの“素直なpip導入”が前提になっていません ([Reaction Mechanism Generator][3])。
→ conda不可の条件では、**当面「入力生成＋Docker等で外部実行」**が安全です。

### 目的

* DFT/GoodVibes出力から、Arkaneの入力ファイル群（species/TS/reaction/network）を生成

### 出力例

* `arkane/inputs/arkane_input.py`
* `arkane/inputs/species/*.py`
* `arkane/inputs/ts/*.py`
* `arkane/README_run_in_docker.md`

---

## Task 10: `cantera_export`（Cantera入力生成：維持）

### 目的

* Canteraの機構ファイル（YAML）に落とすための変換を行う
* Canteraはpipでも導入可能 ([Cantera][4])

### モード

* `mode: prepare`：機構テンプレ生成（反応式、パラメータ枠）
* `mode: fill`：Arkaneなど外部結果を取り込んで埋める（将来）

---

## Task 11: `summarize_rank`（材料探索向けの最終集計）

### 目的（MI観点）

* “材料探索”として見たいのは単なるΔG‡一覧ではなく、少なくとも：

  * 反応タイプ（TS系 / barrierless系）
  * 錯体基準ΔG‡、分離基準ΔG‡、ΔG_assoc
  * QC（TS/IRC/虚数の数、収束、再現性）
  * 計算レベル（xTB/DFT）
    を含む表。

### 出力

* `summary/results.parquet`（推奨：大規模に強い）
* `summary/results.csv`
* `summary/topN.md`

---

# 8. プラグイン設計（手法の追加・改良を第三者がしやすい）

## 8.1 “インターフェース（ABC）”を先に固定する

例：PES探索エンジン

```python
# interfaces/pes.py
class PESExplorer(ABC):
    name: str
    @abstractmethod
    def prepare(self, reaction: ReactionCandidate, params: dict, workdir: Path) -> PreparedJob: ...
    @abstractmethod
    def run(self, job: PreparedJob) -> PESResult: ...
```

* `impl/pes_pysisyphus_xtb.py` はこのIFに従う
* 将来 `impl/pes_ase_sella.py` を追加しても、タスク本体は変更不要

同様に DFT も：

```python
class DFTEngine(ABC):
    def prepare_input(...)
    def run(...)
    def parse_output(...)
```

NWChem実装（今）／Psi4（将来）／PySCF（将来）を並列に保持できます。

---

# 9. “ツール導入（conda禁止）”を満たすための実装方針

## 9.1 Pythonライブラリ（pip前提）

* RDKit：PyPIのwheelが提供されており `pip install rdkit` が案内されています ([PyPI][5])
* pysisyphus：PyPIで提供（GPLv3） ([PyPI][8])
* GoodVibes：PyPIで `pip install goodvibes` が明記 ([PyPI][2])
* Cantera：pip導入の公式ドキュメントあり ([Cantera][4])
* PySCF：pip導入が推奨されている ([pyscf.org][14])
* gpu4pyscf：PyPIで配布（例：2025/12/26リリース情報） ([PyPI][15])
* ASE：PyPIで配布（LGPL-2.1-or-later） ([PyPI][9])
* geomeTRIC：pipで導入可能 ([GeomeTRIC Documentation][10])
* ClearML：pipで導入可能（将来） ([PyPI][16])

`requirements.txt`（例）：

```text
pydantic>=2
typer>=0.12
rich>=13
ruamel.yaml>=0.18
numpy
scipy
pandas
pyarrow
networkx
rdkit
pysisyphus
goodvibes
cantera
# 将来差し替え用（今はoptional）
ase
sella
geometric
pyscf
gpu4pyscf
clearml
```

> ※ `ase/sella/geometric/pyscf/gpu4pyscf/clearml` は “extras” として分離しても良いです（`pip install .[ase]` のように）。

## 9.2 外部バイナリ（PATHに置く：pipでは入らない）

* xTB：precompiled binariesの案内がある ([Xtb Docs][7])
* CREST：リリースバイナリを展開してPATHへ、が案内されている ([Crest Lab][6])
* NWChem：ソースはGitHub releases、コンパイルページあり ([NWChem][12])

### コード側の扱い

`tools.yaml` で明示し、`ToolRegistry` が起動時に `--version` で検査して `run_manifest.json` に記録します（再現性とデバッグ性が上がる）。

---

# 10. ClearML（将来）を見越した “今の実装” の入れ方

## 10.1 追跡（Tracking）を抽象化する

* `Tracker` IF を作り、今は `LocalTracker`（JSONとファイルコピー）で実装
* 将来 `ClearMLTracker` に差し替え

ClearMLはパイプラインをローカルで動かせる（サブプロセス）ことが明記 ([ClearML][1])。
したがって、

* **今：pipeline runner = subprocess実行**
* **将来：ClearML PipelineDecorator = subprocess実行**

という一致が取れます（移行が最小）。

---

# 11. 最後に：この設計が “半導体ガスエッチング×材料探索” に効く理由（MI観点）

1. **予算制御（budget_select）**でDFT投入数を固定し、探索が破綻しない
2. **barrierless分岐**を設計に含め、HF×塩基の“TSが無い/浅い”ケースで止まらない
3. **錯体/分離の二基準**でΔG‡を出せるので、会合が効く装置条件でも解釈がぶれにくい
4. すべてのタスクが **同じI/O規約（inputs/outputs/metrics/qc_flags）**で動くため、第三者が途中から入っても追える
5. ClearMLの将来導入に合わせて「サブプロセス実行」「成果物・パラメータの規約化」を先に整えている

---

[1]: https://clear.ml/docs/latest/docs/pipelines/?utm_source=chatgpt.com "ClearML Pipelines"
[2]: https://pypi.org/project/goodvibes/3.2/ "goodvibes · PyPI"
[3]: https://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/index.html "3. Installation — RMG-Py 3.3.0 Documentation"
[4]: https://cantera.org/3.1/install/pip.html?utm_source=chatgpt.com "Installing with Pip — Cantera 3.1.0 documentation"
[5]: https://pypi.org/project/rdkit/?utm_source=chatgpt.com "rdkit"
[6]: https://crest-lab.github.io/crest-docs/page/installation/install_basic.html?utm_source=chatgpt.com "Step-by-step Installation | CREST Docs"
[7]: https://xtb-docs.readthedocs.io/en/latest/setup.html?utm_source=chatgpt.com "Setup and Installation — xtb doc 2023 documentation"
[8]: https://pypi.org/project/pysisyphus/?utm_source=chatgpt.com "pysisyphus"
[9]: https://pypi.org/project/ase/?utm_source=chatgpt.com "ase"
[10]: https://geometric.readthedocs.io/en/1.1/install.html?utm_source=chatgpt.com "Installation — geomeTRIC 1.1 documentation"
[11]: https://pypi.org/project/Sella/?utm_source=chatgpt.com "Sella"
[12]: https://nwchemgit.github.io/Download.html?utm_source=chatgpt.com "How to download and install NWChem"
[13]: https://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/installation.html "2. Installation — RMG-Py 3.3.0 Documentation"
[14]: https://pyscf.org/user/install.html?utm_source=chatgpt.com "How to install PySCF"
[15]: https://pypi.org/project/gpu4pyscf/?utm_source=chatgpt.com "gpu4pyscf"
[16]: https://pypi.org/project/clearml/?utm_source=chatgpt.com "clearml"
