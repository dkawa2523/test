以下では **DFT（NWChem）フェーズ**を、あなたがこれまで定義してきた全体パイプライン設計（タスク分割・パイプライン実行・将来ClearML対応・将来Psi4/PySCF追加可能）を維持したまま、**アーキテクト観点＋計算化学/MI観点**で「実装に落ちる粒度」まで具体化します。
（ここでは **ガス相**、入力は前フェーズ（CREST/xTB・pysisyphus）から渡ってくる **3D構造（複合体/TS候補など）**を前提にしています。）

---

## 1. DFT（NWChem）フェーズの役割と“やらないこと”の線引き

### このフェーズでやること（責務）

* **前フェーズ（xTB/PES探索）で得た構造**（反応物複合体、生成物複合体、TS候補…）を

  * **DFTレベルで最適化（minima/TS）**
  * **DFTレベルのエネルギー（E）を取得**
  * **必要に応じて振動数（freq）でTS検証・熱補正用データを取得**
* “計算結果の**標準化**”と“後段（GoodVibes/Arkane/Cantera）に渡すための**フォーマット化**”

  * ただし、あなたの方針どおり **熱化学/速度論の実計算自体をこのコード内で完結させる必要はない**
    → このフェーズの出力を「後段が使える形」に整えることを最重要にする

### このフェーズで “やらないこと”（境界）

* 反応経路探索（NEB/GSM/IRC 等）の主体は **pysisyphusフェーズ**に残す（あなたの設計維持）
* GoodVibes/Arkane/Cantera を **必ず実行**する責務は持たない（将来タスク化は可能）
* “どの反応経路が本命か”の探索ロジック自体は PES探索側（候補生成側）に寄せる
  → DFTは「**候補の精密評価**」に集中

---

## 2. NWChem採用の前提・インストール（conda禁止への対応）

### NWChemの導入（condaを使わない）

NWChem公式ドキュメントでは、Linuxディストリのパッケージや Homebrew などで導入できます。例として Debian/Ubuntu は `apt-get install nwchem`、macOS は `brew install nwchem` が案内されています。並列実行例も示されています。 ([NWChem][1])

> 注意: OS配布パッケージはバージョンが古い場合があります。**結果再現性**のため、出力から **NWChem version** を必ず取得してメタデータに保存してください（後述）。

### 実行形式（stdout）

Getting Started には、入力が `n2.nw` の場合の実行コマンド例 `nwchem n2` と、**出力はUNIX標準出力へ出る**旨が明記されています。 ([NWChem][2])
→ あなたのパイプラインでは **stdoutをログファイルへ保存する**設計が自然です。

---

## 3. DFTフェーズのI/O契約（Artifacts）を固定する

タスク分割・将来拡張のため、DFTフェーズは **“入力/出力スキーマ”を最優先で固定**します。
（コードは変えやすいが、Artifactsを変えると全フェーズに波及するため）

### 入力（前フェーズから来る想定）

最小セット:

* `Structure3D`（あなたの内部表現）

  * `symbols: list[str]`
  * `coords_angstrom: (N,3)`
  * `charge: int`
  * `multiplicity: int`
  * `tags`: 断片ID/役割（reactant_complex / ts_guess / product_complex など）
* `DFTJobPlan`

  * job種別（minima最適化、TS最適化、freq、SP など）
  * レベル（functional/basis/grid/disp 等）
  * 実行資源（nproc, memory, scratch）

> 重要: **SDF由来の電荷・プロトン化状態が正しい**ことが前提です。DFTフェーズでは「与えられた charge/mult を信じて回す」設計にし、もし不整合が起きたら **入力側（構造処理フェーズ）へフィードバックする**のが責務分離として健全です。

### 出力（後フェーズへ渡す標準成果物）

* `DFTResult`（内部JSON）

  * `engine`: `"nwchem"`
  * `engine_version`: 出力ログから抽出（必須）
  * `level_of_theory`: functional/basis/disp/grid/…（必須）
  * `job_type`: optimize/saddle/frequencies/energy
  * `status`: success/failed + error分類
  * `final_geometry`: coords
  * `energies`: electronic energy など（最低 `E_elec`）
  * `frequencies`: list[float]（cm^-1）+ imaginary_count
  * `files`: `input.nw`, `stdout.log`, （必要なら `.db/.movecs` など）のパス
  * `provenance`: 実行host, cmdline, timestamp, git commit, pipeline run id, hash…

---

## 4. NWChem入力生成の要点（テンプレート設計の核心）

### 4.1 “1 deckに複数TASK”をどう扱うか

NWChemは **TASKが現れるまで入力を読み、TASKを実行し、次のTASKへ進む**という挙動で、**DBは永続**なので「同一job内の複数TASKは restart job と同等」と説明されています。 ([NWChem][3])

この性質を利用すると、例えば

* minima: `task dft optimize` → `task dft frequencies`
* TS: `task dft saddle` → `task dft frequencies`

を **1つの入力ファイル**で行えます。

ただし運用上は2つの設計があり得ます:

1. **結合deck方式（推奨デフォルト）**

   * 1ジョブで optimize→freq まで完結
   * DB引き継ぎが自然で、ファイル管理も単純
2. **分割ジョブ方式（障害復旧が強い）**

   * optimize成功後に別ジョブでfreq
   * freq失敗時に optimize結果を保持しやすい
   * ClearML化したときに再実行粒度を小さくできる

**設計としては両方に対応**できるようにし、`jobpack: true/false` のような設定で切り替えるのがよいです。

---

### 4.2 TASK種別（DFTで必要なもの）

NWChemの TASK operation は、energy/gradient/optimize/saddle/hessian/frequencies が提供されます。 ([NWChem][3])
TS最適化は `saddle`、振動数は `frequencies`（`freq`）を使います。

---

### 4.3 DFTの基底（ao basis / cd basis）設計は最重要

#### ao basis

DFTモジュールは **Kohn–Sham軌道の基底が “ao basis” というデフォルト名に入っている必要**がある、と明記されています。 ([NWChem][4])
→ 生成する deck では **`basis "ao basis"` を明示**すると安全です（曖昧さ排除）。

#### cd basis（密度フィッティング）

DFTでは `cd basis` を指定することで Dunlap scheme によるクーロン評価が使え、未指定だと **O(N^4)のexact Coulomb**になると説明されています。 ([NWChem][4])
→ 高スループットでは魅力。

しかし重大な落とし穴があります。

#### “cd basisを使うと解析的ヘシアンが使えない”問題

Hessiansの公式ページで、DFTの解析的ヘシアンは利用可能だが、**“charge fitting with DFT”では解析的ヘシアンが利用できない**と明記されています。 ([NWChem][5])
→ `cd basis` を使うと frequency 計算が有限差分になり、計算量と失敗率が跳ね上がりやすい。

**対策（設計に落とす）**

* DFTフェーズの `LevelOfTheory` を **用途別に2系統**持つ

  * `opt_level`: 速度優先（cd basis OK）
  * `freq_level`: 解析ヘシアン優先（cd basis OFF）
* これを **ジョブタイプ**と結びつける（後述の設定例参照）

---

### 4.4 functional選定と “freqが重くなる”落とし穴

DFTドキュメントの「Minnesota Functionals」一覧にて、**Minnesota系（M06-2X等）は解析二階微分が未サポート**と明記されています。 ([NWChem][4])
→ M06-2Xで `task dft frequencies` をやると、数値ヘシアン寄りになりがちで重い/不安定になり得ます。

**対策（実務で効く）**

* **幾何最適化＋freq**は解析ヘシアンが安定な機能（例: B3LYP, PBE0など）で実施
* **single-point（エネルギーだけ）**を M06-2X 等で行い、
  `E_elec(high) + Gcorr(low)` のような **二層法**を標準機能として持つ

  * これはGaussianワークフローでも一般的で、MIでも計算コストを抑えやすい

---

### 4.5 分散補正（D3 / D3BJ）と対応functional

NWChemのDFT分散補正 `disp vdw 3`（D3）と `vdw 4`（D3BJ）について、対応functional一覧と、**energy gradients と Hessian でもサポート**される旨が記載されています。 ([NWChem][4])
→ TS最適化・freqに分散補正を含めたい場合、**対応functionalを選ぶ**必要があります。

設計対策:

* `dispersion` を `{"model": "vdw4"|"vdw3"|"off"}` として設定可能にし
* `functional × dispersion` の互換性チェックを **プリフライト**で行い

  * 非対応なら「自動で vdw3 へ落とす」or「エラーで止める」を選べるようにする

---

### 4.6 数値積分グリッド（GRID）を “設定として露出” させる

DFTの `GRID` は `xcoarse/coarse/medium/fine/xfine/huge` を持ち、**エネルギー目標精度（1e-6 など）**とともに説明されています。デフォルトは `medium`。 ([NWChem][4])
活性化エネルギーは差分なので、**TSとReactantでグリッド設定がブレると誤差が乗る** → パイプラインでは原則固定。

設計対策:

* `grid_level` を LevelOfTheory に必須項目として含める
* `opt` は `medium/fine`、`sp` は `xfine` など “プロファイル”を持つ
* どのレベルを使ったかを出力メタデータに必ず残す

---

### 4.7 BSSE（カウンターポイズ）を“オプション機能”として入れる

弱い複合体（HF遭遇複合体、前駆体複合体）では **BSSEが無視できない**場合があります。
NWChemのDFTでは counterpoise で **ghost原子を `bq` だけでなく `bqH` `bqO` のように“元素記号付きでラベル”する必要がある**と明記されています。 ([NWChem][4])

設計としては:

* 基本は “BSSEなし” で高速スクリーニング
* 上位候補のみ `dft.bsse_cp` タスクで

  * `E_complex`
  * `E_monomerA(with ghost)`
  * `E_monomerB(with ghost)`
    を自動生成し、補正値を保存
    （NWChemのTASKのBSSE例もありますが、DFT側のghostラベル注意は上記が重要です）

---

## 5. “タスクとして分割”したDFTフェーズの具体タスク設計

あなたの要件「各計算が個別タスクで実行でき、パイプラインでもつながる」を満たす最小構成を提示します。

### 5.1 タスク一覧（DFTフェーズ内）

* `dft.nwchem_minima_optfreq`

  * 入力: minima候補（reactant_complex / product_complex / intermediate）
  * 出力: optimize後構造 + freq（任意）
* `dft.nwchem_ts_saddlefreq`

  * 入力: TS guess
  * 出力: saddle最適化TS + freq（TS判定）
* `dft.nwchem_singlepoint`

  * 入力: 任意構造（最終minima/TS）
  * 出力: 高精度SPエネルギー（必要なら dispersion別）
* `dft.nwchem_bsse_cp`（オプション）

  * 入力: 複合体構造 + monomer定義
  * 出力: CP補正エネルギー一式

> ClearML将来対応の観点では、この粒度がちょうど良いです（1タスク=1計算or計算セット）。
> いまはLocal実行でも、後でClearML Agentへ移行しやすい。

---

## 6. コード構成（アーキテクト視点：拡張しやすい骨格）

### 6.1 ディレクトリ例

```text
src/
  gasrxn/
    core/
      artifacts.py          # 共通Artifact定義（Pydantic）
      hashing.py            # job hash（再計算回避）
      exec.py               # ローカルExecutor（subprocess・並列）
      logging.py
      config.py             # YAML/JSON読み込み

    dft/
      __init__.py
      models.py             # DFTJobPlan, LevelOfTheory, DFTResult
      engine/
        base.py             # DFTEngine interface
        nwchem.py           # NWChem実装
        # future:
        # psi4.py
        # pyscf.py

      nwchem/
        templates/
          minima_optfreq.nw.j2
          ts_saddlefreq.nw.j2
          singlepoint.nw.j2
          bsse_cp.nw.j2
        input_builder.py    # Jinja2でdeck生成
        runner.py           # 実行・stdout保存・リトライ
        parser.py           # ログ解析（必要最小を堅牢に）
        validators.py       # 互換性チェック（disp/functional等）
        presets.py          # “screen/refine/sp” プロファイル

      tasks/
        minima.py
        ts.py
        sp.py
        bsse.py

    cli/
      main.py               # Typer等でCLI化
```

### 6.2 DFTEngineの抽象（将来Psi4/PySCF追加に備える）

```python
# dft/engine/base.py
from abc import ABC, abstractmethod
from gasrxn.dft.models import DFTJobPlan, DFTResult

class DFTEngine(ABC):
    @abstractmethod
    def prepare(self, plan: DFTJobPlan) -> None: ...
    @abstractmethod
    def run(self, plan: DFTJobPlan) -> None: ...
    @abstractmethod
    def parse(self, plan: DFTJobPlan) -> DFTResult: ...
```

NWChemはこのインターフェースを満たすだけ。
Psi4/PySCFは同じインターフェースの別実装として追加できる。

---

## 7. NWChem deck 生成の具体（テンプレート + 設定）

### 7.1 “ジョブフォルダ”規約（再現性と後処理のため）

各計算は以下を作る:

```text
runs/{reaction_id}/{species_role}/{calc_id}/
  input.nw
  stdout.log
  metadata.json        # 入力設定の固定化（ClearML前提）
  result.json          # DFTResult（パース済み）
  scratch/             # NWChem scratch_dir（必要なら）
  perm/                # permanent_dir（必要なら）
```

Getting Startedでも START prefix がファイルprefixになり、scratch/permanent共有でもprefix違いで共存できると説明されています。 ([NWChem][2])
→ `start {calc_id}` を使い、フォルダとprefixが一致するよう統一すると追跡が簡単です。

---

### 7.2 minima（optimize + frequencies）テンプレ例

（例: B3LYP + D3BJ、grid fine、freqは解析ヘシアンを期待するので cd basis は使わない）

```nwchem
start {{ prefix }}
title "{{ title }}"

permanent_dir {{ perm_dir }}
scratch_dir {{ scratch_dir }}

charge {{ charge }}
geometry units angstrom
  symmetry c1
{{ geometry_lines }}
end

basis "ao basis" spherical
{{ ao_basis_lines }}
end

dft
  xc {{ xc_keyword }}
  grid {{ grid_level }}
  disp vdw {{ disp_vdw }}   # vdw 4 -> D3BJ など
  {{ dft_extra_lines }}
end

driver
  maxiter {{ maxiter }}
end

task dft optimize
task dft frequencies
```

根拠:

* `task dft optimize` / `task dft frequencies` はTASK仕様上、最適化・振動数解析の操作として定義されています。 ([NWChem][3])
* `GRID` の精度レベルとデフォルトが示されています。 ([NWChem][4])
* 分散補正 `vdw 4` の対応と、勾配/ヘシアンでサポートされることが示されています。 ([NWChem][4])

---

### 7.3 TS（saddle + frequencies）テンプレ例

```nwchem
start {{ prefix }}
title "{{ title }}"

permanent_dir {{ perm_dir }}
scratch_dir {{ scratch_dir }}

charge {{ charge }}
geometry units angstrom
  symmetry c1
{{ geometry_lines }}
end

basis "ao basis" spherical
{{ ao_basis_lines }}
end

dft
  xc {{ xc_keyword }}
  grid {{ grid_level }}
  disp vdw {{ disp_vdw }}
end

driver
  maxiter {{ maxiter }}
end

task dft saddle
task dft frequencies
```

根拠:

* `saddle` はTS（鞍点）探索操作としてTASKに定義。 ([NWChem][3])

TS判定は後述のQCで必須（虚数振動数が1本か）。

---

### 7.4 single-point（高精度SP）テンプレ例（2層法の上位）

Minnesota系（例: m06-2x）は freq が重くなる可能性があるため、SP専用で使うのが安全です（解析二階微分未サポート）。 ([NWChem][4])

```nwchem
start {{ prefix }}
title "{{ title }}"

permanent_dir {{ perm_dir }}
scratch_dir {{ scratch_dir }}

charge {{ charge }}
geometry units angstrom
  symmetry c1
{{ geometry_lines }}
end

basis "ao basis" spherical
{{ ao_basis_lines }}
end

dft
  xc m06-2x
  grid xfine
  disp vdw 3
end

task dft energy
```

---

## 8. Basisの扱い：NWChem内蔵 + Basis Set Exchange（BSE）で“外れない”実装にする

### 8.1 NWChemのlibrary指定（内蔵ベース）

NWChemは `basis` ブロック内で `library` 指定できます（Getting Startedやサンプルで例示）。 ([NWChem][2])
ただし、あなたが将来いろいろな分子・原子種へ拡張することを考えると、
**「内蔵名に存在しない基底」をどうするか**が必ず問題になります。

### 8.2 BSE（basis_set_exchange）を使う（pipで入る）

Basis Set ExchangeのPython APIでは、`get_basis` で基底を取得でき、**出力フォーマット一覧に `nwchem` が含まれる**ことが明記されています。
→ `pip install basis-set-exchange` で、**def2系やaug-cc系も含めて** NWChem形式文字列を生成できます。

設計（重要）:

* `BasisResolver` を作る

  * まず “NWChem library名で指定” を試し
  * 見つからない or 明示したい場合は BSEから `nwchem` 形式を生成して deck に埋め込む
* 生成した基底文字列は **キャッシュ**（同じ原子種・同じ基底を大量に使う）

---

## 9. 実行（runner）設計：Localで回しつつClearML移行を見据える

### 9.1 コマンド生成

Getting Startedにあるように、入力 `n2.nw` を `nwchem n2` で実行できます。 ([NWChem][2])
→ 実装は “prefix規約”を採用すると安全:

* `input_path = run_dir / f"{prefix}.nw"`
* `cmd = [nwchem_bin, prefix]` もしくは `cmd = [nwchem_bin, input_path.name]`（環境差があるため設定化）

標準出力へ出るので、`stdout.log` にリダイレクトして保存する。 ([NWChem][2])

### 9.2 失敗時のリトライ（SCF収束など）

DFTには `convergence fast` の例があり、`quickguess` などを自動で使う旨が示されています。 ([NWChem][4])
また `convergence nolevelshifting` の記載もあり、特定ケースでレベルシフトを避けたい意図が示されています。 ([NWChem][4])

**設計提案（Runnerに組み込む）**

* `RetryPolicy` を設定で定義

  * 例:
    1回目: 標準
    2回目: `convergence fast`, `maxiter`増, gridを一段粗く
    3回目: 初期guessを変える（必要なら）
* ただし勝手に変えすぎると比較が壊れるので

  * 変更した内容は `result.json` に **必ず記録**（“この結果はfallback条件で得られた”）

---

## 10. パース（parser）設計：cclib/QCEngineを使いつつ“最小限は自前で堅牢に”

### 10.1 まず“最小限”は自前で拾えるようにする

高スループットではログフォーマット微差でパーサが壊れるのが致命傷です。
なので **最低限の値**（最終エネルギー、最終座標、freq一覧、虚数本数、収束成否）は

* 正規表現 + “最後に出たセクションを採用” のような堅牢ルールで自前抽出

### 10.2 cclibは有力だが“対応バージョン注意”

cclibはNWChem outputをサポートしますが、ドキュメント上の対応バージョン記述は古めで、最新NWChemへの完全追従は保証しづらい可能性があります（この点は実ログで検証してから採用が安全）。 ([cclib][6])
→ 設計としては:

* `ParserChain = [CclibParser (optional), RegexParser (fallback)]`
  という“チェーン”にしておくと壊れにくい

### 10.3 QCEngine/QCElementalを“将来の標準I/O”として採用する余地

NWChem公式の “Software supporting NWChem” にも QCEngine（実行とI/O標準化）が挙げられています。 ([NWChem][7])
また QCEngine は “quantum chemistry program executor and IO standardizer (QCSchema)” として提供されています。 ([MOLSSI][8])
→ 今すぐ全面採用しなくても、**内部ArtifactをQCSchema寄りにしておく**と将来Psi4/PySCF移行が楽です。

---

## 11. freq/ヘシアンの実務上の注意（半導体ガス相の温度条件にも直結）

### 11.1 振動数の温度・再利用

NWChemのVIBモジュールは

* ZPEを計算し
* デフォルト温度は **298.15 K**
* `temp` で複数温度指定も可能
* Hessian再利用も可能
  と明記されています。 ([NWChem][9])

半導体装置では温度が298Kとは限らないので、少なくとも

* `freq` データ（振動数そのもの）
* 解析温度（どの温度で熱補正値を作ったか）
  はメタデータに保持し、後段（GoodVibes/Arkane）で温度再計算できる形にします。

### 11.2 解析的ヘシアンが“in-coreで対称性なし”→ メモリ要求に注意

Hessianの公式ページで、解析ヘシアンのアルゴリズムは **fully in-core** で **symmetryを使わない**とされています。 ([NWChem][5])
→ freqを入れる場合は、`memory_mb` を設定で必須にして、失敗時は分割ジョブ（opt→freq）に落とすなどの対策が必要です。

---

## 12. GoodVibes連携（NWChemログが使えるか）

GoodVibesのリリースノートで **“Adding NWChem compatibility”** が明記されています（v3.1.0）。 ([GitHub][10])
→ あなたが「このコード内で熱化学計算まではまだ未定」としても、

* DFTフェーズの成果物として **NWChem stdout.log を必ず保存**しておけば
* 後段タスクで GoodVibes をそのまま回せる可能性が高い

---

## 13. 設定ファイル例（YAML）：二層法・ジョブ種別で切替

```yaml
dft:
  engine: nwchem
  nwchem:
    bin: /usr/bin/nwchem
    default_nproc: 8
    default_memory_mb: 8000
    scratch_root: ./runs/_scratch

  levels:
    geom_freq:            # 最適化+freq用（解析ヘシアン優先）
      xc: b3lyp
      disp: {model: vdw4}
      grid: fine
      basis:
        ao: aug-cc-pvdz
        use_bse_if_missing: true
      use_cd_basis: false   # 解析ヘシアンを守る（重要）

    sp_high:              # 高精度SP（freq不要）
      xc: m06-2x
      disp: {model: vdw3}
      grid: xfine
      basis:
        ao: aug-cc-pvtz
        use_bse_if_missing: true
      use_cd_basis: true    # SPなら高速化OK（任意）

  jobs:
    minima:
      run_opt: true
      run_freq: true
      run_sp_high: true

    ts:
      run_saddle: true
      run_freq: true
      run_sp_high: true

  retry_policy:
    max_attempts: 3
    attempt_overrides:
      2:
        dft_extra: ["convergence fast"]
      3:
        grid: medium
        maxiter: 300
```

---

## 14. DFTフェーズのQC（品質判定）ロジック：結果を“材料探索”で使える形にする

MI用途で重要なのは、計算が終わったかではなく **“比較可能な品質か”**です。

### minima QC

* 最適化収束（NWChemの収束メッセージ/gradient norm）
* freqがあるなら **虚数振動数 0**（小さい負値は閾値で扱う）

### TS QC

* saddle収束
* freqで **虚数振動数が1本**

  * もし 0本 → minima化している可能性
  * 2本以上 → TSとして不適
* 重要: 虚数モードが“反応座標か”は理想的にはIRCで確認
  → ただしIRC主体は PESフェーズなので、DFTフェーズでは「虚数1本」を最低条件にする、という線引きが妥当

---

## 15. まとめ：DFT（NWChem）フェーズの“最終ブラッシュアップ版”指針

* **NWChemはopenで実用的**。導入もapt/brew等で可能（conda不要）。 ([NWChem][1])
* **DFT入力生成はテンプレ化**し、`TASK dft optimize/saddle/frequencies/energy` をジョブタイプで切替。 ([NWChem][3])
* **cd basisは高速化に効くが、freq（解析ヘシアン）と相性が悪い**（charge fittingで解析ヘシアン不可）。 ([NWChem][5])
* **Minnesota系（M06-2X等）は解析二階微分未サポート** → freq用途には避け、SP用途で使う二層法が堅い。 ([NWChem][4])
* GRID/dispersionは **差分エネルギーの比較可能性**に直結するので、設定として必ず固定＆記録する。 ([NWChem][4])
* VIBは温度指定やHessian reuseが可能で、ガス相高温条件の後段解析に繋げやすい。 ([NWChem][9])
* GoodVibesはNWChem互換が入っており、後段タスク化しやすい。 ([GitHub][10])

---

必要なら次のステップとして、あなたの既存フェーズ（pysisyphus）出力を仮定し、
**「pysisyphusのTS guess（xyz）→ NWChem TS saddle+freq → SP high → ΔE‡/ΔG‡」** までを一気通貫で回すための

* `DFTJobPlan` / `DFTResult` のPydanticモデル定義案（完全版）
* NWChemログの“最低限パース正規表現”案（エネルギー・最終座標・freq・虚数判定）
* `RetryPolicy` の実装パターン（変更履歴をresult.jsonに残す）

を、実装にほぼコピペできる形で提示できます。

[1]: https://nwchemgit.github.io/Download.html "https://nwchemgit.github.io/Download.html"
[2]: https://nwchemgit.github.io/Getting-Started.html "https://nwchemgit.github.io/Getting-Started.html"
[3]: https://nwchemgit.github.io/TASK.html "https://nwchemgit.github.io/TASK.html"
[4]: https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html "https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html"
[5]: https://nwchemgit.github.io/Hessians-and-Vibrational-Frequencies.html "https://nwchemgit.github.io/Hessians-and-Vibrational-Frequencies.html"
[6]: https://cclib.github.io/data.html "https://cclib.github.io/data.html"
[7]: https://nwchemgit.github.io/Software-supporting-NWChem.html "https://nwchemgit.github.io/Software-supporting-NWChem.html"
[8]: https://molssi.github.io/QCEngine/ "https://molssi.github.io/QCEngine/"
[9]: https://nwchemgit.github.io/Vibration.html "https://nwchemgit.github.io/Vibration.html"
[10]: https://github.com/patonlab/GoodVibes/releases "https://github.com/patonlab/GoodVibes/releases"
