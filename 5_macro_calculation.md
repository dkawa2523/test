以下では、あなたが提示した全体ワークフロー（RDKit → CREST/xTB → pysisyphus → DFT(NWChem) → 熱化学/速度論 → Cantera、将来ClearML）を維持したまま、**「熱化学/速度論（GoodVibes / Arkane / Cantera）」フェーズ**だけを、アーキテクト + MI（材料探索）観点で“実装できるレベル”まで具体化します。
（注）ここは「このコード内で全シミュレーターを完結させる」よりも、**入力/出力フォーマット整備と、後段（Arkane/Cantera等）へ渡すためのデータ整形を主軸**にしています、という前提に沿います。

---

## 0. まず押さえるべき前提（このフェーズの難所）

### A) GoodVibes は NWChem 対応が入っている（=今の構成と相性が良い）

GoodVibes のリリースノートに **“Adding NWChem compatibility”** が明記されています。([GitHub][1])
→ **DFTをNWChemに固定したまま熱化学（G, H, S）を自動化しやすい**。

### B) Arkane は「対応QCコード」に NWChem が入っていない（重要）

Arkane の README で “compatible … Gaussian, Q‑Chem, Molpro, Orca, Psi4, TeraChem” とあり **NWChemは列挙されていません**。([GitHub][2])
→ そのままでは **ArkaneがNWChem出力をパースするルートが詰まりがち**。
**対策設計が必須**です（後述）。

### C) Cantera への接続は「Arkane→Chemkin→ck2yaml」が最短

Arkane は実行後に **`chem.inp`（Chemkin）を出力**し、そこに thermo/kinetics が入ります。([反応機構生成器][3])
Cantera には Chemkin→YAML 変換ツール **ck2yaml** があり、`--input` は THERMO を含んでいてもよい（＝1ファイルでもいける）と書かれています。([Cantera][4])

---

## 1. このフェーズの「目的」と「成果物」

### 目的（MI/材料探索のために必要な最小セット）

* DFT（NWChem）で得た **E / ZPE / 周波数 / 構造**から

  1. **ΔE‡ / ΔH‡ / ΔG‡**（温度依存）
  2. **k(T)**（TSTベース、必要ならトンネリング補正）
  3. （可能なら）**Arrhenius/modified Arrhenius へのフィット**
  4. さらに必要なら **圧力依存 k(T,P)**（Arkaneのmaster equation）
* それらを **材料探索で使えるテーブル**に落とす
  （候補分子ごと、反応タイプごと、温度点ごと、TSごと）

### 成果物（パイプラインのアーティファクト）

最低限：

* `thermo_summary.json`（speciesごとの E, H(T), G(T) など）
* `barrier_summary.json`（reactionごとの ΔG‡(T), ΔE‡, ΔH‡）
* `rates_table.parquet`（reaction×T の k(T), 単位明示）
* `arrhenius_fit.json`（A,n,Ea や PLOG/Chebyshev の係数）

任意（Arkane/Canteraを使う場合）：

* `arkane/input.py` + `species/*.py` + `TS/*.py`
* `arkane/output.py`, `arkane/chem.inp`, `arkane/Arkane.log`([反応機構生成器][3])
* `cantera/mechanism.yaml`（ck2yaml で生成）([Cantera][4])

---

## 2. タスク分割（個別実行可能＋パイプライン化）

このフェーズは「熱化学（thermo）」と「速度論（kinetics）」と「機構出力（export）」に分けます。

### タスク一覧（推奨）

1. **`thermo.collect_qc_outputs`**

   * DFT（NWChem）出力の場所を集約（species/TS/complex/conformer）
   * 必要なファイルのみ抽出し、命名規則を統一（GoodVibesや後段用）

2. **`thermo.goodvibes.prepare`**

   * GoodVibes 用に

     * 入力ファイルのシンボリックリンク/コピー
     * **PES YAML（`--pes` 用）**を自動生成
   * conformer 群を GoodVibes が認識できるように整理

3. **`thermo.goodvibes.run`**（任意：外部実行でもOK）

   * `python -m goodvibes ... --csv --pes ...` で実行しCSVを得る
   * GoodVibes は `--pes` で PES 比較や accessible conformer 補正を実施可能([GoodVibes][5])

4. **`thermo.goodvibes.parse`**

   * CSV/標準出力から

     * species の G(T), qh-G(T)
     * TS の imaginary frequency 等（使える範囲で）
     * ΔG‡(T), ΔE‡ 等を計算して整形

5. **`kinetics.tst.compute`**

   * ΔG‡(T) から **Eyring/TSTで k(T)** を計算
   * 反応次数（uni/bi）で標準状態と単位変換を厳密に扱う（重要）

6. **`kinetics.fit.arrhenius`**

   * k(T) を温度範囲でフィットし

     * Arrhenius
     * modified Arrhenius（Arkane形式）
   * Canteraに入れやすい形へ

7. **`arkane.export_input`**（必要時のみ）

   * Arkane input 生成
   * **NWChem非対応問題を吸収**する設計（後述）

8. **`arkane.run`**（Docker 実行）

   * Arkane は `python Arkane.py INPUTFILE` で実行([反応機構生成器][6])
   * RMG は Docker が推奨インストール([反応機構生成器][7])

9. **`arkane.parse_outputs`**

   * `output.py`, `chem.inp` を回収([反応機構生成器][3])
   * 圧力依存をやったなら PLOG/Chebyshev を抽出

10. **`cantera.export_yaml`**

* Arkaneの `chem.inp` を Cantera YAML に変換
* `ck2yaml --input=chem.inp ...`（`input` にTHERMOが含まれてもOK）([Cantera][4])

11. **`cantera.validate`**（任意）

* `ct.Solution(mechanism.yaml)` で読めるか検証
* 最小の0Dリアクタで “メカとして破綻してないか” を確認

---

## 3. ライブラリ/インストール方針（conda禁止前提）

### 必須（pipでOK）

* GoodVibes: `pip install goodvibes`([GitHub][8])
* Cantera: pipインストールが公式に用意されている([Cantera][9])
* 変換・整形用：`pydantic`, `ruamel.yaml`（YAML生成）, `pandas`, `numpy`, `scipy`
* 量子出力の汎用パース補助（推奨）：`cclib`（NWChem含む多コード対応のログパーサとして一般的）([AIP Publishing][10])
  ※GoodVibesや自作パーサの“保険”として価値が高い

### Arkane（RMG-Py）

* condaが使えないなら、**Dockerが公式推奨**([反応機構生成器][7])
* 実行コマンドも文書化されている（`python Arkane.py INPUTFILE`）([反応機構生成器][6])

---

## 4. GoodVibes 層を「中核」にする設計（NWChem固定でも回る）

### 4.1 GoodVibes がこのフェーズで担う役割

* RRHO（＋quasi-harmonic）で **H(T), S(T), G(T)** を高速に計算
* 低周波数の扱い改善（qRRHO）

  * `-q` quasi-harmonic ON（Grimme/Head‑Gordon など）([GoodVibes][5])
  * cutoff `-f` 指定（デフォルト100 cm⁻¹ 等）([GoodVibes][5])
* 温度指定 `-t`、温度範囲 `--ti` ([GoodVibes][5])
* 反応座標（PES）比較を **YAMLで定義**して処理できる（`--pes`）([GoodVibes][5])
* conformer群がある場合、**accessible conformer correction**（Boltzmann重みの補正）を自動で入れられる([GoodVibes][5])

> あなたのパイプラインは CREST で大量の配座を出すので、GoodVibes の `--pes` + conformer補正は非常に相性が良いです。

### 4.2 GoodVibesのCLI設計（=あなたのタスクに落とすべき仕様）

GoodVibesの使用例・オプションはドキュメントにまとまっています。([GoodVibes][5])
特にこのフェーズで使うべきは以下：

* 温度点（例：装置温度 300–1200K を想定）

  * 単一点：`-t 800`
  * 温度スイープ：`--ti '300,1200,50'` ([GoodVibes][5])

* qRRHO（低周波補正）

  * `-q` + `--qs grimme/truhlar` + `--qh` + `-f <cutoff>` ([GoodVibes][5])

* PES比較（あなたの “反応ごと” の取りまとめに直結）

  * `--pes <pes.yaml>` ([GoodVibes][5])

* 出力の機械可読化

  * `--csv`（CSV出力）([GoodVibes][5])

### 4.3 PES YAML を「あなたのデータモデル」から自動生成する

GoodVibesのPES YAMLは例が示されており、`# PES` と `# SPECIES` ブロックを持ちます。([GoodVibes][5])
ここを自動生成するのが `thermo.goodvibes.prepare` の主タスクです。

#### あなたのパイプライン側の“論理種”の考え方（推奨）

* `SpeciesKey`（論理種）

  * 例：`amine_A`、`HF`、`encounter_complex_A_HF`、`TS1`、`product1` …
* `Conformer`（物理インスタンス）

  * 例：`encounter_complex_A_HF_conf001`, `conf002`, …

GoodVibes側には

* `enc_AHF*` のようなワイルドカードで conformer 群を食わせる
  （例の `Int_I_Ph*` のように `*` を使う）([GoodVibes][5])

#### 実装イメージ（prepareタスク）

* DFT出力ファイルを `work/<rxn_id>/thermo/goodvibes/` に集約
* GoodVibesが期待する命名に揃える（後述）
* `pes.yaml` を生成

  * `# SPECIES` に論理種→ファイルprefix を割り当て
  * `# PES` に反応経路（R + HF → TS → P 等）を定義

> **重要**：PES YAML とファイル名の対応が崩れると GoodVibes が見つけられません。
> ドキュメントにも “PES & Graph は `.yaml` の `# SPECIES` とファイル名が対応する必要” が書かれています。([GoodVibes][5])

### 4.4 命名規則（GoodVibes連携のためのローカル規約）

おすすめ（例）：

* `SPC`（高精度 single point）を後で混ぜる可能性もあるので、同一root + suffix 方式を採用

  * `TS1.out`（freq/opt）
  * `TS1_TZ.out`（高基底SP）
    GoodVibesは “root + _descriptor” を `--spc TZ` で拾える、と説明があります。([GoodVibes][5])

---

## 5. 速度論（TST/Eyring）を “自前で確実に” 作る（Arkane無しでも回る）

Arkaneは強力ですが、NWChemとの互換問題があるので、**まずは GoodVibes → k(T) 算出 → Arrhenius fit → Cantera投入**までを最短で回せる設計が堅いです。

### 5.1 どの量を k(T) に使うか

* screening段階では **ΔG‡(T)** が最も実用的

  * エッチング装置は温度が効く
  * ΔE‡だけだと温度・エントロピー効果が落ちる

GoodVibesは `G(T)` と qRRHO補正 `qh-G(T)` を出すので、それを採用するのが合理的です。([GoodVibes][5])

### 5.2 標準状態（ここは“半導体ガスプロセス”で致命的になり得る）

* GoodVibesは “1 atm ↔ 1 mol/L” の標準状態変換について説明しており、`-c` で濃度を与える設計です。([GoodVibes][5])
* 一方、**ガスエッチング装置は低圧（mTorr〜Torr）**が多く、

  * “標準状態” と “実運転圧力” を混同すると k(T) が桁でズレます。

#### 推奨アーキ設計

* このコードでは

  1. **標準状態での ΔG‡°(T)**（例えば 1 bar or 1 atm）をまず固定で算出
  2. 実機条件（P, 流量, 混合比）での反応速度は **Cantera側で濃度を与えて評価**
  3. どうしても ΔG(T,P) が欲しい場合は別タスクで “理想気体の μ補正” を適用

という分離にすると事故りにくいです。

### 5.3 `kinetics.tst.compute` タスクの入出力

**入力**

* `barrier_summary.json`（ΔG‡(T) が温度点ごとにある）
* 反応の分子数変化（Δν）
* 反応次数（uni/bi）と標準状態定義（C° or P°）

**出力**

* `rates_table.parquet`：

  * columns 例：`reaction_id, T, k, units, method=tst, kappa(wigner)=...`
* `arrhenius_fit.json`（次タスクで生成してもOK）

### 5.4 `kinetics.fit.arrhenius`

Arkaneは「modified Arrhenius（3パラ）」をデフォルトで出す、と記載されています。([反応機構生成器][11])
あなた側も同形式で出しておくと、Arkaneに切替えてもデータ形式が揃います。

---

## 6. Arkane をどう組み込むか（NWChem問題への“設計上の吸収”）

### 6.1 Arkaneを使うべき場面（半導体ガスプロセス視点）

* 反応が **多井戸PES**（複合体→中間体→…）になり、
  **低圧で falloff / chemically activated** が効く
* “遭遇複合体（encounter complex）” を経由して

  * 生成物に落ちる
  * 元に戻る
  * 別経路へ進む
    などが競合する

この領域は Arkane の **pressureDependence**（master equation）が刺さります。
Arkane入力では

* `network()` と `pressureDependence()` が必要で、
* `pressureDependence()` の `method`（MSC/RS/CSE）や `interpolationModel`（chebyshev / pdeparrhenius(plog)）を指定できます。([反応機構生成器][11])

### 6.2 Arkane入力の骨格（生成対象）

Arkane入力ファイルは、構成要素がドキュメントに列挙されています（modelChemistry, atomEnergies, frequencyScaleFactor, species, transitionState, reaction …）。([反応機構生成器][11])

反応定義では `tunneling='Eckart'/'Wigner'` を指定可能。([反応機構生成器][11])

### 6.3 “NWChem非対応”への対策（設計パターン）

#### 対策パターンA：Arkaneは**オプション**、まずは GoodVibes+自前TST で完走

* 多くの材料探索では
  “相対比較ができる barrier / k(T)” がまず必要
* pressure dependence が必要な候補だけ Arkane に回す
  → **計算コスト/実装コストを制御**できる

#### 対策パターンB：Arkane入力を Option #2（分子物性を直接記述）で生成

Arkaneは species/TS を “量子出力ファイルをパース” 以外に
“分子物性を直接入力” できる、と書かれています。([反応機構生成器][11])
つまりあなたのコード側で

* 質量
* 慣性モーメント
* 振動数（+内部回転の近似）
* E0
  を組み立てて `transitionState()` / `species()` を書く。

**現実的な落とし所（初期実装）**

* screeningでは内部回転を全て HO（HarmonicOscillator）近似
* 後で必要になったら pysisyphus/DFT層で回転障壁スキャンを追加し、Arkaneの hindered rotor に拡張

#### 対策パターンC：Arkane対象だけ Psi4/Orca などに差し替え（将来）

Arkaneは対応QCとして Psi4/Orca 等を公式に挙げています。([GitHub][2])
→ 将来的に DFT層が Psi4 を追加できる設計なら、このルートも自然。

---

## 7. Arkane実行・出力・Cantera接続（実務上の流れ）

### 7.1 Arkane実行

Arkaneドキュメントでは実行方法が明記されています：
`python Arkane.py INPUTFILE`（出力先は同ディレクトリ、`-o`で変更可能）([反応機構生成器][6])

### 7.2 Arkaneインストール（conda無しならDocker）

RMGは Docker が推奨インストールです（`docker pull reactionmechanismgenerator/rmg:3.3.0` 等）。([反応機構生成器][7])
→ `arkane.run` タスクは **DockerRunner** を呼ぶだけにしておくと、ローカルでも将来ClearMLでも同じ。

### 7.3 Arkane出力

Arkane出力として

* `output.py`（thermo/kinetics/pdep結果が input に似た構文で出る）
* `chem.inp`（Chemkin、thermo/kinetics含む）
* `Arkane.log`
  が生成されると説明されています。([反応機構生成器][3])

### 7.4 Canteraへの変換（ck2yaml）

Canteraには `ck2yaml` があり、`--input` は thermo を含んでいてもよいと記載されています。([Cantera][4])
よって最短は：

```bash
python -m cantera.ck2yaml --input chem.inp --output mechanism.yaml --permissive
```

（`--permissive` は重複等の“回復可能エラー”を無視できる）([Cantera][4])

---

## 8. コード構成（アーキテクト観点での具体案）

### 8.1 ディレクトリ構成（このフェーズ部分だけ）

例：`src/gasrxn/` 配下

```
gasrxn/
  core/
    task.py                  # Task抽象、入出力、キャッシュ、ログ
    artifact_store.py         # ローカルartifact管理（将来ClearMLに置換）
    units.py                  # 単位・変換
    models/
      thermo.py               # SpeciesThermo, Barrier, ThermoGrid...
      kinetics.py             # RateExpression, ArrheniusFit, Plog...
  thermo_kinetics/
    goodvibes/
      prepare.py              # PES YAML生成、命名規則、symlink作成
      runner.py               # subprocessラッパ、stdout/csv回収
      parser.py               # csv/stdoutから構造化
    tst/
      eyring.py               # k(T)算出、標準状態、単位、Wigner補正など
      fit.py                  # Arrhenius/modified Arrhenius fitting
    arkane/
      inputgen.py             # Arkane input.py / species.py / TS.py 生成
      docker_runner.py        # RMG docker で Arkane 実行
      parser.py               # output.py/chem.inp 解析・抽出
    cantera/
      ck2yaml.py              # ck2yaml呼び出し
      validate.py             # Solutionロード検証
      exporters.py            # mechanism.yaml + metadata 出力
  cli/
    thermo.py                 # `gasrxn thermo ...`
    kinetics.py               # `gasrxn kinetics ...`
    export.py                 # `gasrxn export ...`
```

### 8.2 “エンジン差し替え可能”にするための抽象

* `ThermoEngine`（GoodVibes/Arkane/将来別）

  * `prepare(inputs) -> PreparedRun`
  * `run(prepared) -> RawOutputs`
  * `parse(raw) -> ThermoResults`
* `KineticsEngine`（TST/Arkane/将来ML）
* `MechanismExporter`（Cantera YAML / RMG library / JSON）

これにより

* GoodVibesだけ先に使う
* Arkaneは後から有効化
* Cantera exportだけ先に整える
  が容易になります。

### 8.3 データモデル（Pydantic推奨）

材料探索に使いやすいように “テーブル化できる最小単位” を設計します。

* `SpeciesThermoRecord`

  * `species_id`
  * `charge`, `multiplicity`
  * `method_tag`（例：`nwchem/b97-3c` など）
  * `T_grid: list[float]`
  * `G: list[float]`（kJ/mol or Hartree、どちらか統一）
  * `H, S` も必要なら
  * `conformer_ensemble`（Boltzmann平均/最安/採用個数）
  * `source_files[]`（再現性のための参照）

* `BarrierRecord`

  * `reaction_id`
  * `reactant_ids[]`, `product_ids[]`, `ts_id`
  * `dG_dagger(T)`、`dH_dagger(T)`、`dE_dagger`
  * `standard_state`（1 bar, 1 atm, 1 M など）
  * `qh_settings`（cutoff, method）

* `RateRecord`

  * `reaction_id`
  * `T`
  * `k`
  * `units`
  * `model`（tst / arkane / plog / chebyshev）
  * `tunneling`（none/wigner/eckart）

---

## 9. “半導体ガスエッチングスケールで適切か？”の評価（このフェーズ）

### GoodVibes

* 低周波補正（qRRHO）を簡単に入れられるのが強い([GoodVibes][5])
* CREST由来の多配座を扱うのに `--pes` が便利（conformer補正）([GoodVibes][5])
* ただし **圧力依存の速度論**は直接は扱わない
  → ここは Cantera or Arkane に委譲、という分割が合理的

### Arkane

* 圧力依存（master equation）まで行けるのが強い([反応機構生成器][11])
* しかし **NWChemが互換対象に入っていない**([GitHub][2])
  → “必要になった時だけ使う”＋“入力生成で吸収”が現実的

### Cantera

* Arrhenius/PLOG/Chebyshev/falloff など反応速度表現が豊富([Cantera][12])
* 0D/1Dで装置条件の検討ができ、材料探索の“評価器”にできる
* 変換の起点は YAML or ck2yaml で整理できる([Cantera][4])

---

## 10. このフェーズの「実装優先順位」おすすめ

1. **GoodVibes → ΔG‡(T) テーブル化（最重要）**
2. **TSTで k(T) を自前計算 → Arrhenius fit**
3. **Cantera用mechanism.yaml（最初は gas-phase only）**
4. 必要な反応だけ **Arkane（Docker）でpdep**
5. Arkane結果を ck2yaml → Cantera に統合

---



[1]: https://github.com/patonlab/GoodVibes/releases?utm_source=chatgpt.com "Releases · patonlab/GoodVibes"
[2]: https://raw.githubusercontent.com/ReactionMechanismGenerator/RMG-Py/main/arkane/README.md "raw.githubusercontent.com"
[3]: https://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/output.html?utm_source=chatgpt.com "6. Parsing Output Files — RMG-Py 3.3.0 Documentation"
[4]: https://cantera.org/dev/yaml/ck2yaml.html "Chemkin to YAML conversion — Cantera 4.0.0a1 documentation"
[5]: https://goodvibespy.readthedocs.io/en/latest/source/README.html "Introduction — GoodVibes 3.1.0 documentation"
[6]: https://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/running.html?utm_source=chatgpt.com "5. Running Arkane — RMG-Py 3.3.0 Documentation"
[7]: https://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/index.html?utm_source=chatgpt.com "3. Installation — RMG-Py 3.3.0 Documentation"
[8]: https://github.com/patonlab/GoodVibes?utm_source=chatgpt.com "patonlab/GoodVibes: Calculate quasi-harmonic free ..."
[9]: https://cantera.org/3.1/install/pip.html?utm_source=chatgpt.com "Installing with Pip — Cantera 3.1.0 documentation"
[10]: https://pubs.aip.org/aip/jcp/article/161/4/042501/3304757/cclib-2-0-An-updated-architecture-for?utm_source=chatgpt.com "cclib 2.0: An updated architecture for interoperable ..."
[11]: https://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/input.html "3. Creating Input Files — RMG-Py 3.3.0 Documentation"
[12]: https://cantera.org/3.2/reference/kinetics/reaction-rates.html?utm_source=chatgpt.com "Reaction Rates — Cantera 3.2.0 documentation"
