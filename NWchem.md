

以下は「**各 SDF ファイルに 1 分子（1構造）**が入っている」前提で、
**いまは（アミン × HF）**、将来は **（アミン以外 × HF）**や **（アミン × HF以外のガス分子）**、さらに **（複数の反応物/副反応候補）**まで拡張できるように、**全体設計（アーキテクチャ）**と **詳細機能（モジュール仕様）**を“壊れにくい・運用しやすい”視点でブラッシュアップした提案です。

前回の有益な骨格（SDF→構造標準化→錯体生成→経路探索/TS→DFT精密化→ΔG‡→（必要なら）反応器）は基本維持し、**拡張点・データ品質・例外処理**を強化します。

---

## 1. 設計ゴール（将来拡張を見据えた要件定義）

### 1.1 目的

* 入力：

  * **候補分子群**（SDF/1分子/1構造）
  * **反応相手ガス分子群**（SDF または SMILES/組成/電荷・スピンなどの必要情報）
  * **反応クラス**（例：プロトン移動、付加、H引き抜き、会合/解離…）
  * 条件（T, P、必要なら滞留時間、混合比、反応器モデルなど）
* 出力：

  * 反応ごとの **ΔE‡/ΔG‡（錯体基準・分離基準の両方）**
  * TS/IRC 検証済みかどうかの **QCフラグ**
  * （任意）速度定数 **k(T)** / **k(T,P)** と、0D/1D 反応器での消費推定

### 1.2 設計思想

* **“反応クラス”をプラグイン化**して、分子がアミンでなくても HFでなくても回る
* PubChem由来SDFの“ゆらぎ”（塩・混合物、立体未定義、2Dしかない等）を **入力ゲートで吸収**し、後段の計算を安定化
* **多忠実度（xTB→DFT）**を前提に、

  * 低レベルは「候補生成・失敗分類・粗スクリーニング」
  * 高レベルは「最終値確定」
    と役割分担
* **失敗も資産化**（MI/運用改善のために必須）

---

## 2. 入力仕様（SDF 1分子/1構造を中核にした拡張可能な形）

### 2.1 基本入力（最小）

* `candidates/`：候補分子 SDF（各ファイルに1分子）
* `reactants/`：反応相手ガス分子

  * SDF でもよい（各ファイル1分子）
  * もしくは `reactants.yaml` に SMILES / 電荷 / スピン（多重度）を記述

### 2.2 メタデータ（推奨：将来の拡張に効く）

SDFそのものは「座標と結合」中心なので、将来のHF以外・アミン以外・ラジカル種などに備えて、**分子ごとの sidecar metadata（YAML/JSON）**を持てる設計が便利です。

例：`candidates/ABC123.sdf` と `candidates/ABC123.meta.yaml`

* `source`: PubChem / 社内 / 推定生成 など
* `pubchem_cid`: （あれば）
* `charge`: 0 / +1 / -1 …
* `multiplicity`: 1/2/3…（ラジカル等のため）
* `preferred_states`: tautomer/protomer の指定（任意）

※xTB docking は `.CHRG` と `.UHF`（分子全体と各分子の電荷・不対電子数）を読める仕様があり、**開殻・イオンを将来扱うときの基礎**になります。([XTB Docs][1])

---

## 3. 全体アーキテクチャ（層構造）

大きく 4 層に分けると破綻しにくいです。

### Layer A：データ層（正規化・来歴・DB）

* Molecule Registry（入力SDFの登録、ID採番、ハッシュで重複排除）
* Transform Log（標準化や修正の履歴を全て保存）
* Result Store（計算結果、TS/IRCログ、QCフラグ）

※スケールするなら QCArchive（QCFractal/QCPortal/QCFractalCompute）で「大量計算＋DB」を中核にするのが堅牢です。([QCArchive][2])

### Layer B：化学前処理層（PubChem由来SDFのゆらぎ吸収）

* 構造読み込み・サニタイズ
* “必要なら”構造修正（desalt、立体、H付与、3D化 etc）
* 状態列挙（必要なら tautomer/protomer、立体異性体）

### Layer C：反応生成・PES探索層（反応クラスプラグイン）

* 反応サイト抽出
* 錯体生成（複合体コンフォーマ）
* 生成物推定（反応テンプレ or 探索）
* 経路探索（NEB/GSM/scan → TS refine → IRC）

pysisyphus は NEB・Growing String などの Chain-of-States、TS探索、IRC を備え、YAML入力で使える旨が明記されています。([Pysisyphus][3])

### Layer D：物性化・速度論層（ΔG‡/k(T,P)/反応器）

* 熱化学補正（GoodVibes）
* 圧力依存（必要なら Arkane）
* 0D/1D反応器（Cantera）

GoodVibesは低周波振動に対するRRHOの問題を補正する目的で使われていること、また温度・圧力/濃度条件の設定ができることが説明されています。([GitHub][4])
Cantera の 0D 理想気体リアクタ（IdealGasReactor等）もドキュメント化されています。([Cantera][5])

---

## 4. PubChem由来SDFを“計算入力”として使うときの問題点と修正方針

PubChem由来SDFは非常に有用ですが、**量子化学（特にTS探索）にそのまま入れると壊れやすい**典型パターンがあります。ここを“入力ゲート”で吸収します。

### 4.1 3D構造が無い/不十分（2D座標のみ）

PubChemは 3D 構造（コンフォーマ）を提供できますが、**全化合物にあるわけではありません**。3Dモデル生成には「大きすぎない」「柔らかすぎない」「塩/混合物でない」など複数条件があり、条件を満たさない化合物は 3D ダウンロード対象から外れることがある、と説明されています。([PMC][6])
また、PubChemの3D構造は**実験構造ではなく計算生成**であることも明記されています。([PMC][6])

**対策（仕様化）**

* 入力SDFが 2D/3Dどちらでも、必ず

  1. **H付与**
  2. RDKit埋め込み（ETKDG）で3D化（必要時）
  3. xTB/CREST で**再最適化**（全件）
     を行い、「以降の計算は自前で整えた3D」を正とする
* PubChem 3D でも “そのまま信じない”
  → 初期構造として利用しつつ、同じ低レベル（xTB）で整合させる

### 4.2 塩・混合物・多成分（SDF 1分子でも“切断されてる”）

PubChemの3Dコンフォーマ生成条件には「単一の共有結合ユニット（塩や混合物ではない）」が含まれています。([PMC][6])
つまり、PubChem由来のSDFには**塩（カウンターイオン含む）**などが現実に混ざり得ます。

**対策（デフォルト挙動＋例外扱い）**

* デフォルト：**最大フラグメントを主成分（parent）として採用**し、他成分はログに残して除外
* 例外：反応器内で「塩として」評価したい場合は、明示フラグで全成分を保持する分岐も用意

RDKitの `rdMolStandardize` には FragmentParent/Uncharger/Reionizer 等を使った標準化の枠組みがあり、**desalt/中和/正規化などの部品が揃っている**ことが示されています。([GitHub][7])

### 4.3 立体化学が未定義（stereocenter未指定）

PubChemの3Dモデル生成条件には「未定義の立体中心が少ない」旨が含まれます。([PMC][6])
将来の候補分子が増えると、立体未定義が混ざる可能性が上がります。

**対策**

* 入力ゲートで

  * 立体が未定義 → “未定義”タグ付け
  * ルール：

    * 重要候補のみ立体異性体を列挙（上限数を設定）
    * それ以外は「未定義のまま」代表1種で走らせ、結果に不確かさフラグを付ける
* “必ず列挙”ではなく **計算資源と目的に応じて切り替え**可能にする

### 4.4 プロトン化状態/互変異性（tautomer/protomer）のズレ

PubChemの表現（中性形など）が、ガス相条件での主要状態と一致しない場合があります（特に強塩基/強酸、双性イオンなど）。

**対策（拡張可能な設計）**

* デフォルトは「SDFの状態を尊重」しつつ、
* 反応クラスが要求する場合のみ（例：プロトン移動、付加反応でプロトンが鍵）

  * tautomer/protomer を**限定列挙**して “最も起こりやすい経路” を拾う
* 列挙で爆発しないように、**優先順位ルール**（反応中心近傍のみ列挙、上限数、エネルギー窓）を仕様化

RDKit標準化系は正規化・再イオン化・中和などの処理を組めることが示されています（例：Normalize/Reionize/Uncharger等）。([GitHub][7])

---

## 5. 反応クラス（テンプレート）プラグイン設計

将来「アミン以外」「HF以外」に広げるには、反応を **“反応クラス”**として抽象化し、以下のI/Fを持たせるのが強いです。

### 5.1 反応クラスが提供すべき機能（インターフェース）

各 ReactionClass（例：ProtonTransfer、NucleophilicAddition…）は少なくとも：

1. **反応サイト検出**

* `find_sites(molecule)`：SMARTSや部分構造規則で候補サイト一覧
* 例：酸（H供与）/塩基（受容）/求核/求電子/ラジカル中心 など

2. **錯体構築ルール**（Directed complex building）

* `build_complexes(reactantA, reactantB, siteA, siteB)`
* “どの原子同士を近づけるべきか”を反応クラスが知っていることが重要

3. **生成物推定**（テンプレ駆動 or 探索）

* `build_products(...)`：結合変換（bond order/charge）を反映して product を作る
* productが曖昧なクラスは「探索モード」へフォールバック

4. **経路探索戦略**（推奨アルゴリズムを返す）

* `suggest_path_method()`：scan/NEB/GSM/単端TS探索 など
* `reaction_coordinate()`：scan用の座標定義（例：距離差）

5. **TS妥当性判定**

* `validate_ts(ts, irc_endpoints)`：虚数モードの性質、接続先の正しさ等

---

### 5.2 具体例：現行（アミン × HF）の “ProtonTransfer” を一般化する

* サイト検出：

  * Donor側（HFなど）：H–X（X=F, Cl, O, N…）
  * Acceptor側：塩基性ヘテロ原子（N,O,S…）
* 錯体構築：H（donor）を acceptor に向けて配置
* 生成物推定：Hを acceptor に移し、X側はアニオン化（必要なら接触イオン対として初期配置）
* 経路探索：

  * scan座標 `s = r(X–H) - r(H–A)` を第一候補
  * うまくいかなければ NEB/GSM
* バリアレス判定：scanが単調なら “TS無し” として分類（ΔG‡_intra ≈ 0扱い＋QCフラグ）

---

### 5.3 将来追加しやすい反応クラス例（ガス相・半導体プロセス想定）

* **Association/Dissociation（会合/解離）**：吸着ではなく気相会合（錯体形成自由エネルギー）
* **Halogen acid proton transfer**：HF/HCl/HBr… × 塩基一般
* **Nucleophilic substitution / addition**：ハロゲン化物・シラン系・カルボニル系（候補次第）
* **H-abstraction（引き抜き）**：ラジカル/反応性ガスが対象になった場合
* **Ligand exchange**：Si–F/Si–Cl などの交換反応（将来の化学系次第）

---

## 6. 錯体生成（Complex Builder）を “汎用化”する設計

### 6.1 基本方針

錯体生成は「HF特化」から脱却し、**どの2分子でも**回るようにします。

* **一般モード**：xTB docking (aISS) を使用
  aISS は「任意分子を dimers/aggregates に追加でき、interaction site screening と xTB‑IFF の遺伝的最適化 → GFN最適化」という流れで、デフォルトで上位15構造最適化、NCI ensembleも可能と説明されています。([XTB Docs][1])
* **Directed モード**：反応クラスが指定した原子集合へ誘導
  xTB docking の “directed docking” 入力例（atoms/elements、scaling factor）が示されています。([XTB Docs][1])

### 6.2 開殻・イオンへの備え

将来 HF以外（例えばラジカル種）が入ると、電荷・スピンが必須になります。
xTB docking は `.CHRG` と `.UHF` を読んで分子1/2の電荷・不対電子数を扱えるので、ここを標準I/Fにしておくのが得策です。([XTB Docs][1])

---

## 7. 経路探索・TS・DFT精密化：汎用設計（“反応が増えても壊れにくい”）

### 7.1 低忠実度（xTB）段階：**探索の母集団を作る**

* 複合体コンフォーマ（複数） × 生成物コンフォーマ（複数）を作り、
  NEB/GSM/scan を回して TS候補を複数出す
* pysisyphus を「共通の経路探索ランナー」とする
  （NEB/Growing String/IRCがあること、YAML/ライブラリ両対応は運用面で強い）([Pysisyphus][3])

### 7.2 高忠実度（DFT）段階：**上位候補だけ確定**

将来 HF以外の相手でも、基本は同じです。

* NWChem の TASK には `saddle`（遷移状態/鞍点探索）があり、`task dft saddle` のように使えることが明記されています。([NWChem][8])
* さらに NWChem は BSSE（counterpoise）計算の仕組みも TASK の説明内にあり、錯体の取り扱いで必要になったときに“後付け”できます。([NWChem][8])

**設計上のポイント**

* DFT精密化は「全件」ではなく

  * xTBで成功した上位
  * バリアレス疑いだが重要なもの
  * QCフラグが怪しいもの
    に限定できるよう、ワークフローに **選別ステージ**を置く

---

## 8. 熱化学・速度論の汎用化（ΔG‡とkの扱い）

### 8.1 ΔG‡は“定義を固定”して二本立て出力

将来反応が増えるほど、会合の有無で ΔG‡ の意味が揺れます。したがって：

* **錯体基準**：ΔG‡(TS − complex)
* **分離基準**：ΔG‡(TS − (A+B))

を必ず出し、さらに会合自由エネルギー ΔG_assoc も別出力にするのが堅牢です。

### 8.2 GoodVibesは“条件補正・低周波補正の標準ツール”

GoodVibesは低周波振動のRRHO問題を補正する目的で使われ、温度・圧力/濃度条件も指定できる旨が説明されています。([GitHub][4])
→ 反応相手がHFでなくても、会合が絡むガス相反応全般で効きます。

### 8.3 反応器モデルは Cantera を共通出口にする

0D理想気体リアクタ等が整備されており、計算した速度定数を“プロセス条件”に落とす出口として汎用です。([Cantera][5])

---

## 9. 具体的な機能一覧（MVP→拡張まで）

ここは「実装時にそのままチケットに切れる粒度」で書きます。

### 9.1 Molecule Ingestion

* SDF読み込み（1ファイル=1分子）
* RDKitサニタイズ
* 分子ID生成：InChIKey/SMILESハッシュ
* メタデータ結合（meta.yamlのcharge/multiplicityなど）
* 入力QC：

  * 多フラグメント検出（塩/混合物）
  * 立体未定義数
  * 元素種

### 9.2 Standardization（PubChem由来修正を含む）

* `raw_mol` を保存（オリジナル）
* `std_mol` を生成（標準化した作業用）

  * FragmentParent（主成分化）
  * Normalize/Reionize/Uncharger 等（反応クラスに応じてON/OFF）([GitHub][7])
* 修正ログを必ず残す（MI資産化）

### 9.3 3D/Conformer Manager

* 3Dが無ければ埋め込み（RDKit）
* xTB/CRESTで再最適化（“計算用3D”の統一）
* コンフォーマ集合管理（エネルギー窓、RMSDクラスタ）

### 9.4 Reactant Library

* 反応相手分子を SDF/SMILES で登録
* 反応相手ごとに「代表状態」（charge/multiplicity）を指定可能
* HFなど頻出種はテンプレを用意（ただし将来拡張は同じI/F）

### 9.5 Reaction Enumeration

* ReactionClassプラグインの登録/選択
* 候補×相手×サイトの組み合わせ生成
* 計算資源に応じた組み合わせ削減（上限、優先順位）

### 9.6 Complex Builder

* 一般：xTB docking/aISS（NCI ensemble）([XTB Docs][1])
* Directed：反応クラスが指定した atoms/elements で誘導([XTB Docs][1])
* 生成物初期構造の自動生成（反応クラスごと）

### 9.7 Path Search / TS / IRC Runner

* pysisyphus で NEB/GSM/scan を回す([Pysisyphus][3])
* TS refine
* IRC（重要ケースのみでも可）
* 失敗分類（NEB発散、TS虚数>1、IRC不一致…）

### 9.8 DFT Refiner

* NWChem `task dft saddle` でTS精密化([NWChem][8])
* 必要ならBSSE（錯体が支配的な場合だけ）([NWChem][8])
* 周波数→TS QC（虚数1本など）
* （重要例）IRC再確認

### 9.9 Thermochemistry & Kinetics

* GoodVibesで ΔG‡・ΔG_assoc・標準状態補正([GitHub][4])
* 出力：

  * ΔE‡（錯体/分離）
  * ΔG‡(T,P)（錯体/分離）
  * QCフラグ
* 反応器：Cantera 0D/1Dへ出力([Cantera][5])

### 9.10 Orchestration / Provenance

* 小規模：Snakemake/Make + JSONログでも可
* 大規模：QCArchiveでDB化・再実行性・重複排除([QCArchive][2])

---

## 10. “PubChem参照SDFからの変更”をどう扱うべきか（結論）

PubChem由来SDFは出発点として非常に有用ですが、

* 3Dが無い/計算生成である（再最適化が必要）([PMC][6])
* 塩/混合物・立体未定義などが混ざり得る（入力ゲートで吸収）([PMC][6])

という性質があるため、設計としては

1. **raw（元SDF）を必ず保持**
2. **std（標準化/修正後）を計算の正**
3. 変換履歴と理由を残す（MI/監査/再現性）

を標準にするのが最も運用に強いです。

---

## 11. 実装ロードマップ（無理に変えず、段階的に強くする）

### Phase 1（現行：アミン×HFを堅牢化）

* ProtonTransfer 反応クラス
* xTB docking/aISS + directed docking
* pysisyphus で scan→NEB/GSMのフォールバック
* 上位だけ NWChem saddle
* GoodVibesで ΔG‡二本立て

### Phase 2（HF以外へ）

* ProtonTransfer を HX（HF/HCl…）に一般化
* 開殻/電荷メタデータ（.UHF/.CHRG）運用を確立([XTB Docs][1])
* Cantera出力を標準化

### Phase 3（アミン以外の候補へ）

* 反応クラス追加（会合、付加、引き抜き等）
* 反応サイト検出の拡張（SMARTS/ルール）
* 失敗分類とデータベース化（QCArchive推奨）([QCArchive][2])

---

必要であれば、次のステップとして「**反応クラスI/FをYAMLで記述できる設計（SMARTS・結合変換・初期配置ルール・scan座標定義）**」まで落として、**“新しい反応（新しい相手ガス）が来てもコードを書き換えずに追加できる”**形の仕様案も作れます。

[1]: https://xtb-docs.readthedocs.io/en/latest/xtb_docking.html "Docking Submodule (aISS) — xtb doc 2023 documentation"
[2]: https://docs.qcarchive.molssi.org/overview/index.html?utm_source=chatgpt.com "QCArchive Overview — QCArchive documentation - MolSSI"
[3]: https://pysisyphus.readthedocs.io/?utm_source=chatgpt.com "Welcome to pysisyphus documentation! — pysisyphus 1.0.1 ..."
[4]: https://github.com/patonlab/GoodVibes?utm_source=chatgpt.com "patonlab/GoodVibes: Calculate quasi-harmonic free ..."
[5]: https://cantera.org/3.1/python/zerodim.html?utm_source=chatgpt.com "Zero-Dimensional Reactor Networks"
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8363119/ "
            Exploring Chemical Information in PubChem - PMC
        "
[7]: https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Notebooks/MolStandardize.ipynb "raw.githubusercontent.com"
[8]: https://nwchemgit.github.io/TASK.html "TASK - NWChem"


以下は「**各 SDF ファイルに 1 分子（1構造）**が入っている」前提で、
**いまは（アミン × HF）**、将来は **（アミン以外 × HF）**や **（アミン × HF以外のガス分子）**、さらに **（複数の反応物/副反応候補）**まで拡張できるように、**全体設計（アーキテクチャ）**と **詳細機能（モジュール仕様）**を“壊れにくい・運用しやすい”視点でブラッシュアップした提案です。

前回の有益な骨格（SDF→構造標準化→錯体生成→経路探索/TS→DFT精密化→ΔG‡→（必要なら）反応器）は基本維持し、**拡張点・データ品質・例外処理**を強化します。

---

## 1. 設計ゴール（将来拡張を見据えた要件定義）

### 1.1 目的

* 入力：

  * **候補分子群**（SDF/1分子/1構造）
  * **反応相手ガス分子群**（SDF または SMILES/組成/電荷・スピンなどの必要情報）
  * **反応クラス**（例：プロトン移動、付加、H引き抜き、会合/解離…）
  * 条件（T, P、必要なら滞留時間、混合比、反応器モデルなど）
* 出力：

  * 反応ごとの **ΔE‡/ΔG‡（錯体基準・分離基準の両方）**
  * TS/IRC 検証済みかどうかの **QCフラグ**
  * （任意）速度定数 **k(T)** / **k(T,P)** と、0D/1D 反応器での消費推定

### 1.2 設計思想

* **“反応クラス”をプラグイン化**して、分子がアミンでなくても HFでなくても回る
* PubChem由来SDFの“ゆらぎ”（塩・混合物、立体未定義、2Dしかない等）を **入力ゲートで吸収**し、後段の計算を安定化
* **多忠実度（xTB→DFT）**を前提に、

  * 低レベルは「候補生成・失敗分類・粗スクリーニング」
  * 高レベルは「最終値確定」
    と役割分担
* **失敗も資産化**（MI/運用改善のために必須）

---

## 2. 入力仕様（SDF 1分子/1構造を中核にした拡張可能な形）

### 2.1 基本入力（最小）

* `candidates/`：候補分子 SDF（各ファイルに1分子）
* `reactants/`：反応相手ガス分子

  * SDF でもよい（各ファイル1分子）
  * もしくは `reactants.yaml` に SMILES / 電荷 / スピン（多重度）を記述

### 2.2 メタデータ（推奨：将来の拡張に効く）

SDFそのものは「座標と結合」中心なので、将来のHF以外・アミン以外・ラジカル種などに備えて、**分子ごとの sidecar metadata（YAML/JSON）**を持てる設計が便利です。

例：`candidates/ABC123.sdf` と `candidates/ABC123.meta.yaml`

* `source`: PubChem / 社内 / 推定生成 など
* `pubchem_cid`: （あれば）
* `charge`: 0 / +1 / -1 …
* `multiplicity`: 1/2/3…（ラジカル等のため）
* `preferred_states`: tautomer/protomer の指定（任意）

※xTB docking は `.CHRG` と `.UHF`（分子全体と各分子の電荷・不対電子数）を読める仕様があり、**開殻・イオンを将来扱うときの基礎**になります。([XTB Docs][1])

---

## 3. 全体アーキテクチャ（層構造）

大きく 4 層に分けると破綻しにくいです。

### Layer A：データ層（正規化・来歴・DB）

* Molecule Registry（入力SDFの登録、ID採番、ハッシュで重複排除）
* Transform Log（標準化や修正の履歴を全て保存）
* Result Store（計算結果、TS/IRCログ、QCフラグ）

※スケールするなら QCArchive（QCFractal/QCPortal/QCFractalCompute）で「大量計算＋DB」を中核にするのが堅牢です。([QCArchive][2])

### Layer B：化学前処理層（PubChem由来SDFのゆらぎ吸収）

* 構造読み込み・サニタイズ
* “必要なら”構造修正（desalt、立体、H付与、3D化 etc）
* 状態列挙（必要なら tautomer/protomer、立体異性体）

### Layer C：反応生成・PES探索層（反応クラスプラグイン）

* 反応サイト抽出
* 錯体生成（複合体コンフォーマ）
* 生成物推定（反応テンプレ or 探索）
* 経路探索（NEB/GSM/scan → TS refine → IRC）

pysisyphus は NEB・Growing String などの Chain-of-States、TS探索、IRC を備え、YAML入力で使える旨が明記されています。([Pysisyphus][3])

### Layer D：物性化・速度論層（ΔG‡/k(T,P)/反応器）

* 熱化学補正（GoodVibes）
* 圧力依存（必要なら Arkane）
* 0D/1D反応器（Cantera）

GoodVibesは低周波振動に対するRRHOの問題を補正する目的で使われていること、また温度・圧力/濃度条件の設定ができることが説明されています。([GitHub][4])
Cantera の 0D 理想気体リアクタ（IdealGasReactor等）もドキュメント化されています。([Cantera][5])

---

## 4. PubChem由来SDFを“計算入力”として使うときの問題点と修正方針

PubChem由来SDFは非常に有用ですが、**量子化学（特にTS探索）にそのまま入れると壊れやすい**典型パターンがあります。ここを“入力ゲート”で吸収します。

### 4.1 3D構造が無い/不十分（2D座標のみ）

PubChemは 3D 構造（コンフォーマ）を提供できますが、**全化合物にあるわけではありません**。3Dモデル生成には「大きすぎない」「柔らかすぎない」「塩/混合物でない」など複数条件があり、条件を満たさない化合物は 3D ダウンロード対象から外れることがある、と説明されています。([PMC][6])
また、PubChemの3D構造は**実験構造ではなく計算生成**であることも明記されています。([PMC][6])

**対策（仕様化）**

* 入力SDFが 2D/3Dどちらでも、必ず

  1. **H付与**
  2. RDKit埋め込み（ETKDG）で3D化（必要時）
  3. xTB/CREST で**再最適化**（全件）
     を行い、「以降の計算は自前で整えた3D」を正とする
* PubChem 3D でも “そのまま信じない”
  → 初期構造として利用しつつ、同じ低レベル（xTB）で整合させる

### 4.2 塩・混合物・多成分（SDF 1分子でも“切断されてる”）

PubChemの3Dコンフォーマ生成条件には「単一の共有結合ユニット（塩や混合物ではない）」が含まれています。([PMC][6])
つまり、PubChem由来のSDFには**塩（カウンターイオン含む）**などが現実に混ざり得ます。

**対策（デフォルト挙動＋例外扱い）**

* デフォルト：**最大フラグメントを主成分（parent）として採用**し、他成分はログに残して除外
* 例外：反応器内で「塩として」評価したい場合は、明示フラグで全成分を保持する分岐も用意

RDKitの `rdMolStandardize` には FragmentParent/Uncharger/Reionizer 等を使った標準化の枠組みがあり、**desalt/中和/正規化などの部品が揃っている**ことが示されています。([GitHub][7])

### 4.3 立体化学が未定義（stereocenter未指定）

PubChemの3Dモデル生成条件には「未定義の立体中心が少ない」旨が含まれます。([PMC][6])
将来の候補分子が増えると、立体未定義が混ざる可能性が上がります。

**対策**

* 入力ゲートで

  * 立体が未定義 → “未定義”タグ付け
  * ルール：

    * 重要候補のみ立体異性体を列挙（上限数を設定）
    * それ以外は「未定義のまま」代表1種で走らせ、結果に不確かさフラグを付ける
* “必ず列挙”ではなく **計算資源と目的に応じて切り替え**可能にする

### 4.4 プロトン化状態/互変異性（tautomer/protomer）のズレ

PubChemの表現（中性形など）が、ガス相条件での主要状態と一致しない場合があります（特に強塩基/強酸、双性イオンなど）。

**対策（拡張可能な設計）**

* デフォルトは「SDFの状態を尊重」しつつ、
* 反応クラスが要求する場合のみ（例：プロトン移動、付加反応でプロトンが鍵）

  * tautomer/protomer を**限定列挙**して “最も起こりやすい経路” を拾う
* 列挙で爆発しないように、**優先順位ルール**（反応中心近傍のみ列挙、上限数、エネルギー窓）を仕様化

RDKit標準化系は正規化・再イオン化・中和などの処理を組めることが示されています（例：Normalize/Reionize/Uncharger等）。([GitHub][7])

---

## 5. 反応クラス（テンプレート）プラグイン設計

将来「アミン以外」「HF以外」に広げるには、反応を **“反応クラス”**として抽象化し、以下のI/Fを持たせるのが強いです。

### 5.1 反応クラスが提供すべき機能（インターフェース）

各 ReactionClass（例：ProtonTransfer、NucleophilicAddition…）は少なくとも：

1. **反応サイト検出**

* `find_sites(molecule)`：SMARTSや部分構造規則で候補サイト一覧
* 例：酸（H供与）/塩基（受容）/求核/求電子/ラジカル中心 など

2. **錯体構築ルール**（Directed complex building）

* `build_complexes(reactantA, reactantB, siteA, siteB)`
* “どの原子同士を近づけるべきか”を反応クラスが知っていることが重要

3. **生成物推定**（テンプレ駆動 or 探索）

* `build_products(...)`：結合変換（bond order/charge）を反映して product を作る
* productが曖昧なクラスは「探索モード」へフォールバック

4. **経路探索戦略**（推奨アルゴリズムを返す）

* `suggest_path_method()`：scan/NEB/GSM/単端TS探索 など
* `reaction_coordinate()`：scan用の座標定義（例：距離差）

5. **TS妥当性判定**

* `validate_ts(ts, irc_endpoints)`：虚数モードの性質、接続先の正しさ等

---

### 5.2 具体例：現行（アミン × HF）の “ProtonTransfer” を一般化する

* サイト検出：

  * Donor側（HFなど）：H–X（X=F, Cl, O, N…）
  * Acceptor側：塩基性ヘテロ原子（N,O,S…）
* 錯体構築：H（donor）を acceptor に向けて配置
* 生成物推定：Hを acceptor に移し、X側はアニオン化（必要なら接触イオン対として初期配置）
* 経路探索：

  * scan座標 `s = r(X–H) - r(H–A)` を第一候補
  * うまくいかなければ NEB/GSM
* バリアレス判定：scanが単調なら “TS無し” として分類（ΔG‡_intra ≈ 0扱い＋QCフラグ）

---

### 5.3 将来追加しやすい反応クラス例（ガス相・半導体プロセス想定）

* **Association/Dissociation（会合/解離）**：吸着ではなく気相会合（錯体形成自由エネルギー）
* **Halogen acid proton transfer**：HF/HCl/HBr… × 塩基一般
* **Nucleophilic substitution / addition**：ハロゲン化物・シラン系・カルボニル系（候補次第）
* **H-abstraction（引き抜き）**：ラジカル/反応性ガスが対象になった場合
* **Ligand exchange**：Si–F/Si–Cl などの交換反応（将来の化学系次第）

---

## 6. 錯体生成（Complex Builder）を “汎用化”する設計

### 6.1 基本方針

錯体生成は「HF特化」から脱却し、**どの2分子でも**回るようにします。

* **一般モード**：xTB docking (aISS) を使用
  aISS は「任意分子を dimers/aggregates に追加でき、interaction site screening と xTB‑IFF の遺伝的最適化 → GFN最適化」という流れで、デフォルトで上位15構造最適化、NCI ensembleも可能と説明されています。([XTB Docs][1])
* **Directed モード**：反応クラスが指定した原子集合へ誘導
  xTB docking の “directed docking” 入力例（atoms/elements、scaling factor）が示されています。([XTB Docs][1])

### 6.2 開殻・イオンへの備え

将来 HF以外（例えばラジカル種）が入ると、電荷・スピンが必須になります。
xTB docking は `.CHRG` と `.UHF` を読んで分子1/2の電荷・不対電子数を扱えるので、ここを標準I/Fにしておくのが得策です。([XTB Docs][1])

---

## 7. 経路探索・TS・DFT精密化：汎用設計（“反応が増えても壊れにくい”）

### 7.1 低忠実度（xTB）段階：**探索の母集団を作る**

* 複合体コンフォーマ（複数） × 生成物コンフォーマ（複数）を作り、
  NEB/GSM/scan を回して TS候補を複数出す
* pysisyphus を「共通の経路探索ランナー」とする
  （NEB/Growing String/IRCがあること、YAML/ライブラリ両対応は運用面で強い）([Pysisyphus][3])

### 7.2 高忠実度（DFT）段階：**上位候補だけ確定**

将来 HF以外の相手でも、基本は同じです。

* NWChem の TASK には `saddle`（遷移状態/鞍点探索）があり、`task dft saddle` のように使えることが明記されています。([NWChem][8])
* さらに NWChem は BSSE（counterpoise）計算の仕組みも TASK の説明内にあり、錯体の取り扱いで必要になったときに“後付け”できます。([NWChem][8])

**設計上のポイント**

* DFT精密化は「全件」ではなく

  * xTBで成功した上位
  * バリアレス疑いだが重要なもの
  * QCフラグが怪しいもの
    に限定できるよう、ワークフローに **選別ステージ**を置く

---

## 8. 熱化学・速度論の汎用化（ΔG‡とkの扱い）

### 8.1 ΔG‡は“定義を固定”して二本立て出力

将来反応が増えるほど、会合の有無で ΔG‡ の意味が揺れます。したがって：

* **錯体基準**：ΔG‡(TS − complex)
* **分離基準**：ΔG‡(TS − (A+B))

を必ず出し、さらに会合自由エネルギー ΔG_assoc も別出力にするのが堅牢です。

### 8.2 GoodVibesは“条件補正・低周波補正の標準ツール”

GoodVibesは低周波振動のRRHO問題を補正する目的で使われ、温度・圧力/濃度条件も指定できる旨が説明されています。([GitHub][4])
→ 反応相手がHFでなくても、会合が絡むガス相反応全般で効きます。

### 8.3 反応器モデルは Cantera を共通出口にする

0D理想気体リアクタ等が整備されており、計算した速度定数を“プロセス条件”に落とす出口として汎用です。([Cantera][5])

---

## 9. 具体的な機能一覧（MVP→拡張まで）

ここは「実装時にそのままチケットに切れる粒度」で書きます。

### 9.1 Molecule Ingestion

* SDF読み込み（1ファイル=1分子）
* RDKitサニタイズ
* 分子ID生成：InChIKey/SMILESハッシュ
* メタデータ結合（meta.yamlのcharge/multiplicityなど）
* 入力QC：

  * 多フラグメント検出（塩/混合物）
  * 立体未定義数
  * 元素種

### 9.2 Standardization（PubChem由来修正を含む）

* `raw_mol` を保存（オリジナル）
* `std_mol` を生成（標準化した作業用）

  * FragmentParent（主成分化）
  * Normalize/Reionize/Uncharger 等（反応クラスに応じてON/OFF）([GitHub][7])
* 修正ログを必ず残す（MI資産化）

### 9.3 3D/Conformer Manager

* 3Dが無ければ埋め込み（RDKit）
* xTB/CRESTで再最適化（“計算用3D”の統一）
* コンフォーマ集合管理（エネルギー窓、RMSDクラスタ）

### 9.4 Reactant Library

* 反応相手分子を SDF/SMILES で登録
* 反応相手ごとに「代表状態」（charge/multiplicity）を指定可能
* HFなど頻出種はテンプレを用意（ただし将来拡張は同じI/F）

### 9.5 Reaction Enumeration

* ReactionClassプラグインの登録/選択
* 候補×相手×サイトの組み合わせ生成
* 計算資源に応じた組み合わせ削減（上限、優先順位）

### 9.6 Complex Builder

* 一般：xTB docking/aISS（NCI ensemble）([XTB Docs][1])
* Directed：反応クラスが指定した atoms/elements で誘導([XTB Docs][1])
* 生成物初期構造の自動生成（反応クラスごと）

### 9.7 Path Search / TS / IRC Runner

* pysisyphus で NEB/GSM/scan を回す([Pysisyphus][3])
* TS refine
* IRC（重要ケースのみでも可）
* 失敗分類（NEB発散、TS虚数>1、IRC不一致…）

### 9.8 DFT Refiner

* NWChem `task dft saddle` でTS精密化([NWChem][8])
* 必要ならBSSE（錯体が支配的な場合だけ）([NWChem][8])
* 周波数→TS QC（虚数1本など）
* （重要例）IRC再確認

### 9.9 Thermochemistry & Kinetics

* GoodVibesで ΔG‡・ΔG_assoc・標準状態補正([GitHub][4])
* 出力：

  * ΔE‡（錯体/分離）
  * ΔG‡(T,P)（錯体/分離）
  * QCフラグ
* 反応器：Cantera 0D/1Dへ出力([Cantera][5])

### 9.10 Orchestration / Provenance

* 小規模：Snakemake/Make + JSONログでも可
* 大規模：QCArchiveでDB化・再実行性・重複排除([QCArchive][2])

---

## 10. “PubChem参照SDFからの変更”をどう扱うべきか（結論）

PubChem由来SDFは出発点として非常に有用ですが、

* 3Dが無い/計算生成である（再最適化が必要）([PMC][6])
* 塩/混合物・立体未定義などが混ざり得る（入力ゲートで吸収）([PMC][6])

という性質があるため、設計としては

1. **raw（元SDF）を必ず保持**
2. **std（標準化/修正後）を計算の正**
3. 変換履歴と理由を残す（MI/監査/再現性）

を標準にするのが最も運用に強いです。

---

## 11. 実装ロードマップ（無理に変えず、段階的に強くする）

### Phase 1（現行：アミン×HFを堅牢化）

* ProtonTransfer 反応クラス
* xTB docking/aISS + directed docking
* pysisyphus で scan→NEB/GSMのフォールバック
* 上位だけ NWChem saddle
* GoodVibesで ΔG‡二本立て

### Phase 2（HF以外へ）

* ProtonTransfer を HX（HF/HCl…）に一般化
* 開殻/電荷メタデータ（.UHF/.CHRG）運用を確立([XTB Docs][1])
* Cantera出力を標準化

### Phase 3（アミン以外の候補へ）

* 反応クラス追加（会合、付加、引き抜き等）
* 反応サイト検出の拡張（SMARTS/ルール）
* 失敗分類とデータベース化（QCArchive推奨）([QCArchive][2])

---

必要であれば、次のステップとして「**反応クラスI/FをYAMLで記述できる設計（SMARTS・結合変換・初期配置ルール・scan座標定義）**」まで落として、**“新しい反応（新しい相手ガス）が来てもコードを書き換えずに追加できる”**形の仕様案も作れます。

[1]: https://xtb-docs.readthedocs.io/en/latest/xtb_docking.html "Docking Submodule (aISS) — xtb doc 2023 documentation"
[2]: https://docs.qcarchive.molssi.org/overview/index.html?utm_source=chatgpt.com "QCArchive Overview — QCArchive documentation - MolSSI"
[3]: https://pysisyphus.readthedocs.io/?utm_source=chatgpt.com "Welcome to pysisyphus documentation! — pysisyphus 1.0.1 ..."
[4]: https://github.com/patonlab/GoodVibes?utm_source=chatgpt.com "patonlab/GoodVibes: Calculate quasi-harmonic free ..."
[5]: https://cantera.org/3.1/python/zerodim.html?utm_source=chatgpt.com "Zero-Dimensional Reactor Networks"
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8363119/ "
            Exploring Chemical Information in PubChem - PMC
        "
[7]: https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Notebooks/MolStandardize.ipynb "raw.githubusercontent.com"
[8]: https://nwchemgit.github.io/TASK.html "TASK - NWChem"





-----------------
以下は、**「SDF（1ファイル=1分子）を入力して、ガス相（半導体ガスエッチング装置チャンバー相当）での反応活性化エネルギー（できれば速度定数まで）を自動計算し、材料探索に使う」**という全体ワークフロー／設計は維持しつつ、**各フェーズのモデル・ライブラリ妥当性を“最新情報も踏まえて”再点検**した上で、**問題点の指摘→対策→ブラッシュアップ案**まで落とし込んだ提案です。
（※「今のものが良い場合は変えない」方針に従い、置換が必須でない箇所は“維持”を基本にし、**追加/代替の選択肢**として整理します。）

---

## 0. 前提（ガスエッチング装置スケールでの“落とし穴”を先に明確化）

半導体のガスエッチング装置での“ガス相反応”に寄せると、**研究室の標準的な活性化エネルギー計算（単一TS→TST）だけでは不十分になりやすい**です。理由は主に3つです：

1. **低圧で「会合→安定化」や「多段反応」の圧力依存が効く**
   　→ ΔE‡だけでなく、場合によっては**マスター方程式（ME）**が必要。

2. **HF + 塩基（アミン等）は“障壁なし（barrierless）/超浅いTS”になり得る**
   　→ TSが見つからない/IRCが崩れるケースが現実に多い。
   　→ “活性化エネルギー”をどう定義してスクリーニングするか（ΔG‡？捕獲律速？）が重要。

3. **候補分子数が多いので、DFT一段で網羅は破綻しやすい**
   　→ **階層化（xTB→DFT→必要なら高精度）**と、ジョブ管理/データ基盤が本質。

この3点を前提に、以下のワークフローを再設計（ただし骨格は維持）します。

---

## 1. 維持する全体アーキテクチャ（層構造はそのまま、強化点を明確化）

### 層A：入力・正規化（SDF→量子化学で扱える状態へ）

* SDF読込、構造正規化、立体/電荷/多重度、プロトン化状態、フラグメント除去
* PubChem由来SDFの“必要な修正”もこの層で吸収

### 層B：前処理（配座・会合複合体・反応初期/終状態の用意）

* 配座探索（候補分子）
* HF（将来は他ガス）との**遭遇複合体**生成
* 反応ごとの初期状態/終状態（リアクタント/プロダクト）候補生成

### 層C：反応生成・PES探索（自動で反応経路/TS候補を出す）

* 反応テンプレート（当面HF×アミン、将来一般化）
* NEB/GSM等で経路→TS初期推定
* TS最適化、虚振動数、IRCで検証
* “TSが無い/障壁が無い”系は別分岐（捕獲律速/MEへ）

### 層D：DFT精密化（活性化自由エネルギー/速度定数に足る精度へ）

* TS/IRC/振動数をDFTで確定
* 必要に応じて基底関数/汎関数/分散/BSSE対策

### 層E：物性化・速度論・装置スケール評価

* 熱化学補正（RRHO/準調和/ローター等）
* TST/VTST、圧力依存（ME）
* Canteraで反応器モデル化（ここは維持とのこと）

### 層F：データ基盤・自動化（大量計算に必須）

* 計算投入・再実行・エラー処理・重複排除・トレーサビリティ

---

## 2. 各フェーズのモデル/ライブラリ妥当性（最新も踏まえて再評価）

以下、ユーザー要望の3観点に合わせて整理します。

---

# 2-1. DFT計算：NWChem以外の“無料で現実的な”候補と、結論

## 結論（おすすめの位置づけ）

* **NWChemは「維持してOK」**：オープンソース(ECL 2.0)でTS探索（saddle）まで一通り可能。 ([NWChem][1])
* **追加で最優先に検討する価値が高いのは Psi4**（オープンソース/LGPL-3.0、TS/IRC/分散補正、Python駆動が強い）。 ([GitHub][2])
* **スケール（大量計算）重視なら PySCF + GPU4PySCF が強力**（Apache-2.0、GPUでSCF/DFT/勾配/ヘッセが高速化、最近の更新が活発）。 ([PySCF][3])
* “Gaussian並み以上”は**コードではなく手法選定（汎関数/基底/分散/グリッド/検証）**で達成するのが本筋。ただし、Psi4/PySCFはその実装・自動化のしやすさが強い。

以下、具体比較。

---

## (A) NWChem（現状維持の妥当性）

* **ライセンス**：ECL 2.0 のオープンソース ([NWChem][1])
* **TS探索**：`TASK ... saddle` で遷移状態（鞍点）探索が可能 ([NWChem][4])
* **StepperでもTS探索**：ヘッセ最小固有ベクトル追跡でTS探索できる旨の記述 ([NWChem][5])

**評価**

* “無料で、Gaussianの典型的なDFTワークフロー（最適化・TS・振動数）を回す”という意味で**依然として有力**。
* 一方で「大量自動化」「Python統合」「ジョブの標準化I/O」という観点では、NWChem単体より**上位のオーケストレーション層**（後述 QCArchive/QCEngine 等）と組むのが効きます。

---

## (B) Psi4（最重要の代替候補：無料・高精度・自動化が強い）

* **ライセンス**：LGPL-3.0 ([GitHub][2])
* **TS最適化**：OptKingで `OPT_TYPE TS` によりTS最適化が可能。最新マニュアルにもTS設定が明記。 ([PsiCode][6])
* **IRC**：OptKingでIRC計算（`opt_type` にIRCがある） ([optking.readthedocs.io][7])
* **分散補正**：DFT-D3/DFT-D4 インターフェースが公式に存在 ([American Chemical Society Publications][8])
* **高精度化のための機能**：CBS外挿やフォーカルポイント等の高精度手法も扱えることがPsi4の論文で述べられている ([PMC][9])

**評価（ガス相反応バリアに向く理由）**

* TS/IRC/振動数まで含めた**“反応障壁ワークフロー”をPythonから制御**しやすい。
* “Gaussianと同等のDFT”は、同一汎関数・同一基底・同等の数値設定に寄せれば概ね可能（コード差より設定差が支配的）。
* **不確実性の出る箇所**（格子/積分、SCF収束、虚振動数の扱い）をテンプレート化しやすい。

---

## (C) PySCF（大量探索・高速化の切り札：GPU含めて現実的）

* **PySCF本体は Apache-2.0**（商用利用の観点で扱いやすい） ([PySCF][3])
* **GPU4PySCF**：PySCFのSCF/DFT等をGPUで高速化するプラグイン。公式ドキュメントでDF（密度フィッティング）条件ではA100で大きな高速化が述べられている。 ([PySCF][10])
* **GPU4PySCFの更新**：PyPI上で2025/12/26に1.5.2が出ている（比較的最近まで継続更新） ([PyPI][11])
* **TS最適化**：PySCF拡張にTS最適化（qsdopt）や geomeTRIC 連携がある ([American Chemical Society Publications][12])

**評価**

* “Gaussian代替”というより、**大量計算の実行基盤（エネルギー/勾配/ヘッセ）として非常に強い**。
* TS探索そのものは**外部最適化器（Sella/geomeTRIC/ASE等）と組む**設計にすると伸びます（後述）。
* 反応探索を回す場合、**GPUが使える環境なら極めて有利**になり得ます。

---

## (D) Dalton / OpenMolcas（無料だが“用途を選ぶ”）

* **Dalton**：LGPL-2.1のオープンソース ([daltonprogram.org][13])

  * TS探索として“image法”の記述あり ([GitHub][14])
* **OpenMolcas**：LGPL-2.1、マルチコンフィギュレーショナル（多参照）に強い ([GitHub][15])

**評価**

* HF系・ラジカル・イオンなどで**多参照性が疑われるケース**（例：開殻/近接スピン状態）にOpenMolcasは選択肢。
* ただし装置スケール探索（大量バリア計算）の主力には重いので、**“例外処理枠”**に置くのが現実的。

---

## (E) GAMESS（“無償だが非オープンソース”枠）

* “学術・産業に対し無償のサイトライセンスがある”旨が複数箇所で確認できる一方、**再配布等に制限**がある（ライセンス同意が必要）。 ([about.gitlab.com][16])
* TS探索（SADPOINT）などの入力仕様はマニュアルに明記。

**評価**

* 会社利用で“無償”が許されるなら候補。ただし、**ライセンス管理・配布形態・社内利用範囲**の確認が必須。
* オープンソース縛りの方針なら外す。

---

## (F) ORCA（参考：学術無料/商用ライセンス）

* ORCAは**学術・個人は無料、商用はライセンス**と明記されている ([FACCTs][17])
  → 今回の「有償不可」という条件では原則対象外（ただし社内規程次第）。

---

### DFT層の推奨まとめ（変更を強制しない形）

* **維持**：NWChem（既存資産活かす） ([NWChem][1])
* **追加推奨**：Psi4（“反応障壁ワークフロー”の自動化がしやすい） ([GitHub][2])
* **スケール用に追加推奨**：PySCF + GPU4PySCF（大量・高速化） ([PySCF][3])

---

# 2-2. 反応生成・PES探索層：用途別の処理とライブラリ妥当性

ここが“自動化の成否”を決めます。ガスエッチングスケールでは、**反応数 × 配座数 × 複合体数**が爆発しやすいので、以下のように“用途で分ける”のがポイントです。

---

## 2-2-1. 配座探索・会合複合体生成：xTB + CREST は妥当（むしろ必須寄り）

### xTB（準DFTレベルの前段としての事実上の標準）

* `xtb` はLGPL（オープンソース） ([GitHub][18])
* 直近のリリースページでは安定版 v6.7.1 が示され、bleeding-edgeのプレリリースも存在 ([GitHub][19])
* リリースノートに directed docking / aISS などの記述（会合構造探索に有用） ([GitHub][19])

### CREST（配座・回転異性体の系統探索）

* CRESTはGitHubで公開され、LGPLライセンスを含むオープンソースとして配布されている ([GitHub][20])

**ガスエッチング用途での妥当性**

* “チャンバー内の遭遇複合体（pre-reactive complex）”が重要になる反応（HF会合、プロトン移動）で、**複合体の探索をしないとΔG‡が壊れる**ことが多い。
* xTB/CRESTは、ここを現実的コストで回す手段として妥当。

---

## 2-2-2. 反応経路探索（NEB/GSM/TS最適化）：pysisyphusは機能面◎、ただしライセンス面で要注意

### pysisyphus（機能・自動化の観点では非常に強い）

* 反応経路（Chain-of-States）やTS最適化等を一通り提供する旨が論文/ドキュメントに明記 ([Wiley Online Library][21])
* TS最適化はヘッセを使う手法が有望だが、勾配のみのdimer法でも可能と説明 ([pysisyphus.readthedocs.io][22])

### ただし：pysisyphusは **GPL-3.0**

* GitHub上でGPL-3.0と明記 ([GitHub][23])

**影響（重要）**

* 社内専用で使うだけなら問題にならないケースも多いですが、

  * ソフトを外部配布する、
  * 製品/提供物に組み込む、
  * “派生物”扱いになる形で配布する、
    などの可能性があるなら、**法務/ライセンス判断が必要**です（強いコピーレフト）。

✅ **結論**：

* **研究用途/社内限定運用が主なら「維持」でOK**（機能が強い）。
* **将来“配布や提供”が視野なら、置換可能な設計にしておく**（後述の代替案）。

---

## 2-2-3. pysisyphusの代替（“GPLを避けたい”場合の実務的スタック）

### 代替スタック案（用途別に役割分担）

* **ASE（NEB等の経路探索）**：GNU LGPLで提供され、商用ソフトと同梱する条件などがライセンスページに整理されている ([ASE Library][24])
* **Sella（鞍点/TS最適化）**：TS探索のオープンソース実装として文献・配布物がある。conda-forge情報ではパッケージライセンスがLGPL-3.0-onlyとされている ([American Chemical Society Publications][25])
* **geomeTRIC（安定した幾何最適化）**：MITライセンスで、外部QMコードを呼んで幾何最適化するパッケージとしてPyPIでも説明されている ([GitHub][26])

**この代替が“ガスエッチングのスケール”に合う理由**

* pysisyphusほど“全部入り”ではないが、

  * NEB（ASE）
  * TS最適化（Sella）
  * 収束の安定化・拘束（geomeTRIC）
    を**疎結合で差し替え可能**にできる。
* 背後のQMエンジンはPsi4/NWChem/PySCF等に差し替えられる（QCEngine等を使うとさらに楽）。

---

## 2-2-4. “反応プロファイル自動生成”の上位フレームワーク候補：autodE（追加の選択肢）

* autodEは**MITライセンス**で、TS探索・配座探索・複合体生成・複数QMコードのラッパ等を持つとREADMEに明記 ([GitHub][27])

**注意点**

* README上の“依存QMコード”には NWChem が含まれるが、ORCA/Gaussian等も列挙されているので、**今回の“有償不可”条件ではNWChemや（実装があれば）PySCF側に寄せる必要**がある。 ([GitHub][27])
* HF反応に対して“テンプレが最初から最適化されている”保証はないため、あなたの用途では「反応テンプレ層」を自作/プラグイン化した方が堅い。

---

# 2-3. 物性化・速度論層：Cantera維持でOK。周辺は“圧力依存”を強化すると装置スケールに合う

## 2-3-1. Cantera（維持でOK）

* 装置スケール（反応器）モデルとして適切、という判断は妥当（ここは維持）。

---

## 2-3-2. 熱化学・速度論：Arkane（RMG）を“圧力依存の主役”として再評価

* ArkaneはRMGプロジェクト内で、**熱化学・速度論（TST等）**を扱うツールとしてREADMEにまとまっている ([GitHub][28])
* RMG-Py自体がMITライセンス ([GitHub][14])

**装置条件との整合**

* 低圧で“会合→再解離”や“多段”が効くと、単純TSTより**圧力依存**（falloff/ME）が重要になる。
* Arkane（あるいは次のMEソルバ）を“必要時に発動する分岐”として持つのが、ガスエッチング向けに自然。

---

## 2-3-3. マスター方程式（ME）ソルバの追加候補（無料）

* **MESS（Auto-Mech）**：GitHubでApache-2.0ライセンス、condaで入る旨がREADMEに記載 ([GitHub][29])
* **MESMER**：オープンソースのMEソルバとして文献で整理されている ([American Chemical Society Publications][8])

**提案**

* “全ケースME”は重いので、

  * 会合（R + HF ⇄ [R···HF]）が強い
  * 生成物が安定化しにくい
  * 反応が多段（井戸が複数）
    のときだけME分岐（Arkane or MESS/MESMER）を入れるのが現実的です。

---

# 2-4. 大量自動計算（設計層）：QCArchive/QCEngine を推奨（最新状況も確認）

この層は“正解に近い/遠い”というより、**スケールと再現性に直結**します。

* QCArchiveは、サーバ（QCFractal）・クライアント（QCPortal）・ワーカー（QCFractalCompute）からなるアーキテクチャとして公式ドキュメントに整理 ([docs.qcarchive.molssi.org][30])
* QCFractalはBSD-3で公開され、PyPIでは **v0.63 が 2025/09/26 リリース**と記載 ([GitHub][31])
* QCEngineはBSD-3で、QCSchemaの入出力標準化/実行エンジンとしてGitHubに明記 ([GitHub][32])
* QCSchema自体も仕様としてドキュメント化されている ([molssi-qc-schema.readthedocs.io][33])

**メリット（あなたの用途に刺さる点）**

* 反応候補×条件×手法×配座、で計算数が爆増しても、

  * “どの計算がどの分子/反応/条件に対応するか”
  * “失敗をどう再投入したか”
  * “同一計算の重複排除”
    がデータベースで扱える。
* NWChem/Psi4/PySCFといった複数バックエンドを“同じスキーマ”で扱う設計に寄せられる。

---

## 3. 化学理論×MI観点での「問題点」指摘と、対策を織り込んだブラッシュアップ

ここからが実務で効く部分です。以前の設計方針（階層化・自動化）を尊重しつつ、半導体ガスエッチング向けに“落ちるところ”を塞ぎます。

---

### 問題1：HF×塩基（アミン等）はTSが存在しない/不安定になり得る

**症状**

* NEB/GSMの最高点が“TS”にならず、そのままイオン対へ滑る
* DFTで虚振動数が複数出る/IRCが分岐する

**対策（ワークフローに組み込む）**

* TS探索を“前提”にせず、最初に **反応タイプ判定**を入れる

  * (a) 明確な鞍点がある → 通常TSワークフロー
  * (b) barrierless/捕獲律速っぽい → **捕獲律速モデル + ME（必要なら）**
* スクリーニング指標も切り替える

  * barrierless系：ΔE‡よりも **会合自由エネルギー（ΔG_assoc）**、反応熱、生成物の安定性、再解離のしやすさ
  * TS系：ΔG‡（T,P条件込み）を主指標に

---

### 問題2：配座・複合体の取りこぼしがΔG‡を壊す（ガス相は特にエントロピーが効く）

**症状**

* “たまたまの配座”でバリアが上下してランキングが崩れる
* 反応座標に沿う会合様式が見つからずTS探索が失敗

**対策**

* “反応1件につき1構造”をやめ、**反応=（複数配座×複数複合体）**で扱う
* CREST/xTBで

  * 候補分子配座（単体）
  * 遭遇複合体（HFとの複合体）
  * プロダクト側も必要なら
    を列挙してからPES探索へ（この設計は維持推奨）。 ([GitHub][18])

---

### 問題3：低圧チャンバー条件では“圧力依存”が無視できないケースが出る

**症状**

* 会合しても安定化せず戻る
* 多段反応で井戸が複数あり、単純TSTが効かない

**対策**

* **ME分岐を実装**：Arkane or MESS/MESMERで必要時に評価 ([GitHub][28])
* “全件ME”ではなく、

  * 会合が強い（深い井戸）
  * 生成物が複数/多段
    などのフラグで発動。

---

### 問題4：PubChem由来SDFの“そのまま投入”は危険（気相・高温・装置条件と不整合）

**ありがちなズレ**

* プロトン化状態/塩が混じる
* フラグメントが複数入る（溶媒・対イオン）
* 立体化学不確定、Hが省略

**対策（入力層に仕様として固定）**

* すべてのSDFに対して必ず：

  * **単一フラグメント化**（最大フラグメントのみ採用等）
  * **明示H付与**
  * 電荷・多重度を「明示入力できる」設計（将来のHF以外にも必須）
  * 気相前提の再最適化（まずxTB、その後DFT）

---

### 問題5：ライセンスが将来の製品化/共有の障害になる

* pysisyphus：GPL-3.0 ([GitHub][23])
* xTB：LGPL-3.0-or-later ([GitHub][18])
* CREST：オープンソース（LGPL-3.0の記述あり） ([GitHub][20])
* Psi4：LGPL-3.0 ([GitHub][2])
* PySCF：Apache-2.0 ([PySCF][3])

**対策**

* 設計として「PES探索エンジン」を抽象化し、

  * 研究用途：pysisyphus（機能優先）
  * 将来配布：ASE+Sella+geomeTRIC（置換）
    のどちらにも差し替え可能にする。 ([CAMD Wiki][34])

---

## 4. ブラッシュアップ後の“推奨デフォルト構成”（無料縛り、将来拡張も意識）

### 4-1. まず現実的に回る「デフォルト（推奨）」

* **構造処理**：RDKit（SDF正規化/フラグメント/原子タイプ/反応テンプレの下地）
* **配座/複合体**：CREST + xTB（遭遇複合体も含めて列挙） ([GitHub][20])
* **PES探索**：

  * 研究/社内運用→ pysisyphus（NEB/GSM/TS/IRCを一括） ([Wiley Online Library][21])
  * GPLが懸念→ ASE（NEB） + Sella（TS） + geomeTRIC（最適化） ([CAMD Wiki][34])
* **DFT**：

  * 既存維持→ NWChem ([NWChem][1])
  * 追加推奨→ Psi4（TS/IRC/分散が組みやすい） ([GitHub][2])
  * 大量高速化→ PySCF + GPU4PySCF ([PySCF][3])
* **熱化学/速度論**：GoodVibes（高速）＋ Arkane（圧力依存が必要な時だけ）＋ Cantera（維持） ([GitHub][28])
* **データ/実行管理**：QCArchive（QCFractal）＋ QCEngine（標準化） ([docs.qcarchive.molssi.org][30])

---

### 4-2. “Gaussian相当以上”を狙うための実務ガイド（無料コードで精度を上げる）

ここはコード変更より**計算プロトコル**が支配的です。設計に以下のスイッチを持たせるのが効きます：

* **汎関数/基底の標準セット**（例：スクリーニング用、最終用を分ける）
* **分散補正ONを基本**（会合複合体が重要なため。Psi4ならD3/D4を組み込み可能） ([American Chemical Society Publications][8])
* **TS確認の標準手順**：虚振動数=1、IRCで両側が目的の極小に接続
* **会合反応は標準状態の扱いを厳密化**（1 bar基準 vs 実圧換算、装置条件の圧力へ）
* **必要なら高精度単点（小分子限定）**：Psi4のCBS外挿等でベンチマークを作り、DFTの系統誤差を補正 ([PMC][9])

---

## 5. 将来拡張（アミン以外 × HF以外）を見据えた全体設計の具体案

「入力がSDF（1分子）である」前提で拡張するなら、**“Species（分子）”と“Reaction（反応）”を完全に分離**して設計するのが重要です。

### 5-1. 入力仕様（最小限）

* `species/` ディレクトリにSDF（1分子/ファイル）
* 追加のメタ情報（YAML/JSON推奨）：

  * charge, multiplicity（不明なら推定→ただしログに残す）
  * gas conditions（T, P, carrier gasなど）
  * “この分子は反応物/添加剤/生成物候補”などロール

### 5-2. 反応テンプレートをプラグイン化（HF以外にも対応）

* `reaction_templates/` に

  * 反応タイプ（例：プロトン移動、付加、脱離、抽出）
  * 反応中心の探索規則（SMARTS/グラフ変換）
  * 初期/終状態の構造生成（複合体の向き、距離拘束）
    を“モジュール”として登録できるようにする。

### 5-3. 手法選択（Level of Theory: LoT）を宣言的に

* `lot_screen`：xTB/CREST（配座・複合体・粗経路）
* `lot_refine`：DFT（Psi4/NWChem/PySCF）
* `lot_benchmark`：必要時のみ（Psi4でCBS等） ([PMC][9])

---

## 6. 最後に：今回の再検討で“変えるべきでない/変えるべき”ポイントの整理

### 変える必要がない（維持推奨）

* **階層化（前段xTB→後段DFT）**：ガスエッチングのスケールで必須
* **Cantera**：装置スケール反応器として妥当（維持でOK）

### 追加/強化すべき（ワークフローの成功率を上げる）

* **DFTエンジンを“複数対応”にする**：NWChem維持 + Psi4追加が最も効く ([NWChem][1])
* **大量計算の基盤**：QCArchive/QCEngineでジョブとデータを統合管理 ([docs.qcarchive.molssi.org][30])
* **圧力依存/障壁なし分岐**：Arkane/MEソルバ（MESS/MESMER）を“必要時だけ”発動 ([GitHub][28])
* **pysisyphus採用時のGPLリスクを設計で吸収**（差し替え可能に） ([GitHub][23])

---

もし次に進めるなら、あなたの既存設計を崩さずに、実装としては

1. **SDF標準化 + CREST/xTB配座 + HF遭遇複合体生成**
2. **“反応タイプ判定（TSあり/なし）”分岐**
3. **TSあり：NEB/GSM→TS→IRC→DFT精密化**
4. **TSなし：捕獲律速 +（必要なら）ME**
5. **（上位）QCFractalでデータと実行を一元化**

の順に固めるのが最短距離です。

必要なら、上の提案を **(i) 具体的なディレクトリ構成**, **(ii) 計算入力テンプレ（Psi4/NWChem）**, **(iii) 失敗時の自動リカバリ規則（SCF収束/虚振動数/IRC不一致）**, **(iv) Canteraに落とすための機構生成フォーマット** まで、さらに具体化して提示できます。

[1]: https://nwchemgit.github.io/index.html "https://nwchemgit.github.io/index.html"
[2]: https://github.com/psi4/psi4 "https://github.com/psi4/psi4"
[3]: https://pyscf.org/about.html "https://pyscf.org/about.html"
[4]: https://nwchemgit.github.io/TASK.html "https://nwchemgit.github.io/TASK.html"
[5]: https://nwchemgit.github.io/Geometry-Optimization.html "https://nwchemgit.github.io/Geometry-Optimization.html"
[6]: https://psicode.org/psi4manual/master/optking.html "https://psicode.org/psi4manual/master/optking.html"
[7]: https://optking.readthedocs.io/en/latest/algorithms.html "https://optking.readthedocs.io/en/latest/algorithms.html"
[8]: https://pubs.acs.org/doi/10.1021/jp3051033 "https://pubs.acs.org/doi/10.1021/jp3051033"
[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8614229/ "https://pmc.ncbi.nlm.nih.gov/articles/PMC8614229/"
[10]: https://pyscf.org/user/gpu.html "https://pyscf.org/user/gpu.html"
[11]: https://pypi.org/project/gpu4pyscf/ "https://pypi.org/project/gpu4pyscf/"
[12]: https://pubs.acs.org/doi/10.1021/acs.jctc.3c00421 "https://pubs.acs.org/doi/10.1021/acs.jctc.3c00421"
[13]: https://daltonprogram.org/ "https://daltonprogram.org/"
[14]: https://github.com/ReactionMechanismGenerator/RMG-Py/blob/main/LICENSE.txt "https://github.com/ReactionMechanismGenerator/RMG-Py/blob/main/LICENSE.txt"
[15]: https://github.com/Molcas/OpenMolcas "https://github.com/Molcas/OpenMolcas"
[16]: https://gitlab.com/dalton/dalton/-/tree/master "https://gitlab.com/dalton/dalton/-/tree/master"
[17]: https://www.faccts.de/orca/ "https://www.faccts.de/orca/"
[18]: https://github.com/grimme-lab/xtb "https://github.com/grimme-lab/xtb"
[19]: https://github.com/grimme-lab/xtb/releases "https://github.com/grimme-lab/xtb/releases"
[20]: https://github.com/crest-lab/crest "https://github.com/crest-lab/crest"
[21]: https://onlinelibrary.wiley.com/doi/full/10.1002/qua.26390 "https://onlinelibrary.wiley.com/doi/full/10.1002/qua.26390"
[22]: https://pysisyphus.readthedocs.io/en/latest/tsoptimization.html "https://pysisyphus.readthedocs.io/en/latest/tsoptimization.html"
[23]: https://github.com/eljost/pysisyphus "https://github.com/eljost/pysisyphus"
[24]: https://ase-lib.org/development/licenseinfo.html "https://ase-lib.org/development/licenseinfo.html"
[25]: https://pubs.acs.org/doi/abs/10.1021/acs.jctc.2c00395 "https://pubs.acs.org/doi/abs/10.1021/acs.jctc.2c00395"
[26]: https://github.com/HarryStevens/geometric/blob/master/LICENSE "https://github.com/HarryStevens/geometric/blob/master/LICENSE"
[27]: https://github.com/duartegroup/autodE "https://github.com/duartegroup/autodE"
[28]: https://github.com/ReactionMechanismGenerator/RMG-Py/blob/main/arkane/README.md "https://github.com/ReactionMechanismGenerator/RMG-Py/blob/main/arkane/README.md"
[29]: https://github.com/Auto-Mech/MESS "https://github.com/Auto-Mech/MESS"
[30]: https://docs.qcarchive.molssi.org/overview/index.html "https://docs.qcarchive.molssi.org/overview/index.html"
[31]: https://github.com/MolSSI/QCFractal "https://github.com/MolSSI/QCFractal"
[32]: https://github.com/MolSSI/QCEngine "https://github.com/MolSSI/QCEngine"
[33]: https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html "https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html"
[34]: https://wiki.fysik.dtu.dk/~askhl/ase-doc/index.html "https://wiki.fysik.dtu.dk/~askhl/ase-doc/index.html"
