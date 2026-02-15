以下は、これまでの議論（FNO/DeepONet、PINO、非局所電子運動論、シース境界、形状入力、SDF、境界材料物性など）を踏まえて、**「定常のプラズマ空間分布（多物理量）」を目的変数**、**入力＝プロセス条件＋装置形状/寸法＋境界材料物性**として、**学習データをできるだけ少なく**、かつ**高精度・高速推論**を狙うためのサロゲートモデル構築案を“実装に落ちる粒度”まで具体化します。

---

# 1. 問題の定義（この設計案が狙う“写像”）

## 1.1 目標の写像

* 入力

  * **プロセス条件**：圧力、流量（各ガス）、RFパワー（複数源なら各源）、周波数、位相、Duty、コイル電流、バイアス、温度条件など（ベクトル）
  * **装置形状/寸法**：パーツ寸法・位置・形状パラメータ（ベクトル）＋形状場表現（後述）
  * **境界材料物性**：誘電率 εr、導電率 σ、透磁率 μr、場合によりSEE係数、表面再結合係数など（空間場としても入力可能）
* 出力（定常の空間分布）

  * **電子密度** (n_e(\mathbf{x}))（logスケール推奨）
  * **イオン種ごとの密度** (n_{i,k}(\mathbf{x}))（log）
  * **電子温度** (T_e(\mathbf{x}))（準線形〜場合によりlog）
  * **イオン温度** (T_{i,k}(\mathbf{x}))
  * **電位/電場** (\phi(\mathbf{x}), \mathbf{E}(\mathbf{x}))
  * **磁場** (\mathbf{B}(\mathbf{x}))（必要なら）

COMSOLのPlasma Moduleの基本は、電子密度と電子エネルギー密度等を（反応を含む）輸送方程式で解き、Poisson方程式と自己無撞着に結合します。したがって、**サロゲート側も“密度＋エネルギー（または温度）＋電位”を一貫して出す**と、後段の物理制約や同定がやりやすいです。 ([COMSOL Documentation][1])

---

# 2. データ表現と前処理（4カテゴリ共通の“勝ち筋”）

ここが最重要です。**ネットワークの種類より先に、形状と境界・スケーリングを正しく食わせる**と、データが少なくても当たりやすくなります。

---

## 2.1 形状/寸法の特徴量化（最も実務的な3層構造）

### 推奨：形状を「ベクトル＋空間場」に分離

1. **寸法パラメータ（ベクトル）**

   * 例：外径、ギャップ、コイル位置、シャワーヘッド穴配置パラメータ、誘電体厚み、ウェハ台高さ等
   * 目的：設計最適化・感度解析・同定に直結（微分可能）

2. **SDF（Signed Distance Function）場（グリッド上のスカラー）**

   * 各格子点で「境界までの符号付き距離」
   * 形状変化を連続量として表せるので学習が安定しやすい
   * 可変形状のNeural Operator系でもSDF利用が中核になっている（例：GINOはSDFと点群形状表現を使う）。 ([arXiv][2])
   * “SDF vs バイナリマスク”の比較評価も近年増えており、SDFの定義（内外で符号、境界で0）も明確に整理されている。 ([Nature][3])

3. **境界タイプ/材料物性マップ（多チャネル）**

   * 例：

     * 物体/空間マスク（プラズマ領域）
     * 境界種別（電極/誘電体/壁/入口/出口/ウェハ/コイル等）
     * **εr(x), σ(x), μr(x)**（必要なら温度依存も）
   * これを入れることで「境界で急変する」現象（シース、誘電体表面電荷、EM境界条件）が学習しやすい

---

## 2.2 プロセス条件の特徴量化（“派生量”が効く）

生の条件だけでなく、**派生量（物理的に意味のある無次元・スケール量）**が効きます。

* **中性粒子数密度** (N \approx p/(k_B T_g))
* **パワー密度**（例：総投入電力/チャンバー体積、局所ならコイル近傍の体積で規格化）
* **ガス組成の比率**（流量比、分圧比）
* **周波数・スキン深さ関連の無次元量**（ICP/マイクロ波で特に効く）
* **磁化度の指標**（(\Omega_c/\nu)など：磁場あり装置で）

これらは“確定値として入力”にしてよいです（出力を通じて学習させるよりデータ効率が高いことが多い）。

---

## 2.3 出力のスケーリング（ログ密度・境界急峻に対応）

### 密度（logが基本）

* ( \tilde{n} = \log_{10}(n + n_{\text{floor}})) を推奨

  * (n_{\text{floor}}) は数値安定用（シミュレーションの数値床に合わせる）
* 学習は (\tilde{n}) のL1/Huber + 相対誤差系が安定しやすい
* 推論後に逆変換して物理量に戻す

### 温度（準線形だが“正値”と“境界”が要注意）

* (T) は線形でもよいが、**softplusで正値制約**するか、logで学習して線形で評価する設計も有効
* Teは条件で広がる場合があるので、z-score（平均0分散1）＋Huberが無難

### 電場・磁場（符号あり・スケール差あり）

* **Eは直接出さず φを出して E = −∇φ**（静電近似が成り立つ範囲）
  → これだけで境界付近の破綻が大幅に減りやすい（“ハード制約”）
* **Bは A を出して B = ∇×A**で **∇·B=0を自動満足**（必要なら）

---

## 2.4 “境界で急激に変化”への学習側の対策（必須）

シース/壁近傍・誘電体界面・コイル近傍のEMなど、急峻な領域が当たらないと意味が薄いので、以下を組み合わせるのが定番です。

1. **境界距離（SDF）で重み付け損失**

   * 例：境界から距離 d が小さいほど重み↑（ただしやり過ぎるとバルクが崩れるので上限クリップ）

2. **境界近傍のパッチ学習（サンプリングを偏らせる）**

   * 全領域を等確率サンプル＋境界近傍を追加サンプル、の混合が安定

3. **勾配損失（Gradient loss）**

   * (|\nabla \hat{u}-\nabla u|) を薄く入れる
   * 特にφ→Eにする場合、φの勾配品質が効く

4. **マルチスケール（粗→細）**

   * まず粗解像で全体を合わせ、後から高解像で境界を締める
   * PINOが「粗データ＋高解像PDE制約」で強いのはこの思想。 ([arXiv][4])

---

# 3. 物理制約ライブラリ（用途ごとにON/OFFできる設計）

**“全部入れると学習が死ぬ”**ので、用途別にON/OFFできるよう、制約を階層化します。

---

## 3.1 物理制約の階層（おすすめ）

* **Level 0（データ駆動）**：データ損失のみ
* **Level 1（壊れやすい関係だけハード化）**：正値、E=-∇φ、B=∇×A、領域マスク
* **Level 2（流体モデル相当のPDE残差）**：Poisson、連続式、電子エネルギー式、準中性（バルクのみ）、境界フラックス条件
* **Level 3（電子運動論/非局所）**：EEDF/FP近似、非局所導電率など（低圧ICP/ECR向け）

COMSOLのドリフト拡散系（電子密度＋電子エネルギー密度）やPoisson結合は、Level 2の制約設計に非常に合わせやすいです。 ([COMSOL][5])

また、輸送係数・反応係数をBoltzmannソルバで求める考え方（BOLSIG+）は、**“係数の整合性”を制約として入れる**のに強力です。 ([ResearchGate][6])

---

## 3.2 物理制約（候補）テーブル：何を出すなら何が入れられるか

| 制約（ON/OFF部品）    | 代表式/内容                                       | 必要な出力       | 使いどころ（想定用途）        | 注意点                                        |
| --------------- | -------------------------------------------- | ----------- | ------------------ | ------------------------------------------ |
| 正値制約            | (n>0, T>0)                                   | n,T         | ほぼ常時ON（最低限）        | log/softplusでハード化推奨                        |
| Eのハード化          | (\mathbf{E}=-\nabla \phi)                    | φ           | 静電支配（CCP/低周波）で強い   | ICP/ECRの誘導Eは別成分が必要                         |
| Bのハード化          | (\mathbf{B}=\nabla\times \mathbf{A})（→∇·B=0） | A           | 磁場分布も推定する場合        | Aのゲージは実装設計が要る                              |
| Poisson/ガウス     | (\nabla\cdot(\epsilon \nabla \phi)= -\rho) 等 | φ, n_e, n_i | 流体モデル整合、外挿耐性UP     | シース未解像なら「バルクのみ」など領域分けが安全                   |
| 準中性（バルク）        | (\sum z_i n_i \approx n_e)                   | n_e, n_i    | バルク領域を締める（少データで効く） | シース領域に強制すると逆効果                             |
| 種連続（定常）         | (\nabla\cdot \Gamma_s = R_s)                 | n_s, φ,（係数） | 要因分析・同定で効く         | 反応源項Rが不確かなときは弱く入れる                         |
| 電子エネルギー式        | エネルギー収支（加熱−損失）                               | T_e or ε_e  | Te分布の外挿耐性UP        | 係数閉じ（LFA/LMEA）が鍵                           |
| 係数整合（Boltzmann） | (\mu_e,D_e,k_r) をE/Nやεで閉じる                   | E/N or ε    | “反応テーブル一貫性”を担保     | 非局所条件では破れうるが、まず有効 ([ResearchGate][6])      |
| 非局所電子運動論（条件付き）  | 「系サイズ＜エネルギー緩和長」で非局所重要                        | EEDF系       | 低圧・大面積で電子が非局所      | Kolobovは半導体リアクタで非局所重要と指摘 ([サイエンスダイレクト][7]) |

---

# 4. 4カテゴリ別：サロゲートモデル構築案（具体構成）

以降は、**同じ入力/出力仕様を、どのネットワークで実装するか**の比較としてまとめます。

---

## 4.1 CNN / U-Net系（固定格子テーブルがあるなら最も実務的な本命）

### 推奨構成（定常場回帰向け）

* **バックボーン候補**

  * UNet++（ネストされた密スキップ＋深い監督） ([arXiv][8])
  * UNet 3+（フルスケール接続＋深い監督） ([arXiv][9])
  * Attention U-Net（境界/重要領域に注意を寄せる） ([arXiv][10])

### 条件（プロセス条件・寸法パラメータ）の入れ方

* **FiLMで条件付け**：プロセス条件ベクトル → 各ブロックの特徴量をスケール/シフト ([arXiv][11])
* **CoordConv/座標チャネル**：x,y,z（またはr,z,θ）を入力チャネルに追加して位置依存を学びやすく ([arXiv][12])
* 境界タイプを入力に入れる場合は、正規化層が情報を消しがちなので **SPADE（空間適応正規化）**の思想が刺さることがあります（境界タグ保持に有効）。 ([arXiv][13])

### 形状・材料物性の入れ方（おすすめ）

* 入力チャネル例（2DならH×W、3DならD×H×W）

  * SDF（1ch）
  * プラズマ領域マスク（1ch）
  * 境界種別One-hot（電極/誘電体/壁/入口/出口/ウェハ…）(数ch)
  * εr、σ、μr（各1ch）
  * 追加で：表面係数（SEE γ、再結合確率）を壁セルに埋める（必要なら）

### 出力の出し方（破綻しにくい）

* 直接：(\log n_e, \log n_{i,k}, T_e, T_{i,k})
* φを出して **Eは派生**（E=-∇φ）
* Bを出すなら Aを出して **Bは派生**（∇×A）

### 物理制約（U-Netでも普通に入る）

* **離散残差（有限差分/Stencil）**で、Poissonや連続式の残差を計算して損失化

  * CNN系にPDE残差を入れる“theory-guided/physics-informed”系の研究は多数あり、離散残差を損失に入れる枠組みも整理されています。 ([サイエンスダイレクト][14])

### 期待できる特性

* **強み**：境界層（シース等）を“局所多スケール”で拾いやすい、実装が安定、推論が速い
* **弱み**：Poisson/Maxwell的な“非局所結合”が強い場合に、純CNNだとデータ要求が増えたり、外挿が崩れやすい（→ハイブリッド/FNOへ）

---

## 4.2 FNO / DeepONet系（非局所結合・解像度跨ぎ・任意点出力に強い）

### (A) FNO（Fourier Neural Operator）

* 特徴：フーリエ空間でカーネルを表現し、演算子（関数→関数）を学習する枠組み ([arXiv][15])
* 代表的な利点として、ベンチマークPDEで「大幅な高速化」や「ゼロショット超解像」などが報告されています（ただしプラズマでの同等保証ではなく“性質として参考”）。 ([arXiv][15])

### (B) PINO（Physics-Informed Neural Operator）

* FNO等のオペレータ学習に **PDE制約を高解像度で課す**設計（粗データ＋細PDE） ([arXiv][4])
* 「データが少ない/粗い」ときに、**物理で補正して高忠実化**する思想が、あなたの目的（データ高コスト）に合います。 ([arXiv][4])

### (C) DeepONet

* Branch（入力関数）＋Trunk（座標）で、**出力点が格子でなくてもよい**（任意の点 y で評価可能） ([arXiv][16])
* PAPの測定ライン、ウェハ面の必要点、壁面フラックス点など「点が限られる」用途でデータ効率が上がりやすい

### (D) 形状が変わるとき：GINO / Geom-DeepONet

* **GINO**：SDF＋点群形状＋（Graph + Fourier）で可変形状PDEを扱う ([arXiv][2])
* **Geom-DeepONet**：点群＋SDF等でパラメタ化形状に対応するDeepONet派生 ([arXiv][17])
  → 装置形状が入力に入る場合、**“形状の持ち込み方”が勝負**なので、この系統は要検討です。

---

## 4.3 ハイブリッド（CNN/U-Net × FNO/DeepONet）：局所＋非局所を両取りする本命候補

プラズマは「境界層（局所）＋電位/電磁場（非局所）」の混合なので、ハイブリッドは理にかなっています。

### 有力ハイブリッドの型

1. **U-FNO**：FNOをU字系で強化し、精度・速度・データ効率を狙う ([arXiv][18])
2. **U-NO（U-shaped Neural Operator）**：より深いオペレータを可能にし、データ効率・ロバスト性を主張 ([arXiv][19])
3. **Conv-FNO**：CNNで局所特徴を抽出してからFNOでグローバルを学ぶ ([arXiv][20])
4. **HUFNO**：周期方向（FNO）と非周期方向（U-Net）を分担し、混在境界条件で精度改善を報告 ([arXiv][21])

   * 円筒チャンバーで **θ方向が近似的に周期**なら、まさにこの思想が刺さる可能性があります
5. **CNN-DeepONet派生**：BranchにCNNを入れて入力の構造を抽出する拡張例（Porous-DeepONet等） ([サイエンスダイレクト][22])

### 物理制約はどうする？

* 基本：**融合後の最終出力に対して物理残差を計算して損失化**（PINOと同じ思想でOK）
* ハイブリッドの“うまい使い方”：

  * CNN側＝境界・局所補正
  * FNO側＝ポテンシャル/電磁場など全体結合
    に自然に分担しやすいので、**Poisson系の制約が効きやすい**ことが多い

---

## 4.4 シンプルMLP（ただし“そのまま全格子出力”は非推奨）

MLP単体でH×W×Dの全場を直接出すのは、パラメータ数・データ要求ともに厳しいです。MLPは「圧縮」と組み合わせると強い。

### (A) ROM（POD/Autoencoder）＋MLP（最速推論・少データの有力解）

* 手順

  1. 出力場を **POD**や **（畳み込み）オートエンコーダ**で低次元潜在変数 z に圧縮
  2. **MLPで (プロセス条件＋寸法) → z** を回帰
  3. Decoderで場に復元
* 深層ROMの系統は、POD/AEを用いたDL-ROMや、物理情報を組み込むROMなどが整理されています。 ([mate.polimi.it][23])
* 長所：推論が極めて速い、最適化ループに向く
* 短所：境界の高周波（シース等）を落としやすい → “境界パッチAE”や“2段（バルク＋境界補正）”が必要

### (B) 座標入力のINR（Implicit Neural Representation）＋MLP（連続表現）

* MLPに(x,y,z)と条件を入れて (u(\mathbf{x})) を直接出す
* 高周波が必要なら **Fourier Features**や **SIREN**が有効 ([arXiv][24])
* ただし推論時に全格子点で評価するので、**点数が多いと推論が遅くなる**（GPUでも重い）

---

# 5. 学習データを減らすための“周辺技術”（実装に効く順）

## 5.1 対称性データ拡張（最安で効く）

* 円筒対称に近い装置なら **θ回転による増強**
* ミラー対称があるなら反転増強
* これだけで必要データが実質的に増える（条件が許す限り優先）

## 5.2 アクティブ・ラーニング（「次に回すべきシミュ」を賢く選ぶ）

PDEサロゲートはデータが高価なので、**不確実性/誤差推定に基づいて追加サンプルを選ぶ**ALが効きます。
ニューラルPDEソルバーに対するALが「データを減らす目的で有効」と明確に述べられ、ベンチマーク（AL4PDE）も提案されています。 ([OpenReview][25])

実装案（簡単な順）：

* アンサンブル（3〜5個）で分散を見る
* MC Dropout
* 物理残差の大きい条件点を優先して追加計算（PINOと相性良い）

## 5.3 マルチフィデリティ（2D/粗格子/簡略モデルで事前学習 → 3D/PICで補正）

* 例：

  * Low：2D COMSOL（定常、簡略反応）を多数
  * High：3D COMSOL または PIC/ハイブリッドを少数
* 深層オペレータでマルチフィデリティを扱う研究もあり（例：DeepONetのマルチフィデリティ）、高価なデータを節約する方向性が示されています。 ([APS Link][26])
* PINNs側でもマルチフィデリティの枠組みが提案されています。 ([サイエンスダイレクト][27])

---

# 6. 手法まとめテーブル（実装優先度・有用度つき）

> 記号：
> 実装コスト＝Low/Med/High（主にコード量・チューニング難度・計算資源）
> 有用度＝★1〜5（あなたの目的：少データ・高精度・高速推論・形状入力）

| カテゴリ / 構成案                         | 概要                      | メリット                | デメリット               | 使い分け（おすすめ条件）           | 物理制約との相性            | 実装コスト    | 主要ライブラリ例                | 参考文献                                   | 実装優先度 | 有用度   |
| ---------------------------------- | ----------------------- | ------------------- | ------------------- | ---------------------- | ------------------- | -------- | ----------------------- | -------------------------------------- | ----- | ----- |
| **MLP+ROM（AE/POD）**                | 圧縮→潜在zをMLP回帰→復元         | 最速推論、少データで強い、最適化に向く | 境界層が潰れやすい（工夫必要）     | まず高速化・形状最適化を回したい       | 〇（潜在で物理は弱め、後処理射影は◎） | Med      | PyTorch                 | DL-ROM/POD系、物理情報付きROM ([OSTI.gov][28]) | **高** | ★★★★☆ |
| **MLP(INR)+Fourier/SIREN**         | (x,y,z,条件)→u(x) を連続表現   | 任意点出力、微分が取りやすい      | 全格子推論は遅い、学習が不安定なことも | 出力点が少ない（ウェハ面・PAP線）     | 〇（残差計算しやすい）         | Med      | PyTorch                 | Fourier Features/SIREN ([arXiv][24])   | 中     | ★★★☆☆ |
| **UNet++ / UNet3+（条件付き）**          | SDF+物性+FiLMで条件付けU-Net回帰 | 安定・速い・境界に強い         | 非局所結合が強いと外挿が弱い      | 固定格子で形状変化が中程度まで        | ◎（離散残差で入れやすい）       | Med      | PyTorch                 | UNet++/UNet3+ ([arXiv][8])             | **高** | ★★★★★ |
| **Attention U-Net + SPADE/境界タグ**   | 境界/重要部に注意、タグ保持          | 境界近傍の精度が上がりやすい      | 過度に局所へ寄ると全体が崩れる     | 壁/ウェハ近傍が支配的、材料差が効く     | ◎                   | Med      | PyTorch                 | Attention U-Net/SPADE ([arXiv][10])    | 高     | ★★★★☆ |
| **Phys-guided CNN（離散PDE残差）**       | データ損失＋離散残差損失            | 少データで外挿耐性UP         | 残差定義と重み付けが難しい       | “係数同定/要因分析”までやりたい      | ◎                   | Med〜High | PyTorch                 | TgFCNN / f-PICNN ([サイエンスダイレクト][29])    | 高     | ★★★★☆ |
| **FNO（固定格子）**                      | 演算子学習で非局所を捉える           | 長距離結合/解像度跨ぎに強い      | 境界の扱い・非周期で工夫要       | Poisson/EMが効く、解像度を変えたい | 〇                   | Med      | neuraloperator/ PyTorch | FNO ([arXiv][15])                      | 中     | ★★★★☆ |
| **PINO（FNO+高解像PDE制約）**             | 粗データ＋細PDE制約（PINO思想）     | 少データ/粗データで高忠実化しやすい  | PDE/境界の設計が必要        | データ高価・外挿耐性が重要          | ◎                   | High     | PhysicsNeMo等            | PINO ([arXiv][4])                      | 中〜高   | ★★★★★ |
| **DeepONet（任意点出力）**                | Branch+Trunkで任意点に出力     | PAP/ウェハ面など“点集合”に強い  | 全格子出力は重くなる          | 観測点と結びつける/同化する         | 〇（PI-DeepONetも可）    | Med      | PyTorch                 | DeepONet ([arXiv][30])                 | 中     | ★★★★☆ |
| **GINO / Geom-DeepONet（可変形状）**     | SDF/点群で形状入力に強い          | 形状汎化の本命、メッシュ非依存     | 実装が重い               | 形状が大きく変わる設計探索          | 〇〜◎                 | High     | PyTorch+GNN             | GINO/Geom-DeepONet ([arXiv][2])        | 中     | ★★★★★ |
| **Hybrid：Conv-FNO / U-FNO / U-NO** | CNNで局所＋FNOで非局所          | 境界＋全体結合を両取り         | 構造が複雑、チューニング増       | プラズマの性質に合いやすい本命枠       | ◎                   | High     | PyTorch                 | Conv-FNO/U-FNO/U-NO ([arXiv][20])      | **高** | ★★★★★ |
| **Hybrid：HUFNO（周期×非周期）**           | 方向で役割分担（混在境界）           | 混在BCで精度向上報告         | 適用条件を選ぶ             | 円筒装置でθ周期性が使える          | ◎                   | High     | PyTorch                 | HUFNO ([arXiv][21])                    | 中     | ★★★★☆ |

---

# 7. 最後に：有用な3パターン（推奨レシピ）

ここでは「最短で回る」「少データで伸びる」「形状汎化まで見据える」の3本を提示します。

---

## パターン1：最短で強いベースライン（固定格子×形状入力あり）

**目的**：まず“プラズマ分布がそれっぽく当たる”を最速で作り、以降拡張する

* モデル：**UNet++ or UNet3+**（回帰） ([arXiv][8])
* 入力：

  * SDF、境界タグ、εr/σ/μrマップ
  * CoordConv（座標チャネル） ([arXiv][12])
  * プロセス条件はFiLMで各ブロックを条件付け ([arXiv][11])
* 出力：log密度、Te、φ（Eは派生）
* 学習：

  * Huber + 境界距離重み + 勾配損失（φのみ薄く）
* 物理制約：Level 1（正値、E=-∇φ）＋（可能なら）Poissonを“弱く”

**強い理由**：固定格子との相性が良く、実装が速い。境界急峻対策（SDF重み＋U字多スケール）が効きやすい。

---

## パターン2：データ極少・推論最速（最適化ループ向け）

**目的**：装置寸法・条件の最適化を高速に回す／ブラックボックス係数同定の土台

* モデル：**（畳み込み）Autoencoderで場を圧縮**＋**MLPで条件→潜在**＋Decoder復元 ([OSTI.gov][28])
* 入力：プロセス条件＋寸法パラメータ（ベクトル）＋必要なら“形状コード”
* 出力：潜在z → 場
* 物理制約：

  * Level 1（正値、Eはφ派生）
  * 追加で“射影”：推論後にPoissonを1回解いてφを整合（可能なら）
* 追加：

  * 境界層だけ別AE（バルクAE＋境界補正AEの2段）にすると精度が伸びやすい

**強い理由**：高価な3Dシミュのサンプルが少なくても、解空間が低次元に近いなら強烈に効く（推論も最速）。

---

## パターン3：形状汎化＋非局所結合まで狙う本命（少データ志向）

**目的**：寸法・配置が大きく変わっても外挿し、Poisson/EM的な“全体結合”も壊れにくく

* モデル候補（優先順）

  1. **Hybrid：Conv-FNO / U-FNO / U-NO**（局所＋非局所） ([arXiv][20])
  2. θ周期が使えるなら **HUFNO** ([arXiv][21])
  3. 形状変化が大きいなら **GINO**（SDF＋点群形状） ([arXiv][2])
* 物理制約：**PINO思想（粗データ＋細PDE）**で、少データでも締める ([arXiv][4])
* データ削減：

  * Active Learningで“次に回すべき条件/形状”を選ぶ ([OpenReview][25])
  * 2D/簡略モデルで事前学習→3D/PICで補正（マルチフィデリティ） ([APS Link][26])

**強い理由**：プラズマの本質（局所境界層＋全体場結合）にアーキテクチャが合う。形状汎化の伸び代が大きい。

---

## 付記：低圧で非局所電子が効くなら（物理制約の上位オプション）

Kolobovは、**系サイズがエネルギー緩和長より小さいと非局所効果が重要**になり、半導体製造の多くのリアクタで非局所を考慮すべきと述べています。 ([サイエンスダイレクト][7])
この領域では、Level 3（EEDF/非局所導電率）に踏み込む価値があり、係数閉じ（BOLSIG+）の使い方もより重要になります。 ([ResearchGate][6])

---

もし次に進めるなら、上の「パターン1〜3」のうち、あなたの現状（装置タイプ：CCP/ICP/ECR、圧力レンジ、2D/3D、Bが外部既知か、PICをどこまで参照するか）に合わせて、

* **入力チャネル構成（SDFの作り方、境界タグ設計、材料物性チャネル）**
* **出力の分解（φ/Aを出すか、E/Bを直接出すか）**
* **損失関数（境界重み、残差の式、スケジューリング）**
  を“そのまま実装できるレベル”の仕様書（擬似コード含む）に落として提示できます。

[1]: https://doc.comsol.com/6.3/doc/com.comsol.help.plasma/PlasmaModuleUsersGuide.pdf?utm_source=chatgpt.com "The Plasma Module User's Guide"
[2]: https://arxiv.org/abs/2309.00583?utm_source=chatgpt.com "Geometry-Informed Neural Operator for Large-Scale 3D PDEs"
[3]: https://www.nature.com/articles/s44172-025-00513-3?utm_source=chatgpt.com "Benchmarking scientific machine-learning approaches for ..."
[4]: https://arxiv.org/abs/2111.03794?utm_source=chatgpt.com "Physics-Informed Neural Operator for Learning Partial Differential Equations"
[5]: https://www.comsol.com/model/download/177899/models.plasma.drift_diffusion_tutorial.pdf?utm_source=chatgpt.com "Drift Diffusion Tutorial"
[6]: https://www.researchgate.net/profile/L-Pitchford-2/publication/200702750_Solving_the_Boltzmann_equation_to_obtain_electron_transport_coefficients_and_rate_coefficients_for/links/561261d308ae6b29b49e8785/Solving-the-Boltzmann-equation-to-obtain-electron-transport-coefficients-and-rate-coefficients-for.pdf?utm_source=chatgpt.com "Solving the Boltzmann equation to obtain electron transport ..."
[7]: https://www.sciencedirect.com/science/article/abs/pii/S0167931703003526?utm_source=chatgpt.com "Deterministic Boltzmann solver for electron kinetics in ..."
[8]: https://arxiv.org/abs/1807.10165?utm_source=chatgpt.com "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
[9]: https://arxiv.org/abs/2004.08790?utm_source=chatgpt.com "UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation"
[10]: https://arxiv.org/abs/1804.03999?utm_source=chatgpt.com "Attention U-Net: Learning Where to Look for the Pancreas"
[11]: https://arxiv.org/abs/1709.07871?utm_source=chatgpt.com "FiLM: Visual Reasoning with a General Conditioning Layer"
[12]: https://arxiv.org/abs/1807.03247?utm_source=chatgpt.com "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"
[13]: https://arxiv.org/abs/1903.07291?utm_source=chatgpt.com "Semantic Image Synthesis with Spatially-Adaptive Normalization"
[14]: https://www.sciencedirect.com/science/article/pii/S0021999124005321?utm_source=chatgpt.com "f-PICNN: A physics-informed convolutional neural network ..."
[15]: https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com "Fourier Neural Operator for Parametric Partial Differential ..."
[16]: https://arxiv.org/pdf/1910.03193?utm_source=chatgpt.com "DeepONet: Learning nonlinear operators for identifying ..."
[17]: https://arxiv.org/abs/2403.14788?utm_source=chatgpt.com "[2403.14788] Geom-DeepONet: A Point-cloud-based Deep ..."
[18]: https://arxiv.org/abs/2109.03697?utm_source=chatgpt.com "U-FNO -- An enhanced Fourier neural operator-based ..."
[19]: https://arxiv.org/abs/2204.11127?utm_source=chatgpt.com "U-NO: U-shaped Neural Operators"
[20]: https://arxiv.org/abs/2503.17797?utm_source=chatgpt.com "Enhancing Fourier Neural Operators with Local Spatial ..."
[21]: https://arxiv.org/abs/2504.13126?utm_source=chatgpt.com "A hybrid U-Net and Fourier neural operator framework for the large-eddy simulation of turbulent flows over periodic hills"
[22]: https://www.sciencedirect.com/science/article/pii/S2095809924003904?utm_source=chatgpt.com "Porous-DeepONet: Learning the Solution Operators of ..."
[23]: https://www.mate.polimi.it/biblioteca/add/qmox/72-2021.pdf?utm_source=chatgpt.com "enhancing deep learning-based reduced order models for ..."
[24]: https://arxiv.org/abs/2006.10739?utm_source=chatgpt.com "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
[25]: https://openreview.net/forum?id=x4ZmQaumRg&utm_source=chatgpt.com "Active Learning for Neural PDE Solvers"
[26]: https://link.aps.org/doi/10.1103/PhysRevResearch.4.023210?utm_source=chatgpt.com "Multifidelity deep neural operators for efficient learning of ..."
[27]: https://www.sciencedirect.com/science/article/abs/pii/S0021999121007397?utm_source=chatgpt.com "Multifidelity modeling for Physics-Informed Neural ..."
[28]: https://www.osti.gov/biblio/1843130?utm_source=chatgpt.com "A fast and accurate physics-informed neural network ..."
[29]: https://www.sciencedirect.com/science/article/pii/S0309170821002050?utm_source=chatgpt.com "Theory-guided full convolutional neural network"
[30]: https://arxiv.org/abs/1910.03193?utm_source=chatgpt.com "[1910.03193] DeepONet: Learning nonlinear operators for ..."
