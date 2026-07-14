以下をそのまま .md ファイルに貼り付けて使えます。

# SDF候補分子に対するHF気相反応活性化エネルギーの自動計算手法
## 0. 目的と前提
本検討の目的は、複数の候補分子、特にアミン系分子に対して、**HFをどの程度活性化できるか**を気相反応として自動評価することである。
前提は以下とする。
| 項目 | 前提 |
|---|---|
| 入力 | 複数候補分子を含むSDFファイル |
| 対象プロセス | 半導体製造向けガスプロセス |
| 反応相 | 気相反応に限定 |
| 主対象 | HFと候補分子の会合、プロトン移動、イオン対形成 |
| 比較精度 | Gaussianを用いた通常のDFT計算と同程度を目指す |
| ソフトウェア | 可能な限り無料またはオープンソース |
| 自動化 | SDFから構造生成、経路探索、DFT計算、ランキングまで自動化 |
---
## 1. 評価すべきHF活性化反応
候補分子を `M` とし、HFを1分子とする。
基本反応は以下で定義する。
```text
M(g) + HF(g)
  → M···HF
  → TS
  → [MH]+···F−
```
より詳しく書くと以下である。
```text
R_sep : M(g) + HF(g)
C_HB  : M···H–F
        HFが候補分子のN, Oなどに水素結合した錯体
TS_PT : M···H···F
        HFから候補分子側へプロトンが移動する遷移状態
P_IP  : [MH]+···F−
        プロトン化分子とフッ化物イオンの接触イオン対
```
アミン系候補の場合、特に以下の反応が重要である。
```text
R3N + HF → R3N···HF → [R3NH]+···F−
```
---
## 2. 計算すべき活性化エネルギー
HF活性化能を比較するため、少なくとも以下の2種類の障壁を出す。
### 2.1 会合錯体基準の内部障壁
```text
ΔG‡_int = G(TS_PT) − G(C_HB)
```
これは、すでに `M···HF` 錯体ができている状態から、HFがどれだけ容易に活性化されるかを示す。
HF活性化剤そのものの能力を比較する場合、この値が重要である。
---
### 2.2 分離分子基準の見かけ障壁
```text
ΔG‡_app = G(TS_PT) − {G(M) + G(HF)}
```
これは、気相中で分離した候補分子とHFから見た見かけの障壁である。
気相では会合に伴う並進エントロピー損失が大きいため、`ΔG‡_int` だけでなく `ΔG‡_app` も必ず評価する。
---
### 2.3 温度・分圧補正
半導体ガスプロセスでは、標準状態だけでなく実プロセス条件での自由エネルギー補正が重要である。
各成分について以下の補正を入れる。
```text
G_i(T, p_i) = G_i°(T) + RT ln(p_i / p°)
```
ここで、
| 記号 | 意味 |
|---|---|
| `T` | プロセス温度 |
| `p_i` | 成分 `i` の分圧 |
| `p°` | 標準圧力 |
| `R` | 気体定数 |
---
## 3. SDFファイルを前提にした全体ワークフロー
全体の自動化フローは以下とする。
```text
candidates.sdf
  ↓
RDKit
  - SDF読込
  - H付加
  - 電荷・スピン確認
  - 受容原子の抽出
  ↓
RDKit / CREST / xTB
  - 候補分子Mの配座探索
  ↓
HF初期配置生成
  - N, Oなどの受容原子にHFを配置
  - 複数方向・複数配座を生成
  ↓
xTBまたはr2SCAN-3c
  - M···HF錯体を高速最適化
  ↓
生成物guess作成
  - [MH]+···F− 構造を生成
  ↓
反応経路探索
  - relaxed scan
  - NEB
  - CI-NEB
  - growing string
  ↓
DFT TS最適化
  - OptTS
  - Frequency
  ↓
IRC計算
  - TSが正しい反応物・生成物を接続するか確認
  ↓
高精度single point
  - Gaussian同等レベルへ精度向上
  ↓
熱補正・分圧補正
  - GoodVibes
  - Arkane
  - 自作TST補正
  ↓
ランキング
  - ΔG‡
  - HF結合長
  - HF振動数シフト
  - 電荷
  - 結合次数
```
---
## 4. 推奨ソフトウェア構成
### 4.1 完全無料・オープンソース寄り構成
商用利用も考慮して、できるだけライセンスリスクを避ける場合は以下が候補になる。
```text
RDKit
+ CREST / xTB
+ ASE
+ Sella
+ autodE
+ Psi4
+ PySCF
+ NWChem
+ GoodVibes
+ cclib
+ Snakemake
```
| 目的 | 推奨ツール | 備考 |
|---|---|---|
| SDF読込・構造処理 | RDKit | SDF処理、H付加、配座生成に有用 |
| ファイル変換 | Open Babel | xyz, sdf, mol2変換など |
| 初期配座探索 | CREST / xTB | 高速な配座探索 |
| HF配置生成 | RDKit + 自作Python | サイトごとにHFを自動配置 |
| NEB経路探索 | ASE | NEBをPythonから制御可能 |
| TS探索 | Sella / autodE | TS guess生成や最適化に利用 |
| DFT計算 | Psi4 | オープンソース量子化学コード |
| DFT計算 | PySCF | Pythonベースで自動化しやすい |
| DFT計算 | NWChem | HPC向けオープンソース量子化学コード |
| 出力解析 | cclib | 複数QCコードの出力をparse可能 |
| 熱補正 | GoodVibes | 準調和補正に有用 |
| 反応速度 | Arkane | TST速度定数や熱力学量を計算可能 |
| workflow管理 | Snakemake | 多数候補のジョブ管理に有用 |
---
### 4.2 ORCA利用可能な場合の実用構成
ORCAを利用できる場合は、実務上は以下が非常に強い。
```text
RDKit
+ CREST / xTB
+ ORCA
+ GoodVibes
+ cclib
+ Snakemake
```
ORCAは学術利用では無償で使えるが、企業利用ではライセンス確認が必要である。
ORCAを使える場合、以下の理由で有利である。
| 項目 | 利点 |
|---|---|
| DFT最適化 | 安定している |
| TS探索 | `OptTS` が使いやすい |
| 振動解析 | 虚振動確認が容易 |
| IRC | TS検証が可能 |
| DLPNO-CCSD(T) | 高精度single pointに使える |
| 3c method | r2SCAN-3c, ωB97X-3cなどが使える |
---
## 5. 計算レベルの推奨
Gaussianと同程度の精度を狙う場合、重要なのはGaussianを使うこと自体ではなく、以下を揃えることである。
```text
- 汎関数
- 基底関数
- 分散補正
- SCF収束条件
- 積分グリッド
- 振動解析
- 熱補正
- TS検証方法
```
HF、F−、水素結合、接触イオン対を扱うため、**拡散関数つき基底関数**を使うことが重要である。
---
### 5.1 スクリーニング用
| 用途 | 推奨レベル |
|---|---|
| 初期配座探索 | GFN2-xTB |
| HF会合錯体探索 | GFN2-xTB |
| 粗い反応経路探索 | GFN2-xTB, r2SCAN-3c |
| 粗いTS探索 | r2SCAN-3c |
---
### 5.2 DFT最適化用
| 用途 | 推奨レベル |
|---|---|
| 会合錯体最適化 | r2SCAN-3c |
| 生成物最適化 | r2SCAN-3c |
| TS最適化 | r2SCAN-3c |
| 振動解析 | r2SCAN-3c |
| IRC | r2SCAN-3c |
r2SCAN-3cは、分散補正や基底関数重なり誤差補正を含む実用的なcomposite DFT法であり、大量スクリーニングに適している。
---
### 5.3 高精度single point用
上位候補については、構造最適化後に高精度single pointを行う。
推奨候補は以下である。
| レベル | 用途 |
|---|---|
| ωB97X-D/def2-TZVPD | 水素結合・イオン対の高精度評価 |
| ωB97X-D4/def2-TZVPPD | 分散補正つき高精度評価 |
| ωB97M-V/def2-TZVPD | 非共有結合・反応障壁評価に有用 |
| revDSD-PBEP86-D4/def2-TZVPPD | 高精度DFT benchmark用 |
| DLPNO-CCSD(T)/def2-TZVPP | 小規模benchmark用 |
---
### 5.4 最終的な推奨計算階層
```text
Level 0:
  RDKit / MMFF
  構造確認、初期配座生成
Level 1:
  CREST / GFN2-xTB
  配座探索、HF配置スクリーニング
Level 2:
  r2SCAN-3c
  錯体、生成物、TS、IRC
Level 3:
  ωB97X-D4/def2-TZVPPD
  または
  ωB97M-V/def2-TZVPD
  高精度single point
Level 4:
  DLPNO-CCSD(T)/def2-TZVPP
  subset benchmark
```
---
## 6. HF初期配置の自動生成
SDFから読み込んだ候補分子に対して、以下の原子をHF受容サイトとして抽出する。
| サイト | 例 | 優先度 |
|---|---|---:|
| 脂肪族アミンN | RNH2, R2NH, R3N | 高 |
| ピリジン型N | pyridine, imine | 高 |
| アニリン型N | aniline | 中 |
| エーテルO | ROR | 中 |
| アルコールO | ROH | 中 |
| カルボニルO | ketone, ester, amide | 中〜低 |
| S原子 | thioetherなど | 低〜中 |
| π面 | 芳香環 | 低 |
HFは以下の向きに置く。
```text
X···H–F
```
ここで `X` は候補分子のNやOなどの受容原子である。
初期配置の目安は以下とする。
| パラメータ | 初期値 |
|---|---:|
| X···H距離 | 1.5〜1.8 Å |
| H–F距離 | 0.92〜0.95 Å |
| X···H–F角 | 150〜180° |
| HF回転方向 | 複数方向を生成 |
| 1サイトあたり配置数 | 10〜50個程度 |
---
## 7. SDFから候補構造を生成するPython骨格
```python
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
SDF = "candidates.sdf"
OUT = Path("jobs")
OUT.mkdir(exist_ok=True)
def prepare_mol(mol, mol_id):
    """
    SDFから読んだ分子にHを付加し、3D配座がなければ生成する。
    """
    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 20260508
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=50,
            params=params
        )
        AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            maxIters=500
        )
    return mol
def find_acceptor_atoms(mol):
    """
    HFのHを受け取れる可能性がある原子を抽出する。
    初期実装ではN, Oを対象とする。
    実運用ではSMARTSでアミドN、四級アンモニウムなどを除外する。
    """
    acceptors = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        if atomic_num in (7, 8) and formal_charge <= 0:
            acceptors.append(atom.GetIdx())
    return acceptors
supplier = Chem.SDMolSupplier(SDF, removeHs=False)
for i, mol in enumerate(supplier):
    if mol is None:
        continue
    if mol.HasProp("_Name"):
        name = mol.GetProp("_Name")
    else:
        name = f"mol_{i:04d}"
    mol = prepare_mol(mol, name)
    acceptors = find_acceptor_atoms(mol)
    mol_dir = OUT / name
    mol_dir.mkdir(exist_ok=True)
    for conf in mol.GetConformers():
        conf_id = conf.GetId()
        conf_dir = mol_dir / f"conf_{conf_id:03d}"
        conf_dir.mkdir(exist_ok=True)
        for atom_idx in acceptors:
            site_dir = conf_dir / f"site_{atom_idx:03d}"
            site_dir.mkdir(exist_ok=True)
            # ここで以下を実装する：
            # 1. 分子座標をxyzに書き出す
            # 2. acceptor atom周りにHFを複数方向で配置する
            # 3. complex_guess.xyzを作る
            # 4. proton_transfer_product_guess.xyzを作る
            # 5. xTB / DFT計算用入力ファイルを生成する
```
---
## 8. 推奨ジョブディレクトリ構成
候補数が多い場合、以下のようにディレクトリを分けると管理しやすい。
```text
jobs/
  mol_0001/
    conf_000/
      site_005_N/
        00_input/
          complex_guess.xyz
          product_guess.xyz
        01_xtb_complex_opt/
        02_xtb_product_opt/
        03_scan_or_neb/
        04_dft_complex_opt/
        05_dft_product_opt/
        06_ts_opt/
        07_freq/
        08_irc/
        09_high_level_sp/
        result.json
    conf_001/
      site_005_N/
        ...
  mol_0002/
    ...
```
---
## 9. 反応経路探索の方法
### 9.1 方法A：relaxed scan
最も堅牢でデバッグしやすい方法である。
反応座標は以下で定義する。
```text
q = r(H–F) − r(X–H)
```
ここで、
| 記号 | 意味 |
|---|---|
| `r(H–F)` | HF結合距離 |
| `r(X–H)` | 候補分子の受容原子XとHの距離 |
`q` の値により以下の状態を表す。
```text
q < 0 : M···H–F
q = 0 : HがXとFの中間付近
q > 0 : [MH]+···F−
```
scanのイメージは以下である。
```text
for q in reaction_coordinate:
    constrain q
    optimize all other coordinates
    store energy and geometry
```
エネルギー最大付近の構造をTS初期構造として使う。
---
### 9.2 方法B：NEB / CI-NEB
反応物と生成物の両端構造を用意できる場合、NEBを使う。
```text
Reactant:
  M···HF
Product:
  [MH]+···F−
Path:
  Reactant → image_1 → image_2 → ... → Product
```
NEBで得られた最大エネルギーimageをTS初期構造としてDFTのTS最適化に渡す。
---
### 9.3 方法C：growing string / freezing string
より複雑な反応経路ではstring法が有効である。
候補分子が柔らかく、HFの向きや分子内回転が大きく変わる場合は、1次元scanよりNEBやstring法の方がよい。
---
## 10. ORCA入力例
### 10.1 r2SCAN-3cによる会合錯体最適化
```orca
! r2SCAN-3c TightSCF Opt Freq
%pal
  nprocs 16
end
* xyzfile 0 1 complex_guess.xyz
```
---
### 10.2 r2SCAN-3cによるTS最適化
```orca
! r2SCAN-3c TightSCF OptTS Freq
%pal
  nprocs 16
end
%geom
  Calc_Hess true
end
* xyzfile 0 1 ts_guess.xyz
```
---
### 10.3 IRC計算
```orca
! r2SCAN-3c TightSCF IRC
%pal
  nprocs 16
end
* xyzfile 0 1 ts_optimized.xyz
```
---
### 10.4 高精度single point例
```orca
! wB97X-D4 def2-TZVPPD TightSCF
%pal
  nprocs 16
end
* xyzfile 0 1 ts_optimized.xyz
```
---
## 11. Psi4入力例
Psi4を使う場合の概念例を示す。
```python
import psi4
psi4.set_memory("16 GB")
psi4.set_num_threads(16)
mol = psi4.geometry("""
0 1
N      0.000000   0.000000   0.000000
H      0.000000   0.000000   1.020000
H      0.960000   0.000000  -0.340000
H     -0.480000   0.830000  -0.340000
H      0.000000   0.000000   2.600000
F      0.000000   0.000000   3.520000
""")
psi4.set_options({
    "basis": "def2-tzvpd",
    "scf_type": "df",
    "dft_spherical_points": 590,
    "dft_radial_points": 99,
    "e_convergence": 1e-8,
    "d_convergence": 1e-8,
})
energy = psi4.energy("wb97x-d", molecule=mol)
print(energy)
```
---
## 12. PySCF入力例
PySCFを使う場合のsingle point概念例を示す。
```python
from pyscf import gto, dft
mol = gto.M(
    atom="""
    N      0.000000   0.000000   0.000000
    H      0.000000   0.000000   1.020000
    H      0.960000   0.000000  -0.340000
    H     -0.480000   0.830000  -0.340000
    H      0.000000   0.000000   2.600000
    F      0.000000   0.000000   3.520000
    """,
    basis="def2-tzvpd",
    charge=0,
    spin=0,
)
mf = dft.RKS(mol)
mf.xc = "wb97x-d"
mf.grids.level = 5
energy = mf.kernel()
print(energy)
```
---
## 13. 自動判定すべき計算結果
各候補分子について、以下を `result.json` に保存する。
```json
{
  "mol_id": "mol_0001",
  "site": "N_005",
  "method_opt": "r2SCAN-3c",
  "method_sp": "wB97X-D4/def2-TZVPPD",
  "status": "normal",
  "charge": 0,
  "multiplicity": 1,
  "delta_g_bind_kcal_mol": -4.2,
  "delta_g_act_internal_kcal_mol": 2.8,
  "delta_g_act_apparent_kcal_mol": 12.5,
  "delta_g_rxn_kcal_mol": -1.1,
  "r_hf_complex_angstrom": 0.98,
  "r_hf_ts_angstrom": 1.25,
  "r_xh_ts_angstrom": 1.20,
  "hf_stretch_shift_cm-1": -450,
  "imag_freq_cm-1": -950,
  "num_imag_freq": 1,
  "irc_status": "connected",
  "scf_converged": true,
  "opt_converged": true
}
```
---
## 14. ランキング指標
HF活性化剤として候補を比較する場合、以下の指標を使う。
| 指標 | 意味 | 望ましい方向 |
|---|---|---|
| `ΔG‡_int` | 会合錯体からの活性化自由エネルギー | 小さい |
| `ΔG‡_app` | 分離分子からの見かけ活性化自由エネルギー | 小さい |
| `ΔG_bind` | HFとの会合自由エネルギー | 適度に負 |
| `ΔG_rxn` | イオン対生成自由エネルギー | 負または小さい正 |
| `r_HF` | 錯体中HF結合長 | 長い |
| `Δr_HF` | 孤立HFからの伸長 | 大きい |
| `ν_HF` | HF伸縮振動数 | 低い |
| `Δν_HF` | HF伸縮の赤方シフト | 大きい |
| `q_F` | F原子の負電荷 | より負 |
| `BO_HF` | H–F結合次数 | 小さい |
| `imag_freq` | TSの虚振動数 | 1つだけ |
| `IRC_status` | IRC接続確認 | connected |
---
## 15. TS判定基準
TSとして採用する条件は以下である。
```text
1. 最適化が収束している
2. 虚振動数が1つだけである
3. 虚振動モードがH移動方向に対応している
4. IRC計算で片側がM···HFに戻る
5. IRC計算でもう片側が[MH]+···F−に進む
6. 同一サイト・同一配座内で最も低いTSである
```
単に虚振動数が1つあるだけでは不十分である。
HFの回転、分子内配座変化、水素結合組み替えなどがTSとして見つかることがあるため、IRCによる接続確認を必須にする。
---
## 16. barrierlessの場合の扱い
強いアミンでは、以下の反応が実質的にバリアレスになる可能性がある。
```text
R3N···HF → [R3NH]+···F−
```
この場合、無理にTSを探すのではなく、以下の指標で評価する。
| 指標 | 意味 |
|---|---|
| `ΔG_bind` | HF会合しやすさ |
| `ΔG_rxn` | イオン対形成しやすさ |
| `r_HF` | HF結合の伸長 |
| `ν_HF` | HF伸縮振動数の赤方シフト |
| `q_F` | Fの負電荷化 |
| `BO_HF` | H–F結合次数低下 |
`status` は以下のように分類する。
```text
normal       : TSあり
barrierless  : TSなし、単調に生成物へ進む
no_product   : イオン対が安定化しない
failed       : 計算失敗
ambiguous    : IRC接続が不明
```
---
## 17. GoodVibesによる熱補正
低振動数モードを含む会合錯体では、通常のRRHO近似によりエントロピーが過大評価されることがある。
そのため、GoodVibesなどで準調和補正を行う。
概念的には以下の流れとする。
```text
DFT Opt + Freq
  ↓
GoodVibes
  ↓
quasi-RRHO補正済みG
  ↓
ΔG‡, ΔG_bind, ΔG_rxnを再計算
```
出力値は以下の両方を保存する。
```text
G_RRHO
G_qRRHO
```
ランキングには基本的に `G_qRRHO` を使う。
---
## 18. TST速度定数
必要であれば、Eyring式で速度定数を見積もる。
```text
k(T) = (k_B T / h) exp(−ΔG‡ / RT)
```
ここで、
| 記号 | 意味 |
|---|---|
| `k_B` | Boltzmann定数 |
| `h` | Planck定数 |
| `R` | 気体定数 |
| `T` | 温度 |
| `ΔG‡` | 活性化自由エネルギー |
プロセス温度を複数設定し、以下のようなテーブルを作る。
| mol_id | site | ΔG‡_int | ΔG‡_app | k_300K | k_400K | k_500K |
|---|---:|---:|---:|---:|---:|---:|
| mol_0001 | N_005 | 2.8 | 12.5 | ... | ... | ... |
| mol_0002 | N_003 | 5.1 | 14.8 | ... | ... | ... |
---
## 19. Snakemakeによる自動化イメージ
多数候補を扱う場合、Snakemakeで各段階を分離する。
```python
rule all:
    input:
        "summary/ranking.csv"
rule prepare:
    input:
        "candidates.sdf"
    output:
        directory("jobs")
    shell:
        "python scripts/prepare_from_sdf.py {input} jobs"
rule xtb_complex:
    input:
        "jobs/{mol}/{conf}/{site}/complex_guess.xyz"
    output:
        "jobs/{mol}/{conf}/{site}/01_xtb_complex_opt/xtbopt.xyz"
    shell:
        """
        mkdir -p jobs/{wildcards.mol}/{wildcards.conf}/{wildcards.site}/01_xtb_complex_opt
        cd jobs/{wildcards.mol}/{wildcards.conf}/{wildcards.site}/01_xtb_complex_opt
        xtb ../complex_guess.xyz --opt > xtb.out
        """
rule dft_ts:
    input:
        "jobs/{mol}/{conf}/{site}/03_scan_or_neb/ts_guess.xyz"
    output:
        "jobs/{mol}/{conf}/{site}/06_ts_opt/ts.out"
    shell:
        """
        mkdir -p jobs/{wildcards.mol}/{wildcards.conf}/{wildcards.site}/06_ts_opt
        python scripts/write_orca_ts_input.py {input} > jobs/{wildcards.mol}/{wildcards.conf}/{wildcards.site}/06_ts_opt/ts.inp
        cd jobs/{wildcards.mol}/{wildcards.conf}/{wildcards.site}/06_ts_opt
        orca ts.inp > ts.out
        """
rule summarize:
    input:
        directory("jobs")
    output:
        "summary/ranking.csv"
    shell:
        "python scripts/summarize_results.py jobs {output}"
```
---
## 20. MLポテンシャル・最新手法の位置づけ
近年は、反応経路探索に機械学習ポテンシャルを使う方法が増えている。
ただし、HF、F−、接触イオン対、プロトン移動は外挿リスクが高いため、最終値をMLだけで決めるのは危険である。
MLポテンシャルは以下に限定して使うのが安全である。
```text
1. HF会合体の初期構造探索
2. NEB/string経路の初期guess生成
3. TS guessの候補順位付け
4. DFTで失敗した経路の再初期化
5. 大量候補の粗いスクリーニング
```
最終的な障壁は必ず以下で確認する。
```text
DFT OptTS
+ Frequency
+ IRC
+ 高精度single point
```
---
## 21. 推奨する実運用フロー
### 21.1 Screening workflow
多数候補を高速に順位付けする段階。
```text
SDF
  ↓
RDKit
  - sanitize
  - H付加
  - 電荷確認
  - 受容サイト抽出
  ↓
CREST / GFN2-xTB
  - 候補分子の配座探索
  ↓
HF自動配置
  - 各N/Oサイト
  - 各配座
  - 複数HF方向
  ↓
xTB
  - M···HF錯体最適化
  - 重複除去
  ↓
r2SCAN-3c
  - 錯体最適化
  - 生成物最適化
  ↓
relaxed scan / NEB
  - TS guess生成
  ↓
r2SCAN-3c OptTS/Freq/IRC
  ↓
GoodVibes
  - qRRHO補正
  ↓
ranking.csv
```
---
### 21.2 High-accuracy workflow
上位候補だけを高精度化する段階。
```text
上位10〜30候補
  ↓
ωB97X-D4/def2-TZVPPD single point
  または
ωB97M-V/def2-TZVPD single point
  ↓
小分子subsetでDLPNO-CCSD(T) benchmark
  ↓
GoodVibesで熱補正再評価
  ↓
温度・分圧補正
  ↓
final_ranking.csv
```
---
## 22. 最初に検証すべき小分子セット
本番のSDF候補に入る前に、以下の小分子でワークフローを検証する。
```text
NH3
methylamine
dimethylamine
trimethylamine
pyridine
aniline
H2O
methanol
isopropanol
```
このセットにより、以下を確認できる。
| 分子 | 確認できること |
|---|---|
| NH3 | 最小アミンモデル |
| methylamine | 一級アミン |
| dimethylamine | 二級アミン |
| trimethylamine | 三級アミン |
| pyridine | 芳香族N |
| aniline | 共役により塩基性が弱いN |
| H2O | O系添加剤の基準 |
| methanol | アルコールO |
| isopropanol | 半導体プロセス関連のO系添加剤 |
---
## 23. 最終出力ファイル
最終的には以下のファイルを出力する。
```text
summary/
  ranking.csv
  ranking.xlsx
  failed_jobs.csv
  barrierless_cases.csv
  method_summary.json
  structures/
    best_complexes/
    best_ts/
    best_products/
```
---
## 24. ranking.csvの列案
```text
mol_id
mol_name
site
site_atom_type
conf_id
status
method_opt
method_sp
charge
multiplicity
G_M
G_HF
G_complex
G_TS
G_product
delta_G_bind
delta_G_act_internal
delta_G_act_apparent
delta_G_rxn
delta_E_act_internal
delta_E_act_apparent
r_HF_free
r_HF_complex
r_HF_TS
r_XH_TS
delta_r_HF
nu_HF_free
nu_HF_complex
delta_nu_HF
charge_H
charge_F
bond_order_HF
num_imag_freq
imag_freq
irc_status
scf_converged
opt_converged
comments
```
---
## 25. 実行優先順位
最初から完全自動化を狙うのではなく、以下の順に構築する。
```text
Step 1:
  RDKitでSDFを読み、N/Oサイトを抽出する
Step 2:
  各サイトにHFを自動配置する
Step 3:
  xTBでM···HF錯体を最適化する
Step 4:
  代表10分子でr2SCAN-3c最適化を行う
Step 5:
  relaxed scanでTS guessを作る
Step 6:
  DFT OptTS/Freq/IRCを自動実行する
Step 7:
  result.jsonを生成する
Step 8:
  ranking.csvを作る
Step 9:
  上位候補だけ高精度single pointを行う
Step 10:
  DLPNO-CCSD(T)または高精度DFTでbenchmarkする
```
---
## 26. 結論
SDFで与えられた複数候補分子に対してHF気相活性化能を評価するには、以下の多段階ワークフローが最も現実的である。
```text
RDKit
→ CREST/xTB
→ HF会合体自動生成
→ xTB/r2SCAN-3c最適化
→ scanまたはNEBでTS guess生成
→ DFT OptTS/Freq/IRC
→ 高精度single point
→ GoodVibes熱補正
→ 温度・分圧補正
→ ランキング
```
Gaussian同等精度を目指す場合、最終評価は以下のいずれかにする。
```text
r2SCAN-3c Opt/Freq
+ ωB97X-D4/def2-TZVPPD single point
または
r2SCAN-3c Opt/Freq
+ ωB97M-V/def2-TZVPD single point
さらに必要に応じて
DLPNO-CCSD(T)/def2-TZVPP benchmark
```
HF活性化の比較では、単一の `ΔG‡` だけでなく、以下を併用するべきである。
```text
ΔG‡_int
ΔG‡_app
ΔG_bind
ΔG_rxn
HF結合長伸長
HF伸縮振動数赤方シフト
H–F結合次数低下
F原子の負電荷化
IRC接続確認
```
特に重要なのは、TS候補について以下を必ず確認することである。
```text
虚振動数が1つだけ
かつ
IRCで M···HF と [MH]+···F− を接続する
```
この条件を満たしたものだけを、HF活性化エネルギーとして採用する。
---
## 27. 参考リンク
### 構造処理・配座探索
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Open Babel](https://openbabel.org/)
- [xTB Documentation](https://xtb-docs.readthedocs.io/)
- [CREST Documentation](https://crest-lab.github.io/crest-docs/)
### 量子化学計算
- [ORCA](https://www.faccts.de/orca/)
- [ORCA Manual](https://www.faccts.de/docs/orca/)
- [Psi4](https://psicode.org/)
- [PySCF](https://pyscf.org/)
- [NWChem](https://nwchemgit.github.io/)
### 経路探索・自動化
- [ASE NEB](https://wiki.fysik.dtu.dk/ase/ase/neb.html)
- [Sella](https://github.com/zadorlab/sella)
- [autodE](https://duartegroup.github.io/autodE/)
- [ASH](https://ash.readthedocs.io/)
- [Snakemake](https://snakemake.readthedocs.io/)
### 熱補正・反応速度
- [GoodVibes](https://github.com/patonlab/GoodVibes)
- [Arkane / RMG-Py](https://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/introduction.html)
- [cclib](https://cclib.github.io/)
### HF気相反応・半導体プロセス関連
- [Gas-phase etching mechanism of silicon oxide by HF/NH4F](https://pubs.aip.org/avs/jva/article/41/3/032604/2886869/Gas-phase-etching-mechanism-of-silicon-oxide-by-a)
- [Hydrogen-bond-mediated activation of HF in SiO2 etching](https://pubs.aip.org/avs/jva/article/44/2/023205/3379990/Hydrogen-bond-mediated-activation-of-HF-in-SiO2)