以下の方針が最も現実的です。SDFの候補分子をRDKitで読み込み、CREST/xTBで配座とHF会合体を大量生成し、TS候補をNEB/scan/string法で作り、最終的にDFT＋振動解析＋IRCで活性化自由エネルギーを確定する、という多段ワークフローにします。Gaussianと同等精度を狙う場合、重要なのはGaussianそのものではなく、同等の汎関数・基底関数・分散補正・グリッド・熱補正・TS検証条件をそろえることです。

1. 気相HF反応として定義すべき反応経路

半導体ドライプロセスの文脈では、HF単独よりもHF···添加剤分子の水素結合錯体がHFを分極・弱化し、反応性を高める、という見方が重要です。2026年の第一原理研究でも、H₂O、IPA、アニリン、ピリジン、ジメチルアミン、トリメチルアミンなどの添加剤について、HF···additive錯体が有効なHF活性化種として議論され、N系添加剤はO系よりHF活性化効果が大きい傾向が示されています。 ￼ また、HF/NH₄F混合ガスによるSiO₂エッチング機構をDFTで調べた報告もあり、ガス相HF化学をDFTで扱う先行事例として使えます。 ￼

SDF候補分子を M、HFを1分子とすると、まずは次の最小モデルを標準化します。

R_sep : M(g) + HF(g)
C_HB  : M···H–F        水素結合錯体
TS_PT : M···H···F      プロトン移動 / HF活性化の遷移状態
P_IP  : [MH]+···F−     接触イオン対

出力すべき活性化量は2種類に分けます。

ΔG‡_int = G(TS_PT) − G(C_HB)

これは会合錯体から見た内部障壁です。HFをどれだけ活性化しやすいかを見る指標になります。

ΔG‡_app = G(TS_PT) − {G(M) + G(HF)}

これは分離した気相分子から見た見かけ障壁です。実プロセスの温度・分圧を入れる場合は、各分子の化学ポテンシャル補正、

G_i(T, p_i) = G_i°(T) + RT ln(p_i / p°)

を入れて評価します。気相では会合反応の並進エントロピー損失が大きいので、ΔG‡_int と ΔG‡_app は必ず両方出した方がよいです。

アミン系では、強塩基性分子の場合、M···HF → [MH]+···F− が実質的にバリアレスになることがあります。その場合は、無理にTSを探すのではなく、会合自由エネルギー、HF結合長の伸長、HF伸縮振動数の赤方シフト、F/Hの電荷、生成イオン対の安定性をランキング指標にします。

⸻

2. SDF複数候補を前提にした自動計算フロー

全体像

candidates.sdf
  ↓
RDKit: 構造読込、H付加、電荷・スピン確認、受容原子N/O/F/Sなどの抽出
  ↓
CREST/xTB: 候補分子Mの配座探索
  ↓
HF配置生成: 各配座・各受容原子に対して M···HF 初期構造を複数生成
  ↓
xTBまたはr2SCAN-3c: 会合錯体 C_HB の高速最適化
  ↓
生成物 P_IP guess: HをHFからN/O側へ移した [MH]+···F− を生成
  ↓
経路探索: scan / NEB / CI-NEB / growing string / autodE
  ↓
DFT TS最適化: OptTS + Frequency
  ↓
IRC: TSが目的のC_HBとP_IPを接続するか検証
  ↓
高精度single point: Gaussian同等レベルへ引き上げ
  ↓
GoodVibes/Arkane: 気相熱補正、温度・分圧補正、TST速度定数
  ↓
CSV/Parquet/SQLite: 候補ランキング

RDKitはSDFの読み込み、3D配座生成、MMFF初期最適化に使えます。RDKitには SDMolSupplier や EmbedMultipleConfs などの機能があり、SDF起点の自動前処理に向いています。 ￼ 配座探索はCREST＋GFN-xTBが実用的です。CRESTはxTB法を使った低エネルギー配座探索ワークフローで、xTB自体は半経験的量子化学パッケージです。 ￼

受容サイトの自動抽出

アミン系ならまず以下を候補サイトにします。

サイト	例	優先度
脂肪族アミンN	RNH₂, R₂NH, R₃N	高
ピリジン型N	pyridine, imine	高
アニリン型N	aniline	中：孤立電子対がπ共役で弱まる
エーテル/アルコールO	ROH, ROR	中
カルボニルO	amide, ester, ketone	中〜低
ハロゲン/π面	補助的相互作用	低

各サイトに対して、HFを X···H–F 型に配置します。初期値としては X···H を約1.5–1.8 Å、H–F を約0.92 Åに置き、X周りに数十方向回転させます。xTBまたはr2SCAN-3cで最適化し、重複構造はRMSDとエネルギーでクラスタリングします。

⸻

3. 経路探索とTS探索の具体的方法

方法A：1次元relaxed scanからTSを作る

最も堅牢でデバッグしやすい方法です。反応座標を

q = r(H–F) − r(X–H)

とし、q < 0 が M···HF、q > 0 が [MH]+···F− に対応するようにします。q を段階的に固定して、その他の座標を最適化します。エネルギー最大付近の構造をTS初期構造としてDFTの OptTS に渡します。

利点は、失敗時にどこで経路が崩れたか分かりやすいことです。欠点は、単純なプロトン移動以外の座標、たとえばHFの回転、F−の再配向、多点水素結合を取り逃がしやすいことです。

方法B：NEB / CI-NEB / growing string

C_HB と P_IP の両端構造を用意し、NEBまたはstring法で経路を作ります。ASEのNEBは初期状態と終状態の間の遷移経路と障壁探索に使えます。 ￼ pysisyphusはPES探索、一次鞍点、IRC、NEB、Growing Stringに対応していますが、2024年11月時点でメンテナが「unmaintained」と明記しているため、研究試作には便利でも、量産ワークフローでは代替手段を残すべきです。 ￼ autodEは、反応物・生成物構造からconformer探索、NEB/CI-NEB/adaptive経路探索、TS探索までを自動化する設計なので、候補数が多い場合のTS guess生成に適しています。 ￼

方法C：ORCA NEB-TS / OptTS / IRCを中心にする

ORCAを使える環境なら、最も実務的です。ORCAはTS最適化で OptTS を使え、正しいTSは虚振動が1つだけであることを振動解析で確認します。ORCAのチュートリアルでも、TS検証には同じ計算レベルでのFrequency計算とIRCが推奨されています。 ￼

最小テンプレートは次のようになります。

! r2SCAN-3c TightSCF Opt Freq
%pal nprocs 16 end
* xyzfile 0 1 complex.xyz

TS精密化：

! r2SCAN-3c TightSCF OptTS Freq
%pal nprocs 16 end
%geom
  Calc_Hess true
end
* xyzfile 0 1 ts_guess.xyz

IRC：

! r2SCAN-3c TightSCF IRC
%pal nprocs 16 end
* xyzfile 0 1 ts_optimized.xyz

IRC後、両端を再最適化し、片側が M···HF、もう片側が [MH]+···F− または目的生成物に戻るかを確認します。

⸻

4. Gaussian同等精度を狙う計算レベル

Gaussianと比較するなら、「Gaussian B3LYP/6-31+G(d,p)」のような古典的設定に合わせるより、現在は長距離補正・分散補正・拡散関数つき基底を使う方がよいです。HF、F−、接触イオン対、水素結合を扱うため、拡散関数は重要です。

推奨レベル

用途	推奨	理由
大量初期探索	GFN2-xTB / CREST	配座・会合錯体・経路guessを高速生成
DFT構造最適化	r2SCAN-3c または ωB97X-3c	分散・BSSE補正込みの実用的composite DFT
TS最適化・freq	r2SCAN-3c → 必要に応じて ωB97X-D/def2-SVP(D)	TS探索の安定性とコストのバランス
最終single point	ωB97M-V/def2-TZVPD、ωB97X-D4/def2-TZVPPD、revDSD-PBEP86-D4/def2-TZVPPD	水素結合、イオン対、反応障壁の精度向上
小規模benchmark	DLPNO-CCSD(T)/def2-TZVPP(D) または canonical CCSD(T)	DFTランキングの較正
熱補正	quasi-RRHO / quasi-harmonic	低振動数の過大エントロピーを抑制

ORCAにはHF-3c、B97-3c、r2SCAN-3c、PBEh-3cなどのcomposite methodがあり、ωB97X-3cも利用できます。 ￼ Psi4はオープンソースで、DFT、MP2、SAPT、coupled clusterなどに対応し、WB97M-VやWB97X-D系の汎関数も扱えます。 ￼ PySCFはApache-2.0の自由なPython量子化学基盤で、DFTやD3/D4分散補正を組み込んだワークフローを作れます。 ￼ NWChemもHPC向けのオープンソース量子化学コードで、DFTモジュールを備えています。 ￼

DLPNO-CCSD(T)はGaussian的なDFT比較を超えた較正用として有効です。ORCAはDLPNO-CCSD(T)やDLPNO-MP2を扱え、Psi4にもDLPNO-CCSD(T)のドキュメントがあります。 ￼ ただしORCAは学術利用では無償配布されていますが、商用利用は別ライセンス扱いです。半導体企業内で完全にフリー/OSSに限定するなら、Psi4、PySCF、NWChemを主エンジンにし、ORCAはライセンス確認後の選択肢にするのが安全です。 ￼

⸻

5. 無料ツール構成の現実的な比較

役割	第1候補	代替	コメント
SDF読込・前処理	RDKit	Open Babel	RDKit中心で十分
配座探索	CREST/GFN2-xTB	RDKit MMFF, ETKDG	CRESTを推奨
HF会合体生成	自作Python/RDKit	ASE	幾何配置は自作が最も制御しやすい
経路探索	ORCA NEB-TS, autodE	ASE NEB + Sella, pysisyphus, ASH	ORCA不可ならASE/Sella/ASH
DFT opt/freq	ORCA	Psi4, PySCF, NWChem	商用完全フリーならPsi4/PySCF/NWChem
高精度single point	ORCA DLPNO-CCSD(T)	Psi4 CCSD(T)/DLPNO, NWChem CCSD(T)	小分子subsetで較正
熱補正	GoodVibes	Arkane	GoodVibesはORCA/Psi4/NWChem/xTB出力に対応
速度定数	Arkane	自作TST	T依存rateが必要ならArkane
出力parse	cclib	OPI, 自作parser	cclibは複数QCコード出力を読める
workflow	Snakemake/quacc	FireWorks, Parsl	job array化しやすい

ASHはORCA、xTB、CP2K、Psi4、PySCF、Gaussian、NWChemなどへのインターフェースを持ち、最適化、振動数、MD、scan、NEBなどのjob typeを扱えるため、複数エンジンをまたぐワークフローの上位層として有用です。 ￼ GoodVibesはGaussian、ORCA、NWChem、Q-Chem、xTB、ASEなどの出力から準調和熱補正を計算でき、低振動数のRRHO問題を扱えます。 ￼ Arkaneは量子化学計算から熱力学量やTST速度定数を計算でき、ORCAやPsi4などの出力にも対応しています。 ￼

NBO解析は多くの場合、有償NBOプログラムが必要になるため、完全フリー範囲では避けた方がよいです。代わりに、ORCAのMayer bond orderやLöwdin/Mulliken解析、またはMultiwfnによる波動関数解析を使うのが現実的です。 ￼

⸻

6. 実装イメージ：SDFから候補ジョブを作る

最初の自動化は、以下のようなPythonスクリプトで組みます。

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
SDF = "candidates.sdf"
OUT = Path("jobs")
OUT.mkdir(exist_ok=True)
def prepare_mol(mol, mol_id):
    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 20260508
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=50,
            params=params
        )
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    return mol
def find_acceptor_atoms(mol):
    # 例：中性アミンN、ピリジン型N、O原子を抽出。
    # 実運用ではSMARTSを増やし、アミドNや四級アンモニウムを除外する。
    acceptors = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        if z in (7, 8) and charge <= 0:
            acceptors.append(atom.GetIdx())
    return acceptors
supplier = Chem.SDMolSupplier(SDF, removeHs=False)
for i, mol in enumerate(supplier):
    if mol is None:
        continue
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i:04d}"
    mol = prepare_mol(mol, name)
    acceptors = find_acceptor_atoms(mol)
    mol_dir = OUT / name
    mol_dir.mkdir(exist_ok=True)
    for conf in mol.GetConformers():
        conf_id = conf.GetId()
        # ここでXYZを書き出し、CREST/xTBへ渡す。
        # その後、各acceptor atomにHFを複数方向で配置し、
        # complex.xyz, product_guess.xyzを生成する。
        pass

この後に作るジョブディレクトリは、たとえば次のようにします。

jobs/
  mol_0001/
    conf_000/
      site_N_05/
        00_xtb_complex/
        01_dft_complex/
        02_product_guess/
        03_neb_or_scan/
        04_ts_opt/
        05_irc/
        06_sp_highlevel/
        result.json

各 result.json に、計算レベル、SCF収束、虚振動数、IRC成功可否、エネルギー、自由エネルギー、HF結合長、電荷などを保存します。最後に全候補を集約してランキングします。

⸻

7. 出力すべきランキング指標

候補分子ごとに、少なくとも以下を出すべきです。

指標	意味
ΔG_bind	M + HF → M···HF の会合自由エネルギー
ΔE‡_int	会合錯体基準の電子エネルギー障壁
ΔG‡_int	会合錯体基準の自由エネルギー障壁
ΔG‡_app	分離気相分子基準の見かけ自由エネルギー障壁
ΔG_rxn	[MH]+···F− 形成の自由エネルギー
r_HF	錯体中のHF結合長
Δr_HF	孤立HFからの伸長
ν_HF	HF伸縮振動数、赤方シフト
q_F, q_H	F/Hの電荷変化
BO_HF	H–F結合次数
imag_freq	TSの虚振動数
IRC_status	正しい反応物・生成物に接続したか
status	normal / barrierless / no_product / failed

HF活性化剤の探索では、単に障壁だけを見るより、HF結合長の伸長、HF振動数の赤方シフト、H–F結合次数低下、F側の電荷増大を併用した方が頑健です。水素結合錯体の結合エネルギーを比較する場合は、counterpoise補正または大きめの基底関数を使います。ORCAやPsi4にはcounterpoise/BSSE補正の機能があり、3c系methodにはgCP補正が組み込まれています。 ￼

⸻

8. 最新手法：MLポテンシャルは「TS guess生成」に使う

2025–2026年時点では、分子反応経路探索にMLポテンシャルを使う流れが急速に進んでいます。OMol25はωB97M-V/def2-TZVPDレベルの1億件超のDFT計算を含む大規模データセットとして報告され、広範な元素・電荷・スピン・反応構造を対象にしています。 ￼ 2026年のpreprintでは、MACE-OMol25、UMA、eSEN、AIMNet2、GFN2-xTBなどの自由に使えるポテンシャルを組み合わせ、freezing stringやCI-NEBでTSを自動探索する手法が報告され、有機系反応でDFT勾配回数を大きく削減できるとされています。 ￼ UMA系モデルはFAIRChemのASE calculator経由で使え、AIMNet2もエネルギー、力、電荷、Hessianなどを予測する反応系向けモデルとして公開されています。 ￼

ただし、HF、F−、接触イオン対、強いプロトン移動はMLモデルの外挿になりやすいので、最終障壁は必ずDFTで再最適化・freq・IRC検証してください。MLは次の用途に限定するのが安全です。

1. HF会合体の初期構造スクリーニング
2. NEB/string経路の初期guess生成
3. TS guessの候補順位付け
4. DFTで失敗した経路の再初期化

⸻

9. 推奨する実運用スタック

学術・ORCA利用可の場合

RDKit
+ CREST/GFN2-xTB
+ ORCA 6.x / OPI
+ GoodVibes
+ cclib
+ Snakemake or quacc

ORCAは学術利用では無償で、semiempiricalからDFT、多参照・相関ab initioまで広い手法を備えています。 ￼ ORCA 6.1以降ではPython連携のOPIも整備されており、入力生成・出力解析の自動化に使えます。 ￼

企業内で完全フリー/OSSを優先する場合

RDKit
+ CREST/GFN2-xTB
+ ASE/Sella/autodE/ASH
+ Psi4 and/or PySCF and/or NWChem
+ GoodVibes
+ cclib
+ Snakemake/quacc

この構成なら商用ライセンス制約をかなり避けられます。Psi4、PySCF、NWChemはいずれも無料・オープンソース系の量子化学基盤です。 ￼ cclibは量子化学出力ファイルを横断的にparseできるため、複数エンジンを使う自動化で便利です。 ￼

⸻

10. 最初に作るべき検証セット

SDF本番候補に入る前に、以下の小セットでワークフローを検証するとよいです。

NH3
methylamine
dimethylamine
trimethylamine
pyridine
aniline
H2O
methanol or isopropanol

理由は、N系/O系、脂肪族/芳香族、強塩基/弱塩基の差が出るためです。2026年のHF活性化研究でも、H₂O、IPA、アニリン、ピリジン、ジメチルアミン、トリメチルアミンが比較対象に含まれており、候補分子の傾向確認に使いやすいです。 ￼

この検証セットで以下を確認します。

1. M···HF錯体が妥当に最適化されるか
2. HF結合長・振動数シフトが化学直感と合うか
3. TSが1つの虚振動を持つか
4. IRCが正しいC_HBとP_IPを結ぶか
5. r2SCAN-3c, ωB97X-D, ωB97M-V の順位が大きく矛盾しないか
6. DLPNO-CCSD(T) single pointでDFT順位を較正できるか

⸻

11. 提案する最終ワークフロー

本件では、以下の二段階設計を推奨します。

Screening workflow

SDF
→ RDKit sanitize / H付加 / 配座生成
→ CREST/GFN2-xTB配座探索
→ HF会合体を各N/Oサイトに自動配置
→ xTBまたはr2SCAN-3cで錯体最適化
→ relaxed scanまたはNEBでTS guess生成
→ r2SCAN-3c OptTS/Freq/IRC
→ ΔG‡_int, ΔG‡_app, HF活性化指標でランキング

High-accuracy workflow

上位候補 10–30件
→ ωB97X-3c または ωB97X-D/def2-TZVP(D)で再opt/freq
→ ωB97M-V/def2-TZVPD または ωB97X-D4/def2-TZVPPD single point
→ 小分子・上位候補subsetでDLPNO-CCSD(T)較正
→ GoodVibesで温度・準調和補正
→ 実プロセス温度・HF分圧・添加剤分圧で ΔG‡_app 補正

これで、SDFに入った複数候補を自動処理しつつ、最終的にはGaussianで一般に行うDFT障壁計算と同等、場合によってはそれ以上に体系的な比較ができます。特に重要なのは、TSの虚振動1つだけでは不十分で、IRCで本当に M···HF と [MH]+···F− を接続していることを確認する点です。HF活性化では、見かけの低障壁TSが別の配座変換やHF回転であることがあるため、IRC検証を自動判定に入れるべきです。