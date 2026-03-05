以下では、あなたが既に決めた全体設計（**タスク分割／パイプライン実行／RDKit中心／成果物規約／将来ClearML**）をそのまま前提にして、**構造処理フェーズ（RDKit）だけ**を「アーキテクト観点＋処理の具体内容＋コード構成」まで徹底的に掘り下げます。

---

# 1. 構造処理フェーズの役割と“ここでやらないこと”

## 1.1 役割（下流の計算品質と計算コストを決める）

構造処理（RDKit）は、後続の **CREST+xTB / PES探索 / DFT / 速度論**のすべてに効きます。ここでの目的は大きく3つです。

1. **入力SDF（PubChem由来）の揺れを吸収**して「計算に載る分子」を作る
2. **化学的同一性・変更履歴を明示**し、後から監査・再現できる
3. **下流の爆発（配座数・複合体数・反応候補数）を抑えるための“構造メタ”**を付与する

## 1.2 ここで“やらない”こと（境界の明確化）

* **本格的なプロトマー/タウトマー列挙**（→反応生成層 or 別タスクでやる。ここで無制限にやると爆発）
* **エネルギー最小化の保証**（PubChem 3D は「エネルギー最小点ではない」旨が明示されています。構造処理では“初期座標の整形”に留め、エネルギー整合は xTB/CREST が担当）([pubchem.ncbi.nlm.nih.gov][1])
* **反応の可否判断（化学反応性の本判断）**（→PES探索/DFT層へ）

---

# 2. 構造処理フェーズのタスク分割（DAGの中での位置づけ）

あなたのワークフローに沿って、構造処理は最低でも次の2タスクに分けるのがよいです（すでに提案済みの粒度を維持）。

* `ingest_dataset`：SDF取り込み・ID採番・“生データ保全”
* `standardize_dataset`：RDKit標準化・3D下準備・QCフラグ付与・下流向け形式出力

さらにMI観点の効率化を強めるなら、任意で以下を足せます（足しても全体設計は崩れません）：

* `annotate_sites`：反応テンプレの“下地”になる官能基・部位注釈（SMARTSマッチなど）
* `compute_complexity`：原子数・回転結合数など、下流の予算（配座数、複合体数）の自動調整に使う

---

# 3. ディレクトリ／成果物規約（“第三者が見れば分かる”を最優先）

1分子ごとに成果物を固定し、**raw と std を必ず並べて保存**します（「どこを変えたか」を消さない）。

```text
artifacts/
  molecules/
    MOL_<hash>/
      raw/
        input.sdf
        input_text.sdf   # 原文（MolBlock含む）をそのまま保存（監査用）
        source_meta.json # PubChem CID等、SDF propertyのスナップショット
      std/
        std.sdf          # 標準化後（H無し or 最小限）
        std_withH.sdf    # QM用に explicit H を付けた版
        std.xyz          # CREST/xTB入力向け
        mol.json         # 内部モデル（Pydantic）
        transform_log.json
        qc_flags.json
      cache_key.txt
```

---

# 4. データモデル（Pydantic）を“構造処理で確定”させる

構造処理で確定すべき情報は、下流全層が共通で参照するため、**ここで型（モデル）を固定**します。

## 4.1 `MoleculeRecord`（例）

* `molecule_id`：`MOL_<hash>`
* `source`：ファイルパス、PubChem CID、取得元URL（URLは保存するがレスポンスには出さない運用でもOK）
* `formula` / `n_atoms` / `elements` / `n_rotatable_bonds`
* `canonical_smiles`（RDKit）
* `formal_charge`（RDKit）
* `spin_multiplicity`（推定＋ユーザー上書き可能）
* `has_3d` / `is_2d_like` / `coord_qc`
* `fragments`（何フラグメントだったか、採用した親フラグメント情報）
* `stereo`（未定義数、問題部位）
* `props_snapshot`（SDFプロパティのスナップショット：必要最小限に絞っても良い）
* `qc_flags[]`
* `provenance`：RDKit version / config hash / standardize version など

## 4.2 QCフラグ（例：Enum）

構造処理で“原因を特定できる”フラグを揃えると、後工程の失敗が激減します。

* `SDF_PARSE_FAIL`
* `SANITIZE_FAIL`
* `VALENCE_ERROR`
* `KEKULIZE_FAIL`
* `MULTI_FRAGMENT_INPUT`
* `COUNTERION_REMOVED`
* `METAL_PRESENT`
* `UNSUPPORTED_ELEMENT`
* `UNDEFINED_STEREO`
* `MISSING_COORDINATES`
* `COORDINATES_2D_LIKE`
* `ATOM_OVERLAP_SUSPECT`
* `ODD_ELECTRON_COUNT`（多重度推定の警告）
* `STRUCTURE_MODIFIED`（標準化で変更が入った）

---

# 5. 処理の詳細（アルゴリズム）

ここが本題です。実装がブレないように、**“順序”を固定**し、各ステップを **独立した関数／クラス**にします。

---

## 5.1 `ingest_dataset`：SDF読み取り、raw保全、ID採番

### 5.1.1 SDF読み取りの設計ポイント

RDKitは `SDMolSupplier(fileName, sanitize=True, removeHs=True, strictParsing=True)` のような引数を持ち、SDFのプロパティは `mol.GetProp()` で取れる、という形で動きます。([GitHub][2])
ただし sanitize を最初から True にすると、**sanitize失敗で mol=None になって属性も取れない**パターンがあり得ます（実務でよく踏む）。そこで ingest は以下を推奨します：

* 読み取り：`sanitize=False` で最大限読む
* その後に `Chem.SanitizeMol()` を try/except で実行して「失敗理由を保存」
  このやり方自体も RDKit コミュニティで言及されています（sanitize=False で読んでから sanitize する）([RDKit Discuss][3])

### 5.1.2 ID採番（再現性のための基準）

IDは「入力ファイル名依存」だとデータ追加で崩れるので、**構造由来**にします。

* 第一候補：`canonical_smiles + formal_charge` のハッシュ
* ただし標準化で SMILES が変わることがあるので、**raw由来ID と std由来ID の二段**が良いです。

  * `raw_id`：rawのSMILES（またはMolBlockハッシュ）
  * `molecule_id`：標準化後の canonical_smiles + charge + std_version のハッシュ
    → 追跡性と重複排除の両立

### 5.1.3 ingestの出力（重要）

* `raw/input.sdf`：元ファイルそのまま
* `raw/input_text.sdf`：ファイル内容をそのまま保存（監査と将来の再パース用）
* `raw/source_meta.json`：SDF props（CID、IUPAC name 等）を保存

---

## 5.2 `standardize_dataset`：RDKit標準化と“下流のための整形”

### 5.2.1 標準化は「ポリシーが分岐」する（化学的に勝手に変えない）

半導体ガスプロセス用途（気相）では、**中和・再イオン化・タウトマー正規化**が “正しいとは限らない” ことが多いです。
よって標準化は「常にやる処理」と「ポリシーで切り替える処理」を分けます。

#### 常にやる（推奨）

* sanitize（必須）
* フラグメント検出と記録（必須）
* stereo再計算と未定義チェック（推奨）
* explicit H 付与（QM入力用に別出力として確保）

#### ポリシーで切り替える（デフォルトは保守的）

* Normalize（官能基表現の統一）
* Reionize（酸塩基部位の再イオン化）
* Uncharge（完全中和）
* タウトマー canonicalization

この標準化フローは、RDKit `rdMolStandardize`（Normalize/Reionize/Uncharger 等）に基づき構成できます。たとえば datamol の `standardize_mol()` は **RDKit rdMolStandardize を使い、Normalize → Reionize → Uncharge（任意） → AssignStereochemistry** の順で実装されています。([Datamol][4])

---

### 5.2.2 “PubChem由来SDF”に対する注意と対策

PubChem の3Dは便利ですが、**理論3D conformerがエネルギー最小点ではない**ことが明記されています。([pubchem.ncbi.nlm.nih.gov][1])
また、PubChem3Dが対象にする分子にはサイズ・柔軟性・元素・塩/混合物除外などの条件があることも説明されています。([PMC][5])

**設計としての対策（構造処理でやること）**

* 3D座標があっても「そのまま信じない」

  * 下流（CREST/xTB）に渡す前に **座標QC**だけ必ず実施（重原子重なり、全z=0、異常距離）
* 塩（多フラグメント）で来る可能性は高い

  * 多フラグメントは `MULTI_FRAGMENT_INPUT` を立てる
  * `keep_largest_fragment` の方針は config で明示し、カウンターイオン除去は `COUNTERION_REMOVED` を立てる
  * “除去前の情報”は raw と transform_log に残す

---

### 5.2.3 標準化のステップ順（提案：固定順序）

以下の順序を「設計として固定」しておくと、再現性・監査性が上がります。

#### Step 0: RDKit Mol作成と property 引き継ぎ

* raw mol から `mol = Chem.Mol(mol_raw)` 的にコピー
* SDF property はすべて `props_snapshot` として JSON 保存

  * ただし巨大化する場合は whitelist（CID, name, formula 等）を推奨

#### Step 1: sanitize（例外分類）

sanitize 失敗は下流で必ず爆死するので、ここで止めるか隔離します。

* `Chem.SanitizeMol()` を try
* 失敗なら:

  * `SANITIZE_FAIL` + 例外種別（valence/kekulize等）を `qc_flags.details` に保存
  * その分子は “std成果物は作るが下流へは流さない” モードが安全（後で手修正できる）

#### Step 2: フラグメント（塩/混合物）処理

* `GetMolFrags(asMols=True)` 等でフラグメント列挙
* configで選択：

  * `keep_policy = largest`（デフォルト）
  * `keep_policy = parent`（酸塩基の親化合物が欲しい場合）
  * `keep_policy = all_split`（分割して別分子として登録：将来用途）
* どれを選んでも、**元のフラグメント情報を transform_log に残す**

#### Step 3: Normalize（官能基の表現統一）※任意

RDKit MolStandardizeは “反応SMARTSによる変換”で正規化でき、必要なら独自ルールも追加できます。([Greg Landrum][6])
ここは「描画の揺れ」を減らすメリットがある一方、用途によっては “勝手に変える” になるので、**on/off可能**に。

#### Step 4: Reionize / Uncharge ※任意（特に気相用途は慎重に）

* Reionize は酸塩基部位の状態を調整する処理（datamol docsにアルゴリズム概要あり）([Datamol][4])
* 気相のHF×塩基では “どのプロトマーが前駆体か” が議論になり得るため、

  * **デフォルトは reionize=True, uncharge=False** くらいの保守設定が多い
  * ただし「候補分子が塩として登録されている」場合は uncharge を検討余地あり
    → いずれにせよ **変換した事実は transform_log に必ず残す**

#### Step 5: ステレオ再計算と未定義の扱い

datamol の標準化でも `AssignStereochemistry(cleanIt=True)` が行われます。([Datamol][4])

* 未定義ステレオ数が閾値（例：>0 または >3）なら `UNDEFINED_STEREO`
* 将来「立体列挙」を入れるならここを separate task にすると設計が綺麗です（構造処理で爆発しやすい）

#### Step 6: explicit H 付与（QM入力用）

* RDKitの表現としては implicit H でも良いが、QM入力（xyz）では明示Hが必要
* よって出力を2系統に分ける

  * `std.sdf`：implicit Hベース（軽い・扱いやすい）
  * `std_withH.sdf` + `std.xyz`：explicit H付き（QM用）

#### Step 7: 3D座標のQCと補完（“最小限”）

PubChem座標は最小点でない可能性があるため、ここでは **品質チェック＋必要最低限の補完**に留めます。([pubchem.ncbi.nlm.nih.gov][1])

* 3D有無判定：

  * conformerが無い → `MISSING_COORDINATES`
  * あるが z が全て同じ/ほぼ0 → `COORDINATES_2D_LIKE`
* 原子重なり検出：

  * 非結合原子間距離 < 0.7 Å（閾値は要調整） → `ATOM_OVERLAP_SUSPECT`
* 補完ポリシー：

  * `embed_3d = if_missing_or_2d_like`（推奨）
  * `embed_3d = never`（座標は後段CRESTに全部任せる）
  * `embed_3d = always`（入力座標を信用しない）

> ここで embed する場合も、**乱数seed固定**で再現性を確保しておくのがMI運用上重要です。

#### Step 8: charge / multiplicity の確定（最重要：後段xTB/DFTに直結）

* `formal_charge`：RDKitの formal charge を採用
* `spin_multiplicity`：SDFだけでは曖昧になりがちなので

  * ユーザー指定（reactants.yaml / molecule_meta.yaml）を最優先
  * 無い場合は「総電子数の偶奇」から暫定推定（偶数→1、奇数→2）
  * ただし誤りうるので `ODD_ELECTRON_COUNT` を立てて人間が見直せるようにする

---

# 6. “変更履歴（transform_log）”を第一級の成果物にする

PubChem由来の標準化では、通過した構造の **44%が標準化で変更される**と報告されています。([PMC][7])
つまり「標準化したら変わる」は普通に起こるため、**何が変わったかが残らない設計はNG**です。

## 6.1 transform_log の例（JSON）

```json
{
  "molecule_id": "MOL_8b2a...",
  "rdkit_version": "2025.9.5",
  "standardize_config_hash": "cfg_91c4...",
  "steps": [
    {"name": "sanitize", "status": "ok", "time_s": 0.01},
    {"name": "fragment_keep_largest", "status": "ok",
     "details": {"n_fragments": 2, "kept_fragment_atoms": 23, "dropped_fragments": ["Cl-"]}},
    {"name": "normalize", "status": "skipped"},
    {"name": "reionize", "status": "ok"},
    {"name": "uncharged", "status": "skipped"},
    {"name": "assign_stereo", "status": "ok", "details": {"undefined_centers": 1}},
    {"name": "add_hs", "status": "ok"},
    {"name": "coord_qc", "status": "warn", "details": {"is_2d_like": true}}
  ],
  "before": {"smiles": "...", "charge": 0},
  "after":  {"smiles": "...", "charge": 0}
}
```

## 6.2 Normalize の “どのルールが当たったか”も取れる設計

RDKit MolStandardize の Normalize は内部で適用ルールをログに出します。RDKit blog には、**カスタム変換の入れ方**や **RDKitのC++ログをPython loggerに流して解析する**方法が解説されています。([Greg Landrum][6])
→ これを使って `transform_log` に「適用されたルール名」を保存すると、監査性が非常に上がります。

---

# 7. 標準化が“重くなる”問題への対策（MI運用の現実）

PubChemの標準化解析では、**全体時間の90%が、最も重い2.05%の構造に費やされる**と報告されています（“エッジケースが支配する”）。([PMC][7])
この構造処理パイプラインでも同じことが起こり得るので、次の設計を入れておくと運用が安定します。

* 1分子あたり `max_standardize_seconds`（ソフトなタイムアウト）

  * 超過したら `STANDARDIZE_TIMEOUT` として隔離、下流へ流さない
* “best_effort”モードと“strict”モードを持つ

  * strict：sanitize失敗は即落とす
  * best_effort：std成果物だけ作って隔離（人が後で直せる）

---

# 8. コード構成（構造処理に関係するモジュールだけ具体化）

あなたのリポジトリ構成案を前提に、構造処理部分を“実装しやすい形”まで落とします。

## 8.1 `interfaces/standardize.py`

```python
from abc import ABC, abstractmethod
from pathlib import Path
from gasrxn.core.models.molecule import MoleculeRecord
from gasrxn.core.models.calc_result import TaskArtifact

class MoleculeStandardizer(ABC):
    name: str

    @abstractmethod
    def standardize_one(self, sdf_path: Path) -> MoleculeRecord:
        """1 SDF(=1分子) -> MoleculeRecord（std成果物パスも含む）"""
        raise NotImplementedError
```

## 8.2 `impl/standardize_rdkit.py`（中心）

* ここは「小さな関数の集合」にして、第三者が差し替え・テストしやすくします。

```python
@dataclass
class StandardizeConfig:
    keep_policy: Literal["largest", "parent", "all_split"] = "largest"
    normalize: bool = False
    reionize: bool = True
    uncharge: bool = False
    assign_stereo: bool = True
    add_hs: bool = True
    embed_3d: Literal["if_missing_or_2d_like", "never", "always"] = "if_missing_or_2d_like"
    random_seed: int = 0
    strict: bool = True
```

内部関数例（分解が重要）：

* `read_sdf_raw()`
* `sanitize_or_flag()`
* `choose_fragment()`
* `apply_molstandardize()`（Normalize/Reionize/Uncharge）
* `stereo_check()`
* `ensure_explicit_h()`
* `coord_qc_and_embed()`
* `export_std_files()`
* `build_molecule_record()`

## 8.3 `tasks/ingest_dataset.py` と `tasks/standardize_dataset.py`

タスク側は “制御・並列・I/O規約” を持ち、化学処理ロジックは impl に寄せます。

* タスクの責務：

  * dataset scope / molecule scope の切替
  * 並列度
  * キャッシュ判定（cache_key）
  * inputs.json / outputs.json / metrics.json 生成

* implの責務：

  * RDKitでの化学処理
  * 例外分類とQCフラグ付与
  * transform_log生成

---

# 9. config（pipeline.yaml）で構造処理を“完全に制御可能”にする

例：

```yaml
structure:
  rdkit:
    strict: true
    keep_policy: largest
    normalize: false
    reionize: true
    uncharge: false
    assign_stereo: true
    add_hs: true
    embed_3d: if_missing_or_2d_like
    random_seed: 0

  qc:
    allow_elements: [H, C, N, O, F, Si, P, S, Cl, Br, I]
    max_undefined_stereo: 0
    overlap_distance_angstrom: 0.7
```

allow_elements は PubChem3D の対象元素にも似ていますが、これはあくまで “あなたの計算系で許容する元素” として定義します（将来拡張）。([PMC][5])

---

# 10. pip-only環境でのRDKit導入（実装条件に合わせた現実解）

PyPI の `rdkit` は wheel を配布しており `pip install rdkit` が可能で、2026-02-16 時点の最新版（例）は `2025.9.5` としてリリースされています。([PyPI][8])
→ conda無しでも構造処理は現実的に成立します。

---

# 11. 構造処理フェーズが下流に渡す“最重要インターフェース”

構造処理は下流へ最低限これだけ渡せば回ります：

1. **QM入力**：`std.xyz`（explicit H, charge, multiplicity）
2. **RDKitグラフ**：`mol.json`（canonical_smiles / site annotation / fragments / qc_flags）
3. **監査・再現**：`raw/input_text.sdf` + `transform_log.json`

特に (3) がないと「後から結果が説明できない」状況になりやすいです（標準化で変わるのが普通だから）。([PMC][7])

---

# 12. 次に提示できる具体物（必要ならこのまま続けて出せます）

あなたの要望に沿って、次のどれからでも“そのまま実装に落とせるレベル”で書けます（質問なしで作ります）：

* `MoleculeRecord` / `QCFlag` / `StandardizeConfig` の **Pydantic定義全文**
* `standardize_rdkit.py` の **関数分割案＋疑似コード（ほぼ実装）**
* `transform_log` の **JSON Schema**
* PubChem由来SDFでよく出る失敗（valence/kekulize/塩/2D）に対する **best_effort修復ポリシー**の具体案（どこまで自動修復するか）

希望があれば、次は **「standardize_dataset タスクのI/O設計（inputs.json/outputs.json/metrics.json）＋1分子処理の疑似実装」**まで一気に具体化します。

[1]: https://pubchem.ncbi.nlm.nih.gov/release3d.html "PubChem3D release notes"
[2]: https://raw.githubusercontent.com/rdkit/rdkit/master/Code/GraphMol/Wrap/SDMolSupplier.cpp "raw.githubusercontent.com"
[3]: https://rdkit-discuss.narkive.com/eHJckt72/sdf-properties-in-case-of-error?utm_source=chatgpt.com "[Rdkit-discuss] SDF properties in case of error"
[4]: https://docs.datamol.io/0.8.6/api/datamol.mol.html "datamol.mol - datamol"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3269824/ "
            PubChem3D: a new resource for scientists - PMC
        "
[6]: https://greglandrum.github.io/rdkit-blog/posts/2024-02-23-custom-transformations-and-logging.html "Using custom standardization rules / Using the python logger from the RDKit – RDKit blog"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6086778/ "
            PubChem chemical structure standardization - PMC
        "
[8]: https://pypi.org/project/rdkit/ "rdkit · PyPI"
