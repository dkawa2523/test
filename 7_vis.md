以下は、あなたが刷新した「論文っぽい図（2D反応スキーム／3Dボール&スティック／エネルギーダイアグラム／IRC／反応ネットワーク等）」を、既存アーキ（タスク独立＋パイプライン、run_dir規約、extract→正規化→render分離、将来ClearML）を維持したまま、**make_figures タスク設計に“落として”**具体化した提案です。

⸻

1. make_figures タスクの位置づけ（DAG）

可視化は “ログ直読みで場当たり描画” を禁止し、extract → canonical tables(parquet) → render を強制します。既存の report.extract_tables 方針を維持しつつ、「論文図に必要な構造・結合・経路・エネルギー」を追加抽出できるようにします。  ￼

推奨DAG：
	1.	計算タスク群（RDKit/CREST/pysisyphus/NWChem/thermo/kinetics…）
	2.	report.extract_tables（既存思想のまま。parquetを整備）
	3.	report.make_figures（←今回設計するタスク）

重要：make_figures は parquet + 参照用artifactパスだけを読む。NWChem stdoutやpysisログ等を直接パースしない（パースは extract_tables 側）。  ￼

⸻

2. report.make_figures のタスク仕様

2.1 目的
	•	研究者・レビュー担当が一目で理解できる 論文スタイルの図（Figure） を自動生成
	•	図ごとに「使ったデータ」「条件」「依存ツール」をメタとして保存し、再現性を担保
	•	外部バイナリ（POV-Ray / graphviz dot 等）が無い環境でも 落とさず skip できる

2.2 CLI（例）

# 全図生成
gasrxn report make-figures --run-dir runs/<run_id> --figures all --formats png,svg

# いくつかだけ（例：2D反応スキーム + エネルギーダイアグラム + IRC）
gasrxn report make-figures --run-dir runs/<run_id> --figures F01,F06,F05 --formats png,svg

# strict：不足データがあれば失敗扱い（CI/品質ゲート用）
gasrxn report make-figures --run-dir runs/<run_id> --figures all --strict

2.3 入力（Task I/O規約に沿う）
	•	inputs.json（参照情報）
	•	run_dir
	•	tables_dir（例：runs/<run_id>/reports/tables）
	•	artifacts_index（extract側で作る「構造/計算成果物パスの索引」parquet/JSON）
	•	params.json
	•	figures: ["F01","F06",...] or "all"
	•	formats: ["png","svg"]
	•	strict: bool
	•	overwrite: bool
	•	style: font/dpi 等
	•	config/figures.yaml（任意。後述）

2.4 出力（Task I/O規約に沿う）

タスクディレクトリ：

run_dir/tasks/<task_run_id>/
  inputs.json
  params.json
  outputs.json
  metrics.json
  qc_flags.jsonl
  logs/{stdout.log,stderr.log,tool.log}
  artifacts/
    output/
      figures/
        F01_scheme_2d.svg
        F01_scheme_2d.png
        F01_scheme_2d.meta.json
        F01_scheme_2d.data.parquet
        ...
      figures_manifest.json

	•	outputs.json：生成した図ファイル一覧とパス
	•	metrics.json：生成数、skip数、失敗数、総時間など
	•	qc_flags.jsonl：図ごとの MISSING_DATA / MISSING_TOOL / RENDER_FAILED など

さらに利便性として、run_dir/reports/figures_latest -> tasks/<task_run_id>/artifacts/output/figures/ のシンボリックリンクを作るのは有益（ただし「正」はタスク成果物に置く）。  ￼

⸻

3. make_figures が読む “canonical tables” の契約

report.extract_tables が作る parquet テーブル（既存の tables/）を拡張し、論文図に必要な最小列を固定します。  ￼

3.1 必須テーブル（図のコア）
	•	molecules.parquet：RDKit標準化済みの分子情報
	•	molecule_id, smiles, molblock(任意), charge, mult, qc_flags...
	•	structures.parquet：図で描く“構造（3D含む）”の索引（ここが重要）
	•	structure_id, species_id, role（reactant/complex/ts/product）
	•	xyz_path（artifactへの相対パス）
	•	bond_list（[(i,j,order),...] をJSONで格納 or 別テーブル）
	•	highlight_atoms（反応中心）
	•	forming_bonds, breaking_bonds（TS注釈用）
	•	reactions.parquet
	•	reaction_id, candidate_id, reactant_species_id, ts_species_id, product_species_id
	•	reaction_class, atom_map（必要なら）
	•	energies.parquet
	•	species_id, role, level_tag（xtb/dft/…）
	•	E_elec, ZPE, G_T, T, units
	•	barriers.parquet（温度依存ΔG‡など）
	•	reaction_id, T, dG_dagger, dE_dagger, standard_state
	•	rates.parquet（Arrhenius用）
	•	reaction_id, T, k, units, model

3.2 任意テーブル（ある場合だけ図が生える）
	•	irc_profiles.parquet（pysisyphusの irc_data.h5 を extract 側で展開）
	•	reaction_id, s, E, direction
	•	network_nodes.parquet, network_edges.parquet（探索結果ネットワーク）
	•	cantera_flux.parquet（Canteraが回っている場合）

⸻

4. コード構成（第三者が図を追加・改造しやすい）

設計思想は “1図 = 1ファイル” ＋ “Registry登録” で統一します。  ￼

src/gasrxn/reporting/
  figures/
    models.py          # FigureSpec, FigureContext, FigureResult
    registry.py        # FIGURES: dict[str, FigureSpec]
    store.py           # TableStore (lazy parquet load), AssetResolver (xyz_path解決)
    render/
      mpl.py           # save_png/save_svg + 共通スタイル
      rdkit2d.py       # RDKit 2D描画ヘルパ
      ase_povray.py    # ASE→POV-Ray（外部povrayは任意）
      graphviz.py      # dotがあればレンダ、無ければ .dot 保存
    impl/
      f01_scheme_2d.py
      f02_reaction_center_2d.py
      f03_ballstick_complex.py
      f04_ballstick_ts.py
      f05_irc_profile.py
      f06_energy_diagram.py
      f07_multi_candidate_profile.py
      f08_arrhenius.py
      f09_pathway_network.py
      f10_cantera_path.py
  tasks/
    make_figures.py    # Task wrapper（I/O規約、manifest、skip/strict）


⸻

5. FigureSpec / FigureContext / FigureResult の契約

5.1 FigureSpec（登録情報）
	•	figure_id: "F01" など
	•	name: "scheme_2d" など
	•	title: 人間用
	•	requires_tables: ["molecules","reactions",...]
	•	requires_python: ["rdkit"] 等（pip依存チェック）
	•	requires_tools: ["povray","dot"] 等（外部バイナリ）
	•	default_formats: ["svg","png"] 等
	•	make(ctx) -> FigureResult：実装関数

5.2 FigureContext（図が使う共通入力）
	•	store: TableStore（parquetを lazy load）
	•	assets: AssetResolver（xyz_path等を絶対パスに解決）
	•	params: 当該図のパラメータ（YAMLで上書き）
	•	work_dir: 図の一時ファイル置場（povやdot）
	•	out_dir: 保存先

5.3 FigureResult（戻り値）
	•	status: succeeded | skipped
	•	outputs: 生成したファイルパスリスト
	•	data_exports: 図が使用した集計データ（parquetに落として保存）
	•	meta: 依存・条件・選抜ロジック・unit等
	•	skip_reason: optional

「図で使った集計表」を毎回 .data.parquet に落とすのが、第三者が後から検証しやすい最重要ポイントです。  ￼

⸻

6. figures.yaml（設定の外出し）

make_figures はコード固定、見た目や選抜条件はYAMLで差し替えできるようにします。

global:
  dpi: 300
  formats: ["png", "svg"]
  strict_missing: false

selection:
  temperature_K: 800
  topN_reactions: 10
  require_qc_pass: true

F01:
  annotate_atom_map: true
  show_smiles: false

F03:
  renderer: "povray"     # povray | skip_if_missing | fallback_2d
  ball_radius: 0.32
  bond_radius: 0.12
  image_size_px: [1800, 1400]

F06:
  energy_kind: "dG"      # dE | dG
  reference: "separated" # separated | complex
  show_labels: true

F09:
  layout: "graphviz"     # graphviz | spring
  edge_weight: "dG_dagger"  # or k(T)


⸻

7. 図（F01〜F10）を make_figures の実装仕様に落とし込み

ここが「タスク設計に落とす」中核です。各Figureが 何を入力にし、何を出すかを固定します。

⸻

F01 2D反応スキーム（R + HF → TS → P）
	•	入力：molecules.parquet, reactions.parquet
	•	選抜：selection.topN_reactions（rankは barriers の dG‡(T=温度) で決めるのが定石）
	•	描画：RDKit 2D（SVGが主、PNGは可能なら）
	•	出力：F01_scheme_2d.svg/png
	•	skip条件：RDKit不可 / 反応定義が無い

“反応そのもの”をRDKit Reactionとして描くか、R/TS/Pを並べて矢印を自前で描くかは実装選択。後者の方が自由度高い（ハイライトと整合しやすい）。

⸻

F02 2D反応中心ハイライト（形成/切断結合・原子強調）
	•	入力：structures.parquet, reactions.parquet
	•	highlight_atoms, forming_bonds, breaking_bonds を使う
	•	描画：RDKit 2D（原子色や結合スタイルはパラメータ化）
	•	出力：F02_reaction_center.svg/png
	•	skip条件：forming/breaking 定義が無い（→ハイライト無し版で代替も可）

⸻

F03 3Dボール&スティック（遭遇複合体）
	•	入力：structures.parquet（role=complex）, energies.parquet
	•	描画：
	•	標準：ASE→POV-Ray（外部 povray がある場合）
	•	無い場合：skip か fallback_2d（configで選択）
	•	出力：
	•	F03_complex_ballstick.png
	•	併せて F03_complex_ballstick.pov（再レンダ用）をartifact保存
	•	skip条件：
	•	povray 無し＆fallback無し
	•	bond_list 無し（→距離推定BondGuesserに落とすかを設定で制御）

⸻

F04 3Dボール&スティック（TS、距離注釈付き）
	•	入力：structures.parquet（role=ts）, reactions.parquet
	•	描画：F03同様（povray優先）
	•	注釈：
	•	forming_bonds の距離
	•	breaking_bonds の距離
	•	反応中心原子ラベル（N/H/F 等）
	•	出力：F04_ts_ballstick.png + .pov
	•	skip条件：TS構造が無い／注釈定義が無い（→注釈無し版で代替可）

⸻

F05 IRCプロファイル（E vs reaction coordinate）
	•	入力：irc_profiles.parquet
	•	描画：matplotlib（統一スタイルでPNG/SVG）
	•	出力：F05_irc_profile.png/svg
	•	skip条件：IRC未実行（テーブル無し）

ここで pysisplot を直接呼ぶ方式もあり得ますが、「extract→table→render」分離を守るなら、irc_data.h5 → irc_profiles.parquet は extract 側、描画は make_figures 側が堅いです。  ￼

⸻

F06 エネルギーダイアグラム（R → TS → P、ΔEとΔGの両対応）
	•	入力：energies.parquet, reactions.parquet（必要なら barriers.parquet）
	•	処理：
	•	R, complex, TS, P のレベル（y）を作る（この集計表を .data.parquet に保存）
	•	energy_kind: dE|dG / reference: separated|complex を切替
	•	描画：
	•	energydiagram が入っていればそれを使用
	•	無ければ matplotlib 自前実装にフォールバック（どちらも同じIF）
	•	出力：F06_energy_diagram.svg/png
	•	skip条件：必要なエネルギーが揃わない（strictならfail）

⸻

F07 複数候補のエネルギープロファイル（Top Nを横並び比較）
	•	入力：barriers.parquet, reactions.parquet
	•	選抜：topN_reactions を dG‡(T=温度) で並べる
	•	描画：matplotlib（線図 or 小さなエネルギーダイアグラムを並べる）
	•	出力：F07_multi_candidate_profile.svg/png
	•	skip条件：barriers無し

⸻

F08 Arrhenius（log k vs 1/T）
	•	入力：rates.parquet
	•	選抜：Top N（温度点の中央値でkが大きい順など）
	•	描画：matplotlib
	•	出力：F08_arrhenius.svg/png
	•	skip条件：rates無し

⸻

F09 反応経路ネットワーク（ノード=種、エッジ=素反応）
	•	入力：network_nodes.parquet, network_edges.parquet
	•	無い場合は reactions + barriers から簡易生成してもよい（ただし “生成した” ことをmetaに残す）
	•	描画：
	•	graphviz dot があれば dot→svg/png（見栄え優先）
	•	無ければ networkx spring layout（matplotlib）
	•	出力：
	•	F09_pathway_network.svg/png
	•	F09_pathway_network.dot（再レンダ用）
	•	skip条件：ネットワーク情報が無い＆簡易生成もOFF

⸻

F10 Cantera Reaction Path Diagram（Flux/ROP）
	•	入力：cantera_flux.parquet（または Canteraの出力をextractしたdot）
	•	描画：
	•	dot があればレンダ
	•	無ければ dot保存のみ（図生成はskip扱いでも良い）
	•	出力：F10_cantera_path.svg/png/dot
	•	skip条件：Cantera結果無し

⸻

8. make_figures の実行ロジック（擬似コード）

def run_make_figures(run_dir, figures, config, strict):
    tables_dir = run_dir/"reports/tables"
    out_dir = task_dir/"artifacts/output/figures"
    store = TableStore(tables_dir)
    assets = AssetResolver(run_dir)

    manifest = []
    for fig_id in resolve(figures):
        spec = FigureRegistry.get(fig_id)

        # 1) 依存チェック（table/python/tool）
        ok, reason = spec.check_requirements(store, tools=which(...), py=importlib)
        if not ok:
            manifest.append(skipped(fig_id, reason))
            if strict: raise MissingRequirement(reason)
            continue

        # 2) 描画
        try:
            ctx = FigureContext(store=store, assets=assets, params=config.for_fig(fig_id), ...)
            result = spec.make(ctx)

            # 3) 保存（図＋meta＋data）
            save_outputs(result, out_dir)
            manifest.append(succeeded(fig_id, result.outputs))
        except MissingData as e:
            manifest.append(skipped(fig_id, str(e)))
            if strict: raise
        except Exception as e:
            manifest.append(failed(fig_id, repr(e)))
            if strict: raise

    write_json(out_dir/"figures_manifest.json", manifest)


⸻

9. “第三者が修正・追加しやすい”運用ルール

9.1 新しい図を追加する手順（最短）
	1.	src/gasrxn/reporting/figures/impl/fXX_new.py を追加（make(ctx)->FigureResult だけ実装）
	2.	registry.py に1行登録（FigureSpec追加）
	3.	figures.yaml にパラメータ（任意）
	4.	tests/test_make_figures_smoke.py に “missing dataでも落ちない” スモークテストを追加

9.2 既存図のロジック修正
	•	図のロジック変更は impl/fXX_*.py のみ
	•	データ列名や単位が変わる場合は extract_tables 側で互換列を出す（alias）
→ 図側に “if 列名A else 列名B” を散らさない（破綻防止）

⸻

10. pip / 外部バイナリ依存の扱い（skipで壊れない設計）
	•	pip依存（例：ase, energydiagram, networkx）は extras で管理
	•	pip install .[viz] で図を全部出せる、が理想
	•	外部バイナリ（例：povray, dot）は tools.yaml / PATH で検出
	•	ない場合は F03/F04/F09/F10はskip（strictならfail）

“計算はできるが図は作れない” を許容するのが運用上は正しい（CIはstrictにすればよい）。  ￼

⸻

ここまでを踏まえた最小の結論
	•	report.make_figures は FigureSpec登録型で、parquet＋artifact参照のみを使って描画
	•	出力は figureファイル＋meta＋使用データを1セットで保存（再現性・レビュー容易性）
	•	3D/ネットワークは外部バイナリに依存するため 「あれば生成／なければskip」 をタスク仕様に内蔵
	•	これを既存の run_dir・タスクI/O規約にそのまま乗せる

⸻
