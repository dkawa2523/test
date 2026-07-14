以下は、あなたの既存の全体設計（**各フェーズ＝独立タスク**、それらを**パイプラインとして直列/分岐実行**、現在はLocal、将来は**ClearMLでDB/ワークフロー管理**）を維持したまま、**「実行管理（将来ClearML）」フェーズ**を、アーキテクト観点で“実装に落ちる粒度”まで具体化した提案です。

---

# 1. 実行管理フェーズの責務（この層が解決すること）

化学計算パイプラインでは、アルゴリズムよりも **「実行の再現性・追跡性・失敗復旧・スケール」** が律速になりがちです。実行管理層の責務は次のとおりです。

## 1.1 必須要件（LocalでもClearMLでも同じ）

* **再現性**：入力（SDF/設定/ツール版）→出力（構造/ログ/数値）が再実行可能
* **追跡性**：どの分子・どの複合体・どのTS候補が、どの条件で計算され、どこに結果があるかを一意に辿れる
* **個別タスク実行**：各フェーズ（RDKit/CREST/xTB/pysisyphus/NWChem/GoodVibes/…）が **単体で走る**
* **パイプライン実行**：依存関係を解決しつつ DAG を回せる
* **失敗復旧**：途中で落ちても再開できる（キャッシュ・リトライ・部分再実行）
* **計算資源管理**：1ジョブの内部並列（crest -T / xtb -P / nwchem MPI 等）と、外側の並列（分子並列）を衝突させない

## 1.2 ClearMLで将来やりたいこと（この層で“準備”しておく）

* 実行結果を **Task** として記録（メタデータ、パラメータ、成果物、ログ）
* 多数候補のスイープを **Queues/Agents** で回す（CPU隊列/大メモリ隊列/GPU隊列など）
* パイプラインを **ClearML Pipelines** で可視化し、UIから再実行
* 入力SDF/設定/結果を **Dataset/Artifacts** として版管理

ClearMLのPipelinesは、**Remote Mode（デフォルト）**と**Local Mode（サブプロセス実行）**、**Debugging Mode（関数を同期呼び出し）**が明確に定義されています。Local Modeは「パイプラインをローカルで実行し、各ステップをサブプロセスで実行、同一Python環境を使う」とされ、Remote Modeは「パイプラインコントローラが指定Queueで実行され、各ステップが各Queueで独立実行され、git・依存・コンテナを各タスクが制御できる」と説明されています。([ClearML][1])

---

# 2. Localで先に作るべき「ClearML互換の実行管理コア」

将来ClearMLへ移行する前提なら、Local版でも **“ClearMLの概念モデルに寄せる”**のが最も移行コストを下げます。

## 2.1 まずは概念を揃える（Localオブジェクト＝ClearMLオブジェクトの写像）

| あなたのLocal実装で持つ概念                   | 将来ClearMLで対応する概念                                                     |
| ---------------------------------- | -------------------------------------------------------------------- |
| `Run`（パイプライン全体の実行ID）               | Pipeline Controller Task（パイプラインの親Task）([ClearML][2])                 |
| `TaskRun`（各フェーズ1回の実行）              | ClearML Task（1実験/1ジョブ）([ClearML][3])                                 |
| `Params/Config`（入力パラメータ群）          | Task.connect / Task.connect_configuration（パラメータ・設定の接続）([ClearML][4]) |
| `Artifacts`（ログ/xyz/nw入力/結果json）    | Task.upload_artifact（ファイル/フォルダ/DFなど）([ClearML][5])                   |
| `Metrics`（runtime, success, ΔG‡など） | Logger.report_scalar / report_table など([ClearML][6])                 |
| `Dataset`（SDF入力や結果集合）              | clearml Dataset / clearml-data CLI([ClearML][7])                     |
| `Queue`（CPU/GPU/大メモリなど）            | ClearML Queue + Agent([ClearML][8])                                  |

この対応関係を崩さないように Local実装を作るのがポイントです。

---

# 3. ディレクトリとメタデータ設計（Local＝“ClearMLのTaskページ”相当を作る）

## 3.1 推奨のRunフォルダ構造（強く推奨）

```
runs/
  <run_id>/                         # パイプライン1回（親）
    run_manifest.json               # 全体入力・全体設定・DAG・バージョン
    controller/
      stdout.log
      pipeline_graph.json           # DAG展開結果
      summary_table.parquet         # 材料探索向けまとめ
    tasks/
      <task_run_id>/                # 例: dft.nwchem_ts_saddlefreq の1回
        task_manifest.json          # そのTaskの入力・依存・hash・リソース
        params.json                 # connect相当（後でClearMLへそのまま）
        config/                     # YAML等（connect_configuration相当）
          task_config.yaml
        logs/
          stdout.log
          stderr.log
          tool.log                  # 外部ツール原ログ（nwchem/crestなど）
        artifacts/
          input/                    # 入力幾何、入力SDF、生成したdeckなど
          output/                   # xyz, h5, csv, json, chem.inp etc
        metrics/
          metrics.json              # scalars/tablesの元
        status.json                 # success/fail + error分類 + 再試行情報
```

**狙い**

* `task_manifest.json` は ClearML Task の “Info + Configuration + Artifacts + Console” を **ローカルで再現**するための最小要素
* 後でClearMLへ載せ替える際、`params.json` / `config.yaml` / `artifacts` をそのまま `Task.connect(...)`, `Task.connect_configuration(...)`, `Task.upload_artifact(...)` に流し込めます ([ClearML][4])

## 3.2 `task_manifest.json` に必ず入れる項目（超重要）

* `task_name`（例：`dft.nwchem_ts_saddlefreq`）
* `task_run_id`（UUID推奨）
* `inputs`（依存TaskRunのartifact参照、入力ファイルのhash）
* `tool_versions`（nwchem/crest/xtb/pysis/goodvibes…のversion文字列）
* `resource_spec`

  * `nproc`, `mem_mb`, `walltime`, `requires_gpu`, `scratch_dir`
* `cache_key`

  * `hash = H(inputs + params + tool_versions + code_version)`
* `status`

  * `state: queued/running/succeeded/failed/skipped_cached`
  * `error_type: scf_fail/geom_fail/io_fail/timeout/...`
  * `attempt: 1/2/3`

ここが揃うと「結果が出た/出てない」以上に、「**使える結果か**」「**比較可能か**」が判定できます。

---

# 4. “実行”の抽象：LocalとClearMLを切り替えるための設計

## 4.1 3つの抽象を分ける

実行管理は、次の3レイヤに分けると第三者が理解・拡張しやすいです。

1. **Orchestrator（DAG実行）**

* 依存解決、スケジューリング、失敗時ポリシー（fail-fast / best-effort）

2. **Executor（1タスクをどう走らせるか）**

* Localなら `subprocess` でCLI起動
* 将来ClearMLなら `Queue` にenqueue or Pipeline stepとして起動

3. **Tracker（記録/ログ/成果物管理）**

* LocalTracker：runフォルダへ書く
* ClearMLTracker：Task/Logger/Artifact/Datasetへ書く ([ClearML][3])

### 推奨インターフェース（例）

```python
class Tracker:
    def connect_params(self, d: dict) -> None: ...
    def connect_config(self, name: str, path: str) -> None: ...
    def log_scalar(self, title: str, series: str, value: float, step: int) -> None: ...
    def log_table(self, title: str, df_or_csv_path) -> None: ...
    def upload_artifact(self, name: str, path_or_obj) -> None: ...
    def set_user_properties(self, d: dict) -> None: ...
```

ClearML側は `Task.connect`/`Task.connect_configuration`/`Logger.report_scalar`/`Task.upload_artifact`/`Task.set_user_properties` で実装できます。([ClearML][4])
Local側は同等の情報を `params.json`/`config/`/`metrics.json`/`artifacts/`/`status.json` に書くだけです。

---

# 5. パイプライン実行（Local版）を“ClearML Pipelines互換”にする

## 5.1 Localの実行モードを3つ用意（ClearMLの思想に合わせる）

ClearML Pipelinesには **Remote / Local / Debugging** の3モードがあります。([ClearML][1])
Local実装も同様の3モードを持つと移行が楽です。

### (A) Debugモード（最優先で実装）

* 各タスクを Python関数として同期呼び出し
* 例外をそのまま投げてデバッグしやすい
  ClearMLでも Debugging Mode は「通常のPython関数として同期実行しデバッグ可能」と説明されています。([ClearML][1])

### (B) Local Subprocessモード（実運用に近い）

* 1タスク=1プロセスで起動（外部ツールの環境が汚れない）
* ClearMLのLocal Modeも「ステップをサブプロセス実行、同一Python環境」とされています。([ClearML][1])

### (C) “擬似Queue”モード（将来ClearML Queueへ置換）

* ローカルでジョブキュー（SQLite/ファイル）を作り
* ワーカーが拾って実行する（CPU隊列/大メモリ隊列/DFT隊列など）
* 後で ClearML Queue/Agent に差し替えるだけにする

---

# 6. 将来ClearMLに載せるときの具体像（どう実装すれば無理なく移行できるか）

## 6.1 最小統合（各タスクをClearML Taskとして記録）

各タスクのエントリポイント（CLI main）で以下を行うだけで十分に効果が出ます。

* `Task.init(project_name=..., task_name=...)`
* `task.connect(params)` または `Task.connect_configuration` で設定保存 ([ClearML][4])
* 主要出力（json/csv/xyz/zip）を `task.upload_artifact` でアップロード ([ClearML][5])
* runtimeや成功率などを `Logger.report_scalar` で記録 ([ClearML][6])

「Artifactsはファイル/フォルダ/辞書/NumPy/DFなど幅広い型に対応」とされ、`Task.upload_artifact` の使用が推奨されています。([ClearML][5])

## 6.2 パイプライン統合（ClearML Pipelines）

ClearMLのPipelineDecoratorは、`@PipelineDecorator.pipeline` によりパイプラインコントローラが **独立Task** として作られる、と説明されています。([ClearML][2])
また、Pipelinesページには「各ステップは通常のLoggerフローで成果物/メトリクスをログでき、実行中のステップTaskは `Task.current_task()` で取得できる」とあります。([ClearML][1])

### 重要：モードの使い分け（開発→運用）

* 開発：Debugging Mode（関数同期）([ClearML][1])
* ローカル運用：Local Mode（サブプロセス）([ClearML][1])
* スケール運用：Remote Mode（Queue + Agent）([ClearML][1])

### パイプラインパラメータ設計（ClearML UIで差し替えるため）

PipelineDecoratorには `add_parameter` があり、パイプラインTaskのHyper-parametersにパラメータを露出できる、と説明されています。([ClearML][9])
→ 例えば以下を pipeline parameter にしておくと、UI/再実行が強力になります。

* 入力データセットID（SDF集合）
* 対象反応タイプ（proton_transfer 等）
* スクリーニング閾値（上位N件、ΔG‡閾値）
* DFTレベル（geom/freq/sp）
* 温度レンジ（GoodVibes/Cantera）

## 6.3 Queue/Agent設計（半導体ガスプロセスの計算に必須）

ClearMLのQueues/Agentsの挙動は明確に文書化されています。

* Taskをqueueにenqueue
* Agentがpull
* Dockerコンテナを起動してコードを実行
* 実行環境セットアップ（setup script → system packages → git clone → uncommitted changes適用 → Python環境/依存構築）
  という流れです。([ClearML][8])

さらにAgentは **pip（デフォルト）/ conda / poetry** をサポートすると書かれており、あなたの「conda不可」条件でも **pip運用が前提にできる**のが重要です。([ClearML][8])

### Queueの切り方（このパイプライン向け）

* `cpu_short`：RDKit/整形/小さいxTB
* `cpu_long`：CREST/複合体列挙
* `cpu_mem`：pysisyphus（画像多い）/一部DFT
* `dft_mpi`：NWChem MPI実行（ノード/スロット管理）
* `gpu`：将来GPU4PySCF等用

PipelineController側には **default execution queue** と step-specific override があり、全体のデフォルトと例外を設計しやすいです。([ClearML][10])

## 6.4 “ローカルからリモートへ飛ばす”スイッチ（execute_remotely）

ClearMLには `Task.execute_remotely()` があり、ローカル実行を終えてタスクをキューにenqueueし、Agentが再実行するフローが説明されています。([ClearML][11])
→ 重いDFTだけを「一定条件でリモートに投げる」設計も可能です（例：xTB barrierが閾値以下のものだけ enqueue）。

---

# 7. データ/成果物管理（ClearML Dataをどう使うか）

あなたの用途では「入力SDF」「中間生成物」「最終サマリ」が膨大です。
ClearMLには、オープンソースの **clearml Dataset / clearml-data** と、エンタープライズの **Hyper-Datasets** の区別があります。Hyper-Datasetsは “paid offering” と説明されています。([ClearML][12])

## 7.1 まずは open-source ClearML Data で十分（推奨）

* Dataset classで

  * create → add_files → upload → finalize の流れが示されています。([ClearML][13])
* finalizeすると閉じられて再現性を保証する、とあります。([ClearML][13])
* CLIでも `clearml-data create` / `clearml-data sync` があり、syncはcreate+upload+finalize相当をまとめられます。([ClearML][7])

### このパイプラインでのDatasetの切り方（実務的）

* `dataset:inputs/sdf/<version>`：候補分子SDF群（1分子/ファイルの集合）
* `dataset:inputs/reactants/<version>`：反応相手（HFなど）定義
* `dataset:results/<run_id>`：結果のまとめ（barriers/rates/選抜リスト）
* `dataset:artifacts/<run_id>`：大きいログや構造（必要なら）

**重要**：巨大ログ（NWChem stdout 等）を常にDataset化すると破裂しやすいので、

* “結果抽出済みJSON/CSV”はArtifactsとして必ず上げる
* “生ログ丸ごと”は上位候補だけ圧縮してArtifacts or Datasetへ
  という二段階が扱いやすいです（Artifactsはファイル/フォルダをzipで上げられる）。([ClearML][5])

---

# 8. 設定管理（Hydra採用の有無）

Hydraを使う場合、ClearMLはOmegaConf（設定とoverride）を自動ログできる、と説明されています。([ClearML][14])
あなたは「第三者が把握・改良しやすい設計」を重視しているので、Hydraか、少なくとも **YAML設定をtask.connect_configurationで保存**するのは強く推奨です。([ClearML][4])

---

# 9. 実装の具体（Local→ClearML移行で最小差分にするための“設計ルール”）

## 9.1 ルール1：各タスクは「純粋入力→純粋出力」に寄せる

* 入力は `inputs.json + config.yaml + artifacts/input/` のみ
* 出力は `artifacts/output/ + result.json + metrics.json`
* 外部に書くのは run_dir のみ
  → ClearMLに載せるときも“Task単位の自己完結”が保てます。

## 9.2 ルール2：タスクのCLIは固定（パイプラインはCLIを呼ぶ）

* pipelineは Python関数呼び出しでも良いが、Local Subprocessモードのために
  **CLIで必ず動く**ことを担保する
  → ClearML Local Modeもステップをサブプロセスで回す思想なので一致します。([ClearML][1])

## 9.3 ルール3：パラメータは「connect可能な辞書」にする

ClearMLでは Task.connect で辞書/オブジェクトを接続し、値の変化も追えるとされています。([ClearML][4])
→ `params.json` は常に「フラット寄りの辞書（ネストOK）」で持つ。

## 9.4 ルール4：巨大成果物は“段階的に上げる”

Artifactsは多様な型をアップロードできる一方、巨大生ログを常に上げると帯域/ストレージが詰まります。([ClearML][5])

* 常に上げる：`result.json`, `summary.csv`, `selected_ts.xyz`
* 条件付き：`nwchem_stdout.log.gz`, `irc_data.h5`, `crest_ensemble.zip` など

## 9.5 ルール5：Queue設計のために「resource_spec」をタスクに必須化

Queue/Agentは“必要資源に合わせてenqueueする”前提で説明されています。([ClearML][8])
→ `resource_spec`（cores, mem, gpu, walltime, docker image）を task_manifest に入れる。

---

# 10. ここまでを踏まえた「実装ステップ（Local→ClearML）」の最短ルート

## Phase A（今すぐ：Localのみ）

1. `RunManager`（run_id生成、run_manifest、DAG解決）
2. `LocalExecutor`（subprocess実行、stdout/stderr保存、timeout）
3. `LocalTracker`（params/config/artifacts/metrics/status を所定フォルダに書く）
4. `CacheManager`（cache_key一致ならskip）
5. `SummaryBuilder`（最終的な材料探索用テーブルを生成）

## Phase B（ClearML導入：各タスクからTask.init）

1. `ClearMLTracker` を追加（LocalTrackerと同じIF）
2. 各タスクのCLI main の最初に `Task.init`（project/task名規約を統一）([ClearML][3])
3. `connect_params/connect_config/upload_artifact/log_scalar` を ClearMLへ送る ([ClearML][4])

## Phase C（ClearML Pipelines導入）

1. `@PipelineDecorator.pipeline` で controller を作る（独立Taskになる）([ClearML][2])
2. 開発は Debugging Mode、運用は Local Mode、スケールは Remote Modeへ ([ClearML][1])
3. queue設計を固定し、Agentを配置（pip運用）([ClearML][8])

## Phase D（データ版管理）

1. 入力SDFを Dataset化（Dataset class でも clearml-data CLIでも可）([ClearML][13])
2. 出力サマリを Dataset/Artifactsへ
3. run_manifest に dataset_id を記録（再現性の鍵）

---

[1]: https://clear.ml/docs/latest/docs/pipelines/ "ClearML Pipelines | ClearML"
[2]: https://clear.ml/docs/latest/docs/pipelines/pipelines_sdk_function_decorators/ "PipelineDecorator | ClearML"
[3]: https://clear.ml/docs/latest/docs/references/sdk/task/?utm_source=chatgpt.com "Task"
[4]: https://clear.ml/docs/latest/docs/fundamentals/hyperparameters/ "Hyperparameters | ClearML"
[5]: https://clear.ml/docs/latest/docs/guides/reporting/artifacts/ "Artifacts Reporting | ClearML"
[6]: https://clear.ml/docs/latest/docs/references/sdk/logger/?utm_source=chatgpt.com "Logger"
[7]: https://clear.ml/docs/latest/docs/getting_started/data_management/?utm_source=chatgpt.com "Managing Your Data"
[8]: https://clear.ml/docs/latest/docs/fundamentals/agents_and_queues/ "Workers and Queues | ClearML"
[9]: https://clear.ml/docs/latest/docs/references/sdk/automation_controller_pipelinedecorator/?utm_source=chatgpt.com "PipelineDecorator"
[10]: https://clear.ml/docs/latest/docs/pipelines/pipelines_sdk_tasks/?utm_source=chatgpt.com "PipelineController"
[11]: https://clear.ml/docs/latest/docs/guides/advanced/execute_remotely/?utm_source=chatgpt.com "Remote Execution"
[12]: https://clear.ml/docs/latest/docs/getting_started/video_tutorials/hyperdatasets_data_versioning/?utm_source=chatgpt.com "Hyper-Datasets Data Versioning"
[13]: https://clear.ml/docs/latest/docs/clearml_data/data_management_examples/data_man_python/ "Data Management with Python | ClearML"
[14]: https://clear.ml/docs/latest/docs/integrations/hydra/?utm_source=chatgpt.com "Hydra"
