# AutoML Proto

本リポジトリは、表形式データを対象としたカスタム AutoML パイプラインです。設定ファイル駆動で学習・推論を制御し、複数モデルの比較・最適化・可視化を自動化します。学習コードと推論コードを明確に分離し、保守性を高めた構成にリファクタリング済みです。

## 従来の課題と設計思想

### 従来の課題
- モデル選定・特徴量前処理・ハイパーパラメータ調整を手作業で行うと、再現性が低く属人化しやすい。
- 学習と推論の設定が混在すると、実行環境や出力管理が複雑化しがち。
- モデルごとの予測比較・可視化・一括推論がスクリプト化されておらず、再利用が難しい。

### 設計思想 / 狙い
- **設定ファイル駆動**: YAML に全処理を記述し、コード改変を最小化。
- **モジュール分割**: `auto_ml/train.py` と `auto_ml/inference` パッケージで役割を分け、拡張しやすいアーキテクチャに。
- **一貫した出力管理**: 学習結果は `outputs/train/`、推論結果は `outputs/inference/` に集約。
- **再利用可能な可視化・集計**: 学習・推論時に自動でメトリクス集計や可視化を生成。

### ターゲットと想定ユースケース
- **データサイエンスチーム / アナリスト**: 少量〜中規模の tabular データで素早くベースラインを構築したい場合。
- **MLOps / プロトタイピング**: モデル比較や設定変更の頻度が高いプロジェクトでの反復検証。
- **教育用途**: AutoML の構成要素（前処理、モデル選択、評価）の理解・実践。

### 本リファクタの効果
- 学習・推論の設定ファイルを独立させたことで、用途ごとのバリエーション管理が容易に。
- 推論ロジックを `auto_ml/inference/` 以下に分割し、保守・拡張・テストがしやすくなった。
- CLI エントリーポイント (`train.py` / `inference.py`) をシンプルに保ち、実行方法を統一。

## プロジェクト構成

```text
AutoML_proto/
├─ train.py                     # ルートから学習を実行する CLI
├─ inference.py                 # 推論用 CLI（設定ファイル/CLI入力に対応）
├─ config.yaml                  # 学習設定（標準テンプレート）
├─ inference_config.yaml        # 推論設定（パラメータ探索例）
├─ outputs/                     # 学習・推論で生成される成果物
│  ├─ train/                    # 学習結果（モデル、指標、可視化）
│  └─ inference/                # 推論結果（予測 CSV、グラフ、統計）
├─ auto_ml/
│  ├─ train.py                  # 学習パイプライン本体
│  ├─ config.py                 # YAML -> dataclass 変換
│  ├─ data_loader.py            # データ読み込み・分割
│  ├─ preprocessing/            # 前処理パイプライン生成
│  ├─ model_factory.py          # モデル生成とパラメータ正規化
│  ├─ evaluation.py             # CV 評価とメトリクス
│  ├─ ensemble.py               # Stacking / Voting 構築
│  ├─ visualization.py          # 学習時の各種プロット
│  ├─ inference/                # 推論用ユーティリティ群（下記参照）
│  │  ├─ __init__.py
│  │  ├─ model_utils.py         # モデルロード、特徴量整形、単一予測
│  │  ├─ parameters.py          # パラメータ仕様の読み込みと展開
│  │  ├─ search.py              # グリッド探索・Optuna 最適化
│  │  ├─ plots.py               # 推論結果の可視化
│  │  ├─ metrics.py             # 一貫性指標の算出
│  │  └─ results.py             # CSV 書き出しユーティリティ
│  └─ ...
├─ data/example.csv             # サンプルデータ
└─ venv/                        # （任意）仮想環境
```

## 環境構築

推奨 Python バージョンは **3.10 以降** です。以下の手順でプロジェクト専用の仮想環境を構築できます。

```bash
# 1) 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate          # Windows の場合は venv\Scripts\activate

# 2) パッケージのインストール
pip install --upgrade pip
pip install \
    numpy pandas scipy scikit-learn joblib \
    lightgbm xgboost catboost \
    optuna cmaes \
    matplotlib seaborn shap \
    pyyaml tqdm

# 3) TabNet / TabPFN を利用する場合の追加（任意）
pip install pytorch-tabnet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tabpfn

# 4) 開発補助（任意）
pip install black isort mypy
```

> **メモ**: GPU を利用する場合は、PyTorch のインストールコマンドを環境に合わせて調整してください。`catboost` など一部モデルは追加ライブラリ（OpenMP 等）を必要とすることがあります。

## 実行方法

### 学習（Training）

```bash
# 既定の設定で学習
python3 train.py --config config.yaml

# パッケージ経由で学習（同等）
python3 -m auto_ml.train --config config.yaml
```

学習結果は `outputs/train/`（モデル本体、指標 CSV、画像）に保存されます。`config.yaml` をコピーして編集することで、新しい実験を管理できます。

### 推論（Inference）

```bash
# 設定ファイル駆動（推奨）
python3 inference.py --config inference_config.yaml

# CLI で直接指定する例
python3 inference.py \
    --model-dir outputs/train/models \
    --models Ridge,RandomForest \
    --input-params params.json \
    --search-method tpe \
    --n-trials 50 \
    --goal max \
    --output-dir outputs/inference
```

`inference_config.yaml` を複製し、入力モード（CSV / パラメータ探索）や探索メソッド（`grid` / `random` / `tpe` / `cmaes`）を切り替えることで、様々な推論シナリオをワンコマンドで再現できます。

## 設定ファイルの詳細

### 学習設定 (`config.yaml`)

| セクション | 主なキー | 説明 |
| --- | --- | --- |
| `data` | `csv_path`, `target_column`, `feature_columns`, `test_size`, `random_seed` | 入力データの位置と基本設定。`test_size` が 0 の場合、クロスバリデーションのみで評価。 |
| `preprocessing` | `numeric_imputation`, `categorical_imputation`, `scaling`, `categorical_encoding`, `polynomial_degree`, `target_standardize` | 数値/カテゴリの欠損補完、スケーリング、エンコーディング、ターゲット標準化などを候補リストで記述。 |
| `models` | `name`, `params`, `enable` | 探索対象モデルとハイパーパラメータ候補。リストの直積を作成し、前処理パイプラインと組み合わせて検証。 |
| `ensembles` | `stacking`, `voting` | Stacking / Voting アンサンブルの利用可否と構成。 |
| `cross_validation` | `n_folds`, `shuffle`, `random_seed` | クロスバリデーション戦略の設定。 |
| `output` | `output_dir`, `save_models`, `generate_plots`, `results_csv` | 学習出力の保存場所、内容。 |
| `evaluation` | `regression_metrics`, `classification_metrics`, `primary_metric` | 指標の種類と最優先指標。 |
| `optimization` | `method`, `n_iter` | ハイパーパラメータ探索戦略（`grid` / `random` / `bayesian`）。 |
| `interpretation` / `visualizations` | `compute_feature_importance`, `compute_shap`, 各種フラグ | 解釈性指標と可視化の有効化。 |

### 推論設定 (`inference_config.yaml`)

| セクション | 主なキー | 説明 |
| --- | --- | --- |
| `model_dir` | - | 読み込む学習済みモデル（`.joblib`）を格納したディレクトリ。 |
| `models` | `name`, `enable` | 推論に使用するモデルを絞り込み。ファイル名プレフィックスと一致させる。 |
| `input` | `mode`, `csv_path`, `variables`, `params_path` | 推論データの取得方法。`mode: csv` でファイル指定、`mode: params` でパラメータ探索。 |
| `search` | `method`, `n_trials`, `goal` | Optuna を用いた探索手法と試行回数、最大化/最小化方向。 |
| `output_dir` | - | 推論結果 CSV やグラフを保存するディレクトリ。 |

## モデル一覧と活用メモ

| モデル | 概要 | 主なメリット | 主なデメリット | 主ライブラリ | 推奨データ規模 | 活用事例 / 参考リンク |
| --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 最小二乗による線形回帰。 | 実装が軽量・解釈しやすい。 | 非線形関係や外れ値に弱い。 | scikit-learn | 数十〜数万件 | [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) |
| Ridge | L2 正則化付き線形回帰。 | 多重共線性に強く安定。 | 係数が縮小し解釈しづらい場合あり。 | scikit-learn | 数十〜数万件 | [Hoerl & Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634) |
| Lasso | L1 正則化付き線形回帰。 | 特徴量選択を自動で実施。 | 強い相関がある特徴量が任意に選択される。 | scikit-learn | 数十〜数万件 | [Tibshirani (1996)](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) |
| ElasticNet | L1/L2 を組み合わせた線形モデル。 | Lasso と Ridge の折衷で汎用的。 | ハイパーパラメータが増える。 | scikit-learn | 数十〜数万件 | [Zou & Hastie (2005)](https://doi.org/10.1111/j.1467-9868.2005.00503.x) |
| SVR | カーネルサポートベクタ回帰。 | 非線形パターンの表現力。 | 計算コストが高くスケール調整が必須。 | scikit-learn | 数百〜数万件 | [Cortes & Vapnik (1995)](https://doi.org/10.1007/BF00994018) |
| RandomForest | 決定木のバギング。 | 過学習しづらく特徴量重要度が得られる。 | 高次元データでの解釈が難しい。 | scikit-learn | 数千〜数十万件 | [Breiman (2001)](https://doi.org/10.1023/A:1010933404324) |
| ExtraTrees | 極端ランダム木。 | 高速・高精度になりやすい。 | ノイズへの感度がやや高い。 | scikit-learn | 数千〜数十万件 | [Geurts et al. (2006)](https://doi.org/10.1007/s10994-006-6226-1) |
| GradientBoosting | 勾配ブースティング決定木。 | 高精度で柔軟。 | 学習時間が長い、ハイパーパラメータ依存。 | scikit-learn | 数千〜数十万件 | [Friedman (2001)](https://doi.org/10.1214/aos/1013203451) |
| LightGBM | 勾配ブースティング（Leaf-wise）。 | 大規模データでも高速。 | 小規模データでは過学習に注意。 | lightgbm | 数千〜数百万件 | [LightGBM Paper](https://www.microsoft.com/en-us/research/publication/lightgbm-a-highly-efficient-gradient-boosting-decision-tree/) |
| XGBoost | 勾配ブースティング（正則化付き）。 | 汎用性が高く実績豊富。 | ハイパーパラ調整が煩雑。 | xgboost | 数千〜数百万件 | [Chen & Guestrin (2016)](https://doi.org/10.1145/2939672.2939785) |
| CatBoost | カテゴリ特徴量に強いブースティング。 | エンコーディング不要で性能安定。 | ライブラリサイズが大きい。 | catboost | 数千〜数百万件 | [Dorogush et al. (2018)](https://arxiv.org/abs/1810.11363) |
| GaussianProcess | ガウス過程回帰。 | 不確実性の推定が可能。 | 大規模データで計算負荷が急増。 | scikit-learn | 〜数千件 | [Rasmussen & Williams (2006)](http://www.gaussianprocess.org/gpml/) |
| KNeighbors | k近傍回帰。 | 実装が簡単・局所構造に敏感。 | 特徴量スケールと次元の呪いに弱い。 | scikit-learn | 数百〜数万件 | [Cover & Hart (1967)](https://ieeexplore.ieee.org/document/1053964) |
| MLP | 多層パーセプトロン。 | 非線形表現力が高い。 | ハイパーパラ調整が難しく収束に時間。 | scikit-learn | 数千〜数十万件 | [Rumelhart et al. (1986)](https://www.nature.com/articles/323533a0) |
| TabNet | 注意機構ベースの表形式 NN。 | 特徴選択とインタープリタビリティを両立。 | 学習が不安定でハードウェア要件が高い。 | pytorch-tabnet | 数万〜数百万件 | [Arik & Pfister (2021)](https://arxiv.org/abs/1908.07442) |
| TabPFN | 事前学習済みトランスフォーマ。 | 少量データで強力な性能。 | モデルサイズが大きく依存関係が多い。 | tabpfn | 数十〜数千件 | [Hollmann et al. (2023)](https://arxiv.org/abs/2209.13474) |
| Stacking Ensemble | 複数モデルのメタ学習。 | 複雑な関係を学習し精度向上が見込める。 | 構築・チューニングが難しい。 | scikit-learn | モデル数に依存 | [Wolpert (1992)](https://doi.org/10.1016/S0893-6080(05)80023-1) |
| Voting Ensemble | ハード/ソフト投票のアンサンブル。 | 実装が簡単で安定化に有効。 | 構成モデルに性能が依存。 | scikit-learn | モデル数に依存 | [Kuncheva (2004)](https://doi.org/10.1002/0471660264) |

> **想定データ規模** は線形的な目安です。実際には特徴量数や前処理内容、ハードウェアに依存します。

## 処理面での工夫点

| 項目 | 内容 | 効果 |
| --- | --- | --- |
| 設定ファイル駆動 | `config.yaml` / `inference_config.yaml` に全処理を定義。 | コードに手を入れずに実験条件を再現・比較できる。 |
| 学習・推論の明確な分離 | `auto_ml/train.py` と `auto_ml/inference/` を分割。 | 役割間の干渉を防ぎ、保守やテストが容易に。 |
| 動的な特徴量整列 | 推論時に必須列を推定し DataFrame を整形。 | モデル保存時と入力列が多少異なっても推論を継続可能。 |
| アンサンブル内製化 | Stacking / Voting を標準で提供。 | ベンチマーク超過や安定化をワンコマンドで実施。 |
| 可視化と指標の自動保存 | 学習・推論ともに CSV と PNG を自動生成。 | 結果レビューやレポーティングが迅速化。 |
| Optuna 最適化の統合 | `grid` / `random` / `tpe` / `cmaes` を選択可能。 | 推論段階でのパラメータ探索や感度分析を自動化。 |

## 次のステップ

- 追加データセット毎に `config.yaml` をコピーし、`data.csv_path` やモデル候補を調整してください。
- 環境差異がある場合は、`requirements` ファイルの整備や Docker 化を検討するとより再現性が向上します。
- notebook などと組み合わせる際は、生成物 (`outputs/` 配下) をバージョン管理・共有することでチーム内コミュニケーションを円滑化できます。
