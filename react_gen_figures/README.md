# react_gen の反応・処理概念図

対象：[dkawa2523/react_gen — update2_db](https://github.com/dkawa2523/react_gen/tree/update2_db)。

元の生成画像8枚を基準に、分子の大きさと配置、原子・結合・電荷・電子、処理の接続を再構成した日本語スライドです。既存の反応概念図8枚に、処理・判定・外部データ・実行結果を説明する12枚を追加した全20枚です。

## ダウンロード

- [編集用 PowerPoint](react_gen_editable_figures_ja.pptx)
- [閲覧用 PDF](react_gen_editable_figures_ja.pdf)
- [PowerPoint・PDF・全20枚のPNG・実行結果のZIP](react_gen_PowerPoint.zip)

PPTX がコードエディターで開く場合は、ZIP を保存・展開し、PowerPoint の「開く」から PPTX を選択してください。

## 編集できる要素

原子、結合線、電荷、電子、イオンの括弧、矢印、文字、表のセルは PowerPoint の図形です。分子はグループ化しており、グループ解除で各パーツを編集できます。断面積グラフは独立した画像として配置しています。日本語フォントは Noto Sans CJK JP を使用しています。

## 収録ファイル

| 場所 | 内容 |
| --- | --- |
| `images/slides/` | 修正版スライドの PNG 20枚（1536 × 1024） |
| `images/originals/` | 再構成の基準となった元の生成画像8枚 |
| `images/graphs/` | スライドに配置した個別グラフ画像2枚 |

図は反応機構・データ構造・処理を説明する模式図です。断面積曲線は計算値や測定値ではありません。登録反応の探索、任意の熱化学補完、DNT 入力の出力を説明しており、外部の速度論・DNT ソルバーによる計算結果は含みません。各スライドのノートに実装の参照先と補足を記載しています。

## 実装と実行条件

追加資料は `update2_db` のコミット `62c20092f48e45a1299dd3330b48c56223008177` を対象に確認しました。処理フロー・比較表・文字・矢印は個別に編集可能です。

付属3ケースを `--registry registry` 明示、配布パック未使用で実行しました。結果は `case_results/` に収録しています。反応一覧の構築、断面積の登録、DNT入力の準備を別々に評価しています。反応数は当該登録簿の収録・到達状況であり、実際の化学の網羅率や予測精度を表しません。

## スライド一覧

### 01 入力ガスから反応ネットワークへ

![入力ガスから反応ネットワークへ](images/slides/01_network_generation.png)

[元の生成画像](images/originals/01_network_generation.png)

### 02 電子衝突による反応チャネル

![電子衝突による反応チャネル](images/slides/02_electron_collision.png)

[元の生成画像](images/originals/02_electron_collision.png)

### 03 重粒子間の反応チャネル

![重粒子間の反応チャネル](images/slides/03_heavy_particle_reactions.png)

[元の生成画像](images/originals/03_heavy_particle_reactions.png)

### 04 反応記録と速度モデルの対応

![反応記録と速度モデルの対応](images/slides/04_reaction_rate_data.png)

[元の生成画像](images/originals/04_reaction_rate_data.png)

### 05 生成エンタルピーから反応エネルギーを補完

![生成エンタルピーから反応エネルギーを補完](images/slides/05_thermochemistry.png)

[元の生成画像](images/originals/05_thermochemistry.png)

### 06 イオン–中性種反応の DNT 入力

![イオン–中性種反応の DNT 入力](images/slides/06_dnt_inputs.png)

[元の生成画像](images/originals/06_dnt_inputs.png)

### 07 反応候補の生成と確認項目

![規則に基づく反応候補の生成](images/slides/07_rule_based_candidates.png)

[元の生成画像](images/originals/07_rule_based_candidates.png)

### 08 出典データから再利用可能な反応知識へ

![出典データから再利用可能な反応知識へ](images/slides/08_data_curation.png)

[元の生成画像](images/originals/08_data_curation.png)

### 09 コードが担う範囲と、専門家が担う範囲

![コードが担う範囲と、専門家が担う範囲](images/slides/09_code_scope.png)

### 10 反応網の展開：新しく現れた種を起点に繰り返す

![反応網の展開：新しく現れた種を起点に繰り返す](images/slides/10_expansion_workflow.png)

### 11 判定レイヤー①：反応一覧へ入れる条件

![判定レイヤー①：反応一覧へ入れる条件](images/slides/11_decision_layers.png)

### 12 機械的な反応式生成：現在のテンプレートは2種類

![機械的な反応式生成：現在のテンプレートは2種類](images/slides/12_reaction_templates.png)

### 13 推定候補は、どの段階でふるいにかけられるか

![推定候補は、どの段階でふるいにかけられるか](images/slides/13_candidate_screening.png)

### 14 保存則で分かること：CF₄ の解離電離を例に

![保存則で分かること：CF₄ の解離電離を例に](images/slides/14_conservation_check.png)

### 15 外部データは「何を判断したいか」で必要量が変わる

![外部データは「何を判断したいか」で必要量が変わる](images/slides/15_external_data_requirements.png)

### 16 データ補完：断面積の対応付けと、ΔE の計算を分離

![データ補完：断面積の対応付けと、ΔE の計算を分離](images/slides/16_data_enrichment.png)

### 17 DNT の準備判定：物性がそろっても、まだ不足し得る

![DNT の準備判定：物性がそろっても、まだ不足し得る](images/slides/17_dnt_readiness.png)

### 18 判定レイヤー②：出力を何の判断に使うか

![判定レイヤー②：出力を何の判断に使うか](images/slides/18_output_interpretation.png)

### 19 付属ケースの実行で見える、実用上の到達点

![付属ケースの実行で見える、実用上の到達点](images/slides/19_case_results.png)

### 20 プラズマ研究での使い方：レビューとデータ整備を反復する

![プラズマ研究での使い方：レビューとデータ整備を反復する](images/slides/20_expert_workflow.png)
