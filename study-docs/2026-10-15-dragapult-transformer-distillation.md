# ドラパルトのGBDTからTransformerへの蒸留

2026年10月15日実施。[400試合版のGBDT模倣学習](2026-10-15-dragapult-imitation.md)で作成したモデルを教師に、新しく400試合を収集してTransformerを学習した。学習・モデル選択に使わなかったテスト40試合で、GBDT教師との選択一致率は85.2%だった。PolicyとValueを持つモデルをC++用に出力し、対戦とPython/C++推論の動作確認まで完了した。

既存の学習スクリプトを使用し、今回の作業では元の実装・ルートREADMEを変更していない。PPOの更新はまだ行っていない。

## 蒸留で何を学ぶか

前回のGBDTはルール方策の選択を模倣した。今回は、そのGBDTが実際に対戦したときの観測・合法手・候補スコア・選択・終局結果をTransformerへ学習させる。

- **Policy** は合法手の選び方を学ぶ。教師が選んだ1手だけでなく、候補スコアから作る確率分布も模倣する。
- **Value** はその局面から得られる最終結果を予測する。教師データの手番プレイヤーが勝った場合を+1、引き分けを0、負けを−1として学習する。Value自体は勝率ではない。

学習損失は `KL + 0.1 × 行動NLL + 0.25 × Valueの交差エントロピー`。KLは各判断の最初の選択における教師分布との違いを測り、行動NLLは複数選択の順序やSTOPを含む選択列を扱う。Valueは51段階の分布を出力し、その期待値を局面の評価値とする。

この段階のValueはGBDT教師の対戦結果から学んだ予測である。Transformer自身の方策で対戦して更新するPPOとは、データの集め方が異なる。

## 教師データと分割

教師モデルは `outputs/dragapult/imitation_v2/model/model.gbdt`。教師モデルIDは `4a22ba62d79b2063582da46458993f7d88e9189e8ed5fd61ea31fb355f6c246b`。

相手はメガルカリオex、ドラパルトex、ナンジャモ、メガユキノオーexの公式ルール方策。各対面100試合を両席50試合ずつ収集した。ドラパルト相手もGBDT教師対ルール方策であり、教師同士の自己対戦ではない。収集seedは `202612000` から、収集並列数は4。400試合のエージェント・エンジンエラーは0だった。

分割seed `204` で、試合単位に学習80%・検証10%・テスト10%へ分けた。同じ試合が複数の分割へ混ざらないことを学習時にも確認している。

| 分割 | 試合数 | 判断数 | 選択クエリ数 | 候補が1つだけのクエリ数 |
|---|---:|---:|---:|---:|
| 学習 | 320 | 25,439 | 26,412 | 1,103 |
| 検証 | 40 | 3,118 | 3,236 | 156 |
| テスト | 40 | 3,393 | 3,529 | 117 |

複数選択では、1つの判断から複数の選択クエリが生じる。今回の蒸留と一致率の集計には、候補が1つだけの選択も含まれる。前回のGBDT学習ではそれらを除外しており、教師も評価局面も違うため、前回の「ルール方策との一致率83.1%」と今回の数値は直接比較できない。

抽出時には、GBDT特徴量・候補順・選択順・選んだ候補を記録と照合した。教師スコアのPython/C++最大絶対差は0で、許容誤差は絶対・相対とも `1e-10`。観測は手番プレイヤーに見える情報を使う。

## モデルと学習条件

| 項目 | 設定 |
|---|---|
| モデル | 128次元・6層・4ヘッドのTransformer |
| パラメータ数 | 980,916 |
| トークン | Global・Board・Zone・Option・STOP |
| 1トークンの入力幅 | 198 |
| 最大トークン数 | 構造上120、今回のデータでは最大57 |
| 出力 | Pointer Policyと51段階のValue分布 |
| エポック数 | 5 |
| バッチサイズ | 256判断 |
| 学習率 | 0.0003 |
| 教師分布の温度 | 1.0 |
| 行動損失とValue損失の係数 | 0.1と0.25 |
| 最適化 | AdamW、weight decay 0.0001、勾配clip 0.5 |
| 学習seed | 204 |
| 更新回数 | 500 |
| 学習環境 | PyTorch 2.14.0+cu130、NVIDIA GeForce RTX 2080 Ti |
| PyTorch CPUスレッド数 | 4 |

検証データでの合計損失が最小のチェックポイントを選んだ。テストデータはモデル選択に使っていない。実行時間の性能測定は行っていない。

## 学習経過とテスト結果

| エポック | 学習損失 | 検証損失 | 検証の教師一致率 | 検証Value MSE |
|---|---:|---:|---:|---:|
| 1 | 0.7330 | 0.4035 | 70.3% | 0.9752 |
| 2 | 0.3465 | 0.3199 | 80.8% | 0.9640 |
| 3 | 0.3039 | 0.2994 | 82.8% | 0.9025 |
| 4 | 0.2739 | 0.2825 | 83.7% | 0.8468 |
| 5 | 0.2556 | 0.2735 | 85.4% | 0.8086 |

5エポック目を採用した。テスト40試合での結果は次のとおり。

| 指標 | 値 |
|---|---:|
| GBDT教師との選択一致率 | 85.2% |
| 合計損失 | 0.2455 |
| 教師分布とのKL | 0.0340 |
| 選択列のNLL | 0.8808 |
| Valueの交差エントロピー | 0.4935 |
| Valueの平均二乗誤差 | 0.6575 |

一致率は選択クエリ単位であり、1つの行動全体を正しく再現した割合ではない。Value MSEは各判断の予測値と終局結果の二乗誤差の平均で、勝率や「正解率」への読み替えはできない。

今回は教師の模倣と実行の確認であり、教師より強くなったことや勝率向上を示す評価ではない。また、教師が到達した局面での一致率と、Transformer自身の対戦で到達する局面の質は別の問題になる。

## C++出力と対戦の動作確認

`model.pt` を単独で利用できるC++用の `model.bin` に出力した。バイナリのサイズは3,927,779バイト。出力したバイナリだけを別の一時ディレクトリへコピーして対戦を行い、単独で推論できることを確認した。

| 組み合わせ | 試合数 |
|---|---:|
| Transformer対GBDT教師 | 2 |
| GBDT教師対Transformer | 2 |
| Transformer同士 | 2 |

上記6試合はエージェント・エンジンエラー0で完走した。さらにTransformer同士の2試合を同じseedで再実行し、リプレイのSHA-256が一致した。実行回数は計8試合で、推論照合の対象は再実行分を除く6試合。

| Python/C++の照合項目 | 結果 |
|---|---:|
| 判断数と選択クエリ数 | 827判断・858クエリ |
| トークン行数 | 22,327 |
| 特徴量の最大絶対差 | 約2.38e-7 |
| Policyスコアの最大絶対差 | 約2.49e-6 |
| float32計算でのPolicyスコア最大絶対差 | 約3.11e-6 |
| Valueの最大絶対差 | 約1.66e-6 |
| 行動選択の不一致 | 0 |

特徴量の許容誤差は絶対・相対とも `2e-6`。PolicyとValueの許容誤差は絶対・相対とも `2e-5` で、すべて通過した。保存された重みはfloat32であり、照合ではC++に合わせたPythonの倍精度計算と、Pythonのfloat32計算の両方を確認した。

追加でテストデータから1,000クエリ・26,939トークン行をC++ probeへ渡し、Policyスコアを照合した。最大絶対差は約2.41e-6。テストデータ全クエリをprobeで確認したわけではない。

## 再実行コマンド

環境とC++ビルド、400試合版のGBDTモデルを用意し、リポジトリ直下で実行する。出力先は未使用のものを指定する。GPU学習にはCUDAが利用可能な実行環境が必要で、今回の環境ではサンドボックス外でGPUを利用した。

```bash
.venv/bin/python src/agents/dragapult/scripts/collect_replays.py \
  --output-dir outputs/dragapult/distillation_v1/teacher_replays \
  --teacher dragapult_gbdt \
  --model outputs/dragapult/imitation_v2/model/model.gbdt \
  --games-per-opponent 100 --seed 202612000 --workers 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python src/agents/dragapult/transformer/scripts/extract.py \
  --replay-dir outputs/dragapult/distillation_v1/teacher_replays \
  --model outputs/dragapult/imitation_v2/model/model.gbdt \
  --output-dir outputs/dragapult/distillation_v1/features \
  --seed 204 --shard-games 25 --workers 4 \
  --validation-fraction 0.1 --test-fraction 0.1

.venv/bin/python src/agents/dragapult/transformer/scripts/train.py \
  --feature-dir outputs/dragapult/distillation_v1/features \
  --output-dir outputs/dragapult/distillation_v1/model \
  --epochs 5 --batch-size 256 --learning-rate 0.0003 \
  --temperature 1 --hard-weight 0.1 --value-weight 0.25 \
  --seed 204 --device cuda

.venv/bin/python src/agents/dragapult/transformer/scripts/export.py \
  --checkpoint outputs/dragapult/distillation_v1/model/model.pt \
  --output outputs/dragapult/distillation_v1/model/model.bin

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python src/agents/dragapult/transformer/scripts/validate.py \
  --checkpoint outputs/dragapult/distillation_v1/model/model.pt \
  --model outputs/dragapult/distillation_v1/model/model.bin \
  --teacher-model outputs/dragapult/imitation_v2/model/model.gbdt \
  --output-dir outputs/dragapult/distillation_v1/validation \
  --feature-dir outputs/dragapult/distillation_v1/features \
  --probe-queries 1000
```

## 成果物と次の段階

以下のパスは実行環境内の成果物で、データ・重み・実行レポートはGit管理対象外。

| 内容 | リポジトリ直下からのパス |
|---|---|
| 教師対戦の記録 | `outputs/dragapult/distillation_v1/teacher_replays/manifest.json` |
| データ分割と抽出結果 | `outputs/dragapult/distillation_v1/features/manifest.json` |
| 学習済みチェックポイント | `outputs/dragapult/distillation_v1/model/model.pt` |
| C++用モデル | `outputs/dragapult/distillation_v1/model/model.bin` |
| 学習経過とテスト結果 | `outputs/dragapult/distillation_v1/model/training.json` |
| 対戦と決定性の動作確認 | `outputs/dragapult/distillation_v1/validation/validation.json` |
| Python/C++照合結果 | `outputs/dragapult/distillation_v1/validation/parity.json` |

チェックポイントのSHA-256は `52c76f6c8e0c57fead013b6e098c56f9add4aa8dcd785e60fad4c045d6a98061`。

次の段階は、このPolicyとValueを出発点としてTransformer自身の新しい対戦を収集し、収集時の行動確率・Value・終局結果からGAEを計算してPPOを更新すること。今回の蒸留用対戦はGBDT教師が収集したものであり、そのままTransformerのon-policy PPO rolloutとしては使えない。
