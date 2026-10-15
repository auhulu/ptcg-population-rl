# ドラパルトの教師対戦を400試合に増やした模倣学習

2026年10月15日実施。ドラパルトexのルール方策を教師にしてGBDTを学習し、教師対戦を80試合から400試合へ増やした。検証20試合を固定した比較で、教師との選択一致率は全体で73.1%から83.1%、MAINで64.8%から78.6%になった。

今回の段階は模倣学習であり、PPOによる更新ではない。モデルは勝敗を報酬として改善するのではなく、ルール方策が選んだ候補を他の合法手より上位にするように学習する。

## 学習と検証の条件

教師はC++版ドラパルトのルール方策。相手はメガルカリオex、ドラパルトex、ナンジャモ、メガユキノオーexの公式ルール方策とし、各対面で両方の席を均等に使った。

| 項目 | 80試合版 | 400試合版 |
|---|---:|---:|
| 各対面の試合数 | 20 | 100 |
| 合計試合数 | 80 | 400 |
| 学習用の試合数 | 60 | 380 |
| 検証用の試合数 | 20 | 20 |
| 学習用の選択クエリ数 | 6,099 | 39,824 |
| 検証用の選択クエリ数 | 2,280 | 2,280 |

最初の80試合はseed `202610150` から収集し、分割seed `42`、検証比率25%で試合単位に分割した。追加320試合はseed `202611000` から収集し、すべて学習用へ追加した。検証20試合の所属、特徴量ファイル、選択メタデータのハッシュは前後で完全一致し、学習用との試合の重複もない。

両モデルの学習条件は同じ `quick` 設定とした。

| 設定 | 値 |
|---|---|
| 学習方式 | LightGBM LambdaRank |
| 木の分け方 | MAINとそれ以外の2系統 |
| 特徴量数 | 5,078 |
| 学習率 | 0.08 |
| 最大葉数と最大深さ | 15と5 |
| 最大学習ラウンド | 200 |
| 早期終了 | 検証指標が10ラウンド改善しなければ終了 |
| 学習seedとCPUスレッド数 | 42と4 |
| 検証データでの再学習 | なし |

実際に採用されたラウンド数は、80試合版がMAIN 60・それ以外32、400試合版がMAIN 163・それ以外77だった。

## MAINと一致率の意味

MAINは、自分の番にカードを使う、特性を使う、にげる、ワザを使うなど「次に何をするか」を選ぶ場面。それ以外には、行動の対象や効果で選ぶカードなどの追加選択が含まれる。

`query_top1` は、モデルが最上位とした候補が教師の選択と一致した割合。複数選択の途中段階も別のクエリとして数える。候補が1つしかない選択は除外するため、行動全体の一致率や勝率とは異なる。

## 同じ検証データでの結果

| 選択の種類 | 検証クエリ数 | 80試合版 | 400試合版 | 差 |
|---|---:|---:|---:|---:|
| MAIN | 1,055 | 64.8% | 78.6% | +13.7ポイント |
| それ以外 | 1,225 | 80.2% | 86.9% | +6.7ポイント |
| 全体 | 2,280 | 73.1% | 83.1% | +10.0ポイント |

差は丸める前の値から計算した。全体の一致率は、2種類の一致率の単純平均ではなく、クエリ数で重み付けした値。

今回の条件では、教師対戦を増やすことで同じ検証局面での模倣精度が改善した。ただし検証データは早期終了にも使用しており、独立したテストデータではない。また、今回の対戦は動作確認が目的なので、勝率向上や対戦の強さはこの結果から判断しない。

## 対戦とPython C++の動作確認

400試合版を公式4方策と各2試合、計8試合対戦させ、学習済みモデル同士でも2試合対戦させた。対戦seedは公式方策相手が `202610300` から、モデル同士が `202610400` からで、80試合版の動作確認と同じ設定を使った。

| 確認項目 | 400試合版の結果 |
|---|---:|
| 完走した対戦 | 10試合 |
| エージェントエラー | 0 |
| エンジンエラー | 0 |
| 照合した判断 | 994 |
| 照合した選択クエリ | 1,034 |
| 対戦から照合した候補行 | 6,726 |
| Python/C++特徴量の最大絶対差 | 0 |
| Python/C++予測値の最大絶対差 | 0 |
| Python/C++選択の不一致 | 0 |

予測値の許容誤差は絶対・相対とも `1e-10`。加えて、初回80試合から抽出した54,654候補行を新モデルへ入力し、C++ probeとPythonの予測値が一致した。400試合分の特徴量自体のPython/C++一致は抽出時に確認した。新しい学習データ全候補行を新モデルのprobeへ再投入したわけではない。

検証セットを固定する `--split-manifest` の追加についても、既存の分割の保持、分割間の重複拒否、既存試合の欠落とスキーマ変更の拒否を3件のユニットテストで確認した。変更したPythonファイルのRuffチェックも通過した。

## 再実行の手順

環境準備・C++ビルドと80試合版の作成は[ルートの手順](../README.ja.md)を参照。以下は80試合版のローカル成果物がある状態で、リポジトリ直下から実行したコマンド。出力先は未使用のものを指定する。

```bash
.venv/bin/python src/agents/dragapult/scripts/collect_replays.py \
  --output-dir outputs/dragapult/imitation_v2/additional_replays \
  --games-per-opponent 80 --seed 202611000 --workers 4

mkdir -p outputs/dragapult/imitation_v2/rule_replays
cp -al outputs/dragapult/imitation_v1/rule_replays outputs/dragapult/imitation_v2/rule_replays/base
cp -al outputs/dragapult/imitation_v2/additional_replays outputs/dragapult/imitation_v2/rule_replays/additional

.venv/bin/python src/agents/dragapult/gbdt/scripts/extract.py \
  --replay-dir outputs/dragapult/imitation_v2/rule_replays \
  --output-dir outputs/dragapult/imitation_v2/features \
  --split-manifest outputs/dragapult/imitation_v1/features/manifest.json

.venv/bin/python src/agents/dragapult/gbdt/scripts/train.py \
  --feature-dir outputs/dragapult/imitation_v2/features \
  --output-dir outputs/dragapult/imitation_v2/model \
  --profile quick --rounds 200 --threads 4 --seed 42
```

`cp -al` はLinux上で収集済みリプレイへのハードリンクを作る。元データとリンク先は同じファイルを指すため、作成後は両方とも変更しない。

学習済みモデルの動作確認には以下を実行した。

```bash
.venv/bin/python src/agents/dragapult/scripts/collect_replays.py \
  --output-dir outputs/dragapult/imitation_v2/validation/rule_opponents \
  --teacher dragapult_gbdt \
  --model outputs/dragapult/imitation_v2/model/model.gbdt \
  --games-per-opponent 2 --seed 202610300 --workers 2

build/arena_cpp \
  --agent-a dragapult_gbdt --agent-b dragapult_gbdt \
  --deck-a src/agents/dragapult/deck.csv --deck-b src/agents/dragapult/deck.csv \
  --model-a outputs/dragapult/imitation_v2/model/model.gbdt \
  --model-b outputs/dragapult/imitation_v2/model/model.gbdt \
  --games 2 --seed 202610400 \
  --replay-dir outputs/dragapult/imitation_v2/validation/self

.venv/bin/python src/agents/dragapult/gbdt/scripts/audit.py \
  --replay-dir outputs/dragapult/imitation_v2/validation \
  --model outputs/dragapult/imitation_v2/model/model.gbdt \
  --output outputs/dragapult/imitation_v2/parity.json \
  --feature-dir outputs/dragapult/imitation_v1/features \
  --probe build/dragapult_gbdt_probe
```

## 保存した成果物

以下は実行環境内のパスで、データ・重み・実行レポートはGit管理対象外。上記コマンドでモデルと照合結果を作成できる。`training.json` はモデルに同梱された学習情報を書き出したもの、`comparison.json` は両モデルの学習情報と分割情報から今回別途作成した比較レポート。

| 内容 | リポジトリ直下からのパス |
|---|---|
| 前回モデル | `outputs/dragapult/imitation_v1/model/model.gbdt` |
| 今回モデル | `outputs/dragapult/imitation_v2/model/model.gbdt` |
| 分割と特徴量の記録 | `outputs/dragapult/imitation_v2/features/manifest.json` |
| 学習結果 | `outputs/dragapult/imitation_v2/training.json` |
| 前回との比較 | `outputs/dragapult/imitation_v2/comparison.json` |
| Python/C++の照合結果 | `outputs/dragapult/imitation_v2/parity.json` |

今回の実行に対応するコード・手順のコミットは `e4520c2`、80試合版の手順は `73c88c2`。

## 今回の学びと次の課題

データを増やす比較では、検証局面を固定すると「評価対象が変わったために数値が変わった」という要因を避けられる。試合単位で分割することも、同じ試合の近い盤面が学習用と検証用に混ざるのを防ぐために必要。

次に学習内容を理解するなら、まだ教師と一致しないMAINの局面を確認し、どの行動の選択を間違えたかを見るとよい。PPOへ進む前には、GBDTを教師としたTransformer蒸留でPolicyとValueを用意する段階が残っている。
