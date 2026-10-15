# ドラパルトのTransformerをPPOで更新する

2026年10月15日実施。[Transformerへの蒸留](2026-10-15-dragapult-transformer-distillation.md)で作成したPolicyとValueを出発点として、新しく対戦を収集してPPOで更新する。既存のスクリプトを使用し、元の実装・ルートREADMEは変更していない。

240試合の収集、GAE、32回のPPO更新、C++用重みの出力、更新後の対戦とPython/C++照合まで完了した。学習が動くことと、強くなることは別であり、この1サイクルだけで勝率向上は判断しない。

## 蒸留からPPOへ変わるところ

蒸留はGBDT教師の選択を模倣する学習だった。PPOでは、更新前のTransformer自身が確率的に選んだ行動と、その対戦結果から学ぶ。過去のGBDT教師の対戦記録をそのままon-policyデータとして使うことはできない。

今回の1サイクルは次の順で進む。

1. 蒸留済みPolicyとValueを重み変更なしでコピーし、収集方策を固定する。
2. C++で新しい対戦を行い、各選択の前の観測・合法手・選択と、収集時の対数行動確率・Value・終局結果を記録する。
3. 記録を抽出し、Pythonで収集方策を再計算してC++の確率・Valueとの整合性を確認する。
4. GAEを計算し、GPUでPPOの更新を行う。
5. 更新した重みをC++用に出力し、対戦とPython/C++推論の動作確認を行う。

収集方策とデータはSHA-256で対応づける。学習時にはチェックポイント・rollout・カード構成・特徴量の契約を照合し、収集時と異なるモデルを誤って更新しないよう確認する。複数選択では選択順とSTOPを含む選択列全体の対数確率を使う。

## GAEとPPOの読み方

報酬は勝ち+1、引き分け0、負け−1、途中報酬0。Valueは現在の方策でその局面からプレイした場合の将来の報酬の予測であり、勝率そのものではない。

GAEは「その行動の後の結果が、局面の予測よりどれくらいよかったか」を推定する。各プレイヤーの判断列について、終局側から逆順に計算する。

```text
δ[t] = reward[t] + γ × continuation[t] × Value[t+1] − Value[t]
Advantage[t] = δ[t] + γ × λ × continuation[t] × Advantage[t+1]
```

`continuation` は終局で0、それ以外で1。今回の `γ=1` は将来の報酬を割り引かず、`λ=0.99` は後の予測のずれも強く反映する設定。Value用の学習目標は、正規化前の `Advantage + old Value` から作る。Policyの更新に使うAdvantageは学習対象の判断の平均・標準偏差で正規化する。

PPOでは、収集時と更新中の方策が同じ選択列に与える確率の比を使う。

```text
ratio = exp(new_log_probability − old_log_probability)
```

Advantageが正ならその選択を増やし、負なら減らす方向へ更新する。Policyのclip幅0.2は目的関数の確率比を0.8〜1.2に制限した項と比較するための幅であり、実際の確率比が必ずこの範囲に収まる保証ではない。Valueも収集時の予測からの変化をclipした誤差と比較する。実装では51段階のValue分布への交差エントロピーも使う。Entropy項は選択分布が過度に集中するのを抑える方向に働く。

## 対戦相手と実行条件

| 相手 | 試合数 | 方策 |
|---|---:|---|
| メガルカリオex | 40 | 公式ルール方策 |
| ドラパルトex | 40 | 公式ルール方策 |
| ナンジャモ | 40 | 公式ルール方策 |
| メガユキノオーex | 40 | 公式ルール方策 |
| ドラパルトexのGBDT | 40 | 400試合版の固定モデル |
| 自己対戦 | 40 | 今回の収集時点で固定したTransformer同士 |

相手の重みは均等で、各相手について両席20試合ずつ。自己対戦は両プレイヤーの判断列を収録するため、240試合から280エピソードを得た。ここでエピソードは1人のプレイヤーの1試合分の判断列を指す。

| 項目 | 設定 |
|---|---|
| 初期モデル | `outputs/dragapult/distillation_v1/model/model.pt` |
| 固定GBDT相手 | `outputs/dragapult/imitation_v2/model/model.gbdt` |
| サイクル数・収集試合数 | 1サイクル・240試合 |
| 収集並列数 | 4ワーカー |
| BLAS・OpenMPスレッド数 | 各1 |
| PPO学習のPyTorch CPUスレッド数 | 1 |
| 学習デバイス | CUDA、NVIDIA GeForce RTX 2080 Ti |
| 収集seed・行動サンプリングseed | 202640000・202650000 |
| 更新seed | 204 |
| 行動サンプリング温度 | 1 |
| 更新回数上限 | 32 |
| バッチ・マイクロバッチ | 512判断・64判断 |
| 診断用holdout指定数 | 1,024判断 |
| 学習率 | 0.00003 |
| γ・GAE λ | 1・0.99 |
| Policy clip・Value clip | 0.2・0.2 |
| Value係数・Entropy係数 | 0.25・0.01 |
| 最適化 | AdamW、weight decay 0.0001、勾配clip 0.5 |

holdoutは最大1,024判断、かつ全判断数の約10%までで、エピソード・ステップ順に等間隔で選び、最適化対象から除外する。これは更新前後の方策の変化をKLで診断するための判断単位の集合である。同じ試合の別の判断は学習に含まれるため、独立した試合での汎化性能の検証ではない。

設定ファイルには `kl_target` があるが、今回使用する `run_cycle.py` の学習ではKLによる早期停止に使われない。1 epochで学習判断を重複使用せず、最大32回・16,384判断まで更新する。データが尽きた場合は32回未満で終了する。実行時間の性能測定は行っていない。

## 並列化について

収集は今回4ワーカーだったが、実行環境はRyzen 7 5700X（8コア・16論理CPU）、メモリ約47 GiB。6相手×2席の12ジョブがあるため、既存の `--workers` で8または12に増やせる。今回の構成では12を超えても同時実行ジョブ数は増えない。次回は8ワーカーを開始点とするが、4・8・12の速度比較は未実施で、最適値は未確認。

抽出はreplayを順次処理し、収集方策の照合もCPU・float64・PyTorch 1スレッドで逐次実行する。収集の `--workers` はこの2段階には効かない。並列化には実装変更が必要であり、元のコードを変更しない今回の範囲では実施していない。記録作成・実装確認・成果物の独立確認は別担当で並行して進めた。

## 再実行コマンド

環境・C++ビルド・蒸留モデル・GBDTモデルを用意し、リポジトリ直下から実行する。出力先は空のディレクトリを指定する。今回はCUDAが利用可能な実行環境を使用した。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python src/agents/dragapult/ppo/scripts/run_cycle.py \
  --checkpoint outputs/dragapult/distillation_v1/model/model.pt \
  --gbdt-model outputs/dragapult/imitation_v2/model/model.gbdt \
  --output-dir outputs/dragapult/ppo_v1 \
  --cycles 1 --games 240 --workers 4 \
  --seed 202640000 --sample-seed 202650000 --update-seed 204 \
  --temperature 1 --updates 32 \
  --batch-size 512 --micro-batch-size 64 --holdout-decisions 1024 \
  --learning-rate 0.00003 --gamma 1 --gae-lambda 0.99 \
  --clip 0.2 --value-clip 0.2 \
  --value-coefficient 0.25 --entropy-coefficient 0.01 \
  --weight-decay 0.0001 --max-grad-norm 0.5 --device cuda
```

## 結果と成果物

240試合はすべて完了し、エージェント・エンジンエラーは0。280エピソード、20,701判断、21,494クエリ、546,026トークン行を抽出した。各エピソードの終局だけに報酬があり、勝ち54・負け226・引き分け0だった。これは更新前の確率的方策の収集結果で、自己対戦の両席を含む。更新後の強さを表す数値ではない。

収集方策の再計算では、対数行動確率・Value・Entropyの最大絶対誤差がそれぞれ `2.936e-6`、`8.338e-7`、`1.726e-6` で、許容誤差 `atol=rtol=2e-5` を満たした。GAEと学習目標はすべて有限だった。

| 更新結果 | 値 |
|---|---:|
| 実際の更新数 | 32 |
| holdoutを除いた学習可能判断数 | 19,677 |
| 実際に更新に使った判断数 | 16,384 |
| 診断用holdout | 1,024 |
| 重み変化のL2ノルム | 0.715808 |
| 更新中の平均Policy loss | 0.010375 |
| 更新中の平均clip fraction | 0.138611 |
| 更新後holdoutの近似KL | 0.038358 |
| 更新後holdoutの標本forward KL | 0.044345 |

近似KLは `mean(exp(delta) - 1 - delta)`、標本forward KLは `mean(-delta)`（`delta = new_logp - old_logp`）である。全合法行動の分布を列挙した厳密なKLではない。近似KLは設定ファイルの `kl_target=0.004` より大きいが、この経路では早期停止に使われないため32回更新された。次の実験では更新予算や学習率を減らして方策変化と対戦成績を比較する余地がある。

更新後は固定GBDTとの両席各2試合、自己対戦2試合の計6試合を実行し、自己対戦2試合を再実行して記録の決定性を確認した。単体の `.bin` での実行も成功した。6試合の772判断・798クエリについて、Python/C++の行動不一致0、エージェント・エンジンエラー0。最大絶対誤差は特徴量 `3.576e-7`、スコア `2.018e-6`、Value `5.800e-7` で、特徴量の許容誤差 `2e-6`、推論の `atol=rtol=2e-5` を満たした。追加のデータセットprobeは実行していない。

収集方策のチェックポイントSHA-256は `52c76f6c8e0c57fead013b6e098c56f9add4aa8dcd785e60fad4c045d6a98061`、更新後は `f8d98c08a52f0ffff6fb0aecfcf6b6e9c0e231f4a72730d086deaa16cff2b9ab`。蒸留済み重みを引き継いだ上で更新されたことを確認した。

以下の成果物を生成した。データ・重み・実行レポートはGit管理対象外。

| 内容 | リポジトリ直下からのパス |
|---|---|
| 初期化した収集方策 | `outputs/dragapult/ppo_v1/initial.pt` |
| 収集時のC++用モデル | `outputs/dragapult/ppo_v1/cycle-000/rollout/behavior.bin` |
| rolloutの記録と収集方策の照合結果 | `outputs/dragapult/ppo_v1/cycle-000/rollout/manifest.json` |
| 更新後チェックポイント | `outputs/dragapult/ppo_v1/cycle-000/models/model.pt` |
| 更新後C++用モデル | `outputs/dragapult/ppo_v1/cycle-000/models/model.bin` |
| 更新条件・損失・KL・重み変化 | `outputs/dragapult/ppo_v1/cycle-000/models/training.json` |
| 更新後の対戦の動作確認 | `outputs/dragapult/ppo_v1/cycle-000/validation/validation.json` |
| Python/C++推論照合 | `outputs/dragapult/ppo_v1/cycle-000/validation/parity.json` |
| サイクル全体のレポート | `outputs/dragapult/ppo_v1/cycles.json` |

強さを比較する場合は、更新前を固定ベースラインとして、共通seed・席・相手で更新前後を評価し、不確実性と対面ごとの悪化を確認する必要がある。収集中の勝敗や少数の更新後対戦だけで「PPOで強くなった」とは結論づけない。

この後に実施した[2,200試合の強さ比較](2026-10-15-dragapult-ppo-strength.md)では、固定5相手への平均スコアが28.60%から25.45%へ下がり、改善の証拠は得られなかった。差の95%信頼区間は−6.30〜+0.05ポイントで、悪化を断定することもできない。
