# Pokémon TCG AI Battle — Population-based RLのサンプル実装

[English version](README.md)

Kaggleの[Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)向けの最小学習パイプラインです。C++対戦環境、GBDTによる模倣学習、Transformer蒸留、PPOを一つにまとめています。

[24th Solution: A Population-Based RL Ecosystem](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/writeups/24th-solution-a-population-based-rl-ecosystem)のサンプル実装として、公式4デッキに対応しています。

## コードの配置

```text
src/
├── agents/
│   ├── <agent名>/
│   │   ├── main.py, deck.csv    # 公式Python方策と同梱デッキ
│   │   ├── cpp/                # ルール方策・factory・学習記録
│   │   ├── scripts/            # 教師対戦収集・動作確認
│   │   ├── gbdt/               # 特徴・木の学習、cpp/、scripts/
│   │   ├── transformer/        # トークン・蒸留、cpp/、scripts/
│   │   └── ppo/                # rollout・GAE・更新、cpp/、scripts/
│   └── cpp/registry.h          # エージェントの登録のみ
├── engine/cpp/                 # 観測・合法手・対戦進行・記録
├── engine/cg/                  # 無変更の外部エンジン、Git対象外
└── tools/                      # 取得・ビルド・公式照合・レーティング
```

重み・リプレイ・特徴データ・ビルド成果物はGit追跡対象外です。

### 対象の4エージェント

| エージェント名 | 公式デッキ | GBDT特徴数 | コード・公式Notebook |
|---|---|---:|---|
| `mega_lucario` | メガルカリオex | 4,926 | [実装](src/agents/mega_lucario/)・[Notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck) |
| `dragapult` | ドラパルトex | 5,078 | [実装](src/agents/dragapult/)・[Notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-dragapult-ex-deck) |
| `iono` | ナンジャモ | 4,882 | [実装](src/agents/iono/)・[Notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck) |
| `mega_abomasnow` | メガユキノオーex | 4,714 | [実装](src/agents/mega_abomasnow/)・[Notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck) |

全4種に公式Python/C++ルール方策、GBDT、Transformer、PPOがあります。
C++方策名はルールなら上記の名前、学習済み方策なら `<名前>_gbdt` / `<名前>_transformer` です。PPOもTransformer方策として読み込みます。
学習スクリプトは各方式の `scripts/` にあり、引数は `--help` で確認できます。

## uv環境を用意する

Linux x86-64、Python 3.12以上、C++20対応の `g++` が必要です。GBDTはCPU、Transformer/PPOの更新はCUDA版PyTorchを使います。

uvが未導入なら、[公式インストーラ](https://docs.astral.sh/uv/getting-started/installation/)を実行してターミナルを開き直します。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

以降のコマンドはリポジトリ直下で実行します。`uv sync` が `.venv` を作成するため、手動のactivateは不要です。

```bash
uv sync --locked --dev
uv run kaggle --version
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

lockに記録されたLinux用PyTorchはCUDA版です。CUDAが `False` の場合はGPUとドライバへのアクセスを確認してください。
CPUで学習する場合は `--device cpu` を指定します。[uvとPyTorchの公式説明](https://docs.astral.sh/uv/guides/integration/pytorch/)も参照してください。

Kaggle CLIはdev依存に含めています。利用者自身のKaggle認証とコンペデータへのアクセスを用意し、外部資産を取得・ビルドします。

```bash
uv run python src/tools/fetch_official_assets.py
uv run python src/tools/fetch_cg_engine.py
uv run python src/tools/build_cpp_engine.py
```

デッキは各エージェントの `deck.csv` として同梱しています。配布C++エンジンは `src/engine/cg/` に取得します。
配布ソースを変更せずincludeします。C++推論はuv環境に含まれるOpenBLASを利用できます。
取得元・配置変更・オフライン検証は[エンジンの案内](src/engine/README.md)にあります。

## 解法概要

コンペでは、観測と合法手一覧を受け取り、選んだ候補の番号を返すAIを作ります。相手の手札や山札の順序が見えない中で、カードの展開・攻撃・エネルギーの使い方を判断します。ゲームルールはコンペ配布のC++エンジンを使います。

![解法全体：リプレイ、GBDT、Transformer、PPO、対戦評価](docs/assets/Fig1.png)

1. **ルール方策の対戦を記録する。** 選択前の観測と実際の行動をリプレイに保存します。
2. **GBDTで行動を模倣する。** 盤面と合法手候補から特徴量を作り、LightGBM LambdaRankで教師が選ぶ候補を上位にします。MAINとそれ以外の判断に別々の木を使い、単一の `.gbdt` に保存します。
3. **Transformerへ蒸留する。** GBDTを対戦させ、候補スコア・行動・終局結果からPolicyとValueを学びます。
4. **PPOで更新する。** 現在の方策で新しい対戦を収集し、収集時の行動確率とValue、終局報酬からGAEとPPO更新を行います。
5. **C++で再対戦し、評価する。** `.pt` を `.bin` にexportし、Pythonを呼ばずに推論します。固定した比較対象・共通seed・席交換で対戦し、レーティングの推定値と不確かさを保存します。

図は公開Writeupの全体像です。人間・LBエージェントのリプレイ、対面別Expert、母集団の自動選抜・大規模運用はこのサンプルに含めません。図中の試合数や処理時間は、このリポジトリの実測値ではありません。

![Transformerの構造](docs/assets/Fig2.png)

Transformerは全デッキとも **128次元・6層・4 heads・最大120トークン・980,916パラメータ**です。
Global・Board・Zone・Option・STOPをEncoderへ渡し、Pointer Policyと51段階のValue分布を出します。デッキ語彙と固有特徴が異なり、モデル構造は共通です。
追加3デッキでは公開カードの配置、進化組、エネルギー、攻撃IDなどを特徴化しています。メガルカリオ固有の攻撃結果計算は流用せず、未実装の結果チャネルは未知として扱います。

## ドラパルトのルール方策を模倣する

環境準備とC++ビルド後、公式4方策を相手に各20試合、計80試合を収集します。
各対面で両席を均等に使い、ドラパルト同士の対戦も含めます。
リポジトリ直下で、未使用の出力ディレクトリを指定して実行してください。

```bash
.venv/bin/python src/agents/dragapult/scripts/collect_replays.py \
  --output-dir outputs/dragapult/imitation_v1/rule_replays \
  --games-per-opponent 20 --seed 202610150 --workers 2
.venv/bin/python src/agents/dragapult/gbdt/scripts/extract.py \
  --replay-dir outputs/dragapult/imitation_v1/rule_replays \
  --output-dir outputs/dragapult/imitation_v1/features \
  --validation-fraction 0.25 --test-fraction 0 --seed 42
.venv/bin/python src/agents/dragapult/gbdt/scripts/train.py \
  --feature-dir outputs/dragapult/imitation_v1/features \
  --output-dir outputs/dragapult/imitation_v1/model \
  --profile quick --rounds 200 --threads 4 --seed 42
```

抽出時にPython/C++の特徴量の完全一致を確認し、試合単位で学習用60試合と
検証用20試合に分割します。教師はC++版ドラパルトのルール方策です。
LightGBMは教師が選んだ候補を他の合法手より上位にするよう、MAINとそれ以外を
別々の木で学習します。候補が1つしかない選択は学習対象から除きます。
`quick` は小さい木と検証結果による早期終了を使い、検証用の試合では再学習しません。
モデルの保存先は `outputs/dragapult/imitation_v1/model/model.gbdt` です。

出力の `query_top1` は選択クエリ単位での教師との一致率です。複数選択の途中段階も
数えるため、行動全体の一致率や勝率ではありません。この入門用実行には独立した
テスト分割がなく、検証値は早期終了にも使います。

上記の実行では学習6,099クエリ、検証2,280クエリが得られました。
検証での教師との一致率はMAINが64.8%、それ以外が80.2%、全体が73.1%でした。
学習済みモデルで公式4方策との8試合とモデル同士の2試合を動作確認し、
エージェント・エンジンエラーはありませんでした。確認したPython/C++の特徴量と
予測値の差は0（予測値の許容誤差は `atol=rtol=1e-10`）、選択の不一致も0でした。
データ全54,654候補行についてもC++ probeで予測値を照合しました。
これは限定した模倣精度と動作確認の結果であり、強さの向上を示すものではありません。

## レーティングを計測する

対戦結果からベイズBradley–Terryモデルでレーティングを推定します。以下は公式4方策を各組2試合ずつ対戦させる例です。

```bash
uv run python src/tools/rating_arena.py \
  --output-dir outputs/rating_example --games-per-pair 2 --workers 2
```

`--games-per-pair` は**席交換の両方向を合わせた偶数の試合数**です。両方向で同じseedを使います。席0/1はエンジン上の先攻/後攻の直接指定ではありません。
同じ出力先に `--resume` を指定すると完了済み対戦を再利用でき、試合数を増やせます。

`ratings.json` / `ratings.csv` に推定値、標準偏差、下側評価値、対面別W/D/Lを保存します。基準は比較集団の平均1,000です。
**公式LBレーティングとは別の内部尺度**であり、比較相手を変えると値も変わります。

学習済みモデルを比較する `--roster`、推定条件、再開手順は[レーティングの案内](src/tools/README.md)を参照してください。

## 開発

RuffでPythonの書式を整え、import・型アノテーション・docstringを確認します。

```bash
uv run ruff format src
uv run ruff check src
```

## ライセンス

このリポジトリのコード・文書・図は [Apache License 2.0](LICENSE) で公開しています。
