# Voice Clone Check

Qwen3-TTS の日本語 voice clone で、どの参照セリフが本人らしく自然な音声を作るかをローカルで比較する実験ツールです。

## 起動

Apple Silicon Mac と `ffmpeg` が必要です。

```bash
uv sync --extra dev
uv run voice-clone-check
```

ブラウザで表示された画面から、実験の作成、候補音声の準備、合成、評価、ブラインド比較を行います。初回の合成・評価ではモデルがダウンロードされます。

## サンプル音声から事前選定

新しい実験の方式で「サンプル音声から事前選定」を選ぶと、12候補をすべて録音せずに絞り込めます。

1. 本人の許諾を得た、1人だけが話す5〜15秒の音声を用意します。BGM、効果音、強い反響、言い直しのある素材は避けます。
2. 「元音声を登録」でブラウザ録音または音声ファイルを選び、音声と一字一句対応する台本を保存します。ASRとの読み仮名CERが0.10以下であることを確認します。
3. 「候補生成スモーク」で動作を確認し、「12候補×2テイクを合成」で24件の候補参照音声を作ります。不合格音声は別seedで最大4回まで自動再試行されます。
4. 「評価スモーク」の後、「384件の評価を開始」を実行します。
5. 結果を更新し、「上位3件の実録音検証を作成」を押します。作成された検証実験では、表示される3文だけを各2テイク録音します。
6. 検証実験で96件を評価し、ブラインド試聴と比較レポートで最終候補を決めます。

元音声は最低1件、可能なら異なる発話を2〜3件登録してください。音声とモデルはローカルで処理され、外部サービスには送信されません。合成音声による順位は事前選定用であり、最終判断では実録音検証を優先してください。

元音声スロットを選択すると、現在保存されている音声を再生でき、保存した台本、ASR結果、CER、長さ、SNR、品質上の注意を確認できます。録り直す場合は「新しい音声」で録音し、「スロットを再録音して上書き」を押してください。同じスロットの以前の音声は上書きされます。新しい音声を指定せずに台本だけ変更して保存すると、現在の保存音声を使って台本を再照合します。

CLIから保存済み実験を再開することもできます。

```bash
uv run voice-clone-check run --experiment <experiment-id>
uv run voice-clone-check report --experiment <experiment-id>
```

合成事前選定はCLIからも登録・再開できます。

```bash
uv run voice-clone-check create --mode synthetic --name "事前選定"
uv run voice-clone-check add-source --experiment <experiment-id> --audio sample.wav --transcript "正確な台本"
uv run voice-clone-check generate-references --experiment <experiment-id> --smoke
uv run voice-clone-check generate-references --experiment <experiment-id>
uv run voice-clone-check run --experiment <experiment-id> --smoke
uv run voice-clone-check run --experiment <experiment-id>
uv run voice-clone-check create-validation --experiment <experiment-id>
uv run voice-clone-check add-candidate --experiment <validation-id> --prompt c01 --take 1 --audio c01-take1.wav
```

音声と評価結果は `data/experiments/` に保存されます。Web UIはlocalhostだけで待ち受け、音声を外部サービスへ送信しません。

## 評価

- 話者類似度: VoxCeleb学習済みERes2Netと共通アンカー音声の埋め込み
- 自然さ: UTMOS22 strong
- 読みの正確性: Qwen3-ASR 0.6B と読み仮名CER
- 最終確認: 自動上位3候補の匿名A/B試聴

UTMOSを含む自動指標には限界があります。最終判断では、個別指標、信頼区間、失敗率、ブラインド試聴を合わせて確認してください。
