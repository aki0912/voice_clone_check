# Voice Clone Check

Qwen3-TTS の日本語 voice clone で、どの参照セリフが本人らしく自然な音声を作るかをローカルで比較する実験ツールです。

## 起動

Apple Silicon Mac と `ffmpeg` が必要です。

```bash
uv sync --extra dev
uv run voice-clone-check
```

ブラウザで表示された画面から、実験の作成、候補音声とアンカー音声の録音、合成、評価、ブラインド比較を行います。初回の合成・評価ではモデルがダウンロードされます。

CLIから保存済み実験を再開することもできます。

```bash
uv run voice-clone-check run --experiment <experiment-id>
uv run voice-clone-check report --experiment <experiment-id>
```

音声と評価結果は `data/experiments/` に保存されます。Web UIはlocalhostだけで待ち受け、音声を外部サービスへ送信しません。

## 評価

- 話者類似度: VoxCeleb学習済みERes2Netと共通アンカー音声の埋め込み
- 自然さ: UTMOS22 strong
- 読みの正確性: Qwen3-ASR 0.6B と読み仮名CER
- 最終確認: 自動上位3候補の匿名A/B試聴

UTMOSを含む自動指標には限界があります。最終判断では、個別指標、信頼区間、失敗率、ブラインド試聴を合わせて確認してください。
