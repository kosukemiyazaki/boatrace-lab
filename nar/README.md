# 地方競馬（NAR）の3連単：ローカルで動かす手順（IDEA-019）

地方競馬全国協会の利用規約により、取得したデータは**公開リポジトリに置けません**。
そのため、**コードはこのリポジトリ、データと実行はあなたの PC の中だけ**にします（`nar_data/`・`nar_out/` は `.gitignore` 済み）。
結果のうち**集計値（`nar_out/summary.md`）だけ**を共有してください。生データは共有・コミットしないでください。

## 準備（初回のみ）
```bash
git clone https://github.com/kosukemiyazaki/boatrace-lab.git   # 既にあれば git pull
cd boatrace-lab
python3 -m pip install pandas pyarrow numpy scipy scikit-learn lightgbm tabulate
```

## Windows の場合（PowerShell）
```powershell
git clone https://github.com/kosukemiyazaki/boatrace-lab.git   # git がなければ winget install Git.Git
cd boatrace-lab
py -m venv .venv                 # Python がなければ winget install Python.Python.3.12（入れたら PowerShell を開き直す）
.\.venv\Scripts\Activate.ps1      # 「スクリプトの実行が無効」と出たら: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install pandas pyarrow numpy scipy scikit-learn lightgbm tabulate
$env:PYTHONUTF8 = "1"             # 日本語の表示・保存を UTF-8 にする（PowerShell を開くたびに必要）
python nar\selftest.py nar_selftest; python nar\build.py --data nar_selftest; python nar\run.py --data nar_selftest --out nar_selftest\out
python nar\fetch.py; python nar\build.py; python nar\run.py
```

## 1. 動作確認（作り物のデータ、1分ほど）
```bash
python3 nar/selftest.py nar_selftest && python3 nar/build.py --data nar_selftest && python3 nar/run.py --data nar_selftest --out nar_selftest/out
```
最後に表が出れば OK です（数字は乱数なので意味はありません）。

## 2. 本番
```bash
python3 nar/fetch.py               # 2015-01〜2026-08 のレース情報と、2026-03〜08 の確定オッズ（約150ファイル、10秒間隔で約40分）
python3 nar/build.py               # 表にする（数分）
python3 nar/run.py                 # モデル → 市場との比較 → 上限のバックテスト（数十分）
```
- 取得は `nar_data/` に保存され、途中で止めても続きから再開できます。
- 公式の推奨（月次ファイルは1日1回を上限）を守るため、同じファイルは取り直しません。
- `nar/build.py` の出力に「賭式の値」「着順の値の例」などが出ます。想定と違う場合はその行をそのまま教えてください。

## 3. 共有してほしいもの
`nar_out/summary.md` の中身（対数損失・回収率などの集計値のみ）と、`nar/build.py`・`nar/run.py` の画面出力。

## 期間（テスト期間 2026-09 以降は使わない）
学習 〜2025-10 / 早期終了 2025-11〜12 / 3連単の温度・割引 2026-01〜02 / 市場との混ぜ合わせ 2026-03〜05 / 評価 2026-06〜08。
確定オッズは 2026-03 以降しか公開されていないため、評価は約3か月分です。

## 判定（docs/ideas.md の IDEA-019）
「モデル×市場」の3連単対数損失が、市場（確定オッズ）単独より **0.02 以上**小さい → 地方競馬を本格的に検討する。
（ボートレースでは同じ比較で約 0.005 だった）

## キャリーオーバー（IDEA-021、トリプル馬単）
`nar/build.py` まで終わっている前提（`nar_data/races.parquet` を使って、南関東4場・門別の開催日だけを取得します）。
```powershell
python nar\carryover.py fetch                 # 2018-01〜2026-08 の当日払戻金ページ（約2,000〜3,000ページ、5秒間隔で数時間。途中で止めても再開可）
python nar\carryover.py report                # 集計 → results\nar_carryover.md（集計値のみ。共有・コミットしてよい）
```
- 取得したページの値は `nar_data\carryover.csv` にだけ保存します（コミットしない）。
- 最初の1件と50件ごとに、読み取れた値（繰越・売上・次回への繰越・的中口数）を表示します。空欄ばかりなら、ページの形が想定と違うので `parse()` を直してください。

## 券種間のずれ（IDEA-024：3連単の市場を答えにして、単勝・複勝・馬単・馬連・ワイドのずれを探す）
すでに取得・作成済みの `nar_data`（`build.py` の後）があれば、追加の取得は不要です（数分）。
```
git pull
python nar\cross_pool.py                      # → results\nar_cross_pool.md（集計値のみ。共有・コミットしてよい）
git add results\nar_cross_pool.md
git commit -m "IDEA-024 result"
git push
```
