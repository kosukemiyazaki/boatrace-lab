# オートレースの3連単：ローカルで動かす手順（IDEA-029）

オートレース公式サイト（autorace.jp）の利用規約は「私的利用の範囲」に限り、事前の許可のない転載・複製・頒布・公開と、大量アクセスを禁じています。
そのため **コードはこのリポジトリ、データと実行はあなたの PC の中だけ** にします（`auto_data/` は `.gitignore` 済み）。
共有してよいのは **集計値（`results/auto_summary.md`）だけ** です。生データは共有・コミットしないでください。
取得は2秒以上の間隔をあけ、1台の PC から順番に取ります（並列にしない）。

## Windows（PowerShell）
地方競馬のときに作った `.venv` があれば、そのまま使えます。
```powershell
cd boatrace-lab
git pull
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
# 1. 動作確認（作り物のデータ、1〜2分）
python auto\selftest.py auto_selftest; python auto\build.py --data auto_selftest; python auto\run.py --data auto_selftest --out auto_selftest\summary.md
# 2. 取得（2024-09〜2026-08、約2万回のアクセスで 10時間前後。途中で止めても、同じコマンドで続きから再開できます）
python auto\fetch.py
# 3. 表にする → モデルと市場の比較
python auto\build.py
python auto\run.py
# 4. 結果（集計値のみ）を共有
git add results\auto_summary.md results\auto_summary.json
git commit -m "IDEA-029 result (auto race)"
git push
```
- 取得を短くしたいときは `python auto\fetch.py --from 2025-01-01`（学習が 2025年だけになる。約7〜8時間）。
- `build.py` の最後に「3連単の最終オッズ×100 と払戻の一致率」が出ます。0.9 以上なら、保存したオッズは最終オッズです。

## 期間（テスト期間 2026-09 以降は使わない）
学習 〜2025-12 / 早期終了 2026-01〜02 / 温度・割引 2026-03 / 市場との混ぜ合わせ 2026-04〜05 / 評価 2026-06〜08

## 判定（docs/ideas.md の IDEA-029）
「モデル×市場」の3連単対数損失が、最終オッズの市場単独より 0.02 以上小さく、95%信頼区間が0をまたがない → オートレースを本格的に検討。
控除率は 30%（ボートは 25%）なので、回収率 110% には「自分の確率 ÷ 市場の確率」が約 1.57 倍以上の買い目が必要。
