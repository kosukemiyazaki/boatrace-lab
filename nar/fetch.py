"""地方競馬の公式データ（keiba.go.jp）を手元にダウンロードする。**ローカル専用**（利用規約により、取得したデータは公開リポジトリに置かない）。
usage: python nar/fetch.py [--data nar_data] [--from 2015-01] [--to 2026-08] [--odds-from 2026-03] [--interval 10]

- レース情報（月次 ZIP：racelist / horselist / payback）: /KeibaWeb/DataDownload/RaceDataDownload?type=monthly&k_year=YYYY&k_month=M
- 確定オッズ（月次 ZIP：10日ごとに3ファイル、全賭式）  : /KeibaWeb/DataDownload/OddsDataDownload?type=monthly&k_year=YYYY&k_month=M
  確定オッズは 2026-03 以降のみ公開されている
- 既にあるファイルは取り直さない。取得の間隔は既定で10秒（公式の推奨：月次ファイルは1日1回を上限）。
- 保存先: DATA/race/YYYYMM.zip, DATA/odds/YYYYMM.zip（nar_data/ は .gitignore 済み。コミットしないこと）
"""
import argparse, datetime as dt, os, sys, time, urllib.request

K = "https://www.keiba.go.jp/KeibaWeb/DataDownload"
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research; personal analysis)"}

def months(a, b):
    y, m = map(int, a.split("-")); y2, m2 = map(int, b.split("-"))
    while (y, m) <= (y2, m2):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1

def get(url):
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                b = r.read()
            return b if b[:2] == b"PK" else None  # ZIP でなければ（データなし・エラー）None
        except Exception as e:
            print("  retry", i + 1, e, flush=True); time.sleep(30 * (i + 1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="nar_data"); ap.add_argument("--from", dest="start", default="2015-01")
    ap.add_argument("--to", default="2026-08"); ap.add_argument("--odds-from", default="2026-03")
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args()
    jobs = [("race", y, m, f"{K}/RaceDataDownload?type=monthly&k_year={y}&k_month={m}") for y, m in months(a.start, a.to)]
    jobs += [("odds", y, m, f"{K}/OddsDataDownload?type=monthly&k_year={y}&k_month={m}") for y, m in months(a.odds_from, a.to)]
    todo = [j for j in jobs if not os.path.exists(f"{a.data}/{j[0]}/{j[1]}{j[2]:02d}.zip")]
    print(f"全 {len(jobs)} ファイル、未取得 {len(todo)}（間隔 {a.interval} 秒、目安 {len(todo) * (a.interval + 5) / 60:.0f} 分）", flush=True)
    for kind, y, m, url in todo:
        out = f"{a.data}/{kind}/{y}{m:02d}.zip"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        b = get(url)
        if b is None:
            print(f"{kind} {y}-{m:02d}: データなし", flush=True)
        else:
            open(out + ".tmp", "wb").write(b); os.replace(out + ".tmp", out)
            print(f"{kind} {y}-{m:02d}: {len(b) / 1e6:.1f} MB", flush=True)
        time.sleep(a.interval)

if __name__ == "__main__":
    main()
