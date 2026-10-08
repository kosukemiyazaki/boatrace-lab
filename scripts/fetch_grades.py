"""グレード別の開催日程（boatrace.jp gradesch）を取得して STORE/grades.csv に保存する。
usage: python scripts/fetch_grades.py STORE [YEAR ...]
- 年を省略すると、2023年から今年までのうち未取得の年と、今年・来年（日程が更新されるため毎回取り直す）を取得する。
- 1ページずつ間隔をあけて取得（1年6ページ）。
- ここに載らない開催は一般戦。ヴィーナス・ルーキー・マスターズは grade 列のクラス名で G3 などが分かる。
"""
import csv, datetime as dt, os, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import GRADE_COLS, GRADE_HCD, parse_gradesch

UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}

def main():
    store = sys.argv[1]
    path = f"{store}/grades.csv"
    old = list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []
    this = dt.datetime.now(dt.timezone.utc).year
    have = {r["year"] for r in old}
    years = [int(y) for y in sys.argv[2:]] or sorted({y for y in range(2023, this + 1) if str(y) not in have} | {this, this + 1})
    new = []
    for y in years:
        for hcd in GRADE_HCD:
            url = f"https://www.boatrace.jp/owpc/pc/race/gradesch?year={y}&hcd={hcd}"
            html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")
            rows = parse_gradesch(html, hcd, y)
            print(y, hcd, GRADE_HCD[hcd], len(rows), flush=True)
            new += [dict(year=str(y), **r) for r in rows]
            time.sleep(1.5)
    keep = [r for r in old if int(r["year"]) not in years]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year"] + GRADE_COLS); w.writeheader()
        w.writerows(sorted(keep + new, key=lambda r: (r["start"], r["jcd"], r["hcd"])))

if __name__ == "__main__":
    main()
