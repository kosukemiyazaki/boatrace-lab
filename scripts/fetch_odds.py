"""締切時3連単オッズ(boatrace.jp odds3t)を月単位で取得する。
usage: python scripts/fetch_odds.py YYYYMM STOREDIR [--workers 4] [--budget-min 320]
レース一覧は STOREDIR/bk の競走成績(K)から作る。
保存先: STOREDIR/odds/YYYYMM.csv.gz  (date,jcd,rno,<120通りのオッズ 1-2-3..6-5-4>)
途中まで取れているファイルがあれば続きから取る（時間切れでも途中結果を保存する）。
"""
import argparse, csv, glob, gzip, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, parse_odds3t, race_keys_k

UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}
BASE = "https://www.boatrace.jp/owpc/pc/race/odds3t"

class Pacer:
    """スレッドごとに最低 interval 秒あける"""
    def __init__(self, interval):
        self.interval = interval; self.local = threading.local()
    def wait(self):
        last = getattr(self.local, "t", 0.0)
        dt_ = time.monotonic() - last
        if dt_ < self.interval:
            time.sleep(self.interval - dt_)
        self.local.t = time.monotonic()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("month"); ap.add_argument("store")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--budget-min", type=float, default=320)
    a = ap.parse_args()
    t0 = time.monotonic()

    keys = []
    for p in sorted(glob.glob(f"{a.store}/bk/{a.month[:4]}/k{a.month[2:]}*.txt.gz")):
        date = "20" + os.path.basename(p)[1:7]
        for jcd, rno in race_keys_k(gzip.open(p, "rt", encoding="utf-8").read()):
            keys.append((date, jcd, rno))
    out = f"{a.store}/odds/{a.month}.csv.gz"
    done = {}
    if os.path.exists(out):
        with gzip.open(out, "rt") as f:
            for r in csv.reader(f):
                if r[0] != "date":
                    done[(r[0], r[1], int(r[2]))] = r
    todo = [k for k in keys if k not in done]
    print(f"{a.month}: races={len(keys)} done={len(done)} todo={len(todo)}", flush=True)

    pacer = Pacer(a.interval)
    stop = threading.Event()
    errors = [0]

    def job(k):
        if stop.is_set():
            return k, None
        date, jcd, rno = k
        for i in range(4):
            pacer.wait()
            try:
                req = urllib.request.Request(f"{BASE}?rno={rno}&jcd={jcd}&hd={date}", headers=UA)
                with urllib.request.urlopen(req, timeout=30) as r:
                    html = r.read().decode("utf-8", "replace")
                o = parse_odds3t(html)
                # 中止レースなどオッズ表がない場合は空行で記録
                return k, [date, jcd, rno] + ([("" if o[c] is None else o[c]) for c in COMBOS] if o else [""] * 120)
            except urllib.error.HTTPError as e:
                if e.code in (403, 429):
                    print("rate-limited", e.code, k, flush=True); time.sleep(60 * (i + 1))
                else:
                    time.sleep(5 * (i + 1))
            except Exception:
                time.sleep(5 * (i + 1))
        errors[0] += 1
        if errors[0] > 50:
            stop.set()
        return k, None

    got = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for k, row in ex.map(job, todo):
            if row:
                done[k] = row; got += 1
            if got and got % 500 == 0:
                print(f"  {got}/{len(todo)} {(time.monotonic()-t0)/60:.1f}min", flush=True)
            if (time.monotonic() - t0) / 60 > a.budget_min:
                stop.set()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "jcd", "rno"] + COMBOS)
        for k in sorted(done):
            w.writerow(done[k])
    print(f"{a.month}: saved={len(done)}/{len(keys)} errors={errors[0]} {(time.monotonic()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
