"""当日のレースについて、締切 N 分前に 3連単オッズ・単勝複勝オッズ・直前情報を取得して保存する。
usage: python scripts/live_snapshot.py STOREDIR --until HH:MM [--lead 6] [--date YYYYMMDD]
- 当日の番組表(B)から各レースの締切予定時刻を取り、締切 lead 分前〜1分前の間に1回取得する
- 保存先: STOREDIR/live/{odds3t,oddstf,beforeinfo}/YYYYMMDD.csv.gz（取得時刻 fetched_at 付き。既存分に追記）
- 15分ごとと終了時に scripts/push_store.sh で data ブランチへ push（--no-push で無効）
時刻はすべて日本時間。
"""
import argparse, csv, datetime as dt, gzip, os, subprocess, sys, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import BI_COLS, COMBOS, ODDSTF_COLS, parse_b, parse_beforeinfo, parse_odds3t, parse_oddstf

JST = dt.timezone(dt.timedelta(hours=9))
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}
BASE = "https://www.boatrace.jp/owpc/pc/race/"
PAGES = {"odds3t": (COMBOS, parse_odds3t), "oddstf": (ODDSTF_COLS, parse_oddstf), "beforeinfo": (BI_COLS, parse_beforeinfo)}
HEAD = ["date", "jcd", "rno", "deadline", "fetched_at"]

def now():
    return dt.datetime.now(JST)

def http(url, timeout=40):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()

def today_schedule(date):
    """番組表から [(jcd, rno, deadline datetime)]"""
    d = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    name = f"b{d:%y%m%d}"
    raw = http(f"https://www1.mbrace.or.jp/od2/B/{d:%Y%m}/{name}.lzh", 60)
    with tempfile.TemporaryDirectory() as tmp:
        open(f"{tmp}/a.lzh", "wb").write(raw)
        subprocess.run(["lha", f"xqw={tmp}", f"{tmp}/a.lzh"], check=True)
        txt = [p for p in os.listdir(tmp) if not p.endswith(".lzh")][0]
        txt = open(f"{tmp}/{txt}", "rb").read().decode("cp932", "replace")
    seen = {}
    for r in parse_b(txt, date):
        hh, mm = map(int, r["deadline"].split(":"))
        seen[(r["jcd"], r["rno"])] = dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=JST)
    return sorted(((j, n, t) for (j, n), t in seen.items()), key=lambda x: x[2])

def load_existing(store, date):
    rows = {p: {} for p in PAGES}
    for p in PAGES:
        f = f"{store}/live/{p}/{date}.csv.gz"
        if os.path.exists(f):
            with gzip.open(f, "rt") as fh:
                for r in csv.reader(fh):
                    if r[0] != "date":
                        rows[p][(r[1], int(r[2]))] = r
    return rows

def save(store, date, rows, lock):
    with lock:
        for p, (cols, _) in PAGES.items():
            f = f"{store}/live/{p}/{date}.csv.gz"
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with gzip.open(f + ".tmp", "wt", newline="") as fh:
                w = csv.writer(fh); w.writerow(HEAD + cols)
                for k in sorted(rows[p]):
                    w.writerow(rows[p][k])
            os.replace(f + ".tmp", f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store"); ap.add_argument("--until", required=True)
    ap.add_argument("--lead", type=float, default=6.0)
    ap.add_argument("--date", default=now().strftime("%Y%m%d"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    date = a.date
    hh, mm = map(int, a.until.split(":"))
    until = now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    sched = today_schedule(date)
    print(f"{date}: races={len(sched)} first={sched[0][2]:%H:%M} last={sched[-1][2]:%H:%M} until={until:%H:%M}", flush=True)
    rows = load_existing(a.store, date)
    lock = threading.Lock()
    started = {(j, n) for p in rows.values() for (j, n) in p}

    def snap(jcd, rno, deadline):
        for p, (cols, parser) in PAGES.items():
            t = now()
            try:
                html = http(f"{BASE}{p}?rno={rno}&jcd={jcd}&hd={date}").decode("utf-8", "replace")
                o = parser(html)
            except Exception as e:
                print("NG", p, jcd, rno, repr(e), flush=True); o = None
            row = [date, jcd, rno, f"{deadline:%H:%M}", t.strftime("%H:%M:%S")] + \
                  ([("" if o[c] is None else o[c]) for c in cols] if o else [""] * len(cols))
            with lock:
                rows[p][(jcd, rno)] = row
        print(f"snap {jcd}-{rno:02d} deadline={deadline:%H:%M} done={now():%H:%M:%S}", flush=True)

    last_push = time.monotonic()
    def push(msg):
        save(a.store, date, rows, lock)
        if not a.no_push:
            subprocess.run(["bash", "scripts/push_store.sh", msg], check=False)

    with ThreadPoolExecutor(a.workers) as ex:
        while now() < until:
            t = now()
            for jcd, rno, deadline in sched:
                k = (jcd, rno)
                if k in started:
                    continue
                if deadline - dt.timedelta(minutes=a.lead) <= t < deadline - dt.timedelta(minutes=1):
                    started.add(k); ex.submit(snap, jcd, rno, deadline)
                elif t >= deadline - dt.timedelta(minutes=1):
                    started.add(k)  # 取り逃し（ジョブ開始前など）
            if all(t >= d for _, _, d in sched):
                break
            if time.monotonic() - last_push > 900:
                push(f"live {date} {t:%H:%M}"); last_push = time.monotonic()
            time.sleep(10)
    push(f"live {date} end {now():%H:%M}")
    print("saved", {p: len(v) for p, v in rows.items()})

if __name__ == "__main__":
    main()
