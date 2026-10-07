"""当日のレースについて、締切15分前・10分前・6分前にオッズを、6分前に直前情報を取得して保存する。
usage: python scripts/live_snapshot.py STOREDIR --until HH:MM [--date YYYYMMDD]
- 当日の番組表(B)から各レースの締切予定時刻を取り、各時点の窓の中で1回ずつ取得する
    15分前: [締切-15, 締切-11) に 3連単・単勝複勝オッズ   -> STOREDIR/live_t15/{odds3t,oddstf}/YYYYMMDD.csv.gz
    10分前: [締切-10, 締切-7)  に 3連単・単勝複勝オッズ   -> STOREDIR/live_t10/{odds3t,oddstf}/YYYYMMDD.csv.gz
     6分前: [締切-6,  締切-1)  に 3連単・単勝複勝オッズ・直前情報 -> STOREDIR/live/{odds3t,oddstf,beforeinfo}/YYYYMMDD.csv.gz
  （6分前の保存場所・形式は撤退判定 scripts/exit_check.py が読むので変えない）
- --model と --datadir を指定すると、6分前の取得の直後に凍結モデルで予想（各艇の1着確率と3連単120通りの確率）を計算し、
  STOREDIR/live_pred/YYYYMMDD_<ジョブID>.csv.gz に保存する（予想した時刻 predicted_at 付き。買い目は出さない）
- 取得時刻 fetched_at 付き。保存ファイルはジョブごとに分ける（YYYYMMDD_<ジョブID>.csv.gz）。
  ジョブが重なっても上書きし合わない。同じ日の他のファイルにあるレースは取得済みとして飛ばす。
  読む側（exit_check.py, compare_live.py）は日付のファイルをまとめて読み、重複を除く
- 15分ごとと終了時に scripts/push_store.sh で data ブランチへ push（--no-push で無効）
時刻はすべて日本時間。
"""
import argparse, csv, datetime as dt, glob, gzip, os, subprocess, sys, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import BI_COLS, COMBOS, ODDSTF_COLS, parse_b, parse_beforeinfo, parse_odds3t, parse_oddstf

JST = dt.timezone(dt.timedelta(hours=9))
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}
BASE = "https://www.boatrace.jp/owpc/pc/race/"
PAGES = {"odds3t": (COMBOS, parse_odds3t), "oddstf": (ODDSTF_COLS, parse_oddstf), "beforeinfo": (BI_COLS, parse_beforeinfo)}
HEAD = ["date", "jcd", "rno", "deadline", "fetched_at"]
# (名前, 保存先ディレクトリ, 窓の開始[分前], 窓の終了[分前], 取得ページ)
LEADS = [("t15", "live_t15", 15, 11, ("odds3t", "oddstf")),
         ("t10", "live_t10", 10, 7, ("odds3t", "oddstf")),
         ("t6", "live", 6, 1, ("odds3t", "oddstf", "beforeinfo"))]

def now():
    return dt.datetime.now(JST)

def http(url, timeout=40):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()

def today_schedule(date):
    """番組表から ([(jcd, rno, deadline datetime)], 番組表テキスト)"""
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
    return sorted(((j, n, t) for (j, n), t in seen.items()), key=lambda x: x[2]), txt

TAG = os.environ.get("GITHUB_RUN_ID") or dt.datetime.now(JST).strftime("%H%M%S")

def load_existing(store, date):
    """同じ日の既存ファイル（他のジョブの分も含む）で取得済みのレースを返す"""
    done = set()
    for ld, d, _, _, pages in LEADS:
        for p in pages:
            for f in glob.glob(f"{store}/{d}/{p}/{date}*.csv.gz"):
                with gzip.open(f, "rt") as fh:
                    for r in csv.reader(fh):
                        if r[0] != "date":
                            done.add((ld, r[1], int(r[2])))
    return done

PRED_HEAD = HEAD[:4] + ["bi_fetched_at", "predicted_at", "model", "has_bi"] + [f"win{b}" for b in range(1, 7)] + COMBOS

def save(store, date, rows, lock, preds=None):
    with lock:
        if preds:
            f = f"{store}/live_pred/{date}_{TAG}.csv.gz"
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with gzip.open(f + ".tmp", "wt", newline="") as fh:
                w = csv.writer(fh); w.writerow(PRED_HEAD)
                for k in sorted(preds):
                    w.writerow(preds[k])
            os.replace(f + ".tmp", f)
        for ld, d, _, _, pages in LEADS:
            for p in pages:
                if not rows[(ld, p)]:
                    continue
                f = f"{store}/{d}/{p}/{date}_{TAG}.csv.gz"
                os.makedirs(os.path.dirname(f), exist_ok=True)
                with gzip.open(f + ".tmp", "wt", newline="") as fh:
                    w = csv.writer(fh); w.writerow(HEAD + PAGES[p][0])
                    for k in sorted(rows[(ld, p)]):
                        w.writerow(rows[(ld, p)][k])
                os.replace(f + ".tmp", f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store"); ap.add_argument("--until", required=True)
    ap.add_argument("--date", default=now().strftime("%Y%m%d"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--model", help="凍結モデルのディレクトリ（予想する場合）")
    ap.add_argument("--datadir", help="build_dataset.py の出力（予想する場合）")
    a = ap.parse_args()
    date = a.date
    hh, mm = map(int, a.until.split(":"))
    until = now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    sched, btext = today_schedule(date)
    # 予想の準備は裏で進める（データセットの作成完了 = DATADIR/READY を待ってから）。準備ができるまでのレースは予想なしで取得だけする
    pred_box = {"p": None}
    def init_predictor():
        try:
            t0 = time.monotonic()
            while not os.path.exists(f"{a.datadir}/READY"):
                if time.monotonic() - t0 > 1800:
                    print("predictor init: dataset not ready in 30 min", flush=True); return
                time.sleep(10)
            from boatlib.predictor import Predictor
            p = Predictor(a.datadir, a.model, btext, date)
            pred_box["p"] = p
            print(f"predictor ready ({p.model_name}, {time.monotonic() - t0:.0f}s, races={len(p.today)})", flush=True)
        except Exception as e:
            print("predictor init failed:", repr(e), flush=True)
    if a.model and a.datadir:
        threading.Thread(target=init_predictor, daemon=True).start()
    print(f"{date}: races={len(sched)} first={sched[0][2]:%H:%M} last={sched[-1][2]:%H:%M} until={until:%H:%M}", flush=True)
    rows = {(ld, p): {} for ld, _, _, _, pages in LEADS for p in pages}
    preds = {}
    lock = threading.Lock()
    started = load_existing(a.store, date)
    print("already fetched (other jobs):", len(started), flush=True)

    def snap(ld, pages, jcd, rno, deadline):
        parsed = {}
        for p in pages:
            cols, parser = PAGES[p]
            t = now()
            try:
                html = http(f"{BASE}{p}?rno={rno}&jcd={jcd}&hd={date}").decode("utf-8", "replace")
                o = parser(html)
            except Exception as e:
                print("NG", p, jcd, rno, repr(e), flush=True); o = None
            parsed[p] = (o, t)
            row = [date, jcd, rno, f"{deadline:%H:%M}", t.strftime("%H:%M:%S")] + \
                  ([("" if o[c] is None else o[c]) for c in cols] if o else [""] * len(cols))
            with lock:
                rows[(ld, p)][(jcd, rno)] = row
        print(f"snap {ld} {jcd}-{rno:02d} deadline={deadline:%H:%M} done={now():%H:%M:%S}", flush=True)
        predictor = pred_box["p"]
        if ld == "t6" and predictor is not None:
            try:
                bi, bt = parsed.get("beforeinfo", (None, None))
                win, P, has_bi = predictor.predict(jcd, rno, bi)
                pt = now()
                with lock:
                    preds[(jcd, rno)] = [date, jcd, rno, f"{deadline:%H:%M}", bt.strftime("%H:%M:%S") if bt else "",
                                         pt.strftime("%H:%M:%S"), predictor.model_name, int(has_bi)] + \
                                        [round(float(x), 5) for x in win] + [round(float(x), 6) for x in P]
                print(f"pred {jcd}-{rno:02d} at {pt:%H:%M:%S} (deadline {deadline:%H:%M}) win={[round(float(x), 2) for x in win]}", flush=True)
            except Exception as e:
                print("pred failed", jcd, rno, repr(e), flush=True)

    last_push = time.monotonic()
    def push(msg):
        save(a.store, date, rows, lock, preds)
        if not a.no_push:
            subprocess.run(["bash", "scripts/push_store.sh", msg], check=False)

    with ThreadPoolExecutor(a.workers) as ex:
        while now() < until:
            t = now()
            for jcd, rno, deadline in sched:
                for ld, _, w0, w1, pages in LEADS:
                    k = (ld, jcd, rno)
                    if k in started:
                        continue
                    if deadline - dt.timedelta(minutes=w0) <= t < deadline - dt.timedelta(minutes=w1):
                        started.add(k); ex.submit(snap, ld, pages, jcd, rno, deadline)
                    elif t >= deadline - dt.timedelta(minutes=w1):
                        started.add(k)  # 取り逃し（ジョブ開始前など）
            if all(t >= d for _, _, d in sched):
                break
            if time.monotonic() - last_push > 900:
                push(f"live {date} {t:%H:%M}"); last_push = time.monotonic()
            time.sleep(10)
    push(f"live {date} end {now():%H:%M}")
    print("saved", {f"{ld}/{p}": len(v) for (ld, p), v in rows.items()})

if __name__ == "__main__":
    main()
