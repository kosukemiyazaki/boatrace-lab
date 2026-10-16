"""boatrace.jp のレース別ページを月単位で取得する（既定は締切時3連単オッズ odds3t）。
--page oddstf     : 締切時 単勝・複勝オッズ   -> STOREDIR/oddstf/YYYYMM.csv.gz
--page beforeinfo : 直前情報（展示・チルト・部品交換・スタート展示・水面気象） -> STOREDIR/beforeinfo/YYYYMM.csv.gz
--page pcexpect   : 公式コンピュータ予想（3連・2連の予想組番、自信度。IDEA-006 用） -> STOREDIR/pcexpect/YYYYMM.csv.gz
--page odds3f     : 締切時 3連複オッズ（20通り。IDEA-013 用） -> STOREDIR/odds3f/YYYYMM.csv.gz
--page odds2tf    : 締切時 2連単（30通り）・2連複（15通り）オッズ（IDEA-013 用） -> STOREDIR/odds2tf/YYYYMM.csv.gz
--page bc_oriten  : BOATCAST のオリジナル展示（一周・まわり足・直線、場によって項目が違う。IDEA-015 用） -> STOREDIR/bc_oriten/YYYYMM.csv.gz
--page bc_votes   : BOATCAST の3連単確定票数（120通り＋合計。売上・自分の賭けの影響の計算用） -> STOREDIR/bc_votes/YYYYMM.csv.gz
usage: python scripts/fetch_odds.py YYYYMM STOREDIR [--workers 4] [--budget-min 320] [--stride 3]
--stride N: N日に1日だけ取る（日付の通し番号 % N == 0 の日）。GitHub Actions からは1リクエスト約9秒かかるため
（サーバ側で遅延される）、件数を絞るのに使う。
レース一覧は STOREDIR/bk の競走成績(K)から作る。
保存先: STOREDIR/odds/YYYYMM.csv.gz  (date,jcd,rno,<120通りのオッズ 1-2-3..6-5-4>)
途中まで取れているファイルがあれば続きから取る（時間切れでも途中結果を保存する）。
"""
import argparse, csv, datetime as dt, glob, gzip, os, signal, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import (BC_ORI_COLS, BC_VOTE1_COLS, BC_VOTE2_COLS, BC_VOTE_COLS, BI_COLS, COMBOS, COMBOS_2F, COMBOS_2T, COMBOS_3F, ODDSK_COLS, ODDSTF_COLS, PCEXPECT_COLS, parse_bc_hyousu1, parse_bc_hyousu2, parse_bc_hyousu3,
                           parse_bc_oriten, parse_beforeinfo, parse_odds2tf, parse_odds3f, parse_odds3t, parse_oddsk, parse_oddstf, parse_pcexpect, race_keys_k)

UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}
BASE = "https://www.boatrace.jp/owpc/pc/race/"
PAGES = {  # page -> (列名, パーサ)
    "odds3t": (COMBOS, parse_odds3t),
    "oddstf": (ODDSTF_COLS, parse_oddstf),
    "beforeinfo": (BI_COLS, parse_beforeinfo),
    "pcexpect": (PCEXPECT_COLS, parse_pcexpect),
    "odds3f": (COMBOS_3F, parse_odds3f),            # 締切時 3連複オッズ（IDEA-013）
    "odds2tf": (COMBOS_2T + COMBOS_2F, parse_odds2tf),
    "oddsk": (ODDSK_COLS, parse_oddsk),                  # 締切時 拡連複（ワイド）オッズの下限・上限（IDEA-039）  # 締切時 2連単・2連複オッズ（IDEA-013）
    "bc_oriten": (BC_ORI_COLS, parse_bc_oriten),        # BOATCAST オリジナル展示（一周・まわり足・直線。IDEA-015）
    "bc_votes": (BC_VOTE_COLS, parse_bc_hyousu3),       # BOATCAST 3連単の確定票数（売上・自分の賭けの影響）
    "bc_votes1": (BC_VOTE1_COLS, parse_bc_hyousu1),     # BOATCAST 単勝・複勝の確定票数（IDEA-022：単勝の売上と自分の賭けの影響）
    "bc_votes2": (BC_VOTE2_COLS, parse_bc_hyousu2),     # BOATCAST 2連単・2連複の確定票数（IDEA-023：2連複の売上と自分の賭けの影響）
}
BC = "https://race.boatcast.jp/txt"
def page_url(page, date, jcd, rno):
    if page == "bc_oriten":
        return f"{BC}/{jcd}/bc_oriten_{date}_{jcd}_{int(rno):02d}.txt"
    if page == "bc_votes":
        return f"{BC}/{jcd}/bc_kakutei_hyousu3_{date}_{jcd}_{int(rno):02d}.txt"
    if page == "bc_votes1":
        return f"{BC}/{jcd}/bc_kakutei_hyousu1_{date}_{jcd}_{int(rno):02d}.txt"
    if page == "bc_votes2":
        return f"{BC}/{jcd}/bc_kakutei_hyousu2_{date}_{jcd}_{int(rno):02d}.txt"
    return f"{BASE}{page}?rno={rno}&jcd={jcd}&hd={date}"

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
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--page", default="odds3t", choices=list(PAGES))
    a = ap.parse_args()
    t0 = time.monotonic()

    keys = []
    for p in sorted(glob.glob(f"{a.store}/bk/{a.month[:4]}/k{a.month[2:]}*.txt.gz")):
        date = "20" + os.path.basename(p)[1:7]
        if dt.date(int(date[:4]), int(date[4:6]), int(date[6:])).toordinal() % a.stride:
            continue
        for jcd, rno in race_keys_k(gzip.open(p, "rt", encoding="utf-8").read()):
            keys.append((date, jcd, rno))
    cols, parser = PAGES[a.page]
    out = f"{a.store}/{'odds' if a.page == 'odds3t' else a.page}/{a.month}.csv.gz"
    done = {}
    if os.path.exists(out):
        with gzip.open(out, "rt") as f:
            for r in csv.reader(f):
                if r[0] != "date":
                    done[(r[0], r[1], int(r[2]))] = r
    todo = [k for k in keys if k not in done]
    print(f"{a.page} {a.month}: races={len(keys)} done={len(done)} todo={len(todo)}", flush=True)

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
                req = urllib.request.Request(page_url(a.page, date, jcd, rno), headers=UA)
                with urllib.request.urlopen(req, timeout=30) as r:
                    html = r.read().decode("utf-8", "replace")
                o = parser(html)
                # 中止レースなど表がない場合は空行で記録
                return k, [date, jcd, rno] + ([("" if o[c] is None else o[c]) for c in cols] if o else [""] * len(cols))
            except urllib.error.HTTPError as e:
                if a.page.startswith("bc_") and e.code in (403, 404):
                    # BOATCAST はファイルがないとき 403/404 を返す（江戸川・計測なし・中止など）。データなしとして記録する
                    return k, [date, jcd, rno] + [""] * len(cols)
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

    def save():
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        with gzip.open(tmp, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "jcd", "rno"] + cols)
            for k in sorted(done):
                w.writerow(done[k])
        os.replace(tmp, out)
    # キャンセル(SIGTERM/SIGINT)時も取れた分は残す
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    got = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for k, row in ex.map(job, todo):
            if row:
                done[k] = row; got += 1
            if row and got % 200 == 0:
                print(f"  {got}/{len(todo)} {(time.monotonic()-t0)/60:.1f}min", flush=True)
                save()
            if (time.monotonic() - t0) / 60 > a.budget_min:
                stop.set()
    save()
    print(f"{a.month}: saved={len(done)}/{len(keys)} errors={errors[0]} {(time.monotonic()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
