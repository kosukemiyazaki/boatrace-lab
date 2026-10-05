"""odds3t 取得の遅さの原因調査: 逐次と並列でレイテンシを測る"""
import time, urllib.request, statistics
from concurrent.futures import ThreadPoolExecutor
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}
def one(rno, jcd="02", hd="20260704", page="odds3t"):
    t = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(
            f"https://www.boatrace.jp/owpc/pc/race/{page}?rno={rno}&jcd={jcd}&hd={hd}", headers=UA), timeout=60) as r:
            n = len(r.read())
    except Exception as e:
        n = repr(e)
    return time.monotonic() - t, n
def rep(name, xs):
    ls = [x[0] for x in xs]
    print(f"{name}: n={len(ls)} median={statistics.median(ls):.2f}s max={max(ls):.2f}s sizes={set(type(x[1]).__name__ for x in xs)}", flush=True)
rep("seq", [one(r) for r in range(1, 9)])
rep("seq-mbrace-like-small(raceindex)", [one(r, page="raceindex") for r in range(1, 4)])
for w in (4, 8, 16):
    t = time.monotonic()
    with ThreadPoolExecutor(w) as ex:
        xs = list(ex.map(lambda i: one(i % 12 + 1, jcd=f"{(i // 12) % 24 + 1:02d}", hd="20260703"), range(w * 3)))
    rep(f"par{w} wall={time.monotonic()-t:.1f}s", xs)
