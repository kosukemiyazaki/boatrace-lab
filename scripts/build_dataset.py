"""data ブランチの B/K/odds から学習用テーブルを作る。
usage: python scripts/build_dataset.py STOREDIR OUTDIR
出力: OUTDIR/entries.parquet (1行=1艇), races.parquet (1行=1レース), odds.parquet (1行=1レース, 120列)
      beforeinfo.parquet / oriten.parquet / votes.parquet / odds3f.parquet / odds2tf.parquet（あれば）
"""
import glob, gzip, os, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import parse_b, parse_k

def one_day(kpath):
    base = os.path.basename(kpath)
    date = "20" + base[1:7]
    bpath = kpath.replace("/k", "/b")
    rd = lambda p: gzip.open(p, "rt", encoding="utf-8").read()
    krows, races = parse_k(rd(kpath), date)
    brows = parse_b(rd(bpath), date) if os.path.exists(bpath) else []
    return brows, krows, races

def main(store, out, with_odds=True):
    os.makedirs(out, exist_ok=True)
    ks = sorted(glob.glob(f"{store}/bk/*/k*.txt.gz"))
    B, K, R = [], [], []
    with ProcessPoolExecutor() as ex:
        for b, k, r in ex.map(one_day, ks, chunksize=8):
            B += b; K += k; R += r
    b = pd.DataFrame(B); k = pd.DataFrame(K); r = pd.DataFrame(R)
    key = ["date", "jcd", "rno", "boat"]
    k = k.drop_duplicates(key)
    e = b.merge(k.drop(columns=["motor_no", "boat_no"]), on=key, how="inner", suffixes=("", "_k"))
    e = e[e["toban"] == e["toban_k"]].drop(columns=["toban_k"])
    e.to_parquet(f"{out}/entries.parquet")
    r.to_parquet(f"{out}/races.parquet")
    print("entries", len(e), "races", len(r), "days", len(ks))

    frames = []
    for p in (sorted(glob.glob(f"{store}/odds/*.csv.gz")) if with_odds else []):
        frames.append(pd.read_csv(p, dtype={"date": str, "jcd": str}))
    if frames:
        o = pd.concat(frames, ignore_index=True)
        o.to_parquet(f"{out}/odds.parquet")
        print("odds races", len(o), "with table", int(o.iloc[:, 3:].notna().any(axis=1).sum()))

    bfs = sorted(glob.glob(f"{store}/beforeinfo/*.csv.gz"))
    if bfs:
        b = pd.concat([pd.read_csv(p, dtype={"date": str, "jcd": str}) for p in bfs], ignore_index=True)
        b.to_parquet(f"{out}/beforeinfo.parquet")
        print("beforeinfo races", len(b))

    # BOATCAST のオリジナル展示（oriten.parquet）・3連単の確定票数（votes.parquet）・他券種オッズ（odds3f / odds2tf）。--no-odds のときは票数と他券種は読まない
    for page, name, need_odds in (("bc_oriten", "oriten", False), ("bc_votes", "votes", True), ("odds3f", "odds3f", True), ("odds2tf", "odds2tf", True)):
        if need_odds and not with_odds:
            continue
        fs = sorted(glob.glob(f"{store}/{page}/*.csv.gz"))
        if fs:
            x = pd.concat([pd.read_csv(p, dtype={"date": str, "jcd": str}) for p in fs], ignore_index=True)
            x["jcd"] = x["jcd"].str.zfill(2)
            x = x.drop_duplicates(["date", "jcd", "rno"], keep="last")
            x.to_parquet(f"{out}/{name}.parquet")
            print(name, "races", len(x), "with data", int(x.iloc[:, 3:].notna().any(axis=1).sum()))

if __name__ == "__main__":
    # --no-odds: オッズを読み込まない（live のジョブ開始時など、出走表・成績・直前情報だけ必要なとき）
    main(sys.argv[1], sys.argv[2], with_odds="--no-odds" not in sys.argv)
