"""IDEA-042 大村・下関のミッドナイト後半で1番人気の3連単を1点（docs/ideas.md の登録どおり。2026-10-17 以降で判定）
usage: python scripts/idea042.py DATADIR STOREDIR [--frozen models/idea040_frozen.npz] [--out results/idea042.md]
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from idea040 import apply, f_features, market
from live_eval import KEY, _read

START = "20261017"; VENUES = ("19", "24"); RNG = np.random.default_rng(0)

def roi_ci(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(5000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--frozen", default="models/idea040_frozen.npz"); ap.add_argument("--out", default="results/idea042.md")
    a = ap.parse_args()
    t1 = _read(f"{a.store}/live_t1/odds3t/*.csv.gz")
    t1 = t1[t1["fetched_at"] < t1["deadline"] + ":00"].drop_duplicates(KEY, keep="last")
    t1 = t1[(t1["date"] >= START) & t1["jcd"].isin(VENUES)]
    last = t1.groupby(["date", "jcd"])["deadline"].max()
    mid = last[last.astype(str) >= "22:30"].reset_index()[["date", "jcd"]]
    t1 = t1.merge(mid, on=["date", "jcd"]); t1 = t1[t1["rno"].astype(int) >= 7]
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    cl = norm(pd.read_parquet(f"{a.datadir}/odds.parquet"))
    t1 = norm(t1)
    m = r.merge(t1[KEY], on=KEY).merge(cl[KEY], on=KEY)
    if not len(m):
        msg = f"実行前: {START} 以降のミッドナイト（大村・下関）7R以降で、締切直前オッズ・確定オッズ・結果がそろうレースなし"
        open(a.out, "w").write(msg + "\n"); print(msg); return
    O1 = m[KEY].merge(t1, on=KEY, how="left")[COMBOS].to_numpy(float); OC = m[KEY].merge(cl, on=KEY, how="left")[COMBOS].to_numpy(float)
    P1, v1 = market(O1); PC, vc = market(OC)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values; okY = ~pd.isna(y); n = len(m)
    hit = np.zeros((n, 120), bool); hit[np.where(okY)[0], y[okY].astype(int)] = True
    prm = dict(np.load(a.frozen)); PI = apply(P1, f_features(a.datadir, m[KEY]), prm)
    rows = []
    def add(name, pick, mask):
        st = np.where(mask, 100.0, 0); rt = np.where(mask & hit[np.arange(n), pick] & vc[np.arange(n), pick], OC[np.arange(n), pick] * 100, 0)
        nb = int(mask.sum()); roi = rt.sum() / st.sum() if nb else np.nan; lo, hi = roi_ci(m["date"].values, st, rt)
        rows.append({"買い方": name, "点数": nb, "的中": int((mask & hit[np.arange(n), pick]).sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi})
    fav1 = np.argmin(np.where(v1, O1, np.inf), 1); has = v1.any(1)
    add("締切直前の1番人気（判定に使う）", fav1, has)
    ev = PI[np.arange(n), fav1] * np.where(v1[np.arange(n), fav1], O1[np.arange(n), fav1], 0)
    add("参考：うち IDEA-038 補正の期待値 0.85 以上", fav1, has & (ev >= 0.85))
    favc = np.argmin(np.where(vc, OC, np.inf), 1); add("参考：確定オッズの1番人気（上限）", favc, vc.any(1))
    R = pd.DataFrame(rows); nb = rows[0]["点数"]
    state = ("採用候補" if (rows[0]["回収率"] >= 1.1 and rows[0]["95%CI下"] > 1.0) else "不採用") if nb >= 300 else f"判定前（{nb}/300レース）"
    md = [f"# IDEA-042 大村・下関のミッドナイト後半で1番人気の3連単（{START}〜{m['date'].max()}、{n}レース）", "", f"**{state}**", "",
          R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
