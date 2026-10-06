"""ベースライン: 締切時オッズが上限以下の組番を全部（各100円）買った場合の回収率。
Step 2（walkforward.py）と同じ評価期間・同じ払戻計算（F/L/欠場は返還）で出し、モデルの上乗せ分を明示する。
usage: python scripts/baseline_allcombos.py DATADIR WALKFORWARD_JSON [--out results/baseline_allcombos.md]
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from walkforward import load, TEST_START
from boatlib.parse import COMBOS

RNG = np.random.default_rng(0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("wf"); ap.add_argument("--out", default="results/baseline_allcombos.md")
    a = ap.parse_args()
    wf = json.load(open(a.wf)); lo, hi = wf["summary"]["eval_period"]
    r, X, mask, refund, pos, O = load(a.datadir)
    d = r["date"].to_numpy(dtype=object).astype(str)
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    pay = r["tri_pay"].values.astype(float)
    ref = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref[:, ci] |= refund[:, int(b) - 1]
    sel = (d >= lo) & (d <= hi) & (np.isfinite(O).sum(1) >= 60)
    idx = np.where(sel)[0]
    rows = []
    for omax in (30, 50, 100):
        buy = np.isfinite(O[idx]) & (O[idx] <= omax)
        hit = buy & (np.arange(120)[None, :] == y[idx][:, None]) & ~ref[idx]
        rf = buy & ref[idx]
        ret_race = (hit.any(1) * pay[idx]) + rf.sum(1) * 100
        st_race = buy.sum(1) * 100
        df = pd.DataFrame({"day": d[idx], "st": st_race, "rt": ret_race}).groupby("day").sum()
        st, rt = df["st"].values, df["rt"].values
        bi = RNG.integers(0, len(st), size=(5000, len(st)))
        rois = rt[bi].sum(1) / st[bi].sum(1)
        roi = rt.sum() / st.sum()
        models = [g for g in wf["results"] if g["omax"] == omax]
        rows.append({"オッズ上限": omax, "点数": int(buy.sum()), "回収率（全組番）": roi,
                     "95%CI下": float(np.quantile(rois, .025)), "95%CI上": float(np.quantile(rois, .975)),
                     **{f"モデル EV>{g['thr']}": g["roi"] for g in models},
                     **{f"上乗せ EV>{g['thr']}（pt）": (g["roi"] - roi) * 100 for g in models}})
    md = [f"評価期間 {lo}〜{hi}（{sel.sum()}レース）、Step 2 と同じ期間・払戻計算", "",
          pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
