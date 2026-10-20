"""IDEA-056 偏りを重ねて1号艇が強いレースで、単勝・2連単（1号艇1着）・複勝を買う（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea056.py DATADIR STOREDIR [--out results/idea056.md]
- IDEA-053 の21区分の重み（探索 2023-11〜2025-12 で学習）で 補正後÷市場 を出し、1.15・1.25 以上のレースで
  (a) 1号艇の単勝、(b) 1号艇1着の2連単5点、(c) 1号艇の複勝 を各100円
- 対象は確定の単勝（store oddstf）と2連単（odds2tf）がそろうレース（2025-10 以降）。払戻は K ファイルの実額（確定オッズと同じ）
- 探索 2025-10〜2025-12、評価 2026-01〜08
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY, TR, EV, FIRST, boat_feats, segments
from exp_idea053 import design, fit, boot
from idea052 import k_pay

XP = ("20251001", "20251231")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea056.md")
    a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]))
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(boat_feats(e), on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); p1 = P[:, FIRST == 1].sum(1); m["p1"] = p1
    win1 = (FIRST[y] == 1).astype(float)
    days = m["date"].values; tr = (days >= TR[0]) & (days <= TR[1])
    S = segments(m, tr, True, a.store); X = design(S, n)
    off = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1))
    w = fit(X[tr], off[tr], win1[tr])
    padj = 1 / (1 + np.exp(-(off + X @ w)))
    good = p1 > 0                                   # 市場の1号艇の確率が0のレース（割り算が無限大）は除く
    ratio = np.where(good, padj / np.where(good, p1, 1), np.nan)
    # 確定の単勝・2連単がそろうレースに絞る
    import glob
    tf = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in glob.glob(f"{a.store}/oddstf/*.csv.gz")])
    tf = norm(tf).drop_duplicates(KEY, keep="last")[KEY + ["tan1"]]
    o2 = norm(pd.read_parquet(f"{a.datadir}/odds2tf.parquet"))[KEY + ["1-2"]].rename(columns={"1-2": "o12"})
    mm = m[KEY].assign(i=np.arange(n)).merge(tf, on=KEY).merge(o2, on=KEY)
    mm = mm[pd.to_numeric(mm["tan1"], errors="coerce").gt(0) & pd.to_numeric(mm["o12"], errors="coerce").gt(0)]
    idx = mm["i"].values
    kp = k_pay(a.store, XP[0]); kp = kp.set_index(KEY + ["kind", "combo"])["pay"]
    def pay(kind, combo):
        keys = list(zip(m["date"].values[idx], m["jcd"].values[idx], m["rno"].values[idx]))
        return np.array([kp.get((d, j, int(rn), kind, combo), 0) for d, j, rn in keys], float)
    rt_tan = pay("tan", "1")
    rt_fuku = pay("fuku", "1")
    rt_2t = sum(pay("2t", f"1-{b}") for b in range(2, 7))
    st = {"(a) 単勝1": 100.0, "(b) 2連単 1-x 5点": 500.0, "(c) 複勝1": 100.0}
    rt = {"(a) 単勝1": rt_tan, "(b) 2連単 1-x 5点": rt_2t, "(c) 複勝1": rt_fuku}
    # K の払戻が取れていないレース（単勝の行がない）は除く
    haveK = np.array([kp.get((d, j, int(rn), "tan", str(b)), None) is not None for d, j, rn, b in
                      zip(m["date"].values[idx], m["jcd"].values[idx], m["rno"].values[idx], (FIRST[y[idx]]))])
    d_ = days[idx]; rat = ratio[idx]; xp = (d_ >= XP[0]) & (d_ <= XP[1]) & haveK; ev = (d_ >= EV[0]) & (d_ <= EV[1]) & haveK
    rows = []; judge_ok = False
    for thr in (None, 1.15, 1.25):
        for per, msk in (("探索 2025-10〜12", xp), ("評価 2026-01〜08", ev)):
            s = msk if thr is None else (msk & (rat >= thr))
            row = {"対象": "全レース（基準）" if thr is None else f"補正後÷市場 ≥{thr}", "期間": per, "レース": int(s.sum()),
                   "1号艇の1着率": win1[idx][s].mean() if s.any() else np.nan, "市場の見込み": p1[idx][s].mean() if s.any() else np.nan}
            for k in st:
                roi, lo, hi = boot(d_[s], np.full(s.sum(), st[k]), rt[k][s]) if s.sum() else (np.nan, np.nan, np.nan)
                row[k] = roi; row[k + " CI"] = f"{lo:.2f}〜{hi:.2f}"
                if thr is not None and per.startswith("評価") and s.sum() >= 300 and roi >= 1.1 and lo > 1.0:
                    judge_ok = True
            rows.append(row)
    R = pd.DataFrame(rows)
    judge = "採用候補（締切前オッズでの前向き検証へ）" if judge_ok else "不採用"
    md = [f"# IDEA-056 偏りを重ねて1号艇が強いレースで 単勝・2連単・複勝（重みは IDEA-053 の探索 {TR[0]}〜{TR[1]}）", "",
          f"**判定：{judge}**", "",
          f"対象：確定の単勝・2連単がそろい、K ファイルの払戻があるレース（探索 {int(xp.sum())}・評価 {int(ev.sum())}レース）。払戻は K の実額（複勝は1号艇が2着以内のときの実額）。市場の1号艇の確率が0のレースは除外。", "",
          R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
