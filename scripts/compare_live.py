"""締切6分前オッズ(live)と締切時オッズの差を人気帯別に集計する。
usage: python scripts/compare_live.py STOREDIR [--out results/live_vs_closing.md]
- 3連単: 6分前オッズの人気順位帯ごとに、締切時/6分前 のオッズ比（中央値・平均・四分位）と、
  市場の見込み確率（オッズ逆数の正規化）の比を出す
- 単勝: 同じく艇ごとに
- 評価（6分前オッズでの回収率など）はここではしない（約2,000レースたまるまで実施しない）
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

BANDS = [(1, 1), (2, 3), (4, 10), (11, 30), (31, 60), (61, 120)]

def read(pattern, key=("date", "jcd", "rno")):
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs], ignore_index=True)
    df["jcd"] = df["jcd"].str.zfill(2)
    return df.drop_duplicates(list(key), keep="last")

def summarize(ratio, rank, bands):
    rows = []
    for lo, hi in bands:
        m = (rank >= lo) & (rank <= hi) & np.isfinite(ratio)
        v = ratio[m]
        if len(v) == 0:
            continue
        rows.append(dict(band=f"{lo}-{hi}番人気" if lo != hi else f"{lo}番人気", n=int(len(v)),
                         median=float(np.median(v)), mean=float(v.mean()),
                         q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)),
                         down_share=float((v < 1).mean())))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("store"); ap.add_argument("--out", default="results/live_vs_closing.md")
    a = ap.parse_args()
    out = []
    # 3連単
    lv = read(f"{a.store}/live/odds3t/*.csv.gz"); cl = read(f"{a.store}/odds/*.csv.gz")
    if len(lv) and len(cl):
        m = lv.merge(cl, on=["date", "jcd", "rno"], suffixes=("_live", "_close"))
        L = m[[c + "_live" for c in COMBOS]].to_numpy(float); C = m[[c + "_close" for c in COMBOS]].to_numpy(float)
        ok = np.isfinite(L).sum(1) >= 60
        L, C = L[ok], C[ok]
        rank = (-np.where(np.isfinite(L), 1 / L, 0)).argsort(1).argsort(1) + 1  # 6分前の人気順位
        lead = (pd.to_datetime(m["deadline"][ok], format="%H:%M") - pd.to_datetime(m["fetched_at"][ok], format="%H:%M:%S")).dt.total_seconds() / 60
        out.append(f"## 3連単（{ok.sum()}レース、{m['date'][ok].nunique()}日、取得は締切 {lead.median():.1f} 分前（中央値））\n")
        out.append("締切時オッズ ÷ 6分前オッズ（1未満 = 締切までにオッズが下がった）\n")
        out.append(summarize((C / L).ravel(), rank.ravel(), BANDS).to_markdown(index=False, floatfmt=".3f"))
        pl = np.where(np.isfinite(L), 1 / L, 0); pl /= pl.sum(1, keepdims=True)
        pc = np.where(np.isfinite(C), 1 / C, 0); pc /= pc.sum(1, keepdims=True)
        out.append("\n\n市場の見込み確率の比（締切時 ÷ 6分前）\n")
        out.append(summarize((pc / np.where(pl > 0, pl, np.nan)).ravel(), rank.ravel(), BANDS).to_markdown(index=False, floatfmt=".3f"))
    # 単勝
    lv = read(f"{a.store}/live/oddstf/*.csv.gz"); cl = read(f"{a.store}/oddstf/*.csv.gz")
    if len(lv) and len(cl):
        cols = [f"tan{b}" for b in range(1, 7)]
        m = lv.merge(cl, on=["date", "jcd", "rno"], suffixes=("_live", "_close"))
        L = m[[c + "_live" for c in cols]].to_numpy(float); C = m[[c + "_close" for c in cols]].to_numpy(float)
        rank = (-np.where(np.isfinite(L), 1 / L, 0)).argsort(1).argsort(1) + 1
        out.append(f"\n\n## 単勝（{len(m)}レース）\n\n締切時オッズ ÷ 6分前オッズ\n")
        out.append(summarize((C / L).ravel(), rank.ravel(), [(1, 1), (2, 2), (3, 3), (4, 6)]).to_markdown(index=False, floatfmt=".3f"))
    txt = "\n".join(out) if out else "データなし"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(txt + "\n")
    print(txt)

if __name__ == "__main__":
    main()
