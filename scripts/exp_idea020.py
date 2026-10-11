"""IDEA-020: 売上の小さいレースほど市場の値付けが粗いか（BOATCAST の確定票数で売上を出す。docs/ideas.md の登録どおり）
usage: python scripts/exp_idea020.py DATADIR [--out results/idea020]
1. 売上五分位 × 締切時オッズ帯の全組番回収率（2025-10〜2026-08、各100円、F/L/欠場は返還）
2. 売上五分位ごとの「B'（IDEA-011 のスタッキング×市場）− 市場」の3連単対数損失（評価 2026Q1〜Q3）。最小五分位と最大五分位の差の差と95%CI
3. 売上五分位ごとの、1000円（10票）を足したときのオッズの下がり方（10〜50倍の組番、中央値）
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS_C1
from boatlib.model import blend, fit_blend
from boatlib.parse import COMBOS
from exp_idea011 import MK, fit_predict, market, market_feats
from walkforward import TEST_START, load, month_add

QSTARTS = ["202601", "202604", "202607"]
START, EVAL_END = "20251001", "20260831"
BANDS = [(0, 10), (10, 30), (30, 100), (100, 1e9)]
RNG = np.random.default_rng(0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea020")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    key = ["date", "jcd", "rno"]
    r, X, mask, refund, pos, O = load(a.datadir, FEATS_C1)
    d = r["date"].to_numpy(dtype=object).astype(str); ym = np.array([s[:6] for s in d])
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    v = pd.read_parquet(f"{a.datadir}/votes.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    rr = r[key].assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    vm = rr.merge(v, on=key, how="left")
    T = vm["votes_total"].to_numpy(float)
    V = vm[[f"v{c}" for c in COMBOS]].to_numpy(float)
    has = (np.isfinite(O).sum(1) >= 60) & np.isfinite(T) & (T > 0) & (d >= START) & (d <= EVAL_END)
    qs = np.full(len(r), -1)
    edges = np.quantile(T[has], [0.2, 0.4, 0.6, 0.8])
    qs[has] = np.searchsorted(edges, T[has], side="right")
    qlab = {i: f"Q{i + 1}（{'〜' if i == 0 else f'{edges[i - 1] * 100 / 1e4:,.0f}万円〜'}{f'{edges[i] * 100 / 1e4:,.0f}万円' if i < 4 else ''}）" for i in range(5)}
    pay = r["tri_pay"].values.astype(float)
    ref = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref[:, ci] |= refund[:, int(b) - 1]
    valid = np.isfinite(O)
    win = (np.arange(120)[None, :] == y[:, None]) & ~ref
    ret = np.where(win, pay[:, None], 0.0) + np.where(ref, 100.0, 0.0)
    # 1. 五分位 × オッズ帯の回収率
    rows = []
    for qi in range(5):
        m = (qs == qi)
        row = {"売上五分位": qlab[qi], "レース": int(m.sum()), "売上の中央値（万円）": float(np.median(T[m]) * 100 / 1e4)}
        for lo, hi in BANDS:
            mm = valid & (O >= lo) & (O < hi) & m[:, None]
            st = (mm * 100.0).sum(1); rt = (ret * mm).sum(1)
            df = pd.DataFrame({"d": d[m], "st": st[m], "rt": rt[m]}).groupby("d").sum()
            bi = RNG.integers(0, len(df), size=(2000, len(df)))
            b = df["rt"].values[bi].sum(1) / df["st"].values[bi].sum(1)
            lab = f"{lo}〜{hi}倍" if hi < 1e9 else f"{lo}倍〜"
            row[lab] = df["rt"].sum() / df["st"].sum(); row[f"{lab} CI下"] = float(np.quantile(b, .025))
        rows.append(row)
    t1 = pd.DataFrame(rows)
    # 3. 1000円（10票）を足したときのオッズの下がり方（10〜50倍の組番）
    rows3 = []
    for qi in range(5):
        m = (qs == qi)[:, None] & valid & (O >= 10) & (O <= 50) & np.isfinite(V) & (V > 0)
        Tm = np.repeat(T[:, None], 120, 1)
        before = Tm * 0.75 / V; after = (Tm + 10) * 0.75 / (V + 10)
        drop = (1 - after / before)[m]
        rows3.append({"売上五分位": qlab[qi], "組番数": int(m.sum()), "オッズの下がり（中央値）": float(np.median(drop)), "90%点": float(np.quantile(drop, .9))})
    t3 = pd.DataFrame(rows3)
    # 2. B' − 市場（評価 2026Q1〜Q3）
    PC = market(O); hasO = np.isfinite(O).sum(1) >= 60
    XB = np.concatenate([X, market_feats(PC, hasO)], axis=2); FB = FEATS_C1 + MK
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    PBp = np.full((len(r), 120), np.nan)
    for s in QSTARTS:
        ev = np.isin(ym, [s, month_add(s, 1), month_add(s, 2)]) & (d <= EVAL_END) & hasO
        tr = (d >= warm) & (ym <= month_add(s, -3)); es = ym == month_add(s, -2); ca = ym == month_add(s, -1)
        cm = ca & hasO
        PB = fit_predict(XB, mask, pos, FB, tr, es, ca, ev, y)
        (ba, bb), _ = fit_blend(PB[cm], PC[cm], y[cm])
        PBp[ev] = blend(PB[ev], PC[ev], ba, bb)
        print("quarter", s, "done", flush=True)
    ev = np.isfinite(PBp).all(1) & (qs >= 0)
    nll = lambda Q: -np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, None))
    diff = nll(PBp) - nll(PC)
    rows2 = []
    for qi in range(5):
        m = ev & (qs == qi)
        rows2.append({"売上五分位": qlab[qi], "レース": int(m.sum()), "市場": float(nll(PC)[m].mean()), "B'": float(nll(PBp)[m].mean()), "B'−市場": float(diff[m].mean())})
    t2 = pd.DataFrame(rows2)
    # 差の差（最小 − 最大）の日単位ブートストラップ
    df = pd.DataFrame({"d": d[ev], "q": qs[ev], "x": diff[ev]})
    g = df.groupby(["d", "q"])["x"].agg(["sum", "count"]).unstack(fill_value=0)
    days = g.index.values; bi = RNG.integers(0, len(days), size=(3000, len(days)))
    S0, N0 = g[("sum", 0)].values, g[("count", 0)].values; S4, N4 = g[("sum", 4)].values, g[("count", 4)].values
    with np.errstate(invalid="ignore", divide="ignore"):
        bd = S0[bi].sum(1) / N0[bi].sum(1) - S4[bi].sum(1) / N4[bi].sum(1)
    dd = float(t2.loc[0, "B'−市場"] - t2.loc[4, "B'−市場"]); ci = [float(np.nanquantile(bd, .025)), float(np.nanquantile(bd, .975))]
    adopt = dd <= -0.01 and ci[1] < 0
    md = [f"### IDEA-020 売上と市場の正確さ（{START}〜{EVAL_END}、売上のあるレース {int((qs >= 0).sum())}）", "",
          "#### 1. 売上五分位 × 締切時オッズ帯の全組番回収率", "", t1.to_markdown(index=False, floatfmt=".3f"), "",
          "#### 2. B'（スタッキング×市場）− 市場（3連単対数損失、評価 2026Q1〜Q3）", "", t2.to_markdown(index=False, floatfmt=".4f"), "",
          f"最小五分位 − 最大五分位 = {dd:+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}） → **{'小さい市場に絞る買い方を候補にする' if adopt else '売上による違いは基準に届かない'}**", "",
          "#### 3. 1000円（10票）を足したときのオッズの下がり方（10〜50倍の組番）", "", t3.to_markdown(index=False, floatfmt=".4f")]
    json.dump(dict(t1=rows, t2=rows2, t3=rows3, diff_of_diff=dd, ci95=ci, adopt=adopt), open(f"{a.out}/idea020.json", "w"), ensure_ascii=False, indent=1)
    open(f"{a.out}/idea020.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
