"""IDEA-045 荒れる条件のレースで、穴（100倍以上）を薄く買う（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea045.py DATADIR [--out results/idea045.md]
- 条件 C1〜C6 ごとに、万舟の出現率 ÷ 市場の見込み、確定100倍以上の組を市場の確率の高い順に5点（各100円）の回収率
- 学習 2023-11〜2025-12 で回収率が一番高い条件を選び、評価 2026-01〜08 で判定
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm

KEY = ["date", "jcd", "rno"]; TR = ("20231101", "20251231"); EV = ("20260101", "20260831")
RNG = np.random.default_rng(0)

def boot(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(g["rt"].sum() / g["st"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea045.md")
    a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["wind", "wave", "tri_combo"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= TR[0]) & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "cls"]))
    c1 = e[e["boat"] == 1][KEY + ["cls"]].drop_duplicates(KEY).rename(columns={"cls": "cls1"})
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(c1, on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    first = np.array([int(c[0]) for c in COMBOS]); p1 = P[:, first == 1].sum(1)
    hit = np.zeros_like(v); hit[np.arange(n), y] = True
    big = v & (O >= 100)
    # 100倍以上の中で市場の確率が高い順に5点
    Pb = np.where(big, P, -1); top = np.argsort(-Pb, 1)[:, :5]
    buy5 = np.zeros_like(v); buy5[np.arange(n)[:, None], top] = True; buy5 &= big
    st5 = buy5.sum(1) * 100.0; rt5 = np.where(buy5 & hit, O * 100, 0).sum(1)
    stA = big.sum(1) * 100.0; rtA = np.where(big & hit, O * 100, 0).sum(1)
    manshu = (O[np.arange(n), y] >= 100).astype(float); exp_m = np.where(big, P, 0).sum(1)
    days = m["date"].values; tr = (days <= TR[1]); ev = (days >= EV[0])
    wind = pd.to_numeric(m["wind"], errors="coerce").values; wave = pd.to_numeric(m["wave"], errors="coerce").values
    # 荒れやすい場：学習期間の万舟率 上位6場
    vr = pd.Series(manshu[tr]).groupby(m["jcd"].values[tr]).mean().sort_values(ascending=False); rough = set(vr.index[:6])
    conds = {"全レース": np.ones(n, bool), "C1 風速5m以上": wind >= 5, "C2 波高5cm以上": wave >= 5, "C3 1号艇の1着確率40%未満": p1 < 0.4,
             "C4 1号艇がB級": m["cls1"].astype(str).str.startswith("B").values, f"C5 荒れやすい場（{','.join(sorted(rough))}）": m["jcd"].isin(rough).values,
             "C6 風5m以上かつ1号艇50%未満": (wind >= 5) & (p1 < 0.5)}
    rows = []
    for name, c in conds.items():
        for per, msk in (("学習", tr), ("評価", ev)):
            s = c & msk
            if s.sum() < 30:
                continue
            roi5, lo5, hi5 = boot(days[s], st5[s], rt5[s]); roiA = rtA[s].sum() / stA[s].sum()
            rows.append({"条件": name, "期間": per, "レース": int(s.sum()), "万舟率": manshu[s].mean(), "市場の見込み": exp_m[s].mean(),
                         "実際÷見込み": manshu[s].sum() / exp_m[s].sum(), "穴5点の回収率": roi5, "95%CI下": lo5, "95%CI上": hi5, "100倍以上を全部": roiA})
    R = pd.DataFrame(rows)
    trR = R[(R["期間"] == "学習") & (R["条件"] != "全レース")].sort_values("穴5点の回収率", ascending=False)
    best = trR.iloc[0]["条件"]; evb = R[(R["期間"] == "評価") & (R["条件"] == best)].iloc[0]
    judge = "採用候補（締切前オッズでの前向き検証へ）" if (evb["レース"] >= 300 and evb["穴5点の回収率"] >= 1.1 and evb["95%CI下"] > 1.0) else "不採用"
    md = [f"# IDEA-045 荒れる条件で穴を薄く買う（学習 {TR[0]}〜{TR[1]}／評価 {EV[0]}〜{EV[1]}、{n}レース）", "",
          f"**判定：{judge}**　学習で選んだ条件「{best}」の評価：{int(evb['レース'])}レース、穴5点の回収率 {evb['穴5点の回収率']:.3f}（95%CI {evb['95%CI下']:.3f}〜{evb['95%CI上']:.3f}）", "",
          R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
