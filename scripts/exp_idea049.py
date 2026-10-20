"""IDEA-049 市場が「荒れる」と見ている日に、堅いレースの本命を買う（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea049.py DATADIR [--out results/idea049.md]
- D1 その場・その日の市場の万舟見込みの平均（3分位）、D2 同じ場でそのレースより前に出た万舟 − 見込み
- 堅いレース = 1号艇の市場の1着確率 60%以上。(a) 1番人気1点 (b) 1号艇1着を全部 (c) 1号艇1着で π×オッズ>1（π=P^1.06）
- 探索 2023-11〜2025-12 で区分を1つ選び、評価 2026-01〜08 で判定
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
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea049.md")
    a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= TR[0]) & (r["date"] <= EV[1])]
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).sort_values(KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    first = np.array([int(c[0]) for c in COMBOS]); p1 = P[:, first == 1].sum(1)
    hit = np.zeros_like(v); hit[np.arange(n), y] = True
    exp_m = np.where(v & (O >= 100), P, 0).sum(1); manshu = (O[np.arange(n), y] >= 100).astype(float)
    days = m["date"].values; tr = days <= TR[1]; ev = days >= EV[0]
    # D1: 場×日の万舟見込みの平均（3分位は探索期間で決める）
    g = m[KEY].copy(); g["exp_m"] = exp_m; d1 = g.groupby(["date", "jcd"])["exp_m"].transform("mean").values
    q = np.quantile(d1[tr], [1 / 3, 2 / 3]); D1 = np.digitize(d1, q)
    # D2: 同じ場でそのレースより前の 万舟 − 見込み
    res = manshu - exp_m; prior = g.assign(res=res).groupby(["date", "jcd"])["res"].transform(lambda s: s.cumsum().shift(fill_value=0)).values
    D2 = np.digitize(prior, [0.0001, 1, 2])  # 0: ≤0, 1: 0〜1, 2: 1〜2, 3: 2以上
    hard = p1 >= 0.6
    fav = np.argmin(np.where(v, O, np.inf), 1)
    bA = np.zeros_like(v); bA[np.arange(n), fav] = True
    bB = v & (first[None, :] == 1)
    pi = P ** 1.06; pi /= pi.sum(1, keepdims=True); bC = bB & (pi * np.where(v, O, 0) > 1.0)
    buys = {"(a) 1番人気1点": bA, "(b) 1号艇1着を全部": bB, "(c) 1号艇1着でπ×オッズ>1": bC}
    rows = []
    for dn, D, labels in (("D1 市場の見込み", D1, ["低", "中", "高"]), ("D2 ここまでの荒れ", D2, ["0以下", "0〜1", "1〜2", "2以上"])):
        for k, lab in enumerate(labels):
            for bn, b in buys.items():
                st = b.sum(1) * 100.0; rt = np.where(b & hit, O * 100, 0).sum(1)
                row = {"定義": dn, "区分": lab, "買い方": bn}
                for per, msk in (("探索", tr), ("評価", ev)):
                    s = hard & msk & (D == k); roi, lo, hi = boot(days[s], st[s], rt[s])
                    row[f"{per} レース"] = int(s.sum()); row[f"{per} 回収率"] = roi; row[f"{per} CI下"] = lo; row[f"{per} CI上"] = hi
                rows.append(row)
    R = pd.DataFrame(rows)
    cand = R[R["探索 レース"] >= 300].sort_values("探索 回収率", ascending=False); best = cand.iloc[0]
    judge = "採用候補（前向きの検証へ）" if (best["評価 レース"] >= 300 and best["評価 回収率"] >= 1.1 and best["評価 CI下"] > 1.0) else "不採用"
    md = [f"# IDEA-049 荒れる日に堅いレースの本命（探索 {TR[0]}〜{TR[1]}／評価 {EV[0]}〜{EV[1]}、堅いレース = 1号艇60%以上）", "",
          f"**判定：{judge}**　探索で一番よかった区分「{best['定義']}・{best['区分']}・{best['買い方']}」（探索 {best['探索 回収率']:.3f}）の評価：{int(best['評価 レース'])}レース、回収率 {best['評価 回収率']:.3f}（95%CI {best['評価 CI下']:.3f}〜{best['評価 CI上']:.3f}）", "",
          f"参考：堅いレース全体での回収率（評価）— " + "、".join(f"{bn} {np.where(b & hit, O * 100, 0)[hard & ev].sum() / (b.sum(1) * 100.0)[hard & ev].sum():.3f}" for bn, b in buys.items()), "",
          R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
