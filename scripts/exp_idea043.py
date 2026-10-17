"""IDEA-043 安定板を使うレースは1号艇がさらに有利か（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea043.py DATADIR STOREDIR [--out results/idea043.md]
- 安定板の有無：STOREDIR/stab/*.csv.gz（直前情報ページの見出し）。期間 2026-01〜08
- (1) 1号艇の1着率 ÷ 市場の見込み（日単位95%CI）、(2) β、(3) 1番人気の3連単・1号艇1着の3連単の回収率
- 補正して買う：β を安定板ありの 2026-01〜04 で推定し、2026-05〜08 の1号艇1着の組で π×オッズ > 1.0 を各100円
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm

KEY = ["date", "jcd", "rno"]; RNG = np.random.default_rng(0)

def beta(L, y):
    def nll(b):
        s = b * L; s = s - s.max(1, keepdims=True); return -(s[np.arange(len(y)), y] - np.log(np.exp(s).sum(1))).mean()
    return minimize_scalar(nll, bounds=(0.5, 1.8), method="bounded").x

def boot_ratio(days, num, den):
    g = pd.DataFrame({"d": days, "n": num, "e": den}).groupby("d").sum()
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["n"].values[bi].sum(1) / g["e"].values[bi].sum(1)
    return float(g["n"].sum() / g["e"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea043.md")
    a = ap.parse_args()
    st = norm(pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in glob.glob(f"{a.store}/stab/*.csv.gz")])).drop_duplicates(KEY, keep="last")
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    m = r.merge(st[KEY + ["stab", "dist_hdr"]], on=KEY).merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY)
    m = m[(m["date"] >= "20260101") & (m["date"] <= "20260831")].reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 40) & ~pd.isna(y); m, O, v, y = m[ok].reset_index(drop=True), O[ok], v[ok], y[ok].astype(int)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); L = np.log(np.clip(P, 1e-12, None))
    first = np.array([int(c[0]) for c in COMBOS]); hit = np.zeros_like(v); hit[np.arange(len(y)), y] = True
    p1 = P[:, first == 1].sum(1); w1 = (first[y] == 1).astype(float); days = m["date"].values
    rows = []
    for g, name in ((1, "安定板あり"), (0, "安定板なし")):
        s = m["stab"].values == g
        ratio, lo, hi = boot_ratio(days[s], w1[s], p1[s])
        b = beta(L[s], y[s])
        fav = np.argmin(np.where(v[s], O[s], np.inf), 1); n = s.sum()
        fav_rt = np.where(hit[s][np.arange(n), fav], O[s][np.arange(n), fav], 0)
        b1 = v[s] & (first[None, :] == 1)
        r1 = np.where(b1 & hit[s], O[s], 0).sum() / b1.sum()
        rows.append({"区分": name, "レース": int(n), "1号艇の1着率": w1[s].mean(), "市場の見込み": p1[s].mean(),
                     "実際÷見込み": ratio, "95%CI下": lo, "95%CI上": hi, "β": b,
                     "1番人気の3連単の回収率": fav_rt.mean(), "1号艇1着の3連単を全部": r1})
    R = pd.DataFrame(rows)
    # 補正して買う（安定板あり、β は前半で推定・後半で評価）
    s = m["stab"].values == 1; tr = s & (m["date"].values <= "20260430"); ev = s & (m["date"].values >= "20260501")
    bt = beta(L[tr], y[tr]) if tr.sum() > 50 else np.nan
    Pb = np.exp(bt * L[ev]); Pb /= Pb.sum(1, keepdims=True)
    buy = v[ev] & (first[None, :] == 1) & (Pb * np.where(v[ev], O[ev], 0) > 1.0)
    stt = buy.sum(1) * 100.0; rt = np.where(buy & hit[ev], O[ev] * 100, 0).sum(1)
    g = pd.DataFrame({"d": days[ev], "st": stt, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    nb = int(buy.sum()); roi = g["rt"].sum() / g["st"].sum() if nb else np.nan
    if len(g) >= 2:
        bi = RNG.integers(0, len(g), size=(4000, len(g))); bb = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1); clo, chi = np.quantile(bb, [.025, .975])
    else:
        clo = chi = np.nan
    stronger = rows[0]["95%CI下"] > 1.0
    judge = "採用候補（前向きの検証へ）" if (stronger and nb >= 200 and roi >= 1.1 and clo > 1.0) else ("1号艇は見込みより強いが、市場は織り込んでいる（儲からない）" if stronger else "安定板で1号艇が見込み以上に強いとは言えない")
    md = [f"# IDEA-043 安定板と1号艇（2026-01〜08、安定板あり {int(s.sum())}レース）", "", f"**判定：{judge}**", "",
          R.to_markdown(index=False, floatfmt=".3f"), "",
          f"補正して買う（安定板あり、β={bt:.3f} を 2026-01〜04 で推定、2026-05〜08 で1号艇1着の π×オッズ > 1.0）：{nb}点、回収率 {roi:.3f}（95%CI {clo:.3f}〜{chi:.3f}）"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
