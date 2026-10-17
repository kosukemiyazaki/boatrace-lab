"""IDEA-044 展示の進入が枠なりでないレースを市場は正しく値付けしているか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea044.py DATADIR [--out results/idea044.md]
- 区分：展示の進入（beforeinfo の ex_course1〜6）が枠なり／枠なりでない。期間 2025-10〜2026-08
- (1) β、(2) 1番人気の的中率 ÷ 見込み、(3) 内へ動いた艇・押し出された艇の1着率 ÷ 見込み、(4) 1番人気・全組の回収率
- 買い方：枠なりでないレースで β を 2025-10〜2026-03 で推定し、2026-04〜08 で π×オッズ > 1.0・300倍以下を各100円
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea043 import beta, boot_ratio

KEY = ["date", "jcd", "rno"]; START, END, SPLIT = "20251001", "20260831", "20260331"
RNG = np.random.default_rng(0)

def boot_roi(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(g["rt"].sum() / g["st"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea044.md")
    a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    b = norm(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet", columns=KEY + [f"ex_course{i}" for i in range(1, 7)])).drop_duplicates(KEY, keep="last")
    m = r.merge(b, on=KEY).merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY)
    m = m[(m["date"] >= START) & (m["date"] <= END)].reset_index(drop=True)
    ex = m[[f"ex_course{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    # 展示の進入：6艇とも 1〜6 が入っていて重複なし（欠場のレースは除く）
    exok = np.isfinite(ex).all(1) & (np.sort(np.nan_to_num(ex), 1) == np.arange(1, 7)).all(1)
    ok = (v.sum(1) >= 60) & ~pd.isna(y) & exok
    m, O, v, ex = m[ok].reset_index(drop=True), O[ok], v[ok], ex[ok].astype(int); y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); L = np.log(np.clip(P, 1e-12, None))
    first = np.array([int(c[0]) for c in COMBOS]); pw = np.stack([P[:, first == bb].sum(1) for bb in range(1, 7)], 1)
    won = np.zeros((n, 6)); won[np.arange(n), first[y] - 1] = 1
    hit = np.zeros_like(v); hit[np.arange(n), y] = True
    boatno = np.arange(1, 7)[None, :]; wakunari = (ex == boatno).all(1)
    days = np.asarray(m["date"].astype(str)); fav = np.argmin(np.where(v, O, np.inf), 1)
    rows = []
    for name, s in (("枠なり", wakunari), ("枠なりでない", ~wakunari)):
        ns = s.sum(); fh = hit[s][np.arange(ns), fav[s]]; fp = P[s][np.arange(ns), fav[s]]
        rr, lo, hi = boot_ratio(days[s], fh.astype(float), fp)
        rf, _, _ = boot_roi(days[s], np.full(ns, 100.0), np.where(fh, O[s][np.arange(ns), fav[s]] * 100, 0))
        ra = np.where(v[s] & hit[s], O[s], 0).sum() / v[s].sum()
        rows.append({"区分": name, "レース": int(ns), "β": beta(L[s], y[s]), "1番人気の的中率": fh.mean(), "見込み": fp.mean(),
                     "実際÷見込み": rr, "95%CI下": lo, "95%CI上": hi, "1番人気の回収率": rf, "全組の回収率": ra})
    R = pd.DataFrame(rows)
    # 動いた艇（枠なりでないレースのみ）
    s = ~wakunari; rows2 = []
    for name, msk in (("内へ動いた艇", ex < boatno), ("外へ押し出された艇", ex > boatno), ("枠どおりの艇", ex == boatno)):
        sel = msk & s[:, None]; rr, lo, hi = boot_ratio(np.repeat(days[:, None], 6, 1)[sel], won[sel], pw[sel])
        rows2.append({"艇": name, "艇数": int(sel.sum()), "1着": int(won[sel].sum()), "見込み": float(pw[sel].sum()), "実際÷見込み": rr, "95%CI下": lo, "95%CI上": hi})
    # 艇番 × 展示の進入（枠なりでない、艇数の多い組）
    rows3 = []
    for bb in range(6):
        for c in range(1, 7):
            sel = s & (ex[:, bb] == c) & (c != bb + 1)
            if sel.sum() >= 300:
                rr, lo, hi = boot_ratio(days[sel], won[sel, bb], pw[sel, bb])
                rows3.append({"艇番": bb + 1, "展示の進入": c, "艇数": int(sel.sum()), "1着率": won[sel, bb].mean(), "見込み": pw[sel, bb].mean(), "実際÷見込み": rr, "95%CI下": lo, "95%CI上": hi})
    # 買い方（枠なりでない）
    tr = s & (days <= SPLIT); ev = s & (days > SPLIT)
    bt = beta(L[tr], y[tr]); Pb = np.exp(bt * L[ev]); Pb /= Pb.sum(1, keepdims=True)
    buy = v[ev] & (O[ev] <= 300) & (Pb * np.where(v[ev], O[ev], 0) > 1.0)
    nb = int(buy.sum()); roi, clo, chi = boot_roi(days[ev], buy.sum(1) * 100.0, np.where(buy & hit[ev], O[ev] * 100, 0).sum(1))
    judge = "採用候補（締切前オッズでの前向きの検証へ）" if (nb >= 200 and roi >= 1.1 and clo > 1.0) else "不採用"
    md = [f"# IDEA-044 展示の進入が枠なりでないレース（{START}〜{END}、{n}レース、枠なりでない {int(s.sum())}レース）", "",
          f"**判定：{judge}**　枠なりでないレースで β={bt:.3f}（〜{SPLIT} で推定）、{SPLIT[:6]}より後で π×オッズ > 1.0・300倍以下：{nb}点、回収率 {roi:.3f}（日単位95%CI {clo:.3f}〜{chi:.3f}）", "",
          "## 区分ごと", "", R.to_markdown(index=False, floatfmt=".3f"), "",
          "## 枠なりでないレースで、動いた艇の1着", "", pd.DataFrame(rows2).to_markdown(index=False, floatfmt=".3f"), "",
          "## 艇番 × 展示の進入（枠なりでないレース、300艇以上）", "", pd.DataFrame(rows3).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
