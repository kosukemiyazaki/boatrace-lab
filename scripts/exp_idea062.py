"""IDEA-062 公式コンピュータ予想の逆（IDEA-006 の続き）：単勝と自信度で見る（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea062.py DATADIR STOREDIR [--out results/idea062.md]
- 予想：STOREDIR/pcexpect/*.csv.gz（pc_tri = 予想の3連単10組、pc_conf = 自信度）
- (1) 予想の10組の1着に1度も出てこない艇の単勝、(2) 予想の1組目の1着の艇（予想の本命）の単勝。艇番・自信度の区分ごと
- 1着の市場の見込み = 確定3連単オッズの逆数を正規化した1着確率。単勝の払戻は K ファイルの実額（各100円）
- 探索 2025-10〜2026-03 で区分を選ぶ（300点以上・回収率100%以上 または 実際÷見込み 1.08以上）→ 評価 2026-04〜08 で判定
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm

KEY = ["date", "jcd", "rno"]; EXP = ("20251001", "20260331"); EVL = ("20260401", "20260831")
RNG = np.random.default_rng(0)

def k_tan(bkdir, start, end):
    """K ファイルの単勝：date, jcd, rno, win_boat, tan_pay（単勝の行がないレースは返還などとして除く）"""
    rows = []
    for fn in sorted(glob.glob(f"{bkdir}/*/k*.txt.gz")):
        date = "20" + os.path.basename(fn)[1:7]
        if not (start <= date <= end):
            continue
        txt = gzip.open(fn, "rt", encoding="utf-8", errors="replace").read()
        for jcd, blk in BP._blocks(txt, "K"):
            rno = None
            for line in blk.splitlines():
                z = BP.z2h(line)
                m = BP.K_RACE.match(z)
                if m:
                    rno = int(m.group(1)); continue
                if rno is None:
                    continue
                mm = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", z)
                if mm:
                    rows.append((date, jcd, rno, int(mm.group(1)), int(mm.group(2)))); rno = None
    return pd.DataFrame(rows, columns=KEY + ["win_boat", "tan_pay"]).drop_duplicates(KEY, keep="last")

def boot(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(g["rt"].sum() / g["st"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def ratio_ci(days, won, p):
    g = pd.DataFrame({"d": days, "w": won, "p": p}).groupby("d").sum()
    if len(g) < 2 or g["p"].sum() <= 0:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["w"].values[bi].sum(1) / g["p"].values[bi].sum(1)
    return float(g["w"].sum() / g["p"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def conf_band(c):
    c = pd.to_numeric(c, errors="coerce")
    return np.where(c >= 5, "自信度5以上", np.where(c >= 3, "自信度3〜4", np.where(c >= 0, "自信度2以下", "不明")))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea062.md")
    a = ap.parse_args()
    pc = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str, "pc_tri": str}) for f in sorted(glob.glob(f"{a.store}/pcexpect/*.csv.gz"))], ignore_index=True)
    pc = norm(pc).drop_duplicates(KEY, keep="last"); pc = pc[pc["pc_tri"].notna() & (pc["date"] >= EXP[0]) & (pc["date"] <= EVL[1])]
    od = norm(pd.read_parquet(f"{a.datadir}/odds.parquet")); od = od[(od["date"] >= EXP[0]) & (od["date"] <= EVL[1])]
    kt = norm(k_tan(f"{a.store}/bk", EXP[0], EVL[1]))
    m = pc.merge(od, on=KEY).merge(kt, on=KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); ok = v.sum(1) >= 60
    m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    first = np.array([int(c[0]) for c in COMBOS]); pw = np.stack([P[:, first == b].sum(1) for b in range(1, 7)], 1)
    picks = [s.split(";") for s in m["pc_tri"].values]
    firsts = np.zeros((n, 6), bool); fav = np.zeros(n, int)
    for i, ps in enumerate(picks):
        ps = [p for p in ps if re.match(r"^\d-\d-\d$", p)]
        for p in ps:
            firsts[i, int(p[0]) - 1] = True
        fav[i] = int(ps[0][0]) - 1 if ps else -1
    won = np.zeros((n, 6)); won[np.arange(n), m["win_boat"].values - 1] = 1
    pay = np.zeros((n, 6)); pay[np.arange(n), m["win_boat"].values - 1] = m["tan_pay"].values
    days = m["date"].values; band = conf_band(m["pc_conf"].values)
    per = np.where(days <= EXP[1], "探索", "評価")
    rows = []
    def add(kind, name, sel):  # sel: (n,6) bool
        for pr in ("探索", "評価"):
            s = sel & (per == pr)[:, None]; r_i, b_i = np.where(s)
            if len(r_i) == 0:
                continue
            d = days[r_i]; st = np.full(len(r_i), 100.0); rt = pay[r_i, b_i]
            roi, lo, hi = boot(d, st, rt); rr, rlo, rhi = ratio_ci(d, won[r_i, b_i], pw[r_i, b_i])
            rows.append({"種類": kind, "区分": name, "期間": pr, "点数": int(len(r_i)), "的中": int(won[r_i, b_i].sum()),
                         "実際÷見込み": rr, "比CI下": rlo, "比CI上": rhi, "回収率": roi, "95%CI下": lo, "95%CI上": hi})
    absent = ~firsts
    for b in range(6):
        col = np.zeros((n, 6), bool); col[:, b] = True
        add("(1) 予想の1着に出ない艇", f"{b + 1}号艇・全自信度", absent & col)
        for cb in ("自信度2以下", "自信度3〜4", "自信度5以上"):
            add("(1) 予想の1着に出ない艇", f"{b + 1}号艇・{cb}", absent & col & (band == cb)[:, None])
    add("(1) 予想の1着に出ない艇", "全艇・全自信度", absent)
    for cb in ("自信度2以下", "自信度3〜4", "自信度5以上"):
        add("(1) 予想の1着に出ない艇", f"全艇・{cb}", absent & (band == cb)[:, None])
    favm = np.zeros((n, 6), bool); favm[np.arange(n)[fav >= 0], fav[fav >= 0]] = True
    add("(2) 予想の本命", "全自信度", favm)
    for cb in ("自信度2以下", "自信度3〜4", "自信度5以上"):
        add("(2) 予想の本命", cb, favm & (band == cb)[:, None])
    allb = np.ones((n, 6), bool); add("参考：全艇の単勝", "全艇", allb)
    for b in range(6):
        col = np.zeros((n, 6), bool); col[:, b] = True; add("参考：全艇の単勝", f"{b + 1}号艇", col)
    R = pd.DataFrame(rows)
    ex = R[(R["期間"] == "探索") & ~R["種類"].str.startswith("参考") & (R["点数"] >= 300) & ((R["回収率"] >= 1.0) | (R["実際÷見込み"] >= 1.08))]
    sel = ex[["種類", "区分"]].merge(R[R["期間"] == "評価"], on=["種類", "区分"])
    passed = sel[(sel["点数"] >= 300) & (sel["回収率"] >= 1.1) & (sel["95%CI下"] > 1.0)]
    judge = "採用候補（前向きの検証へ）" if len(passed) else "不採用"
    selmd = ex.merge(sel, on=["種類", "区分"], suffixes=("_探索", "_評価"))[["種類", "区分", "点数_探索", "回収率_探索", "実際÷見込み_探索", "点数_評価", "回収率_評価", "95%CI下_評価", "95%CI上_評価", "実際÷見込み_評価"]] if len(ex) else pd.DataFrame()
    md = [f"# IDEA-062 公式コンピュータ予想の逆：単勝と自信度（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}、{n}レース）", "",
          f"**判定：{judge}**　探索で選ばれた区分 {len(ex)} 個、評価で基準（300点以上・110%以上・CI下限>100%）を満たした区分 {len(passed)} 個。", "",
          "- 市場の見込み：確定3連単オッズの逆数を正規化した1着確率。単勝の払戻は K ファイルの実額（各100円）。自信度は pc_conf（2以下／3〜4／5以上）。", "",
          "## 探索で選ばれた区分と評価", "", selmd.to_markdown(index=False, floatfmt=".3f") if len(selmd) else "（なし）", "",
          "## 全区分", "", R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:8])); print(selmd.to_string() if len(selmd) else "no sel")
    print(R[R["区分"].isin(["全艇・全自信度", "全自信度", "全艇"])].to_string())

if __name__ == "__main__":
    main()
