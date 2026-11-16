"""IDEA-077 逃げ下手の1号艇のレースで、2・3号艇のスタートの速いA級を買う（docs/ideas.md の登録どおり、1回だけ）
usage: python scripts/exp_idea077.py DATADIR HISTDS STOREDIR [--histbk /tmp/claude-0/hist] [--out results/idea077.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ）、STOREDIR / histbk：K ファイル（bk/YYYY/k*.txt.gz）
- L：1号艇の選手が IDEA-060 の縮めた比（探索 2023-11〜2025-12、見込み5回以上）の下位10%（exp_idea069.racer_top10 の下側）
- X：2・3号艇で A1・A2、かつレース前180日（当日を含まない）の実際のST平均（F・L・欠場の行を除く、10走以上）が1号艇の選手より小さい艇
- 買い方：(i) X の単勝、(ii) 2連単 X-1、(iii) 2連単 X-全（5点）。各100円、K の実額。返還（K0/K1/F/L0/L1）のあるレースと、K に単勝・2連単の行が無いレースは除く
- 12年の参考（オッズなし）：2016〜2023 の各年、前の2年で作った下位10%（1号艇の1着を年×場の平均で見込み、経験ベイズで縮める）
- 逃げ下手の選手のリストは出力しない（件数だけ）
"""
import argparse, glob, gzip, os, re, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import ratio_ci, boot_roi
from exp_idea060 import est_k, shrink, per_racer

KEY = ["date", "jcd", "rno"]
TR = ("20231101", "20251231"); EV = ("20260101", "20260831"); ADD = ("20260901", "20261115")
FIRST = np.array([int(c[0]) for c in COMBOS])
REFUND = {"F", "L0", "L1", "K0", "K1", "L", "K"}

def _k(f):
    """K ファイル1日分 -> [(date, jcd, rno, 単勝の艇, 単勝の払戻, 2連単の組, 2連単の払戻)]"""
    date = "20" + os.path.basename(f)[1:7]; out = []
    txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
    for jcd, blk in BP._blocks(txt, "K"):
        rno = None; tb = tp = c2 = p2 = None
        def flush():
            if rno is not None and tb is not None and c2 is not None:
                out.append((date, jcd, rno, tb, tp, c2, p2))
        for line in blk.splitlines():
            l = BP.z2h(line); m = BP.K_RACE.match(l)
            if m:
                flush(); rno = int(m.group(1)); tb = tp = c2 = p2 = None; continue
            if rno is None:
                continue
            if tb is None and (mm := re.match(r"^\s+単勝\s+(\d)\s+(\d+)", l)):
                tb, tp = int(mm.group(1)), int(mm.group(2)); continue
            if c2 is None and (mm := re.match(r"^\s+2連単\s+(\d)-(\d)\s+(\d+)", l)):
                c2, p2 = f"{mm.group(1)}-{mm.group(2)}", int(mm.group(3)); continue
        flush()
    return out

def kpay(dirs, start, end):
    fs = [f for d in dirs for f in sorted(glob.glob(f"{d}/bk/*/k*.txt.gz")) if start <= "20" + os.path.basename(f)[1:7] <= end]
    with Pool(4) as p:
        rows = [x for r in p.map(_k, fs, chunksize=20) for x in r]
    d = pd.DataFrame(rows, columns=KEY + ["tan_boat", "tan_pay", "c2", "p2"]); d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return d.drop_duplicates(KEY, keep="last")

def st180(e):
    """各出走（1〜3号艇）に、その選手のレース前180日のST平均と件数（当日を含まない）"""
    ok = e["st"].notna() & e["st_raw"].astype(str).str.match(r"^\d?\.\d+$")
    s = e[ok][["toban", "date", "st"]].copy()
    s["day"] = (pd.to_datetime(s["date"]) - pd.Timestamp("2000-01-01")).dt.days
    g = s.groupby(["toban", "day"])["st"].agg(["sum", "count"]).reset_index().sort_values(["toban", "day"]).reset_index(drop=True)
    tb = g["toban"].astype(np.int64).values; dy = g["day"].astype(np.int64).values
    key = tb * 100000 + dy
    cs = np.concatenate([[0], np.cumsum(g["sum"].values)]); cn = np.concatenate([[0], np.cumsum(g["count"].values)])
    q = e[["toban", "date"]].copy(); qd = (pd.to_datetime(q["date"]) - pd.Timestamp("2000-01-01")).dt.days.values.astype(np.int64)
    qt = pd.to_numeric(q["toban"], errors="coerce").fillna(0).astype(np.int64).values
    hi = np.searchsorted(key, qt * 100000 + qd, side="left"); lo = np.searchsorted(key, qt * 100000 + qd - 180, side="left")
    n = cn[hi] - cn[lo]; sm = cs[hi] - cs[lo]
    return np.where(n >= 10, sm / np.maximum(n, 1), np.nan)

def race_table(e):
    """1レース1行：1〜3号艇の選手・級別・ST平均、返還の有無、X の艇（2・3号艇）"""
    e = e.copy()
    e["st180"] = np.nan
    m123 = e["boat"] <= 3
    e.loc[m123, "st180"] = st180(e[m123])
    refund = e[e["pos_raw"].astype(str).isin(REFUND)][KEY].drop_duplicates().assign(refund=True)
    wide = e[m123].pivot_table(index=KEY, columns="boat", values=["toban", "cls", "st180"], aggfunc="first")
    wide.columns = [f"{a}{b}" for a, b in wide.columns]; wide = wide.reset_index()
    wide = wide.merge(refund, on=KEY, how="left"); wide["refund"] = wide["refund"].fillna(False).astype(bool)
    for b in (2, 3):
        wide[f"X{b}"] = wide[f"cls{b}"].isin(["A1", "A2"]) & wide["st1801"].notna() & wide[f"st180{b}"].notna() & (wide[f"st180{b}"] < wide["st1801"])
    return wide

def bets(T):
    """各レースの (i)(ii)(iii) の賭け金と払戻（100円単位）"""
    st = {k: np.zeros(len(T)) for k in ("i", "ii", "iii")}; rt = {k: np.zeros(len(T)) for k in ("i", "ii", "iii")}
    hit = {k: np.zeros(len(T)) for k in ("i", "ii", "iii")}
    tb = T["tan_boat"].values; tp = T["tan_pay"].values; c2 = T["c2"].astype(str).values; p2 = T["p2"].values
    first2 = np.array([int(c[0]) for c in c2]); second2 = np.array([int(c[2]) for c in c2])
    for b in (2, 3):
        x = T[f"X{b}"].values.astype(bool)
        st["i"] += x * 100; w = x & (tb == b); rt["i"] += np.where(w, tp, 0); hit["i"] += w
        st["ii"] += x * 100; w = x & (first2 == b) & (second2 == 1); rt["ii"] += np.where(w, p2, 0); hit["ii"] += w
        st["iii"] += x * 500; w = x & (first2 == b); rt["iii"] += np.where(w, p2, 0); hit["iii"] += w
    return st, rt, hit

NAMES = {"i": "(i) X の単勝", "ii": "(ii) 2連単 X-1", "iii": "(iii) 2連単 X-全（5点）"}

def roi_rows(T, label):
    st, rt, hit = bets(T); d = T["date"].values; nd = T["date"].nunique(); rows = []
    for k in ("i", "ii", "iii"):
        roi, lo, hi = boot_roi(d, st[k], rt[k]) if len(T) else (np.nan,) * 3
        rows.append({"期間": label, "買い方": NAMES[k], "レース": len(T), "1日あたり": len(T) / max(nd, 1), "点数": int(st[k].sum() / 100),
                     "的中": int(hit[k].sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi})
    t1 = np.where(T["tan_boat"].values == 1, T["tan_pay"].values, 0)
    roi, lo, hi = boot_roi(d, np.full(len(T), 100.0), t1) if len(T) else (np.nan,) * 3
    rows.append({"期間": label, "買い方": "参考：1号艇の単勝", "レース": len(T), "1日あたり": len(T) / max(nd, 1), "点数": len(T),
                 "的中": int((T["tan_boat"].values == 1).sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi})
    return rows

def low10_market(m, P, y):
    """IDEA-060 と同じ：探索の1号艇の1着の比を縮めた値の下位10%（見込み5回以上）"""
    e1 = (FIRST[y] == 1).astype(float); p = P[:, FIRST == 1].sum(1)
    d = pd.DataFrame({"t1": m["t1"].values, "w": e1, "p": p, "v": p * (1 - p)})[((m["date"] >= TR[0]) & (m["date"] <= TR[1])).values]
    g = per_racer(d, "t1", "w", "p", "v"); k, _ = est_k(g["o"].values, g["e"].values, g["v"].values)
    g["s"] = shrink(g["o"].values, g["e"].values, k); g = g[g["e"] >= 5]
    q10 = np.quantile(g["s"], .1)
    return set(g.loc[g["s"] <= q10, "t1"]), k, q10, len(g)

def low10_hist(E1, years):
    """オッズなし：その期間の1号艇の1着を、年×場の平均を見込みにして縮めた比の下位10%"""
    d = E1[E1["year"].isin(years)].copy()
    d["p"] = d.groupby(["year", "jcd"])["w"].transform("mean"); d["v"] = d["p"] * (1 - d["p"])
    g = per_racer(d, "t1", "w", "p", "v"); k, _ = est_k(g["o"].values, g["e"].values, g["v"].values)
    g["s"] = shrink(g["o"].values, g["e"].values, k); g = g[g["e"] >= 5]
    q10 = np.quantile(g["s"], .1)
    return set(g.loc[g["s"] <= q10, "t1"]), len(g)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("store")
    ap.add_argument("--histbk", default="/tmp/claude-0/hist"); ap.add_argument("--out", default="results/idea077.md"); a = ap.parse_args()
    cols = KEY + ["boat", "toban", "cls", "pos_raw", "st", "st_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in (a.histds, a.datadir)], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last").reset_index(drop=True)
    e["toban"] = pd.to_numeric(e["toban"], errors="coerce")
    W = race_table(e); print("race table", len(W), flush=True)
    K = kpay([a.histbk, a.store], "20140101", ADD[1]); print("K", len(K), flush=True)
    W = W.merge(K, on=KEY, how="inner")
    W = W[~W["refund"]].reset_index(drop=True)
    W["anyX"] = W["X2"] | W["X3"]
    # 市場（2023-10〜）と L のリスト
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= ADD[1])]
    t1 = e[e["boat"] == 1][KEY + ["toban"]].drop_duplicates(KEY).rename(columns={"toban": "t1"})
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(t1, on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    low, k60, q10, nrac = low10_market(m, P, y); print("low10", len(low), k60, q10, flush=True)
    W["L"] = W["toban1"].isin(low)
    mk = m[KEY].copy()
    for b in (1, 2, 3):
        mk[f"pm{b}"] = P[:, FIRST == b].sum(1)
    mk["win"] = FIRST[y]
    rows, ratios, lonly = [], [], []
    for lab, (s0, s1) in (("探索 2023-11〜2025-12（参考）", TR), ("評価 2026-01〜08（主）", EV), ("追加 2026-09-01〜11-15", ADD)):
        per = (W["date"] >= s0) & (W["date"] <= s1)
        T = W[per & W["L"] & W["anyX"]].reset_index(drop=True)
        rows += roi_rows(T, lab)
        # 実際÷見込み（1号艇、X）
        Tm = T.merge(mk, on=KEY, how="inner")
        if len(Tm):
            r1 = ratio_ci(Tm["date"].values, (Tm["win"] == 1).astype(float).values, Tm["pm1"].values)
            ax = np.concatenate([((Tm["win"] == b) & Tm[f"X{b}"]).astype(float).values for b in (2, 3)])
            ex = np.concatenate([np.where(Tm[f"X{b}"], Tm[f"pm{b}"], 0) for b in (2, 3)])
            dd = np.concatenate([Tm["date"].values] * 2)
            rx = ratio_ci(dd, ax, ex)
            ratios.append({"期間": lab, "レース（オッズあり）": len(Tm), "1号艇 実際÷見込み": r1[0], "1号艇 CI下": r1[1], "1号艇 CI上": r1[2],
                           "X 実際÷見込み": rx[0], "X CI下": rx[1], "X CI上": rx[2]})
        # L だけのレースの単勝
        TL = W[per & W["L"]].reset_index(drop=True); dL = TL["date"].values
        rr = {"期間": lab, "L のレース": len(TL)}
        for b in (1, 2, 3):
            pay = np.where(TL["tan_boat"].values == b, TL["tan_pay"].values, 0)
            rr[f"{b}号艇の単勝"] = boot_roi(dL, np.full(len(TL), 100.0), pay)[0] if len(TL) else np.nan
        lonly.append(rr)
    R = pd.DataFrame(rows); RT = pd.DataFrame(ratios); LO = pd.DataFrame(lonly)
    # 12年の参考（オッズなし）
    E1 = W[["date", "jcd", "toban1", "tan_boat"]].rename(columns={"toban1": "t1"}).copy()
    E1["year"] = E1["date"].str[:4].astype(int); E1["w"] = (E1["tan_boat"] == 1).astype(float)
    hrows = []; st_all = {k: 0.0 for k in ("i", "ii", "iii")}; rt_all = {k: 0.0 for k in ("i", "ii", "iii")}
    for Y in range(2016, 2024):
        lowY, nY = low10_hist(E1, [Y - 2, Y - 1])
        T = W[(W["date"].str[:4].astype(int) == Y) & W["toban1"].isin(lowY) & W["anyX"]].reset_index(drop=True)
        st, rt, hit = bets(T); row = {"年": Y, "リストの選手（下位10%／対象）": f"{len(lowY)}／{nY}", "レース": len(T)}
        for kk in ("i", "ii", "iii"):
            row[NAMES[kk]] = rt[kk].sum() / st[kk].sum() if st[kk].sum() else np.nan
            st_all[kk] += st[kk].sum(); rt_all[kk] += rt[kk].sum()
        t1p = np.where(T["tan_boat"].values == 1, T["tan_pay"].values, 0); row["参考：1号艇の単勝"] = t1p.sum() / (100 * len(T)) if len(T) else np.nan
        hrows.append(row)
    H = pd.DataFrame(hrows)
    tot = {kk: rt_all[kk] / st_all[kk] for kk in st_all}
    # 判定
    ev = R[R["期間"].str.startswith("評価")]; tr = R[R["期間"].str.startswith("探索")]
    passed = []
    for kk in ("i", "ii", "iii"):
        e_ = ev[ev["買い方"] == NAMES[kk]].iloc[0]; t_ = tr[tr["買い方"] == NAMES[kk]].iloc[0]
        if e_["回収率"] >= 1.1 and e_["95%CI下"] > 1.0 and e_["点数"] >= 300 and t_["回収率"] >= 1.0 and tot[kk] >= 1.0:
            passed.append(NAMES[kk])
    verdict = "採用候補（前向きの検証へ）：" + "・".join(passed) if passed else "不採用"
    md = ["# IDEA-077 逃げ下手の1号艇のレースで、2・3号艇のスタートの速いA級を買う", "",
          f"L：探索の縮めた比の下位10%（見込み5回以上の {nrac} 人中 {len(low)} 人、k={k60:.0f}、下位10%の境 {q10:.3f}）。リストは出力しない。",
          "X：2・3号艇で A1・A2、かつレース前180日のST平均が1号艇の選手より速い。返還のあるレースは除く。払戻は K の実額。", "",
          f"**判定：{verdict}**", "",
          "## L＋X のレースでの買い方", "", R.to_markdown(index=False, floatfmt=".3f"), "",
          "## 1号艇と X の1着 実際÷見込み（確定3連単オッズから）", "", RT.to_markdown(index=False, floatfmt=".3f"), "",
          "## 参考：L だけのレース（X の有無を問わない）の単勝", "", LO.to_markdown(index=False, floatfmt=".3f"), "",
          "## 12年の参考（オッズなし、リストは前の2年で作る）", "", H.to_markdown(index=False, floatfmt=".3f"), "",
          "通算：" + "、".join(f"{NAMES[kk]} {tot[kk]:.3f}" for kk in tot), ""]
    open(a.out, "w", encoding="utf-8").write("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    main()
