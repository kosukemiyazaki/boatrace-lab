"""IDEA-072 1号艇がB級で相手にA級がいるとき、市場は1号艇を過大評価しているか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea072.py DATADIR HISTDS STOREDIR [--histbk /tmp/claude-0/hist] [--out results/idea072.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ）、STOREDIR：K ファイル（bk/）・確定単勝オッズ（oddstf/、2025-10〜）・live_t1/oddstf
- 区分：1号艇の級別 × 相手（2〜6号艇）のA級の数（0／1／2以上）、B級1号艇は A級の相手の位置（2〜3号艇にいる／4〜6号艇だけ）。位置は艇番（entries.course はレース後の値なので使わない）
- 比：1号艇の実際の1着率 ÷ 市場の見込み（確定3連単オッズから。参考に確定単勝オッズを正規化した見込み）。日単位ブートストラップ
- 買い方（級別だけで選ぶ）：(a) B級1号艇・A級の相手あり → A級の相手の単勝 各100円、(b) B級1号艇・A級の相手なし → 1号艇の単勝・複勝 各100円、
  (c) B級1号艇・A級の相手がちょうど1人（X） → 3連単 X-1-全・X-全-1 の8点 各100円。返還のあるレースは除く
"""
import argparse, glob, gzip, os, re, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL, ratio_ci, boot_roi

FWD = ("20260901", "20261020")
FIRST = np.array([int(c[0]) for c in COMBOS])

def _k(f):
    """K ファイル1日分 -> [(date, jcd, rno, 単勝の艇, 単勝の払戻, 複勝1の払戻)]"""
    date = "20" + os.path.basename(f)[1:7]; out = []
    txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
    for jcd, blk in BP._blocks(txt, "K"):
        rno = None; tb = tp = fu = None
        for line in blk.splitlines():
            l = BP.z2h(line); m = BP.K_RACE.match(l)
            if m:
                if rno is not None and tb is not None:
                    out.append((date, jcd, rno, tb, tp, fu or 0))
                rno = int(m.group(1)); tb = tp = fu = None; continue
            if rno is None:
                continue
            if tb is None and (mm := re.match(r"^\s+単勝\s+(\d)\s+(\d+)", l)):
                tb, tp = int(mm.group(1)), int(mm.group(2)); continue
            if fu is None and (mm := re.match(r"^\s+複勝\s+(\d)\s+(\d+)(?:\s+(\d)\s+(\d+))?", l)):
                fu = int(mm.group(2)) if mm.group(1) == "1" else (int(mm.group(4)) if mm.group(3) == "1" else 0)
        if rno is not None and tb is not None:
            out.append((date, jcd, rno, tb, tp, fu or 0))
    return out

def kpay(dirs, start, end):
    fs = [f for d in dirs for f in sorted(glob.glob(f"{d}/bk/*/k*.txt.gz")) if start <= "20" + os.path.basename(f)[1:7] <= end]
    with Pool(4) as p:
        rows = [x for r in p.map(_k, fs, chunksize=20) for x in r]
    d = pd.DataFrame(rows, columns=KEY + ["tan_boat", "tan_pay", "fuku1"]); d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return d.drop_duplicates(KEY, keep="last")

def race_groups(e):
    """レースごとの区分（1号艇の級別、相手のA級の数・位置、A級の艇番）"""
    e = e.copy(); e["isA"] = e["cls"].isin(["A1", "A2"])
    one = e[e["boat"] == 1][KEY + ["cls"]].rename(columns={"cls": "cls1"})
    op = e[e["boat"] >= 2]
    g = op.groupby(KEY).agg(nA=("isA", "sum")).reset_index()
    a = op[op["isA"]]
    a23 = a[a["boat"] <= 3].groupby(KEY).size().rename("nA23").reset_index()
    aboats = a.groupby(KEY)["boat"].apply(lambda s: tuple(sorted(s))).rename("Aboats").reset_index()
    g = g.merge(a23, on=KEY, how="left").merge(aboats, on=KEY, how="left")
    g["nA23"] = g["nA23"].fillna(0).astype(int); g["Aboats"] = g["Aboats"].apply(lambda x: x if isinstance(x, tuple) else ())
    g = one.merge(g, on=KEY)
    g["nAbin"] = np.where(g["nA"] == 0, "A級0", np.where(g["nA"] == 1, "A級1", "A級2以上"))
    g["B1go"] = g["cls1"].isin(["B1", "B2"])
    g["区分"] = g["cls1"] + "・相手" + g["nAbin"]
    g["位置"] = np.where(g["nA"] == 0, "相手A級なし", np.where(g["nA23"] > 0, "A級が2〜3号艇にいる", "A級は4〜6号艇だけ"))
    g["B級×位置"] = np.where(g["B1go"], "B級1号艇・" + g["位置"], "nan")
    return g

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("store")
    ap.add_argument("--histbk", default="/tmp/claude-0/hist"); ap.add_argument("--out", default="results/idea072.md"); a = ap.parse_args()
    cols = KEY + ["boat", "cls", "pos_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in (a.histds, a.datadir)], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last")
    bad = e[e["pos_raw"].astype(str).str.match(r"^[KFL]")][KEY].drop_duplicates().assign(bad=1)
    six = e.groupby(KEY).size().rename("nb").reset_index(); six = six[six["nb"] == 6][KEY]
    G = race_groups(e).merge(six, on=KEY).merge(bad, on=KEY, how="left"); G = G[G["bad"].isna()].drop(columns="bad")
    r = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["tri_combo", "tri_pay"])) for p in (a.histds, a.datadir)])
    r = r.dropna(subset=["tri_combo"]).drop_duplicates(KEY, keep="last")
    G = G.merge(r, on=KEY)
    print("K payouts...", flush=True)
    K = kpay([a.histbk, a.store], "20140101", FWD[1]); G = G.merge(K, on=KEY)
    G["win1"] = (G["tri_combo"].str[0] == "1").astype(float)
    G["per"] = np.select([(G["date"] >= EXP[0]) & (G["date"] <= EXP[1]), (G["date"] >= EVL[0]) & (G["date"] <= EVL[1]),
                          (G["date"] >= FWD[0]) & (G["date"] <= FWD[1])], ["探索", "評価", "前向き参考"], "12年のみ")
    G = G[G["date"] <= FWD[1]].reset_index(drop=True)
    print("races", len(G), flush=True)
    # --- 市場の見込み（確定3連単）
    o = norm(pd.read_parquet(f"{a.datadir}/odds.parquet")); O = o[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    P = np.where(v, 1 / np.where(v, O, 1), 0); s = P.sum(1, keepdims=True); P = P / np.where(s > 0, s, 1)
    o = o[KEY].assign(p1=P[:, FIRST == 1].sum(1), nv=v.sum(1)); o = o[o["nv"] >= 60].drop(columns="nv")
    G = G.merge(o, on=KEY, how="left")
    # 確定単勝オッズ（2025-10〜）
    tf = pd.concat([pd.read_csv(f, dtype={"jcd": str, "date": str}) for f in sorted(glob.glob(f"{a.store}/oddstf/*.csv.gz"))])
    T = tf[[f"tan{i}" for i in range(1, 7)]].to_numpy(float); tv = np.isfinite(T) & (T > 0)
    Q = np.where(tv, 1 / np.where(tv, T, 1), 0); tf["q1"] = np.where(tv.all(1), Q[:, 0] / Q.sum(1), np.nan)
    tf["jcd"] = tf["jcd"].str.zfill(2); G = G.merge(tf[KEY + ["q1"]].drop_duplicates(KEY), on=KEY, how="left")
    # --- 比の表
    rows = []
    for col in ("区分", "B級×位置"):
        for lab, g in G[G["per"].isin(["探索", "評価", "前向き参考"])].groupby(col):
            if lab == "nan":
                continue
            row = {"区分": lab}
            for pp in ("探索", "評価"):
                gg = g[(g["per"] == pp) & g["p1"].notna()]
                rr, lo, hi = ratio_ci(gg["date"].values, gg["win1"].values, gg["p1"].values) if len(gg) >= 100 else (np.nan,) * 3
                row[f"{pp} レース"] = len(gg); row[f"{pp} 1着率"] = gg["win1"].mean(); row[f"{pp} 見込み"] = gg["p1"].mean()
                row[f"{pp} 比"] = rr; row[f"{pp} CI"] = f"{lo:.2f}〜{hi:.2f}" if np.isfinite(lo) else ""
                gq = g[(g["per"] == pp) & g["q1"].notna()]
                rq = ratio_ci(gq["date"].values, gq["win1"].values, gq["q1"].values) if len(gq) >= 100 else (np.nan,) * 3
                row[f"{pp} 比（単勝）"] = rq[0]; row[f"{pp} 単勝 n"] = len(gq)
                ge = g[g["per"] == pp]
                t = boot_roi(ge["date"].values, np.full(len(ge), 100.0), np.where(ge["tan_boat"] == 1, ge["tan_pay"], 0).astype(float))
                f = boot_roi(ge["date"].values, np.full(len(ge), 100.0), ge["fuku1"].values.astype(float))
                row[f"{pp} 単勝1"] = t[0]; row[f"{pp} 単勝1 CI"] = f"{t[1]:.2f}〜{t[2]:.2f}" if np.isfinite(t[1]) else ""
                row[f"{pp} 複勝1"] = f[0]
            rows.append(row)
    R = pd.DataFrame(rows)
    # --- 買い方
    def bets(g):
        out = {}
        # (a)
        ga = g[g["B1go"] & (g["nA"] >= 1)]
        st, rt, dd = [], [], []
        for d, ab, tb, tp in zip(ga["date"], ga["Aboats"], ga["tan_boat"], ga["tan_pay"]):
            for b in ab:
                st.append(100.0); rt.append(float(tp) if tb == b else 0.0); dd.append(d)
        out["(a) A級の相手の単勝"] = (np.array(dd), np.array(st), np.array(rt))
        gb = g[g["B1go"] & (g["nA"] == 0)]
        out["(b) 1号艇の単勝（相手A級なし）"] = (gb["date"].values, np.full(len(gb), 100.0), np.where(gb["tan_boat"] == 1, gb["tan_pay"], 0).astype(float))
        out["(b) 1号艇の複勝（相手A級なし）"] = (gb["date"].values, np.full(len(gb), 100.0), gb["fuku1"].values.astype(float))
        gc = g[g["B1go"] & (g["nA"] == 1)]
        hit = [((c[0] == str(ab[0])) and (c[2] == "1" or c[4] == "1")) for c, ab in zip(gc["tri_combo"], gc["Aboats"])]
        out["(c) 3連単 X-1-全・X-全-1"] = (gc["date"].values, np.full(len(gc), 800.0), np.where(hit, gc["tri_pay"], 0).astype(float))
        # 参考：A級の相手の位置別の (a)
        for pos in ("A級が2〜3号艇にいる", "A級は4〜6号艇だけ"):
            gp = ga[ga["位置"] == pos]; st, rt, dd = [], [], []
            for d, ab, tb, tp in zip(gp["date"], gp["Aboats"], gp["tan_boat"], gp["tan_pay"]):
                for b in ab:
                    st.append(100.0); rt.append(float(tp) if tb == b else 0.0); dd.append(d)
            out[f"参考 (a) {pos}"] = (np.array(dd), np.array(st), np.array(rt))
        return out
    brows = []
    G["year"] = G["date"].str[:4]
    for pp in ("探索", "評価", "前向き参考", "12年 2014〜2026-08"):
        g = G[G["per"] == pp] if pp != "12年 2014〜2026-08" else G[(G["date"] >= "20140101") & (G["date"] <= EVL[1])]
        for nm, (d, st, rt) in bets(g).items():
            roi, lo, hi = boot_roi(d, st, rt) if len(st) else (np.nan,) * 3
            brows.append({"買い方": nm, "期間": pp, "点数": int(st.sum() / 100) if len(st) else 0, "回収率": roi, "CI下": lo, "CI上": hi})
    B = pd.DataFrame(brows)
    # 12年の年別
    yrows = []
    for y, g in G[(G["date"] >= "20140101") & (G["date"] <= EVL[1])].groupby("year"):
        row = {"年": y}
        for nm, (d, st, rt) in bets(g).items():
            if nm.startswith("参考"):
                continue
            row[nm] = rt.sum() / st.sum() if st.sum() else np.nan
        gb = g[g["B1go"]]
        for lab, gg in gb.groupby("nAbin"):
            row[f"B級1号艇 {lab} 1着率"] = gg["win1"].mean()
        yrows.append(row)
    Y = pd.DataFrame(yrows)
    # 判定
    ev = B[B["期間"] == "評価"]; ex = B[B["期間"] == "探索"]; hy = B[B["期間"] == "12年 2014〜2026-08"]
    win = []
    for _, b in ev[~ev["買い方"].str.startswith("参考")].iterrows():
        if b["点数"] >= 300 and b["CI下"] > 1.0:
            e1 = ex[ex["買い方"] == b["買い方"]]["回収率"].iloc[0]; h1 = hy[hy["買い方"] == b["買い方"]]["回収率"].iloc[0]
            if e1 > 1.0 and h1 > 1.0:
                win.append(b["買い方"])
    judge = ("採用候補（前向きの検証へ）：" + "、".join(win)) if win else "不採用"
    # 前向き：締切2分前の単勝オッズでの比（live_t1、2026-10-11〜）
    lt = [pd.read_csv(f, dtype={"jcd": str, "date": str}) for f in sorted(glob.glob(f"{a.store}/live_t1/oddstf/*.csv.gz"))]
    lrow = ""
    if lt:
        L = pd.concat(lt).sort_values("fetched_at").drop_duplicates(KEY, keep="last"); L["jcd"] = L["jcd"].str.zfill(2)
        T = L[[f"tan{i}" for i in range(1, 7)]].to_numpy(float); tv = np.isfinite(T) & (T > 0)
        Q = np.where(tv, 1 / np.where(tv, T, 1), 0); L["q1t1"] = np.where(tv.all(1), Q[:, 0] / Q.sum(1), np.nan)
        gl = G.merge(L[KEY + ["q1t1"]], on=KEY).dropna(subset=["q1t1"])
        parts = []
        for lab, gg in gl[gl["B1go"]].groupby("位置"):
            parts.append(f"{lab}：{len(gg)}レース、1着率 {gg['win1'].mean():.3f}、見込み {gg['q1t1'].mean():.3f}、比 {gg['win1'].sum() / gg['q1t1'].sum():.2f}")
        lrow = "；".join(parts)
    show = lambda df: df.to_markdown(index=False, floatfmt=".3f")
    keyc = ["区分", "探索 レース", "探索 1着率", "探索 見込み", "探索 比", "探索 CI", "評価 レース", "評価 1着率", "評価 見込み", "評価 比", "評価 CI", "評価 比（単勝）", "評価 単勝 n"]
    roic = ["区分", "探索 単勝1", "探索 単勝1 CI", "探索 複勝1", "評価 単勝1", "評価 単勝1 CI", "評価 複勝1"]
    md = [f"# IDEA-072 1号艇がB級で相手にA級がいるとき（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}／12年 2014〜）", "",
          f"**判定：{judge}**", "",
          "- 比 = 1号艇の実際の1着率 ÷ 市場の見込み（確定3連単オッズから）。1 より下なら市場が1号艇を買いかぶっている。CI は日単位ブートストラップ95%。",
          "- 比（単勝）は確定単勝オッズを正規化した見込みで割った比（2025-10〜しかないので評価期間だけ）。",
          "- 単勝1・複勝1 は1号艇の単勝・複勝を各100円の回収率（K の実額）。返還のあるレースは除く。位置は艇番（進入はレース後の値なので使わない）。", "",
          "## 1号艇の1着：実際 ÷ 市場の見込み", "", show(R[keyc]), "",
          "## 1号艇の単勝・複勝の回収率", "", show(R[roic]), "",
          "## 買い方（級別だけで選ぶ。オッズで絞らないので確定オッズの見かけは無い）", "", show(B), "",
          "## 12年の年別（買い方の回収率と、B級1号艇の1着率）", "", show(Y), "",
          f"## 前向きの参考：締切2分前の単勝オッズでの比（live_t1、B級1号艇）", "", lrow or "データなし", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
