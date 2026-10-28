"""IDEA-076 利用者の本線「イン逃げが決まりやすいレースを厳選し、3連単を2〜6点に絞る」を1回だけ確かめる（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea076.py DATADIR STOREDIR [--hist /tmp/claude-0/ds_hist] [--histbk /tmp/claude-0/hist] [--out results/idea076.md]
- S：(1) 展示スタートで1号艇が2・3号艇の早い方+0.02秒以内（展示の F は負のタイミング）・展示で1号艇が1コース
       (2) 1号艇のモーター2連率がレース内2位以内 (3) 当地勝率 ≥ 全国勝率−0.3（当地0は不可）かつ A級か全国勝率≥6.0 (4) 1号艇の展示タイムがレース内3位以内
- S+壁：2号艇がA級で、平均ST（レース前180日の実際のSTの平均、10走以上）≤ 1号艇の平均ST+0.02
- S'：S から (1) を外したもの（2014〜）
- 買い目：2〜6号艇を展示タイム順に a,b,c。2点 1-a-b,1-b-a／4点 +1-a-c,1-b-c／6点 {a,b,c} の並び6通り。各100円、払戻は実額、F・L・欠場の艇を含む組は返還
"""
import argparse, glob, gzip, os, re, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as P
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY
from exp_idea053 import boot

REF = {"F", "L0", "L1", "K0", "K1"}

def _kfile(f):
    """K 1日分 -> [(date,jcd,rno,単勝1の払戻(1号艇が外れは0), 2連単の組, 2連単の払戻)]"""
    date = "20" + os.path.basename(f)[1:7]; out = []
    txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
    for jcd, blk in P._blocks(txt, "K"):
        rno = None; tan = None; t2 = None; p2 = None
        def flush():
            if rno is not None and tan is not None:
                out.append((date, jcd, rno, tan, t2, p2))
        for line in blk.splitlines():
            l = P.z2h(line); m = P.K_RACE.match(l)
            if m:
                flush(); rno = int(m.group(1)); tan = None; t2 = None; p2 = None; continue
            if rno is None:
                continue
            if (mm := re.match(r"^\s+単勝\s+(\d)\s+(\d+)", l)):
                tan = int(mm.group(2)) if mm.group(1) == "1" else 0; continue
            if t2 is None and (mm := re.match(r"^\s+2連単\s+(\d)-(\d)\s+(\d+)", l)):
                t2 = f"{mm.group(1)}-{mm.group(2)}"; p2 = int(mm.group(3))
        flush()
    return out

def kpays(dirs):
    fs = [f for d in dirs for f in sorted(glob.glob(f"{d}/bk/*/k*.txt.gz"))]
    with Pool(4) as p:
        rows = [x for r in p.map(_kfile, fs, chunksize=20) for x in r]
    d = pd.DataFrame(rows, columns=KEY + ["tan1", "t2", "p2"]); d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return d.drop_duplicates(KEY, keep="last")

def avg_st(e):
    """選手ごと、レース前180日（当日を含まない）の実際のSTの平均と走数"""
    x = e[(e["st"] > 0) & ~e["pos_raw"].astype(str).isin(REF)][["toban", "date", "st"]].copy()
    x["dt"] = pd.to_datetime(x["date"]); g = x.groupby(["toban", "dt"])["st"].agg(["sum", "count"]).reset_index().sort_values(["toban", "dt"])
    out = []
    for t, h in g.groupby("toban", sort=False):
        h = h.set_index("dt")
        s = h["sum"].rolling("180D", closed="left").sum(); c = h["count"].rolling("180D", closed="left").sum()
        out.append(pd.DataFrame({"toban": t, "dt": h.index, "ast": (s / c).values, "ncnt": c.values}))
    a = pd.concat(out); a["date"] = a["dt"].dt.strftime("%Y%m%d"); return a.drop(columns="dt")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--hist", default="/tmp/claude-0/ds_hist"); ap.add_argument("--histbk", default="/tmp/claude-0/hist")
    ap.add_argument("--out", default="results/idea076.md"); a = ap.parse_args()
    cols = KEY + ["boat", "toban", "cls", "nat_win", "loc_win", "motor_2r", "exh_time", "st", "pos_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in (a.hist, a.datadir)], ignore_index=True).drop_duplicates(KEY + ["boat"], keep="last")
    r = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["tri_combo", "tri_pay"])) for p in (a.hist, a.datadir)], ignore_index=True).drop_duplicates(KEY, keep="last").dropna(subset=["tri_combo"])
    print("avg st...", flush=True)
    ast = avg_st(e); e = e.merge(ast, on=["toban", "date"], how="left")
    e["exh_time"] = pd.to_numeric(e["exh_time"], errors="coerce").where(lambda s: s > 0)
    W = e.pivot_table(index=KEY, columns="boat", values=["exh_time", "motor_2r", "nat_win", "loc_win", "ast", "ncnt"], aggfunc="first")
    W.columns = [f"{c}{b}" for c, b in W.columns]; W = W.reset_index()
    C = e.pivot_table(index=KEY, columns="boat", values=["cls", "pos_raw"], aggfunc="first"); C.columns = [f"{c}{b}" for c, b in C.columns]; C = C.reset_index()
    m = r.merge(W, on=KEY).merge(C, on=KEY)
    ex = m[[f"exh_time{b}" for b in range(1, 7)]].to_numpy(float)
    m = m[np.isfinite(ex).all(1)].reset_index(drop=True); ex = m[[f"exh_time{b}" for b in range(1, 7)]].to_numpy(float); n = len(m)
    # (2)(3)(4)
    mot = m[[f"motor_2r{b}" for b in range(1, 7)]].to_numpy(float)
    c2 = (mot > mot[:, [0]]).sum(1) <= 1
    nat1, loc1 = m["nat_win1"].astype(float).values, m["loc_win1"].astype(float).values
    c3 = (loc1 > 0) & (loc1 >= nat1 - 0.3) & (m["cls1"].isin(["A1", "A2"]).values | (nat1 >= 6.0))
    c4 = (ex < ex[:, [0]]).sum(1) <= 2
    Sp = c2 & c3 & c4
    wall = m["cls2"].isin(["A1", "A2"]).values & (m["ncnt1"].astype(float).values >= 10) & (m["ncnt2"].astype(float).values >= 10) & \
        (m["ast2"].astype(float).values <= m["ast1"].astype(float).values + 0.02)
    # (1) 展示スタート
    bi = norm(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet", columns=KEY + ["ex_course1", "ex_st1", "ex_f1", "ex_st2", "ex_f2", "ex_st3", "ex_f3"])).drop_duplicates(KEY, keep="last")
    for b in (1, 2, 3):
        bi[f"t{b}"] = np.where(bi[f"ex_f{b}"].fillna(0) > 0, -bi[f"ex_st{b}"], bi[f"ex_st{b}"])
    bi["c1ok"] = (bi["ex_course1"] == 1) & bi[["t1", "t2", "t3"]].notna().all(1) & (bi["t1"] <= np.minimum(bi["t2"], bi["t3"]) + 0.02 + 1e-9)
    m = m.merge(bi[KEY + ["c1ok"]], on=KEY, how="left"); hasbi = m["c1ok"].notna().values; c1 = m["c1ok"].fillna(False).astype(bool).values
    S = Sp & c1 & hasbi
    # 買い目
    order = np.argsort(ex[:, 1:] + np.arange(5) * 1e-6, axis=1) + 2  # 2〜6号艇の展示順（同タイムは艇番の小さい順）
    A, B, Cc = order[:, 0], order[:, 1], order[:, 2]
    sets = {2: [(A, B), (B, A)], 4: [(A, B), (A, Cc), (B, A), (B, Cc)], 6: [(A, B), (A, Cc), (B, A), (B, Cc), (Cc, A), (Cc, B)]}
    win = m["tri_combo"].astype(str).values; tp = pd.to_numeric(m["tri_pay"], errors="coerce").fillna(0).values
    pr = np.stack([m[f"pos_raw{b}"].astype(str).isin(REF).values for b in range(1, 7)], 1)
    def ret(k):
        out = np.zeros(n)
        for x, y in sets[k]:
            combo = np.char.add(np.char.add("1-", x.astype(str)), np.char.add("-", y.astype(str)))
            refund = pr[:, 0] | pr[np.arange(n), x - 1] | pr[np.arange(n), y - 1]
            out += np.where(refund, 100.0, np.where(combo == win, tp, 0.0))
        return out
    R = {k: ret(k) for k in (2, 4, 6)}
    hit = {k: sum((np.char.add(np.char.add("1-", x.astype(str)), np.char.add("-", y.astype(str))) == win) for x, y in sets[k]) > 0 for k in (2, 4, 6)}
    print("K pays...", flush=True)
    kp = kpays([a.histbk, a.store]); m2 = m[KEY].assign(i=np.arange(n)).merge(kp, on=KEY, how="left")
    m2 = m2.sort_values("i"); tan = m2["tan1"].astype(float).values; t2 = m2["t2"].astype(str).values; p2 = m2["p2"].astype(float).values
    ok2 = np.isfinite(tan)
    tanR = np.where(pr[:, 0], 100.0, np.nan_to_num(tan))
    r2 = np.zeros(n)
    for x in (A, B):
        combo = np.char.add("1-", x.astype(str)); refund = pr[:, 0] | pr[np.arange(n), x - 1]
        r2 += np.where(refund, 100.0, np.where(combo == t2, np.nan_to_num(p2), 0.0))
    d = m["date"].values
    # 市場の見込み（確定3連単）
    o = norm(pd.read_parquet(f"{a.datadir}/odds.parquet")); mo = m[KEY].assign(i=np.arange(n)).merge(o, on=KEY)
    O = mo[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); Pm = np.where(v, 1 / np.where(v, O, 1), 0); Pm /= np.where(Pm.sum(1, keepdims=True) > 0, Pm.sum(1, keepdims=True), 1)
    p1 = np.full(n, np.nan); p1[mo["i"].values] = Pm[:, [i for i, c in enumerate(COMBOS) if c[0] == "1"]].sum(1)
    w1 = np.array([c[0] == "1" for c in win])
    PER = {"S": [("探索 2025-10〜12", "20251001", "20251231"), ("評価 2026-01〜08", "20260101", "20260831"), ("追加 2026-09〜10-27", "20260901", "20261027")],
           "S'": [("長い参考 2014-01〜2023-09", "20140101", "20230930"), ("探索 2023-11〜2025-12", "20231101", "20251231"), ("評価 2026-01〜08", "20260101", "20260831")]}
    rows = []
    def add(sel_name, per, msk):
        nd = len(np.unique(d[msk])); row = {"絞り込み": sel_name, "期間": per, "レース": int(msk.sum()), "1日あたり": round(msk.sum() / nd, 2) if nd else 0}
        for k in (2, 4, 6):
            if msk.sum():
                roi, lo, hi = boot(d[msk], np.full(msk.sum(), 100.0 * k), R[k][msk])
                row[f"{k}点"] = f"{roi:.1%}（{lo:.0%}〜{hi:.0%}）"; row[f"{k}点 的中"] = f"{hit[k][msk].mean():.1%}"
                hh = msk & hit[k]; row[f"{k}点 当たりの平均払戻"] = int(R[k][hh].mean()) if hh.sum() else 0
                row[f"_{k}"] = roi; row[f"_{k}lo"] = lo
        mk = msk & ok2
        if mk.sum():
            roi, lo, hi = boot(d[mk], np.full(mk.sum(), 100.0), tanR[mk]); row["単勝1"] = f"{roi:.1%}（{lo:.0%}〜{hi:.0%}）"
            roi, lo, hi = boot(d[mk], np.full(mk.sum(), 200.0), r2[mk]); row["2連単 1-a・1-b"] = f"{roi:.1%}（{lo:.0%}〜{hi:.0%}）"
        pm = msk & np.isfinite(p1)
        row["1号艇1着率／市場の見込み"] = f"{w1[pm].mean():.3f}／{p1[pm].mean():.3f}（比 {w1[pm].mean() / p1[pm].mean():.3f}）" if pm.sum() >= 30 else "—"
        rows.append(row)
    for per, s, t in PER["S"]:
        msk = (d >= s) & (d <= t) & hasbi
        add("全レース（基準・展示スタートあり）", per, msk); add("S", per, msk & S); add("S+壁", per, msk & S & wall)
    for per, s, t in PER["S'"]:
        msk = (d >= s) & (d <= t)
        add("全レース（基準）", per, msk); add("S'", per, msk & Sp); add("S'+壁", per, msk & Sp & wall)
    T = pd.DataFrame(rows)
    def J(nm):
        ev = T[(T["絞り込み"] == nm) & (T["期間"] == "評価 2026-01〜08")].iloc[0]; xp = T[(T["絞り込み"] == nm) & (T["期間"] == "探索 2025-10〜12")].iloc[0]
        lg = T[(T["絞り込み"] == ("S'" if nm == "S" else "S'+壁")) & (T["期間"] == "長い参考 2014-01〜2023-09")].iloc[0]
        res = []
        for k in (2, 4, 6):
            ok = ev["レース"] >= 100 and ev[f"_{k}"] >= 1.10 and ev[f"_{k}lo"] > 1.0 and xp[f"_{k}"] >= 1.0 and lg[f"_{k}"] >= 1.0
            res.append(f"{k}点：{'採用候補' if ok else '不採用'}")
        return "、".join(res)
    verdict = {nm: J(nm) for nm in ("S", "S+壁")}
    show = [c for c in T.columns if not c.startswith("_")]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# IDEA-076 イン逃げ厳選＋3連単2〜6点（利用者の考えを1回だけ確かめる）\n\n")
        f.write("登録どおりの1回だけの実行。払戻は実額（3連単は結果ファイル、単勝・2連単は K）、各組100円、F・L・欠場の艇を含む組は返還。括弧は日単位ブートストラップ95%CI。\n\n")
        f.write("- 展示スタートは 2025-10 以降しかないため、S（展示スタートを使う）はその期間、S'（展示スタート抜き）は 2014 年から。\n")
        f.write("- 平均STは過去分の番組表に無いため、レース前180日の実際のSTの平均で代用（10走以上）。\n\n")
        f.write("## 結果\n\n" + T[show].to_markdown(index=False) + "\n\n")
        f.write("## 判定（登録どおり：評価で回収率110%以上・CI下限>100%・100レース以上、探索≥100%、長い参考 S'≥100%）\n\n")
        for nm, v_ in verdict.items():
            f.write(f"- {nm}：{v_}\n")
    print(open(a.out, encoding="utf-8").read())

if __name__ == "__main__":
    main()
