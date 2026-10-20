"""IDEA-052 売上の小さいレースなら、締切2分前の割安は確定まで残るか（docs/ideas.md の登録どおり）
usage: python scripts/idea052.py STOREDIR [--start 20261011] [--out results/idea052.md]
- 売上の大きさ：t1 時点の3連単の票数（live_t1/bc_votes の votes_total）。レースを5分位に分ける（1 = 一番小さい）
- t1 の3連単 → P → π = P^1.06（正規化）。単勝・複勝（2着以内）・2連単・2連複・拡連複（3着以内）・3連複 の確率を π から出す
- 確率 × t1 オッズ（複勝・拡連複は下限）> 1.1 の組を各100円、払戻は K ファイルの実額
- 判定：一番小さい5分位（券種まとめて）で 300点以上・回収率110%以上・日単位95%CI下限 > 100% → 採用候補。300点未満は途中経過、1,000点で最終判定
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import boatlib.parse as P
from boatlib.parse import COMBOS, COMBOS_3F, COMBOS_2T, COMBOS_2F
from idea039_stage2 import load

KEY = ["date", "jcd", "rno"]; RNG = np.random.default_rng(0); BETA = 1.06; THR = 1.1
LABELS = {"tan": "単勝", "fuku": "複勝", "2t": "2連単", "2f": "2連複", "wide": "拡連複", "3f": "3連複"}

def k_pay(store, start):
    """K ファイルの 単勝・複勝・2連単・2連複・拡連複・3連複 の払戻（100円あたり）。(kind, combo) -> pay"""
    rows = []
    for f in sorted(glob.glob(f"{store}/bk/*/k*.txt.gz")):
        date = "20" + os.path.basename(f)[1:7]
        if date < start:
            continue
        txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
        for jcd, blk in P._blocks(txt, "K"):
            rno = None; inw = 0
            for line in blk.splitlines():
                l = P.z2h(line); m = P.K_RACE.match(l)
                if m:
                    rno = int(m.group(1)); inw = 0; continue
                if rno is None:
                    continue
                add = lambda k, c, p: rows.append((date, jcd, rno, k, c, int(p)))
                if (mm := re.match(r"^\s+単勝\s+(\d)\s+(\d+)", l)):
                    add("tan", mm.group(1), mm.group(2)); inw = 0; continue
                if (mm := re.match(r"^\s+複勝\s+(\d)\s+(\d+)(?:\s+(\d)\s+(\d+))?", l)):
                    add("fuku", mm.group(1), mm.group(2))
                    if mm.group(3):
                        add("fuku", mm.group(3), mm.group(4))
                    inw = 0; continue
                if (mm := re.match(r"^\s+2連単\s+(\d)-(\d)\s+(\d+)", l)):
                    add("2t", f"{mm.group(1)}-{mm.group(2)}", mm.group(3)); inw = 0; continue
                if (mm := re.match(r"^\s+2連複\s+(\d)-(\d)\s+(\d+)", l)):
                    add("2f", "=".join(sorted(mm.group(1, 2))), mm.group(3)); inw = 0; continue
                if (mm := re.match(r"^\s+拡連複\s+(\d)-(\d)\s+(\d+)", l)):
                    add("wide", "=".join(sorted(mm.group(1, 2))), mm.group(3)); inw = 1; continue
                if inw and inw < 3 and (mm := re.match(r"^\s+(\d)-(\d)\s+(\d+)\s+人気", l)):
                    add("wide", "=".join(sorted(mm.group(1, 2))), mm.group(3)); inw += 1; continue
                if (mm := re.match(r"^\s+3連複\s+(\d)-(\d)-(\d)\s+(\d+)", l)):
                    add("3f", "=".join(sorted(mm.group(1, 2, 3))), mm.group(4)); inw = 0; continue
                inw = 0
    d = pd.DataFrame(rows, columns=KEY + ["kind", "combo", "pay"]); d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return d.drop_duplicates(KEY + ["kind", "combo"], keep="last")

def boot(g):
    if len(g) < 2:
        return np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("store"); ap.add_argument("--start", default="20261011"); ap.add_argument("--out", default="results/idea052.md")
    a = ap.parse_args()
    o3 = load(a.store, "live_t1", "odds3t", a.start)
    pay = k_pay(a.store, a.start); done = set(map(tuple, pay[KEY].drop_duplicates().values))
    paymap = {(r.date, r.jcd, r.rno, r.kind, r.combo): r.pay for r in pay.itertuples()}
    vt = load(a.store, "live_t1", "bc_votes", a.start)[KEY + ["votes_total"]]
    o3 = o3[[k in done for k in map(tuple, o3[KEY].values)]].merge(vt, on=KEY).reset_index(drop=True)
    O = o3[COMBOS].apply(pd.to_numeric, errors="coerce").to_numpy(float); v = np.isfinite(O) & (O > 0)
    ok = (v.sum(1) >= 40) & np.isfinite(pd.to_numeric(o3["votes_total"], errors="coerce").values)
    o3, O, v = o3[ok].reset_index(drop=True), O[ok], v[ok]
    Pm = np.where(v, 1 / np.where(v, O, 1), 0); Pm /= Pm.sum(1, keepdims=True)
    pi = Pm ** BETA; pi /= pi.sum(1, keepdims=True)
    vol = pd.to_numeric(o3["votes_total"]).values; edges = np.quantile(vol, [.2, .4, .6, .8]); o3["q5"] = np.digitize(vol, edges) + 1
    f1 = np.array([int(c[0]) for c in COMBOS]); f2 = np.array([int(c[2]) for c in COMBOS]); f3 = np.array([int(c[4]) for c in COMBOS])
    top2 = [frozenset((x, y)) for x, y in zip(f1, f2)]; top3 = [frozenset((x, y, z)) for x, y, z in zip(f1, f2, f3)]
    idx = {"tan": {str(b): f1 == b for b in range(1, 7)}, "fuku": {str(b): (f1 == b) | (f2 == b) for b in range(1, 7)},
           "2t": {c: (f1 == int(c[0])) & (f2 == int(c[2])) for c in COMBOS_2T},
           "2f": {c: np.array([s == frozenset(map(int, c.split("="))) for s in top2]) for c in COMBOS_2F},
           "wide": {c: np.array([set(map(int, c.split("="))) <= s for s in top3]) for c in COMBOS_2F},
           "3f": {c: np.array([s == frozenset(map(int, c.split("="))) for s in top3]) for c in COMBOS_3F}}
    pages = {"tan": ("oddstf", lambda c: f"tan{c}"), "fuku": ("oddstf", lambda c: f"fuku_lo{c}"), "2t": ("odds2tf", lambda c: c),
             "2f": ("odds2tf", lambda c: c), "wide": ("oddsk", lambda c: f"wide_lo{c}"), "3f": ("odds3f", lambda c: c)}
    cache, rows = {}, []
    for kind, (page, col) in pages.items():
        if page not in cache:
            cache[page] = load(a.store, "live_t1", page, a.start)
        m = o3[KEY + ["q5"]].reset_index().merge(cache[page], on=KEY)
        if not len(m):
            continue
        for c, mask in idx[kind].items():
            cc = col(c)
            if cc not in m:
                continue
            q = pi[m["index"].values][:, mask].sum(1); od = pd.to_numeric(m[cc], errors="coerce").values
            p = np.array([paymap.get((d, j, r, kind, c), 0) for d, j, r in zip(m["date"], m["jcd"], m["rno"])], float)
            rows.append(pd.DataFrame({"kind": kind, "date": m["date"], "q5": m["q5"], "q": q, "od": od, "pay": p}))
    d = pd.concat(rows, ignore_index=True); d = d[np.isfinite(d["od"]) & (d["od"] > 0) & (d["q"] > 0)]
    b = d[d["q"] * d["od"] > THR].copy(); b["hit"] = b["pay"] > 0
    out = []
    def summ(x, label, q5):
        g = x.assign(st=100.0, rt=x["pay"]).groupby("date")[["st", "rt"]].sum(); n = len(x)
        roi = g["rt"].sum() / g["st"].sum() if n else np.nan; lo, hi = boot(g); h = x[x["hit"]]
        return {"売上5分位": q5, "券種": label, "点数": n, "的中": int(x["hit"].sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi,
                "確定払戻÷t1オッズ（的中）": (h["pay"] / 100 / h["od"]).mean() if len(h) else np.nan}
    for q5 in range(1, 6):
        x = b[b["q5"] == q5]
        out.append(summ(x, "全券種", q5))
        for kind, lab in LABELS.items():
            out.append(summ(x[x["kind"] == kind], lab, q5))
    R = pd.DataFrame(out)
    s1 = R[(R["売上5分位"] == 1) & (R["券種"] == "全券種")].iloc[0]
    if s1["点数"] < 300:
        verdict = f"途中経過（一番小さい5分位 {int(s1['点数'])}点、判定は300点）"
    elif s1["回収率"] >= 1.1 and s1["95%CI下"] > 1.0:
        verdict = "採用候補（前向きに続けて確認、1,000点で最終判定）"
    else:
        verdict = f"不採用（一番小さい5分位：{int(s1['点数'])}点・回収率 {s1['回収率']:.3f}、CI下 {s1['95%CI下']:.3f}）"
    vr = pd.DataFrame({"売上5分位": range(1, 6), "3連単の票数（t1）の範囲": [f"〜{int(edges[0]):,}"] + [f"{int(edges[i]):,}〜{int(edges[i+1]):,}" for i in range(3)] + [f"{int(edges[3]):,}〜"],
                       "レース": [int((o3["q5"] == k).sum()) for k in range(1, 6)]})
    md = [f"# IDEA-052 売上の小さいレースで締切2分前の割安は残るか（t1、{a.start}〜、{len(o3)}レース・{o3['date'].nunique()}日、払戻は K の実額）", "",
          f"**判定：{verdict}**", "", vr.to_markdown(index=False), "", R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
