"""IDEA-056 段階2 (A) 12年分の確認：偏りを重ねて1号艇が強いレースの 1号艇の単勝・複勝（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea056_stage2.py DATADIR STOREDIR [--hist /tmp/claude-0/ds_hist] [--histbk /tmp/claude-0/hist] [--out results/idea056_stage2.md]
- 重み：IDEA-053 と同じ21区分・同じ学習（探索 2023-11〜2025-12、市場の1号艇の確率を offset にした L2 ロジスティック）。
  学習した重みを models/idea056_weights.json に保存（前向きの検証 scripts/idea056_fwd.py はこれを使う。以後変更しない）
- 12年分（2014〜2026-08、オッズなし）：区分の重みの合計 Σw（市場の項なし、グレードは過去分に無いので使わない）の上位 4.5%／1.5% のレース。
  しきい値は 2014〜2019 で決めて固定。年ごとに「1号艇の単勝・複勝を各100円」の回収率（払戻は K ファイルの実額）と、全レースの単勝1・複勝1。
- 返還のあるレース（欠場・フライング・出遅れ）と、K に単勝の払戻が無いレースは除く
"""
import argparse, glob, gzip, json, os, re, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as P
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY, TR, EV, FIRST, boat_feats, segments
from exp_idea053 import SEL, LAM, design, fit

COLS = KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]
XP = ("20140101", "20191231")

def _tanfuku(f):
    """K ファイル1日分 -> [(date, jcd, rno, 単勝1の払戻, 複勝1の払戻)]（1号艇が外れなら 0）。単勝の行が無いレースは入れない"""
    date = "20" + os.path.basename(f)[1:7]; out = []
    txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
    for jcd, blk in P._blocks(txt, "K"):
        rno = None; tan = fuku = None
        for line in blk.splitlines():
            l = P.z2h(line); m = P.K_RACE.match(l)
            if m:
                if rno is not None and tan is not None:
                    out.append((date, jcd, rno, tan, fuku or 0))
                rno = int(m.group(1)); tan = fuku = None; continue
            if rno is None:
                continue
            if (mm := re.match(r"^\s+単勝\s+(\d)\s+(\d+)", l)):
                tan = int(mm.group(2)) if mm.group(1) == "1" else 0; continue
            if (mm := re.match(r"^\s+複勝\s+(\d)\s+(\d+)(?:\s+(\d)\s+(\d+))?", l)):
                fuku = int(mm.group(2)) if mm.group(1) == "1" else (int(mm.group(4)) if mm.group(3) == "1" else 0)
        if rno is not None and tan is not None:
            out.append((date, jcd, rno, tan, fuku or 0))
    return out

def tanfuku(dirs, start, end):
    fs = [f for d in dirs for f in sorted(glob.glob(f"{d}/bk/*/k*.txt.gz")) if start <= "20" + os.path.basename(f)[1:7] <= end]
    with Pool(4) as p:
        rows = [x for r in p.map(_tanfuku, fs, chunksize=20) for x in r]
    d = pd.DataFrame(rows, columns=KEY + ["tan1", "fuku1"]); d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return d.drop_duplicates(KEY, keep="last")

def fit_weights(datadir, store):
    """IDEA-053 と同じ手順で重みを学習（探索 2023-11〜2025-12）"""
    r = norm(pd.read_parquet(f"{datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{datadir}/entries.parquet", columns=COLS))
    m = r.merge(norm(pd.read_parquet(f"{datadir}/odds.parquet")), on=KEY).merge(boat_feats(e), on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int)
    Pm = np.where(v, 1 / np.where(v, O, 1), 0); Pm /= Pm.sum(1, keepdims=True); p1 = Pm[:, FIRST == 1].sum(1); m["p1"] = p1
    win1 = (FIRST[y] == 1).astype(float); days = m["date"].values; tr = (days >= TR[0]) & (days <= TR[1])
    X = design(segments(m, tr, True, store), len(m))
    off = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1))
    return fit(X[tr], off[tr], win1[tr]), r, e

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--hist", default="/tmp/claude-0/ds_hist"); ap.add_argument("--histbk", default="/tmp/claude-0/hist")
    ap.add_argument("--weights", default="models/idea056_weights.json"); ap.add_argument("--out", default="results/idea056_stage2.md")
    a = ap.parse_args()
    w, r, e = fit_weights(a.datadir, a.store)
    os.makedirs(os.path.dirname(a.weights) or ".", exist_ok=True)
    json.dump({"idea": "IDEA-053/056", "fit": f"{TR[0]}〜{TR[1]}", "lambda": LAM, "ratio_threshold": 1.25,
               "segments": [[s, v, float(x)] for (s, v), x in zip(SEL, w)]}, open(a.weights, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("weights saved", a.weights, flush=True)
    # 12年分（オッズなし）
    rh = norm(pd.read_parquet(f"{a.hist}/races.parquet"))[KEY + ["tri_combo"]]
    eh = norm(pd.read_parquet(f"{a.hist}/entries.parquet", columns=COLS))
    ra = pd.concat([rh, r[["date", "jcd", "rno", "tri_combo"]]]).dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    ea = pd.concat([eh, e]); bad = ea[ea["pos_raw"].astype(str).str.match(r"^[KFL]")][KEY].drop_duplicates()
    ra = ra.merge(bad.assign(bad=1), on=KEY, how="left"); ra = ra[ra["bad"].isna()].drop(columns="bad")
    ra = ra[(ra["date"] >= XP[0]) & (ra["date"] <= EV[1])]
    tf = tanfuku([a.histbk, a.store], XP[0], EV[1]); print("K tan/fuku", len(tf), flush=True)
    ra = ra.merge(tf, on=KEY).merge(boat_feats(ea), on=KEY, how="left").reset_index(drop=True)
    xp = ((ra["date"] >= XP[0]) & (ra["date"] <= XP[1])).values
    score = design(segments(ra, xp, False, None), len(ra)) @ w
    year = ra["date"].str[:4].values; tan = ra["tan1"].values.astype(float); fuku = ra["fuku1"].values.astype(float)
    win = (tan > 0).astype(float)
    base = pd.DataFrame({"y": year, "tan": tan, "fuku": fuku, "win": win}).groupby("y").agg(n=("tan", "size"), tan=("tan", "mean"), fuku=("fuku", "mean"), win=("win", "mean"))
    base[["tan", "fuku"]] /= 100
    out = []; tables = []
    for top in (0.045, 0.015):
        thr = float(np.quantile(score[xp], 1 - top)); sel = score >= thr
        g = pd.DataFrame({"y": year[sel], "tan": tan[sel], "fuku": fuku[sel], "win": win[sel]}).groupby("y").agg(
            レース=("tan", "size"), 単勝1=("tan", "mean"), 複勝1=("fuku", "mean"), 一号艇の1着率=("win", "mean"))
        g[["単勝1", "複勝1"]] /= 100
        g = g.join(base.rename(columns={"n": "全レース", "tan": "全 単勝1", "fuku": "全 複勝1", "win": "全 1着率"}))
        tables.append((top, thr, sel.mean(), g))
        tot_tan = tan[sel].sum() / (100 * sel.sum()); tot_fuku = fuku[sel].sum() / (100 * sel.sum())
        out.append({"上位": f"{top:.1%}", "Σw しきい値": thr, "実際に選ばれた割合": float(sel.mean()), "レース": int(sel.sum()),
                    "単勝1（全期間）": tot_tan, "単勝1 100%以上の年": f"{int((g['単勝1'] >= 1).sum())}／{len(g)}",
                    "単勝1 基準以上の年": f"{int((g['単勝1'] > g['全 単勝1']).sum())}／{len(g)}",
                    "複勝1（全期間）": tot_fuku, "複勝1 100%以上の年": f"{int((g['複勝1'] >= 1).sum())}／{len(g)}",
                    "複勝1 基準以上の年": f"{int((g['複勝1'] > g['全 複勝1']).sum())}／{len(g)}"})
    S = pd.DataFrame(out)
    md = ["# IDEA-056 段階2 偏りを重ねて1号艇が強いレースの 単勝1・複勝1", "",
          "## (A) 12年分の確認（2014-01〜2026-08、オッズなし、判定には使わない）", "",
          f"重み：IDEA-053 と同じ21区分・探索 {TR[0]}〜{TR[1]} で学習（`{a.weights}` に保存、前向きの検証で使う）。選び方：Σw（市場の項なし、グレードは使わない）の上位 4.5%・1.5%（しきい値は 2014〜2019 で決めて固定）。"
          f"払戻は K ファイルの実額。返還のあるレースと、K に単勝の払戻が無いレースを除く（{len(ra)}レース）。", "",
          S.to_markdown(index=False, floatfmt=".3f"), ""]
    for top, thr, frac, g in tables:
        md += [f"### Σw 上位 {top:.1%}（しきい値 {thr:.3f}、選ばれた割合 {frac:.2%}）の年別", "", g.to_markdown(floatfmt=".3f"), ""]
    W = pd.DataFrame({"区分": [s for s, _ in SEL], "値": [v_ for _, v_ in SEL], "係数 w": w})
    md += ["### 重み（探索で学習、固定）", "", W.to_markdown(index=False, floatfmt=".3f"), "",
           "## (B) 前向きの検証", "", "`scripts/idea056_fwd.py` の出力（下）を参照。", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
