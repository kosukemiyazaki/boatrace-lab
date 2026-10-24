"""IDEA-075 の準備：IDEA-069/070 の絞り込みのうち、過去データで決まっていて当日は動かない部分を固定して保存する
usage: python scripts/freeze_idea075.py DATADIR STOREDIR [--hist /tmp/claude-0/ds_hist] [--out models/idea075_frozen.json]
- (c) 逃げ上手の選手：IDEA-060 の探索（2023-11〜2025-12）で縮めた比の上位10%（exp_idea069.racer_top10 と同じ計算）。登録番号だけを保存する
- (d) 1コースに有利な風の群：IDEA-066 の群分け（2014〜2019、exp_idea066.wind_groups と同じ計算）
- 参考：直前情報（beforeinfo）の風向きの番号 → 結果ファイルの風向き（8方位）の対応。場ごとに一番多い組み合わせ（2023-10 以降の daily）。
  展示の時点の風なのでレース時と一致するのは場により 4〜8割。締切前の (d) は「推定」として扱う
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY, EV
from exp_idea056_stage2 import COLS
from exp_idea066 import wind_groups, DIRS
from exp_idea069 import racer_top10

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--hist", default="/tmp/claude-0/ds_hist"); ap.add_argument("--out", default="models/idea075_frozen.json")
    a = ap.parse_args()
    # (c) exp_idea069 と同じデータの組み方
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=COLS))
    t1 = e[e["boat"] == 1][KEY + ["toban"]].drop_duplicates(KEY).rename(columns={"toban": "t1"})
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(t1, on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    top10, k60, q90 = racer_top10(m, P, y)
    # (d)
    ce = KEY + ["boat", "course", "pos_raw"]
    eall = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=ce)) for p in (a.hist, a.datadir)], ignore_index=True).drop_duplicates(KEY + ["boat"], keep="last")
    rall = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["wind", "wind_dir"])) for p in (a.hist, a.datadir)], ignore_index=True).drop_duplicates(KEY, keep="last")
    G = wind_groups(eall, rall)
    fav = G.loc[G["grp"] == "1コース有利", ["jcd", "wind_dir"]].values.tolist()
    # 直前情報の風向きの番号 → 8方位（場ごと）
    b = norm(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet", columns=KEY + ["wind_dir"]))
    rk = norm(pd.read_parquet(f"{a.datadir}/races.parquet", columns=KEY + ["wind_dir"]))
    x = b.merge(rk, on=KEY, suffixes=("_b", "_k")).dropna(subset=["wind_dir_b"])
    x = x[x["wind_dir_k"].isin(DIRS + ["無風"])]
    x["code"] = x["wind_dir_b"].astype(float).astype(int).astype(str)
    cmap, purity = {}, {}
    for jcd, g in x.groupby("jcd"):
        cmap[jcd] = {}
        hit = 0
        for code, h in g.groupby("code"):
            vc = h["wind_dir_k"].value_counts()
            if vc.sum() >= 20:
                cmap[jcd][code] = vc.index[0]; hit += int(vc.iloc[0])
        purity[jcd] = round(hit / len(g), 3)
    out = {"idea": "IDEA-075（IDEA-069/070 の絞り込みを当日に使うための固定値）",
           "c_top10_toban": sorted(int(t) for t in top10), "c_k": float(k60), "c_q90": float(q90),
           "d_fav": fav, "wind_code_map": cmap, "wind_code_purity": purity}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("top10", len(top10), "fav", len(fav), "purity", purity)

if __name__ == "__main__":
    main()
