"""IDEA-075 の準備：IDEA-069/070 の絞り込みのうち、過去データで決まっていて当日は動かない部分を固定して保存する
usage: python scripts/freeze_idea075.py DATADIR STOREDIR [--hist /tmp/claude-0/ds_hist] [--out models/idea075_frozen.json]
- (c) 逃げ上手の選手：IDEA-060 の探索（2023-11〜2025-12）で縮めた比の上位10%（exp_idea069.racer_top10 と同じ計算）。登録番号だけを保存する
- (d) 1コースに有利な風の群：IDEA-066 の群分け（2014〜2019、exp_idea066.wind_groups と同じ計算）
- 直前情報（beforeinfo）の風向きの番号（is-wind1〜16、17 は無風）→ 結果ファイルの風向き（8方位）の対応（2026-10-25 に作り直し）。
  番号は方位ではなく、場ごとの水面の図に対する矢印の向き（時計回りの16方位）で、場ごとに回転がずれている。
  場ごとに「回転のずれ」を1つだけ当てはめ（2025-10〜2026-08 の直前情報と K）、16方位の中間の番号はその場でよく出る隣の8方位に寄せる。
  もとの「番号ごとに一番多い方位」の対応表は、隣どうしの番号が同じ方位になる重複があった（精度はほぼ同じ）。
  展示の時点の風なので、レース時の風と一致するのは場により 4〜9割
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
    # 直前情報の風向きの番号 → 8方位（場ごとに回転のずれを1つ当てはめる。2025-10〜2026-08）
    D8 = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
    b = norm(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet", columns=KEY + ["wind_dir"]))
    rk = norm(pd.read_parquet(f"{a.datadir}/races.parquet", columns=KEY + ["wind_dir"]))
    x = b.merge(rk, on=KEY, suffixes=("_b", "_k")).dropna(subset=["wind_dir_b"])
    x = x[x["wind_dir_k"].isin(DIRS + ["無風"]) & (x["date"] <= EV[1])]
    x["cb"] = x["wind_dir_b"].astype(float).astype(int)
    cmap, purity, offs = {}, {}, {}
    for jcd, g in x.groupby("jcd"):
        y = g[(g["cb"] <= 16) & (g["wind_dir_k"] != "無風")]
        k = y["wind_dir_k"].map({dd: i for i, dd in enumerate(D8)}).values; c = y["cb"].values
        sc = []
        for o in range(16):
            i16 = (c - 1 + o) % 16
            sc.append(np.mean((k == (i16 // 2) % 8) | (k == ((i16 + 1) // 2) % 8)))
        o = int(np.argmax(sc)); offs[jcd] = o; mp = {}
        for code in range(1, 17):
            i16 = (code - 1 + o) % 16
            if i16 % 2 == 0:
                mp[str(code)] = D8[i16 // 2]
            else:
                lo, hi = D8[(i16 // 2) % 8], D8[((i16 + 1) // 2) % 8]
                sub = y.loc[y["cb"] == code, "wind_dir_k"]
                mp[str(code)] = lo if (sub == lo).sum() >= (sub == hi).sum() else hi
        mp["17"] = "無風"; cmap[jcd] = mp
        purity[jcd] = round(float((g["cb"].astype(str).map(mp) == g["wind_dir_k"]).mean()), 3)
    out = {"idea": "IDEA-075（IDEA-069/070 の絞り込みを当日に使うための固定値）",
           "c_top10_toban": sorted(int(t) for t in top10), "c_k": float(k60), "c_q90": float(q90),
           "d_fav": fav, "wind_code_map": cmap, "wind_code_offset": offs, "wind_code_purity": purity,
           "wind_code_note": "番号は場ごとの水面の図に対する矢印（時計回り16方位）。場ごとの回転のずれ（offset）で8方位に直す。purity はレース時の風（K）との一致率（2025-10〜2026-08）"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("top10", len(top10), "fav", len(fav), "purity", purity)

if __name__ == "__main__":
    main()
