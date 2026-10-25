"""IDEA-075 の見直し（2026-10-25、独立レビューの指摘への対応）：IDEA-069/070 の評価を、前向きと同じ定義でやり直す
usage: python scripts/exp_idea075_recheck.py DATADIR STOREDIR [--hist /tmp/claude-0/ds_hist] [--out results/idea075_recheck.md]
- 条件 (a)(b)(c)(d)・重み・(c) の選手・(d) の群は exp_idea069 と同じ（新しく学習しない）
- 比べるもの（評価 2026-01〜08、参考 2023-11〜2025-12）：
  V0 もとの評価（どれかの艇が欠場・F・L のレースを除く、(b) は確定の3連単、(d) は結果ファイル K のレース時の風）
  V1 除外を前向きと同じに（1号艇の欠場・F・L だけ返還＝100円戻し、ほかの艇の欠場・F・L のレースは残す）
  V2 V1 ＋ (d) を直前情報（beforeinfo）の風向きで判定（models/idea075_frozen.json の対応表。直前情報は 2025-10 以降だけ）
  V3 V2 から (b) を外す（(a)(c)(d) のうち2つ以上）
- 締切前の3連単は過去にないので、(b) は確定オッズでしか作れない（買う時点では分からない値）。V2 と V3 の差が「(b) を確定オッズで作った分」
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY, TR, EV, FIRST, boat_feats, segments
from exp_idea053 import SEL, design, boot
from exp_idea056_stage2 import COLS, tanfuku
from idea075_lib import wind_from_code

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--weights", default="models/idea056_weights.json"); ap.add_argument("--frozen", default="models/idea075_frozen.json")
    ap.add_argument("--out", default="results/idea075_recheck.md"); a = ap.parse_args()
    W = json.load(open(a.weights, encoding="utf-8")); w = np.array([x[2] for x in W["segments"]])
    assert [(s, v) for s, v, _ in W["segments"]] == list(SEL)
    Z = json.load(open(a.frozen, encoding="utf-8")); top10 = set(Z["c_top10_toban"]); fav = set(map(tuple, Z["d_fav"]))
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo", "wind_dir"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=COLS))
    t1 = e[e["boat"] == 1][KEY + ["toban", "pos_raw"]].drop_duplicates(KEY).rename(columns={"toban": "t1", "pos_raw": "pos1"})
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(boat_feats(e), on=KEY, how="left").merge(t1, on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); p1 = P[:, FIRST == 1].sum(1); m["p1"] = p1
    days = m["date"].values; tr = (days >= TR[0]) & (days <= TR[1])
    X = design(segments(m, tr, True, a.store), n)
    off = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1)); padj = 1 / (1 + np.exp(-(off + X @ w)))
    ratio = padj / np.clip(p1, 1e-9, 1)
    bad_any = e[e["pos_raw"].astype(str).str.match(r"^[KFL]")][KEY].drop_duplicates().assign(bad=1)
    m = m.merge(bad_any, on=KEY, how="left")
    tf = tanfuku([a.store], "20231101", EV[1])
    bi = norm(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet", columns=KEY + ["wind_dir"])).drop_duplicates(KEY).rename(columns={"wind_dir": "bi_code"})
    mm = m[KEY + ["bad", "pos1", "wind_dir", "t1", "age1"]].assign(i=np.arange(n)).merge(tf, on=KEY).merge(bi, on=KEY, how="left")
    idx = mm["i"].values; d = days[idx]
    ref1 = mm["pos1"].astype(str).str.match(r"^[KFL]").values
    tan = np.where(ref1, 100.0, mm["tan1"].values.astype(float)); fuku = np.where(ref1, 100.0, mm["fuku1"].values.astype(float))
    A = (pd.to_numeric(mm["age1"], errors="coerce").values <= 22)
    B = ratio[idx] >= 1.15
    C = mm["t1"].isin(top10).values
    Dk = np.array([(j, x) in fav for j, x in zip(mm["jcd"], mm["wind_dir"])])
    wb = [wind_from_code(Z, j, c) for j, c in zip(mm["jcd"], mm["bi_code"])]
    Db = np.array([(j, x) in fav for j, x in zip(mm["jcd"], wb)]); has_bi = mm["bi_code"].notna().values
    old_ok = mm["bad"].isna().values
    V = {"V0 もとの評価（欠場・F・L のあるレースを除く、(b) 確定、(d) レース時の風）": (old_ok, A.astype(int) + B + C + Dk >= 2),
         "V1 除外を前向きと同じに（1号艇の欠場・F・L だけ返還）": (np.ones(len(mm), bool), A.astype(int) + B + C + Dk >= 2),
         "V1' V1 を直前情報のあるレースだけで（V2 との比べ用）": (has_bi, A.astype(int) + B + C + Dk >= 2),
         "V2 V1 ＋ (d) を直前情報の風で（直前情報のあるレース）": (has_bi, A.astype(int) + B + C + Db >= 2),
         "V3 V2 から (b) を外す（(a)(c)(d) のうち2つ以上）": (has_bi, A.astype(int) + C + Db >= 2),
         "参考：全レース（1号艇の単勝・複勝、除外は V1 と同じ）": (np.ones(len(mm), bool), np.ones(len(mm), bool))}
    rows = []
    for per, pm in (("評価 2026-01〜08", (d >= EV[0]) & (d <= EV[1])), ("参考 2023-11〜2025-12", (d >= TR[0]) & (d <= TR[1]))):
        for nm, (base, sel) in V.items():
            s = pm & base & sel
            row = {"版": nm, "期間": per, "レース": int(s.sum())}
            for lab, pay in (("単勝1", tan), ("複勝1", fuku)):
                roi, lo, hi = boot(d[s], np.full(s.sum(), 100.0), pay[s]) if s.sum() else (np.nan,) * 3
                row[lab] = roi; row[f"{lab} CI下"] = lo; row[f"{lab} CI上"] = hi
            rows.append(row)
    R = pd.DataFrame(rows)
    ev = (d >= EV[0]) & (d <= EV[1]) & has_bi
    agree = float(np.mean(Dk[ev] == Db[ev])) if ev.sum() else np.nan
    md = ["# IDEA-075 の見直し：IDEA-069/070 の評価を前向きと同じ定義でやり直す（2026-10-25）", "",
          "独立レビュー（2026-10-25）の指摘への対応。条件・重み・選手・風の群はもとのまま（新しく学習しない）。払戻は K の実額。", "",
          "- V0 は `results/idea069.md` の (e) 2つ以上と同じ作り方（再現の確認）。",
          "- V1 は除外を前向きと同じにしたもの：1号艇の欠場・F・L だけ返還（100円戻し）、ほかの艇の欠場・F・L があってもレースは残す。",
          "- V2 は (d) を直前情報の風（締切前に見える値）で判定。直前情報は 2025-10 以降しかないので、参考期間はほぼ空。",
          "- V3 は (b) を外したもの。過去には締切前の3連単がないので (b) は確定オッズでしか作れない（買う時点では分からない値）。**V2 と V3 の差が、確定オッズで作った (b) の分**。",
          f"- 評価期間で、直前情報の風で判定した (d) とレース時の風の (d) が一致した割合：{agree:.3f}", "",
          R.to_markdown(index=False, floatfmt=".3f"), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
