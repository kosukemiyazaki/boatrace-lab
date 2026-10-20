"""IDEA-066 風向き×コース：追い風なら差し、向かい風ならまくり、を市場は織り込んでいるか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea066.py DATADIR HISTDS STORE [--out results/idea066.md]
- 風向きの群（場ごと、2014〜2019 のデータだけで決める）：その場・その風向き（8方位）の 1コースの1着率で風向きを並べ、
  レース300以上の風向きの上位3分の1 = 「1コース有利」、下位3分の1 = 「1コース不利」、それ以外（中位・300未満・無風）= 「中間」
- 風速：〜2m／3〜4m／5m〜。進入コース（entries.course）：1／2〜3／4〜6
- 区分：風向きの群 × 進入コース、風向きの群 × 風速 × 進入コース。1着率 ÷ 市場の見込み（確定3連単、2023-10 以降）
- 探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で判定（IDEA-063 と同じ基準）。12年の参考は群分けに使っていない 2020〜2026
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL
from exp_idea065 import market_boats, attach, evaluate, bgroup

FIN = {"01", "02", "03", "04", "05", "06"}
DIRS = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]

def wind_groups(e, r):
    """場ごと・風向きごとの群（2014〜2019 の 1コースの1着率から）"""
    c1 = e[(e["course"] == 1) & e["pos_raw"].astype(str).isin(FIN) & (e["date"] >= "20140101") & (e["date"] <= "20191231")]
    c1 = c1.merge(r[KEY + ["wind_dir"]], on=KEY)
    g = c1.assign(w=(c1["pos_raw"].astype(str) == "01").astype(float)).groupby(["jcd", "wind_dir"])["w"].agg(["mean", "size"]).reset_index()
    rows = []
    for jcd, d in g.groupby("jcd"):
        d = d[d["wind_dir"].isin(DIRS) & (d["size"] >= 300)].sort_values("mean", ascending=False)
        n = len(d); k = n // 3
        for i, (_, x) in enumerate(d.iterrows()):
            grp = "1コース有利" if i < k else ("1コース不利" if i >= n - k else "中間")
            rows.append({"jcd": jcd, "wind_dir": x["wind_dir"], "grp": grp, "c1rate": x["mean"], "n": int(x["size"])})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("store")
    ap.add_argument("--out", default="results/idea066.md"); a = ap.parse_args()
    cols = KEY + ["boat", "course", "pos_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in (a.histds, a.datadir)], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last")
    r = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["wind", "wind_dir"])) for p in (a.histds, a.datadir)], ignore_index=True)
    r = r.drop_duplicates(KEY, keep="last")
    G = wind_groups(e, r)
    r = r.merge(G[["jcd", "wind_dir", "grp"]], on=["jcd", "wind_dir"], how="left"); r["grp"] = r["grp"].fillna("中間")
    w = pd.to_numeric(r["wind"], errors="coerce")
    r["spd"] = np.where(w <= 2, "〜2m", np.where(w <= 4, "3〜4m", np.where(w >= 5, "5m〜", None)))
    x = e[~e["pos_raw"].astype(str).str.startswith("K") & e["course"].between(1, 6)].merge(r[KEY + ["grp", "spd"]], on=KEY)
    x["cg"] = bgroup(x["course"]) + "コース"
    x["d1"] = x["grp"] + "・" + x["cg"]
    x["d2"] = (x["grp"] + "・" + x["spd"].astype(str) + "・" + x["cg"]).where(x["spd"].notna())
    dims = {"風向きの群×コース": "d1", "風向きの群×風速×コース": "d2"}
    mk = market_boats(a.datadir)
    L = attach(x, mk)
    R, S, B, pooled, judge = evaluate(L, dims, mk, a.store, metrics=("1着",))
    # 12年の参考（2020〜2026、オッズなし）：同じ年・同じコース群の全出走の1着率との比
    H = x[x["date"] >= "20200101"].copy(); H["year"] = H["date"].str[:4]
    H["won"] = (H["pos_raw"].astype(str) == "01").astype(float)
    base = H.groupby(["year", "cg"])["won"].mean().rename("bw")
    out = []
    for dn, col in dims.items():
        g = H.dropna(subset=[col]).groupby([col, "year", "cg"])["won"].agg(["sum", "size"]).reset_index()
        g = g.merge(base, left_on=["year", "cg"], right_index=True); g["ew"] = g["size"] * g["bw"]
        yr = g.groupby([col, "year"])[["sum", "ew", "size"]].sum().reset_index(); yr["r"] = yr["sum"] / yr["ew"]
        t = yr.groupby(col).agg(年数=("year", "size"), 出走=("size", "sum"), r=("r", "mean"), u=("r", lambda s: int((s > 1).sum()))).reset_index().rename(columns={col: "値"})
        t.insert(0, "区分", dn); out.append(t)
    Hs = pd.concat(out, ignore_index=True)
    Hs.columns = ["区分", "値", "年数", "出走", "1着率÷同じ年・コース群の全出走（年平均）", "上回った年"]
    Gs = G.pivot_table(index="jcd", columns="grp", values="wind_dir", aggfunc=lambda s: "・".join(s)).reset_index()
    Rshow = R[[c for c in R.columns if not c.startswith("_")]]
    md = [f"# IDEA-066 風向き×コース（群分け 2014〜2019／探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}）", "",
          f"**判定：{judge}**", "",
          "## 定義", "",
          "- 公式の水面の向きのデータがないため、登録どおり 2014〜2019 の 1コースの1着率で場ごとに風向きを3群に分けた（評価・探索の期間とは重ならない）。",
          "- 場ごとに、8方位のうちレースが300以上ある風向きを 1コースの1着率の高い順に並べ、上位3分の1 =「1コース有利」、下位3分の1 =「1コース不利」、残り・300未満・無風 =「中間」。",
          "- 風向き・風速は結果ファイルの値。進入コースは結果ファイルの実際のコース。市場の見込みは確定3連単オッズから出したその艇の1着確率。", "",
          "## 場ごとの風向きの群（2014〜2019 で決めたもの）", "", Gs.to_markdown(index=False), "",
          "## 選ばれた区分（探索で 300艇以上・比 1.08 以上か 0.92 以下・CI が 1 をまたがない）と評価", "",
          (S.to_markdown(index=False, floatfmt=".3f") if len(S) else "なし"), "",
          "## 買い方（評価 2026-01〜08、単勝は K の実額・強い向きのみ、3連単は確定オッズで π×オッズ > 1.0・300倍以下）", "",
          (B[[c for c in B.columns if not c.startswith("_")]].to_markdown(index=False, floatfmt=".3f") if len(B) else "なし"), "", pooled, "",
          "## 全区分（1着の 実際÷市場の見込み）", "", Rshow.to_markdown(index=False, floatfmt=".3f"), "",
          "## 12年の参考（群分けに使っていない 2020〜2026、オッズなし。同じ年・同じコース群の全出走との比）", "", Hs.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(judge); print(S.to_string() if len(S) else "選択なし"); print(B[[c for c in B.columns if not c.startswith("_")]].to_string() if len(B) else ""); print(pooled)
    print(Rshow[Rshow["区分"] == "風向きの群×コース"].to_string())

if __name__ == "__main__":
    main()
