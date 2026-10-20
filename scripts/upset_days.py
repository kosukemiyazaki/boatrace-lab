"""「不思議な日」の一覧（IDEA-046 の観察用、判定には使わない）
usage: python scripts/upset_days.py DATADIR STOREDIR [--out results/upset_days.md] [--top 30]
- 場 × 日ごとに、万舟（確定100倍以上）の実際の本数 − 市場の見込み（各レースの100倍以上の組の確率の合計）
- 大きい順に並べ、その日の天候・平均風速・平均波高・水質・グレードを添える
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp_idea046 import market, grades, WATER

NAMES = {"01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
         "09": "津", "10": "三国", "11": "びわこ", "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
         "17": "宮島", "18": "徳山", "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--out", default="results/upset_days.md"); ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    m, *_ = market(a.datadir)
    gr, _ = grades(m, a.store); m["grade"] = gr
    m["wind"] = pd.to_numeric(m["wind"], errors="coerce"); m["wave"] = pd.to_numeric(m["wave"], errors="coerce")
    g = m.groupby(["date", "jcd"]).agg(レース=("rno", "count"), 万舟=("manshu", "sum"), 見込み=("exp_m", "sum"),
                                       天候=("weather", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
                                       平均風速=("wind", "mean"), 平均波高=("wave", "mean"), グレード=("grade", "max")).reset_index()
    g = g[g["レース"] >= 8]
    g["実際−見込み"] = g["万舟"] - g["見込み"]
    g["場"] = g["jcd"].map(NAMES); g["水質"] = g["jcd"].map(WATER).map({0: "淡水", 1: "海水", 2: "汽水"})
    g["グレード"] = g["グレード"].map({4: "SG", 3: "G1", 2: "G2", 1: "G3", 0: "一般"})
    cols = ["date", "場", "レース", "万舟", "見込み", "実際−見込み", "天候", "平均風速", "平均波高", "水質", "グレード"]
    top = g.sort_values("実際−見込み", ascending=False).head(a.top)[cols]
    recent = g[g["date"] >= (pd.Timestamp(g["date"].max()) - pd.Timedelta(days=13)).strftime("%Y%m%d")].sort_values("実際−見込み", ascending=False).head(10)[cols]
    sd = g["実際−見込み"].std()
    md = [f"# 不思議な日（市場の見込みより万舟が多かった 場×日、{g['date'].min()}〜{g['date'].max()}、{len(g)}開催日）", "",
          f"1日あたりの「実際 − 見込み」の標準偏差は {sd:.2f} 本。偶然でも上位はこのくらい出るので、共通点は IDEA-046 の評価期間で確かめる。", "",
          "## 全期間の上位", "", top.to_markdown(index=False, floatfmt=".2f"), "",
          "## 直近2週間の上位", "", recent.to_markdown(index=False, floatfmt=".2f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
