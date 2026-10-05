"""特徴量。レース前に分かる情報だけを使う。
- 番組表: 級別・勝率・2連率・当地・モーター・ボート・年齢・体重
- 展示タイム（直前情報。締切前に公表される）
  ※ 風・波はKファイルの値がレース時点のものなので使わない
- 選手の過去成績（当該レースより前のレースのみ）: 直近30走の平均着順/勝率/3連対率/平均ST、
  当該艇番コースでの過去の1着率・3連対率、直近のF回数
"""
import numpy as np
import pandas as pd

CLS = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

def history_features(e):
    """e: entries（全期間）。時系列順に racer ごとの過去成績を付ける。"""
    e = e.copy()
    e["t"] = e["date"].astype(int) * 100 + e["rno"]
    e = e.sort_values(["toban", "t"]).reset_index(drop=True)
    started = ~e["pos_raw"].isin(["K0", "K1"])
    fin = e["pos"].where(e["pos"].notna(), 6.0)  # 失格・F等は6着扱い
    e["_fin"] = fin.where(started)
    e["_win"] = (e["pos"] == 1).astype(float).where(started)
    e["_top2"] = (e["pos"] <= 2).astype(float).where(started)
    e["_top3"] = (e["pos"] <= 3).astype(float).where(started)
    e["_st"] = e["st"]
    e["_f"] = e["pos_raw"].astype(str).str.startswith("F").astype(float)
    g = e.groupby("toban", sort=False)
    for c, w in [("_fin", 30), ("_win", 30), ("_top2", 30), ("_top3", 30), ("_st", 30), ("_f", 60)]:
        e[f"h{c}{w}"] = g[c].transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
    e["h_n"] = g.cumcount()
    # 過去の「実際の進入コース」別成績を、今回の艇番コースについて引く（as-of join）
    e["_lane"] = e["course"].fillna(e["boat"]).astype(float)
    lane = e[["toban", "_lane", "t", "_win", "_top3"]].rename(columns={"_lane": "boat_lane"})
    lane["cw"] = lane.groupby(["toban", "boat_lane"])["_win"].transform(lambda s: s.fillna(0).cumsum())
    lane["c3"] = lane.groupby(["toban", "boat_lane"])["_top3"].transform(lambda s: s.fillna(0).cumsum())
    lane["cn"] = lane.groupby(["toban", "boat_lane"])["_win"].transform(lambda s: s.notna().cumsum())
    lane = lane.sort_values("t")
    q = e[["toban", "boat", "t"]].copy()
    q["boat_lane"] = q["boat"].astype(float)
    q["_i"] = np.arange(len(q))
    q = q.sort_values("t")
    m = pd.merge_asof(q, lane[["toban", "boat_lane", "t", "cw", "c3", "cn"]], on="t",
                      by=["toban", "boat_lane"], allow_exact_matches=False).sort_values("_i")
    prior = 2.0  # 縮小推定の擬似件数
    e["hl_win"] = ((m["cw"].fillna(0) + prior * e["boat"].map(BASE_WIN)) / (m["cn"].fillna(0) + prior)).values
    e["hl_top3"] = ((m["c3"].fillna(0) + prior * e["boat"].map(BASE_TOP3)) / (m["cn"].fillna(0) + prior)).values
    e["hl_n"] = m["cn"].fillna(0).values
    return e.drop(columns=[c for c in e.columns if c.startswith("_")])

BASE_WIN = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.11, 5: 0.06, 6: 0.02}
BASE_TOP3 = {1: 0.85, 2: 0.60, 3: 0.55, 4: 0.50, 5: 0.35, 6: 0.20}

FEATS = ["boat", "jcd_i", "cls_i", "age", "weight", "nat_win", "nat_2r", "loc_win", "loc_2r",
         "motor_2r", "boat_2r", "exh_time", "dist",
         "h_fin30", "h_win30", "h_top230", "h_top330", "h_st30", "h_f60", "h_n", "hl_win", "hl_top3", "hl_n",
         "r_nat_win", "r_exh", "r_exh_rank", "r_motor", "r_hst", "r_hfin", "r_cls"]

def race_features(e):
    e["jcd_i"] = e["jcd"].astype(int)
    e["cls_i"] = e["cls"].map(CLS)
    e["loc_win"] = e["loc_win"].where(e["loc_win"] > 0)
    e["loc_2r"] = e["loc_2r"].where(e["loc_win"].notna())
    key = ["date", "jcd", "rno"]
    g = e.groupby(key)
    rel = lambda c: e[c] - g[c].transform("mean")
    e["r_nat_win"] = rel("nat_win")
    e["r_exh"] = rel("exh_time")
    e["r_exh_rank"] = g["exh_time"].rank(method="min")
    e["r_motor"] = rel("motor_2r")
    e["r_hst"] = rel("h_st30")
    e["r_hfin"] = rel("h_fin30")
    e["r_cls"] = rel("cls_i")
    return e
