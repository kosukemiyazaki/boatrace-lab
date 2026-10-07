"""締切前の予想（live のジョブの中で使う）。凍結モデルの3連単確率を計算するだけで、買い目は出さない。
- 起動時に1回: 前日までの成績・直前情報と当日の番組表から、当日全レースの特徴量（直前情報以外）を計算しておく
- 締切6分前: そのレースの直前情報を入れて、レース内の特徴量（展示タイムの相対値・順位、普段との差、進入の差、展示STの相対値）
  を add_beforeinfo と同じ式で計算し、予想する
注意: 選手の「普段の展示」（直近10走の平均）は前日までの値。当日の前のレースの展示は入らない（学習時は入っている）。
"""
import json
import lightgbm as lgb
import numpy as np
import pandas as pd

from boatlib.features import (BI_BOAT_COLS, BI_RACE_COLS, FEATS, FEATS_BI, add_beforeinfo, history_features,
                              race_features)
from boatlib.parse import COMBOS, parse_b
from boatlib.model import PERMS

PI = np.array([p[0] for p in PERMS]); PJ = np.array([p[1] for p in PERMS]); PK = np.array([p[2] for p in PERMS])

def stage_trifecta(s1, s2, s3, mask):
    """scripts/exp_trifecta.py の stage_trifecta と同じ式"""
    e1, e2, e3 = [np.where(mask, np.exp(s - s.max(1, keepdims=True)), 0) for s in (s1, s2, s3)]
    a = e1[:, PI] / e1.sum(1, keepdims=True)
    b = e2[:, PJ] / np.clip(e2.sum(1, keepdims=True) - e2[:, PI], 1e-12, None)
    c = e3[:, PK] / np.clip(e3.sum(1, keepdims=True) - e3[:, PI] - e3[:, PJ], 1e-12, None)
    t = a * b * c
    return t / t.sum(1, keepdims=True)

class Predictor:
    def __init__(self, datadir, model_dir, btext, date):
        self.date = date
        meta = json.load(open(f"{model_dir}/meta.json"))
        self.F = FEATS_BI if meta.get("feats_name") == "bi" else FEATS
        self.model_name = model_dir.rstrip("/").split("/")[-1]
        self.boosters = [lgb.Booster(model_file=f"{model_dir}/stage{k}.txt") for k in (1, 2, 3)]
        self.iso = json.load(open(f"{model_dir}/isotonic.json"))
        hist = pd.read_parquet(f"{datadir}/entries.parquet")
        hist = hist[hist["date"] < date]
        today = pd.DataFrame(parse_b(btext, date))
        for c in ("pos", "exh_time", "course", "st"):
            today[c] = np.nan
        today["pos_raw"] = ""; today["st_raw"] = ""
        e = pd.concat([hist, today[[c for c in hist.columns if c in today.columns]]], ignore_index=True)
        e = race_features(history_features(e))
        hb = pd.read_parquet(f"{datadir}/beforeinfo.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
        e = add_beforeinfo(e, hb[hb["date"] < date])
        self.today = {k: g.sort_values("boat").copy() for k, g in e[e["date"] == date].groupby(["jcd", "rno"])}

    def predict(self, jcd, rno, bi):
        """bi: parse_beforeinfo の結果（dict。None なら直前情報なし）。返り値: (1着確率6個, 3連単確率120個, 直前情報の有無)"""
        r = self.today[(jcd, int(rno))].copy()
        has_bi = bool(bi) and bi.get("exh1") is not None
        if has_bi:
            for c in BI_RACE_COLS:
                r[f"bi_{c}"] = bi.get(c)
            for c in BI_BOAT_COLS:
                r[f"bi_{c}"] = [bi.get(f"{c}{b}") for b in r["boat"]]
            r = r.astype({f"bi_{c}": float for c in BI_RACE_COLS + BI_BOAT_COLS})
            # add_beforeinfo と同じ式（1レース分）
            r["exh_time"] = r["exh_time"].fillna(r["bi_exh"])
            r["r_exh"] = r["exh_time"] - r["exh_time"].mean()
            r["r_exh_rank"] = r["exh_time"].rank(method="min")
            r["d_exh_self"] = r["r_exh"] - r["h_rexh10"]
            r["d_course"] = r["bi_ex_course"] - r["boat"]
            r["r_ex_st"] = r["bi_ex_st"] - r["bi_ex_st"].mean()
        X = np.full((1, 6, len(self.F)), np.nan); mask = np.zeros((1, 6), bool)
        X[0, r["boat"].values - 1] = r[self.F].values.astype(float); mask[0, r["boat"].values - 1] = True
        s = [b.predict(X.reshape(-1, len(self.F)), raw_score=True).reshape(-1, 6) for b in self.boosters]
        T = stage_trifecta(*s, mask)
        P = np.interp(T.ravel(), self.iso["x"], self.iso["y"]); P = P / P.sum()
        win = np.zeros(6)
        for ci, c in enumerate(COMBOS):
            win[int(c[0]) - 1] += P[ci]
        return win, P, has_bi
