"""締切前の予想（live のジョブの中で使う）。凍結モデルの3連単確率を計算するだけで、買い目は出さない。
- 起動時: 前日までの成績・直前情報と当日の番組表を読み込み、当日全レースの特徴量を計算する
- 当日の結果・直前情報が入るたびに refresh() で特徴量を計算し直す（学習時と同じ手順を全データに対して行う）
  → 節間成績や過去成績に、当日の前のレースの結果が入る
- 締切6分前: そのレースの直前情報を入れて、レース内の特徴量（展示タイムの相対値・順位、普段との差、進入の差、展示STの相対値）
  を add_beforeinfo と同じ式で計算し、予想する
"""
import json
import threading

import lightgbm as lgb
import numpy as np
import pandas as pd

from boatlib.features import (BI_BOAT_COLS, BI_RACE_COLS, FEATS, FEATS_BI, FEATS_C1, add_beforeinfo,
                              history_features, meet_motor_features, race_features)
from boatlib.model import PERMS
from boatlib.parse import COMBOS, parse_b

PI = np.array([p[0] for p in PERMS]); PJ = np.array([p[1] for p in PERMS]); PK = np.array([p[2] for p in PERMS])
FEATS_BY_NAME = {"base": FEATS, "bi": FEATS_BI, "c1": FEATS_C1}

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
        self.feats_name = meta.get("feats_name", "base")
        self.F = FEATS_BY_NAME[self.feats_name]
        self.model_name = model_dir.rstrip("/").split("/")[-1]
        self.boosters = [lgb.Booster(model_file=f"{model_dir}/stage{k}.txt") for k in (1, 2, 3)]
        self.iso = json.load(open(f"{model_dir}/isotonic.json"))
        hist = pd.read_parquet(f"{datadir}/entries.parquet")
        self.hist = hist[hist["date"] < date]
        t = pd.DataFrame(parse_b(btext, date))
        for c in ("pos", "exh_time", "course", "st"):
            t[c] = np.nan
        t["pos_raw"] = ""; t["st_raw"] = ""
        self.today_b = t[[c for c in hist.columns if c in t.columns]]
        hb = pd.read_parquet(f"{datadir}/beforeinfo.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
        self.hist_bi = hb[hb["date"] < date]
        self.lock = threading.Lock()
        self.version = 0
        self.refresh([], {})

    def refresh(self, results, live_bi):
        """results: [dict(jcd, rno, boat, toban, pos_raw, pos, course, st, st_raw)]（当日の結果）
        live_bi: {(jcd, rno): parse_beforeinfo の dict}（当日の締切前の直前情報）"""
        t = self.today_b.copy()
        if results:
            r = pd.DataFrame(results)[["jcd", "rno", "boat", "toban", "pos_raw", "pos", "course", "st", "st_raw"]]
            r["rno"] = r["rno"].astype(int)
            t = t.drop(columns=["pos_raw", "pos", "course", "st", "st_raw"]).merge(r, on=["jcd", "rno", "boat", "toban"], how="left")
            t["pos_raw"] = t["pos_raw"].fillna("")
        e = pd.concat([self.hist, t], ignore_index=True)
        e = race_features(history_features(e))
        lb = pd.DataFrame([dict(date=self.date, jcd=j, rno=int(n), **{c: v for c, v in bi.items()}) for (j, n), bi in live_bi.items() if bi])
        e = add_beforeinfo(e, pd.concat([self.hist_bi, lb], ignore_index=True) if len(lb) else self.hist_bi)
        if self.feats_name == "c1":
            e = meet_motor_features(e)
        today = {k: g.sort_values("boat").copy() for k, g in e[e["date"] == self.date].groupby(["jcd", "rno"])}
        with self.lock:
            self.today = today
            self.version += 1
            self.n_results = len(results)

    def features(self, jcd, rno, bi):
        """予想に使う特徴量（1レース分、艇番順）と、直前情報の有無"""
        with self.lock:
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
        return r, has_bi

    def predict(self, jcd, rno, bi):
        """bi: parse_beforeinfo の結果（dict。None なら直前情報なし）。返り値: (1着確率6個, 3連単確率120個, 直前情報の有無)"""
        r, has_bi = self.features(jcd, rno, bi)
        X = np.full((1, 6, len(self.F)), np.nan); mask = np.zeros((1, 6), bool)
        X[0, r["boat"].values - 1] = r[self.F].values.astype(float); mask[0, r["boat"].values - 1] = True
        s = [b.predict(X.reshape(-1, len(self.F)), raw_score=True).reshape(-1, 6) for b in self.boosters]
        T = stage_trifecta(*s, mask)
        P = np.interp(T.ravel(), self.iso["x"], self.iso["y"]); P = P / P.sum()
        win = np.zeros(6)
        for ci, c in enumerate(COMBOS):
            win[int(c[0]) - 1] += P[ci]
        return win, P, has_bi
