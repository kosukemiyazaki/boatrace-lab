"""1レースの予想（凍結モデル v3、仮想。投票はしない）
usage: python scripts/predict_race.py DATADIR STOREDIR BFILE --jcd 24 --rno 9 [--date YYYYMMDD] [--model models/frozen_v3]
- 過去成績: DATADIR の entries（前日まで）
- 出走表: BFILE（当日の番組表テキスト、cp932）
- 直前情報・オッズ: STOREDIR/live/（締切6分前に取得したもの。なければ直前情報なしで予想）
"""
import argparse, glob, json, os, sys
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS, FEATS_BI, add_beforeinfo, history_features, race_features
from boatlib.parse import COMBOS, parse_b
from exp_trifecta import stage_trifecta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("bfile")
    ap.add_argument("--jcd", required=True); ap.add_argument("--rno", type=int, required=True)
    ap.add_argument("--date", default=pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y%m%d"))
    ap.add_argument("--model", default="models/frozen_v3")
    a = ap.parse_args()
    key = ["date", "jcd", "rno"]
    hist = pd.read_parquet(f"{a.datadir}/entries.parquet")
    today = pd.DataFrame(parse_b(open(a.bfile, "rb").read().decode("cp932", "replace"), a.date))
    for c in ("pos", "exh_time", "course", "st"):
        today[c] = np.nan
    today["pos_raw"] = ""; today["st_raw"] = ""
    e = pd.concat([hist[hist["date"] < a.date], today[[c for c in hist.columns if c in today.columns]]], ignore_index=True)
    meta = json.load(open(f"{a.model}/meta.json")); F = FEATS_BI if meta.get("feats_name") == "bi" else FEATS
    e = race_features(history_features(e))
    lbi = [pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in glob.glob(f"{a.store}/live/beforeinfo/{a.date}*.csv.gz")]
    lbi = pd.concat(lbi) if lbi else pd.DataFrame()
    if len(lbi):
        lbi["jcd"] = lbi["jcd"].str.zfill(2)
        lbi = lbi[lbi["fetched_at"] < lbi["deadline"] + ":00"].drop_duplicates(key, keep="last")
    hb = pd.read_parquet(f"{a.datadir}/beforeinfo.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    allbi = pd.concat([hb[hb["date"] < a.date], lbi.drop(columns=["deadline", "fetched_at"], errors="ignore")], ignore_index=True)
    e = add_beforeinfo(e, allbi)
    r = e[(e["date"] == a.date) & (e["jcd"] == a.jcd) & (e["rno"] == a.rno)].sort_values("boat")
    if len(r) == 0:
        print("該当レースなし"); return
    has_bi = r["bi_exh"].notna().any()
    X = np.full((1, 6, len(F)), np.nan); mask = np.zeros((1, 6), bool)
    X[0, r["boat"].values - 1] = r[F].values.astype(float); mask[0, r["boat"].values - 1] = True
    s = [lgb.Booster(model_file=f"{a.model}/stage{k}.txt").predict(X.reshape(-1, len(F)), raw_score=True).reshape(-1, 6) for k in (1, 2, 3)]
    T = stage_trifecta(*s, mask)
    iso = json.load(open(f"{a.model}/isotonic.json"))
    P = np.interp(T.ravel(), iso["x"], iso["y"]).reshape(T.shape); P /= P.sum(1, keepdims=True)
    win = np.zeros(6)
    for ci, c in enumerate(COMBOS):
        win[int(c[0]) - 1] += P[0, ci]
    print(f"{a.date} 場{a.jcd} {a.rno}R 締切 {r['deadline'].iloc[0]}  直前情報: {'あり（締切6分前）' if has_bi else 'なし'}")
    print(r[["boat", "name", "cls", "nat_win", "loc_win", "motor_2r", "exh_time", "bi_ex_course", "bi_ex_st"]].assign(win_prob=win[r["boat"].values - 1].round(3)).to_string(index=False))
    lv = [pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in glob.glob(f"{a.store}/live/odds3t/{a.date}*.csv.gz")]
    O = None
    if lv:
        lv = pd.concat(lv); lv["jcd"] = lv["jcd"].str.zfill(2)
        row = lv[(lv["jcd"] == a.jcd) & (lv["rno"] == a.rno) & (lv["fetched_at"] < lv["deadline"] + ":00")]
        if len(row):
            O = row[COMBOS].iloc[-1].to_numpy(float); print("6分前オッズ取得時刻:", row["fetched_at"].iloc[-1])
    order = np.argsort(-P[0])[:12]
    out = []
    for ci in order:
        d = {"組番": COMBOS[ci], "モデル確率": round(P[0, ci], 4)}
        if O is not None and np.isfinite(O[ci]):
            inv = np.where(np.isfinite(O) & (O > 0), 1 / O, 0); pk = inv / inv.sum()
            d.update({"6分前オッズ": O[ci], "市場の確率": round(pk[ci], 4), "EV": round(P[0, ci] * O[ci], 2)})
        out.append(d)
    print(pd.DataFrame(out).to_string(index=False))

if __name__ == "__main__":
    main()
