"""撤退判定と同じ評価用レースを読み込む（IDEA-002・005・006 R2 で共通に使う）。
選び方は scripts/exit_check.py（事前登録 v4）と同じ：
- live で締切前に取得した6分前の3連単オッズ、締切前に記録された予想（model==モデル名、has_bi=1）、
  締切前に取得した直前情報（展示タイムあり）、締切時オッズ・着順がそろうレース
- 6分前・締切時ともオッズが60組以上あるレース
- 日付・締切時刻・場・レース番号の順に並べる
extra_leads に "t15"・"t10" を渡すと、その時点の3連単オッズもそろうレースに絞る（IDEA-002 用）。
exit_check.py は事前登録のコードなので書き換えず、同じ選び方をここに写している（check_same_as_exit で一致を確認できる）。
"""
import glob
import numpy as np
import pandas as pd

from boatlib.parse import COMBOS

KEY = ["date", "jcd", "rno"]
LEAD_DIR = {"t6": "live", "t10": "live_t10", "t15": "live_t15"}

def _read(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs], ignore_index=True)
    df["jcd"] = df["jcd"].str.zfill(2)
    return df

def market(O):
    inv = np.where(np.isfinite(O) & (O > 0), 1 / O, 0)
    return inv / np.clip(inv.sum(1, keepdims=True), 1e-12, None)

def odds_before_deadline(store, lead):
    o = _read(f"{store}/{LEAD_DIR[lead]}/odds3t/*.csv.gz")
    return o[o["fetched_at"] < o["deadline"] + ":00"].drop_duplicates(KEY, keep="last")

def load(datadir, store, model_name="frozen_v4", extra_leads=()):
    """返り値: dict(r=レース表(date,jcd,rno,deadline,tri_combo), P=予想, L={lead: オッズ}, C=締切時オッズ, y=正解の組番の位置)"""
    live = odds_before_deadline(store, "t6")
    lp = _read(f"{store}/live_pred/*.csv.gz")
    lp = lp[(lp["model"] == model_name) & (lp["has_bi"] == 1) & (lp["predicted_at"] < lp["deadline"] + ":00")].drop_duplicates(KEY, keep="first")
    lbi = _read(f"{store}/live/beforeinfo/*.csv.gz")
    lbi = lbi[(lbi["fetched_at"] < lbi["deadline"] + ":00") & lbi["exh1"].notna()].drop_duplicates(KEY, keep="last")
    live = live.merge(lp[KEY], on=KEY).merge(lbi[KEY], on=KEY)
    close = pd.read_parquet(f"{datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    races = pd.read_parquet(f"{datadir}/races.parquet")
    r = races[KEY + ["tri_combo"]].drop_duplicates(KEY)
    r = r[r["tri_combo"].notna()].merge(live[KEY + ["deadline"]], on=KEY)
    L = {"t6": r[KEY].merge(live, on=KEY, how="left")[COMBOS].to_numpy(float)}
    C = r[KEY].merge(close, on=KEY, how="left")[COMBOS].to_numpy(float)
    ok = (np.isfinite(L["t6"]).sum(1) >= 60) & (np.isfinite(C).sum(1) >= 60)
    for lead in extra_leads:
        L[lead] = r[KEY].merge(odds_before_deadline(store, lead), on=KEY, how="left")[COMBOS].to_numpy(float)
        ok &= np.isfinite(L[lead]).sum(1) >= 60
    r, C, L = r[ok].reset_index(drop=True), C[ok], {k: v[ok] for k, v in L.items()}
    order = np.lexsort((r["rno"].values, r["jcd"].values, r["deadline"].values, r["date"].values))
    r, C, L = r.iloc[order].reset_index(drop=True), C[order], {k: v[order] for k, v in L.items()}
    P = r[KEY].merge(lp, on=KEY, how="left")[COMBOS].to_numpy(float)
    P = P / P.sum(1, keepdims=True)
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    return dict(r=r, P=P, L=L, C=C, y=y)

def head(d, n):
    return dict(r=d["r"].iloc[:n].reset_index(drop=True), P=d["P"][:n], L={k: v[:n] for k, v in d["L"].items()}, C=d["C"][:n], y=d["y"][:n])

def nll(Q, y):
    return -np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, None))

def day_boot_ci(diff, days, n_boot=5000, seed=0):
    """レースごとの差の平均の95%信頼区間（日単位のブートストラップ）"""
    dd = pd.Series(diff).groupby(np.asarray(days)).agg(["sum", "count"])
    bi = np.random.default_rng(seed).integers(0, len(dd), size=(n_boot, len(dd)))
    boot = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
    return [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
