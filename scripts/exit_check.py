"""撤退判定（PREREGISTRATION.md の判定1）。設定はコミット済みのものから変えない。
usage: python scripts/exit_check.py DATADIR STOREDIR [--model models/frozen_v4] [--n 2000] [--out results/exit_check.md]
事前登録 v4 以降: モデルの確率は live が締切前に記録した予想（STOREDIR/live_pred、model がこのモデル名、has_bi=1、
predicted_at < 締切）をそのまま使う。記録のないレースは対象外（--recompute で v3 までの再計算方式）。
モデルの特徴量の版は models/<版>/meta.json の feats_name で決まる（bi = 直前情報入り）。
bi の版では、対象レースの直前情報に live で締切前に取得したもの（live/beforeinfo）を使い、ないレースは対象から外す。

- 対象: live で締切前に取得した6分前の3連単オッズがあり、締切時オッズと着順もそろうレース。
  日付・締切時刻の順に並べ、先頭から n レース（既定 2,000）。足りなければ「判定前」と出して終わる。
- 予測: 凍結モデル（--model、既定 models/frozen_v4）の3連単確率（Isotonic 校正後、レース内で正規化）
- モデル×市場（6分前）: π ∝ P_model^a × P_6分前^b。P_6分前 は6分前オッズの逆数をレース内で正規化。
  a, b は前半 n/2 レースで最尤推定し、後半 n/2 レースで評価する。
- 判定: 後半の3連単対数損失で「モデル×市場（6分前）」が「締切時オッズの市場」を下回らなければ（≥）、3連単EV戦略は打ち切り。
  比較対象は締切時の市場（6分前の市場ではない）。
"""
import argparse, glob, json, os, sys
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS, FEATS_BI, FEATS_C1, add_beforeinfo, history_features, race_features
from boatlib.model import blend, fit_blend
from boatlib.parse import COMBOS
from exp_trifecta import stage_trifecta

RNG = np.random.default_rng(0)

def market(O):
    inv = np.where(np.isfinite(O) & (O > 0), 1 / O, 0)
    return inv / np.clip(inv.sum(1, keepdims=True), 1e-12, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--model", default="models/frozen_v4")
    ap.add_argument("--recompute", action="store_true", help="記録された予想ではなく、データから予想を計算し直す（v3 までの方式）"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/exit_check.md")
    a = ap.parse_args()
    key = ["date", "jcd", "rno"]
    live = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in sorted(glob.glob(f"{a.store}/live/odds3t/*.csv.gz"))])
    live["jcd"] = live["jcd"].str.zfill(2)
    live = live[live["fetched_at"] < live["deadline"] + ":00"].drop_duplicates(key, keep="last")  # 締切前の取得のみ
    meta = json.load(open(f"{a.model}/meta.json"))
    model_name = a.model.rstrip("/").split("/")[-1]
    lp = None
    if not a.recompute:
        fs = sorted(glob.glob(f"{a.store}/live_pred/*.csv.gz"))
        lp = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs]) if fs else pd.DataFrame(columns=key + ["deadline", "predicted_at", "model", "has_bi"] + COMBOS)
        lp["jcd"] = lp["jcd"].astype(str).str.zfill(2)
        lp = lp[(lp["model"] == model_name) & (lp["has_bi"] == 1) & (lp["predicted_at"] < lp["deadline"] + ":00")].drop_duplicates(key, keep="first")
        live = live.merge(lp[key], on=key)  # 締切前に記録された予想があるレースだけ
    F = {"base": FEATS, "bi": FEATS_BI, "c1": FEATS_C1}[meta.get("feats_name", "base")]
    if F is not FEATS:
        lbi = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in sorted(glob.glob(f"{a.store}/live/beforeinfo/*.csv.gz"))])
        lbi["jcd"] = lbi["jcd"].str.zfill(2)
        lbi = lbi[(lbi["fetched_at"] < lbi["deadline"] + ":00") & lbi["exh1"].notna()].drop_duplicates(key, keep="last")
        live = live.merge(lbi[key], on=key)  # 締切前の直前情報があるレースだけ
    close = pd.read_parquet(f"{a.datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    races = pd.read_parquet(f"{a.datadir}/races.parquet")
    r = races[key + ["tri_combo"]].drop_duplicates(key)
    r = r[r["tri_combo"].notna()].merge(live[key + ["deadline"]], on=key)
    L = r[key].merge(live, on=key, how="left")[COMBOS].to_numpy(float)
    C = r[key].merge(close, on=key, how="left")[COMBOS].to_numpy(float)
    ok = (np.isfinite(L).sum(1) >= 60) & (np.isfinite(C).sum(1) >= 60)
    r, L, C = r[ok].reset_index(drop=True), L[ok], C[ok]
    order = np.lexsort((r["rno"].values, r["jcd"].values, r["deadline"].values, r["date"].values))
    r, L, C = r.iloc[order].reset_index(drop=True), L[order], C[order]
    n_avail = len(r)
    if n_avail < a.n:
        msg = f"判定前: 対象レース {n_avail} / {a.n}（{r['date'].nunique() if n_avail else 0}日）"
        open(a.out, "w").write(msg + "\n"); print(msg); return
    r, L, C = r.iloc[:a.n].reset_index(drop=True), L[:a.n], C[:a.n]
    if lp is not None:
        # 締切前に記録された予想をそのまま使う
        P = r[key].merge(lp, on=key, how="left")[COMBOS].to_numpy(float)
        P = P / P.sum(1, keepdims=True)
    else:
        # 凍結モデルで予測し直す（v3 までの方式）
        e = race_features(history_features(pd.read_parquet(f"{a.datadir}/entries.parquet")))
        if F is FEATS_BI:
            # 過去レースはデータセットの直前情報、対象レースは live で締切前に取得した直前情報
            hb = pd.read_parquet(f"{a.datadir}/beforeinfo.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
            hb = hb.merge(r[key], on=key, how="left", indicator=True)
            hb = hb[hb["_merge"] == "left_only"].drop(columns="_merge")
            e = add_beforeinfo(e, pd.concat([hb, lbi.drop(columns=["deadline", "fetched_at"])], ignore_index=True))
        e["absent"] = e["pos_raw"].isin(["K0", "K1"])
        r["ri"] = np.arange(len(r))
        m = e.merge(r[key + ["ri"]], on=key)
        X = np.full((len(r), 6, len(F)), np.nan); mask = np.zeros((len(r), 6), bool)
        X[m["ri"].values, m["boat"].values - 1] = m[F].values.astype(float)
        mask[m["ri"].values, m["boat"].values - 1] = ~m["absent"].values
        s = [lgb.Booster(model_file=f"{a.model}/stage{k}.txt").predict(X.reshape(-1, len(F)), raw_score=True).reshape(-1, 6) for k in (1, 2, 3)]
        T = stage_trifecta(*s, mask)
        iso = json.load(open(f"{a.model}/isotonic.json"))
        P = np.interp(T.ravel(), iso["x"], iso["y"]).reshape(T.shape); P /= P.sum(1, keepdims=True)
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    pl, pc = market(L), market(C)
    h = a.n // 2
    fit, ev = np.arange(len(r)) < h, np.arange(len(r)) >= h
    (wa, wb), _ = fit_blend(P[fit], pl[fit], y[fit])
    pb = blend(P, pl, wa, wb)
    nll = lambda Q: -np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, None))
    res = {k: nll(Q) for k, Q in (("model", P), ("market_6min", pl), ("market_close", pc), ("blend_6min", pb))}
    ll = {k: float(v[ev].mean()) for k, v in res.items()}
    diff = res["blend_6min"][ev] - res["market_close"][ev]
    days = r["date"].values[ev]; dd = pd.Series(diff).groupby(days).agg(["sum", "count"])
    bi = RNG.integers(0, len(dd), size=(5000, len(dd)))
    boot = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
    stop = ll["blend_6min"] >= ll["market_close"]
    out = dict(races=a.n, fit_races=int(fit.sum()), eval_races=int(ev.sum()),
               period=[str(r["date"].min()), str(r["date"].max())], blend_ab=[float(wa), float(wb)],
               logloss_eval=ll, diff_blend_minus_close=float(diff.mean()),
               diff_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
               decision="打ち切り（3連単EV戦略）" if stop else "継続（判定1を通過）")
    md = ["### 撤退判定（判定1）", "```", json.dumps(out, ensure_ascii=False, indent=1), "```"]
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
