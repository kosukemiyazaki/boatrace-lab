"""実験A: 3連単確率モデルの比較（オッズ不要。開発期間のみ使用）
usage: python scripts/exp_trifecta.py DATADIR [variant ...]
  学習 〜2025-12 / 早期終了 2026-01〜02 / 評価 2026-03〜09（校正は評価の前半 03〜05、指標は後半 06〜09）
variant:
  harville : 1着モデル + べき乗Harville（現行）
  stage    : 1着・2着・3着を段階ごとの条件付きロジット（LightGBM）で予測
"""
import json, os, sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS, FEATS2, extra_features, history_features, race_features
from boatlib.model import PERMS, fit_gamma, softmax_rows, trifecta
from boatlib.parse import COMBOS

TR, ES, CA, EV = "20251231", "20260228", "20260531", "20260930"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)

def load(datadir, feats):
    e = pd.read_parquet(f"{datadir}/entries.parquet")
    races = pd.read_parquet(f"{datadir}/races.parquet")
    e = race_features(history_features(e))
    if feats is FEATS2:
        e = extra_features(e)
    e["absent"] = e["pos_raw"].isin(["K0", "K1"])
    key = ["date", "jcd", "rno"]
    r = races[key + ["tri_combo"]].drop_duplicates(key)
    r = r[r["tri_combo"].notna()].reset_index(drop=True)
    r["ri"] = np.arange(len(r))
    m = e.merge(r[key + ["ri"]], on=key)
    n = len(r)
    X = np.full((n, 6, len(feats)), np.nan); mask = np.zeros((n, 6), bool)
    bi = m["boat"].values - 1
    X[m["ri"].values, bi] = m[feats].values.astype(float)
    mask[m["ri"].values, bi] = ~m["absent"].values
    pos = np.array([[int(c[0]) - 1, int(c[2]) - 1, int(c[4]) - 1] for c in r["tri_combo"].values])
    return r, X, mask, pos

def stage_model(X, mask, pos, stage, tr, es, feats):
    """stage=0,1,2: それより上位で決まった艇を除いた中で stage 着になる艇を当てる"""
    n = len(X)
    avail = mask.copy()
    for s in range(stage):
        avail[np.arange(n), pos[:, s]] = False
    tgt = np.zeros((n, 6)); tgt[np.arange(n), pos[:, stage]] = 1
    def flat(sel):
        a = avail[sel].reshape(-1)
        return X[sel].reshape(-1, len(feats))[a], tgt[sel].reshape(-1)[a]
    xtr, ytr = flat(tr); xes, yes = flat(es)
    dtr = lgb.Dataset(xtr, ytr, feature_name=feats, categorical_feature=["jcd_i"])
    bst = lgb.train(PARAMS, dtr, 3000, valid_sets=[lgb.Dataset(xes, yes, reference=dtr)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    return bst.predict(X.reshape(-1, len(feats)), raw_score=True).reshape(n, 6), bst.best_iteration

def stage_trifecta(s1, s2, s3, mask):
    n = len(s1)
    e1, e2, e3 = [np.where(mask, np.exp(s - s.max(1, keepdims=True)), 0) for s in (s1, s2, s3)]
    PI = np.array([p[0] for p in PERMS]); PJ = np.array([p[1] for p in PERMS]); PK = np.array([p[2] for p in PERMS])
    a = e1[:, PI] / e1.sum(1, keepdims=True)
    b = e2[:, PJ] / np.clip(e2.sum(1, keepdims=True) - e2[:, PI], 1e-12, None)
    c = e3[:, PK] / np.clip(e3.sum(1, keepdims=True) - e3[:, PI] - e3[:, PJ], 1e-12, None)
    t = a * b * c
    return t / t.sum(1, keepdims=True)

def evaluate(T, y, sel_cal, sel_ev, name):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    yt = np.zeros_like(T, dtype=bool); yt[np.arange(len(y)), y] = True
    iso.fit(T[sel_cal].ravel(), yt[sel_cal].ravel())
    Tc = iso.predict(T.ravel()).reshape(T.shape); Tc = Tc / Tc.sum(1, keepdims=True)
    ll_raw = float(-np.log(np.clip(T[sel_ev, y[sel_ev]], 1e-12, None)).mean())
    ll_cal = float(-np.log(np.clip(Tc[sel_ev, y[sel_ev]], 1e-12, None)).mean())
    top1 = float((T[sel_ev].argmax(1) == y[sel_ev]).mean())
    out = dict(variant=name, tri_ll_raw=ll_raw, tri_ll_cal=ll_cal, tri_top1_hit=top1, uniform=float(np.log(120)))
    print(json.dumps(out), flush=True)
    return out

def main(datadir, *variants):
    variants = variants or ("harville", "stage")
    FE = FEATS2 if "v2" in variants else FEATS
    r, X, mask, pos = load(datadir, FE)
    d = r["date"].to_numpy(dtype=object).astype(str)
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    tr = (d >= warm) & (d <= TR); es = (d > TR) & (d <= ES); ca = (d > ES) & (d <= CA); ev = (d > CA) & (d <= EV)
    print("races", tr.sum(), es.sum(), ca.sum(), ev.sum(), flush=True)
    s1, it1 = stage_model(X, mask, pos, 0, tr, es, FE)
    res = []
    if "harville" in variants:
        p1 = softmax_rows(s1, mask)
        (g2, g3), _ = fit_gamma(p1[ca], y[ca])
        res.append(evaluate(trifecta(p1, g2, g3), y, ca, ev, f"harville(g={g2:.2f},{g3:.2f})"))
    if "stage" in variants:
        s2, it2 = stage_model(X, mask, pos, 1, tr, es, FE)
        s3, it3 = stage_model(X, mask, pos, 2, tr, es, FE)
        res.append(evaluate(stage_trifecta(s1, s2, s3, mask), y, ca, ev, f"stage{'-v2' if FE is FEATS2 else ''}(it={it1},{it2},{it3})"))
    return res

if __name__ == "__main__":
    main(*sys.argv[1:])
