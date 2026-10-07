"""撤退判定（PREREGISTRATION.md）用のモデルを学習して保存する。保存後は再学習しない。
usage: python scripts/freeze_model.py DATADIR OUTDIR [--feats base|bi|c1]
--feats c1: bi に節間成績（今節の出走数・平均着順・1着数・平均ST・展示タイムの平均順位）を加えた版（IDEA-004 の C1）
--feats bi: 直前情報（展示進入・展示ST・チルト・部品交換・直前気象、展示タイムのレース内相対値と普段との差）を加えた版。
  直前情報は 2025-10 以降のみ。それ以前は欠損のまま学習する（LightGBM は欠損を扱える）
- 1〜3着の段階別条件付きロジット（LightGBM, 特徴量 FEATS）
- 学習 〜2026-06 / 早期終了 2026-07 / 3連単確率の Isotonic 校正 2026-08（2026-09以降は使わない）
- 保存: stage{1,2,3}.txt（LightGBM）、isotonic.json（校正の折れ線）、meta.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from walkforward import load
from exp_trifecta import stage_model, stage_trifecta
from boatlib.features import FEATS, FEATS_BI, FEATS_C1
from boatlib.parse import COMBOS

TRAIN_END, ES_MONTH, CAL_MONTH = "20260630", "202607", "202608"

def main(datadir, out, feats_name="base"):
    os.makedirs(out, exist_ok=True)
    F = {"base": FEATS, "bi": FEATS_BI, "c1": FEATS_C1}[feats_name]
    r, X, mask, refund, pos, O = load(datadir, F)  # load は 2026-09 以降を除外する
    d = r["date"].to_numpy(dtype=object).astype(str); ym = np.array([x[:6] for x in d])
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    tr = (d >= warm) & (d <= TRAIN_END); es = ym == ES_MONTH; ca = ym == CAL_MONTH
    scores, its = [], []
    for k in range(3):
        s, it, bst = stage_model(X, mask, pos, k, tr, es, F)
        bst.save_model(f"{out}/stage{k+1}.txt", num_iteration=bst.best_iteration)
        scores.append(s); its.append(it)
    T = stage_trifecta(*scores, mask)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
    yt = np.zeros((ca.sum(), 120), bool); yt[np.arange(ca.sum()), y[ca]] = True
    iso.fit(T[ca].ravel(), yt.ravel())
    json.dump({"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}, open(f"{out}/isotonic.json", "w"))
    P = iso.predict(T[ca].ravel()).reshape(-1, 120); P /= P.sum(1, keepdims=True)
    meta = dict(feats_name=feats_name, features=F, train=[warm, TRAIN_END], early_stopping=ES_MONTH, calibration=CAL_MONTH,
                best_iterations=its, train_races=int(tr.sum()), cal_races=int(ca.sum()),
                cal_tri_logloss=float(-np.log(P[np.arange(ca.sum()), y[ca]]).mean()))
    json.dump(meta, open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("out"); ap.add_argument("--feats", default="base", choices=["base", "bi", "c1"])
    a = ap.parse_args()
    main(a.datadir, a.out, a.feats)
