"""着順予測モデルの学習 → 期待値ベースのバックテスト。
usage: python scripts/train_backtest.py DATADIR OUTDIR
期間: 学習 〜2025-09 / 検証(γ・ブレンド・戦略選択) 2025-10〜2026-03 / テスト 2026-04〜
賭け方: 3連単。EV = π × オッズ が閾値以上の組番を、1レース上限1000円（100円単位）で購入。
払戻は公式Kファイルの3連単払戻金。F/L/欠場艇を含む組番は返還。
"""
import json, os, sys
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS, history_features, race_features
from boatlib.model import blend, fit_blend, fit_gamma, softmax_rows, trifecta
from boatlib.parse import COMBOS

TRAIN_END = os.environ.get("TRAIN_END", "20250930")
VALID_END = os.environ.get("VALID_END", "20260331")
CAP = 1000
BANKROLL = 100_000  # Kelly 計算用の想定資金

def race_matrix(e, races):
    """艇単位 -> レース単位 (n,6) の配列"""
    key = ["date", "jcd", "rno"]
    r = races[key + ["tri_combo", "tri_pay"]].drop_duplicates(key).copy()
    r = r[r["tri_combo"].notna()].reset_index(drop=True)
    r["ri"] = np.arange(len(r))
    m = e.merge(r[key + ["ri"]], on=key)
    return r, m

def main(datadir, outdir):
    os.makedirs(outdir, exist_ok=True)
    e = pd.read_parquet(f"{datadir}/entries.parquet")
    races = pd.read_parquet(f"{datadir}/races.parquet")
    odds = pd.read_parquet(f"{datadir}/odds.parquet")
    e = history_features(e)
    e = race_features(e, races)
    e["y"] = (e["pos"] == 1).astype(int)
    e["absent"] = e["pos_raw"].isin(["K0", "K1"])
    e["refund"] = e["pos_raw"].astype(str).str.match(r"^(F|L|K)")

    r, m = race_matrix(e, races)
    n = len(r)
    X = np.full((n, 6, len(FEATS)), np.nan)
    mask = np.zeros((n, 6), bool); refund = np.zeros((n, 6), bool)
    bi = m["boat"].values - 1
    X[m["ri"].values, bi] = m[FEATS].values.astype(float)
    mask[m["ri"].values, bi] = ~m["absent"].values
    refund[m["ri"].values, bi] = m["refund"].values
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    d = r["date"].to_numpy(dtype=object).astype(str)
    tr, va, te = d <= TRAIN_END, (d > TRAIN_END) & (d <= VALID_END), d > VALID_END
    # 履歴が溜まるまでの最初の2か月は学習から外す
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    tr &= d >= warm
    print("races train/valid/test", tr.sum(), va.sum(), te.sum(), flush=True)

    # 1着モデル（艇単位の二値分類、レース内 softmax）
    def flat(sel):
        idx = np.where(sel)[0]
        xx = X[idx].reshape(-1, len(FEATS)); mm = mask[idx].reshape(-1)
        yy = np.zeros((len(idx), 6)); win = np.array([int(c[0]) - 1 for c in r["tri_combo"].values[idx]])
        yy[np.arange(len(idx)), win] = 1
        return xx[mm], yy.reshape(-1)[mm]
    xtr, ytr = flat(tr); xva, yva = flat(va)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
    dtr = lgb.Dataset(xtr, ytr, feature_name=FEATS, categorical_feature=["jcd_i"])
    dva = lgb.Dataset(xva, yva, reference=dtr)
    bst = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    s = bst.predict(X.reshape(-1, len(FEATS)), raw_score=True).reshape(n, 6)
    p_win = softmax_rows(s, mask)

    win_idx = np.array([int(c[0]) - 1 for c in r["tri_combo"].values])
    def win_ll(sel):
        return -np.log(np.clip(p_win[sel, win_idx[sel]], 1e-12, None)).mean()
    (g2, g3), _ = fit_gamma(p_win[va], y[va])
    pm = trifecta(p_win, g2, g3)

    # 締切時オッズ
    o = r[["date", "jcd", "rno"]].merge(odds.assign(date=odds["date"].astype(str), jcd=odds["jcd"].astype(str).str.zfill(2)),
                                        on=["date", "jcd", "rno"], how="left")
    O = o[COMBOS].values.astype(float)
    has_odds = np.isfinite(O).sum(axis=1) >= 60
    inv = np.where(np.isfinite(O) & (O > 0), 1 / O, 0)
    pk = inv / np.clip(inv.sum(axis=1, keepdims=True), 1e-12, None)
    vo = va & has_odds
    (a, b), _ = fit_blend(pm[vo], pk[vo], y[vo])
    pi = blend(pm, pk, a, b)

    def tri_ll(P, sel):
        return -np.log(np.clip(P[sel, y[sel]], 1e-12, None)).mean()
    to = te & has_odds
    metrics = {
        "races": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum()),
                  "valid_with_odds": int(vo.sum()), "test_with_odds": int(to.sum())},
        "win_logloss": {"uniform": float(np.log(6)), "valid": win_ll(va), "test": win_ll(te)},
        "win_top1_acc_test": float((p_win[te].argmax(1) == win_idx[te]).mean()),
        "gamma": [g2, g3], "blend_ab": [a, b],
        "tri_logloss_test": {"model": tri_ll(pm, to), "market": tri_ll(pk, to), "blend": tri_ll(pi, to)},
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=1), flush=True)

    # ---- 期待値ベースの賭け ----
    # 組番に含まれる艇が返還対象なら返還
    ref_combo = refund[:, [int(c[0]) - 1 for c in COMBOS]] | refund[:, [int(c[2]) - 1 for c in COMBOS]] | refund[:, [int(c[4]) - 1 for c in COMBOS]]
    pay = r["tri_pay"].values.astype(float)

    def simulate(P, sel, thr, pmin, maxpts, sizing, rank="ev"):
        idx = np.where(sel)[0]
        stake = ret = 0.0; nb = hits = 0; per_day = {}
        rows = []
        for i in idx:
            ev = P[i] * O[i]
            ok = np.isfinite(ev) & (ev >= thr) & (P[i] >= pmin)
            if not ok.any():
                continue
            cand = np.where(ok)[0]
            cand = cand[np.argsort(-(ev if rank == "ev" else P[i])[cand])][:maxpts]
            if sizing == "flat":
                yen = np.full(len(cand), 100.0)
            else:  # 1/4 Kelly 相当を 100円単位、上限1000円に収める
                f = np.clip((P[i, cand] * O[i, cand] - 1) / (O[i, cand] - 1), 0, None) * 0.25
                yen = np.maximum(np.round(f * BANKROLL / 100) * 100, 100)
            if yen.sum() > CAP:
                yen = np.floor(yen * CAP / yen.sum() / 100) * 100
                yen[yen == 0] = 0
            keep = yen > 0
            cand, yen = cand[keep], yen[keep]
            if len(cand) == 0:
                continue
            st = yen.sum(); rt = 0.0
            for c, yy in zip(cand, yen):
                if ref_combo[i, c]:
                    rt += yy
                elif c == y[i]:
                    rt += yy / 100 * pay[i]; hits += 1
            stake += st; ret += rt; nb += 1
            per_day[d[i]] = per_day.get(d[i], 0) + rt - st
            rows.append((d[i], r["jcd"].values[i], int(r["rno"].values[i]), int(st), rt))
        cum = np.cumsum([per_day[k] for k in sorted(per_day)]) if per_day else np.array([0.0])
        dd = float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max())
        return dict(thr=thr, pmin=pmin, maxpts=maxpts, sizing=sizing, races_bet=nb, hits=hits,
                    stake=int(stake), ret=int(ret), pnl=int(ret - stake), roi=ret / stake if stake else 0.0,
                    max_drawdown=int(dd)), rows

    grid = [(thr, pmin, mp, sz) for thr in (1.0, 1.1, 1.2, 1.3, 1.5, 2.0) for pmin in (0.0, 0.005, 0.01, 0.02)
            for mp in (3, 10) for sz in ("flat", "kelly")]
    results = {}
    for name, P in (("blend", pi), ("model", pm)):
        val = [simulate(P, vo, *g)[0] for g in grid]
        val = [v for v in val if v["races_bet"] >= int(os.environ.get("MIN_BET_RACES", 300))]
        best = max(val, key=lambda v: v["roi"]) if val else None
        tst = simulate(P, to, best["thr"], best["pmin"], best["maxpts"], best["sizing"]) if best else (None, [])
        results[name] = {"valid_grid_top5": sorted(val, key=lambda v: -v["roi"])[:5], "valid_best": best, "test": tst[0]}
        if tst[1]:
            pd.DataFrame(tst[1], columns=["date", "jcd", "rno", "stake", "return"]).to_csv(f"{outdir}/test_bets_{name}.csv", index=False)
    # 参考: EVを見ずに確率上位10点を毎レース100円ずつ（day1 と同じ買い方）
    results["baseline_top10_prob"] = {nm: simulate(P, to, 0.0, 0.0, 10, "flat", rank="prob")[0]
                                      for nm, P in (("model", pm), ("market", pk), ("blend", pi))}
    metrics["backtest"] = results
    imp = pd.Series(bst.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    metrics["feature_gain_top"] = {k: round(float(v), 1) for k, v in imp.head(15).items()}
    json.dump(metrics, open(f"{outdir}/metrics.json", "w"), ensure_ascii=False, indent=1, default=float)
    bst.save_model(f"{outdir}/model_win.txt")
    print(json.dumps(results, ensure_ascii=False, indent=1, default=float))

if __name__ == "__main__":
    main(*sys.argv[1:3])
