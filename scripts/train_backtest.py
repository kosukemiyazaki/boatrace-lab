"""着順予測モデルの学習 → 校正 → 期待値ベースの仮想投票バックテスト。
usage: python scripts/train_backtest.py DATADIR OUTDIR

検証ルール
- 特徴量は締切前に確定する情報のみ（boatlib/features.py。当該レースの ST・進入・着順・風波は不使用。
  過去レースの成績は当該レースより前のものだけ）
- 時系列分割（未来で学習しない）:
    学習        〜2025-07
    早期終了    2025-08〜09
    校正        2025-10〜12   (1着確率 Isotonic → Harville γ → 3連単確率 Isotonic)
    戦略選択    2026-01〜03   (EV閾値・最低確率・点数・資金配分を選ぶ)
    テスト      2026-04〜     (最後に1回だけ評価)
- 買うのは 校正済み確率 × オッズ > 閾値 の3連単だけ。1レース上限1000円（100円単位）
- 仮想投票のみ。払戻は公式Kファイルの3連単払戻金、F/L/欠場艇を含む組番は返還
- オッズは締切時オッズ（過去分で取れるのはこれだけ）。締切前の実際の判断時とはずれるので、
  判断に使うオッズを1割下げたストレステストも出す
比較: LightGBM と ロジスティック回帰（ベースライン）で同じパイプライン
"""
import json, os, sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS, history_features, race_features
from boatlib.model import fit_gamma, softmax_rows, trifecta
from boatlib.parse import COMBOS

SPLITS = dict(train_end="20250731", es_end="20250930", cal_end="20251231", sel_end="20260331")
SPLITS = {k: os.environ.get(k.upper(), v) for k, v in SPLITS.items()}
CAP = 1000
BANKROLL = 100_000  # Kelly 計算用の想定資金
MIN_SEL_RACES = int(os.environ.get("MIN_SEL_RACES", 500))
RNG = np.random.default_rng(0)
RUN_TEST = os.environ.get("RUN_TEST", "0") == "1"  # テスト期間は明示したときだけ評価する

def load(datadir):
    e = pd.read_parquet(f"{datadir}/entries.parquet")
    races = pd.read_parquet(f"{datadir}/races.parquet")
    odds = pd.read_parquet(f"{datadir}/odds.parquet")
    e = race_features(history_features(e))
    e["absent"] = e["pos_raw"].isin(["K0", "K1"])
    e["refund"] = e["pos_raw"].astype(str).str.match(r"^(F|L|K)")
    key = ["date", "jcd", "rno"]
    r = races[key + ["tri_combo", "tri_pay"]].drop_duplicates(key)
    r = r[r["tri_combo"].notna()].reset_index(drop=True)
    r["ri"] = np.arange(len(r))
    m = e.merge(r[key + ["ri"]], on=key)
    n = len(r)
    X = np.full((n, 6, len(FEATS)), np.nan)
    mask = np.zeros((n, 6), bool); refund = np.zeros((n, 6), bool)
    bi = m["boat"].values - 1
    X[m["ri"].values, bi] = m[FEATS].values.astype(float)
    mask[m["ri"].values, bi] = ~m["absent"].values
    refund[m["ri"].values, bi] = m["refund"].values
    odds = odds.assign(date=odds["date"].astype(str), jcd=odds["jcd"].astype(str).str.zfill(2))
    O = r[key].merge(odds, on=key, how="left")[COMBOS].values.astype(float)
    O[~(O > 0)] = np.nan
    return r, X, mask, refund, O

def main(datadir, outdir):
    os.makedirs(outdir, exist_ok=True)
    r, X, mask, refund, O = load(datadir)
    n = len(r)
    d = r["date"].to_numpy(dtype=object).astype(str)
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    win = np.array([int(c[0]) - 1 for c in r["tri_combo"].values])
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")  # 履歴が溜まるまで除外
    has_odds = np.isfinite(O).sum(axis=1) >= 60
    S = dict(
        train=(d >= warm) & (d <= SPLITS["train_end"]),
        es=(d > SPLITS["train_end"]) & (d <= SPLITS["es_end"]),
        cal=(d > SPLITS["es_end"]) & (d <= SPLITS["cal_end"]),
        sel=(d > SPLITS["cal_end"]) & (d <= SPLITS["sel_end"]) & has_odds,
        test=(d > SPLITS["sel_end"]) & has_odds,
    )
    info = {"splits": SPLITS, "races": {k: int(v.sum()) for k, v in S.items()},
            "days": {k: int(len(set(d[v]))) for k, v in S.items()}}
    print(json.dumps(info), flush=True)
    nd = {k: len(set(d[v])) for k, v in S.items()}

    def flat(sel):
        idx = np.where(sel)[0]
        yy = np.zeros((len(idx), 6)); yy[np.arange(len(idx)), win[idx]] = 1
        mm = mask[idx].reshape(-1)
        return X[idx].reshape(-1, len(FEATS))[mm], yy.reshape(-1)[mm]

    xtr, ytr = flat(S["train"]); xes, yes = flat(S["es"])
    scores = {}
    # --- LightGBM ---
    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
    dtr = lgb.Dataset(xtr, ytr, feature_name=FEATS, categorical_feature=["jcd_i"])
    bst = lgb.train(params, dtr, 3000, valid_sets=[lgb.Dataset(xes, yes, reference=dtr)],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    scores["lgbm"] = bst.predict(X.reshape(-1, len(FEATS)), raw_score=True).reshape(n, 6)
    bst.save_model(f"{outdir}/model_lgbm.txt")
    # --- ロジスティック回帰（ベースライン）: 欠損は学習期間の中央値、艇番・級別・場はone-hot ---
    med = np.nanmedian(xtr, axis=0)
    def lr_x(x):
        x = np.where(np.isnan(x), med, x)
        fi = {f: i for i, f in enumerate(FEATS)}
        oh = [(x[:, fi["boat"]] == b).astype(float) for b in range(1, 7)]
        oh += [(x[:, fi["jcd_i"]] == j).astype(float) for j in range(1, 25)]
        num = np.delete(x, [fi["boat"], fi["jcd_i"]], axis=1)
        return np.column_stack([num] + oh)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    lr.fit(lr_x(xtr), ytr)
    scores["logreg"] = lr.decision_function(lr_x(X.reshape(-1, len(FEATS)))).reshape(n, 6)

    ref_combo = np.zeros((n, 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref_combo[:, ci] |= refund[:, int(b) - 1]
    pay = r["tri_pay"].values.astype(float)

    report = {"info": info, "models": {}}
    for name, s in scores.items():
        # 1) 1着確率: レース内softmax → Isotonic(校正期間) → 再正規化
        p0 = softmax_rows(s, mask)
        iso_w = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1.0)
        cm = S["cal"][:, None] & mask
        yw = np.zeros((n, 6)); yw[np.arange(n), win] = 1
        iso_w.fit(p0[cm], yw[cm])
        p1 = np.where(mask, iso_w.predict(p0.ravel()).reshape(n, 6), 0)
        p1 = p1 / p1.sum(axis=1, keepdims=True)
        # 2) 3連単: Harville γ（校正期間で最尤）→ Isotonic（校正期間, 組番単位）
        (g2, g3), _ = fit_gamma(p1[S["cal"]], y[S["cal"]])
        t0 = trifecta(p1, g2, g3)
        yt = np.zeros((n, 120), bool); yt[np.arange(n), y] = True
        iso_t = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_t.fit(t0[S["cal"]].ravel(), yt[S["cal"]].ravel())
        P = iso_t.predict(t0.ravel()).reshape(n, 120)

        te = S["test"] if RUN_TEST else S["sel"]
        mres = {"evaluated_on": "test" if RUN_TEST else "sel (テスト未実施)",
            "win_logloss": {k: float(-np.log(np.clip(p1[S[k], win[S[k]]], 1e-12, None)).mean()) for k in ("cal", "sel") + (("test",) if RUN_TEST else ())},
            "win_logloss_uniform": float(np.log(6)),
            "win_top1_acc_test": float((p1[te].argmax(1) == win[te]).mean()),
            "tri_logloss_test": float(-np.log(np.clip(P[te, y[te]] / P[te].sum(1), 1e-12, None)).mean()),
            "gamma": [float(g2), float(g3)],
            "calibration_win_test": reliability(p1[te][mask[te]], yw[te][mask[te]]),
            "calibration_tri_test": reliability(P[te].ravel(), yt[te].ravel(), bins=[0, .005, .01, .02, .04, .07, .1, .15, .25, 1]),
        }
        # 3) 戦略選択（選択期間のみ）。高配当頼みの偶然の当たりを選ばないよう、
        #    回収率そのものではなく「日単位ブートストラップの下側5%点」が最大の設定を選ぶ
        grid = [dict(thr=t, pmin=pm, maxpts=mp, sizing=sz) for t in (1.0, 1.1, 1.2, 1.3, 1.5, 2.0)
                for pm in (0.0, 0.005, 0.01, 0.02, 0.04) for mp in (1, 3, 10) for sz in ("flat", "kelly")]
        sel_rows = []
        for g in grid:
            b = bets(P, O, O, S["sel"], **g)
            sm = settle(b, y, pay, ref_combo, d, O, nd["sel"], boot=True, nboot=2000)
            if sm["races_bet"] >= MIN_SEL_RACES:
                sel_rows.append({**g, **{k: sm[k] for k in ("races_bet", "hits", "roi", "roi_q05", "pnl", "mean_odds_bought")}})
        sel_rows.sort(key=lambda v: -v["roi_q05"])
        best = {k: sel_rows[0][k] for k in ("thr", "pmin", "maxpts", "sizing")} if sel_rows else None
        mres["selection_top5"] = sel_rows[:5]
        mres["chosen_strategy"] = best
        if best:
            sm_sel = settle(bets(P, O, O, S["sel"], **best), y, pay, ref_combo, d, O, nd["sel"], boot=True, calib=P)
            mres["selection_period"] = sm_sel
        if best and RUN_TEST:
            # 4) テスト（ここで初めて触る）
            bt = bets(P, O, O, te, **best)
            mres["test"] = settle(bt, y, pay, ref_combo, d, O, nd["test"], boot=True, calib=P)
            pd.DataFrame(bt, columns=["ri", "combo_i", "yen"]).assign(
                date=lambda x: d[x.ri], jcd=lambda x: r["jcd"].values[x.ri], rno=lambda x: r["rno"].values[x.ri],
                combo=lambda x: [COMBOS[i] for i in x.combo_i], prob=lambda x: P[x.ri, x.combo_i],
                odds=lambda x: O[x.ri, x.combo_i], hit=lambda x: y[x.ri] == x.combo_i,
            ).drop(columns=["ri", "combo_i"]).to_csv(f"{outdir}/test_bets_{name}.csv", index=False)
            # ストレステスト: 判断に使うオッズを1割下げる（払戻は実際の払戻金）
            mres["test_stress_odds90"] = settle(bets(P, O * 0.9, O, te, **best), y, pay, ref_combo, d, O, nd["test"], boot=True)
        report["models"][name] = mres
        print(name, json.dumps({k: mres[k] for k in ("win_logloss", "tri_logloss_test", "chosen_strategy")}, default=float), flush=True)
    imp = pd.Series(bst.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    report["lgbm_feature_gain_top15"] = {k: round(float(v), 1) for k, v in imp.head(15).items()}
    json.dump(report, open(f"{outdir}/report.json", "w"), ensure_ascii=False, indent=1, default=float)

def reliability(p, yv, bins=(0, .05, .1, .2, .3, .4, .5, .6, .7, .8, 1.0)):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append({"bin": f"{lo}-{hi}", "n": int(m.sum()), "pred": float(p[m].mean()), "actual": float(yv[m].mean())})
    return out

def bets(P, Odec, Opay, sel, thr, pmin, maxpts, sizing):
    """Odec: 判断に使うオッズ。返り値 [(race_i, combo_i, yen)]"""
    out = []
    for i in np.where(sel)[0]:
        ev = P[i] * Odec[i]
        ok = np.isfinite(ev) & (ev > thr) & (P[i] >= pmin)
        if not ok.any():
            continue
        cand = np.where(ok)[0]
        cand = cand[np.argsort(-ev[cand])][:maxpts]
        if sizing == "flat":
            yen = np.full(len(cand), 100.0)
        else:  # 1/4 Kelly（想定資金10万円）を100円単位
            o = Odec[i, cand]
            f = np.clip((P[i, cand] * o - 1) / (o - 1), 0, None) * 0.25
            yen = np.maximum(np.round(f * BANKROLL / 100) * 100, 100)
        if yen.sum() > CAP:
            yen = np.floor(yen * CAP / yen.sum() / 100) * 100
        for c, yy in zip(cand, yen):
            if yy > 0:
                out.append((i, int(c), float(yy)))
    return out

def settle(b, y, pay, ref_combo, d, O, ndays, boot=False, calib=None, nboot=5000):
    if not b:
        return {"races_bet": 0}
    b = np.array(b)
    ri, ci, yen = b[:, 0].astype(int), b[:, 1].astype(int), b[:, 2]
    refund = ref_combo[ri, ci]
    hit = (y[ri] == ci) & ~refund
    ret = np.where(refund, yen, np.where(hit, yen / 100 * pay[ri], 0.0))
    days = d[ri]
    df = pd.DataFrame({"day": days, "stake": yen, "ret": ret, "ri": ri})
    per_day = df.groupby("day").agg(stake=("stake", "sum"), ret=("ret", "sum"), races=("ri", "nunique"), tickets=("ri", "size"))
    n_days_all = ndays  # 期間中の開催日数（買わなかった日も含む）
    cum = (per_day["ret"] - per_day["stake"]).cumsum().values
    dd = float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max())
    o = O[ri, ci]
    out = dict(
        races_bet=int(len(set(ri))), tickets=int(len(ri)), hits=int(hit.sum()), refunds=int(refund.sum()),
        stake=int(yen.sum()), ret=int(ret.sum()), pnl=int(ret.sum() - yen.sum()),
        roi=float(ret.sum() / yen.sum()),
        hit_rate_per_ticket=float(hit.mean()),
        days_in_period=int(n_days_all), days_with_bets=int(len(per_day)),
        races_per_day=float(len(set(ri)) / n_days_all), tickets_per_day=float(len(ri) / n_days_all),
        stake_per_day=float(yen.sum() / n_days_all),
        mean_odds_bought=float(o.mean()), median_odds_bought=float(np.median(o)),
        mean_odds_hit=float(o[hit].mean()) if hit.any() else None,
        max_drawdown=int(dd),
    )
    if calib is not None:
        # 買い目に限った校正: モデルの予測的中数 / 市場(締切オッズ逆数を正規化)の予測的中数 / 実際
        pr = calib[ri, ci]
        inv = 1 / O[ri]
        mk = inv[np.arange(len(ri)), ci] / np.nansum(inv, axis=1)
        out["expected_hits_model"] = float(pr.sum()); out["expected_hits_market"] = float(mk.sum())
        out["mean_pred_prob"] = float(pr.mean()); out["actual_hit_rate"] = float(hit.mean())
    if boot:
        # 日単位のブートストラップで回収率の95%信頼区間
        st, rt = per_day["stake"].values, per_day["ret"].values
        k = len(st)
        idx = RNG.integers(0, k, size=(nboot, k))
        rois = rt[idx].sum(1) / st[idx].sum(1)
        out["roi_q05"] = float(np.quantile(rois, 0.05))
        out["roi_ci95"] = [float(np.quantile(rois, 0.025)), float(np.quantile(rois, 0.975))]
        out["p_roi_ge_1"] = float((rois >= 1).mean())
    return out

if __name__ == "__main__":
    main(*sys.argv[1:3])
