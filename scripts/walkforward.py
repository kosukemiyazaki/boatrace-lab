"""Step 2: ウォークフォワードでの買い方の比較（設定は事前に固定。結果は全組み合わせを記録する）
usage: python scripts/walkforward.py DATADIR [--out results/walkforward]

事前に固定した設定（実行前にコミット済み。変更する場合はコミットを分けて理由を残す）
- 予測確率
  A. モデル単体: 1〜3着の段階別条件付きロジット（LightGBM, 特徴量 FEATS）→ 3連単確率 → Isotonic 校正
  B. モデル×市場（市場側は6分前オッズのみ）: 過去に6分前オッズがないため、この期間では評価できない（記録のみ）
- 期間: 評価 2024-07〜2026-08（2026-09以降はテスト用として読み込み時に除外）
  四半期ごとに学習し直す: 評価四半期の開始月を s として
    学習 〜s-3か月末 / 早期終了 s-2か月 / 校正 s-1か月 / 評価 s〜s+2か月
- 買い方（9通りのみ）: オッズ上限 {30,50,100倍} × EV閾値 {1.1,1.2,1.3}
  各レースで「締切時オッズ ≤ 上限 かつ 校正後確率×締切時オッズ > 閾値」の組番を、EVの高い順に最大10点・各100円
  （1レース上限1000円）。EV計算のオッズは締切時オッズで代用（過去分はこれしかない）
- 払戻: 公式Kファイルの3連単払戻金。F/L/欠場艇を含む組番は返還
- 指標: 回収率、日単位ブートストラップ95%信頼区間と下側5%点、四半期ごとの回収率、1日あたり購入数、平均オッズ、
  予測的中数（モデル）と実際の的中数
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from boatlib.features import FEATS, history_features, race_features
from boatlib.parse import COMBOS
from exp_trifecta import stage_model, stage_trifecta

EVAL_START, EVAL_END = "20240701", "20260831"
TEST_START = "20260901"
GRID = [dict(omax=om, thr=t) for om in (30, 50, 100) for t in (1.1, 1.2, 1.3)]
MAXPTS, YEN, CAP = 10, 100, 1000
RNG = np.random.default_rng(0)

def load(datadir):
    e = pd.read_parquet(f"{datadir}/entries.parquet")
    e = e[e["date"] < TEST_START]
    races = pd.read_parquet(f"{datadir}/races.parquet")
    races = races[races["date"] < TEST_START]
    odds = pd.read_parquet(f"{datadir}/odds.parquet")
    odds = odds.assign(date=odds["date"].astype(str), jcd=odds["jcd"].astype(str).str.zfill(2))
    odds = odds[odds["date"] < TEST_START]
    e = race_features(history_features(e))
    e["absent"] = e["pos_raw"].isin(["K0", "K1"])
    e["refund"] = e["pos_raw"].astype(str).str.match(r"^(F|L|K)")
    key = ["date", "jcd", "rno"]
    r = races[key + ["tri_combo", "tri_pay"]].drop_duplicates(key)
    r = r[r["tri_combo"].notna()].reset_index(drop=True)
    r["ri"] = np.arange(len(r))
    m = e.merge(r[key + ["ri"]], on=key)
    n = len(r)
    X = np.full((n, 6, len(FEATS)), np.nan); mask = np.zeros((n, 6), bool); refund = np.zeros((n, 6), bool)
    bi = m["boat"].values - 1
    X[m["ri"].values, bi] = m[FEATS].values.astype(float)
    mask[m["ri"].values, bi] = ~m["absent"].values
    refund[m["ri"].values, bi] = m["refund"].values
    pos = np.array([[int(c[0]) - 1, int(c[2]) - 1, int(c[4]) - 1] for c in r["tri_combo"].values])
    O = r[key].merge(odds, on=key, how="left")[COMBOS].values.astype(float)
    O[~(O > 0)] = np.nan
    return r, X, mask, refund, pos, O

def month_add(yyyymm, k):
    t = pd.Timestamp(yyyymm + "01") + pd.DateOffset(months=k)
    return t.strftime("%Y%m")

def bets_for(P, O, idx, omax, thr):
    out = []
    for i in idx:
        ev = P[i] * O[i]
        ok = np.isfinite(ev) & (O[i] <= omax) & (ev > thr)
        if not ok.any():
            continue
        c = np.where(ok)[0]
        c = c[np.argsort(-ev[c])][:MAXPTS]
        out += [(i, int(k)) for k in c]
    return out

def settle(b, P, O, y, pay, ref_combo, d, ndays, quarter_of):
    if not b:
        return dict(tickets=0)
    b = np.array(b); ri, ci = b[:, 0], b[:, 1]
    refund = ref_combo[ri, ci]
    hit = (y[ri] == ci) & ~refund
    ret = np.where(refund, YEN, np.where(hit, pay[ri] * YEN / 100, 0.0))
    df = pd.DataFrame({"day": d[ri], "q": quarter_of[ri], "stake": YEN, "ret": ret})
    per_day = df.groupby("day")[["stake", "ret"]].sum()
    st, rt = per_day["stake"].values, per_day["ret"].values
    idx = RNG.integers(0, len(st), size=(5000, len(st)))
    rois = rt[idx].sum(1) / st[idx].sum(1)
    q = df.groupby("q")[["stake", "ret"]].sum()
    o = O[ri, ci]
    return dict(
        races=int(len(set(ri))), tickets=int(len(ri)), hits=int(hit.sum()), refunds=int(refund.sum()),
        stake=int(YEN * len(ri)), ret=int(ret.sum()), roi=float(ret.sum() / (YEN * len(ri))),
        roi_ci95=[float(np.quantile(rois, .025)), float(np.quantile(rois, .975))],
        roi_q05=float(np.quantile(rois, .05)), p_roi_ge_1_1=float((rois >= 1.1).mean()),
        tickets_per_day=float(len(ri) / ndays), races_per_day=float(len(set(ri)) / ndays),
        mean_odds=float(o.mean()), median_odds=float(np.median(o)),
        expected_hits_model=float(P[ri, ci].sum()),
        roi_by_quarter={k: round(float(v.ret / v.stake), 3) for k, v in q.iterrows()},
        daily=per_day.reset_index().assign(day=lambda x: x["day"].astype(str)).values.tolist(),
    )

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/walkforward")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r, X, mask, refund, pos, O = load(a.datadir)
    d = r["date"].to_numpy(dtype=object).astype(str); ym = np.array([x[:6] for x in d])
    assert d.max() < TEST_START
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    pay = r["tri_pay"].values.astype(float)
    ref_combo = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref_combo[:, ci] |= refund[:, int(b) - 1]
    has_odds = np.isfinite(O).sum(1) >= 60
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    P = np.full((len(r), 120), np.nan)
    quarter_of = np.array([""] * len(r), dtype=object)
    qstarts = [m for m in sorted(set(ym)) if EVAL_START[:6] <= m <= EVAL_END[:6] and int(m[4:]) in (1, 4, 7, 10)]
    log = []
    for s in qstarts:
        ev_months = {s, month_add(s, 1), month_add(s, 2)}
        ev = np.isin(ym, list(ev_months)) & (d <= EVAL_END) & has_odds
        if not ev.any():
            continue
        tr = (d >= warm) & (ym <= month_add(s, -3)); es = ym == month_add(s, -2); ca = ym == month_add(s, -1)
        s123 = [stage_model(X, mask, pos, k, tr, es, FEATS)[0] for k in range(3)]
        T = stage_trifecta(*s123, mask)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
        yt = np.zeros((ca.sum(), 120), bool); yt[np.arange(ca.sum()), y[ca]] = True
        iso.fit(T[ca].ravel(), yt.ravel())
        P[ev] = iso.predict(T[ev].ravel()).reshape(-1, 120)
        q = f"{s[:4]}Q{(int(s[4:]) - 1) // 3 + 1}"
        quarter_of[ev] = q
        ll = float(-np.log(np.clip(P[ev, y[ev]] / P[ev].sum(1), 1e-12, None)).mean())
        log.append(dict(quarter=q, train_races=int(tr.sum()), cal_races=int(ca.sum()), eval_races=int(ev.sum()), tri_logloss=ll))
        print(json.dumps(log[-1]), flush=True)
    evall = np.isfinite(P).all(1)
    idx = np.where(evall)[0]
    ndays = len(set(d[evall]))
    results = []
    for g in GRID:
        sm = settle(bets_for(P, O, idx, **g), P, O, y, pay, ref_combo, d, ndays, quarter_of)
        results.append({**g, **sm})
    summary = dict(
        n_combinations_tried=len(GRID), fixed_rule=f"EV上位から最大{MAXPTS}点・各{YEN}円、1レース上限{CAP}円",
        variant_B="モデル×市場（6分前オッズ）: 過去に6分前オッズがないため評価不能",
        eval_period=[str(d[evall].min()), str(d[evall].max())], eval_races=int(evall.sum()), eval_days=ndays,
        quarters=log,
    )
    json.dump(dict(summary=summary, results=results), open(f"{a.out}/walkforward.json", "w"), ensure_ascii=False, indent=1)
    rows = [{"オッズ上限": g["omax"], "EV閾値": g["thr"], "購入レース": g.get("races", 0), "点数": g.get("tickets", 0),
             "的中": g.get("hits", 0), "回収率": g.get("roi", 0), "95%CI下": g.get("roi_ci95", [0, 0])[0],
             "95%CI上": g.get("roi_ci95", [0, 0])[1], "P(≥110%)": g.get("p_roi_ge_1_1", 0),
             "1日あたり点数": g.get("tickets_per_day", 0), "平均オッズ": g.get("mean_odds", 0),
             "予測的中数": g.get("expected_hits_model", 0)} for g in results]
    md = [f"評価期間 {summary['eval_period'][0]}〜{summary['eval_period'][1]}（{summary['eval_races']}レース / {ndays}日）、"
          f"試した組み合わせ {len(GRID)} 通り（全件）", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"),
          "", "四半期ごとの回収率", "", pd.DataFrame([{"オッズ上限": g["omax"], "EV閾値": g["thr"], **g.get("roi_by_quarter", {})} for g in results]).to_markdown(index=False, floatfmt=".3f"),
          "", "四半期ごとのモデル", "", pd.DataFrame(log).to_markdown(index=False, floatfmt=".3f")]
    open(f"{a.out}/walkforward.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
