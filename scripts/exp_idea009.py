"""IDEA-009: 締切時オッズで買えたとしたらの上限（docs/ideas.md の登録どおり。買い方は Step 2 と同じ9通りのみ）
usage: python scripts/exp_idea009.py DATADIR [--out results/idea009]
- 予測: FEATS_C1 の段階別 LightGBM → 3連単確率 → Isotonic 校正。四半期ごとのウォークフォワード（評価 2026Q1〜Q3、2026-09以降は使わない）
- 組み合わせ: π ∝ P_model^a × P_締切時^b（a, b は各四半期の校正月で推定）
- 買い方: 締切時オッズ ≤ 上限 かつ π×締切時オッズ > 閾値 の組番を EV 上位から最大10点・各100円
- 実際より有利な前提（締切時オッズを見て締切時オッズで買える・オッズが動かない・9通りの最良を見る）なので、回収率は上限として扱う
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS_C1
from boatlib.model import blend, fit_blend
from boatlib.parse import COMBOS
from exp_trifecta import stage_model, stage_trifecta
from walkforward import GRID, TEST_START, bets_for, load, month_add, settle

QSTARTS = ["202601", "202604", "202607"]
EVAL_END = "20260831"

def market(O):
    inv = np.where(np.isfinite(O) & (O > 0), 1 / O, 0)
    return inv / np.clip(inv.sum(1, keepdims=True), 1e-12, None)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea009")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r, X, mask, refund, pos, O = load(a.datadir, FEATS_C1)
    d = r["date"].to_numpy(dtype=object).astype(str); ym = np.array([x[:6] for x in d])
    assert max(d) < TEST_START
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    pay = r["tri_pay"].values.astype(float)
    ref_combo = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref_combo[:, ci] |= refund[:, int(b) - 1]
    has_odds = np.isfinite(O).sum(1) >= 60
    PC = market(O)
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    PM = np.full((len(r), 120), np.nan); PB = np.full((len(r), 120), np.nan)
    quarter_of = np.array([""] * len(r), dtype=object)
    nll = lambda Q, m: float(-np.log(np.clip(Q[m, y[m]], 1e-12, None)).mean())
    log = []
    for s in QSTARTS:
        ev = np.isin(ym, [s, month_add(s, 1), month_add(s, 2)]) & (d <= EVAL_END) & has_odds
        tr = (d >= warm) & (ym <= month_add(s, -3)); es = ym == month_add(s, -2); ca = ym == month_add(s, -1)
        T = stage_trifecta(*[stage_model(X, mask, pos, k, tr, es, FEATS_C1)[0] for k in range(3)], mask)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
        yt = np.zeros((ca.sum(), 120), bool); yt[np.arange(ca.sum()), y[ca]] = True
        iso.fit(T[ca].ravel(), yt.ravel())
        cal = iso.predict(T[ca | ev].ravel()).reshape(-1, 120)
        P = np.full((len(r), 120), np.nan); P[ca | ev] = cal / cal.sum(1, keepdims=True)
        cm = ca & has_odds
        (wa, wb), _ = fit_blend(P[cm], PC[cm], y[cm])  # 校正月で推定
        PM[ev] = P[ev]; PB[ev] = blend(P[ev], PC[ev], wa, wb)
        q = f"{s[:4]}Q{(int(s[4:]) - 1) // 3 + 1}"
        quarter_of[ev] = q
        log.append(dict(quarter=q, train_races=int(tr.sum()), cal_races=int(ca.sum()), eval_races=int(ev.sum()),
                        blend_ab=[round(float(wa), 3), round(float(wb), 3)],
                        ll_model=nll(PM, ev), ll_close=nll(PC, ev), ll_blend=nll(PB, ev)))
        print(json.dumps(log[-1]), flush=True)
    evall = np.isfinite(PB).all(1)
    idx = np.where(evall)[0]; ndays = len(set(d[evall]))
    results = [{**g, **settle(bets_for(PB, O, idx, **g), PB, O, y, pay, ref_combo, d, ndays, quarter_of)} for g in GRID]
    best = max(r_["roi"] for r_ in results if r_.get("tickets"))
    ok = [g for g in results if g.get("tickets") and g["roi"] >= 1.1 and g["roi_ci95"][0] > 1.0]
    decision = ("上限でも届かない（9通りすべて100%未満）→ 6分前オッズでの3連単EV戦略に見込みなし" if best < 1.0 else
                "上限では届く → 撤退判定を待つ意味がある" if ok else "はっきりしない → 撤退判定を待つ")
    allm = evall
    summary = dict(eval_period=[str(min(d[allm])), str(max(d[allm]))], eval_races=int(allm.sum()), eval_days=ndays,
                   ll_model=nll(PM, allm), ll_close=nll(PC, allm), ll_blend=nll(PB, allm), quarters=log,
                   best_roi=best, decision=decision, n_combinations_tried=len(GRID))
    json.dump(dict(summary=summary, results=[{k: v for k, v in g.items() if k != "daily"} for g in results]),
              open(f"{a.out}/idea009.json", "w"), ensure_ascii=False, indent=1)
    rows = [{"オッズ上限": g["omax"], "EV閾値": g["thr"], "購入レース": g.get("races", 0), "点数": g.get("tickets", 0),
             "的中": g.get("hits", 0), "回収率": g.get("roi", 0), "95%CI下": g.get("roi_ci95", [0, 0])[0],
             "95%CI上": g.get("roi_ci95", [0, 0])[1], "1日あたり点数": g.get("tickets_per_day", 0),
             "平均オッズ": g.get("mean_odds", 0)} for g in results]
    md = [f"### IDEA-009 締切時オッズで買えたとしたらの上限（評価 {summary['eval_period'][0]}〜{summary['eval_period'][1]}、"
          f"{summary['eval_races']}レース / {ndays}日、試した買い方 {len(GRID)} 通り（全件））", "",
          f"3連単対数損失：モデル単体 {summary['ll_model']:.4f} / 締切時の市場 {summary['ll_close']:.4f} / 組み合わせ {summary['ll_blend']:.4f}", "",
          pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), "",
          "四半期ごとの回収率", "", pd.DataFrame([{"オッズ上限": g["omax"], "EV閾値": g["thr"], **g.get("roi_by_quarter", {})} for g in results]).to_markdown(index=False, floatfmt=".3f"),
          "", "四半期ごとのモデル", "", pd.DataFrame(log).to_markdown(index=False, floatfmt=".4f"), "", f"**判定：{decision}**"]
    open(f"{a.out}/idea009.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
