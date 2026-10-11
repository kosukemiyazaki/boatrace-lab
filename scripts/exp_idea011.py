"""IDEA-011 スタッキング（市場の確率を特徴量に）と IDEA-012 段階別の分解（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea011.py DATADIR [--out results/idea011]
- A: 従来モデル（FEATS_C1）、B: FEATS_C1 + 市場特徴量（各艇の市場1〜3着確率・log・順位・レースの1番人気確率）
- 四半期ごとのウォークフォワード（評価 2026Q1〜Q3）。Isotonic 校正。A'・B' は市場との混ぜ合わせ（a, b は校正月で推定）
- 市場の確率は締切時3連単オッズ（実運用では締切直前のオッズに置き換える前提。実際より少し有利）
- 判定: B' が A' より対数損失 0.01 以上小さく、日単位ブートストラップの95%CIが0をまたがない → 採用
- IDEA-012: 3連単対数損失を 1着 / 2着|1着 / 3着|1,2着 に分解（市場・A・A'・B・B'）
- 参考: 買い方9通り（締切時オッズで買える前提＝上限）。点数200未満は判断に使わない
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
PI = np.array([int(c[0]) - 1 for c in COMBOS]); PJ = np.array([int(c[2]) - 1 for c in COMBOS]); PK = np.array([int(c[4]) - 1 for c in COMBOS])
MK = ["mk_p1", "mk_p2", "mk_p3", "mk_lp1", "mk_rank1", "mk_fav"]

def market(O):
    inv = np.where(np.isfinite(O) & (O > 0), 1 / O, 0)
    return inv / np.clip(inv.sum(1, keepdims=True), 1e-12, None)

def market_feats(PC, has):
    """各艇の市場1〜3着確率など (n,6,len(MK))。オッズのないレースは NaN"""
    n = len(PC)
    p1 = np.zeros((n, 6)); p2 = np.zeros((n, 6)); p3 = np.zeros((n, 6))
    for b in range(6):
        p1[:, b] = PC[:, PI == b].sum(1); p2[:, b] = PC[:, PJ == b].sum(1); p3[:, b] = PC[:, PK == b].sum(1)
    rank1 = (-p1).argsort(1, kind="stable").argsort(1) + 1.0
    fav = np.repeat(p1.max(1, keepdims=True), 6, 1)
    F = np.stack([p1, p2, p3, np.log(np.clip(p1, 1e-6, None)), rank1, fav], axis=2)
    F[~has] = np.nan
    return F

def stage_ll(P, y):
    """3連単対数損失を 1着 / 2着|1着 / 3着|1,2着 に分解（レースごと）"""
    n = np.arange(len(y)); a, b = PI[y], PJ[y]
    p1 = np.array([P[i, PI == a[i]].sum() for i in n])
    p12 = np.array([P[i, (PI == a[i]) & (PJ == b[i])].sum() for i in n])
    p123 = P[n, y]
    cl = lambda x: np.clip(x, 1e-12, None)
    return -np.log(cl(p1)), -np.log(cl(p12 / cl(p1))), -np.log(cl(p123 / cl(p12)))

def boot_ci(diff, days, n_boot=5000, seed=0):
    dd = pd.Series(diff).groupby(np.asarray(days)).agg(["sum", "count"])
    bi = np.random.default_rng(seed).integers(0, len(dd), size=(n_boot, len(dd)))
    b = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]

def fit_predict(X, mask, pos, feats, tr, es, ca, ev, y):
    T = stage_trifecta(*[stage_model(X, mask, pos, k, tr, es, feats)[0] for k in range(3)], mask)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
    yt = np.zeros((ca.sum(), 120), bool); yt[np.arange(ca.sum()), y[ca]] = True
    iso.fit(T[ca].ravel(), yt.ravel())
    P = np.full(T.shape, np.nan)
    sel = ca | ev
    cal = iso.predict(T[sel].ravel()).reshape(-1, 120); P[sel] = cal / cal.sum(1, keepdims=True)
    return P

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea011")
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
    has = np.isfinite(O).sum(1) >= 60
    PC = market(O)
    X2 = np.concatenate([X, market_feats(PC, has)], axis=2); F2 = FEATS_C1 + MK
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    P = {k: np.full((len(r), 120), np.nan) for k in ("A", "A'", "B", "B'")}
    quarter_of = np.array([""] * len(r), dtype=object)
    nll = lambda Q, m: -np.log(np.clip(Q[m, y[m]], 1e-12, None))
    log = []
    for s in QSTARTS:
        ev = np.isin(ym, [s, month_add(s, 1), month_add(s, 2)]) & (d <= EVAL_END) & has
        tr = (d >= warm) & (ym <= month_add(s, -3)); es = ym == month_add(s, -2); ca = (ym == month_add(s, -1))
        cm = ca & has
        PA = fit_predict(X, mask, pos, FEATS_C1, tr, es, ca, ev, y)
        PB = fit_predict(X2, mask, pos, F2, tr, es, ca, ev, y)
        (aa, ab), _ = fit_blend(PA[cm], PC[cm], y[cm]); (ba, bb), _ = fit_blend(PB[cm], PC[cm], y[cm])
        P["A"][ev] = PA[ev]; P["A'"][ev] = blend(PA[ev], PC[ev], aa, ab)
        P["B"][ev] = PB[ev]; P["B'"][ev] = blend(PB[ev], PC[ev], ba, bb)
        q = f"{s[:4]}Q{(int(s[4:]) - 1) // 3 + 1}"; quarter_of[ev] = q
        log.append(dict(quarter=q, train_races=int(tr.sum()), eval_races=int(ev.sum()),
                        blend_A=[round(float(aa), 3), round(float(ab), 3)], blend_B=[round(float(ba), 3), round(float(bb), 3)],
                        ll_market=float(nll(PC, ev).mean()), **{f"ll_{k}": float(nll(P[k], ev).mean()) for k in P}))
        print(json.dumps(log[-1], ensure_ascii=False), flush=True)
    evall = np.isfinite(P["B'"]).all(1)
    days = d[evall]
    ll = {"market": float(nll(PC, evall).mean()), **{k: float(nll(P[k], evall).mean()) for k in P}}
    diff = nll(P["B'"], evall) - nll(P["A'"], evall)
    ci = boot_ci(diff, days)
    adopt = (diff.mean() <= -0.01) and (ci[1] < 0)
    diff_m = nll(P["B'"], evall) - nll(PC, evall); ci_m = boot_ci(diff_m, days)
    # IDEA-012 段階別
    stage = {}
    for k, Q in [("market", PC)] + list(P.items()):
        s1, s2, s3 = stage_ll(Q[evall], y[evall])
        stage[k] = dict(stage1=float(s1.mean()), stage2=float(s2.mean()), stage3=float(s3.mean()))
    sdf = pd.DataFrame(stage).T
    for k in P:
        for st in ("stage1", "stage2", "stage3"):
            sdf.loc[k, f"{st}−市場"] = sdf.loc[k, st] - sdf.loc["market", st]
    # 参考: 買い方（締切時オッズで買える前提）
    idx = np.where(evall)[0]; ndays = len(set(days))
    rows = []
    for name in ("A'", "B'"):
        for g in GRID:
            sm = settle(bets_for(P[name], O, idx, **g), P[name], O, y, pay, ref_combo, d, ndays, quarter_of)
            rows.append({"確率": name, "オッズ上限": g["omax"], "EV閾値": g["thr"], "点数": sm.get("tickets", 0), "的中": sm.get("hits", 0),
                         "回収率": sm.get("roi", np.nan), "95%CI下": sm.get("roi_ci95", [np.nan, np.nan])[0], "95%CI上": sm.get("roi_ci95", [np.nan, np.nan])[1],
                         "判断に使う": "◯" if sm.get("tickets", 0) >= 200 else ""})
    bets = pd.DataFrame(rows)
    out = dict(eval_period=[str(min(days)), str(max(days))], eval_races=int(evall.sum()), eval_days=ndays, quarters=log,
               logloss=ll, diff_Bp_minus_Ap=float(diff.mean()), ci95=ci, diff_Bp_minus_market=float(diff_m.mean()), ci95_vs_market=ci_m,
               decision_idea011="採用（スタッキングは混ぜるだけより有意に良い）" if adopt else "不採用", stages=stage,
               bets=[{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()} for row in rows])
    json.dump(out, open(f"{a.out}/idea011.json", "w"), ensure_ascii=False, indent=1)
    md = [f"### IDEA-011 スタッキング / IDEA-012 段階別分解（評価 {out['eval_period'][0]}〜{out['eval_period'][1]}、{out['eval_races']}レース / {ndays}日）", "",
          "3連単対数損失（評価期間全体）", "", pd.Series(ll).to_frame("対数損失").to_markdown(floatfmt=".4f"), "",
          f"B' − A' = {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}）／ B' − 市場 = {diff_m.mean():+.4f}（95%CI {ci_m[0]:+.4f}〜{ci_m[1]:+.4f}）", "",
          f"**IDEA-011 判定：{out['decision_idea011']}**", "",
          "四半期ごと", "", pd.DataFrame(log).to_markdown(index=False, floatfmt=".4f"), "",
          "#### IDEA-012 段階別の対数損失（1着 / 2着|1着 / 3着|1,2着）", "", sdf.to_markdown(floatfmt=".4f"), "",
          "#### 参考：買い方（締切時オッズで買える前提＝上限。点数200未満は判断に使わない）", "", bets.to_markdown(index=False, floatfmt=".3f")]
    open(f"{a.out}/idea011.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
