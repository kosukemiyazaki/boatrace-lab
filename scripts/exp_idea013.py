"""IDEA-013: ほかの券種のオッズから3連単の値付けのずれを探す（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea013.py DATADIR [--out results/idea013]
- P_cross(a,b,c) = P_2連単(a,b) × P_3連複({a,b,c}) / Σ_c' P_3連複({a,b,c'})   （各券種の締切時オッズの逆数をレース内で正規化）
- 基準 B' : π ∝ P_model^a × P_3連単^b（P_model は IDEA-011 の B、四半期ウォークフォワード、評価 2026Q1〜Q3）
- C2      : π ∝ P_model^a × P_3連単^b × P_cross^c（重みは各四半期の校正月で推定）
- 判定: C2 が B' より3連単対数損失 0.01 以上小さく、日単位ブートストラップ95%CIが0をまたがない → 採用候補
- 参考: C1（P_cross 単独）と市場の対数損失、買い方9通り（締切時オッズで買える前提＝上限、点数200未満は判断に使わない）
"""
import argparse, itertools, json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS_C1
from boatlib.model import blend, fit_blend
from boatlib.parse import COMBOS, COMBOS_2T, COMBOS_3F
from exp_idea011 import MK, boot_ci, fit_predict, market, market_feats
from walkforward import GRID, TEST_START, bets_for, load, month_add, settle

QSTARTS = ["202601", "202604", "202607"]
EVAL_END = "20260831"
KEY = ["date", "jcd", "rno"]

def loglin(logs, w):
    lg = sum(wi * li for wi, li in zip(w, logs)); lg = lg - lg.max(1, keepdims=True); e = np.exp(lg)
    return e / e.sum(1, keepdims=True)

def cross(r, datadir):
    nz = lambda d: d.assign(date=d["date"].astype(str), jcd=d["jcd"].astype(str).str.zfill(2))
    k = r[KEY].assign(date=r["date"].astype(str), jcd=r["jcd"].astype(str).str.zfill(2))
    f3 = k.merge(nz(pd.read_parquet(f"{datadir}/odds3f.parquet")), on=KEY, how="left")[COMBOS_3F].to_numpy(float)
    t2 = k.merge(nz(pd.read_parquet(f"{datadir}/odds2tf.parquet")), on=KEY, how="left")[COMBOS_2T].to_numpy(float)
    has = (np.isfinite(f3) & (f3 > 0)).sum(1) >= 15
    has &= (np.isfinite(t2) & (t2 > 0)).sum(1) >= 25
    P3F, P2T = market(f3), market(t2)
    i3 = {frozenset(map(int, c.split("="))): i for i, c in enumerate(COMBOS_3F)}; i2 = {c: i for i, c in enumerate(COMBOS_2T)}
    X = np.zeros((len(r), 120))
    for ci, c in enumerate(COMBOS):
        a, b, cc = int(c[0]), int(c[2]), int(c[4])
        den = sum(P3F[:, i3[frozenset((a, b, z))]] for z in range(1, 7) if z not in (a, b))
        X[:, ci] = P2T[:, i2[f"{a}-{b}"]] * P3F[:, i3[frozenset((a, b, cc))]] / np.clip(den, 1e-12, None)
    X = X / np.clip(X.sum(1, keepdims=True), 1e-12, None)
    return X, has

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea013")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r, X, mask, refund, pos, O = load(a.datadir, FEATS_C1)
    d = r["date"].to_numpy(dtype=object).astype(str); ym = np.array([s[:6] for s in d])
    assert max(d) < TEST_START
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    pay = r["tri_pay"].values.astype(float)
    ref_combo = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref_combo[:, ci] |= refund[:, int(b) - 1]
    has = np.isfinite(O).sum(1) >= 60
    PC = market(O); MF = market_feats(PC, has)
    PX, hasx = cross(r, a.datadir)
    XB = np.concatenate([X, MF], axis=2); FB = FEATS_C1 + MK
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    P = {k: np.full((len(r), 120), np.nan) for k in ("B'", "C2")}
    quarter_of = np.array([""] * len(r), dtype=object)
    nll = lambda Q, m: -np.log(np.clip(Q[m, y[m]], 1e-12, None))
    eps = 1e-9; L = lambda Q: np.log(np.clip(Q, eps, None))
    log = []
    for s in QSTARTS:
        ev = np.isin(ym, [s, month_add(s, 1), month_add(s, 2)]) & (d <= EVAL_END) & has & hasx
        tr = (d >= warm) & (ym <= month_add(s, -3)); es = ym == month_add(s, -2); ca = ym == month_add(s, -1)
        cm = ca & has & hasx
        PB = fit_predict(XB, mask, pos, FB, tr, es, ca, ev, y)
        (ba, bb), _ = fit_blend(PB[cm], PC[cm], y[cm])
        f = lambda w: -np.log(np.clip(loglin([L(PB[cm]), L(PC[cm]), L(PX[cm])], w)[np.arange(cm.sum()), y[cm]], 1e-12, None)).mean()
        w = minimize(f, [ba, bb, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000}).x
        P["B'"][ev] = blend(PB[ev], PC[ev], ba, bb)
        P["C2"][ev] = loglin([L(PB[ev]), L(PC[ev]), L(PX[ev])], w)
        q = f"{s[:4]}Q{(int(s[4:]) - 1) // 3 + 1}"; quarter_of[ev] = q
        log.append(dict(quarter=q, eval_races=int(ev.sum()), w_C2=[round(float(x), 3) for x in w], ll_market=float(nll(PC, ev).mean()),
                        ll_C1=float(nll(PX, ev).mean()), **{f"ll_{k}": float(nll(P[k], ev).mean()) for k in P}))
        print(json.dumps(log[-1], ensure_ascii=False), flush=True)
    evall = np.isfinite(P["C2"]).all(1); days = d[evall]
    ll = {"market": float(nll(PC, evall).mean()), "C1": float(nll(PX, evall).mean()), **{k: float(nll(P[k], evall).mean()) for k in P}}
    diff = nll(P["C2"], evall) - nll(P["B'"], evall); ci = boot_ci(diff, days)
    adopt = diff.mean() <= -0.01 and ci[1] < 0
    idx = np.where(evall)[0]; ndays = len(set(days))
    rows = []
    for name in ("B'", "C2"):
        for g in GRID:
            sm = settle(bets_for(P[name], O, idx, **g), P[name], O, y, pay, ref_combo, d, ndays, quarter_of)
            rows.append({"確率": name, "オッズ上限": g["omax"], "EV閾値": g["thr"], "点数": sm.get("tickets", 0), "的中": sm.get("hits", 0),
                         "回収率": sm.get("roi", np.nan), "95%CI下": sm.get("roi_ci95", [np.nan, np.nan])[0], "95%CI上": sm.get("roi_ci95", [np.nan, np.nan])[1],
                         "判断に使う": "◯" if sm.get("tickets", 0) >= 200 else ""})
    out = dict(eval_races=int(evall.sum()), eval_days=ndays, logloss=ll, diff_C2_minus_Bp=float(diff.mean()), ci95=ci,
               decision="採用候補" if adopt else "不採用", quarters=log)
    json.dump(out, open(f"{a.out}/idea013.json", "w"), ensure_ascii=False, indent=1)
    md = [f"### IDEA-013 クロスプール（評価 {out['eval_races']}レース / {ndays}日）", "",
          pd.Series(ll).to_frame("3連単対数損失").to_markdown(floatfmt=".4f"), "",
          f"C2 − B' = {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}）", "", f"**判定：{out['decision']}**", "",
          pd.DataFrame(log).to_markdown(index=False, floatfmt=".4f"), "",
          "#### 参考：買い方（締切時オッズで買える前提＝上限。点数200未満は判断に使わない）", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    open(f"{a.out}/idea013.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
