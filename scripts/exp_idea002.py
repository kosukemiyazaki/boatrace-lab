"""IDEA-002 オッズの動きを特徴量にする（docs/ideas.md の登録どおり。組み合わせは M0・M1・M2 の3通りのみ）。
usage: python scripts/exp_idea002.py DATADIR STOREDIR [--n 2000] [--out results/idea002.md]
- 対象: 撤退判定と同じ選び方（scripts/live_eval.py）で、さらに15分前・10分前の3連単オッズもそろうレース。先頭から n レース。
- 予測モデルの確率: live が締切前に記録した frozen_v4 の予想（撤退判定と同じ）
- M0: π ∝ P_model^a × P_6^b
  M1: π ∝ P_model^a × P_6^b × (P_6 / P_15)^c
  M2: π ∝ P_model^a × P_6^b × (P_6 / P_10)^c × (P_10 / P_15)^d
  重みは前半 n/2 レースで最尤推定し、後半 n/2 レースの3連単対数損失で比べる。
- 判定: M1 または M2 が M0 より小さく、差の95%信頼区間（日単位）が0をまたがない → 採用（小さい方）。それ以外は不採用。
  締切時オッズの市場との比較は参考として記録する（判定には使わない）。
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from live_eval import day_boot_ci, head, load, market, nll

def loglin(logs, w):
    lg = sum(wi * li for wi, li in zip(w, logs))
    lg = lg - lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)

def fit_loglin(logs, y, w0):
    f = lambda w: nll(loglin(logs, w), y).mean()
    r = minimize(f, w0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000})
    return r.x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--out", default="results/idea002.md")
    a = ap.parse_args()
    d = load(a.datadir, a.store, extra_leads=("t10", "t15"))
    if len(d["r"]) < a.n:
        msg = f"実行前: 対象レース {len(d['r'])} / {a.n}"
        open(a.out, "w").write(msg + "\n"); print(msg); return
    d = head(d, a.n)
    y, days = d["y"], d["r"]["date"].values
    lg = lambda Q: np.log(np.clip(Q, 1e-12, None))
    lm, l6, l10, l15 = lg(d["P"]), lg(market(d["L"]["t6"])), lg(market(d["L"]["t10"])), lg(market(d["L"]["t15"]))
    specs = {"M0": [lm, l6], "M1": [lm, l6, l6 - l15], "M2": [lm, l6, l6 - l10, l10 - l15]}
    h = a.n // 2
    fit, ev = np.arange(a.n) < h, np.arange(a.n) >= h
    res, weights = {}, {}
    for k, logs in specs.items():
        w = fit_loglin([x[fit] for x in logs], y[fit], [0.5, 0.5] + [0.0] * (len(logs) - 2))
        weights[k] = [float(v) for v in w]
        res[k] = nll(loglin(logs, w), y)
    res["market_close"] = nll(market(d["C"]), y)
    ll = {k: float(v[ev].mean()) for k, v in res.items()}
    cmp = {}
    for k in ("M1", "M2"):
        diff = res[k][ev] - res["M0"][ev]
        cmp[k] = dict(diff_vs_M0=float(diff.mean()), ci95=day_boot_ci(diff, days[ev]))
    ok = [k for k in ("M1", "M2") if cmp[k]["diff_vs_M0"] < 0 and cmp[k]["ci95"][1] < 0]
    adopt = min(ok, key=lambda k: ll[k]) if ok else None
    ref = {k: dict(diff_vs_close=float((res[k][ev] - res["market_close"][ev]).mean()),
                   ci95=day_boot_ci(res[k][ev] - res["market_close"][ev], days[ev])) for k in specs}
    out = dict(races=a.n, fit_races=int(fit.sum()), eval_races=int(ev.sum()),
               period=[str(d["r"]["date"].min()), str(d["r"]["date"].max())],
               weights=weights, logloss_eval=ll, vs_M0=cmp, reference_vs_market_close=ref,
               decision=f"採用: {adopt}" if adopt else "不採用（オッズの動きは対数損失を有意に下げない）")
    md = ["### IDEA-002 オッズの動き", "```", json.dumps(out, ensure_ascii=False, indent=1), "```"]
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
