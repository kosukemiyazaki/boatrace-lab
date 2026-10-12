"""IDEA-026 締切直前のオッズの動き（docs/ideas.md の登録どおり。組み合わせは N0・N1 の2組のみ）。
usage: python scripts/exp_idea026.py DATADIR STOREDIR [--n 1200] [--out results/idea026.md]
- 対象: live の締切前（2〜0.5分前、live_t1）と6分前（live）の3連単オッズ、締切時オッズ・着順がそろうレース（各60組以上）。
  日付・締切時刻・場・レース番号の順に並べ、先頭から n レース。前半 n/2 で重みを推定、後半 n/2 で評価。
- N0（基準）: π ∝ P_1^b              （P_1 = 締切直前の3連単オッズの逆数をレース内で正規化）
  N1        : π ∝ P_1^b × (P_1 / P_6)^c （P_6 = 6分前）
- 判定（後半の3連単対数損失）: N1 が N0 より小さく、差の95%信頼区間（日単位）が0をまたがない → オッズの動きは締切直前でも情報を持つ。
- 買い方（後半）: N1 の確率 × 締切直前オッズ > 閾値 {1.1, 1.2, 1.3}、締切直前オッズ 100倍以下、各100円、払戻は締切時オッズ×100円。
  点数 200 以上・回収率 110% 以上・95%CI 下限 100% 超 → 採用候補。
- 参考（判定に使わない）: 締切時オッズの市場の対数損失（締切直前より後に入る情報の大きさ）、締切直前→締切時のオッズの変化。
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from live_eval import KEY, _read, day_boot_ci, market, nll

THRS = (1.1, 1.2, 1.3)
OMAX = 100

def before(o):
    return o[o["fetched_at"] < o["deadline"] + ":00"].drop_duplicates(KEY, keep="last")

def loglin(logs, w):
    lg = sum(wi * li for wi, li in zip(w, logs))
    lg = lg - lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)

def fit(logs, y, w0):
    f = lambda w: nll(loglin(logs, w), y).mean()
    return minimize(f, w0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000}).x

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--n", type=int, default=1200); ap.add_argument("--out", default="results/idea026.md")
    a = ap.parse_args()
    t1 = before(_read(f"{a.store}/live_t1/odds3t/*.csv.gz")); t6 = before(_read(f"{a.store}/live/odds3t/*.csv.gz"))
    close = pd.read_parquet(f"{a.datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    races = pd.read_parquet(f"{a.datadir}/races.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    r = races[KEY + ["tri_combo"]].drop_duplicates(KEY)
    r = r[r["tri_combo"].notna()].merge(t1[KEY + ["deadline"]], on=KEY)
    O1 = r[KEY].merge(t1, on=KEY, how="left")[COMBOS].to_numpy(float)
    O6 = r[KEY].merge(t6, on=KEY, how="left")[COMBOS].to_numpy(float)
    OC = r[KEY].merge(close, on=KEY, how="left")[COMBOS].to_numpy(float)
    ok = (np.isfinite(O1).sum(1) >= 60) & (np.isfinite(O6).sum(1) >= 60) & (np.isfinite(OC).sum(1) >= 60)
    r, O1, O6, OC = r[ok].reset_index(drop=True), O1[ok], O6[ok], OC[ok]
    order = np.lexsort((r["rno"].values, r["jcd"].values, r["deadline"].values, r["date"].values))
    r, O1, O6, OC = r.iloc[order].reset_index(drop=True), O1[order], O6[order], OC[order]
    if len(r) < a.n:
        msg = f"実行前: 対象レース {len(r)} / {a.n}（{r['date'].min() if len(r) else '-'}〜{r['date'].max() if len(r) else '-'}）"
        open(a.out, "w").write(msg + "\n"); print(msg); return
    r, O1, O6, OC = r.iloc[:a.n].reset_index(drop=True), O1[:a.n], O6[:a.n], OC[:a.n]
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    days = r["date"].values
    P1, P6, PC = market(O1), market(O6), market(OC)
    eps = 1e-6
    L1, L6 = np.log(np.clip(P1, eps, None)), np.log(np.clip(P6, eps, None))
    D = np.where((P1 > 0) & (P6 > 0), L1 - L6, 0.0)
    h = a.n // 2; fi, ei = np.arange(h), np.arange(h, a.n)
    w0 = fit([L1[fi]], y[fi], [1.0]); w1 = fit([L1[fi], D[fi]], y[fi], [1.0, 0.0])
    Q0 = loglin([L1[ei]], w0); Q1 = loglin([L1[ei], D[ei]], w1)
    ye, de = y[ei], days[ei]
    l0, l1, lc, l6 = nll(Q0, ye), nll(Q1, ye), nll(PC[ei], ye), nll(P6[ei], ye)
    diff = l1 - l0; ci = day_boot_ci(diff, de)
    adopt = diff.mean() < 0 and ci[1] < 0
    # 買い方（後半）
    rows = []
    yy = np.zeros_like(Q1, bool); yy[np.arange(len(ye)), ye] = True
    o1, oc = O1[ei], OC[ei]
    for thr in THRS:
        buy = np.isfinite(o1) & (o1 <= OMAX) & (Q1 * np.nan_to_num(o1) > thr)
        ret = np.where(buy & yy, np.nan_to_num(oc) * 100, 0.0)
        g = pd.DataFrame({"d": de, "st": buy.sum(1) * 100.0, "rt": ret.sum(1)}).groupby("d").sum()
        n = int(buy.sum())
        if n:
            bi = np.random.default_rng(0).integers(0, len(g), size=(5000, len(g)))
            b = g["rt"].values[bi].sum(1) / np.clip(g["st"].values[bi].sum(1), 1, None)
            roi = g["rt"].sum() / g["st"].sum(); lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
        else:
            roi = lo = hi = np.nan
        drift = float(np.nanmedian((oc / o1)[buy])) if n else np.nan
        rows.append({"閾値": thr, "点数": n, "的中": int((buy & yy).sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi,
                     "締切時オッズ/判定時オッズ（中央値）": drift,
                     "判定": "採用候補" if (n >= 200 and roi >= 1.1 and lo > 1.0) else ("点数不足" if n < 200 else "不採用")})
    out = dict(n=a.n, period=[str(days[0]), str(days[-1])], fit_period=[str(days[0]), str(days[h - 1])], eval_period=[str(days[h]), str(days[-1])],
               w_N0=list(map(float, w0)), w_N1=list(map(float, w1)),
               ll=dict(N0=float(l0.mean()), N1=float(l1.mean()), market_t6=float(l6.mean()), market_close=float(lc.mean())),
               diff_N1_minus_N0=float(diff.mean()), ci95=ci, decision="採用（オッズの動きは締切直前でも情報を持つ）" if adopt else "不採用", bets=rows)
    json.dump(out, open(a.out.replace(".md", ".json"), "w"), ensure_ascii=False, indent=1, default=float)
    md = [f"### IDEA-026 締切直前のオッズの動き（{a.n}レース、推定 {out['fit_period'][0]}〜{out['fit_period'][1]}、評価 {out['eval_period'][0]}〜{out['eval_period'][1]}）", "",
          f"重み N0: b={w0[0]:.3f}／N1: b={w1[0]:.3f}, c={w1[1]:.3f}", "",
          pd.Series(out["ll"]).to_frame("3連単対数損失（評価の後半）").to_markdown(floatfmt=".4f"), "",
          f"N1 − N0 = {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}） → **{out['decision']}**", "",
          "#### 買い方（後半、締切直前オッズで判定・締切時オッズで払戻）", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
