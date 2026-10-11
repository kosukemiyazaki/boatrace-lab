"""IDEA-017: 選手・コース・モーターの時系列レーティング（Plackett–Luce の逐次更新、boatlib/rating.py）を特徴量に加える
usage: python scripts/exp_idea017.py DATADIR [--hist HIST_DATADIR] [--out results/idea017]
- レーティングは全期間（--hist があれば長期の entries を先頭につなげる）で日付順に計算し、各レースの前の値を特徴量にする
- B: FEATS_C1 + 市場特徴量（IDEA-011 の B）、C: B + レーティング特徴量。どちらも市場と混ぜた B'・C' で比べる
- 四半期ごとのウォークフォワード（評価 2026Q1〜Q3）、Isotonic 校正、混ぜる重みは校正月で推定（IDEA-011 と同じ）
- 判定: C' が B' より3連単対数損失 0.01 以上小さく、日単位ブートストラップ95%CIが0をまたがない → 採用候補
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS_C1
from boatlib.model import blend, fit_blend
from boatlib.parse import COMBOS
from boatlib.rating import RT_FEATS, compute
from exp_idea011 import MK, boot_ci, fit_predict, market, market_feats
from walkforward import TEST_START, load, month_add

QSTARTS = ["202601", "202604", "202607"]
EVAL_END = "20260831"
COLS = ["date", "jcd", "rno", "boat", "toban", "motor_no", "pos", "course"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--hist"); ap.add_argument("--out", default="results/idea017")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    e = pd.read_parquet(f"{a.datadir}/entries.parquet", columns=COLS)
    e = e[e["date"] < TEST_START]
    if a.hist:
        h = pd.read_parquet(f"{a.hist}/entries.parquet", columns=COLS)
        h = h[h["date"] < e["date"].min()]
        e = pd.concat([h, e], ignore_index=True)
    rt = compute(e.reset_index(drop=True))
    extra = pd.concat([e.reset_index(drop=True)[["date", "jcd", "rno", "boat"]], rt[RT_FEATS]], axis=1)
    F = FEATS_C1 + RT_FEATS
    r, X, mask, refund, pos, O = load(a.datadir, F, extra=extra)
    d = r["date"].to_numpy(dtype=object).astype(str); ym = np.array([x[:6] for x in d])
    assert max(d) < TEST_START
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    has = np.isfinite(O).sum(1) >= 60
    PC = market(O)
    MF = market_feats(PC, has)
    nC1 = len(FEATS_C1)
    XB = np.concatenate([X[:, :, :nC1], MF], axis=2); FB = FEATS_C1 + MK
    XC = np.concatenate([X, MF], axis=2); FC = F + MK
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    P = {k: np.full((len(r), 120), np.nan) for k in ("B", "B'", "C", "C'")}
    nll = lambda Q, m: -np.log(np.clip(Q[m, y[m]], 1e-12, None))
    log = []
    for s in QSTARTS:
        ev = np.isin(ym, [s, month_add(s, 1), month_add(s, 2)]) & (d <= EVAL_END) & has
        tr = (d >= warm) & (ym <= month_add(s, -3)); es = ym == month_add(s, -2); ca = ym == month_add(s, -1)
        cm = ca & has
        PB = fit_predict(XB, mask, pos, FB, tr, es, ca, ev, y)
        PCm = fit_predict(XC, mask, pos, FC, tr, es, ca, ev, y)
        (ba, bb), _ = fit_blend(PB[cm], PC[cm], y[cm]); (ca_, cb), _ = fit_blend(PCm[cm], PC[cm], y[cm])
        P["B"][ev] = PB[ev]; P["B'"][ev] = blend(PB[ev], PC[ev], ba, bb)
        P["C"][ev] = PCm[ev]; P["C'"][ev] = blend(PCm[ev], PC[ev], ca_, cb)
        q = f"{s[:4]}Q{(int(s[4:]) - 1) // 3 + 1}"
        log.append(dict(quarter=q, eval_races=int(ev.sum()), ll_market=float(nll(PC, ev).mean()), **{f"ll_{k}": float(nll(P[k], ev).mean()) for k in P}))
        print(json.dumps(log[-1], ensure_ascii=False), flush=True)
    evall = np.isfinite(P["C'"]).all(1)
    days = d[evall]
    ll = {"market": float(nll(PC, evall).mean()), **{k: float(nll(P[k], evall).mean()) for k in P}}
    diff = nll(P["C'"], evall) - nll(P["B'"], evall); ci = boot_ci(diff, days)
    diff_m = nll(P["C'"], evall) - nll(PC, evall); ci_m = boot_ci(diff_m, days)
    adopt = diff.mean() <= -0.01 and ci[1] < 0
    out = dict(hist=bool(a.hist), rating_races_from=str(e["date"].min()), eval_races=int(evall.sum()), eval_days=len(set(days)),
               logloss=ll, diff_Cp_minus_Bp=float(diff.mean()), ci95=ci, diff_Cp_minus_market=float(diff_m.mean()), ci95_vs_market=ci_m,
               decision="採用候補" if adopt else "不採用", quarters=log)
    json.dump(out, open(f"{a.out}/idea017{'_hist' if a.hist else ''}.json", "w"), ensure_ascii=False, indent=1)
    md = [f"### IDEA-017 レーティング（レーティングの計算は {out['rating_races_from']} から、評価 {out['eval_races']}レース / {out['eval_days']}日）", "",
          pd.Series(ll).to_frame("3連単対数損失").to_markdown(floatfmt=".4f"), "",
          f"C' − B' = {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}）／ C' − 市場 = {diff_m.mean():+.4f}（95%CI {ci_m[0]:+.4f}〜{ci_m[1]:+.4f}）", "",
          f"**判定：{out['decision']}**", "", pd.DataFrame(log).to_markdown(index=False, floatfmt=".4f")]
    open(f"{a.out}/idea017{'_hist' if a.hist else ''}.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
