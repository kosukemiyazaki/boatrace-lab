"""IDEA-015: BOATCAST のオリジナル展示（一周・まわり足・直線）を特徴量に加える（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea015.py DATADIR [--out results/idea015]
- B: FEATS_C1 + 市場特徴量（IDEA-011 の B）、D: B + オリジナル展示の特徴量（FEATS_ORI）。どちらも市場と混ぜた B'・D' で比べる
- 評価はオリジナル展示が取れているレースだけ（江戸川などは除く）。四半期ごとのウォークフォワード（評価 2026Q1〜Q3）
- 判定: D' が B' より3連単対数損失 0.01 以上小さく、日単位ブートストラップ95%CIが0をまたがない → 採用候補
- 参考: 買い方9通り（締切時オッズで買える前提＝上限、点数200未満は判断に使わない）
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.features import FEATS_C1, FEATS_ORI, add_oriten
from boatlib.model import blend, fit_blend
from boatlib.parse import COMBOS
from exp_idea011 import MK, boot_ci, fit_predict, market, market_feats
from walkforward import GRID, TEST_START, bets_for, load, month_add, settle

QSTARTS = ["202601", "202604", "202607"]
EVAL_END = "20260831"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea015")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    e = pd.read_parquet(f"{a.datadir}/entries.parquet", columns=["date", "jcd", "rno", "boat", "toban"])
    e = e[e["date"] < TEST_START]
    ori = pd.read_parquet(f"{a.datadir}/oriten.parquet")
    ori = ori[ori["date"].astype(str) < TEST_START]
    x = add_oriten(e, ori)
    extra = x[["date", "jcd", "rno", "boat"] + FEATS_ORI]
    F = FEATS_C1 + FEATS_ORI
    r, X, mask, refund, pos, O = load(a.datadir, F, extra=extra)
    d = r["date"].to_numpy(dtype=object).astype(str); ym = np.array([s[:6] for s in d])
    assert max(d) < TEST_START
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    pay = r["tri_pay"].values.astype(float)
    ref_combo = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref_combo[:, ci] |= refund[:, int(b) - 1]
    has = np.isfinite(O).sum(1) >= 60
    has_ori = np.isfinite(X[:, :, len(FEATS_C1)]).sum(1) >= 5  # 一周の相対値が5艇以上ある
    PC = market(O); MF = market_feats(PC, has)
    nC1 = len(FEATS_C1)
    XB = np.concatenate([X[:, :, :nC1], MF], axis=2); FB = FEATS_C1 + MK
    XD = np.concatenate([X, MF], axis=2); FD = F + MK
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    P = {k: np.full((len(r), 120), np.nan) for k in ("B", "B'", "D", "D'")}
    quarter_of = np.array([""] * len(r), dtype=object)
    nll = lambda Q, m: -np.log(np.clip(Q[m, y[m]], 1e-12, None))
    log = []
    for s in QSTARTS:
        ev = np.isin(ym, [s, month_add(s, 1), month_add(s, 2)]) & (d <= EVAL_END) & has
        tr = (d >= warm) & (ym <= month_add(s, -3)); es = ym == month_add(s, -2); ca = ym == month_add(s, -1)
        cm = ca & has
        PB = fit_predict(XB, mask, pos, FB, tr, es, ca, ev, y)
        PD = fit_predict(XD, mask, pos, FD, tr, es, ca, ev, y)
        (ba, bb), _ = fit_blend(PB[cm], PC[cm], y[cm]); (da, db), _ = fit_blend(PD[cm], PC[cm], y[cm])
        P["B"][ev] = PB[ev]; P["B'"][ev] = blend(PB[ev], PC[ev], ba, bb)
        P["D"][ev] = PD[ev]; P["D'"][ev] = blend(PD[ev], PC[ev], da, db)
        q = f"{s[:4]}Q{(int(s[4:]) - 1) // 3 + 1}"; quarter_of[ev] = q
        m = ev & has_ori
        log.append(dict(quarter=q, eval_races=int(ev.sum()), eval_with_ori=int(m.sum()), train_with_ori=int((tr & has_ori).sum()),
                        ll_market=float(nll(PC, m).mean()), **{f"ll_{k}": float(nll(P[k], m).mean()) for k in P}))
        print(json.dumps(log[-1], ensure_ascii=False), flush=True)
    evall = np.isfinite(P["D'"]).all(1) & has_ori
    days = d[evall]
    ll = {"market": float(nll(PC, evall).mean()), **{k: float(nll(P[k], evall).mean()) for k in P}}
    diff = nll(P["D'"], evall) - nll(P["B'"], evall); ci = boot_ci(diff, days)
    diff_m = nll(P["D'"], evall) - nll(PC, evall); ci_m = boot_ci(diff_m, days)
    adopt = diff.mean() <= -0.01 and ci[1] < 0
    idx = np.where(evall)[0]; ndays = len(set(days))
    rows = []
    for name in ("B'", "D'"):
        for g in GRID:
            sm = settle(bets_for(P[name], O, idx, **g), P[name], O, y, pay, ref_combo, d, ndays, quarter_of)
            rows.append({"確率": name, "オッズ上限": g["omax"], "EV閾値": g["thr"], "点数": sm.get("tickets", 0), "的中": sm.get("hits", 0),
                         "回収率": sm.get("roi", np.nan), "95%CI下": sm.get("roi_ci95", [np.nan, np.nan])[0], "95%CI上": sm.get("roi_ci95", [np.nan, np.nan])[1],
                         "判断に使う": "◯" if sm.get("tickets", 0) >= 200 else ""})
    out = dict(eval_races=int(evall.sum()), eval_days=ndays, logloss=ll, diff_Dp_minus_Bp=float(diff.mean()), ci95=ci,
               diff_Dp_minus_market=float(diff_m.mean()), ci95_vs_market=ci_m, decision="採用候補" if adopt else "不採用", quarters=log)
    json.dump(out, open(f"{a.out}/idea015.json", "w"), ensure_ascii=False, indent=1)
    md = [f"### IDEA-015 オリジナル展示（評価はオリジナル展示のあるレース {out['eval_races']} / {ndays}日）", "",
          pd.Series(ll).to_frame("3連単対数損失").to_markdown(floatfmt=".4f"), "",
          f"D' − B' = {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}）／ D' − 市場 = {diff_m.mean():+.4f}（95%CI {ci_m[0]:+.4f}〜{ci_m[1]:+.4f}）", "",
          f"**判定：{out['decision']}**", "", pd.DataFrame(log).to_markdown(index=False, floatfmt=".4f"), "",
          "#### 参考：買い方（締切時オッズで買える前提＝上限。点数200未満は判断に使わない）", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    open(f"{a.out}/idea015.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
