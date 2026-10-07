"""IDEA-003: 直前情報入りモデルの対数損失（開発期間のみ。賭けの評価はしない）
usage: python scripts/exp_bi_ll.py DATADIR [--out results/idea003_bi_ll.md]
- A: FEATS（直前情報なし） / B: FEATS_BI（直前情報入り）。どちらも1〜3着の段階別LightGBM、同じ設定
- 学習 〜2026-02 / 早期終了 2026-03 / Isotonic校正 2026-04 / 評価 2026-05〜08（テスト期間 2026-09〜 は load で除外）
- 指標: 3連単対数損失（校正後・レース内で正規化）、単勝対数損失（1着段階のソフトマックス）
  参考: 締切時オッズの市場の3連単対数損失（オッズのある評価レースのみ。判定には使わない）
- 判定: B の3連単対数損失が A より小さく、日単位ブートストラップの差（B−A）の95%信頼区間が0をまたがない → 直前情報は有効
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from walkforward import load
from exp_trifecta import stage_model, stage_trifecta
from boatlib.features import FEATS, FEATS_BI
from boatlib.model import softmax_rows
from boatlib.parse import COMBOS

TRAIN_END, ES, CAL, EV0, EV1 = "20260228", "202603", "202604", "202605", "202608"
RNG = np.random.default_rng(0)

def run(datadir, F):
    r, X, mask, refund, pos, O = load(datadir, F)
    d = r["date"].to_numpy(dtype=object).astype(str); ym = np.array([x[:6] for x in d])
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    warm = (pd.Timestamp(str(min(d))) + pd.DateOffset(months=2)).strftime("%Y%m%d")
    tr = (d >= warm) & (d <= TRAIN_END); es = ym == ES; ca = ym == CAL; ev = (ym >= EV0) & (ym <= EV1)
    out = [stage_model(X, mask, pos, k, tr, es, F) for k in range(3)]
    T = stage_trifecta(*[o[0] for o in out], mask)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
    yt = np.zeros((ca.sum(), 120), bool); yt[np.arange(ca.sum()), y[ca]] = True
    iso.fit(T[ca].ravel(), yt.ravel())
    P = iso.predict(T[ev].ravel()).reshape(-1, 120); P /= P.sum(1, keepdims=True)
    tri = -np.log(np.clip(P[np.arange(ev.sum()), y[ev]], 1e-12, None))
    pw = softmax_rows(out[0][0][ev], mask[ev])
    win = -np.log(np.clip(pw[np.arange(ev.sum()), pos[ev, 0]], 1e-12, None))
    imp = pd.Series(out[0][2].feature_importance("gain"), index=F).sort_values(ascending=False)
    key = pd.DataFrame({"date": d[ev], "jcd": r["jcd"].values[ev], "rno": r["rno"].values[ev]})
    return key, tri, win, O[ev], y[ev], [o[1] for o in out], imp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea003_bi_ll.md")
    a = ap.parse_args()
    ka, ta, wa, Oa, ya, ita, _ = run(a.datadir, FEATS)
    kb, tb, wb, Ob, yb, itb, imp = run(a.datadir, FEATS_BI)
    assert (ka.values == kb.values).all()
    days = ka["date"].values
    diff = pd.Series(tb - ta).groupby(days).agg(["sum", "count"])
    bi = RNG.integers(0, len(diff), size=(5000, len(diff)))
    boot = diff["sum"].values[bi].sum(1) / diff["count"].values[bi].sum(1)
    ci = (float(np.quantile(boot, .025)), float(np.quantile(boot, .975)))
    ok = np.isfinite(Oa).sum(1) >= 60
    inv = np.where(np.isfinite(Oa) & (Oa > 0), 1 / Oa, 0); pk = inv / np.clip(inv.sum(1, keepdims=True), 1e-12, None)
    mk = -np.log(np.clip(pk[ok][np.arange(ok.sum()), ya[ok]], 1e-12, None))
    adopt = tb.mean() < ta.mean() and ci[1] < 0
    rows = [
        {"モデル": "A 直前情報なし", "3連単LL": ta.mean(), "単勝LL": wa.mean(), "3連単LL（オッズのあるレース）": ta[ok].mean(), "best_iter": str(ita)},
        {"モデル": "B 直前情報入り", "3連単LL": tb.mean(), "単勝LL": wb.mean(), "3連単LL（オッズのあるレース）": tb[ok].mean(), "best_iter": str(itb)},
        {"モデル": "参考: 締切時オッズの市場", "3連単LL": None, "単勝LL": None, "3連単LL（オッズのあるレース）": mk.mean(), "best_iter": ""},
    ]
    md = [f"評価 {EV0}〜{EV1}：{len(ta)}レース / {len(diff)}日（オッズのあるレース {ok.sum()}）、一様分布の3連単LL {np.log(120):.3f}・単勝LL {np.log(6):.3f}", "",
          pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"), "",
          f"3連単LLの差（B−A）: {tb.mean() - ta.mean():+.4f}、日単位ブートストラップ95%信頼区間 [{ci[0]:+.4f}, {ci[1]:+.4f}]",
          f"単勝LLの差（B−A）: {wb.mean() - wa.mean():+.4f}",
          f"**判定: {'直前情報は有効（採用）' if adopt else '有効とは言えない（不採用）'}**", "",
          "B の1着段階の特徴量の重要度（上位15、gain）", "", imp.head(15).round(0).to_frame("gain").to_markdown()]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
