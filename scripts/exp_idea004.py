"""IDEA-004: 節間成績・モーターの調子（開発期間のみ。賭けの評価はしない）
usage: python scripts/exp_idea004.py DATADIR [--out results/idea004_meet_motor.md]
- B  = FEATS_BI（IDEA-003 の直前情報入り。基準）
- C1 = B + 節間成績（今節の出走数・平均着順・1着数・平均ST・展示タイムの平均順位）
- C2 = C1 + モーターの調子（場×モーターの直近30走の平均着順・2連対率、直近7日の平均着順）
- 期間・設定は IDEA-003（scripts/exp_bi_ll.py）と同じ
- 判定: C1 または C2 の3連単対数損失が B より小さく、日単位ブートストラップの差の95%信頼区間が0をまたがない → 有効（小さい方を採用候補）
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp_bi_ll import run, EV0, EV1
from boatlib.features import FEATS_BI, FEATS_C1, FEATS_C2

RNG = np.random.default_rng(0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea004_meet_motor.md")
    a = ap.parse_args()
    res = {}
    for name, F in (("B", FEATS_BI), ("C1", FEATS_C1), ("C2", FEATS_C2)):
        key, tri, win, O, y, its, imp = run(a.datadir, F)
        res[name] = dict(key=key, tri=tri, win=win, its=its, imp=imp)
        print(name, tri.mean(), win.mean(), flush=True)
    for n in ("C1", "C2"):
        assert (res[n]["key"].values == res["B"]["key"].values).all()
    days = res["B"]["key"]["date"].values
    rows, verdicts = [], {}
    for n in ("B", "C1", "C2"):
        r = {"モデル": n, "3連単LL": res[n]["tri"].mean(), "単勝LL": res[n]["win"].mean(), "best_iter": str(res[n]["its"])}
        if n != "B":
            d = pd.Series(res[n]["tri"] - res["B"]["tri"]).groupby(days).agg(["sum", "count"])
            bi = RNG.integers(0, len(d), size=(5000, len(d)))
            boot = d["sum"].values[bi].sum(1) / d["count"].values[bi].sum(1)
            lo, hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
            r.update({"差（−B）": res[n]["tri"].mean() - res["B"]["tri"].mean(), "95%CI下": lo, "95%CI上": hi})
            verdicts[n] = (res[n]["tri"].mean() < res["B"]["tri"].mean()) and hi < 0
        rows.append(r)
    ok = [n for n in ("C1", "C2") if verdicts[n]]
    best = min(ok, key=lambda n: res[n]["tri"].mean()) if ok else None
    md = [f"評価 {EV0}〜{EV1}：{len(days)}レース / {len(set(days))}日。試した組み合わせ 3通り（基準 B を含む）の全件", "",
          pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"), "",
          f"**判定: {'有効（採用候補: ' + best + '）' if best else '有効とは言えない（不採用）'}**", "",
          "C2 の1着段階の特徴量の重要度（上位20、gain）", "", res["C2"]["imp"].head(20).round(0).to_frame("gain").to_markdown()]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
