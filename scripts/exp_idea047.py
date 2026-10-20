"""IDEA-047 荒れると予測したレースの「荒れ方」に偏りはあるか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea047.py DATADIR STOREDIR [--out results/idea047.md]
- IDEA-046 のモデル（学習 2023-11〜2025-12）で全レースを予測し、各期間の上位20%を対象にする
- 荒れ方の分類：(a) 1着の艇、(b) 1号艇の着順（1・2・3・4着以下）、(c) (a)×(b)、(d) 2連単の組
- 学習の上位20%で 実際 ÷ 市場の見込み を出し、見込み30回以上・比1.15以上の分類を選ぶ
- 評価の上位20%で、選んだ分類に属する組の確率を (実際+20)/(見込み+20) で補正（複数の分類に属する組は大きい方の比）→ レース内で正規化、
  π × オッズ > 1.0・300倍以下を各100円（払戻は確定オッズ）
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea046 import build, boot

A = 20.0; TOP = 0.2

def classes():
    """分類名 → 120組のどれが属するか（bool 配列）"""
    I = np.array([int(c[0]) for c in COMBOS]); J = np.array([int(c[2]) for c in COMBOS]); K = np.array([int(c[4]) for c in COMBOS])
    pos1 = np.where(I == 1, 1, np.where(J == 1, 2, np.where(K == 1, 3, 4)))
    lab = {1: "1着", 2: "2着", 3: "3着", 4: "4着以下"}
    out = {}
    for k in range(1, 7):
        out[f"(a) 1着 {k}号艇"] = I == k
    for b in range(1, 5):
        out[f"(b) 1号艇 {lab[b]}"] = pos1 == b
    for k in range(1, 7):
        for b in range(1, 5):
            s = (I == k) & (pos1 == b)
            if s.any():
                out[f"(c) 1着 {k}号艇 × 1号艇 {lab[b]}"] = s
    for i in range(1, 7):
        for j in range(1, 7):
            if i != j:
                out[f"(d) 2連単 {i}-{j}"] = (I == i) & (J == j)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea047.md")
    a = ap.parse_args()
    m, O, P, v, y, pred, mdl, feats, target, days, tr, ev, n = build(a.datadir, a.store)
    top = np.zeros(n, bool)
    for msk in (tr, ev):
        thr = np.quantile(pred[msk], 1 - TOP); top |= msk & (pred >= thr)
    hit = np.zeros_like(v); hit[np.arange(n), y] = True
    C = classes()
    rows = []
    for name, s in C.items():
        exp_ = P[:, s].sum(1); act = hit[:, s].any(1).astype(float)
        r = {"分類": name}
        for per, msk in (("学習 上位20%", tr & top), ("評価 上位20%", ev & top), ("評価 下位80%", ev & ~top)):
            r[per + " 見込み"] = exp_[msk].sum(); r[per + " 実際"] = act[msk].sum()
            r[per + " 比"] = act[msk].sum() / exp_[msk].sum() if exp_[msk].sum() > 0 else np.nan
        rows.append(r)
    R = pd.DataFrame(rows)
    sel = R[(R["学習 上位20% 見込み"] >= 30) & (R["学習 上位20% 比"] >= 1.15)]
    fac = np.ones(len(COMBOS))
    for _, r in sel.iterrows():
        f = (r["学習 上位20% 実際"] + A) / (r["学習 上位20% 見込み"] + A)
        fac = np.where(C[r["分類"]], np.maximum(fac, f), fac)
    Pi = P * fac[None, :]; Pi /= Pi.sum(1, keepdims=True)
    e = ev & top
    buy = v[e] & (O[e] <= 300) & (Pi[e] * np.where(v[e], O[e], 0) > 1.0)
    st = buy.sum(1) * 100.0; rt = np.where(buy & hit[e], O[e] * 100, 0).sum(1)
    nb = int(buy.sum()); roi, lo, hi = boot(days[e], st, rt) if nb else (np.nan, np.nan, np.nan)
    judge = "採用候補（前向きの検証へ）" if (nb >= 200 and roi >= 1.1 and lo > 1.0) else "不採用"
    ok = R["学習 上位20% 見込み"] >= 30
    corr = float(np.corrcoef(R.loc[ok, "学習 上位20% 比"], R.loc[ok, "評価 上位20% 比"])[0, 1])
    show = ["分類", "学習 上位20% 見込み", "学習 上位20% 比", "評価 上位20% 見込み", "評価 上位20% 比", "評価 下位80% 比"]
    md = [f"# IDEA-047 荒れると予測したレースの荒れ方（IDEA-046 のモデル、各期間の上位20%：学習 {int((tr & top).sum())}・評価 {int(e.sum())}レース）", "",
          f"**判定：{judge}**　選んだ分類 {len(sel)}個、評価で π×オッズ > 1.0・300倍以下：{nb}点、回収率 {roi:.3f}（日単位95%CI {lo:.3f}〜{hi:.3f}）", "",
          f"学習↔評価の「実際÷見込み」の相関（学習の見込み30回以上の {int(ok.sum())} 分類）：{corr:.3f}", "",
          "注：複数の選んだ分類に属する組は、大きい方の比で補正（二重に掛けない）。", "",
          "## 選んだ分類（学習で見込み30回以上・比1.15以上）", "",
          (sel[show].to_markdown(index=False, floatfmt=".3f") if len(sel) else "なし"), "",
          "## 全分類", "", R[show].to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:12]))

if __name__ == "__main__":
    main()
