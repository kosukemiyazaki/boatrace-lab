"""IDEA-035 展開：市場は「1着がこの艇なら2着・3着は誰か」を正しく値付けしているか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea035.py DATADIR [--out results/idea035.md]
- P(i-j-k) = P(i) × q2(j|i) × q3(k|i,j)（確定オッズの逆数の正規化）
- 学習で c2[i,j,帯]・c3[i,j,k] を推定（+20 で縮める）→ 評価で π × オッズ > 1.1・300倍以下を各100円
- 学習 2023-10〜2025-12 ／ 評価 2026-01〜08。判定：回収率110%以上・点数200以上・日単位95%CI下限 > 100%
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

KEY = ["date", "jcd", "rno"]
TR = ("20231001", "20251231"); EV = ("20260101", "20260831")
A = 20.0
RNG = np.random.default_rng(0)

def norm(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2), rno=df["rno"].astype(int))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea035.md")
    a = ap.parse_args()
    races = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    races = races[(races["date"] >= TR[0]) & (races["date"] <= EV[1])]
    m = races.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY)
    O = m[COMBOS].to_numpy(float); valid = np.isfinite(O) & (O > 0)
    ok = valid.sum(1) >= 60
    m, O, valid = m[ok].reset_index(drop=True), O[ok], valid[ok]
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    okY = ~pd.isna(y); m, O, valid, y = m[okY].reset_index(drop=True), O[okY], valid[okY], y[okY].astype(int)
    n = len(m)
    inv = np.where(valid, 1 / np.where(valid, O, 1), 0); P = inv / inv.sum(1, keepdims=True)
    I = np.array([int(c[0]) for c in COMBOS]) - 1; J = np.array([int(c[2]) for c in COMBOS]) - 1; K = np.array([int(c[4]) for c in COMBOS]) - 1
    # 分解
    P1 = np.zeros((n, 6)); P12 = np.zeros((n, 6, 6))
    for c in range(120):
        P1[:, I[c]] += P[:, c]; P12[:, I[c], J[c]] += P[:, c]
    q2 = P12 / np.where(P1[:, :, None] > 0, P1[:, :, None], 1)                     # q2[r, i, j]
    q3c = P / np.where(P12[:, I, J] > 0, P12[:, I, J], 1)                           # q3(k|i,j) を組番ごとに
    band = np.digitize(P1[:, 0], [0.4, 0.6])                                         # 0,1,2
    yi, yj, yk = I[y], J[y], K[y]
    tr = (m["date"] <= TR[1]).values; ev = (m["date"] >= EV[0]).values
    # c2[i,j,band]
    obs2 = np.zeros((6, 6, 3)); exp2 = np.zeros((6, 6, 3))
    for r in np.where(tr)[0]:
        exp2[yi[r], :, band[r]] += q2[r, yi[r], :]; obs2[yi[r], yj[r], band[r]] += 1
    c2 = (obs2 + A) / (exp2 + A)
    # c3[i,j,k]（組番 c ごと）
    obs3 = np.zeros(120); exp3 = np.zeros(120)
    pair = I * 6 + J
    for r in np.where(tr)[0]:
        sel = pair == (yi[r] * 6 + yj[r]); exp3[sel] += q3c[r, sel]; obs3[y[r]] += 1
    c3 = (obs3 + A) / (exp3 + A)
    # 補正後の確率
    q2a = q2 * np.transpose(c2[:, :, band], (2, 0, 1))
    for i in range(6):
        q2a[:, i, i] = 0
    q2a = q2a / np.where(q2a.sum(2, keepdims=True) > 0, q2a.sum(2, keepdims=True), 1)
    q3a = q3c * c3[None, :]
    den = np.zeros((n, 36))
    for c in range(120):
        den[:, pair[c]] += q3a[:, c]
    q3a = q3a / np.where(den[:, pair] > 0, den[:, pair], 1)
    PI = P1[:, I] * q2a[:, I, J] * q3a
    PI = PI / PI.sum(1, keepdims=True)
    # 評価
    ie = np.where(ev)[0]; days = m["date"].values[ie]
    ll_m = -np.log(np.clip(P[ie, y[ie]], 1e-12, None)); ll_p = -np.log(np.clip(PI[ie, y[ie]], 1e-12, None))
    d = ll_p - ll_m; dd = pd.Series(d).groupby(days).agg(["sum", "count"])
    bi = RNG.integers(0, len(dd), size=(5000, len(dd))); bt = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
    ll_ci = (float(np.quantile(bt, .025)), float(np.quantile(bt, .975)))
    buy = valid[ie] & (O[ie] <= 300) & (PI[ie] * np.where(valid[ie], O[ie], 0) > 1.1)
    hit = np.zeros_like(buy); hit[np.arange(len(ie)), y[ie]] = True
    st = buy.sum(1) * 100.0; rt = np.where(buy & hit, O[ie] * 100, 0).sum(1)
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    nb = int(buy.sum()); roi = g["rt"].sum() / g["st"].sum() if nb else np.nan
    if nb:
        bi = RNG.integers(0, len(g), size=(5000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
        lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
    else:
        lo = hi = np.nan
    judge = "採用候補（段階2へ）" if (nb >= 200 and roi >= 1.1 and lo > 1.0) else "不採用"
    # 参考：c2 が評価期間でも続くか
    obs2e = np.zeros((6, 6, 3)); exp2e = np.zeros((6, 6, 3))
    for r in ie:
        exp2e[yi[r], :, band[r]] += q2[r, yi[r], :]; obs2e[yi[r], yj[r], band[r]] += 1
    msk = (exp2 > 50) & (exp2e > 20)
    rtr, rev = (obs2 / np.where(exp2 > 0, exp2, 1))[msk], (obs2e / np.where(exp2e > 0, exp2e, 1))[msk]
    corr = float(np.corrcoef(rtr, rev)[0, 1]) if msk.sum() > 2 else np.nan
    rows = []
    for i in range(6):
        for j in range(6):
            for bnd in range(3):
                if exp2[i, j, bnd] > 50:
                    rows.append({"1着": i + 1, "2着": j + 1, "1号艇の1着確率": ["〜40%", "40〜60%", "60%〜"][bnd], "学習 実際/市場": obs2[i, j, bnd] / exp2[i, j, bnd],
                                 "学習 件数（市場の見込み）": exp2[i, j, bnd], "評価 実際/市場": obs2e[i, j, bnd] / exp2e[i, j, bnd] if exp2e[i, j, bnd] > 0 else np.nan})
    R = pd.DataFrame(rows); R["ずれ"] = (R["学習 実際/市場"] - 1).abs()
    md = [f"# IDEA-035 展開：2着・3着の値付け（学習 {int(tr.sum())}レース／評価 {len(ie)}レース）", "",
          f"**判定：{judge}**　評価期間で π×オッズ > 1.1（300倍以下）：{nb}点、回収率 {roi:.3f}（日単位95%CI {lo:.3f}〜{hi:.3f}）", "",
          f"参考：3連単対数損失（評価） 市場 {ll_m.mean():.4f} → 補正後 {ll_p.mean():.4f}、差 {d.mean():+.4f}（95%CI {ll_ci[0]:+.4f}〜{ll_ci[1]:+.4f}）", "",
          f"参考：2着の「実際 ÷ 市場の見込み」の学習↔評価の相関（{int(msk.sum())} セル）: {corr:.3f}", "",
          "## 2着の値付けのずれが大きい組（学習、上位20）", "",
          R.sort_values("ずれ", ascending=False).drop(columns="ずれ").head(20).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
