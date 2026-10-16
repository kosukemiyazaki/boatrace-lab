"""IDEA-038 小さな偏りを全部重ねる（β ＋ F持ち ＋ 展開 c2/c3）。docs/ideas.md の登録どおり
usage: python scripts/exp_idea038.py DATADIR [--out results/idea038.md]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import f_status, norm

KEY = ["date", "jcd", "rno"]
TR = ("20231101", "20251231"); EV = ("20260101", "20260831"); A = 20.0
RNG = np.random.default_rng(0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea038.md")
    a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= TR[0]) & (r["date"] <= EV[1])]
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); ok = v.sum(1) >= 60
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values; ok &= ~pd.isna(y)
    m, O, v, y = m[ok].reset_index(drop=True), O[ok], v[ok], y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); L = np.log(np.clip(P, 1e-12, None))
    I = np.array([int(c[0]) for c in COMBOS]) - 1; J = np.array([int(c[2]) for c in COMBOS]) - 1
    tr = (m["date"] <= TR[1]).values; ev = (m["date"] >= EV[0]).values
    # (1) β
    def nllb(b):
        s = b * L[tr]; s -= s.max(1, keepdims=True); return -(s[np.arange(tr.sum()), y[tr]] - np.log(np.exp(s).sum(1))).mean()
    beta = minimize_scalar(nllb, bounds=(0.8, 1.3), method="bounded").x
    Pb = np.exp(beta * L); Pb /= Pb.sum(1, keepdims=True)
    # (2) F持ち（艇番＝コース、展示の進入があればそれ）
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban", "pos_raw"]))
    ef = e.merge(f_status(e), on=["toban", "date"], how="left")
    F = m[KEY].merge(ef.pivot_table(index=KEY, columns="boat", values="fh", aggfunc="first").reset_index(), on=KEY, how="left")[[1, 2, 3, 4, 5, 6]].astype(float).fillna(0).to_numpy()
    b = norm(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet", columns=KEY + [f"ex_course{i}" for i in range(1, 7)]))
    ex = m[KEY].merge(b, on=KEY, how="left")[[f"ex_course{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    cur = np.where(np.isfinite(ex) & (ex >= 1) & (ex <= 6), ex, np.arange(1, 7)[None, :]).astype(int)
    fc = np.zeros((n, 7))
    for bb in range(6):
        fc[np.arange(n), cur[:, bb]] = F[:, bb]
    X = np.zeros((n, 6, 4))
    for bb in range(6):
        c = cur[:, bb]; grp = np.where(c == 1, 0, np.where(c <= 3, 1, 2))
        X[np.arange(n), bb, grp] = F[:, bb]; X[:, bb, 3] = np.where(c >= 2, fc[np.arange(n), np.clip(c - 1, 1, 6)], 0)
    P1 = np.zeros((n, 6))
    for c in range(120):
        P1[:, I[c]] += Pb[:, c]
    Lw = np.log(np.clip(P1, 1e-12, None)); yi = I[y]
    def lossf(w):
        s = Lw[tr] + X[tr] @ w; s -= s.max(1, keepdims=True); q = np.exp(s); q /= q.sum(1, keepdims=True)
        return -np.log(np.clip(q[np.arange(tr.sum()), yi[tr]], 1e-12, None)).mean() + 1e-3 * (w ** 2).sum()
    wf = minimize(lossf, np.zeros(4), method="L-BFGS-B").x
    mult = np.exp(np.einsum("rbk,k->rb", X, wf))
    Pf = Pb * mult[:, I]; Pf /= Pf.sum(1, keepdims=True)
    # (3) 展開 c2/c3（学習で推定）
    P1 = np.zeros((n, 6)); P12 = np.zeros((n, 6, 6))
    for c in range(120):
        P1[:, I[c]] += Pf[:, c]; P12[:, I[c], J[c]] += Pf[:, c]
    q2 = P12 / np.where(P1[:, :, None] > 0, P1[:, :, None], 1); q3 = Pf / np.where(P12[:, I, J] > 0, P12[:, I, J], 1)
    band = np.digitize(P1[:, 0], [0.4, 0.6]); yj = J[y]; pair = I * 6 + J
    obs2 = np.zeros((6, 6, 3)); exp2 = np.zeros((6, 6, 3)); obs3 = np.zeros(120); exp3 = np.zeros(120)
    for r_ in np.where(tr)[0]:
        exp2[yi[r_], :, band[r_]] += q2[r_, yi[r_], :]; obs2[yi[r_], yj[r_], band[r_]] += 1
        sel = pair == (yi[r_] * 6 + yj[r_]); exp3[sel] += q3[r_, sel]; obs3[y[r_]] += 1
    c2 = (obs2 + A) / (exp2 + A); c3 = (obs3 + A) / (exp3 + A)
    q2a = q2 * np.transpose(c2[:, :, band], (2, 0, 1))
    for i in range(6):
        q2a[:, i, i] = 0
    q2a /= np.where(q2a.sum(2, keepdims=True) > 0, q2a.sum(2, keepdims=True), 1)
    q3a = q3 * c3[None, :]; den = np.zeros((n, 36))
    for c in range(120):
        den[:, pair[c]] += q3a[:, c]
    q3a /= np.where(den[:, pair] > 0, den[:, pair], 1)
    PI = P1[:, I] * q2a[:, I, J] * q3a; PI /= PI.sum(1, keepdims=True)
    # 評価
    ie = np.where(ev)[0]; days = m["date"].values[ie]
    d = -np.log(np.clip(PI[ie, y[ie]], 1e-12, None)) + np.log(np.clip(P[ie, y[ie]], 1e-12, None))
    dd = pd.Series(d).groupby(days).agg(["sum", "count"]); bi = RNG.integers(0, len(dd), size=(3000, len(dd)))
    bt = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
    E = PI[ie] * np.where(v[ie], O[ie], 0); hit = np.zeros_like(E, bool); hit[np.arange(len(ie)), y[ie]] = True
    rows = []
    for lo_, hi_ in ((0, .7), (.7, .8), (.8, .9), (.9, 1.0), (1.0, 1.1), (1.1, 9e9)):
        sel = v[ie] & (O[ie] <= 300) & (E > lo_) & (E <= hi_)
        st = sel.sum(1) * 100.0; rt = np.where(sel & hit, O[ie] * 100, 0).sum(1)
        g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
        if len(g):
            bb_ = RNG.integers(0, len(g), size=(3000, len(g))); b2 = g["rt"].values[bb_].sum(1) / g["st"].values[bb_].sum(1)
            rows.append({"π×オッズ": f"{lo_}〜{hi_ if hi_ < 9e9 else ''}", "点数": int(sel.sum()), "回収率": g["rt"].sum() / g["st"].sum(),
                         "95%CI下": float(np.quantile(b2, .025)), "95%CI上": float(np.quantile(b2, .975))})
        else:
            rows.append({"π×オッズ": f"{lo_}〜", "点数": 0})
    R = pd.DataFrame(rows)
    top = R.iloc[-1]
    judge = "採用候補（段階2へ）" if (top.get("点数", 0) >= 200 and top.get("回収率", 0) >= 1.1 and top.get("95%CI下", 0) > 1.0) else "不採用"
    md = [f"# IDEA-038 小さな偏りを全部重ねる（学習 {int(tr.sum())}／評価 {len(ie)}レース）", "",
          f"**判定：{judge}**", "",
          f"推定値：β = {beta:.3f}、F持ちの係数 {np.round(wf, 3).tolist()}（1コース／2〜3／4〜6／内側がF）", "",
          f"3連単対数損失（評価）の市場との差：{d.mean():+.4f}（95%CI {np.quantile(bt, .025):+.4f}〜{np.quantile(bt, .975):+.4f}）", "",
          "## 補正後の π × 確定オッズ の帯ごとの回収率（300倍以下、評価期間）", "", R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
