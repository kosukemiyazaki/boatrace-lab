"""IDEA-028: 展示のあとの風・波の変化（docs/ideas.md の登録どおり。段階1＝本番の気象を知っていたらの上限）
usage: python scripts/exp_idea028.py DATADIR [--out results/idea028.md]
- 市場: 確定3連単オッズから出した各艇の1着確率
- 特徴量（コース 1〜6 ごと）: Δ風速（本番 K − 展示 beforeinfo）、Δ波、風向が変わったか（展示の風向番号→本番の方位の対応は学習期間で場ごとに最頻値）
- 条件付きロジット: 得点 = log(市場の1着確率) + w·特徴量、L2 1e-3。学習 2025-10〜2026-04、評価 2026-05〜08
- 判定: 評価の1着対数損失が市場より 0.005 以上小さく、日単位ブートストラップ95%CIが0をまたがない → 段階2
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

KEY = ["date", "jcd", "rno"]
RNG = np.random.default_rng(0)

def nz(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea028.md")
    a = ap.parse_args()
    r = nz(pd.read_parquet(f"{a.datadir}/races.parquet"))
    b = nz(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet"))
    o = nz(pd.read_parquet(f"{a.datadir}/odds.parquet"))
    m = r[r["tri_combo"].notna()].merge(b[KEY + ["wind", "wind_dir", "wave"] + [f"ex_course{i}" for i in range(1, 7)]], on=KEY, suffixes=("_race", "_exh"))
    m = m.merge(o, on=KEY)
    m = m[(m["date"] >= "20251001") & (m["date"] <= "20260831")].reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); ok = (np.isfinite(O) & (O > 0)).sum(1) >= 60
    for c in ("wind_race", "wind_exh", "wave_race", "wave_exh"):
        ok &= pd.to_numeric(m[c], errors="coerce").notna().values
    ok &= m["wind_dir_exh"].notna().values & m["wind_dir_race"].notna().values
    m, O = m[ok].reset_index(drop=True), O[ok]
    inv = np.where(np.isfinite(O) & (O > 0), 1 / np.where(O > 0, O, 1), 0); p3 = inv / inv.sum(1, keepdims=True)
    first = np.array([int(c[0]) for c in COMBOS])
    pw = np.stack([p3[:, first == bb].sum(1) for bb in range(1, 7)], 1)
    y = m["tri_combo"].str[0].astype(int).values - 1
    tr = (m["date"] <= "20260430").values; ev = (m["date"] >= "20260501").values
    # 風向の対応（学習期間で場ごとに、展示の番号 → 本番の方位の最頻値）
    mt = m[tr].groupby(["jcd", "wind_dir_exh"])["wind_dir_race"].agg(lambda s: s.value_counts().index[0])
    mapped = pd.Series([mt.get((j, d), None) for j, d in zip(m["jcd"], m["wind_dir_exh"])])
    dchg = ((mapped.notna()) & (mapped.values != m["wind_dir_race"].values)).astype(float).values
    dw = pd.to_numeric(m["wind_race"]).values - pd.to_numeric(m["wind_exh"]).values
    dv = pd.to_numeric(m["wave_race"]).values - pd.to_numeric(m["wave_exh"]).values
    course = np.stack([pd.to_numeric(m[f"ex_course{i}"], errors="coerce").fillna(i).clip(1, 6).astype(int).values for i in range(1, 7)], 1)
    # X: (n, 6艇, 18) — コース c の艇にだけ [dw, dv, dchg] を c 番目の位置に置く
    n = len(m); X = np.zeros((n, 6, 18))
    for bb in range(6):
        for c in range(1, 7):
            sel = course[:, bb] == c
            X[sel, bb, (c - 1) * 3 + 0] = dw[sel]; X[sel, bb, (c - 1) * 3 + 1] = dv[sel]; X[sel, bb, (c - 1) * 3 + 2] = dchg[sel]
    L = np.log(np.clip(pw, 1e-9, None))
    def probs(w, idx):
        s = L[idx] + X[idx] @ w; s -= s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
    def loss(w):
        P = probs(w, tr); return -np.log(P[np.arange(P.shape[0]), y[tr]]).mean() + 1e-3 * (w ** 2).sum()
    w = minimize(loss, np.zeros(18), method="L-BFGS-B").x
    Pe = probs(w, ev); ye = y[ev]
    l_adj = -np.log(np.clip(Pe[np.arange(len(ye)), ye], 1e-12, None)); l_mkt = -np.log(np.clip(pw[ev][np.arange(len(ye)), ye], 1e-12, None))
    diff = l_adj - l_mkt; days = m["date"].values[ev]
    dd = pd.Series(diff).groupby(days).agg(["sum", "count"])
    bi = RNG.integers(0, len(dd), size=(5000, len(dd)))
    boot = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
    ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    go = diff.mean() <= -0.005 and ci[1] < 0
    chg = (np.abs(dw) >= 2) | (dchg == 1) | (np.abs(dv) >= 2)
    W = pd.DataFrame(w.reshape(6, 3), index=[f"{c}コース" for c in range(1, 7)], columns=["Δ風速", "Δ波", "風向変化"])
    md = [f"### IDEA-028 展示のあとの気象の変化（段階1：本番の気象を知っていたらの上限）", "",
          f"対象 {n}レース（学習 {int(tr.sum())}、評価 {int(ev.sum())}）。評価期間で 風速が2m以上変化 {np.mean(np.abs(dw[ev]) >= 2):.1%}、風向が変化 {np.mean(dchg[ev]):.1%}、波が2cm以上変化 {np.mean(np.abs(dv[ev]) >= 2):.1%}", "",
          f"1着の対数損失（評価）: 市場 {l_mkt.mean():.4f} → 調整後 {l_adj.mean():.4f}、差 {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}）", "",
          f"変化があったレースだけ（{int(chg[ev].sum())}レース）: 差 {diff[chg[ev]].mean():+.4f}", "",
          f"**判定：{'段階2へ' if go else '打ち切り'}**", "", "係数（学習期間）", "", W.to_markdown(floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
