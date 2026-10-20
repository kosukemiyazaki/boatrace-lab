"""IDEA-053 荒れにくい偏りを重ねる（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea053.py DATADIR STOREDIR [--hist /tmp/claude-0/ds_hist] [--out results/idea053.md]
- IDEA-051 の探索期間で選ばれた21区分（results/idea051.md）を 0/1 の特徴量にし、logit(市場の1号艇1着確率) に足す補正を
  L2 正則化のロジスティックで学習（探索 2023-11〜2025-12）
- 評価 2026-01〜08：補正後÷市場 が 1.15・1.25 以上のレースで (a) 1番人気1点、(b) 1号艇1着の組で π×オッズ > 1.0
  （π = P^1.06 を、1号艇1着の組の合計が補正後の確率になるよう引き上げ、残りを同じ割合で縮める）
- 参考：2014〜2026（オッズなし）で、補正の合計 Σw が大きいレースの「1号艇1着を全部」の年別回収率
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY, TR, EV, FIRST, boat_feats, segments, boot_ratio

RNG = np.random.default_rng(0)
# IDEA-051 の探索で選ばれた21区分（results/idea051.md）
SEL = [("場", "24"), ("場", "18"), ("場", "13"), ("場", "08"), ("場", "07"), ("場", "05"), ("場", "03"), ("グレード", "SG"),
       ("1号艇の級別", "B1"), ("1号艇の級別", "B2"), ("1号艇の年齢帯", "〜29"), ("節の日目", "7"),
       ("レース番号×時刻帯", "3R・〜15時"), ("レース番号×時刻帯", "3R・〜18時"), ("レース番号×時刻帯", "12R・〜21時"), ("レース番号×時刻帯", "7R・〜18時"),
       ("レース番号×時刻帯", "2R・〜15時"), ("レース番号×時刻帯", "8R・〜18時"), ("級別×F持ち", "B1・なし"), ("級別×F持ち", "B2・なし"), ("級別×F持ち", "B1・F持ち")]
LAM = 1.0

def design(S, n):
    X = np.zeros((n, len(SEL)))
    for k, (s, v) in enumerate(SEL):
        if s in S:
            X[:, k] = (np.asarray(S[s]).astype(str) == v)
    return X

def fit(X, off, y):
    def f(w):
        z = off + X @ w; p = 1 / (1 + np.exp(-z))
        ll = -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))).sum() + 0.5 * LAM * (w ** 2).sum()
        g = X.T @ (p - y) + LAM * w
        return ll, g
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x

def boot(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return (g["rt"].sum() / g["st"].sum() if len(g) else np.nan), np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(g["rt"].sum() / g["st"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--hist", default="/tmp/claude-0/ds_hist")
    ap.add_argument("--out", default="results/idea053.md"); a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo", "tri_pay"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]))
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(boat_feats(e), on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); p1 = P[:, FIRST == 1].sum(1); m["p1"] = p1
    win1 = (FIRST[y] == 1).astype(float); hit = np.zeros_like(v); hit[np.arange(n), y] = True
    days = m["date"].values; tr = (days >= TR[0]) & (days <= TR[1]); ev = days >= EV[0]
    S = segments(m, tr, True, a.store); X = design(S, n)
    off = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1))
    w = fit(X[tr], off[tr], win1[tr])
    padj = 1 / (1 + np.exp(-(off + X @ w))); ratio = padj / p1
    # π：P^1.06 を基準に、1号艇1着の組の合計を padj に、残りを 1−padj に
    Pb = P ** 1.06; Pb /= Pb.sum(1, keepdims=True); f1 = FIRST[None, :] == 1
    q1 = (Pb * f1).sum(1, keepdims=True); pi = np.where(f1, Pb * padj[:, None] / q1, Pb * (1 - padj[:, None]) / (1 - q1))
    fav = np.argmin(np.where(v, O, np.inf), 1)
    stA = np.full(n, 100.0); rtA = np.where(hit[np.arange(n), fav], O[np.arange(n), fav] * 100, 0)
    bB = v & f1 & (pi * np.where(v, O, 0) > 1.0); stB = bB.sum(1) * 100.0; rtB = np.where(bB & hit, O * 100, 0).sum(1)
    rows = []; judge_ok = False
    for thr in (1.0, 1.15, 1.25):
        for per, msk in (("探索", tr), ("評価", ev)):
            s = msk & (ratio >= thr) if thr > 1.0 else msk
            if s.sum() == 0:
                rows.append({"補正後÷市場": f"≥{thr}" if thr > 1 else "全レース", "期間": per, "レース": 0}); continue
            rr, rlo, rhi = boot_ratio(days[s], win1[s], p1[s])
            ra_, alo, ahi = boot(days[s], stA[s], rtA[s]); rb, blo, bhi = boot(days[s], stB[s], rtB[s])
            rows.append({"補正後÷市場": f"≥{thr}" if thr > 1 else "全レース", "期間": per, "レース": int(s.sum()),
                         "1号艇 実際÷見込み": rr, "CI下": rlo, "CI上": rhi, "補正後の見込み÷市場（平均）": padj[s].sum() / p1[s].sum(),
                         "(a) 1番人気1点": ra_, "(a) CI下": alo, "(a) CI上": ahi, "(b) 点数": int(bB[s].sum()), "(b) π×オッズ>1": rb, "(b) CI下": blo, "(b) CI上": bhi})
            if per == "評価" and thr > 1.0:
                if (s.sum() >= 300 and ra_ >= 1.1 and alo > 1.0) or (bB[s].sum() >= 300 and rb >= 1.1 and blo > 1.0):
                    judge_ok = True
    R = pd.DataFrame(rows)
    W = pd.DataFrame({"区分": [s for s, _ in SEL], "値": [v_ for _, v_ in SEL], "係数 w": w, "倍率の目安 exp(w)": np.exp(w),
                      "探索 該当レース": X[tr].sum(0).astype(int), "評価 該当レース": X[ev].sum(0).astype(int)})
    dist = pd.Series(ratio[ev]).describe(percentiles=[.5, .9, .99]).to_frame("評価の 補正後÷市場").T
    judge = "採用候補（前向きの検証へ）" if judge_ok else "不採用"
    # 参考：12年分（オッズなし）
    hist_md = ""
    if os.path.isdir(a.hist):
        rh = norm(pd.read_parquet(f"{a.hist}/races.parquet"))[KEY + ["tri_combo", "tri_pay"]]
        eh = norm(pd.read_parquet(f"{a.hist}/entries.parquet", columns=KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]))
        ra = pd.concat([rh, r]).dropna(subset=["tri_combo"]).drop_duplicates(KEY)
        ea = pd.concat([eh, e]); bad = ea[ea["pos_raw"].astype(str).str.match(r"^[KFL]")][KEY].drop_duplicates()
        ra = ra.merge(bad.assign(bad=1), on=KEY, how="left"); ra = ra[ra["bad"].isna()].drop(columns="bad")
        ra = ra.merge(boat_feats(ea), on=KEY, how="left"); ra = ra[ra["tri_combo"].isin(COMBOS)].reset_index(drop=True)
        SH = segments(ra, ((ra["date"] >= TR[0]) & (ra["date"] <= TR[1])).values, False, None)
        score = design(SH, len(ra)) @ w   # グレード（SG）は過去分に無いので 0
        sc_ev = (X @ w)[ev]; frac = float((ratio[ev] >= 1.15).mean())   # 評価で ≥1.15 に入るレースの割合と同じ割合を上から選ぶ
        thr_s = np.quantile(score, 1 - frac) if frac > 0 else np.inf
        w1 = ra["tri_combo"].str[0] == "1"; pay = np.where(w1, pd.to_numeric(ra["tri_pay"], errors="coerce").fillna(0), 0.0)
        year = ra["date"].str[:4].values; sel = score >= thr_s
        g = pd.DataFrame({"y": year, "pay": pay, "sel": sel})
        base = g.groupby("y").apply(lambda d: d["pay"].sum() / (len(d) * 2000))
        top = g[g["sel"]].groupby("y").apply(lambda d: pd.Series({"レース": len(d), "回収率": d["pay"].sum() / (len(d) * 2000)}))
        H = top.join(base.rename("全レース（基準）")); H["差"] = H["回収率"] - H["全レース（基準）"]
        hist_md = "\n".join([f"## 参考：2014〜2026（オッズなし）、Σw の上位 {frac:.1%}（評価で補正後÷市場≥1.15 に入る割合と同じ）の「1号艇1着を全部」", "",
                             f"Σw のしきい値 {thr_s:.3f}（評価期間の ≥1.15 のレースの Σw の中央値 {np.median(sc_ev[ratio[ev] >= 1.15]) if frac > 0 else float('nan'):.3f}）。グレードは過去分に無いため SG の係数は使わない。", "",
                             H.to_markdown(floatfmt=".3f"), "",
                             f"基準を上回った年：{int((H['差'] > 0).sum())}／{len(H)}、100%を超えた年：{int((H['回収率'] >= 1).sum())}"])
    md = [f"# IDEA-053 荒れにくい偏りを重ねる（学習 {TR[0]}〜{TR[1]}／評価 {EV[0]}〜{EV[1]}、{n}レース、L2 λ={LAM}）", "",
          f"**判定：{judge}**", "", "## 区分ごとの係数（探索で学習）", "", W.to_markdown(index=False, floatfmt=".3f"), "",
          "## 補正後÷市場 の分布（評価）", "", dist.to_markdown(floatfmt=".3f"), "",
          "## 補正後÷市場 のしきい値ごとの結果", "", R.to_markdown(index=False, floatfmt=".3f"), "", hist_md]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
