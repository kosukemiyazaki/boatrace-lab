"""IDEA-036 F持ちを市場は正しく値付けしているか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea036.py DATADIR [--out results/idea036.md]
- F持ち：レース当日より前、いまの審査期間（5/1〜10/31・11/1〜4/30）の中で着順 F がある選手（参考：過去120日以内）
- 特徴量：自分が F持ち × コース帯（1／2〜3／4〜6）、内側の艇が F持ち。条件付きロジットで市場の1着確率を補正
- 学習 2023-11〜2025-12 ／ 評価 2026-01〜08。判定：1着対数損失が 0.005 以上小さく 95%CI上限 < 0 → 段階2
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

KEY = ["date", "jcd", "rno"]
TR = ("20231101", "20251231"); EV = ("20260101", "20260831")
RNG = np.random.default_rng(0)

def norm(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2), rno=df["rno"].astype(int))

def period_start(d):
    y, mth = d.year, d.month
    if 5 <= mth <= 10:
        return pd.Timestamp(y, 5, 1)
    return pd.Timestamp(y if mth >= 11 else y - 1, 11, 1)

def f_status(e, rolling_days=None):
    """選手 × 日付 → F持ちか（当日より前の F）"""
    f = e.loc[e["pos_raw"].astype(str).str.startswith("F"), ["toban", "date"]].copy()
    f["d"] = pd.to_datetime(f["date"]); f = f.sort_values("d")
    q = e[["toban", "date"]].drop_duplicates().copy(); q["d"] = pd.to_datetime(q["date"])
    q["ps"] = q["d"].map(period_start) if rolling_days is None else q["d"] - pd.Timedelta(days=rolling_days)
    m = q.merge(f[["toban", "d"]].rename(columns={"d": "fd"}), on="toban", how="left")
    m["hit"] = (m["fd"] < m["d"]) & (m["fd"] >= m["ps"])
    return m.groupby(["toban", "date"])["hit"].any().rename("fh").reset_index()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea036.md")
    a = ap.parse_args()
    races = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    races = races[(races["date"] >= TR[0]) & (races["date"] <= EV[1])]
    m = races.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY)
    O = m[COMBOS].to_numpy(float); ok = (np.isfinite(O) & (O > 0)).sum(1) >= 60
    m, O = m[ok].reset_index(drop=True), O[ok]
    inv = np.where(np.isfinite(O) & (O > 0), 1 / np.where(O > 0, O, 1), 0); P = inv / inv.sum(1, keepdims=True)
    first = np.array([int(c[0]) for c in COMBOS]); pw = np.stack([P[:, first == b].sum(1) for b in range(1, 7)], 1)
    y = m["tri_combo"].astype(str).str[0].astype(int).values - 1
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban", "pos_raw"]))
    b = norm(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet", columns=KEY + [f"ex_course{i}" for i in range(1, 7)]))
    ex = m[KEY].merge(b, on=KEY, how="left")[[f"ex_course{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    cur = np.where(np.isfinite(ex) & (ex >= 1) & (ex <= 6), ex, np.arange(1, 7)[None, :]).astype(int)
    results = {}
    for label, rd in (("審査期間", None), ("過去120日", 120)):
        fs = f_status(e, rd)
        ef = e.merge(fs, on=["toban", "date"], how="left")
        F = m[KEY].merge(ef.pivot_table(index=KEY, columns="boat", values="fh", aggfunc="first").reset_index(), on=KEY, how="left")[[1, 2, 3, 4, 5, 6]]
        F = F.astype(float).fillna(0).to_numpy()
        n = len(m); X = np.zeros((n, 6, 4))
        # 各コースにいる艇の F
        fc = np.zeros((n, 7))
        for bb in range(6):
            fc[np.arange(n), cur[:, bb]] = F[:, bb]
        for bb in range(6):
            c = cur[:, bb]
            grp = np.where(c == 1, 0, np.where(c <= 3, 1, 2))
            X[np.arange(n), bb, grp] = F[:, bb]
            X[:, bb, 3] = np.where(c >= 2, fc[np.arange(n), np.clip(c - 1, 1, 6)], 0)
        L = np.log(np.clip(pw, 1e-9, None))
        tr = ((m["date"] >= TR[0]) & (m["date"] <= TR[1])).values; ev = (m["date"] >= EV[0]).values
        def probs(w, idx):
            s = L[idx] + X[idx] @ w; s -= s.max(1, keepdims=True); q = np.exp(s); return q / q.sum(1, keepdims=True)
        def loss(w):
            q = probs(w, tr); return -np.log(np.clip(q[np.arange(q.shape[0]), y[tr]], 1e-12, None)).mean() + 1e-3 * (w ** 2).sum()
        w = minimize(loss, np.zeros(4), method="L-BFGS-B").x
        Q = probs(w, ev); ye = y[ev]
        la = -np.log(np.clip(Q[np.arange(len(ye)), ye], 1e-12, None)); lm = -np.log(np.clip(pw[ev][np.arange(len(ye)), ye], 1e-12, None))
        d = la - lm; days = m["date"].values[ev]; dd = pd.Series(d).groupby(days).agg(["sum", "count"])
        bi = RNG.integers(0, len(dd), size=(5000, len(dd))); bt = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
        ci = (float(np.quantile(bt, .025)), float(np.quantile(bt, .975)))
        # 参考：F持ちの艇の 実際の1着 ÷ 市場の見込み、単勝相当の回収率（3連単の頭の確率から）
        rows = []
        won = np.zeros((n, 6)); won[np.arange(n), y] = 1
        for per, msk in (("学習", tr), ("評価", ev)):
            for gname, g in (("1コース", 0), ("2〜3コース", 1), ("4〜6コース", 2)):
                sel = (X[msk][:, :, g] > 0)
                rows.append({"定義": label, "期間": per, "コース帯": gname, "F持ちの艇": int(sel.sum()),
                             "実際の1着": int(won[msk][sel].sum()), "市場の見込み": float(pw[msk][sel].sum()),
                             "実際÷見込み": float(won[msk][sel].sum() / max(pw[msk][sel].sum(), 1e-9))})
        results[label] = dict(w=w, lm=lm.mean(), la=la.mean(), diff=d.mean(), ci=ci, rows=rows, nf=int((F > 0).sum()), n=n,
                              go=(d.mean() <= -0.005 and ci[1] < 0))
    main_r = results["審査期間"]
    md = [f"# IDEA-036 F持ちの値付け（学習 {TR[0]}〜{TR[1]}／評価 {EV[0]}〜{EV[1]}）", "",
          f"**判定（審査期間の定義）：{'段階2へ' if main_r['go'] else '打ち切り'}**", ""]
    for label, r in results.items():
        md += [f"## 定義：{label}（F持ちの出走 {r['nf']}件 / {r['n']}レース）", "",
               f"1着の対数損失（評価）：市場 {r['lm']:.4f} → 補正 {r['la']:.4f}、差 {r['diff']:+.4f}（95%CI {r['ci'][0]:+.4f}〜{r['ci'][1]:+.4f}）", "",
               "係数：自分がF持ち（1コース {:.3f}／2〜3コース {:.3f}／4〜6コース {:.3f}）、内側の艇がF持ち {:.3f}".format(*r["w"]), "",
               pd.DataFrame(r["rows"]).drop(columns="定義").to_markdown(index=False, floatfmt=".3f"), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
