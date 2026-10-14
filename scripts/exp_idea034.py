"""IDEA-034 当日の水面の癖（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea034.py DATADIR [--out results/idea034.md]
- 同じ日・同じ場の前のレースで、進入コースごとの「1着 − 市場の見込み」「2着以内 − 市場の見込み」の合計 ÷ (前のレース数 + 3)
- 条件付きロジット：log(市場の1着確率) + w·特徴量。学習 2023-10〜2025-12 ／ 評価 2026-01〜08
- 判定：評価の1着対数損失が市場より 0.005 以上小さく、日単位95%CI上限 < 0 → 段階2
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

KEY = ["date", "jcd", "rno"]
TR = ("20231001", "20251231"); EV = ("20260101", "20260831")
RNG = np.random.default_rng(0)

def norm(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2), rno=df["rno"].astype(int))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea034.md")
    a = ap.parse_args()
    races = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    races = races[(races["date"] >= TR[0]) & (races["date"] <= EV[1])]
    o3 = norm(pd.read_parquet(f"{a.datadir}/odds.parquet"))
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "course"]))
    b = norm(pd.read_parquet(f"{a.datadir}/beforeinfo.parquet", columns=KEY + [f"ex_course{i}" for i in range(1, 7)]))
    m = races.merge(o3, on=KEY)
    O = m[COMBOS].to_numpy(float); ok = (np.isfinite(O) & (O > 0)).sum(1) >= 60
    m, O = m[ok].reset_index(drop=True), O[ok]
    inv = np.where(np.isfinite(O) & (O > 0), 1 / np.where(O > 0, O, 1), 0); P = inv / inv.sum(1, keepdims=True)
    f1 = np.array([int(c[0]) for c in COMBOS]); f2 = np.array([int(c[2]) for c in COMBOS])
    pw = np.stack([P[:, f1 == k].sum(1) for k in range(1, 7)], 1)                      # 艇ごとの1着確率
    pt = np.stack([P[:, (f1 == k) | (f2 == k)].sum(1) for k in range(1, 7)], 1)        # 2着以内
    tri = m["tri_combo"].astype(str)
    w1 = tri.str[0].astype(int).values; w2 = tri.str[2].astype(int).values
    # 実際の進入コース（結果側）
    crs = e.pivot_table(index=KEY, columns="boat", values="course", aggfunc="first")
    crs = m[KEY].merge(crs.reset_index(), on=KEY, how="left")[[1, 2, 3, 4, 5, 6]].to_numpy(float)
    # このレースでのコース（判定側）：展示の進入があればそれ、なければ艇番
    ex = m[KEY].merge(b, on=KEY, how="left")[[f"ex_course{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    cur = np.where(np.isfinite(ex) & (ex >= 1) & (ex <= 6), ex, np.arange(1, 7)[None, :]).astype(int)
    # レースごとのコース別の残差
    n = len(m); rw = np.zeros((n, 6)); rt = np.zeros((n, 6)); valid = np.isfinite(crs).all(1)
    won = np.zeros((n, 6)); won[np.arange(n), w1 - 1] = 1
    top = won.copy(); top[np.arange(n), w2 - 1] = 1
    for bb in range(6):
        c = np.where(valid, crs[:, bb], 1).astype(int) - 1
        rw[np.arange(n), c] += np.where(valid, won[:, bb] - pw[:, bb], 0)
        rt[np.arange(n), c] += np.where(valid, top[:, bb] - pt[:, bb], 0)
    # 同じ日・同じ場の前のレースの累計（このレースは含めない）
    order = np.lexsort((m["rno"].values, m["jcd"].values, m["date"].values))
    grp = (m["date"] + m["jcd"]).values
    cw = np.zeros((n, 6)); ct = np.zeros((n, 6)); cn = np.zeros(n)
    sw = np.zeros(6); st = np.zeros(6); sn = 0; last = None
    for i in order:
        if grp[i] != last:
            sw[:] = 0; st[:] = 0; sn = 0; last = grp[i]
        cw[i] = sw; ct[i] = st; cn[i] = sn
        if valid[i]:
            sw += rw[i]; st += rt[i]; sn += 1
    fw, ft = cw / (cn[:, None] + 3), ct / (cn[:, None] + 3)
    # 特徴量 X: (n, 6艇, 12) — 艇のこのレースでのコース c の位置に [fw_c, ft_c]
    X = np.zeros((n, 6, 12))
    for bb in range(6):
        c = cur[:, bb] - 1
        X[np.arange(n), bb, c * 2] = fw[np.arange(n), c]; X[np.arange(n), bb, c * 2 + 1] = ft[np.arange(n), c]
    L = np.log(np.clip(pw, 1e-9, None)); y = w1 - 1
    tr = (m["date"] <= TR[1]).values; ev = (m["date"] >= EV[0]).values
    def probs(w, idx):
        s = L[idx] + X[idx] @ w; s -= s.max(1, keepdims=True); q = np.exp(s); return q / q.sum(1, keepdims=True)
    def loss(w):
        q = probs(w, tr); return -np.log(np.clip(q[np.arange(q.shape[0]), y[tr]], 1e-12, None)).mean() + 1e-3 * (w ** 2).sum()
    w = minimize(loss, np.zeros(12), method="L-BFGS-B").x
    Q = probs(w, ev); ye = y[ev]
    l_adj = -np.log(np.clip(Q[np.arange(len(ye)), ye], 1e-12, None)); l_mkt = -np.log(np.clip(pw[ev][np.arange(len(ye)), ye], 1e-12, None))
    diff = l_adj - l_mkt; days = m["date"].values[ev]
    dd = pd.Series(diff).groupby(days).agg(["sum", "count"])
    bi = RNG.integers(0, len(dd), size=(5000, len(dd)))
    boot = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
    ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    go = diff.mean() <= -0.005 and ci[1] < 0
    late = cn[ev] >= 6
    # 当日の偏りの大きさ（参考）：前のレースが6以上のときの「1コースの1着 − 見込み」の平均（÷前のレース数）
    c1 = cw[:, 0] / np.maximum(cn, 1)
    W = pd.DataFrame(w.reshape(6, 2), index=[f"{c}コース" for c in range(1, 7)], columns=["1着の残差", "2着以内の残差"])
    md = [f"# IDEA-034 当日の水面の癖（学習 {int(tr.sum())}レース／評価 {int(ev.sum())}レース）", "",
          f"1着の対数損失（評価）: 市場 {l_mkt.mean():.4f} → 当日の癖で調整 {l_adj.mean():.4f}、差 {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}）", "",
          f"参考：前のレースが6以上のレースだけ（{int(late.sum())}レース）：差 {diff[late].mean():+.4f}", "",
          f"参考：当日の「1コースの1着 − 市場の見込み」（前のレース6以上、1レースあたり）：標準偏差 {np.std(c1[cn >= 6]):.3f}、"
          f"上位5% {np.quantile(c1[cn >= 6], .95):+.3f}、下位5% {np.quantile(c1[cn >= 6], .05):+.3f}", "",
          f"**判定：{'段階2へ' if go else '打ち切り'}**", "", "係数（学習期間）", "", W.to_markdown(floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
