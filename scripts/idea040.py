"""IDEA-040 「市場が一番取りこぼしている組」を前向きに確かめる（docs/ideas.md の登録どおり）
usage:
  python scripts/idea040.py fit DATADIR [--out models/idea040_frozen.npz]        # 2023-11〜2026-08 で β・F持ち・c2・c3 を推定して凍結
  python scripts/idea040.py eval DATADIR STOREDIR [--frozen ...] [--out results/idea040.md]   # 2026-10-17 以降を判定
補正の順番は IDEA-038 と同じ：π ∝ p^β → F持ちの1着補正 → 2着・3着の形（c2[i,j,1号艇の帯]・c3[i,j,k]）
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import f_status, norm
from live_eval import KEY, _read

FIT = ("20231101", "20260831"); START = "20261017"; A = 20.0; THR = 0.9; OMAX = 300
I = np.array([int(c[0]) for c in COMBOS]) - 1; J = np.array([int(c[2]) for c in COMBOS]) - 1; PAIR = I * 6 + J
RNG = np.random.default_rng(0)

def market(O):
    v = np.isfinite(O) & (O > 0)
    P = np.where(v, 1 / np.where(v, O, 1), 0)
    return P / np.where(P.sum(1, keepdims=True) > 0, P.sum(1, keepdims=True), 1), v

def f_features(datadir, keys):
    """keys(date,jcd,rno) → X (n,6,4)：自分がF持ち×コース帯3つ、内側の艇がF持ち。コースは展示の進入があればそれ、なければ艇番"""
    e = norm(pd.read_parquet(f"{datadir}/entries.parquet", columns=KEY + ["boat", "toban", "pos_raw"]))
    ef = e.merge(f_status(e), on=["toban", "date"], how="left")
    F = keys.merge(ef.pivot_table(index=KEY, columns="boat", values="fh", aggfunc="first").reset_index(), on=KEY, how="left")
    F = F.reindex(columns=KEY + [1, 2, 3, 4, 5, 6])[[1, 2, 3, 4, 5, 6]].astype(float).fillna(0).to_numpy()
    b = norm(pd.read_parquet(f"{datadir}/beforeinfo.parquet", columns=KEY + [f"ex_course{i}" for i in range(1, 7)]))
    ex = keys.merge(b, on=KEY, how="left")[[f"ex_course{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = len(keys); cur = np.where(np.isfinite(ex) & (ex >= 1) & (ex <= 6), ex, np.arange(1, 7)[None, :]).astype(int)
    fc = np.zeros((n, 7))
    for bb in range(6):
        fc[np.arange(n), cur[:, bb]] = F[:, bb]
    X = np.zeros((n, 6, 4))
    for bb in range(6):
        c = cur[:, bb]; grp = np.where(c == 1, 0, np.where(c <= 3, 1, 2))
        X[np.arange(n), bb, grp] = F[:, bb]; X[:, bb, 3] = np.where(c >= 2, fc[np.arange(n), np.clip(c - 1, 1, 6)], 0)
    return X

def decompose(Pf):
    n = len(Pf); P1 = np.zeros((n, 6)); P12 = np.zeros((n, 6, 6))
    for c in range(120):
        P1[:, I[c]] += Pf[:, c]; P12[:, I[c], J[c]] += Pf[:, c]
    q2 = P12 / np.where(P1[:, :, None] > 0, P1[:, :, None], 1); q3 = Pf / np.where(P12[:, I, J] > 0, P12[:, I, J], 1)
    return P1, q2, q3

def apply(P, X, prm):
    Pb = np.exp(prm["beta"] * np.log(np.clip(P, 1e-12, None))) * (P > 0); Pb /= Pb.sum(1, keepdims=True)
    mult = np.exp(np.einsum("rbk,k->rb", X, prm["wf"])); Pf = Pb * mult[:, I]; Pf /= Pf.sum(1, keepdims=True)
    P1, q2, q3 = decompose(Pf); band = np.digitize(P1[:, 0], [0.4, 0.6]); n = len(P)
    q2a = q2 * np.transpose(prm["c2"][:, :, band], (2, 0, 1))
    for i in range(6):
        q2a[:, i, i] = 0
    q2a /= np.where(q2a.sum(2, keepdims=True) > 0, q2a.sum(2, keepdims=True), 1)
    q3a = q3 * prm["c3"][None, :]; den = np.zeros((n, 36))
    for c in range(120):
        den[:, PAIR[c]] += q3a[:, c]
    q3a /= np.where(den[:, PAIR] > 0, den[:, PAIR], 1)
    PI = P1[:, I] * q2a[:, I, J] * q3a
    return PI / np.where(PI.sum(1, keepdims=True) > 0, PI.sum(1, keepdims=True), 1)

def fit(a):
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= FIT[0]) & (r["date"] <= FIT[1])]
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY)
    O = m[COMBOS].to_numpy(float); P, v = market(O); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, P, y = m[ok].reset_index(drop=True), P[ok], y[ok].astype(int)
    L = np.log(np.clip(P, 1e-12, None)); n = len(m)
    def nllb(b):
        s = b * L; s -= s.max(1, keepdims=True); return -(s[np.arange(n), y] - np.log(np.exp(s).sum(1))).mean()
    beta = minimize_scalar(nllb, bounds=(0.8, 1.3), method="bounded").x
    Pb = np.exp(beta * L); Pb /= Pb.sum(1, keepdims=True)
    X = f_features(a.datadir, m[KEY]); P1, _, _ = decompose(Pb); Lw = np.log(np.clip(P1, 1e-12, None)); yi = I[y]
    def lossf(w):
        s = Lw + X @ w; s -= s.max(1, keepdims=True); q = np.exp(s); q /= q.sum(1, keepdims=True)
        return -np.log(np.clip(q[np.arange(n), yi], 1e-12, None)).mean() + 1e-3 * (w ** 2).sum()
    wf = minimize(lossf, np.zeros(4), method="L-BFGS-B").x
    Pf = Pb * np.exp(np.einsum("rbk,k->rb", X, wf))[:, I]; Pf /= Pf.sum(1, keepdims=True)
    P1, q2, q3 = decompose(Pf); band = np.digitize(P1[:, 0], [0.4, 0.6]); yj = J[y]
    obs2 = np.zeros((6, 6, 3)); exp2 = np.zeros((6, 6, 3)); obs3 = np.zeros(120); exp3 = np.zeros(120)
    for r_ in range(n):
        exp2[yi[r_], :, band[r_]] += q2[r_, yi[r_], :]; obs2[yi[r_], yj[r_], band[r_]] += 1
        sel = PAIR == (yi[r_] * 6 + yj[r_]); exp3[sel] += q3[r_, sel]; obs3[y[r_]] += 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, beta=beta, wf=wf, c2=(obs2 + A) / (exp2 + A), c3=(obs3 + A) / (exp3 + A), fit=np.array(FIT), n=n)
    print(f"saved {a.out}: β={beta:.4f} wf={np.round(wf, 4)} races={n}")

def evaluate(a):
    prm = dict(np.load(a.frozen))
    t1 = _read(f"{a.store}/live_t1/odds3t/*.csv.gz")
    t1 = t1[t1["fetched_at"] < t1["deadline"] + ":00"].drop_duplicates(KEY, keep="last")
    t1 = t1[t1["date"] >= START]
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    cl = norm(pd.read_parquet(f"{a.datadir}/odds.parquet"))
    m = r.merge(t1[KEY], on=KEY).merge(cl[KEY], on=KEY)
    if not len(m):
        msg = f"実行前: {START} 以降で締切直前オッズ・確定オッズ・結果がそろうレースなし"; open(a.out, "w").write(msg + "\n"); print(msg); return
    O1 = m[KEY].merge(t1, on=KEY, how="left")[COMBOS].to_numpy(float); OC = m[KEY].merge(cl, on=KEY, how="left")[COMBOS].to_numpy(float)
    P1m, v1 = market(O1); PCm, vc = market(OC)
    ok = (v1.sum(1) >= 60) & (vc.sum(1) >= 60); m, O1, OC, P1m, PCm, v1, vc = m[ok].reset_index(drop=True), O1[ok], OC[ok], P1m[ok], PCm[ok], v1[ok], vc[ok]
    X = f_features(a.datadir, m[KEY]); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    hit = np.zeros(O1.shape, bool); okY = ~pd.isna(y); hit[np.where(okY)[0], y[okY].astype(int)] = True
    rows = []
    for name, Pm, Od, vd in (("締切直前で判定（本番と同じ）", P1m, O1, v1), ("確定オッズで判定（参考・上限）", PCm, OC, vc)):
        PI = apply(Pm, X, prm); E = PI * np.where(vd, Od, 0)
        buy = vd & vc & (Od <= OMAX) & (E >= THR)
        st = buy.sum(1) * 100.0; rt = np.where(buy & hit, np.where(vc, OC, 0) * 100, 0).sum(1)
        g = pd.DataFrame({"d": m["date"].values, "st": st, "rt": rt}).groupby("d").sum()
        nb = int(buy.sum()); roi = g["rt"].sum() / g["st"].sum() if nb else np.nan
        if nb and (g["st"] > 0).sum() >= 2:
            gg = g[g["st"] > 0]; bi = RNG.integers(0, len(gg), size=(5000, len(gg))); b = gg["rt"].values[bi].sum(1) / gg["st"].values[bi].sum(1)
            lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
        else:
            lo = hi = np.nan
        drift = float(np.nanmedian((OC / Od)[buy])) if nb else np.nan
        state = ("採用候補" if (roi >= 1.1 and lo > 1.0) else "不採用") if nb >= 300 else f"判定前（{nb}/300点）"
        rows.append({"判定": name, "点数": nb, "的中": int((buy & hit).sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi,
                     "確定÷判定時オッズ（中央値）": drift, "状態": state if name.startswith("締切直前") else ""})
    md = [f"# IDEA-040 「市場が一番取りこぼしている組」の前向きの検証（{START}〜{m['date'].max()}、{len(m)}レース）", "",
          f"凍結したパラメータ：β={float(prm['beta']):.3f}、F持ち {np.round(prm['wf'], 3).tolist()}（{prm['fit'][0]}〜{prm['fit'][1]} で推定）", "",
          f"補正後の π × オッズ ≥ {THR}（{OMAX}倍以下）を各100円、払戻は確定オッズ。", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fit", "eval"]); ap.add_argument("datadir"); ap.add_argument("store", nargs="?")
    ap.add_argument("--frozen", default="models/idea040_frozen.npz"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "fit":
        a.out = a.out or a.frozen; fit(a)
    else:
        a.out = a.out or "results/idea040.md"; evaluate(a)

if __name__ == "__main__":
    main()
