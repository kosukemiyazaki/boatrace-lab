"""IDEA-041 市場の形から「外れやすい艇」を機械に探させる（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea041.py DATADIR STOREDIR [--out results/idea041.md]
- 艇ごとの1着。出発点 = 3連単の確定オッズから出した1着確率 p（init_score = log p）
- 特徴量：各券種（単勝・複勝下限・2連単・2連複・3連複）から出したその艇の確率と p の比、混戦度、オーバーラウンド、艇番、展示の進入、F持ち
- 月ごとの時系列の交差検証：2026-02〜08 の各月を「その月より前の全部（2025-10〜）」で学習して評価
- 判定：評価した月の1着対数損失が市場より 0.005 以上小さく、日単位95%CI上限 < 0 → 段階2（単勝の回収率）
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, COMBOS_2F, COMBOS_2T, COMBOS_3F
from exp_idea036 import norm
from idea040 import f_features

KEY = ["date", "jcd", "rno"]; START, END = "20251001", "20260831"
EVAL_MONTHS = ["202602", "202603", "202604", "202605", "202606", "202607", "202608"]
RNG = np.random.default_rng(0)

def inv_norm(O):
    v = np.isfinite(O) & (O > 0); P = np.where(v, 1 / np.where(v, O, 1), 0)
    s = P.sum(1, keepdims=True); return P / np.where(s > 0, s, 1), s[:, 0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea041.md")
    a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= START) & (r["date"] <= END)]
    o3 = norm(pd.read_parquet(f"{a.datadir}/odds.parquet")); o3f = norm(pd.read_parquet(f"{a.datadir}/odds3f.parquet"))
    o2 = norm(pd.read_parquet(f"{a.datadir}/odds2tf.parquet"))
    tf = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in glob.glob(f"{a.store}/oddstf/*.csv.gz")])
    tf = norm(tf).drop_duplicates(KEY, keep="last")
    m = r.merge(o3, on=KEY).merge(o3f, on=KEY).merge(o2, on=KEY).merge(tf[KEY + [f"tan{i}" for i in range(1, 7)] + [f"fuku_lo{i}" for i in range(1, 7)]], on=KEY)
    O = m[COMBOS].to_numpy(float); ok = (np.isfinite(O) & (O > 0)).sum(1) >= 60
    y = m["tri_combo"].astype(str).str[0]; ok &= y.str.isdigit().values
    m = m[ok].reset_index(drop=True); n = len(m); y = m["tri_combo"].astype(str).str[0]
    P3, ov3 = inv_norm(m[COMBOS].to_numpy(float))
    f1 = np.array([int(c[0]) for c in COMBOS]); f2 = np.array([int(c[2]) for c in COMBOS]); f3 = np.array([int(c[4]) for c in COMBOS])
    p_win = np.stack([P3[:, f1 == b].sum(1) for b in range(1, 7)], 1)
    p_top2 = np.stack([P3[:, (f1 == b) | (f2 == b)].sum(1) for b in range(1, 7)], 1)
    p_top3 = np.stack([P3[:, (f1 == b) | (f2 == b) | (f3 == b)].sum(1) for b in range(1, 7)], 1)
    Pt, ovt = inv_norm(m[[f"tan{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    Pf, ovf = inv_norm(m[[f"fuku_lo{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    P2t, ov2t = inv_norm(m[COMBOS_2T].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    P2f, ov2f = inv_norm(m[COMBOS_2F].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    P3f, ov3f = inv_norm(m[COMBOS_3F].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    t1 = np.array([int(c[0]) for c in COMBOS_2T]); t2 = np.array([int(c[2]) for c in COMBOS_2T])
    q1 = np.array([int(c[0]) for c in COMBOS_2F]); q2 = np.array([int(c[2]) for c in COMBOS_2F])
    r1 = [set(map(int, c.split("="))) for c in COMBOS_3F]
    p2t_win = np.stack([P2t[:, t1 == b].sum(1) for b in range(1, 7)], 1)
    p2f_top2 = np.stack([P2f[:, (q1 == b) | (q2 == b)].sum(1) for b in range(1, 7)], 1)
    p3f_top3 = np.stack([P3f[:, [k for k, s in enumerate(r1) if b in s]].sum(1) for b in range(1, 7)], 1)
    ent = -(np.where(P3 > 0, P3 * np.log(np.where(P3 > 0, P3, 1)), 0)).sum(1); fav = P3.max(1)
    F = f_features(a.datadir, m[KEY])
    eps = 1e-6
    feats = []
    for b in range(6):
        feats.append(np.column_stack([
            np.log(p_win[:, b] + eps), np.log(p_top2[:, b] + eps), np.log(p_top3[:, b] + eps),
            np.log((Pt[:, b] + eps) / (p_win[:, b] + eps)), np.log((p2t_win[:, b] + eps) / (p_win[:, b] + eps)),
            np.log((Pf[:, b] + eps) / (p_top2[:, b] + eps)), np.log((p2f_top2[:, b] + eps) / (p_top2[:, b] + eps)),
            np.log((p3f_top3[:, b] + eps) / (p_top3[:, b] + eps)),
            ent, fav, ov3, ovt, ovf, ov2t, ov2f, ov3f, np.full(n, b + 1), F[:, b, :].sum(1), F[:, b, 3]]))
    names = ["log_p_win", "log_p_top2", "log_p_top3", "r_tan", "r_2t", "r_fuku", "r_2f", "r_3f", "entropy", "fav", "ov3t", "ovtan", "ovfuku", "ov2t", "ov2f", "ov3f", "boat", "f_self", "f_inner"]
    Xb = np.stack(feats, 1)                                     # (n, 6, k)
    yb = np.zeros((n, 6)); yb[np.arange(n), y.astype(int).values - 1] = 1
    base = np.log(np.clip(p_win, 1e-9, None))                   # init_score（log p。レース内の正規化は評価のときに行う）
    month = m["date"].str[:6].values; days = m["date"].values
    ll_m, ll_a, dd, tanroi = [], [], [], []
    for em in EVAL_MONTHS:
        trn = month < em; tst = month == em
        Xtr = Xb[trn].reshape(-1, Xb.shape[2]); ytr = yb[trn].ravel(); btr = base[trn].ravel()
        ds = lgb.Dataset(Xtr, ytr, init_score=btr - np.log1p(-np.clip(np.exp(btr), 0, 0.999)), feature_name=names, free_raw_data=False)
        mdl = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0), ds, num_boost_round=300)
        bt = base[tst]; adj = mdl.predict(Xb[tst].reshape(-1, Xb.shape[2]), raw_score=True).reshape(-1, 6)
        s = bt + adj; s = s - s.max(1, keepdims=True); q = np.exp(s); q /= q.sum(1, keepdims=True)  # predict(raw_score) は init_score を含まないので log p を足す
        yi = y[tst].astype(int).values - 1
        la = -np.log(np.clip(q[np.arange(len(yi)), yi], 1e-12, None)); lm = -np.log(np.clip(p_win[tst][np.arange(len(yi)), yi], 1e-12, None))
        ll_a.append(la); ll_m.append(lm); dd.append(days[tst])
        to = m.loc[tst, [f"tan{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        buy = np.isfinite(to) & (to > 0) & (q * np.nan_to_num(to) > 1.1)
        ret = np.where(buy & (yb[tst] == 1), to * 100, 0)
        tanroi.append(pd.DataFrame({"d": days[tst], "st": buy.sum(1) * 100.0, "rt": ret.sum(1)}))
        print(em, f"市場 {lm.mean():.4f} 補正 {la.mean():.4f}", flush=True)
    la, lm, d_ = np.concatenate(ll_a), np.concatenate(ll_m), np.concatenate(dd)
    diff = la - lm; g = pd.Series(diff).groupby(d_).agg(["sum", "count"])
    bi = RNG.integers(0, len(g), size=(5000, len(g))); bt = g["sum"].values[bi].sum(1) / g["count"].values[bi].sum(1)
    ci = (float(np.quantile(bt, .025)), float(np.quantile(bt, .975)))
    go = diff.mean() <= -0.005 and ci[1] < 0
    T = pd.concat(tanroi).groupby("d").sum(); T = T[T["st"] > 0]
    nt = int(T["st"].sum() / 100); troi = T["rt"].sum() / T["st"].sum() if nt else np.nan
    imp = pd.Series(mdl.feature_importance("gain"), index=names).sort_values(ascending=False)
    md = [f"# IDEA-041 市場の形から外れやすい艇を機械に探させる（LightGBM、{START}〜{END}、評価 {EVAL_MONTHS[0]}〜{EVAL_MONTHS[-1]}）", "",
          f"1着の対数損失（評価した月）：市場 {lm.mean():.4f} → 補正 {la.mean():.4f}、差 {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}） → **{'段階2へ' if go else '打ち切り'}**", "",
          f"参考：補正後の確率 × 単勝確定オッズ > 1.1 の艇を各100円：{nt}点、回収率 {troi:.3f}", "",
          "特徴量の重要度（最後の月のモデル、gain 上位10）", "", imp.head(10).to_frame("gain").to_markdown(floatfmt=".0f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
