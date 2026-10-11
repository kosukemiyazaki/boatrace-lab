"""地方競馬の3連単：着順モデル → 確定オッズの市場との比較 → 締切時オッズで買えたとしたらの上限（IDEA-019）。**ローカル専用**。
usage: python nar/run.py [--data nar_data] [--out nar_out]
期間（テスト期間 2026-09 以降は使わない）
  学習 〜2025-10 / 早期終了 2025-11〜12 / 3連単の温度・割引の推定 2026-01〜02（結果のみ）
  市場との混ぜ合わせの重みの推定 2026-03〜05 / 評価 2026-06〜08（確定オッズは 2026-03 以降のみ公開）
出力（集計値のみ。生データは含まないので共有してよい）: OUT/summary.md, OUT/summary.json
判定（docs/ideas.md の IDEA-019）: 「モデル×市場」の3連単対数損失が市場単独より 0.02 以上小さい → 地方競馬を本格的に検討
"""
import argparse, itertools, json, os, sys
import numpy as np
import pandas as pd

TRAIN_END, ES = "20251031", ("20251101", "20251231")
CAL = ("20260101", "20260228"); FIT = ("20260301", "20260531"); EVAL = ("20260601", "20260831")
TEST_START = "20260901"
GRID = [dict(omax=om, thr=t) for om in (30, 50, 100) for t in (1.1, 1.2, 1.3)]
RNG = np.random.default_rng(0)

def features(H, R):
    key = ["venue", "date", "rno"]
    e = H.merge(R[key + ["post_time", "dist", "surface", "going", "field", "weather"]], on=key, how="left")
    e = e[e["date"] < TEST_START].copy()
    e["absent"] = e["pos"].isna() & e["pos_raw"].str.contains("取|除", na=False)
    e["t"] = e["date"] + e["post_time"].fillna("").str.replace(":", "").str.zfill(4)
    e = e.sort_values(["t", "venue", "rno", "num"]).reset_index(drop=True)
    started = ~e["absent"]
    e["_posn"] = (e["pos"] / e["field"]).where(started)
    e["_win"] = (e["pos"] == 1).astype(float).where(started & e["pos"].notna() | (started & e["pos"].isna()))
    e.loc[started & e["pos"].isna(), "_win"] = 0.0
    e["_top3"] = (e["pos"] <= 3).astype(float).where(e["_win"].notna())
    g = e.groupby(key)
    e["_l3rank"] = g["last3f"].rank(method="min") / e["field"]
    e["_behind"] = e["time"] - g["time"].transform("min")
    e["_d"] = pd.to_datetime(e["date"])
    gh = e.groupby("horse", sort=False)
    for c, w in (("_posn", 5), ("_win", 10), ("_top3", 10), ("_l3rank", 5), ("_behind", 5), ("pop", 5)):
        e[f"h{c}{w}"] = gh[c].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    e["h_last_posn"] = gh["_posn"].shift(1)
    e["h_n"] = gh.cumcount()
    e["h_days"] = (e["_d"] - gh["_d"].shift(1)).dt.days
    e["h_dist_diff"] = e["dist"] - gh["dist"].shift(1)
    e["h_bw_diff_prev"] = e["body_weight"] - gh["body_weight"].shift(1)
    for who, w in (("jockey", 300), ("trainer", 300)):
        gw = e.groupby(who, sort=False)
        e[f"{who[0]}_win{w}"] = gw["_win"].transform(lambda s: s.shift(1).rolling(w, min_periods=20).mean())
        e[f"{who[0]}_top3{w}"] = gw["_top3"].transform(lambda s: s.shift(1).rolling(w, min_periods=20).mean())
    for c in ("venue", "surface", "going", "sex", "weather"):
        e[f"{c}_i"] = e[c].astype("category").cat.codes
    g = e.groupby(key)
    for c in ("h_posn5", "h_win10", "h_l3rank5", "h_behind5", "hpop5", "j_win300", "t_win300", "best_time", "rec_win", "rec_top3", "weight_carried", "age"):
        src = c if c in e.columns else None
        if src is None:
            continue
        e[f"r_{c}"] = e[c] - g[c].transform("mean")
    e = e.rename(columns={"h_posn5": "h_posn5"})
    F = ["waku", "num", "age", "sex_i", "weight_carried", "body_weight", "body_diff", "h_bw_diff_prev", "best_time", "best_time_good",
         "rec_n", "rec_win", "rec_top3", "rec_venue_n", "rec_venue_win", "rec_venue_top3", "rec_dist_n", "rec_dist_win", "rec_dist_top3",
         "jrec_n", "jrec_win", "jrec_top3", "field", "dist", "venue_i", "surface_i", "going_i", "weather_i",
         "h_posn5", "h_win10", "h_top310", "h_l3rank5", "h_behind5", "hpop5", "h_last_posn", "h_n", "h_days", "h_dist_diff",
         "j_win300", "j_top3300", "t_win300", "t_top3300"] + [c for c in e.columns if c.startswith("r_")]
    F = [c for c in F if c in e.columns]
    return e, F

def tri_probs(p, g2, g3):
    """p: 1着確率（n頭）。Harville に割引（2着・3着で強さをべき乗）を入れた3連単確率。返り値: (組番の配列 (m,3), 確率 (m,))"""
    n = len(p)
    perms = np.array(list(itertools.permutations(range(n), 3)))
    i, j, k = perms[:, 0], perms[:, 1], perms[:, 2]
    a = p[i]
    q2 = p ** g2; q3 = p ** g3
    b = q2[j] / np.clip(q2.sum() - q2[i], 1e-12, None)
    c = q3[k] / np.clip(q3.sum() - q3[i] - q3[j], 1e-12, None)
    t = a * b * c
    return perms, t / t.sum()

def race_tables(e, scores):
    """レースごとに（馬番の配列, スコア, 実際の3連単）"""
    e = e.assign(score=scores)
    out = {}
    for k, d in e[~e["absent"]].groupby(["venue", "date", "rno"], sort=False):
        d = d.sort_values("num")
        fin = d[d["pos"].isin([1, 2, 3])].sort_values("pos")
        tri = "-".join(str(int(x)) for x in fin["num"]) if len(fin) == 3 and fin["pos"].tolist() == [1, 2, 3] else None
        out[k] = (d["num"].astype(int).values, d["score"].values, tri)
    return out

def model_ll(tabs, keys, tau, g2, g3):
    ll = []
    for k in keys:
        nums, s, tri = tabs[k]
        if tri is None or len(nums) < 3:
            continue
        p = np.exp((s - s.max()) / tau); p /= p.sum()
        perms, t = tri_probs(p, g2, g3)
        lab = np.array([f"{nums[a]}-{nums[b]}-{nums[c]}" for a, b, c in perms])
        hit = lab == tri
        ll.append(-np.log(max(t[hit][0], 1e-12)) if hit.any() else np.nan)
    return np.nanmean(ll)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="nar_data"); ap.add_argument("--out", default="nar_out")
    a = ap.parse_args()
    import lightgbm as lgb
    from scipy.optimize import minimize
    os.makedirs(a.out, exist_ok=True)
    R = pd.read_parquet(f"{a.data}/races.parquet"); H = pd.read_parquet(f"{a.data}/horses.parquet")
    O = pd.read_parquet(f"{a.data}/tri_odds.parquet"); P = pd.read_parquet(f"{a.data}/tri_pay.parquet")
    e, F = features(H, R)
    print("特徴量", len(F), "頭", len(e), flush=True)
    rel = np.where(e["pos"] == 1, 3, np.where(e["pos"] == 2, 2, np.where(e["pos"] == 3, 1, 0)))
    d = e["date"].values
    tr = (d <= TRAIN_END) & ~e["absent"].values; es = (d >= ES[0]) & (d <= ES[1]) & ~e["absent"].values
    def grp(m):
        return e[m].groupby(["venue", "date", "rno"], sort=False).size().values
    # LambdaRank はグループが連続している必要がある（e はレース順に並んでいる）
    mdl = lgb.LGBMRanker(objective="lambdarank", n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    mdl.fit(e.loc[tr, F], rel[tr], group=grp(tr), eval_set=[(e.loc[es, F], rel[es])], eval_group=[grp(es)], eval_at=[3],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    print("学習完了 best_iter", mdl.best_iteration_, flush=True)
    sel = d >= CAL[0]
    tabs = race_tables(e[sel], mdl.predict(e.loc[sel, F], num_iteration=mdl.best_iteration_))
    keys_cal = [k for k in tabs if CAL[0] <= k[1] <= CAL[1]]
    res = minimize(lambda x: model_ll(tabs, keys_cal, *np.exp(x)), np.log([1.0, 0.8, 0.7]), method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
    tau, g2, g3 = np.exp(res.x)
    print(f"温度 {tau:.3f}, 割引 {g2:.3f} {g3:.3f}, 校正期間の3連単対数損失 {res.fun:.4f}", flush=True)
    # 市場（確定オッズ）
    Ok = {k: dict(zip(g["tri"], g["odds"])) for k, g in O.groupby(["venue", "date", "rno"])}
    Pk = {k: (r.tri, r.tri_pay) for k, r in P.set_index(["venue", "date", "rno"]).iterrows()}
    rows = []
    for k in tabs:
        if not (FIT[0] <= k[1] <= EVAL[1]) or k not in Ok:
            continue
        nums, s, tri = tabs[k]
        if tri is None or len(nums) < 3:
            continue
        p = np.exp((s - s.max()) / tau); p /= p.sum()
        perms, t = tri_probs(p, g2, g3)
        lab = [f"{nums[x]}-{nums[y]}-{nums[z]}" for x, y, z in perms]
        od = np.array([Ok[k].get(c, np.nan) for c in lab], float)
        if np.isfinite(od).sum() < len(lab) * 0.8 or tri not in lab:
            continue
        inv = np.where(np.isfinite(od) & (od > 0), 1 / od, 0); q = inv / inv.sum()
        rows.append(dict(key=k, t=t, q=q, od=od, y=lab.index(tri), pay=Pk.get(k, (None, np.nan))[1]))
    print("市場と比べるレース", len(rows), flush=True)
    fit = [r for r in rows if r["key"][1] <= FIT[1]]; ev = [r for r in rows if r["key"][1] >= EVAL[0]]
    def blend(r, wa, wb):
        lg = wa * np.log(np.clip(r["t"], 1e-12, None)) + wb * np.log(np.clip(r["q"], 1e-12, None))
        x = np.exp(lg - lg.max()); return x / x.sum()
    nll_b = lambda w, rs: np.mean([-np.log(max(blend(r, *w)[r["y"]], 1e-12)) for r in rs])
    wres = minimize(lambda w: nll_b(w, fit), [0.5, 0.5], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6})
    wa, wb = wres.x
    ll_m = np.array([-np.log(max(r["t"][r["y"]], 1e-12)) for r in ev])
    ll_q = np.array([-np.log(max(r["q"][r["y"]], 1e-12)) for r in ev])
    ll_b = np.array([-np.log(max(blend(r, wa, wb)[r["y"]], 1e-12)) for r in ev])
    days = np.array([r["key"][1] for r in ev])
    diff = ll_b - ll_q
    dd = pd.Series(diff).groupby(days).agg(["sum", "count"]); bi = RNG.integers(0, len(dd), size=(5000, len(dd)))
    boot = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1)
    ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    adopt = diff.mean() <= -0.02 and ci[1] < 0
    # 全組番の回収率（オッズ帯別）と、締切時オッズで買えたとしたらの上限
    bands = [(0, 10), (10, 30), (30, 100), (100, 300), (300, 1e9)]
    st = {b: 0.0 for b in bands}; rt = {b: 0.0 for b in bands}
    for r in rows:
        for b in bands:
            m = np.isfinite(r["od"]) & (r["od"] >= b[0]) & (r["od"] < b[1])
            st[b] += m.sum() * 100
            if m[r["y"]] and np.isfinite(r["pay"]):
                rt[b] += r["pay"]
    band_roi = {f"{lo}〜{hi}倍" if hi < 1e9 else f"{lo}倍〜": (rt[(lo, hi)] / st[(lo, hi)] if st[(lo, hi)] else None) for lo, hi in bands}
    bets = []
    for g in GRID:
        stake = ret = n = hits = 0
        for r in ev:
            pb = blend(r, wa, wb); evv = pb * r["od"]
            ok = np.isfinite(evv) & (r["od"] <= g["omax"]) & (evv > g["thr"])
            idx = np.where(ok)[0][np.argsort(-evv[ok])][:10]
            n += len(idx); stake += 100 * len(idx)
            if r["y"] in idx and np.isfinite(r["pay"]):
                ret += r["pay"]; hits += 1
        bets.append({**g, "点数": n, "的中": hits, "回収率": (ret / stake if stake else None)})
    out = dict(races_compared=len(rows), eval_races=len(ev), tau=float(tau), g2=float(g2), g3=float(g3), blend=[float(wa), float(wb)],
               ll_model=float(ll_m.mean()), ll_market=float(ll_q.mean()), ll_blend=float(ll_b.mean()),
               diff_blend_minus_market=float(diff.mean()), ci95=ci, decision="地方競馬を本格的に検討" if adopt else "市場は効率的（ボートと同様）",
               band_roi_all_combos=band_roi, bets_upper_bound=bets)
    json.dump(out, open(f"{a.out}/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = [f"# IDEA-019 地方競馬の3連単（評価 {EVAL[0]}〜{EVAL[1]}、{len(ev)}レース）", "",
          f"3連単対数損失：モデル {out['ll_model']:.4f} / 市場（確定オッズ） {out['ll_market']:.4f} / モデル×市場 {out['ll_blend']:.4f}", "",
          f"モデル×市場 − 市場 = {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}） → **{out['decision']}**（基準 −0.02）", "",
          "全組番の回収率（オッズ帯別、2026-03〜08）", "", pd.Series(band_roi).to_frame("回収率").to_markdown(floatfmt=".3f"), "",
          "締切時オッズで買えたとしたらの上限（9通り全件、各100円、EV上位最大10点）", "", pd.DataFrame(bets).to_markdown(index=False, floatfmt=".3f")]
    open(f"{a.out}/summary.md", "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
