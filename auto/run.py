"""オートレースの3連単：着順モデル → 最終オッズの市場との比較 → 最終オッズで買えたとしたらの上限（IDEA-029）。**ローカル専用**。
usage: python auto/run.py [--data auto_data] [--out results/auto_summary.md]
期間（テスト期間 2026-09 以降は使わない）
  学習 〜2025-12 / 早期終了 2026-01〜02 / 温度・割引 2026-03 / 市場との混ぜ合わせ 2026-04〜05 / 評価 2026-06〜08
出力は集計値のみ（results/ に置いて共有してよい）
判定（docs/ideas.md の IDEA-029）: 「モデル×市場」の3連単対数損失が市場単独より 0.02 以上小さい → オートレースを本格的に検討
参考: 3連単の市場から見た単勝・複勝のずれ（IDEA-022/023 と同じ、最終オッズ＝上限）、全組番のオッズ帯別回収率
"""
import argparse, itertools, json, os, sys
import numpy as np
import pandas as pd

TRAIN_END, ES = "20251231", ("20260101", "20260228")
CAL = ("20260301", "20260331"); FIT = ("20260401", "20260531"); EVAL = ("20260601", "20260831")
TEST_START = "20260901"
GRID = [dict(omax=om, thr=t) for om in (30, 50, 100) for t in (1.1, 1.2, 1.3)]
RNG = np.random.default_rng(0)
KEY = ["date", "pc", "rno"]

def tri_probs(p, g2, g3):
    n = len(p)
    perms = np.array(list(itertools.permutations(range(n), 3)))
    i, j, k = perms[:, 0], perms[:, 1], perms[:, 2]
    q2 = p ** g2; q3 = p ** g3
    t = p[i] * q2[j] / np.clip(q2.sum() - q2[i], 1e-12, None) * q3[k] / np.clip(q3.sum() - q3[i] - q3[j], 1e-12, None)
    return perms, t / t.sum()

def features(D, R):
    e = D.merge(R[KEY + ["p1", "p2", "p3"]], on=KEY, how="inner")
    e = e[(e["date"] < TEST_START) & ~e["absent"]].copy()
    e["rel"] = np.where(e["car"] == e["p1"], 3, np.where(e["car"] == e["p2"], 2, np.where(e["car"] == e["p3"], 1, 0)))
    e["ai_i"] = pd.to_numeric(e["ai"], errors="coerce")
    e["grad_i"] = pd.to_numeric(e["graduation"], errors="coerce")
    e["retry"] = e["trial_retry"].notna().astype(int)
    e["n"] = e.groupby(KEY)["car"].transform("size")
    e["trial_form"] = e["trial"] - e["gt_trial_ave"]
    g = e.groupby(KEY)
    for c in ("trial", "handicap", "race_dev", "st_ave", "gt_race_ave", "gt_race_best", "trial_form"):
        e[f"r_{c}"] = e[c] - g[c].transform("mean")
        e[f"k_{c}"] = g[c].rank(method="min")
    # ハンデ（後ろからのスタート、m）を試走の速さで割り引いた見込み：1周あたりの時間差 × 周回数 と ハンデ の比較の近似
    e["trial_x_hc"] = e["r_trial"] * 6 * 100 + e["r_handicap"]
    F = ["car", "n", "pc", "handicap", "trial", "retry", "race_dev", "st_ave", "gt_trial_ave", "gt_race_ave", "gt_race_best", "ai_i", "grad_i",
         "trial_form", "trial_x_hc"] + [c for c in e.columns if c.startswith(("r_", "k_"))]
    return e.sort_values(KEY + ["car"]).reset_index(drop=True), F

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="auto_data"); ap.add_argument("--out", default="results/auto_summary.md")
    a = ap.parse_args()
    import lightgbm as lgb
    from scipy.optimize import minimize
    R = pd.read_parquet(f"{a.data}/races.parquet"); D = pd.read_parquet(f"{a.data}/riders.parquet"); O = pd.read_parquet(f"{a.data}/odds.parquet")
    e, F = features(D, R)
    print("特徴量", len(F), "台", len(e), flush=True)
    d = e["date"].values
    tr = d <= TRAIN_END; es = (d >= ES[0]) & (d <= ES[1])
    grp = lambda m: e[m].groupby(KEY, sort=False).size().values
    mdl = lgb.LGBMRanker(objective="lambdarank", n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    mdl.fit(e.loc[tr, F], e.loc[tr, "rel"], group=grp(tr), eval_set=[(e.loc[es, F], e.loc[es, "rel"])], eval_group=[grp(es)], eval_at=[3],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    e["score"] = mdl.predict(e[F], num_iteration=mdl.best_iteration_)
    tabs = {}
    for k, g in e[d >= CAL[0]].groupby(KEY, sort=False):
        tri = f"{int(g['p1'].iloc[0])}-{int(g['p2'].iloc[0])}-{int(g['p3'].iloc[0])}" if g["p1"].notna().iloc[0] else None
        tabs[k] = (g["car"].astype(int).values, g["score"].values, tri)
    def model_ll(keys, tau, g2, g3):
        out = []
        for k in keys:
            cars, s, tri = tabs[k]
            if tri is None or len(cars) < 3:
                continue
            p = np.exp((s - s.max()) / tau); p /= p.sum()
            perms, t = tri_probs(p, g2, g3)
            lab = [f"{cars[x]}-{cars[y]}-{cars[z]}" for x, y, z in perms]
            out.append(-np.log(max(t[lab.index(tri)], 1e-12)) if tri in lab else np.nan)
        return np.nanmean(out)
    kc = [k for k in tabs if CAL[0] <= k[0] <= CAL[1]]
    res = minimize(lambda x: model_ll(kc, *np.exp(x)), np.log([1.0, 0.8, 0.7]), method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
    tau, g2, g3 = np.exp(res.x)
    T3 = O[O["kind"] == "3連単"]; Ok = {k: dict(zip(g["combo"], g["odds"])) for k, g in T3.groupby(KEY)}
    rows = []
    for k in tabs:
        if not (FIT[0] <= k[0] <= EVAL[1]) or k not in Ok:
            continue
        cars, s, tri = tabs[k]
        if tri is None or len(cars) < 3:
            continue
        p = np.exp((s - s.max()) / tau); p /= p.sum()
        perms, t = tri_probs(p, g2, g3)
        lab = [f"{cars[x]}-{cars[y]}-{cars[z]}" for x, y, z in perms]
        od = np.array([Ok[k].get(c, np.nan) for c in lab], float)
        if np.isfinite(od).sum() < len(lab) * 0.8 or tri not in lab:
            continue
        inv = np.where(np.isfinite(od) & (od > 0), 1 / np.where(od > 0, od, 1), 0); q = inv / inv.sum()
        rows.append(dict(key=k, cars=cars, t=t, q=q, od=od, lab=lab, y=lab.index(tri)))
    print("市場と比べるレース", len(rows), flush=True)
    fit = [r for r in rows if r["key"][0] <= FIT[1]]; ev = [r for r in rows if r["key"][0] >= EVAL[0]]
    def blend(r, wa, wb):
        lg = wa * np.log(np.clip(r["t"], 1e-12, None)) + wb * np.log(np.clip(r["q"], 1e-12, None)); x = np.exp(lg - lg.max()); return x / x.sum()
    wa, wb = minimize(lambda w: np.mean([-np.log(max(blend(r, *w)[r["y"]], 1e-12)) for r in fit]), [0.5, 0.5], method="Nelder-Mead",
                      options={"xatol": 1e-3, "fatol": 1e-6}).x
    ll_m = np.array([-np.log(max(r["t"][r["y"]], 1e-12)) for r in ev]); ll_q = np.array([-np.log(max(r["q"][r["y"]], 1e-12)) for r in ev])
    ll_b = np.array([-np.log(max(blend(r, wa, wb)[r["y"]], 1e-12)) for r in ev])
    days = np.array([r["key"][0] for r in ev]); diff = ll_b - ll_q
    dd = pd.Series(diff).groupby(days).agg(["sum", "count"]); bi = RNG.integers(0, len(dd), size=(5000, len(dd)))
    boot = dd["sum"].values[bi].sum(1) / dd["count"].values[bi].sum(1); ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    adopt = diff.mean() <= -0.02 and ci[1] < 0
    # 全組番のオッズ帯別回収率（払戻はオッズ×100、2026-04〜08）
    bands = [(0, 10), (10, 30), (30, 100), (100, 300), (300, 1e9)]; st = {b: 0.0 for b in bands}; rt = {b: 0.0 for b in bands}
    for r in rows:
        for b in bands:
            m = np.isfinite(r["od"]) & (r["od"] > 0) & (r["od"] >= b[0]) & (r["od"] < b[1])
            st[b] += m.sum() * 100; rt[b] += r["od"][r["y"]] * 100 if m[r["y"]] else 0
    band_roi = {(f"{lo}〜{hi}倍" if hi < 1e9 else f"{lo}倍〜"): (rt[(lo, hi)] / st[(lo, hi)] if st[(lo, hi)] else None) for lo, hi in bands}
    bets = []
    for g in GRID:
        stake = ret = n = hits = 0
        for r in ev:
            evv = blend(r, wa, wb) * r["od"]; ok = np.isfinite(evv) & (r["od"] <= g["omax"]) & (evv > g["thr"])
            idx = np.where(ok)[0][np.argsort(-evv[ok])][:10]; n += len(idx); stake += 100 * len(idx)
            if r["y"] in idx:
                ret += r["od"][r["y"]] * 100; hits += 1
        bets.append({**g, "点数": n, "的中": hits, "回収率": (ret / stake if stake else None)})
    # 3連単の市場から見た単勝・複勝のずれ（最終オッズで判定・購入できたとしたらの上限、2026-04〜08）
    W = O[O["kind"] == "単勝"].pivot_table(index=KEY, columns="combo", values="odds", aggfunc="first")
    P = O[O["kind"] == "複勝"].pivot_table(index=KEY, columns="combo", values="odds", aggfunc="first")
    Rk = R.set_index(KEY)
    cp = []
    for r in rows:
        k = r["key"]; first = np.array([int(x.split("-")[0]) for x in r["lab"]]); sec = np.array([int(x.split("-")[1]) for x in r["lab"]])
        thd = np.array([int(x.split("-")[2]) for x in r["lab"]])
        fpaid = json.loads(Rk.loc[k, "fns_paid"]) if k in Rk.index else {}
        kplace = max(len(fpaid), 2)
        for c in r["cars"]:
            qw = r["q"][first == c].sum(); qp = r["q"][(first == c) | (sec == c) | ((thd == c) if kplace >= 3 else False)].sum()
            wo = W.loc[k].get(str(c), np.nan) if k in W.index else np.nan; po = P.loc[k].get(str(c), np.nan) if k in P.index else np.nan
            win = r["lab"][r["y"]].split("-")[0] == str(c)
            cp.append(dict(date=k[0], kind="単勝", ev=qw * wo, odds=wo, pay=(wo * 100 if win else 0.0)))
            cp.append(dict(date=k[0], kind="複勝", ev=qp * po, odds=po, pay=(fpaid.get(str(c), 0.0) or 0.0)))
    cp = pd.DataFrame(cp); cross = []
    for kind in ("単勝", "複勝"):
        for thr in (1.1, 1.2, 1.5):
            s = cp[(cp["kind"] == kind) & np.isfinite(cp["ev"]) & (cp["ev"] > thr)]
            if len(s):
                g = s.groupby("date").agg(st=("pay", "size"), rt=("pay", "sum")); g["st"] *= 100.0
                b = RNG.integers(0, len(g), size=(3000, len(g))); bb = g["rt"].values[b].sum(1) / g["st"].values[b].sum(1)
                cross.append({"券種": kind, "閾値": thr, "点数": len(s), "回収率": g["rt"].sum() / g["st"].sum(), "95%CI下": float(np.quantile(bb, .025)), "95%CI上": float(np.quantile(bb, .975))})
    out = dict(races_compared=len(rows), eval_races=len(ev), tau=float(tau), g2=float(g2), g3=float(g3), blend=[float(wa), float(wb)],
               ll_model=float(ll_m.mean()), ll_market=float(ll_q.mean()), ll_blend=float(ll_b.mean()), diff=float(diff.mean()), ci95=ci,
               decision="オートレースを本格的に検討" if adopt else "市場は効率的（ボート・地方競馬と同様）", band_roi=band_roi, bets_upper_bound=bets, cross_pool=cross)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out.replace(".md", ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = [f"# IDEA-029 オートレースの3連単（評価 {EVAL[0]}〜{EVAL[1]}、{len(ev)}レース）", "",
          f"3連単対数損失：モデル {out['ll_model']:.4f} / 市場（最終オッズ） {out['ll_market']:.4f} / モデル×市場 {out['ll_blend']:.4f}", "",
          f"モデル×市場 − 市場 = {diff.mean():+.4f}（95%CI {ci[0]:+.4f}〜{ci[1]:+.4f}） → **{out['decision']}**（基準 −0.02）", "",
          "全組番の回収率（オッズ帯別、2026-04〜08、払戻＝最終オッズ×100）", "", pd.Series(band_roi).to_frame("回収率").to_markdown(floatfmt=".3f"), "",
          "最終オッズで買えたとしたらの上限（モデル×市場、各100円、EV上位最大10点）", "", pd.DataFrame(bets).to_markdown(index=False, floatfmt=".3f"), "",
          "3連単の市場から見た単勝・複勝のずれ（最終オッズで判定・購入できたとしたらの上限、2026-04〜08）", "",
          pd.DataFrame(cross).to_markdown(index=False, floatfmt=".3f") if cross else "（データなし）"]
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main()
