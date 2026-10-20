"""IDEA-060 選手ごとのクセ（この選手が1号艇だと逃げない／この場で崩れる／スタートが遅いと外が来る。docs/ideas.md の登録どおり）
usage: python scripts/exp_idea060.py DATADIR HISTDS --bk BKDIR [--out results/idea060.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ。スタートのクセと12年の参考に使う）
- BKDIR：K ファイル（bk/YYYY/kYYMMDD.txt.gz）。評価の1号艇単勝の払戻に使う
- 単位：1号艇の選手。(a) 1着 (b) 3着以内 (c) 選手×場の1着 (d) 4〜6号艇の1着、それぞれ 実際÷市場の見込み（確定3連単オッズから）
- 縮め方：経験ベイズ (実際 + k) ÷ (見込み + k)、k は探索期間のばらつきから（モーメント法）
- 探索 2023-11〜2025-12 → 評価 2026-01〜08。買い方 (1) 比 ≥ 1.15 の選手が1号艇なら単勝1、(2) 比 ≤ 0.85 なら補正した確率で π×オッズ > 1.0・300倍以下の3連単
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL, ratio_ci, boot_roi
from exp_idea055 import k_tanfuku

HI, LO = 1.15, 0.85

def est_k(o, e, v):
    """O_i ~ 見込み E_i（分散 V_i）に選手ごとの倍率 θ_i（平均1・分散 τ²）が掛かるとして τ² をモーメント法で推定。k = 1/τ²"""
    tau2 = (((o - e) ** 2 - v).sum()) / (e ** 2).sum()
    return (1.0 / tau2 if tau2 > 0 else np.inf), tau2

def shrink(o, e, k):
    return (o + k) / (e + k) if np.isfinite(k) else np.ones_like(o, dtype=float)

def wcorr(x, y, w):
    m = np.isfinite(x) & np.isfinite(y) & (w > 0); x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))

def per_racer(df, key, ocol, ecol, vcol):
    g = df.groupby(key).agg(o=(ocol, "sum"), e=(ecol, "sum"), v=(vcol, "sum"), n=(ocol, "size")).reset_index()
    return g

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("--bk", nargs="+", required=True)
    ap.add_argument("--out", default="results/idea060.md"); a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= EXP[0]) & (r["date"] <= EVL[1])]
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY)
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban", "pos_raw"])).drop_duplicates(KEY + ["boat"], keep="last")
    # 欠場のあるレースは除く（6艇そろい、着順が数字か F/L 以外の失格等も含めて残す）
    absent = e[e["pos_raw"].astype(str).isin(["K0", "K1", "欠"])][KEY].drop_duplicates()
    cnt = e.groupby(KEY).size().rename("nb").reset_index()
    t1 = e[e["boat"] == 1][KEY + ["toban"]].rename(columns={"toban": "t1"})
    m = m.merge(cnt, on=KEY).merge(t1, on=KEY)
    m = m[m["nb"] == 6].merge(absent.assign(ab=1), on=KEY, how="left"); m = m[m["ab"].isna()].drop(columns=["ab", "nb"])
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    f1 = np.array([int(c[0]) for c in COMBOS]); f2 = np.array([int(c[2]) for c in COMBOS]); f3 = np.array([int(c[4]) for c in COMBOS])
    in3 = (f1 == 1) | (f2 == 1) | (f3 == 1); out_first = f1 >= 4
    m["p_win"] = P[:, f1 == 1].sum(1); m["p_top3"] = P[:, in3].sum(1); m["p_out"] = P[:, out_first].sum(1)
    m["w_win"] = (f1[y] == 1).astype(float); m["w_top3"] = in3[y].astype(float); m["w_out"] = out_first[y].astype(float)
    for c in ("win", "top3", "out"):
        m[f"v_{c}"] = m[f"p_{c}"] * (1 - m[f"p_{c}"])
    days = m["date"].values; ex = (days >= EXP[0]) & (days <= EXP[1]); ev = (days >= EVL[0]) & (days <= EVL[1])
    mx, mv = m[ex], m[ev]
    out = [f"# IDEA-060 選手ごとのクセ（1号艇の選手。探索 {EXP[0]}〜{EXP[1]} {int(ex.sum())}レース／評価 {EVL[0]}〜{EVL[1]} {int(ev.sum())}レース）", ""]
    ratios = {}; summary = []
    rows_k = []
    for c, lab in (("win", "(a) 1号艇の1着"), ("top3", "(b) 1号艇の3着以内"), ("out", "(d) 1号艇の選手のとき4〜6号艇が1着")):
        gx = per_racer(mx, "t1", f"w_{c}", f"p_{c}", f"v_{c}"); gv = per_racer(mv, "t1", f"w_{c}", f"p_{c}", f"v_{c}")
        k, tau2 = est_k(gx["o"].values, gx["e"].values, gx["v"].values)
        gx["s"] = shrink(gx["o"].values, gx["e"].values, k)
        j = gx.merge(gv, on="t1", suffixes=("_x", "_v"))
        j["r_v"] = j["o_v"] / j["e_v"]
        cor = wcorr(j["s"].values, j["r_v"].values, j["e_v"].values)
        # 探索の縮めた比で上位・下位10%（探索の見込み5回以上の選手）
        jj = j[j["e_x"] >= 5]; q10, q90 = np.quantile(jj["s"], [.1, .9])
        top = set(jj[jj["s"] >= q90]["t1"]); bot = set(jj[jj["s"] <= q10]["t1"])
        def evr(sel):
            s = mv["t1"].isin(sel).values
            return ratio_ci(mv["date"].values[s], mv[f"w_{c}"].values[s], mv[f"p_{c}"].values[s]) + (int(s.sum()),)
        rt, rb = evr(top), evr(bot)
        # 縮めた比の分布
        dist = np.quantile(gx.loc[gx["e"] >= 5, "s"], [.01, .1, .5, .9, .99]) if np.isfinite(k) else [1] * 5
        rows_k.append({"指標": lab, "探索の選手数": len(gx), "k": round(k, 1) if np.isfinite(k) else "∞（ばらつきなし）", "τ²": tau2,
                       "縮めた比 1%": dist[0], "10%": dist[1], "50%": dist[2], "90%": dist[3], "99%": dist[4],
                       "探索→評価の相関（重み付き）": cor, "上位10%の評価での比": f"{rt[0]:.3f}（{rt[1]:.3f}〜{rt[2]:.3f}、{rt[3]}レース）",
                       "下位10%の評価での比": f"{rb[0]:.3f}（{rb[1]:.3f}〜{rb[2]:.3f}、{rb[3]}レース）"})
        ratios[c] = gx.set_index("t1")["s"]
        if c == "win":
            gx_win, k_win = gx, k
    # (c) 選手×場
    gx = per_racer(mx, ["t1", "jcd"], "w_win", "p_win", "v_win"); gv = per_racer(mv, ["t1", "jcd"], "w_win", "p_win", "v_win")
    k_c, tau2_c = est_k(gx["o"].values, gx["e"].values, gx["v"].values)
    gx["s"] = shrink(gx["o"].values, gx["e"].values, k_c); j = gx.merge(gv, on=["t1", "jcd"], suffixes=("_x", "_v")); j["r_v"] = j["o_v"] / j["e_v"]
    cor_c = wcorr(j["s"].values, j["r_v"].values, j["e_v"].values)
    # 選手の比を除いた残り（選手×場 ÷ 選手）の相関
    j["s_rel"] = j["s"] / j["t1"].map(ratios["win"]).fillna(1.0)
    cor_c2 = wcorr(j["s_rel"].values, j["r_v"].values / j["t1"].map(ratios["win"]).fillna(1.0).values, j["e_v"].values)
    rows_k.append({"指標": "(c) 選手×場の1号艇の1着", "探索の選手数": len(gx), "k": round(k_c, 1) if np.isfinite(k_c) else "∞（ばらつきなし）", "τ²": tau2_c,
                   "縮めた比 1%": np.nan, "10%": np.nan, "50%": np.nan, "90%": np.nan, "99%": np.nan,
                   "探索→評価の相関（重み付き）": cor_c, "上位10%の評価での比": f"選手の比を除いた残りの相関 {cor_c2:.3f}", "下位10%の評価での比": ""})
    out += ["## 選手ごとのクセの大きさと持続", "", pd.DataFrame(rows_k).to_markdown(index=False, floatfmt=".3f"), "",
            "- k：経験ベイズの縮め方の強さ（見込み k 回分の「平均どおり」を足す）。τ² は選手ごとの倍率の分散の推定。", ""]
    # (d) スタートのクセ（2014〜2023-09 の平均 ST・ばらつき、探索・評価より前のデータのみ）
    h = norm(pd.read_parquet(f"{a.histds}/entries.parquet", columns=KEY + ["toban", "st", "st_raw"]))
    h = h[h["st_raw"].astype(str).str.match(r"^[0.]") & h["st"].between(0, 0.5)]
    stt = h.groupby("toban")["st"].agg(["mean", "std", "size"]); stt = stt[stt["size"] >= 50]
    rows_d = []
    for per, mm in (("探索", mx), ("評価", mv)):
        s = mm.join(stt, on="t1", how="inner")
        s["st_m"] = pd.qcut(s["mean"], 3, labels=["速い", "中", "遅い"]); s["st_s"] = pd.qcut(s["std"], 3, labels=["安定", "中", "ばらつく"])
        for col, nm in (("st_m", "平均ST"), ("st_s", "STのばらつき")):
            for lv, ss in s.groupby(col, observed=True):
                r1 = ratio_ci(ss["date"].values, ss["w_out"].values, ss["p_out"].values); r2 = ratio_ci(ss["date"].values, ss["w_win"].values, ss["p_win"].values)
                rows_d.append({"期間": per, "区分": f"{nm}：{lv}", "レース": len(ss), "4〜6号艇1着 実際÷見込み": f"{r1[0]:.3f}（{r1[1]:.3f}〜{r1[2]:.3f}）",
                               "1号艇1着 実際÷見込み": f"{r2[0]:.3f}（{r2[1]:.3f}〜{r2[2]:.3f}）"})
    out += ["## (d) 1号艇の選手のスタートのクセ（2014〜2023-09 の平均ST・ばらつきで3分位）", "", pd.DataFrame(rows_d).to_markdown(index=False), ""]
    # 買い方（評価）
    sw = mv["t1"].map(ratios["win"]).fillna(1.0).values; s3 = mv["t1"].map(ratios["top3"]).fillna(1.0).values
    dv = mv["date"].values
    tf = k_tanfuku(a.bk, EVL[0], EVL[1])
    mt = mv[KEY].reset_index(drop=True).merge(tf, on=KEY, how="left")
    sel1 = (sw >= HI) & mt["tan1"].notna().values
    st1 = np.where(sel1, 100.0, 0); rt1 = np.where(sel1, mt["tan1"].fillna(0).values, 0)
    roi1 = boot_roi(dv, st1, rt1); act1 = ratio_ci(dv[sel1], mv["w_win"].values[sel1], mv["p_win"].values[sel1])
    # (2) 下位の選手：補正した確率で3連単
    Pv, Ov, vv, yv = P[ev], O[ev], v[ev], y[ev]
    pi = Pv ** 1.06; pi /= pi.sum(1, keepdims=True)
    A = pi[:, f1 == 1].sum(1); B = pi[:, in3 & (f1 != 1)].sum(1); C = 1 - A - B
    A2 = np.clip(sw * A, 0, 0.999); B2 = np.clip(s3 * (A + B) - A2, 0, 0.999 - A2); C2 = 1 - A2 - B2
    sc = np.where(f1[None, :] == 1, (A2 / np.where(A > 0, A, 1))[:, None], np.where(in3[None, :], (B2 / np.where(B > 0, B, 1))[:, None], (C2 / np.where(C > 0, C, 1))[:, None]))
    pic = pi * sc
    sel2 = sw <= LO
    buy = vv & sel2[:, None] & (Ov <= 300) & (pic * np.where(vv, Ov, 0) > 1.0)
    hit = np.zeros_like(vv); hit[np.arange(len(yv)), yv] = True
    st2 = buy.sum(1) * 100.0; rt2 = np.where(buy & hit, Ov * 100, 0).sum(1)
    roi2 = boot_roi(dv, st2, rt2); act2 = ratio_ci(dv[sel2], mv["w_win"].values[sel2], mv["p_win"].values[sel2])
    base = boot_roi(dv, np.where(mt["tan1"].notna().values, 100.0, 0), mt["tan1"].fillna(0).values)
    # 参考（判定に使わない）：探索の縮めた比で上位10%・下位10%の選手が1号艇のときの単勝1
    xx0 = gx_win[gx_win["e"] >= 5]; q10, q90 = np.quantile(xx0["s"], [.1, .9])
    ref = []
    for nm, selset in (("参考：探索の比が上位10%の選手が1号艇 → 単勝1", set(xx0[xx0["s"] >= q90]["t1"])), ("参考：探索の比が下位10%の選手が1号艇 → 単勝1", set(xx0[xx0["s"] <= q10]["t1"]))):
        sl = mv["t1"].isin(selset).values & mt["tan1"].notna().values
        rr = boot_roi(dv, np.where(sl, 100.0, 0), np.where(sl, mt["tan1"].fillna(0).values, 0)); aa = ratio_ci(dv[sl], mv["w_win"].values[sl], mv["p_win"].values[sl])
        ref.append({"買い方": nm, "対象レース": int(sl.sum()), "点数": int(sl.sum()), "1号艇1着 実際÷見込み": f"{aa[0]:.3f}（{aa[1]:.3f}〜{aa[2]:.3f}）", "回収率": rr[0], "95%CI下": rr[1], "95%CI上": rr[2]})
    bets = [{"買い方": f"(1) 比 ≥ {HI} の選手が1号艇 → 単勝1（K の実額）", "対象レース": int(sel1.sum()), "点数": int(sel1.sum()),
             "1号艇1着 実際÷見込み": f"{act1[0]:.3f}（{act1[1]:.3f}〜{act1[2]:.3f}）", "回収率": roi1[0], "95%CI下": roi1[1], "95%CI上": roi1[2]},
            {"買い方": f"(2) 比 ≤ {LO} の選手が1号艇 → 補正π×オッズ>1.0・300倍以下の3連単", "対象レース": int(sel2.sum()), "点数": int(buy.sum()),
             "1号艇1着 実際÷見込み": f"{act2[0]:.3f}（{act2[1]:.3f}〜{act2[2]:.3f}）", "回収率": roi2[0], "95%CI下": roi2[1], "95%CI上": roi2[2]},
            {"買い方": "参考：全レースの単勝1", "対象レース": int(mt["tan1"].notna().sum()), "点数": int(mt["tan1"].notna().sum()), "1号艇1着 実際÷見込み": "",
             "回収率": base[0], "95%CI下": base[1], "95%CI上": base[2]}] + ref
    ok1 = sel1.sum() >= 300 and roi1[0] >= 1.1 and roi1[1] > 1.0
    ok2 = buy.sum() >= 300 and roi2[0] >= 1.1 and roi2[1] > 1.0
    judge = "採用候補（前向きの検証へ）" if (ok1 or ok2) else "不採用"
    # 極端な選手（探索の縮めた比、見込み5回以上）
    gx_win["r_ev"] = gx_win["t1"].map(per_racer(mv, "t1", "w_win", "p_win", "v_win").set_index("t1").eval("o / e"))
    gx_win["n_ev"] = gx_win["t1"].map(per_racer(mv, "t1", "w_win", "p_win", "v_win").set_index("t1")["n"])
    xx = gx_win[gx_win["e"] >= 5].sort_values("s")
    ext = pd.concat([xx.tail(5).iloc[::-1], xx.head(5)])[["t1", "n", "o", "e", "s", "n_ev", "r_ev"]]
    ext.columns = ["登録番号", "探索 1号艇の回数", "探索 1着", "探索 見込み", "探索 縮めた比", "評価 1号艇の回数", "評価 実際÷見込み"]
    # 12年の参考：年ごとの1号艇1着率の残差（年×場の平均との差）の、前年→翌年の相関
    hr = norm(pd.read_parquet(f"{a.histds}/races.parquet", columns=KEY + ["tri_combo"]))
    he = norm(pd.read_parquet(f"{a.histds}/entries.parquet", columns=KEY + ["boat", "toban"]))
    cur_r = norm(pd.read_parquet(f"{a.datadir}/races.parquet", columns=KEY + ["tri_combo"])); cur_e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban"]))
    allr = pd.concat([hr, cur_r]).dropna().drop_duplicates(KEY, keep="last"); alle = pd.concat([he, cur_e]).drop_duplicates(KEY + ["boat"], keep="last")
    allr = allr[allr["tri_combo"].astype(str).str[0].str.isdigit()]
    b1 = allr.merge(alle[alle["boat"] == 1][KEY + ["toban"]], on=KEY)
    b1["win"] = (b1["tri_combo"].astype(str).str[0] == "1").astype(float); b1["year"] = b1["date"].str[:4]
    b1["avg"] = b1.groupby(["year", "jcd"])["win"].transform("mean"); b1["res"] = b1["win"] - b1["avg"]
    py = b1.groupby(["toban", "year"]).agg(res=("res", "mean"), n=("res", "size")).reset_index()
    rows_y = []
    yrs = sorted(py["year"].unique())
    for y0, y1 in zip(yrs[:-1], yrs[1:]):
        aa = py[py["year"] == y0].merge(py[py["year"] == y1], on="toban", suffixes=("_0", "_1"))
        aa = aa[(aa["n_0"] >= 10) & (aa["n_1"] >= 10)]
        w = np.minimum(aa["n_0"], aa["n_1"]).values
        rows_y.append({"前年→翌年": f"{y0}→{y1}", "選手数": len(aa), "残差の相関（重み付き）": wcorr(aa["res_0"].values, aa["res_1"].values, w)})
    Y = pd.DataFrame(rows_y)
    out = out[:2] + [f"**判定：{judge}**", "",
                     f"- (1) 単勝1：{int(sel1.sum())}点、回収率 {roi1[0]:.3f}（95%CI {roi1[1]:.3f}〜{roi1[2]:.3f}）。(2) 3連単：{int(buy.sum())}点、回収率 {roi2[0]:.3f}（95%CI {roi2[1]:.3f}〜{roi2[2]:.3f}）。",
                     f"- 選手ごとの1号艇の1着の比：k = {k_win:.1f}、探索→評価の相関 {rows_k[0]['探索→評価の相関（重み付き）']:.3f}。", ""] + out[2:]
    out += ["## 買い方（評価 2026-01〜08）", "", pd.DataFrame(bets).to_markdown(index=False, floatfmt=".3f"), "",
            "## 参考：探索で極端だった選手（縮めた比の上位5・下位5、登録番号のみ）", "", ext.to_markdown(index=False, floatfmt=".3f"), "",
            "## 参考：12年分（2014〜2026-08、オッズなし）の1号艇の1着率の残差（年×場の平均との差）の前年→翌年の相関（両年とも1号艇10回以上）", "",
            Y.to_markdown(index=False, floatfmt=".3f"), "", f"- 平均の相関：{Y['残差の相関（重み付き）'].mean():.3f}", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    main()
