"""IDEA-069 1号艇の複勝を、これまでの絞り込みで買う／IDEA-070 同じ絞り込みで1号艇の単勝（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea069.py DATADIR STOREDIR [--hist /tmp/claude-0/ds_hist] [--histbk /tmp/claude-0/hist] [--out results/idea069.md]
- 絞り込み（すべて既存の探索期間の結果で決まっていて、新しく学習しない）：
  (a) 1号艇が22歳以下（IDEA-054）
  (b) IDEA-053 の重み（models/idea056_weights.json）で 補正後÷市場 ≥ 1.15・≥ 1.25
  (c) 1号艇の選手が IDEA-060 の探索（2023-11〜2025-12）で縮めた比の上位10%（見込み5回以上の選手の中で。IDEA-060 の参考と同じ定義）
  (d) 1コースに有利な風の群（IDEA-066、2014〜2019 で決めた場ごとの群分け。レースの風向きで判定し、実際の進入コースは使わない）
  (e) (a)(b≥1.15)(c)(d) のうち2つ以上・3つ以上
- 買い方：1号艇の複勝（IDEA-069、判定）・単勝（IDEA-070、判定）を各100円。払戻は K ファイルの実額。返還のあるレース・K に単勝の払戻がないレースは除く
- 評価 2026-01〜08（判定）、参考 2023-11〜2025-12（(b)(c) は学習に使った期間）、12年分（2014〜、オッズなし）は (a)(d) と (b) の市場の項なし版（IDEA-056 段階2 と同じ Σw 上位4.5%・1.5%）
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY, TR, EV, FIRST, boat_feats, segments
from exp_idea053 import SEL, design, boot
from exp_idea056_stage2 import COLS, tanfuku
from exp_idea060 import est_k, shrink, per_racer
from exp_idea066 import wind_groups

def racer_top10(m, P, y):
    """IDEA-060 と同じ：探索の1号艇の1着の比（経験ベイズで縮める）が上位10%の選手（見込み5回以上）"""
    e1 = (FIRST[y] == 1).astype(float); p = P[:, FIRST == 1].sum(1)
    d = pd.DataFrame({"t1": m["t1"].values, "w": e1, "p": p, "v": p * (1 - p)})[((m["date"] >= TR[0]) & (m["date"] <= TR[1])).values]
    g = per_racer(d, "t1", "w", "p", "v"); k, _ = est_k(g["o"].values, g["e"].values, g["v"].values)
    g["s"] = shrink(g["o"].values, g["e"].values, k); g = g[g["e"] >= 5]
    q90 = np.quantile(g["s"], .9)
    return set(g.loc[g["s"] >= q90, "t1"]), k, q90

def roi_row(name, per, d, sel, tan, fuku):
    s = sel
    row = {"絞り込み": name, "期間": per, "レース": int(s.sum())}
    for lab, pay in (("複勝1", fuku), ("単勝1", tan)):
        if s.sum():
            roi, lo, hi = boot(d[s], np.full(s.sum(), 100.0), pay[s])
        else:
            roi = lo = hi = np.nan
        row[lab] = roi; row[f"{lab} CI下"] = lo; row[f"{lab} CI上"] = hi
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--hist", default="/tmp/claude-0/ds_hist"); ap.add_argument("--histbk", default="/tmp/claude-0/hist")
    ap.add_argument("--weights", default="models/idea056_weights.json"); ap.add_argument("--out", default="results/idea069.md")
    a = ap.parse_args()
    W = json.load(open(a.weights, encoding="utf-8")); w = np.array([x[2] for x in W["segments"]])
    assert [(s, v) for s, v, _ in W["segments"]] == list(SEL)
    # 市場のあるデータ（2023-10〜）
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo", "wind_dir"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=COLS))
    t1 = e[e["boat"] == 1][KEY + ["toban"]].drop_duplicates(KEY).rename(columns={"toban": "t1"})
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(boat_feats(e), on=KEY, how="left").merge(t1, on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); p1 = P[:, FIRST == 1].sum(1); m["p1"] = p1
    days = m["date"].values; tr = (days >= TR[0]) & (days <= TR[1])
    X = design(segments(m, tr, True, a.store), n)
    off = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1)); padj = 1 / (1 + np.exp(-(off + X @ w)))
    ratio = np.where(p1 > 0, padj / np.where(p1 > 0, p1, 1), np.nan)
    top10, k60, q90 = racer_top10(m, P, y)
    print("weights loaded; racer top10%", len(top10), "k", k60, flush=True)
    # 風向きの群（2014〜2019、全出走から）
    ce = KEY + ["boat", "course", "pos_raw"]
    eall = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=ce)) for p in (a.hist, a.datadir)], ignore_index=True).drop_duplicates(KEY + ["boat"], keep="last")
    rall = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["wind", "wind_dir"])) for p in (a.hist, a.datadir)], ignore_index=True).drop_duplicates(KEY, keep="last")
    G = wind_groups(eall, rall); fav = set(map(tuple, G.loc[G["grp"] == "1コース有利", ["jcd", "wind_dir"]].values))
    # 返還のあるレースを除き、K の単勝・複勝（1号艇）を付ける
    bad = e[e["pos_raw"].astype(str).str.match(r"^[KFL]")][KEY].drop_duplicates()
    tf = tanfuku([a.store], "20231101", EV[1])
    mm = m[KEY].assign(i=np.arange(n)).merge(bad.assign(bad=1), on=KEY, how="left")
    mm = mm[mm["bad"].isna()].merge(tf, on=KEY)
    idx = mm["i"].values; d = days[idx]; tan = mm["tan1"].values.astype(float); fuku = mm["fuku1"].values.astype(float)
    age1 = pd.to_numeric(m["age1"], errors="coerce").values[idx]
    F = {"(a) 1号艇が22歳以下": age1 <= 22,
         "(b) 補正後÷市場 ≥1.15": ratio[idx] >= 1.15,
         "(b) 補正後÷市場 ≥1.25": ratio[idx] >= 1.25,
         "(c) 逃げ上手の選手（IDEA-060 上位10%）": m["t1"].isin(top10).values[idx],
         "(d) 1コースに有利な風": np.array([(j, wd) in fav for j, wd in zip(m["jcd"].values[idx], m["wind_dir"].values[idx])])}
    cnt = (F["(a) 1号艇が22歳以下"].astype(int) + F["(b) 補正後÷市場 ≥1.15"].astype(int) + F["(c) 逃げ上手の選手（IDEA-060 上位10%）"].astype(int)
           + F["(d) 1コースに有利な風"].astype(int))
    F["(e) (a)(b≥1.15)(c)(d) のうち2つ以上"] = cnt >= 2
    F["(e) (a)(b≥1.15)(c)(d) のうち3つ以上"] = cnt >= 3
    ref = (d >= TR[0]) & (d <= TR[1]); ev = (d >= EV[0]) & (d <= EV[1])
    rows = []
    for per, msk in (("評価 2026-01〜08", ev), ("参考 2023-11〜2025-12", ref)):
        rows.append(roi_row("全レース（基準）", per, d, msk, tan, fuku))
        for nm, f in F.items():
            rows.append(roi_row(nm, per, d, msk & f, tan, fuku))
    R = pd.DataFrame(rows)
    E = R[(R["期間"].str.startswith("評価")) & (R["絞り込み"] != "全レース（基準）")]
    def judge(col):
        ok_ = E[(E["レース"] >= 300) & (E[col] >= 1.1) & (E[f"{col} CI下"] > 1.0)]
        return ("採用候補（前向きの検証で確かめ直す）：" + "、".join(ok_["絞り込み"])) if len(ok_) else "不採用"
    j69, j70 = judge("複勝1"), judge("単勝1")
    # 重なり（評価）
    names = list(F)[:5]; ov = pd.DataFrame(index=names, columns=names, dtype=float)
    for x1 in names:
        for x2 in names:
            ov.loc[x1, x2] = int((F[x1] & F[x2] & ev).sum())
    # 12年分（オッズなし）
    print("12-year...", flush=True)
    rh = norm(pd.read_parquet(f"{a.hist}/races.parquet"))[KEY + ["tri_combo", "wind_dir"]]
    eh = norm(pd.read_parquet(f"{a.hist}/entries.parquet", columns=COLS))
    ra = pd.concat([rh, r[KEY + ["tri_combo", "wind_dir"]]]).dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    ea = pd.concat([eh, e]); badh = ea[ea["pos_raw"].astype(str).str.match(r"^[KFL]")][KEY].drop_duplicates()
    ra = ra.merge(badh.assign(bad=1), on=KEY, how="left"); ra = ra[ra["bad"].isna()].drop(columns="bad")
    ra = ra[(ra["date"] >= "20140101") & (ra["date"] <= EV[1])]
    tfh = tanfuku([a.histbk, a.store], "20140101", EV[1])
    ra = ra.merge(tfh, on=KEY).merge(boat_feats(ea), on=KEY, how="left").reset_index(drop=True)
    xp = ((ra["date"] >= "20140101") & (ra["date"] <= "20191231")).values
    score = design(segments(ra, xp, False, None), len(ra)) @ w
    thr45 = float(np.quantile(score[xp], 1 - 0.045)); thr15 = float(np.quantile(score[xp], 1 - 0.015))
    ha = pd.to_numeric(ra["age1"], errors="coerce").values <= 22
    hd = np.array([(j, wd) in fav for j, wd in zip(ra["jcd"].values, ra["wind_dir"].values)])
    hb = score >= thr45
    HF = {"全レース": np.ones(len(ra), bool), "(a) 1号艇が22歳以下": ha, "(d) 1コースに有利な風": hd,
          "(b) Σw 上位4.5%（市場の項なし）": hb, "(b) Σw 上位1.5%（市場の項なし）": score >= thr15,
          "(a)(b4.5%)(d) のうち2つ以上": (ha.astype(int) + hb.astype(int) + hd.astype(int)) >= 2}
    yr = ra["date"].str[:4].values; ht = ra["tan1"].values.astype(float); hf = ra["fuku1"].values.astype(float)
    Hs, Ht = [], []
    for nm, f in HF.items():
        g = pd.DataFrame({"y": yr[f], "t": ht[f], "f": hf[f]}).groupby("y").agg(n=("t", "size"), t=("t", "mean"), f=("f", "mean"))
        g[["t", "f"]] /= 100
        Ht.append(g["f"].rename(nm)); Hs.append({"絞り込み": nm, "レース": int(f.sum()), "複勝1（全期間）": hf[f].sum() / (100 * f.sum()),
                                                  "複勝1 100%以上の年": f"{int((g['f'] >= 1).sum())}／{len(g)}", "単勝1（全期間）": ht[f].sum() / (100 * f.sum()),
                                                  "単勝1 100%以上の年": f"{int((g['t'] >= 1).sum())}／{len(g)}"})
    HS = pd.DataFrame(Hs); HT = pd.concat(Ht, axis=1)
    base = HS.iloc[0]
    HS["複勝1 基準以上の年"] = ""
    for i, (nm, f) in enumerate(HF.items()):
        if i == 0:
            continue
        HS.loc[i, "複勝1 基準以上の年"] = f"{int((HT[nm] > HT['全レース']).sum())}／{len(HT)}"
    show = lambda df: df.to_markdown(index=False, floatfmt=".3f")
    md = [f"# IDEA-069（複勝）・IDEA-070（単勝） 1号艇を既存の絞り込みで買う（評価 {EV[0]}〜{EV[1]}）", "",
          f"**IDEA-069（1号艇の複勝）の判定：{j69}**", "", f"**IDEA-070（1号艇の単勝）の判定：{j70}**", "",
          f"- 払戻は K ファイルの実額。返還のあるレース（欠場・F・L）と K に単勝の払戻がないレースは除く（評価 {int(ev.sum())}・参考 {int(ref.sum())}レース）。",
          f"- (b) の重みは `{a.weights}`（IDEA-053 の探索で学習、固定）。(c) は IDEA-060 と同じ計算（k = {k60:.1f}、上位10%のしきい値 {q90:.4f}、{len(top10)}人）。",
          "- (d) は IDEA-066 の群分け（2014〜2019）で、レースの風向き（結果ファイルの値）が「1コース有利」の群かどうか。進入コースは使わない。",
          "- (b)(c) は 2023-11〜2025-12 で決めたものなので、その期間の数字は参考（学習に使った期間）。判定は評価期間だけ。", "",
          "## 絞り込みごとの回収率（1号艇の複勝・単勝を各100円、日単位95%CI）", "", show(R), "",
          "## IDEA-070（単勝）", "",
          f"判定：{j70}。上の表の「単勝1」の列が IDEA-070 の判定に使う数字（評価期間、300点以上・回収率110%以上・CI下限 > 100%）。", "",
          show(E[["絞り込み", "レース", "単勝1", "単勝1 CI下", "単勝1 CI上"]]), "",
          "## 絞り込みどうしの重なり（評価期間のレース数）", "", ov.astype(int).to_markdown(), "",
          "## 参考：12年分（2014-01〜2026-08、オッズなし。市場の要る (b)(c) は除き、(b) は市場の項なしの Σw 版）", "",
          f"Σw のしきい値（2014〜2019 で決めて固定）：上位4.5% = {thr45:.3f}、上位1.5% = {thr15:.3f}。", "", show(HS), "",
          "### 年別の複勝1の回収率", "", HT.to_markdown(floatfmt=".3f"), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
