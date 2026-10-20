"""IDEA-065 1日2走目：1走目の結果に市場が引っ張られすぎていないか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea065.py DATADIR HISTDS STORE [--out results/idea065.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ、12年の参考用）、STORE：K ファイル（bk/、単勝の払戻）
- 対象：同じ日・同じ場で2回目の出走（2走目）の艇。2走目が欠場の艇は除く
- 区分：1走目の着順（1着／2〜3着／4〜6着／事故・欠場）、1走目のスタートタイミング（2014〜2022 の通常の出走の3分位で 速い／普通／遅い、事故などで不明は除く）、
  1走目の艇番の群 → 2走目の艇番の群（1／2〜3／4〜6）。それぞれ × 2走目の艇番（1／2〜3／4〜6）
- 1着率・3着以内率 ÷ 市場の見込み（確定3連単）。探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で判定（IDEA-063 と同じ基準）
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL, PSEUDO, ratio_ci, boot_roi
from exp_idea061 import market, k_tan

FIN = {"01", "02", "03", "04", "05", "06"}

def bgroup(b):
    b = np.asarray(b, float)
    return np.where(b == 1, "1", np.where(b <= 3, "2〜3", "4〜6"))

def market_boats(datadir):
    """レースごとの市場の艇別 1着・3着以内の確率と結果（行番号 ridx で引く）"""
    m, O, v, P, y = market(datadir)
    first = np.array([int(c[0]) for c in COMBOS]); sets = [set(map(int, c.split("-"))) for c in COMBOS]
    pw = np.stack([P[:, first == b].sum(1) for b in range(1, 7)], 1)
    p3 = np.stack([P[:, [k for k, s in enumerate(sets) if b in s]].sum(1) for b in range(1, 7)], 1)
    fin = np.array([[int(x) for x in COMBOS[k].split("-")] for k in y])
    won = np.zeros((len(m), 6)); won[np.arange(len(m)), fin[:, 0] - 1] = 1
    top = np.zeros((len(m), 6))
    for j in range(3):
        top[np.arange(len(m)), fin[:, j] - 1] = 1
    m = m.copy(); m["ridx"] = np.arange(len(m))
    return m, O, v, P, y, pw, p3, won, top, first, sets

def attach(L, mk):
    """艇の行（KEY + boat）に市場の確率と結果を付ける"""
    m, O, v, P, y, pw, p3, won, top, first, sets = mk
    L = L.merge(m[KEY + ["ridx"]], on=KEY)
    bi = L["boat"].astype(int).values - 1; ri = L["ridx"].values
    L["pw"] = pw[ri, bi]; L["p3"] = p3[ri, bi]; L["won"] = won[ri, bi]; L["top"] = top[ri, bi]
    L = L[L["pw"] > 0].copy()
    L["per"] = np.where((L["date"] >= EXP[0]) & (L["date"] <= EXP[1]), "探索", np.where((L["date"] >= EVL[0]) & (L["date"] <= EVL[1]), "評価", ""))
    return L

def evaluate(L, dims, mk, store, metrics=("1着", "3着内")):
    """dims：{区分名: L の列名}。各区分 × 値 ごとに 実際÷見込み（探索・評価）、選択、評価での買い方"""
    m, O, v, P, y, pw, p3, won, top, first, sets = mk
    MET = {"1着": ("won", "pw"), "3着内": ("top", "p3")}
    rows = []
    for dn, col in dims.items():
        for val in sorted(L[col].dropna().unique()):
            row = {"区分": dn, "値": val}
            for per in ("探索", "評価"):
                d = L[(L[col] == val) & (L["per"] == per)]
                row[f"{per} 艇数"] = len(d)
                for k in metrics:
                    act, exp = MET[k]
                    r_, lo, hi = ratio_ci(d["date"].values, d[act].values, d[exp].values) if len(d) else (np.nan,) * 3
                    row[f"{per} {k}比"] = r_; row[f"{per} {k}CI"] = f"{lo:.2f}〜{hi:.2f}" if np.isfinite(lo) else ""
                    row[f"_{per}{k}lo"] = lo; row[f"_{per}{k}hi"] = hi
            rows.append(row)
    R = pd.DataFrame(rows)
    sel = []
    for _, r_ in R.iterrows():
        for k in metrics:
            if r_["探索 艇数"] >= 300 and ((r_[f"探索 {k}比"] >= 1.08 and r_[f"_探索{k}lo"] > 1) or (r_[f"探索 {k}比"] <= 0.92 and r_[f"_探索{k}hi"] < 1)):
                up = r_[f"探索 {k}比"] > 1
                held = (r_[f"_評価{k}lo"] > 1) if up else (r_[f"_評価{k}hi"] < 1)
                sel.append({"区分": r_["区分"], "値": r_["値"], "指標": k, "向き": "強い" if up else "弱い", "探索 比": r_[f"探索 {k}比"],
                            "評価 比": r_[f"評価 {k}比"], "評価 CI": r_[f"評価 {k}CI"], "評価 艇数": r_["評価 艇数"], "評価で残る": "はい" if held else "いいえ"})
    S = pd.DataFrame(sel)
    bets = []; pool = []
    if len(S):
        tan = k_tan(f"{store}/bk", EVL[0], EVL[1])
        Lv = L[L["per"] == "評価"].merge(tan, on=KEY, how="left")
        for _, s_ in S.iterrows():
            col = dims[s_["区分"]]
            act, exp = MET[s_["指標"]]
            d0 = L[(L["per"] == "探索") & (L[col] == s_["値"])]
            ratio = (d0[act].sum() + PSEUDO) / (d0[exp].sum() + PSEUDO)
            dv = Lv[Lv[col] == s_["値"]]
            boats = dv["boat"].astype(int).values
            st = np.full(len(dv), 100.0)
            rt = np.where(dv["tan_boat"].fillna(-1).values == boats, dv["tan_pay"].fillna(0).values, 0.0)
            roi, lo, hi = boot_roi(dv["date"].values, st, rt)
            ri = dv["ridx"].values; Pb = P[ri] ** 1.06; Pb /= Pb.sum(1, keepdims=True)
            if s_["指標"] == "1着":
                mask = first[None, :] == boats[:, None]
            else:
                mask = np.array([[b in s for s in sets] for b in boats])
            Pc = Pb * np.where(mask, ratio, 1.0); Pc /= Pc.sum(1, keepdims=True)
            Ov = np.where(v[ri], O[ri], 0); buy = v[ri] & (Ov <= 300) & (Pc * Ov > 1.0)
            hit = np.zeros_like(buy); hit[np.arange(len(ri)), y[ri]] = True
            st3 = buy.sum(1) * 100.0; rt3 = np.where(buy & hit, Ov * 100, 0).sum(1)
            r3, l3, h3 = boot_roi(dv["date"].values, st3, rt3)
            strong = s_["向き"] == "強い"
            bets.append({"区分": s_["区分"], "値": s_["値"], "指標": s_["指標"], "向き": s_["向き"], "縮めた探索の比": ratio,
                         "単勝 点数": len(dv) if strong else 0, "単勝 回収率": roi if strong else np.nan,
                         "単勝 CI": (f"{lo:.2f}〜{hi:.2f}" if np.isfinite(lo) else "") if strong else "",
                         "3連単 点数": int(buy.sum()), "3連単 回収率": r3, "3連単 CI": f"{l3:.2f}〜{h3:.2f}" if np.isfinite(l3) else "",
                         "_tn": len(dv) if strong else 0, "_troi": roi, "_tlo": lo, "_3n": int(buy.sum()), "_3roi": r3, "_3lo": l3})
            pool.append(pd.DataFrame({"d": dv["date"].values, "st": st3, "rt": rt3, "ts": st if strong else 0 * st, "tr": rt if strong else 0 * rt}))
    B = pd.DataFrame(bets)
    passed = []
    for _, b_ in B.iterrows():
        if b_["_tn"] >= 300 and b_["_troi"] >= 1.1 and b_["_tlo"] > 1.0:
            passed.append(f"{b_['区分']}={b_['値']}（単勝）")
        if b_["_3n"] >= 300 and b_["_3roi"] >= 1.1 and b_["_3lo"] > 1.0:
            passed.append(f"{b_['区分']}={b_['値']}（3連単）")
    pooled = ""
    if pool:
        T = pd.concat(pool); r3, l3, h3 = boot_roi(T["d"].values, T["st"].values, T["rt"].values)
        rt_, lt_, ht_ = boot_roi(T["d"].values, T["ts"].values, T["tr"].values)
        pooled = (f"選ばれた区分をまとめて（評価）：3連単 {int(T['st'].sum() / 100)}点・回収率 {r3:.3f}（95%CI {l3:.3f}〜{h3:.3f}）、"
                  f"単勝（強い向きの区分のみ）{int(T['ts'].sum() / 100)}点・回収率 {rt_:.3f}（95%CI {lt_:.3f}〜{ht_:.3f}）")
    judge = "採用候補（前向きの検証へ）：" + "、".join(passed) if passed else "不採用"
    return R, S, B, pooled, judge

def baseline(mk):
    """参考：全出走の艇番群ごとの 1着・3着以内の 実際÷市場の見込み（探索・評価）"""
    m, O, v, P, y, pw, p3, won, top, first, sets = mk
    rows = []
    d = m["date"].values
    for per, (a, b) in (("探索", EXP), ("評価", EVL)):
        msk = (d >= a) & (d <= b)
        for g, idx in (("1", [0]), ("2〜3", [1, 2]), ("4〜6", [3, 4, 5])):
            dd = np.repeat(d[msk], len(idx))
            rw = ratio_ci(dd, won[msk][:, idx].ravel(), pw[msk][:, idx].ravel())
            rt = ratio_ci(dd, top[msk][:, idx].ravel(), p3[msk][:, idx].ravel())
            rows.append({"期間": per, "艇番群": g, "1着比": rw[0], "1着CI": f"{rw[1]:.2f}〜{rw[2]:.2f}", "3着内比": rt[0], "3着内CI": f"{rt[1]:.2f}〜{rt[2]:.2f}"})
    return pd.DataFrame(rows)

def second_starts(e):
    e = e.copy(); e["pos_raw"] = e["pos_raw"].astype(str)
    e = e.sort_values(["date", "jcd", "toban", "rno"])
    g = e.groupby(["date", "jcd", "toban"])
    e["k"] = g.cumcount(); e["n"] = g["rno"].transform("size")
    f = e[(e["n"] == 2) & (e["k"] == 0)][["date", "jcd", "toban", "boat", "pos_raw", "st"]].rename(columns={"boat": "b1", "pos_raw": "pos1", "st": "st1"})
    s = e[(e["n"] == 2) & (e["k"] == 1)]
    s = s[~s["pos_raw"].str.startswith("K")]  # 2走目が欠場は除く
    s = s.merge(f, on=["date", "jcd", "toban"])
    return s

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("store")
    ap.add_argument("--out", default="results/idea065.md"); a = ap.parse_args()
    cols = KEY + ["boat", "toban", "pos_raw", "st"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in (a.histds, a.datadir)], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last")
    s = second_starts(e)
    # 1走目のスタートの3分位（2014〜2022 の通常の出走）
    ok = e["pos_raw"].astype(str).isin(FIN) & (e["date"] <= "20221231") & e["st"].between(-0.5, 1.5)
    q1, q2 = np.quantile(e.loc[ok, "st"], [1 / 3, 2 / 3])
    p1 = s["pos1"].astype(str)
    s["first_pos"] = np.where(p1 == "01", "1着", np.where(p1.isin(["02", "03"]), "2〜3着", np.where(p1.isin(["04", "05", "06"]), "4〜6着", "事故・欠場")))
    fin1 = p1.isin(FIN) & s["st1"].between(-0.5, 1.5)
    s["first_st"] = pd.Series(np.where(s["st1"] <= q1, "速い", np.where(s["st1"] <= q2, "普通", "遅い")), index=s.index).where(fin1)
    s["bg"] = bgroup(s["boat"]); s["bg1"] = bgroup(s["b1"])
    s["d_pos"] = s["first_pos"] + "→" + s["bg"] + "号艇"
    s["d_st"] = s["first_st"].astype(str).where(s["first_st"].notna()) + "→" + s["bg"] + "号艇"
    s["d_b"] = s["bg1"] + "号艇→" + s["bg"] + "号艇"
    dims = {"1走目の着順→2走目の艇番": "d_pos", "1走目のST→2走目の艇番": "d_st", "1走目の艇番→2走目の艇番": "d_b"}
    print("2走目", len(s), s["first_pos"].value_counts().to_dict(), f"ST 3分位 {q1:.2f}/{q2:.2f}", flush=True)
    mk = market_boats(a.datadir)
    L = attach(s, mk)
    R, S, B, pooled, judge = evaluate(L, dims, mk, a.store)
    BL = baseline(mk)
    # 12年の参考（オッズなし）
    H = s.copy()
    H["won"] = (H["pos_raw"].astype(str) == "01").astype(float); H["top"] = H["pos_raw"].astype(str).isin(["01", "02", "03"]).astype(float)
    # 基準：同じ年・同じ艇番群の全出走（欠場を除く）
    allp = e[~e["pos_raw"].astype(str).str.startswith("K")].copy(); allp["bg"] = bgroup(allp["boat"])
    allp["won"] = (allp["pos_raw"].astype(str) == "01").astype(float); allp["top"] = allp["pos_raw"].astype(str).isin(["01", "02", "03"]).astype(float)
    allp["year"] = allp["date"].str[:4]; base = allp.groupby(["year", "bg"])[["won", "top"]].mean()
    H["year"] = H["date"].str[:4]
    out = []
    for dn, col in dims.items():
        g = H.dropna(subset=[col]).groupby([col, "year", "bg"]).agg(won=("won", "sum"), top=("top", "sum"), n=("won", "size")).reset_index()
        g = g.merge(base.rename(columns={"won": "bw", "top": "bt"}), left_on=["year", "bg"], right_index=True)
        g["ew"] = g["n"] * g["bw"]; g["et"] = g["n"] * g["bt"]
        yr = g.groupby([col, "year"])[["won", "ew", "top", "et", "n"]].sum().reset_index()
        yr["rw"] = yr["won"] / yr["ew"]; yr["rt"] = yr["top"] / yr["et"]
        t = yr.groupby(col).agg(年数=("year", "size"), 出走=("n", "sum"), r1=("rw", "mean"), u1=("rw", lambda x: int((x > 1).sum())),
                                r3=("rt", "mean"), u3=("rt", lambda x: int((x > 1).sum()))).reset_index().rename(columns={col: "値"})
        t.insert(0, "区分", dn); out.append(t)
    Hs = pd.concat(out, ignore_index=True)
    Hs.columns = ["区分", "値", "年数", "出走", "1着率÷同じ年・艇番群の全出走（年平均）", "1着が上回った年", "3着内率÷同じ年・艇番群の全出走（年平均）", "3着内が上回った年"]
    Rshow = R[[c for c in R.columns if not c.startswith("_")]]
    md = [f"# IDEA-065 1日2走目（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}）", "",
          f"**判定：{judge}**", "",
          "## 定義", "",
          f"- 対象：同じ日・同じ場でちょうど2回走った選手の2走目（2走目が欠場の艇は除く）。2014〜2026 で {len(s):,} 出走、確定オッズのある 2023-10 以降で {len(L):,}。",
          f"- 1走目の着順：1着／2〜3着／4〜6着／事故・欠場（F・L・転覆・失格・欠場など）。1走目のST：2014〜2022 の通常の出走の3分位（{q1:.2f} 以下＝速い、{q2:.2f} 以下＝普通、それ以上＝遅い。事故などでSTがない走は除く）。",
          "- 艇番の群：1／2〜3／4〜6。市場の見込みは確定3連単オッズの逆数を正規化した確率から。", "",
          "## 参考：全出走（2走目に限らない）の艇番群ごとの 実際÷市場の見込み", "", BL.to_markdown(index=False, floatfmt=".3f"), "",
          "## 選ばれた区分（探索で 300艇以上・比 1.08 以上か 0.92 以下・CI が 1 をまたがない）と評価", "",
          (S.to_markdown(index=False, floatfmt=".3f") if len(S) else "なし"), "",
          "## 買い方（評価 2026-01〜08、単勝は K の実額・強い向きのみ、3連単は確定オッズで π×オッズ > 1.0・300倍以下）", "",
          (B[[c for c in B.columns if not c.startswith("_")]].to_markdown(index=False, floatfmt=".3f") if len(B) else "なし"), "", pooled, "",
          "## 全区分（1着・3着以内の 実際÷市場の見込み）", "", Rshow.to_markdown(index=False, floatfmt=".3f"), "",
          "## 12年の参考（2014〜2026、オッズなし。同じ年・同じ艇番群の全出走との比）", "", Hs.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(BL.to_string()); print(judge); print(S.to_string() if len(S) else "選択なし"); print(B[[c for c in B.columns if not c.startswith("_")]].to_string() if len(B) else ""); print(pooled)

if __name__ == "__main__":
    main()
