"""IDEA-067 体重の急な変化：体調・減量の手がかりを市場は見ているか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea067.py DATADIR HISTDS SEXCSV STORE [--out results/idea067.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ、12年の参考用）、SEXCSV：登録番号→性別（ファン手帳、コミットしない）、STORE：K ファイル（bk/）
- 節 = 同じ場の連続開催日（間が1日以上空いたら別の節）。節の体重 = その節で最初に出走した日の出走表の体重
- 体重の変化 = 今の節の体重 − その選手の前の節の体重（前の節の初日が60日以内のときだけ）。区分 −2kg以下／−1kg／±0／+1kg／+2kg以上
- 下限の近く = 男子 53kg 以下（下限52kg+1kg 以内）、女子 48kg 以下（下限47kg+1kg 以内）
- 区分 × 艇番（1／2〜3／4〜6）ごとに 1着率・3着以内率 ÷ 市場の見込み。探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で判定
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL, PSEUDO, ratio_ci, boot_roi
from exp_idea061 import market, k_tan

CHG = ["−2kg以下", "−1kg", "±0", "+1kg", "+2kg以上"]
BG = {1: "1号艇", 2: "2〜3号艇", 3: "2〜3号艇", 4: "4〜6号艇", 5: "4〜6号艇", 6: "4〜6号艇"}

def load(paths, sex):
    cols = KEY + ["boat", "toban", "weight", "pos_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in paths], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last").sort_values(KEY + ["boat"]).reset_index(drop=True)
    e["weight"] = pd.to_numeric(e["weight"], errors="coerce")
    s = e["toban"].map(sex); e["fem"] = np.where(s.notna(), s == 2, e["weight"] < 50)
    d = e[["jcd", "date"]].drop_duplicates().sort_values(["jcd", "date"])
    d["dt"] = pd.to_datetime(d["date"]); gap = d.groupby("jcd")["dt"].diff().dt.days.fillna(99)
    d["meet"] = d["jcd"] + "_" + (gap > 1).groupby(d["jcd"]).cumsum().astype(str)
    e = e.merge(d[["jcd", "date", "meet"]], on=["jcd", "date"], how="left")
    # 選手×節の体重（その節の最初の出走日の値）
    mw = e.dropna(subset=["weight"]).sort_values(KEY).groupby(["toban", "meet"]).agg(d0=("date", "first"), w=("weight", "first")).reset_index()
    mw["dt"] = pd.to_datetime(mw["d0"]); mw = mw.sort_values(["toban", "dt"])
    mw["pw"] = mw.groupby("toban")["w"].shift(); mw["pdt"] = mw.groupby("toban")["dt"].shift()
    mw["gap"] = (mw["dt"] - mw["pdt"]).dt.days
    mw["dw"] = np.where(mw["gap"] <= 60, mw["w"] - mw["pw"], np.nan)
    e = e.merge(mw[["toban", "meet", "dw"]], on=["toban", "meet"], how="left")
    dw = np.round(e["dw"])
    e["chg"] = np.select([dw <= -2, dw == -1, dw == 0, dw == 1, dw >= 2], CHG, default="")
    e["near"] = np.where(e["fem"], e["weight"] <= 48, e["weight"] <= 53)
    e["bg"] = e["boat"].astype(int).map(BG)
    return e

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("sexcsv"); ap.add_argument("store")
    ap.add_argument("--out", default="results/idea067.md"); a = ap.parse_args()
    sex = pd.read_csv(a.sexcsv).set_index("toban")["sex"]
    e = load([a.histds, a.datadir], sex)
    e = e[e["chg"] != ""].copy()
    print("体重の変化が出せた出走", len(e), e["chg"].value_counts().to_dict(), flush=True)
    m, O, v, P, y = market(a.datadir)
    first = np.array([int(c[0]) for c in COMBOS]); sets = [set(map(int, c.split("-"))) for c in COMBOS]
    pw = np.stack([P[:, first == b].sum(1) for b in range(1, 7)], 1)
    p3 = np.stack([P[:, [k for k, s in enumerate(sets) if b in s]].sum(1) for b in range(1, 7)], 1)
    fin = np.array([[int(x) for x in COMBOS[k].split("-")] for k in y])
    won = np.zeros((len(m), 6)); won[np.arange(len(m)), fin[:, 0] - 1] = 1
    top = np.zeros((len(m), 6))
    for j in range(3):
        top[np.arange(len(m)), fin[:, j] - 1] = 1
    m["ridx"] = np.arange(len(m))
    L = e.merge(m[KEY + ["ridx"]], on=KEY)
    bi = L["boat"].astype(int).values - 1; ri = L["ridx"].values
    L["pw"] = pw[ri, bi]; L["p3"] = p3[ri, bi]; L["won"] = won[ri, bi]; L["top"] = top[ri, bi]
    L["per"] = np.where((L["date"] >= EXP[0]) & (L["date"] <= EXP[1]), "探索", np.where((L["date"] >= EVL[0]) & (L["date"] <= EVL[1]), "評価", ""))
    L["sexl"] = np.where(L["fem"], "女子", "男子"); L["nearl"] = np.where(L["near"], "下限の近く", "下限から遠い")
    # 区分：変化×艇番、変化×性別×艇番、下限の近く×変化×艇番
    specs = []
    for c in CHG:
        for g in ("1号艇", "2〜3号艇", "4〜6号艇"):
            specs.append({"体重の変化": c, "性別": "全", "下限": "全", "艇番": g})
            for sx in ("男子", "女子"):
                specs.append({"体重の変化": c, "性別": sx, "下限": "全", "艇番": g})
            specs.append({"体重の変化": c, "性別": "全", "下限": "下限の近く", "艇番": g})
    def mask(D, sp):
        k = (D["chg"] == sp["体重の変化"]) & (D["bg"] == sp["艇番"])
        if sp["性別"] != "全":
            k &= D["sexl"] == sp["性別"]
        if sp["下限"] != "全":
            k &= D["nearl"] == sp["下限"]
        return k
    rows = []
    for sp in specs:
        row = dict(sp)
        for per in ("探索", "評価"):
            d = L[mask(L, sp) & (L["per"] == per)]
            row[f"{per} 艇数"] = len(d)
            for k, (act, exp) in {"1着": ("won", "pw"), "3着内": ("top", "p3")}.items():
                r_, lo, hi = ratio_ci(d["date"].values, d[act].values, d[exp].values) if len(d) else (np.nan,) * 3
                row[f"{per} {k}比"] = r_; row[f"{per} {k}CI"] = f"{lo:.2f}〜{hi:.2f}" if np.isfinite(lo) else ""
                row[f"_{per}{k}lo"] = lo; row[f"_{per}{k}hi"] = hi
        rows.append(row)
    R = pd.DataFrame(rows)
    sel = []
    for _, r_ in R.iterrows():
        for k in ("1着", "3着内"):
            if r_["探索 艇数"] >= 300 and ((r_[f"探索 {k}比"] >= 1.08 and r_[f"_探索{k}lo"] > 1) or (r_[f"探索 {k}比"] <= 0.92 and r_[f"_探索{k}hi"] < 1)):
                up = r_[f"探索 {k}比"] > 1
                held = (r_[f"_評価{k}lo"] > 1) if up else (r_[f"_評価{k}hi"] < 1)
                sel.append({"体重の変化": r_["体重の変化"], "性別": r_["性別"], "下限": r_["下限"], "艇番": r_["艇番"], "指標": k, "向き": "強い" if up else "弱い",
                            "探索 艇数": r_["探索 艇数"], "探索 比": r_[f"探索 {k}比"], "評価 艇数": r_["評価 艇数"], "評価 比": r_[f"評価 {k}比"],
                            "評価 CI": r_[f"評価 {k}CI"], "評価で残る": "はい" if held else "いいえ"})
    S = pd.DataFrame(sel)
    print("選ばれた区分", len(S), flush=True)
    # 買い方（評価）：単勝は K の実額、3連単は確定オッズ（π = P^1.06、その艇の1着か3着以内の組を縮めた探索の比で補正）
    tan = k_tan(f"{a.store}/bk", EVL[0], EVL[1])
    bets = []
    Lv = L[L["per"] == "評価"].merge(tan, on=KEY, how="left")
    for _, s_ in S.iterrows():
        sp = {c: s_[c] for c in ("体重の変化", "性別", "下限", "艇番")}
        d0 = L[mask(L, sp) & (L["per"] == "探索")]
        act, exp = ("won", "pw") if s_["指標"] == "1着" else ("top", "p3")
        ratio = (d0[act].sum() + PSEUDO) / (d0[exp].sum() + PSEUDO)
        dv = Lv[mask(Lv, sp)]
        if not len(dv):
            continue
        st = np.full(len(dv), 100.0)
        rt = np.where((dv["tan_boat"].values == dv["boat"].values), dv["tan_pay"].fillna(0).values, 0.0)
        roi, lo, hi = boot_roi(dv["date"].values, st, rt)
        b = dv["boat"].astype(int).values; ri = dv["ridx"].values
        Pb = P[ri] ** 1.06; Pb /= Pb.sum(1, keepdims=True)
        if s_["指標"] == "1着":
            mk = first[None, :] == b[:, None]
        else:
            mk = np.array([[bb in s for s in sets] for bb in b])
        Pc = Pb * np.where(mk, ratio, 1.0); Pc /= Pc.sum(1, keepdims=True)
        Ov = np.where(v[ri], O[ri], 0); buy = v[ri] & (Ov <= 300) & (Pc * Ov > 1.0)
        hit = np.zeros_like(buy); hit[np.arange(len(ri)), y[ri]] = True
        st3 = buy.sum(1) * 100.0; rt3 = np.where(buy & hit, Ov * 100, 0).sum(1)
        r3, l3, h3 = boot_roi(dv["date"].values, st3, rt3)
        bets.append({**sp, "指標": s_["指標"], "向き": s_["向き"], "縮めた探索の比": ratio,
                     "単勝 点数": len(dv), "単勝 回収率": roi, "単勝 CI": f"{lo:.2f}〜{hi:.2f}" if np.isfinite(lo) else "",
                     "3連単 点数": int(buy.sum()), "3連単 回収率": r3, "3連単 CI": f"{l3:.2f}〜{h3:.2f}" if np.isfinite(l3) else "",
                     "_tlo": lo, "_tn": len(dv), "_troi": roi, "_3lo": l3, "_3n": int(buy.sum()), "_3roi": r3})
    B = pd.DataFrame(bets)
    passed = []
    for _, b_ in B.iterrows():
        nm = f"{b_['体重の変化']}・{b_['性別']}・{b_['下限']}・{b_['艇番']}"
        if b_["向き"] == "強い" and b_["_tn"] >= 300 and b_["_troi"] >= 1.1 and b_["_tlo"] > 1.0:
            passed.append(nm + "（単勝）")
        if b_["_3n"] >= 300 and b_["_3roi"] >= 1.1 and b_["_3lo"] > 1.0:
            passed.append(nm + "（3連単）")
    judge = "採用候補（前向きの検証へ）：" + "、".join(passed) if passed else "不採用"
    # 12年の参考（オッズなし）：区分×艇番の1着率・3着以内率 ÷ 同じ年・同じ艇番の全出走
    ea = load([a.histds, a.datadir], sex)
    ea = ea[ea["pos_raw"].astype(str).str.match(r"^0[1-6]$")].copy()
    ea["year"] = ea["date"].str[:4]; ea["won"] = (ea["pos_raw"] == "01").astype(float); ea["top"] = ea["pos_raw"].isin(["01", "02", "03"]).astype(float)
    base = ea.groupby(["year", "bg"])[["won", "top"]].mean()
    h = ea[ea["chg"] != ""].groupby(["chg", "bg", "year"])[["won", "top"]].agg(["mean", "size"]).reset_index()
    h.columns = ["chg", "bg", "year", "wm", "n", "tm", "n2"]
    h = h.merge(base.reset_index().rename(columns={"won": "bw", "top": "bt"}), on=["year", "bg"])
    h["w_r"] = h["wm"] / h["bw"]; h["t_r"] = h["tm"] / h["bt"]
    Hs = h.groupby(["chg", "bg"]).agg(年数=("year", "size"), 出走=("n", "sum"), w=("w_r", "mean"), wu=("w_r", lambda x: int((x > 1).sum())),
                                     t=("t_r", "mean"), tu=("t_r", lambda x: int((x > 1).sum()))).reset_index()
    Hs["chg"] = pd.Categorical(Hs["chg"], CHG); Hs = Hs.sort_values(["chg", "bg"])
    Hs.columns = ["体重の変化", "艇番", "年数", "出走", "1着率÷全体（年平均）", "1着が全体を上回った年", "3着内率÷全体（年平均）", "3着内が全体を上回った年"]
    Rshow = R[[c for c in R.columns if not c.startswith("_")]]
    nexp = int((L["per"] == "探索").sum()); nev = int((L["per"] == "評価").sum())
    md = [f"# IDEA-067 体重の急な変化（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}）", "",
          f"**判定：{judge}**", "",
          "## 定義", "",
          "- 節 = 同じ場の連続開催日。節の体重 = その節で最初に出走した日の出走表（B ファイル）の体重（kg、整数）。",
          "- 体重の変化 = 今の節の体重 − その選手の前の節の体重（前の節の初日から60日以内のときだけ。それ以外は対象外）。",
          "- 性別はファン手帳の実データ（載っていない選手のみ体重50kg未満を女子とみなす）。下限の近く = 男子 53kg 以下・女子 48kg 以下。",
          f"- 市場の見込みと比べた出走：探索 {nexp:,}／評価 {nev:,}。12年の参考は 2014〜2026 の B/K（オッズなし）。", "",
          "## 選ばれた区分（探索で 300艇以上・比 1.08 以上か 0.92 以下・CI が 1 をまたがない）と評価", "",
          (S.to_markdown(index=False, floatfmt=".3f") if len(S) else "なし"), "",
          "## 買い方（評価 2026-01〜08、単勝は K の実額、3連単は確定オッズで π×オッズ > 1.0・300倍以下）", "",
          (B[[c for c in B.columns if not c.startswith("_")]].to_markdown(index=False, floatfmt=".3f") if len(B) else "なし"), "",
          "## 12年の参考（オッズなし、区分×艇番の 1着率・3着以内率 ÷ 同じ年・同じ艇番の全出走）", "",
          "体重の変化は選手の実力とは関係が薄いはずなので、全体との差がそのまま市場の見落としになるとは限らない（市場との比較は上の表）。", "",
          Hs.to_markdown(index=False, floatfmt=".3f"), "",
          "## 全区分（探索・評価）", "", Rshow.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:12]))
    print(S.to_string() if len(S) else "選ばれた区分なし")
    print(B[[c for c in B.columns if not c.startswith("_")]].to_string() if len(B) else "")
    print(Hs.to_string())

if __name__ == "__main__":
    main()
