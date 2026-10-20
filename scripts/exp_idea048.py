"""IDEA-048 荒れ方の型を12年分で確かめる（docs/ideas.md の登録どおり。2014〜2026、結果・出走表ファイルのみ）
usage: python scripts/exp_idea048.py HISTDS DATADIR [--out results/idea048.md]
- HISTDS：data-hist の B/K（2014-01〜2023-09）から build_dataset で作ったテーブル、DATADIR：既存（2023-10〜）
- 条件 × 荒れ方の型ごとに、型に当てはまる3連単を全部各100円買ったときの回収率（払戻は K の実額）
- 欠場・フライング・出遅れ（返還が出る）のあるレースは除く
- 探索 2014〜2019 → 確認 2020-01〜2023-09 → 最終確認 2023-10〜2026-08
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp_idea036 import norm

KEY = ["date", "jcd", "rno"]
EXP = ("20140101", "20191231"); CONF = ("20200101", "20230930"); FIN = ("20231001", "20260831")
RNG = np.random.default_rng(0)

def load(ds, lo, hi):
    r = norm(pd.read_parquet(f"{ds}/races.parquet", columns=KEY + ["wind", "wave", "tri_combo", "tri_pay"]))
    r = r[(r["date"] >= lo) & (r["date"] <= hi)].drop_duplicates(KEY)
    e = norm(pd.read_parquet(f"{ds}/entries.parquet", columns=KEY + ["boat", "cls", "nat_win", "pos_raw"]))
    e = e[(e["date"] >= lo) & (e["date"] <= hi)]
    return r, e

def classes():
    """荒れ方の型 -> (名前, 当てはまる結果か判定する関数(w,s,p1) , 組の数)"""
    out = []
    for k in range(1, 7):
        out.append((f"(a) 1着{k}号艇", lambda w, s, p, k=k: w == k, 20))
    for p, nm, nc in ((1, "1着", 20), (2, "2着", 20), (3, "3着", 20), (4, "4着以下", 60)):
        out.append((f"(b) 1号艇{nm}", lambda w, s, q, p=p: (q == p) if p < 4 else (q >= 4), nc))
    out.append(("(c) 1着1号艇", lambda w, s, q: w == 1, 20))
    for k in range(2, 7):
        for p, nm, nc in ((2, "1号艇2着", 4), (3, "1号艇3着", 4), (4, "1号艇4着以下", 12)):
            out.append((f"(c) 1着{k}号艇・{nm}", lambda w, s, q, k=k, p=p: (w == k) & ((q == p) if p < 4 else (q >= 4)), nc))
    for i in range(1, 7):
        for j in range(1, 7):
            if i != j:
                out.append((f"(d) 2連単{i}-{j}", lambda w, s, q, i=i, j=j: (w == i) & (s == j), 4))
    return out

def boot(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(2000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("histds"); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea048.md")
    a = ap.parse_args()
    r1, e1 = load(a.histds, EXP[0], CONF[1]); r2, e2 = load(a.datadir, FIN[0], FIN[1])
    r = pd.concat([r1, r2], ignore_index=True); e = pd.concat([e1, e2], ignore_index=True)
    # 返還のあるレース（欠場 K・フライング F・出遅れ L など）と6艇そろわないレースを除く
    pr = e["pos_raw"].astype(str).str.strip()
    bad = e[~pr.str.fullmatch(r"0?[1-6]")][KEY].drop_duplicates()
    cnt = e.groupby(KEY).size().rename("n").reset_index()
    r = r.merge(cnt, on=KEY).query("n == 6").drop(columns="n")
    r = r.merge(bad.assign(bad=1), on=KEY, how="left"); r = r[r["bad"].isna()].drop(columns="bad")
    r = r[r["tri_combo"].astype(str).str.fullmatch(r"[1-6]-[1-6]-[1-6]") & (pd.to_numeric(r["tri_pay"], errors="coerce") > 0)].reset_index(drop=True)
    w = r["tri_combo"].str[0].astype(int).values; s = r["tri_combo"].str[2].astype(int).values; t = r["tri_combo"].str[4].astype(int).values
    p1 = np.where(w == 1, 1, np.where(s == 1, 2, np.where(t == 1, 3, 4)))
    pay = pd.to_numeric(r["tri_pay"]).values.astype(float)
    # 条件
    b1 = e[e["boat"] == 1][KEY + ["cls"]].drop_duplicates(KEY)
    e["rk"] = e.groupby(KEY)["nat_win"].rank(ascending=False, method="min")
    rk1 = e[e["boat"] == 1][KEY + ["rk"]].drop_duplicates(KEY)
    r = r.merge(b1, on=KEY, how="left").merge(rk1, on=KEY, how="left")
    wind = pd.to_numeric(r["wind"], errors="coerce").values; wave = pd.to_numeric(r["wave"], errors="coerce").values
    days = r["date"].values; year = r["date"].str[:4].values
    exp_m = (days >= EXP[0]) & (days <= EXP[1]); conf_m = (days >= CONF[0]) & (days <= CONF[1]); fin_m = days >= FIN[0]
    man = pay >= 10000
    vr = pd.Series(man[exp_m]).groupby(r["jcd"].values[exp_m]).mean().sort_values(ascending=False); rough = sorted(vr.index[:6])
    clsB = r["cls"].astype(str).str.startswith("B").values
    conds = {"全レース": np.ones(len(r), bool), "風速5m以上": wind >= 5, "波高5cm以上": wave >= 5, "1号艇がB級": clsB,
             "1号艇の全国勝率3位以下": r["rk"].values >= 3, f"荒れやすい場（{','.join(rough)}）": r["jcd"].isin(rough).values,
             "風速5m以上かつ1号艇がB級": (wind >= 5) & clsB}
    CL = classes(); years = sorted(set(year))
    rows = []; yr_rows = []
    for cn, cm in conds.items():
        for nm, f, nc in CL:
            hit = f(w, s, p1)
            rt = np.where(hit, pay, 0.0); st = np.full(len(r), nc * 100.0)
            row = {"条件": cn, "型": nm, "点数": nc}
            for per, pm in (("探索", exp_m), ("確認", conf_m), ("最終", fin_m)):
                k = cm & pm; row[f"{per} レース"] = int(k.sum()); row[f"{per} 回収率"] = rt[k].sum() / st[k].sum() if k.any() else np.nan
            yroi = {}
            for y in years:
                k = cm & (year == y); yroi[y] = rt[k].sum() / st[k].sum() if k.any() else np.nan
            row["探索で100%以上の年数"] = int(sum(yroi[y] >= 1.0 for y in years if "2014" <= y <= "2019"))
            row["全年で100%以上の年数"] = int(sum(yroi[y] >= 1.0 for y in years)); row["年数"] = len(years)
            rows.append(row); yr_rows.append({"条件": cn, "型": nm, **{y: yroi[y] for y in years}})
    R = pd.DataFrame(rows); Y = pd.DataFrame(yr_rows)
    sel = R[(R["探索 回収率"] >= 1.0) & (R["探索で100%以上の年数"] >= 5)].copy()
    res = []
    for _, x in sel.iterrows():
        cm = conds[x["条件"]]; f, nc = next((f, nc) for nm, f, nc in CL if nm == x["型"])
        hit = f(w, s, p1); rt = np.where(hit, pay, 0.0); st = np.full(len(r), nc * 100.0)
        k = cm & conf_m; lo, hi = boot(days[k], st[k], rt[k])
        ok = (x["確認 回収率"] >= 1.1) and (lo > 1.0) and (x["最終 回収率"] >= 1.0)
        res.append({**x[["条件", "型", "点数", "探索 回収率", "探索で100%以上の年数", "確認 レース", "確認 回収率"]].to_dict(), "確認 95%CI下": lo, "確認 95%CI上": hi,
                    "最終 レース": x["最終 レース"], "最終 回収率": x["最終 回収率"], "判定": "採用候補" if ok else "不採用"})
    S = pd.DataFrame(res)
    corr = float(np.corrcoef(R["探索 回収率"], R["確認 回収率"])[0, 1]); corr2 = float(np.corrcoef(R["確認 回収率"], R["最終 回収率"].fillna(0))[0, 1])
    base = {y: pay[(year == y)].sum() / (120 * 100.0 * (year == y).sum()) for y in years}
    judge = "採用候補あり" if len(S) and (S["判定"] == "採用候補").any() else "不採用"
    top = R.sort_values(["全年で100%以上の年数", "探索 回収率"], ascending=False).head(15)
    md = [f"# IDEA-048 荒れ方の型を12年分で確かめる（{len(r)}レース、探索 {EXP[0]}〜{EXP[1]}／確認 {CONF[0]}〜{CONF[1]}／最終 {FIN[0]}〜{FIN[1]}）", "",
          f"**判定：{judge}**　探索で選ばれた「条件×型」：{len(S)}個（全 {len(R)}個のうち）", "",
          f"探索↔確認の回収率の相関（全 {len(R)} 個）：{corr:.3f}、確認↔最終：{corr2:.3f}", "",
          "返還のあるレース（欠場・フライング・出遅れ）と6艇そろわないレースは除いた。荒れやすい場は探索期間の万舟率の上位6場。", "",
          "## 全組を買った場合の回収率（年ごと、参考）", "", pd.DataFrame([base]).to_markdown(index=False, floatfmt=".3f"), "",
          "## 探索で選ばれた条件×型と、確認・最終での成績", "", S.to_markdown(index=False, floatfmt=".3f") if len(S) else "（なし）", "",
          "## 100%を超えた年数が多い条件×型（上位15）", "", top[["条件", "型", "点数", "探索 回収率", "確認 回収率", "最終 回収率", "全年で100%以上の年数", "年数"]].to_markdown(index=False, floatfmt=".3f"), "",
          "## 上位15の年ごとの回収率", "", Y.merge(top[["条件", "型"]], on=["条件", "型"]).to_markdown(index=False, floatfmt=".2f"), "",
          "## 全条件×型", "", R.drop(columns=["年数"]).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:30]))

if __name__ == "__main__":
    main()
