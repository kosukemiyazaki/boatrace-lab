"""IDEA-032 確定時に割安になりやすい「場所」は決まっているか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea032.py DATADIR STOREDIR [--out results/idea032.md]
- 2連複 15組（DATADIR/odds2tf.parquet）と単勝 6艇（STOREDIR/oddstf/*.csv.gz）の確定オッズ、的中は3連単の着順から
- 区分 A 組 / B 組×場 / C 組×レース番号帯。学習 2025-10〜2026-04 で 点数300以上・回収率105%以上の区分を選び、評価 2026-05〜08 で判定
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, COMBOS_2F

KEY = ["date", "jcd", "rno"]
TR = ("20251001", "20260430"); EV = ("20260501", "20260831")
RNG = np.random.default_rng(0)

def norm(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2), rno=df["rno"].astype(int))

def boot(g):
    bi = RNG.integers(0, len(g), size=(5000, len(g)))
    b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea032.md")
    a = ap.parse_args()
    races = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    races = races[(races["date"] >= TR[0]) & (races["date"] <= EV[1])]
    o3 = norm(pd.read_parquet(f"{a.datadir}/odds.parquet"))
    q2 = norm(pd.read_parquet(f"{a.datadir}/odds2tf.parquet"))
    tf = norm(pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in sorted(glob.glob(f"{a.store}/oddstf/*.csv.gz"))]))
    m = races.merge(o3, on=KEY).merge(q2[KEY + COMBOS_2F], on=KEY).merge(tf[KEY + [f"tan{i}" for i in range(1, 7)]].drop_duplicates(KEY, keep="last"), on=KEY)
    O = m[COMBOS].to_numpy(float); ok = (np.isfinite(O) & (O > 0)).sum(1) >= 60
    m, O = m[ok].reset_index(drop=True), O[ok]
    inv = np.where(np.isfinite(O) & (O > 0), 1 / np.where(O > 0, O, 1), 0); p = inv / inv.sum(1, keepdims=True)
    first = np.array([int(c[0]) for c in COMBOS]); second = np.array([int(c[2]) for c in COMBOS])
    tri = m["tri_combo"].astype(str)
    w1, w2 = tri.str[0].astype(int).values, tri.str[2].astype(int).values
    rows = []
    for k, c in enumerate(COMBOS_2F):
        x, y = int(c[0]), int(c[2])
        q = p[:, ((first == x) & (second == y)) | ((first == y) & (second == x))].sum(1)
        o = pd.to_numeric(m[c], errors="coerce").values
        hit = ((w1 == x) & (w2 == y)) | ((w1 == y) & (w2 == x))
        rows.append(pd.DataFrame({"date": m["date"], "jcd": m["jcd"], "rno": m["rno"], "kind": "2連複", "combo": c, "odds": o, "q": q, "hit": hit}))
    for b in range(1, 7):
        q = p[:, first == b].sum(1); o = pd.to_numeric(m[f"tan{b}"], errors="coerce").values
        rows.append(pd.DataFrame({"date": m["date"], "jcd": m["jcd"], "rno": m["rno"], "kind": "単勝", "combo": str(b), "odds": o, "q": q, "hit": w1 == b}))
    d = pd.concat(rows, ignore_index=True)
    d = d[np.isfinite(d["odds"]) & (d["odds"] > 0)].reset_index(drop=True)
    d["ret"] = np.where(d["hit"], d["odds"] * 100, 0.0); d["under"] = (d["q"] * d["odds"] > 1.1).astype(float)
    d["band"] = pd.cut(d["rno"], [0, 4, 8, 12], labels=["1-4R", "5-8R", "9-12R"]).astype(str)
    d["period"] = np.where(d["date"] <= TR[1], "tr", np.where((d["date"] >= EV[0]) & (d["date"] <= EV[1]), "ev", ""))
    d = d[d["period"] != ""]
    n_tr, n_ev = d.loc[d.period == "tr", KEY].drop_duplicates().shape[0], d.loc[d.period == "ev", KEY].drop_duplicates().shape[0]
    md = [f"# IDEA-032 確定時に割安になりやすい「場所」（学習 {TR[0]}〜{TR[1]} {n_tr}レース／評価 {EV[0]}〜{EV[1]} {n_ev}レース）", "",
          "区分ごとに全部を各100円買ったときの回収率（確定オッズで払戻）。学習で 点数300以上・回収率105%以上の区分を選び、評価で判定。", ""]
    base = d.groupby(["kind", "period"]).agg(点数=("ret", "size"), 回収率=("ret", lambda s: s.sum() / (100 * len(s))), 確定時に割安の割合=("under", "mean")).reset_index()
    md += ["## 全体（区分なし）", "", base.to_markdown(index=False, floatfmt=".3f"), ""]
    summ = []
    for name, cols in (("A 組", ["kind", "combo"]), ("B 組×場", ["kind", "combo", "jcd"]), ("C 組×レース番号帯", ["kind", "combo", "band"])):
        g = d.groupby(cols + ["period"]).agg(n=("ret", "size"), rt=("ret", "sum"), under=("under", "mean")).reset_index()
        g["roi"] = g["rt"] / (100 * g["n"])
        w = g.pivot_table(index=cols, columns="period", values=["n", "roi", "under"])
        w.columns = [f"{x}_{y}" for x, y in w.columns]; w = w.dropna(subset=["n_tr", "n_ev"]).reset_index()
        corr_roi = float(np.corrcoef(w["roi_tr"], w["roi_ev"])[0, 1]) if len(w) > 2 else np.nan
        corr_under = float(np.corrcoef(w["under_tr"], w["under_ev"])[0, 1]) if len(w) > 2 else np.nan
        sel = w[(w["n_tr"] >= 300) & (w["roi_tr"] >= 1.05)]
        row = {"区分": name, "区分の数": len(w), "選ばれた区分": len(sel), "回収率の相関（学習↔評価）": corr_roi, "割安の割合の相関（学習↔評価）": corr_under}
        if len(sel):
            e = d[d.period == "ev"].merge(sel[cols], on=cols)
            gd = pd.DataFrame({"d": e["date"], "st": 100.0, "rt": e["ret"]}).groupby("d").sum()
            roi = gd["rt"].sum() / gd["st"].sum(); lo, hi = boot(gd)
            tr_roi = float((sel["roi_tr"] * sel["n_tr"]).sum() / sel["n_tr"].sum())
            row.update({"学習の回収率": tr_roi, "評価の点数": len(e), "評価の回収率": roi, "95%CI下": lo, "95%CI上": hi,
                        "判定": "採用候補" if (len(e) >= 200 and roi >= 1.1 and lo > 1.0) else "不採用"})
            top = sel.sort_values("roi_tr", ascending=False).head(15)
            md += [f"### {name}：選ばれた区分（学習の回収率の高い順、上位15）", "",
                   top.rename(columns={"n_tr": "学習点数", "roi_tr": "学習回収率", "n_ev": "評価点数", "roi_ev": "評価回収率", "under_tr": "学習 割安の割合", "under_ev": "評価 割安の割合"}).to_markdown(index=False, floatfmt=".3f"), ""]
        else:
            row["判定"] = "かたよりなし（選ばれた区分なし）"
        summ.append(row)
    S = pd.DataFrame(summ)
    md[4:4] = ["## 判定", "", S.to_markdown(index=False, floatfmt=".3f"), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
