"""IDEA-061 勝負駆け：予選最終日に準優勝戦進出ぎりぎりの選手は、市場の見込みより走るか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea061.py DATADIR HISTDS STORE [--out results/idea061.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ、12年の参考用）、STORE：grades.csv と K ファイル（bk/）
- 節 = 同じ場の連続開催日（間が1日以上空いたら別の節）。準優の日 = 節の中で最初にレース名に「準優」が出る日。予選最終日 = その前の開催日
- 予選最終日の朝の得点率 = それより前の節内のレースの得点（1着10・2着8・3着6・4着4・5着2・6着1、F・L・S・欠場は0）÷ 出走数（欠場 K を除く）。ドリーム戦は除く
- 準優の枠 = その節の準優のレースに出た選手の数。順位（得点率の高い順）で 当確（枠−6以内）／勝負駆け（枠−5〜枠+6）／圏外（枠+7〜）
- 区分 × 艇番（と グレード帯 SG/G1/G2 か 一般/G3）ごとに 1着率・3着以内率 ÷ 市場の見込み。探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で判定
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import grade_map
from exp_idea054 import KEY, EXP, EVL, PSEUDO, ratio_ci, boot_roi

PTS = {"01": 10, "02": 8, "03": 6, "04": 4, "05": 2, "06": 1}
CATS = ["当確", "勝負駆け", "圏外"]

def load(d):
    e = pd.read_parquet(f"{d}/entries.parquet", columns=KEY + ["boat", "toban", "race_name", "pos_raw"])
    return norm(e)

def categorize(e):
    """予選最終日の出走（date, jcd, rno, boat）ごとに 区分・順位・枠"""
    e = e.copy(); e["race_name"] = e["race_name"].astype(str)
    days = e[["date", "jcd"]].drop_duplicates().copy(); days["dt"] = pd.to_datetime(days["date"])
    days = days.sort_values(["jcd", "dt"])
    gap = days.groupby("jcd")["dt"].diff().dt.days.fillna(99) > 1
    days["meet"] = days["jcd"] + "_" + gap.groupby(days["jcd"]).cumsum().astype(str)
    e = e.merge(days[["date", "jcd", "meet"]], on=["date", "jcd"])
    semi = e[e["race_name"].str.contains("準優")]
    first_semi = semi.groupby("meet")["date"].min().rename("semi_day")
    seats = semi.groupby("meet")["toban"].nunique().rename("seats")
    dl = days.merge(first_semi, on="meet")
    dl = dl[dl["date"] < dl["semi_day"]].groupby("meet")["date"].max().rename("last_day")
    info = pd.concat([first_semi, seats, dl], axis=1).dropna()
    e = e.merge(info, left_on="meet", right_index=True)
    before = e[(e["date"] < e["last_day"]) & ~e["race_name"].str.contains("ドリーム") & ~e["pos_raw"].astype(str).str.startswith("K")]
    pts = before.assign(p=before["pos_raw"].map(PTS).fillna(0)).groupby(["meet", "toban"]).agg(pt=("p", "sum"), n=("p", "size")).reset_index()
    pts["rate"] = pts["pt"] / pts["n"]
    pts["rank"] = pts.groupby("meet")["rate"].rank(ascending=False, method="min")
    last = e[(e["date"] == e["last_day"]) & ~e["race_name"].str.contains("準優|優勝|ドリーム")]
    last = last.merge(pts[["meet", "toban", "rank", "rate", "n"]], on=["meet", "toban"])
    s = last["seats"]
    last["cat"] = np.where(last["rank"] <= s - 6, "当確", np.where(last["rank"] <= s + 6, "勝負駆け", "圏外"))
    return last

def market(datadir):
    r = norm(pd.read_parquet(f"{datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    m = r.merge(norm(pd.read_parquet(f"{datadir}/odds.parquet")), on=KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    return m, O, v, P, y

def k_tan(bkdir, start, end):
    """K ファイルの単勝の払戻：date, jcd, rno, tan_boat, tan_pay"""
    rows = []
    for fn in sorted(glob.glob(f"{bkdir}/*/k*.txt.gz")):
        date = "20" + os.path.basename(fn)[1:7]
        if not (start <= date <= end):
            continue
        txt = gzip.open(fn, "rt", encoding="utf-8", errors="replace").read()
        for jcd, blk in BP._blocks(txt, "K"):
            rno = None
            for line in blk.splitlines():
                z = BP.z2h(line); mm = BP.K_RACE.match(z)
                if mm:
                    rno = int(mm.group(1)); continue
                if rno is None:
                    continue
                mm = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", z)
                if mm:
                    rows.append((date, jcd, rno, int(mm.group(1)), int(mm.group(2)))); rno = None
    return pd.DataFrame(rows, columns=KEY + ["tan_boat", "tan_pay"]).drop_duplicates(KEY, keep="last")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("store")
    ap.add_argument("--out", default="results/idea061.md"); a = ap.parse_args()
    e = pd.concat([load(a.histds), load(a.datadir)], ignore_index=True).drop_duplicates(KEY + ["boat"], keep="last")
    last = categorize(e)
    print("予選最終日の出走", len(last), last["cat"].value_counts().to_dict(), flush=True)
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
    L = last.merge(m[KEY + ["ridx"]], on=KEY)
    gm = grade_map(a.store, L); L["gg"] = np.where(pd.Series(gm).isin(["SG", "G1", "G2"]).values, "SG/G1/G2", "一般/G3")
    bi = L["boat"].astype(int).values - 1; ri = L["ridx"].values
    L["pw"] = pw[ri, bi]; L["p3"] = p3[ri, bi]; L["won"] = won[ri, bi]; L["top"] = top[ri, bi]
    L["per"] = np.where((L["date"] >= EXP[0]) & (L["date"] <= EXP[1]), "探索", np.where((L["date"] >= EVL[0]) & (L["date"] <= EVL[1]), "評価", ""))
    cells = []
    for cat in CATS:
        for b in range(1, 7):
            cells.append((cat, b, "全"))
            for g in ("SG/G1/G2", "一般/G3"):
                cells.append((cat, b, g))
    rows = []
    for cat, b, g in cells:
        s0 = (L["cat"] == cat) & (L["boat"] == b) & ((L["gg"] == g) if g != "全" else True)
        row = {"区分": cat, "艇番": b, "グレード": g}
        for per in ("探索", "評価"):
            s = s0 & (L["per"] == per); d = L[s]
            row[f"{per} 艇数"] = int(s.sum())
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
                sel.append({"区分": r_["区分"], "艇番": r_["艇番"], "グレード": r_["グレード"], "指標": k, "向き": "強い" if up else "弱い",
                            "探索 比": r_[f"探索 {k}比"], "評価 比": r_[f"評価 {k}比"], "評価 CI": r_[f"評価 {k}CI"], "評価 艇数": r_["評価 艇数"], "評価で残る": "はい" if held else "いいえ"})
    S = pd.DataFrame(sel)
    print("選ばれた区分", len(S), flush=True)
    # 買い方（評価）
    tan = k_tan(f"{a.store}/bk", EVL[0], EVL[1])
    bets = []; tri_rows = []
    if len(S):
        Lv = L[L["per"] == "評価"].merge(tan, on=KEY, how="left")
        for _, s_ in S.iterrows():
            # 探索の比を縮める（+PSEUDO）
            d0 = L[(L["per"] == "探索") & (L["cat"] == s_["区分"]) & (L["boat"] == s_["艇番"]) & ((L["gg"] == s_["グレード"]) if s_["グレード"] != "全" else True)]
            act, exp = ("won", "pw") if s_["指標"] == "1着" else ("top", "p3")
            ratio = (d0[act].sum() + PSEUDO) / (d0[exp].sum() + PSEUDO)
            dv = Lv[(Lv["cat"] == s_["区分"]) & (Lv["boat"] == s_["艇番"]) & ((Lv["gg"] == s_["グレード"]) if s_["グレード"] != "全" else True)]
            st = np.full(len(dv), 100.0); rt = np.where(dv["tan_boat"].values == s_["艇番"], dv["tan_pay"].fillna(0).values, 0.0)
            rt = np.where(dv["tan_boat"].isna().values, 0.0, rt)
            roi, lo, hi = boot_roi(dv["date"].values, st, rt)
            # 3連単：補正した確率で π×オッズ > 1.0・300倍以下
            b = int(s_["艇番"]); mask_c = (first == b) if s_["指標"] == "1着" else np.array([b in s for s in sets])
            ri = dv["ridx"].values; Pb = P[ri] ** 1.06; Pb /= Pb.sum(1, keepdims=True)
            Pc = Pb * np.where(mask_c[None, :], ratio, 1.0); Pc /= Pc.sum(1, keepdims=True)
            Ov = np.where(v[ri], O[ri], 0); buy = v[ri] & (Ov <= 300) & (Pc * Ov > 1.0)
            hit = np.zeros_like(buy); hit[np.arange(len(ri)), y[ri]] = True
            st3 = buy.sum(1) * 100.0; rt3 = np.where(buy & hit, Ov * 100, 0).sum(1)
            r3, l3, h3 = boot_roi(dv["date"].values, st3, rt3)
            bets.append({"区分": s_["区分"], "艇番": b, "グレード": s_["グレード"], "指標": s_["指標"], "向き": s_["向き"], "縮めた探索の比": ratio,
                         "単勝 点数": len(dv), "単勝 回収率": roi, "単勝 CI": f"{lo:.2f}〜{hi:.2f}" if np.isfinite(lo) else "",
                         "3連単 点数": int(buy.sum()), "3連単 回収率": r3, "3連単 CI": f"{l3:.2f}〜{h3:.2f}" if np.isfinite(l3) else "",
                         "_tlo": lo, "_tn": len(dv), "_troi": roi, "_3lo": l3, "_3n": int(buy.sum()), "_3roi": r3})
            tri_rows.append(pd.DataFrame({"d": dv["date"].values, "st": st3, "rt": rt3, "ts": np.where(s_["向き"] == "強い", st, 0), "tr": np.where(s_["向き"] == "強い", rt, 0)}))
    B = pd.DataFrame(bets)
    passed = []
    for _, b_ in B.iterrows():
        if b_["向き"] == "強い" and b_["_tn"] >= 300 and b_["_troi"] >= 1.1 and b_["_tlo"] > 1.0:
            passed.append(f"{b_['区分']}・{b_['艇番']}号艇・{b_['グレード']}（単勝）")
        if b_["_3n"] >= 300 and b_["_3roi"] >= 1.1 and b_["_3lo"] > 1.0:
            passed.append(f"{b_['区分']}・{b_['艇番']}号艇・{b_['グレード']}（3連単）")
    pooled = ""
    if tri_rows:
        T = pd.concat(tri_rows); r3, l3, h3 = boot_roi(T["d"].values, T["st"].values, T["rt"].values)
        rt_, lt_, ht_ = boot_roi(T["d"].values, T["ts"].values, T["tr"].values)
        pooled = (f"選ばれた区分をまとめて（評価）：3連単 {int(T['st'].sum() / 100)}点・回収率 {r3:.3f}（95%CI {l3:.3f}〜{h3:.3f}）、"
                  f"単勝（強い向きの区分のみ）{int(T['ts'].sum() / 100)}点・回収率 {rt_:.3f}（95%CI {lt_:.3f}〜{ht_:.3f}）")
    judge = "採用候補（前向きの検証へ）：" + "、".join(passed) if passed else "不採用"
    # 12年の参考（オッズなし）：区分×艇番の1着率・3着以内率 ÷ 同じ年・同じ艇番の全出走の平均
    e2 = e[e["pos_raw"].astype(str).str.match(r"^0[1-6]$")].copy()
    e2["year"] = e2["date"].str[:4]; e2["won"] = (e2["pos_raw"] == "01").astype(float); e2["top"] = e2["pos_raw"].isin(["01", "02", "03"]).astype(float)
    base = e2.groupby(["year", "boat"])[["won", "top"]].mean()
    l2 = last[last["pos_raw"].astype(str).str.match(r"^0[1-6]$")].copy()
    l2["year"] = l2["date"].str[:4]; l2["won"] = (l2["pos_raw"] == "01").astype(float); l2["top"] = l2["pos_raw"].isin(["01", "02", "03"]).astype(float)
    g = l2.groupby(["cat", "boat", "year"])[["won", "top"]].agg(["mean", "size"])
    hist = []
    for (cat, b, yr), r_ in g.iterrows():
        bw, bt = base.loc[(yr, b)]
        hist.append({"区分": cat, "艇番": b, "年": yr, "出走": int(r_[("won", "size")]), "1着率÷全体": r_[("won", "mean")] / bw, "3着内率÷全体": r_[("top", "mean")] / bt})
    H = pd.DataFrame(hist)
    Hs = H.groupby(["区分", "艇番"]).agg(年数=("年", "size"), 出走=("出走", "sum"), 一着_平均=("1着率÷全体", "mean"),
                                       一着_上回った年=("1着率÷全体", lambda x: int((x > 1).sum())), 三着内_平均=("3着内率÷全体", "mean"),
                                       三着内_上回った年=("3着内率÷全体", lambda x: int((x > 1).sum()))).reset_index()
    Hs.columns = ["区分", "艇番", "年数", "出走", "1着率÷全体（年平均）", "1着が全体を上回った年", "3着内率÷全体（年平均）", "3着内が全体を上回った年"]
    Rshow = R[[c for c in R.columns if not c.startswith("_")]]
    nd = last[last["per"] != ""] if "per" in last else last
    md = [f"# IDEA-061 勝負駆け（予選最終日、探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}）", "",
          f"**判定：{judge}**", "",
          "## 定義", "",
          "- 節 = 同じ場の連続開催日。準優の日 = 節の中で最初にレース名に「準優」がある日。予選最終日 = その前の開催日（準優・優勝・ドリームの名前のレースは除く）。準優がない節は対象外。",
          "- 得点率 = 予選最終日より前の節内の得点（1着10・2着8・3着6・4着4・5着2・6着1、F・L・S は0、欠場 K は出走に数えない、ドリーム戦は除く）÷ 出走数。事故点の減点・最低出走回数・同点の順位決めは入れていない（近似）。",
          "- 枠 = その節の準優に出た選手の数（多くは18）。得点率の順位で 当確（順位 ≤ 枠−6）／勝負駆け（枠−5〜枠+6）／圏外（> 枠+6）。",
          f"- 予選最終日の出走：{len(last):,}（2014〜2026）、区分 {last['cat'].value_counts().to_dict()}。グレードは store/grades.csv（2023年以降）で SG/G1/G2 と 一般/G3 に分けた。", "",
          "## 選ばれた区分（探索で 300艇以上・比 1.08 以上か 0.92 以下・CI が 1 をまたがない）と評価", "",
          (S.to_markdown(index=False, floatfmt=".3f") if len(S) else "なし"), "",
          "## 買い方（評価 2026-01〜08、単勝は K の実額、3連単は確定オッズ）", "",
          (B[[c for c in B.columns if not c.startswith("_")]].to_markdown(index=False, floatfmt=".3f") if len(B) else "なし"), "", pooled, "",
          "## 区分 × 艇番 × グレードの全表（1着・3着以内の 実際÷市場の見込み）", "", Rshow.to_markdown(index=False, floatfmt=".3f"), "",
          "## 12年の参考（2014〜2026、オッズなし。同じ年・同じ艇番の全出走の平均との比）", "", Hs.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:12])); print(S.to_string() if len(S) else ""); print(pooled)

if __name__ == "__main__":
    main()
