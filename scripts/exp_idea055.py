"""IDEA-055 女性選手は、特定の条件なら男性に勝ちやすいか（混合戦。docs/ideas.md の登録どおり）
usage: python scripts/exp_idea055.py DATADIR HISTDS SEXCSV --bk BKDIR [BKDIR ...] [--out results/idea055.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ、12年の参考用）
- SEXCSV：登録番号→性別（IDEA-054 と同じ、公式のファン手帳から作ったもの。コミットしない）
- BKDIR：K ファイル（bk/YYYY/kYYMMDD.txt.gz）の置き場所。参考の「1号艇22歳以下の単勝・複勝」の払戻に使う
- 対象：女子が1人以上・男子も1人以上いるレース（混合戦）の女子の艇。条件ごとに 1着率・3着以内率 ÷ 市場の見込み（確定3連単オッズから）
- 探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で確かめる（選び方・判定は IDEA-054 と同じ）
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL, PSEUDO, ratio_ci, boot_roi

WATER = {"01": "淡水", "02": "淡水", "03": "汽水", "04": "海水", "05": "淡水", "06": "汽水", "07": "汽水", "08": "海水", "09": "海水", "10": "淡水",
         "11": "淡水", "12": "淡水", "13": "淡水", "14": "海水", "15": "海水", "16": "海水", "17": "海水", "18": "海水", "19": "海水", "20": "海水",
         "21": "淡水", "22": "汽水", "23": "淡水", "24": "海水"}

def load_entries(paths, sex):
    cols = KEY + ["boat", "toban", "age", "weight", "cls", "motor_2r", "pos_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in paths], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last").sort_values(KEY + ["boat"]).reset_index(drop=True)
    s = e["toban"].map(sex); w = pd.to_numeric(e["weight"], errors="coerce")
    e["fem"] = np.where(s.notna(), s == 2, w < 50)
    e["weight"] = w; e["age"] = pd.to_numeric(e["age"], errors="coerce"); e["motor_2r"] = pd.to_numeric(e["motor_2r"], errors="coerce")
    # 節：場ごとに開催日が1日以上空いたら新しい節
    d = e[["jcd", "date"]].drop_duplicates().sort_values(["jcd", "date"])
    d["dt"] = pd.to_datetime(d["date"]); gap = d.groupby("jcd")["dt"].diff().dt.days.fillna(99)
    d["meet"] = (gap > 1).groupby(d["jcd"]).cumsum()
    e = e.merge(d[["jcd", "date", "meet"]], on=["jcd", "date"], how="left")
    # その節のモーター上位3分の1（節で使われたモーターごとの2連対率で比べる）
    mo = e.groupby(["jcd", "meet", "motor_2r"]).size().reset_index()[["jcd", "meet", "motor_2r"]]
    q = mo.groupby(["jcd", "meet"])["motor_2r"].quantile(2 / 3).rename("q").reset_index()
    e = e.merge(q, on=["jcd", "meet"], how="left"); e["motor_top"] = np.where(e["motor_2r"] >= e["q"], "上位3分の1", "それ以外")
    # 混合戦・男子の平均体重
    g = e.groupby(KEY).agg(nf=("fem", "sum"), n=("fem", "size")).reset_index()
    mw = e[~e["fem"]].groupby(KEY)["weight"].mean().rename("male_w").reset_index()
    e = e.merge(g, on=KEY).merge(mw, on=KEY, how="left")
    e["mixed"] = (e["nf"] >= 1) & (e["nf"] < e["n"])
    return e

def conditions(f, races, gap_cut):
    """女子の艇ごとの条件（列名 → 区分のラベル）"""
    x = f.merge(races[KEY + ["wind", "wave"]], on=KEY, how="left")
    c = pd.DataFrame(index=f.index)
    wind = pd.to_numeric(x["wind"], errors="coerce").values; wave = pd.to_numeric(x["wave"], errors="coerce").values
    c["全体"] = "混合戦の女子"
    c["風"] = np.select([wind <= 2, wind <= 4, wind >= 5], ["〜2m", "3〜4m", "5m〜"], "nan")
    c["波"] = np.select([wave <= 2, wave <= 4, wave >= 5], ["〜2cm", "3〜4cm", "5cm〜"], "nan")
    c["水質"] = f["jcd"].map(WATER).values
    c["艇番"] = np.select([f["boat"] == 1, f["boat"] <= 3], ["1", "2〜3"], "4〜6")
    c["モーター"] = f["motor_top"].values
    gap = (f["male_w"] - f["weight"]).values
    c["体重差"] = np.where(np.isfinite(gap), np.select([gap <= gap_cut[0], gap <= gap_cut[1]], ["小", "中"], "大"), "nan")
    c["級別"] = np.where(f["cls"].astype(str).str.startswith("A"), "A", np.where(f["cls"].astype(str).str.startswith("B"), "B", "nan"))
    c["艇番×風"] = c["艇番"] + "・" + c["風"]
    c["艇番×波"] = c["艇番"] + "・" + c["波"]
    c["艇番×体重差"] = c["艇番"] + "・" + c["体重差"]
    return c

def k_tanfuku(bkdirs, start, end):
    """K ファイルの単勝・複勝の払戻（1号艇の分だけ）：date, jcd, rno, tan1, fuku1"""
    rows = []
    for bd in bkdirs:
        for fn in sorted(glob.glob(f"{bd}/*/k*.txt.gz")):
            date = "20" + os.path.basename(fn)[1:7]
            if not (start <= date <= end):
                continue
            txt = gzip.open(fn, "rt", encoding="utf-8", errors="replace").read()
            for jcd, blk in BP._blocks(txt, "K"):
                rno = None
                for line in blk.splitlines():
                    m = BP.K_RACE.match(BP.z2h(line))
                    if m:
                        rno = int(m.group(1)); tan = 0; fuku = 0; continue
                    if rno is None:
                        continue
                    z = BP.z2h(line)
                    mm = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", z)
                    if mm:
                        tan = int(mm.group(2)) if mm.group(1) == "1" else 0
                        continue
                    mm = re.match(r"^\s+複勝\s+(.*)$", z)
                    if mm:
                        pr = re.findall(r"(\d)\s+(\d+)", mm.group(1))
                        fuku = sum(int(p) for b, p in pr if b == "1")
                        rows.append((date, jcd, rno, tan, fuku)); rno = None
    d = pd.DataFrame(rows, columns=KEY + ["tan1", "fuku1"])
    return d.drop_duplicates(KEY, keep="last")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("sexcsv")
    ap.add_argument("--bk", nargs="+", default=[]); ap.add_argument("--out", default="results/idea055.md"); a = ap.parse_args()
    sex = pd.read_csv(a.sexcsv).set_index("toban")["sex"]
    e = load_entries([a.histds, a.datadir], sex)
    races = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["wind", "wave", "tri_combo", "tri_pay"])) for p in (a.histds, a.datadir)])
    races = races.drop_duplicates(KEY, keep="last")
    f = e[e["fem"] & e["mixed"]].copy()
    fx = f[(f["date"] >= EXP[0]) & (f["date"] <= EXP[1])]
    gap_cut = np.quantile((fx["male_w"] - fx["weight"]).dropna(), [1 / 3, 2 / 3])
    C = conditions(f, races, gap_cut); ccols = list(C.columns)
    f = f.join(C)
    # --- 市場（2023-10 以降、確定3連単オッズ）
    r = races[(races["date"] >= EXP[0]) & (races["date"] <= EVL[1])][KEY + ["tri_combo"]].dropna()
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    I = np.array([int(c[0]) for c in COMBOS]); J = np.array([int(c[2]) for c in COMBOS]); K = np.array([int(c[4]) for c in COMBOS])
    pw = np.stack([P[:, I == b].sum(1) for b in range(1, 7)], 1)
    pt3 = np.stack([P[:, (I == b) | (J == b) | (K == b)].sum(1) for b in range(1, 7)], 1)
    aw = np.stack([(I[y] == b) for b in range(1, 7)], 1).astype(float)
    at3 = np.stack([(I[y] == b) | (J[y] == b) | (K[y] == b) for b in range(1, 7)], 1).astype(float)
    days = m["date"].values; per = np.where(days <= EXP[1], "探索", "評価")
    em = m[KEY].reset_index().merge(f[KEY + ["boat"] + ccols], on=KEY, how="inner")
    ri = em["index"].values; bi_ = em["boat"].values.astype(int) - 1
    em["pw"] = pw[ri, bi_]; em["pt3"] = pt3[ri, bi_]; em["aw"] = aw[ri, bi_]; em["at3"] = at3[ri, bi_]; em["per"] = per[ri]
    rows = []
    for col in ccols:
        for lab, g in em.groupby(col):
            if "nan" in str(lab):
                continue
            row = {"条件": col, "区分": lab}
            for pp in ("探索", "評価"):
                gg = g[g["per"] == pp]; row[f"{pp} 艇数"] = len(gg)
                for t, act, ex_ in (("1着", "aw", "pw"), ("3着内", "at3", "pt3")):
                    rr, lo, hi = ratio_ci(gg["date"].values, gg[act].values, gg[ex_].values) if len(gg) >= 30 else (np.nan,) * 3
                    row[f"{pp} {t}比"] = rr; row[f"{pp} {t}CI下"] = lo; row[f"{pp} {t}CI上"] = hi
                    row[f"{pp} {t}実際"] = gg[act].sum(); row[f"{pp} {t}見込み"] = gg[ex_].sum()
            rows.append(row)
    R = pd.DataFrame(rows)
    sel = []
    for t in ("1着", "3着内"):
        c = (R["探索 艇数"] >= 300) & (((R[f"探索 {t}比"] >= 1.10) & (R[f"探索 {t}CI下"] > 1)) | ((R[f"探索 {t}比"] <= 0.90) & (R[f"探索 {t}CI上"] < 1)))
        s = R[c].copy(); s["種類"] = t; sel.append(s)
    S = pd.concat(sel, ignore_index=True)
    def survived(rw):
        t = rw["種類"]
        return rw[f"評価 {t}CI下"] > 1 if rw[f"探索 {t}比"] > 1 else rw[f"評価 {t}CI上"] < 1
    S["評価で残った"] = S.apply(survived, axis=1) if len(S) else []
    # --- 残った区分（市場より強い向き）の買い方：その艇の1着（3着内）を含む組で π×オッズ > 1.0
    pi0 = P ** 1.06; pi0 /= pi0.sum(1, keepdims=True)
    bets = []
    for _, rw in S[S["評価で残った"]].iterrows() if len(S) else []:
        t = rw["種類"]; up = rw[f"探索 {t}比"] > 1
        rr = (rw[f"探索 {t}実際"] + PSEUDO) / (rw[f"探索 {t}見込み"] + PSEUDO)
        tgt = em[(em[rw["条件"]] == rw["区分"]) & (em["per"] == "評価")]
        if not up or len(tgt) == 0:
            bets.append({"条件": rw["条件"], "区分": rw["区分"], "種類": t, "補正": rr, "点数": 0, "回収率": np.nan, "CI下": np.nan, "CI上": np.nan}); continue
        st = np.zeros(len(tgt)); rt = np.zeros(len(tgt)); nb = 0
        for k, (idx, b) in enumerate(zip(tgt["index"].values, tgt["boat"].values.astype(int))):
            mask = (I == b) if t == "1着" else ((I == b) | (J == b) | (K == b))
            pi = pi0[idx].copy(); pi[mask] *= rr; pi /= pi.sum()
            buy = v[idx] & mask & (pi * np.where(v[idx], O[idx], 0) > 1.0)
            st[k] = buy.sum() * 100.0; rt[k] = O[idx, y[idx]] * 100 if buy[y[idx]] else 0.0; nb += int(buy.sum())
        roi, lo, hi = boot_roi(days[tgt["index"].values], st, rt)
        bets.append({"条件": rw["条件"], "区分": rw["区分"], "種類": t, "補正": rr, "点数": nb, "回収率": roi, "CI下": lo, "CI上": hi, "艇数": len(tgt)})
    B = pd.DataFrame(bets)
    win = B[(B["点数"] >= 300) & (B["回収率"] >= 1.1) & (B["CI下"] > 1.0)] if len(B) else B
    judge = "採用候補（前向きの検証へ）" if len(win) else "不採用"
    # --- 12年の参考：その女子の艇が1着の組を全部（20点）の年別回収率。基準＝同じ年・同じ艇番の全艇
    rh = races.dropna(subset=["tri_combo"]); rh = rh[(rh["date"] >= "20140101") & (rh["date"] <= EVL[1])]
    bad = e[e["pos_raw"].astype(str).str.match(r"^[KFLS]")][KEY].drop_duplicates()
    six = e.groupby(KEY).size(); six = six[six == 6].reset_index()[KEY]
    rh = rh.merge(six, on=KEY).merge(bad.assign(x=1), on=KEY, how="left"); rh = rh[rh["x"].isna()].drop(columns="x")
    rh["w"] = rh["tri_combo"].astype(str).str[0]; rh["tri_pay"] = pd.to_numeric(rh["tri_pay"], errors="coerce")
    ea = rh.merge(e[KEY + ["boat"]], on=KEY); ea["year"] = ea["date"].str[:4]
    ea["ret"] = np.where(ea["w"] == ea["boat"].astype(str), ea["tri_pay"], 0)
    base = ea.groupby(["year", "boat"])["ret"].mean() / 2000
    eh = rh.merge(f[KEY + ["boat"] + ccols], on=KEY); eh["year"] = eh["date"].str[:4]
    eh["ret"] = np.where(eh["w"] == eh["boat"].astype(str), eh["tri_pay"], 0)
    eh["base"] = [base.get((yy, bb), np.nan) for yy, bb in zip(eh["year"], eh["boat"])]
    hist = []
    cells = S[S["種類"] == "1着"][["条件", "区分"]].drop_duplicates() if len(S) else pd.DataFrame(columns=["条件", "区分"])
    cells = pd.concat([pd.DataFrame([{"条件": "全体", "区分": "混合戦の女子"}]), cells]).drop_duplicates()
    for _, rw in cells.iterrows():
        g = eh[eh[rw["条件"]] == rw["区分"]].groupby("year").agg(ret=("ret", "sum"), n=("ret", "size"), base=("base", "mean"))
        g["roi"] = g["ret"] / (g["n"] * 2000)
        hist.append({"条件": rw["条件"], "区分": rw["区分"], "年数": len(g), "基準を上回った年": int((g["roi"] > g["base"]).sum()),
                     "回収率の平均": g["roi"].mean(), "最高の年": g["roi"].max(), "100%超の年": int((g["roi"] > 1).sum()),
                     **{yy: g["roi"].get(yy, np.nan) for yy in ("2014", "2017", "2020", "2023", "2025", "2026")}})
    H = pd.DataFrame(hist)
    # --- 参考：1号艇が22歳以下のレースの 単勝1・複勝1（K の実額、各100円）の年別回収率
    tf = k_tanfuku(a.bk, "20140101", EVL[1]) if a.bk else pd.DataFrame(columns=KEY + ["tan1", "fuku1"])
    b1 = e[e["boat"] == 1][KEY + ["age"]]
    t1 = tf.merge(b1, on=KEY).merge(six, on=KEY).merge(bad.assign(x=1), on=KEY, how="left"); t1 = t1[t1["x"].isna()]
    t1["year"] = t1["date"].str[:4]; t1["young"] = t1["age"] <= 22
    yr = t1.groupby(["year", "young"]).agg(n=("tan1", "size"), tan=("tan1", "sum"), fuku=("fuku1", "sum")).reset_index()
    yr["単勝1"] = yr["tan"] / (yr["n"] * 100); yr["複勝1"] = yr["fuku"] / (yr["n"] * 100)
    Y = yr.pivot(index="year", columns="young", values=["n", "単勝1", "複勝1"])
    Y.columns = [f"{c}（{'22歳以下' if yg else '23歳以上'}）" for c, yg in Y.columns]
    Y = Y.reset_index()
    yt = t1[t1["young"]]; tot_t = yt["tan1"].sum() / (len(yt) * 100) if len(yt) else np.nan; tot_f = yt["fuku1"].sum() / (len(yt) * 100) if len(yt) else np.nan
    # --- 出力
    def pick(rw, pp, k):
        return rw[f"{pp} {rw['種類']}{k}"]
    S_out = S.copy()
    if len(S):
        S_out["探索 比"] = [pick(rw, "探索", "比") for _, rw in S.iterrows()]
        S_out["探索 CI"] = ["%.2f〜%.2f" % (pick(rw, "探索", "CI下"), pick(rw, "探索", "CI上")) for _, rw in S.iterrows()]
        S_out["評価 比"] = [pick(rw, "評価", "比") for _, rw in S.iterrows()]
        S_out["評価 CI"] = ["%.2f〜%.2f" % (pick(rw, "評価", "CI下"), pick(rw, "評価", "CI上")) for _, rw in S.iterrows()]
        S_out = S_out[["条件", "区分", "種類", "探索 艇数", "評価 艇数", "探索 比", "探索 CI", "評価 比", "評価 CI", "評価で残った"]].sort_values(["評価で残った", "評価 比"], ascending=[False, False])
    full = R[["条件", "区分", "探索 艇数", "探索 1着比", "探索 3着内比", "評価 艇数", "評価 1着比", "評価 3着内比"]]
    allrow = R[R["条件"] == "全体"].iloc[0]
    nsurv = int(S["評価で残った"].sum()) if len(S) else 0
    md = [f"# IDEA-055 女性選手×条件（混合戦、探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}）", "",
          f"**判定：{judge}**", "",
          "## まとめ", "",
          f"- 性別は公式の選手データ（ファン手帳）から（IDEA-054 と同じ）。混合戦の女子の艇：探索 {int(allrow['探索 艇数'])}・評価 {int(allrow['評価 艇数'])}。",
          f"- 混合戦の女子全体の 1着 実際÷見込み：探索 {allrow['探索 1着比']:.3f}・評価 {allrow['評価 1着比']:.3f}（3着内 探索 {allrow['探索 3着内比']:.3f}・評価 {allrow['評価 3着内比']:.3f}）。",
          f"- 探索で選ばれた区分 {len(S)} 個のうち、評価でも同じ向きに有意に残ったのは {nsurv} 個。",
          f"- 残った区分（市場より強い向き）で π×オッズ > 1.0 を買った場合：{'、'.join(f'{x.条件}={x.区分}（{x.種類}）{x.点数}点 {x.回収率:.0%}' for x in B.itertuples() if x.点数 > 0) if len(B) and (B['点数'] > 0).any() else 'なし'}",
          f"- 参考（判定に使わない）：1号艇が22歳以下のレースで 単勝1・複勝1 を各100円（K の実額、2014〜2026）：通算 単勝 {tot_t:.1%}・複勝 {tot_f:.1%}（{len(yt)}レース）。",
          "", "## 探索で選ばれた区分と評価", "", S_out.to_markdown(index=False, floatfmt=".3f") if len(S) else "なし", "",
          "## 残った区分の買い方（評価、π×オッズ > 1.0、確定オッズで払戻）", "", B.to_markdown(index=False, floatfmt=".3f") if len(B) else "なし", "",
          "## 12年の参考（2014〜2026、その女子の艇が1着の組を全部20点。基準＝同じ年・同じ艇番の全艇）", "", H.to_markdown(index=False, floatfmt=".3f"), "",
          "## 参考：1号艇が22歳以下のレースの 単勝1・複勝1 の年別回収率（K の実額）", "", Y.to_markdown(index=False, floatfmt=".3f"), "",
          "## 全区分（1着比・3着内比）", "", full.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:14]))

if __name__ == "__main__":
    main()
