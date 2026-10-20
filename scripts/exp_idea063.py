"""IDEA-063 休み明けの選手は、市場の見込みより走らないか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea063.py DATADIR HISTDS BKDIR [--out results/idea063.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ、空白日数の計算と12年の参考用）、BKDIR：K ファイル（単勝の払戻）
- 空白日数：その選手の前回の出走日（欠場 K0/K1 を除く、レース当日より前）からの日数。21日以上空いて戻った日から14日以内の出走を「休み明けの節」とし、
  戻った日の空白日数で 21〜45／46〜90／91日〜 に分ける。それ以外は 通常（〜20日）。
  F明け：戻る前の最後の出走日までの60日以内に F（pos_raw が F）があった。何走目：戻った日からその出走の前までの出走数（1〜2走目／3走目以降）。
- 区分 × 艇番（1／2〜3／4〜6）ごとに、1着率・3着以内率 ÷ 市場の見込み（確定3連単オッズ）。探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で確かめる
- このファイルの core() は IDEA-064 からも使う
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL, PSEUDO, ratio_ci, boot_roi

GRP = {1: "1", 2: "2〜3", 3: "2〜3", 4: "4〜6", 5: "4〜6", 6: "4〜6"}

def load_entries(paths, cols):
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=KEY + ["boat"] + cols)) for p in paths], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last").sort_values(KEY + ["boat"]).reset_index(drop=True)
    e["grp"] = e["boat"].astype(int).map(GRP)
    return e

def k_tan(bkdir, start, end):
    """K ファイルの単勝：date, jcd, rno, tan_boat（1着の艇）, tan_pay（100円あたり）"""
    rows = []
    for fn in sorted(glob.glob(f"{bkdir}/*/k*.txt.gz")):
        date = "20" + os.path.basename(fn)[1:7]
        if not (start <= date <= end):
            continue
        txt = gzip.open(fn, "rt", encoding="utf-8", errors="replace").read()
        for jcd, blk in BP._blocks(txt, "K"):
            rno = None
            for line in blk.splitlines():
                z = BP.z2h(line); mk = BP.K_RACE.match(z)
                if mk:
                    rno = int(mk.group(1)); continue
                if rno is None:
                    continue
                mm = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", z)
                if mm:
                    rows.append((date, jcd, rno, int(mm.group(1)), int(mm.group(2)))); rno = None
    return pd.DataFrame(rows, columns=KEY + ["tan_boat", "tan_pay"]).drop_duplicates(KEY, keep="last")

def market(datadir):
    r = norm(pd.read_parquet(f"{datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= EXP[0]) & (r["date"] <= EVL[1])]
    m = r.merge(norm(pd.read_parquet(f"{datadir}/odds.parquet")), on=KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    return m, O, v, y, P

def core(e, ccols, datadir, bkdir, title, extra_md=()):
    """区分の列 ccols（艇ごと、"nan" は対象外）について、選び方・評価・買い方・12年の参考を行い、markdown の行を返す"""
    m, O, v, y, P = market(datadir); n = len(m)
    I = np.array([int(c[0]) for c in COMBOS]); J = np.array([int(c[2]) for c in COMBOS]); K = np.array([int(c[4]) for c in COMBOS])
    pw = np.stack([P[:, I == b].sum(1) for b in range(1, 7)], 1)
    pt3 = np.stack([P[:, (I == b) | (J == b) | (K == b)].sum(1) for b in range(1, 7)], 1)
    aw = np.stack([(I[y] == b) for b in range(1, 7)], 1).astype(float)
    at3 = np.stack([(I[y] == b) | (J[y] == b) | (K[y] == b) for b in range(1, 7)], 1).astype(float)
    days = m["date"].values; per = np.where(days <= EXP[1], "探索", "評価")
    em = m[KEY].reset_index().merge(e[KEY + ["boat", "grp", "st"] + ccols], on=KEY, how="inner")
    ri = em["index"].values; bi_ = em["boat"].values.astype(int) - 1
    em["pw"] = pw[ri, bi_]; em["pt3"] = pt3[ri, bi_]; em["aw"] = aw[ri, bi_]; em["at3"] = at3[ri, bi_]; em["per"] = per[ri]
    rows = []
    for col in ccols:
        for (lab, grp), g in em.groupby([col, "grp"]):
            if "nan" in str(lab):
                continue
            row = {"区分の列": col, "区分": lab, "艇番": grp}
            for pp in ("探索", "評価"):
                gg = g[g["per"] == pp]; row[f"{pp} 艇数"] = len(gg)
                row[f"{pp} ST平均"] = pd.to_numeric(gg["st"], errors="coerce").mean()
                for t, act, ex_ in (("1着", "aw", "pw"), ("3着内", "at3", "pt3")):
                    rr, lo, hi = ratio_ci(gg["date"].values, gg[act].values, gg[ex_].values) if len(gg) >= 30 else (np.nan,) * 3
                    row[f"{pp} {t}比"] = rr; row[f"{pp} {t}CI下"] = lo; row[f"{pp} {t}CI上"] = hi
                    row[f"{pp} {t}実際"] = gg[act].sum(); row[f"{pp} {t}見込み"] = gg[ex_].sum()
            rows.append(row)
    R = pd.DataFrame(rows)
    sel = []
    for t in ("1着", "3着内"):
        c = (R["探索 艇数"] >= 300) & (((R[f"探索 {t}比"] >= 1.08) & (R[f"探索 {t}CI下"] > 1)) | ((R[f"探索 {t}比"] <= 0.92) & (R[f"探索 {t}CI上"] < 1)))
        s = R[c].copy(); s["種類"] = t; sel.append(s)
    S = pd.concat(sel, ignore_index=True)
    def survived(rw):
        t = rw["種類"]
        return rw[f"評価 {t}CI下"] > 1 if rw[f"探索 {t}比"] > 1 else rw[f"評価 {t}CI上"] < 1
    S["評価で残った"] = S.apply(survived, axis=1) if len(S) else []
    # --- 残った区分の買い方（評価）。強い向き：その艇の単勝（K の実額）と、その艇の1着（3着内）を含む3連単で π×オッズ > 1.0。
    #     弱い向き：その艇を比で割り引いた π で、その艇の1着（3着内）を含まない3連単で π×オッズ > 1.0・300倍以下。
    tan = k_tan(bkdir, EVL[0], EVL[1]).set_index(KEY)
    pi0 = P ** 1.06; pi0 /= pi0.sum(1, keepdims=True)
    bets = []
    for _, rw in (S[S["評価で残った"]].iterrows() if len(S) else []):
        t = rw["種類"]; up = rw[f"探索 {t}比"] > 1
        rr = (rw[f"探索 {t}実際"] + PSEUDO) / (rw[f"探索 {t}見込み"] + PSEUDO)
        tgt = em[(em[rw["区分の列"]] == rw["区分"]) & (em["grp"] == rw["艇番"]) & (em["per"] == "評価")]
        st = np.zeros(len(tgt)); rt = np.zeros(len(tgt)); nb = 0
        tst = []; trt = []; tdd = []
        for k, (idx, b, d, j, rn) in enumerate(zip(tgt["index"].values, tgt["boat"].values.astype(int), tgt["date"], tgt["jcd"], tgt["rno"])):
            mask = (I == b) if t == "1着" else ((I == b) | (J == b) | (K == b))
            pi = pi0[idx].copy(); pi[mask] *= rr; pi /= pi.sum()
            cand = (mask if up else ~mask) & v[idx] & (O[idx] <= 300 if not up else True)
            buy = cand & (pi * np.where(v[idx], O[idx], 0) > 1.0)
            st[k] = buy.sum() * 100.0; rt[k] = O[idx, y[idx]] * 100 if buy[y[idx]] else 0.0; nb += int(buy.sum())
            if up and t == "1着" and (d, j, rn) in tan.index:
                tb = tan.loc[(d, j, rn)]; tst.append(100.0); tdd.append(d); trt.append(float(tb["tan_pay"]) if int(tb["tan_boat"]) == b else 0.0)
        roi, lo, hi = boot_roi(days[tgt["index"].values], st, rt)
        row = {"区分の列": rw["区分の列"], "区分": rw["区分"], "艇番": rw["艇番"], "種類": t, "向き": "強い" if up else "弱い", "補正": rr, "艇数": len(tgt),
               "3連単 点数": nb, "3連単 回収率": roi, "3連単 CI下": lo, "3連単 CI上": hi}
        if tst:
            troi, tlo, thi = boot_roi(np.array(tdd), np.array(tst), np.array(trt))
            row.update({"単勝 点数": len(tst), "単勝 回収率": troi, "単勝 CI下": tlo, "単勝 CI上": thi})
        bets.append(row)
    B = pd.DataFrame(bets)
    win = []
    for _, b in B.iterrows() if len(B) else []:
        if b["3連単 点数"] >= 300 and b["3連単 回収率"] >= 1.1 and b["3連単 CI下"] > 1.0:
            win.append(b)
        if b.get("単勝 点数", 0) >= 300 and b.get("単勝 回収率", 0) >= 1.1 and b.get("単勝 CI下", 0) > 1.0:
            win.append(b)
    judge = "採用候補（前向きの検証へ）" if win else "不採用"
    # --- 12年の参考（オッズなし）：区分×艇番の1着率・3着以内率 ÷ 同じ年・同じ艇番の全体
    eh = e[e["pos"].notna()].copy(); eh["year"] = eh["date"].str[:4]
    eh["w"] = (eh["pos"] == 1).astype(float); eh["t3"] = (eh["pos"] <= 3).astype(float)
    base = eh.groupby(["year", "boat"])[["w", "t3"]].mean()
    eh = eh.join(base, on=["year", "boat"], rsuffix="_base")
    hist = []
    cells = S[["区分の列", "区分", "艇番", "種類"]].drop_duplicates() if len(S) else pd.DataFrame(columns=["区分の列", "区分", "艇番", "種類"])
    for _, rw in cells.iterrows():
        a_, b_ = ("w", "w_base") if rw["種類"] == "1着" else ("t3", "t3_base")
        g = eh[(eh[rw["区分の列"]] == rw["区分"]) & (eh["grp"] == rw["艇番"])].groupby("year").agg(a=(a_, "sum"), b=(b_, "sum"), n=(a_, "size"))
        g = g[g["n"] >= 50]; g["ratio"] = g["a"] / g["b"]
        hist.append({"区分の列": rw["区分の列"], "区分": rw["区分"], "艇番": rw["艇番"], "種類": rw["種類"], "年数": len(g),
                     "全体より上の年": int((g["ratio"] > 1).sum()), "比の平均": g["ratio"].mean(), "最低": g["ratio"].min(), "最高": g["ratio"].max(),
                     **{yy: g["ratio"].get(yy, np.nan) for yy in ("2014", "2017", "2020", "2023", "2025", "2026")}})
    H = pd.DataFrame(hist)
    # --- 出力
    def pick(rw, pp, k):
        return rw[f"{pp} {rw['種類']}{k}"]
    S_out = S.copy()
    if len(S):
        S_out["探索 比"] = [pick(rw, "探索", "比") for _, rw in S.iterrows()]
        S_out["探索 CI"] = ["%.2f〜%.2f" % (pick(rw, "探索", "CI下"), pick(rw, "探索", "CI上")) for _, rw in S.iterrows()]
        S_out["評価 比"] = [pick(rw, "評価", "比") for _, rw in S.iterrows()]
        S_out["評価 CI"] = ["%.2f〜%.2f" % (pick(rw, "評価", "CI下"), pick(rw, "評価", "CI上")) for _, rw in S.iterrows()]
        S_out = S_out[["区分の列", "区分", "艇番", "種類", "探索 艇数", "評価 艇数", "探索 比", "探索 CI", "評価 比", "評価 CI", "評価で残った"]].sort_values(["評価で残った", "評価 比"], ascending=[False, False])
    full = R[["区分の列", "区分", "艇番", "探索 艇数", "探索 ST平均", "探索 1着比", "探索 3着内比", "評価 艇数", "評価 1着比", "評価 3着内比"]]
    md = [f"# {title}（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}、{n}レース）", "", f"**判定：{judge}**", "",
          *extra_md,
          f"- 探索で選ばれた区分 {len(S)} 個のうち、評価でも同じ向きに有意に残ったのは {int(S['評価で残った'].sum()) if len(S) else 0} 個。", "",
          "## 探索で選ばれた区分と評価", "", S_out.to_markdown(index=False, floatfmt=".3f") if len(S) else "なし", "",
          "## 残った区分の買い方（評価。単勝は K の実額、3連単は確定オッズ）", "", B.to_markdown(index=False, floatfmt=".3f") if len(B) else "なし", "",
          "## 12年の参考（2014〜2026、オッズなし。比＝区分の1着率（3着以内率）÷ 同じ年・同じ艇番の全体）", "", H.to_markdown(index=False, floatfmt=".3f") if len(H) else "なし", "",
          "## 全区分（1着比・3着内比、ST平均は本番のスタートタイミング）", "", full.to_markdown(index=False, floatfmt=".3f")]
    return md, judge, S, B, H, R

def categories(e):
    st_ = e[~e["pos_raw"].astype(str).str.startswith("K")]
    d = st_.groupby(["toban", "date"]).agg(hasF=("pos_raw", lambda s: s.astype(str).str.startswith("F").any())).reset_index()
    d["dt"] = pd.to_datetime(d["date"]); d = d.sort_values(["toban", "dt"])
    d["prev_dt"] = d.groupby("toban")["dt"].shift(); d["gap"] = (d["dt"] - d["prev_dt"]).dt.days
    # 戻る前の最後の出走日までの60日以内の F
    fdates = d[d["hasF"]][["toban", "dt"]].rename(columns={"dt": "f_dt"}).sort_values("f_dt")
    q = d[d["prev_dt"].notna()][["toban", "date", "prev_dt"]].sort_values("prev_dt")
    q = pd.merge_asof(q, fdates, left_on="prev_dt", right_on="f_dt", by="toban", direction="backward")
    q["Fb"] = (q["prev_dt"] - q["f_dt"]).dt.days <= 60
    d = d.merge(q[["toban", "date", "Fb"]], on=["toban", "date"], how="left")
    # 直近の「戻った日」（空白21日以上）とその空白
    d["ret_dt"] = np.where(d["gap"] >= 21, d["dt"], pd.NaT); d["ret_gap"] = np.where(d["gap"] >= 21, d["gap"], np.nan)
    d["ret_F"] = np.where(d["gap"] >= 21, d["Fb"], np.nan)
    d[["ret_dt", "ret_gap", "ret_F"]] = d.groupby("toban")[["ret_dt", "ret_gap", "ret_F"]].ffill()
    d["ret_dt"] = pd.to_datetime(d["ret_dt"])
    d["in_ret"] = (d["dt"] - d["ret_dt"]).dt.days <= 13
    d["first_seen"] = d["prev_dt"].isna()
    x = e.merge(d[["toban", "date", "in_ret", "ret_gap", "ret_F", "ret_dt", "first_seen"]], on=["toban", "date"], how="left")
    # 戻った日からその出走の前までの出走数
    x = x.sort_values(["toban", "date", "rno"]); x["dt"] = pd.to_datetime(x["date"])
    started = ~x["pos_raw"].astype(str).str.startswith("K")
    x["k"] = started.astype(int)
    x["cum"] = x.groupby("toban")["k"].cumsum() - x["k"]
    first_cum = x[x["dt"] == x["ret_dt"]].groupby(["toban", "ret_dt"])["cum"].min().rename("c0").reset_index()
    x = x.merge(first_cum, on=["toban", "ret_dt"], how="left")
    x["nth"] = x["cum"] - x["c0"] + 1
    gapbin = pd.cut(x["ret_gap"], [20, 45, 90, 1e6], labels=["21〜45日", "46〜90日", "91日〜"]).astype(str)
    lab = np.where(x["in_ret"] == True, gapbin, "通常（〜20日）")
    lab = np.where(x["first_seen"] == True, "nan", lab)
    x["空白"] = lab
    isret = x["空白"].isin(["21〜45日", "46〜90日", "91日〜"])
    x["空白×F明け"] = np.where(isret, x["空白"] + "・" + np.where(x["ret_F"] == True, "F明け", "F以外"), "nan")
    x["空白×何走目"] = np.where(isret, x["空白"] + "・" + np.where(x["nth"] <= 2, "1〜2走目", "3走目以降"), "nan")
    return x.sort_values(KEY + ["boat"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("bkdir")
    ap.add_argument("--out", default="results/idea063.md"); a = ap.parse_args()
    e = load_entries([a.histds, a.datadir], ["toban", "pos", "pos_raw", "st"])
    e["pos"] = pd.to_numeric(e["pos"], errors="coerce")
    x = categories(e)
    ccols = ["空白", "空白×F明け", "空白×何走目"]
    cnt = x[(x["date"] >= EXP[0])]["空白"].value_counts().to_dict()
    extra = [f"- 空白日数は 2014年以降の出走から計算（欠場を除く）。休み明けの節 = 21日以上空いて戻った日から14日以内。2023-11 以降の艇数：{cnt}", ""]
    md, judge, S, B, H, R = core(x, ccols, a.datadir, a.bkdir, "IDEA-063 休み明けの選手", extra)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:40]))

if __name__ == "__main__":
    main()
