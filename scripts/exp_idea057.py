"""IDEA-057 ペルソナの掛け合わせ（6艇ぶんの人物像を1レースで重ねる。docs/ideas.md の登録どおり）
usage: python scripts/exp_idea057.py DATADIR HISTDS SEXCSV --bk BKDIR [BKDIR ...] [--out results/idea057.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ、12年の参考用）
- SEXCSV：登録番号→性別（IDEA-054 と同じ、公式のファン手帳から作ったもの。コミットしない）、BKDIR：K ファイル（単勝・複勝の実額）
- ペルソナ（IDEA-054 の列）× 艇番ごとの1着の「実際÷見込み」を探索期間（2023-11〜2025-12）だけで出し、+30 で 1 に縮める
- 1艇が複数のペルソナに当てはまるときは対数で足し、艇ごとの合計を ±log(1.6) で頭打ちにして、6艇を正規化し直す
- 評価 2026-01〜08：1号艇の 補正後÷市場 ≥ 1.15・1.25・1.35、どの艇でも ≥ 1.25 の艇。その艇の単勝・複勝を各100円（払戻は K の実額）
- 12年の参考：市場の項の代わりに、年ごと・艇番ごとの平均1着率を使って同じ補正をかける
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL, PSEUDO, personas, load_entries, ratio_ci, boot_roi

CAP = np.log(1.6)
BCOLS = ["年齢帯×性別", "年齢帯×級別", "年齢帯×地元"]   # 参考版B（条件付きロジット）で使う列

def fit_clogit(off, feats, nfeat, win, lam=1e-3):
    """条件付きロジット：s_b = off_b + Σ w[feats_b]。off (n,6)、feats：(n,6,k) の特徴量番号（-1 は無し）、win：1着の艇（0〜5）"""
    from scipy.optimize import minimize
    n = off.shape[0]; fk = feats.reshape(-1, feats.shape[2]); valid = fk >= 0; fidx = np.where(valid, fk, 0)
    def f(w):
        s = off + (np.where(valid, w[fidx], 0).sum(1)).reshape(n, 6)
        s = s - s.max(1, keepdims=True); e_ = np.exp(s); z = e_.sum(1, keepdims=True); pr = e_ / z
        ll = s[np.arange(n), win] - np.log(z[:, 0])
        g_s = pr.copy(); g_s[np.arange(n), win] -= 1.0; g_s /= n
        g = np.zeros_like(w); np.add.at(g, fidx[valid], np.repeat(g_s.reshape(-1), valid.sum(1)))
        return -ll.mean() + lam * (w ** 2).sum(), g + 2 * lam * w
    r = minimize(f, np.zeros(nfeat), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return r.x

def k_tanfuku_all(bkdirs, start, end):
    """K ファイルの単勝・複勝の払戻（全艇）：date, jcd, rno, tan_b, tan_pay, fuku（{艇: 払戻} の dict）"""
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
                        rno = int(m.group(1)); tb = 0; tp = 0; continue
                    if rno is None:
                        continue
                    z = BP.z2h(line)
                    mm = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", z)
                    if mm:
                        tb, tp = int(mm.group(1)), int(mm.group(2)); continue
                    mm = re.match(r"^\s+複勝\s+(.*)$", z)
                    if mm:
                        pr = re.findall(r"(\d)\s+(\d+)", mm.group(1))
                        rows.append((date, jcd, rno, tb, tp, {int(b): int(p) for b, p in pr})); rno = None
    d = pd.DataFrame(rows, columns=KEY + ["tan_b", "tan_pay", "fuku"])
    return d.drop_duplicates(KEY, keep="last")

def adjust(base, logs):
    """base：(n,6) 艇ごとの1着の見込み、logs：(n,6) 対数の補正（頭打ち済み）→ 補正後の確率と 補正後÷元"""
    q = base * np.exp(logs); q /= np.where(q.sum(1, keepdims=True) > 0, q.sum(1, keepdims=True), 1)
    return q, q / np.where(base > 0, base, np.nan)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("sexcsv")
    ap.add_argument("--bk", nargs="+", required=True); ap.add_argument("--out", default="results/idea057.md"); a = ap.parse_args()
    sex = pd.read_csv(a.sexcsv).set_index("toban")["sex"]
    e = load_entries([a.histds, a.datadir])
    ex = e[(e["date"] >= EXP[0]) & (e["date"] <= EXP[1])]
    starts_cut = np.unique(np.quantile(ex["starts"].dropna(), [0, .2, .4, .6, .8, 1])); starts_cut[0] = -1; starts_cut[-1] = 1e9
    toban_cut = np.unique(np.quantile(ex["toban"], [0, .2, .4, .6, .8, 1])); toban_cut[0] = 0; toban_cut[-1] = 1e5
    P_ = personas(e, sex, starts_cut, toban_cut); pcols = list(P_.columns)
    e = e.join(P_)
    # --- 市場（2023-10 以降、確定3連単オッズ）
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= EXP[0]) & (r["date"] <= EVL[1])]
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    I = np.array([int(c[0]) for c in COMBOS])
    pw = np.stack([P[:, I == b].sum(1) for b in range(1, 7)], 1)
    aw = np.stack([(I[y] == b) for b in range(1, 7)], 1).astype(float)
    days = m["date"].values; exp_m = days <= EXP[1]; evl_m = days >= EVL[0]
    # 艇ごとの縦持ち（市場の期間）
    em = m[KEY].reset_index().merge(e[KEY + ["boat"] + pcols], on=KEY, how="inner")
    ri = em["index"].values; bi_ = em["boat"].values.astype(int) - 1
    em["pw"] = pw[ri, bi_]; em["aw"] = aw[ri, bi_]; em["exp"] = exp_m[ri]
    # --- 探索期間だけで ペルソナ×艇番 の1着の補正（縮小 +30）
    ratio = {}
    ex_em = em[em["exp"]]
    for col in pcols:
        g = ex_em.groupby([col, "boat"]).agg(a=("aw", "sum"), x=("pw", "sum"))
        for (lab, b), rw in g.iterrows():
            if str(lab).startswith("nan") or "nan" in str(lab):
                continue
            ratio[(col, lab, int(b))] = (rw["a"] + PSEUDO) / (rw["x"] + PSEUDO)
    def log_adj(df):
        """艇ごとの対数補正の合計（頭打ち）"""
        tot = np.zeros(len(df))
        for col in pcols:
            tot += np.log(np.array([ratio.get((col, lab, int(b)), 1.0) for lab, b in zip(df[col].values, df["boat"].values)]))
        return np.clip(tot, -CAP, CAP)
    em["ladj"] = log_adj(em)
    L = np.zeros((n, 6)); L[ri, bi_] = em["ladj"].values
    have = np.zeros(n, int); np.add.at(have, ri, 1); full = have == 6
    q, R = adjust(pw, L)
    # --- 参考版B：重なりを除いた補正（探索期間で条件付きロジットを学習、市場の対数確率を基準に）
    fid = {}
    for col in BCOLS:
        for lab, b in em[[col, "boat"]].drop_duplicates().itertuples(index=False):
            if "nan" not in str(lab):
                fid.setdefault((col, lab, int(b)), len(fid))
    FE = -np.ones((n, 6, len(BCOLS)), int)
    for k, col in enumerate(BCOLS):
        FE[ri, bi_, k] = [fid.get((col, lab, int(b)), -1) for lab, b in zip(em[col].values, em["boat"].values)]
    offm = np.log(np.clip(pw, 1e-9, None)); winb = np.argmax(aw, 1)
    tr_ = np.where(exp_m & full)[0]
    wB = fit_clogit(offm[tr_], FE[tr_], len(fid), winb[tr_])
    LB = np.where(FE >= 0, wB[np.where(FE >= 0, FE, 0)], 0).sum(2)
    qB, RB = adjust(pw, LB)
    # --- K の単勝・複勝（実額）
    kt = k_tanfuku_all(a.bk, "20140101", EVL[1])
    km = m[KEY].reset_index().merge(kt, on=KEY, how="left")
    km = km.sort_values("index"); tan_b = km["tan_b"].fillna(0).astype(int).values; tan_pay = km["tan_pay"].fillna(0).values
    fuku = km["fuku"].values; hasK = km["tan_b"].notna().values & (km["tan_pay"].fillna(0).values > 0)
    def payouts(idx, boats):
        t = np.where(tan_b[idx] == boats, tan_pay[idx], 0.0)
        f = np.array([(fk.get(int(b), 0) if isinstance(fk, dict) else 0) for fk, b in zip(fuku[idx], boats)], float)
        return t, f
    out = []
    def evaluate(label, idx, boats, per_mask_name, qq=None):
        qq = q if qq is None else qq
        t, f = payouts(idx, boats)
        rt_t, lo_t, hi_t = boot_roi(days[idx], np.full(len(idx), 100.0), t)
        rt_f, lo_f, hi_f = boot_roi(days[idx], np.full(len(idx), 100.0), f)
        aw_s = aw[idx, boats - 1].sum(); ex_s = pw[idx, boats - 1].sum(); qx = qq[idx, boats - 1].sum()
        out.append({"期間": per_mask_name, "選び方": label, "レース（艇）": len(idx), "1着 実際÷市場": aw_s / ex_s if ex_s > 0 else np.nan,
                    "補正後÷市場（平均）": qx / ex_s if ex_s > 0 else np.nan, "単勝": rt_t, "単勝CI": f"{lo_t:.2f}〜{hi_t:.2f}",
                    "複勝": rt_f, "複勝CI": f"{lo_f:.2f}〜{hi_f:.2f}"})
    for pname, pm in (("探索", exp_m), ("評価", evl_m)):
        base = pm & full & hasK
        idx = np.where(base)[0]; evaluate("全レース（1号艇）", idx, np.ones(len(idx), int), pname)
        for th in (1.15, 1.25, 1.35):
            idx = np.where(base & (R[:, 0] >= th))[0]; evaluate(f"1号艇 補正後÷市場 ≥ {th}", idx, np.ones(len(idx), int), pname)
        rr, bb = np.where(base[:, None] & (R >= 1.25))
        evaluate("どの艇でも ≥ 1.25（その艇）", rr, bb + 1, pname)
        rr, bb = np.where(base[:, None] & np.isfinite(R))
        evaluate("全艇（基準）", rr, bb + 1, pname)
        for th in (1.15, 1.25, 1.35):
            idx = np.where(base & (RB[:, 0] >= th))[0]; evaluate(f"参考B 1号艇 ≥ {th}", idx, np.ones(len(idx), int), pname, qB)
        rr, bb = np.where(base[:, None] & (RB >= 1.25))
        evaluate("参考B どの艇でも ≥ 1.25", rr, bb + 1, pname, qB)
    T = pd.DataFrame(out)
    ev = T[T["期間"] == "評価"].set_index("選び方")
    win = []
    for lab in ["1号艇 補正後÷市場 ≥ 1.15", "1号艇 補正後÷市場 ≥ 1.25", "1号艇 補正後÷市場 ≥ 1.35", "どの艇でも ≥ 1.25（その艇）"]:
        rw = ev.loc[lab]
        for k in ("単勝", "複勝"):
            if rw["レース（艇）"] >= 300 and rw[k] >= 1.1 and float(rw[f"{k}CI"].split("〜")[0]) > 1.0:
                win.append(f"{lab}・{k}")
    judge = "採用候補（前向きの検証へ）" if win else "不採用"
    # --- 12年の参考：年ごと・艇番ごとの平均1着率を市場の代わりに
    rh = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["tri_combo"])) for p in (a.histds, a.datadir)])
    rh = rh.dropna().drop_duplicates(KEY, keep="last"); rh = rh[(rh["date"] >= "20140101") & (rh["date"] <= EVL[1])]
    bad = e[e["pos_raw"].astype(str).str.match(r"^[KFLS]")][KEY].drop_duplicates()
    six = e.groupby(KEY).size(); six = six[six == 6].reset_index()[KEY]
    rh = rh.merge(six, on=KEY).merge(bad.assign(x=1), on=KEY, how="left"); rh = rh[rh["x"].isna()].drop(columns="x")
    rh = rh[rh["tri_combo"].astype(str).str[0].str.isdigit()].reset_index(drop=True)
    rh["w"] = rh["tri_combo"].astype(str).str[0].astype(int); rh["year"] = rh["date"].str[:4]
    nh = len(rh)
    eh = rh[KEY].reset_index().merge(e[KEY + ["boat"] + pcols], on=KEY)
    eh["ladj"] = log_adj(eh)
    Lh = np.zeros((nh, 6)); Lh[eh["index"].values, eh["boat"].values.astype(int) - 1] = eh["ladj"].values
    wr = pd.crosstab(rh["year"], rh["w"], normalize="index").reindex(columns=range(1, 7), fill_value=0)
    Bh = wr.loc[rh["year"]].to_numpy(float)
    qh, Rh = adjust(Bh, Lh)
    FEh = -np.ones((nh, 6, len(BCOLS)), int); hi_ = eh["index"].values; hb_ = eh["boat"].values.astype(int) - 1
    for k, col in enumerate(BCOLS):
        FEh[hi_, hb_, k] = [fid.get((col, lab, int(b)), -1) for lab, b in zip(eh[col].values, eh["boat"].values)]
    LBh = np.where(FEh >= 0, wB[np.where(FEh >= 0, FEh, 0)], 0).sum(2)
    _, RBh = adjust(Bh, LBh)
    kh = rh[KEY].reset_index().merge(kt, on=KEY, how="left").sort_values("index")
    tb_h = kh["tan_b"].fillna(0).astype(int).values; tp_h = kh["tan_pay"].fillna(0).values; fk_h = kh["fuku"].values
    okh = kh["tan_b"].notna().values & (tp_h > 0)
    yrs = rh["year"].values; hist = []
    def hrow(label, idx, boats):
        t = np.where(tb_h[idx] == boats, tp_h[idx], 0.0)
        f = np.array([(fk.get(int(b), 0) if isinstance(fk, dict) else 0) for fk, b in zip(fk_h[idx], boats)], float)
        d = pd.DataFrame({"year": yrs[idx], "t": t, "f": f}).groupby("year").agg(n=("t", "size"), t=("t", "sum"), f=("f", "sum"))
        d["単勝"] = d["t"] / (d["n"] * 100); d["複勝"] = d["f"] / (d["n"] * 100)
        return d
    sels = {"全レース（1号艇）": (np.where(okh)[0], None), "1号艇 ≥ 1.15": (np.where(okh & (Rh[:, 0] >= 1.15))[0], None),
            "1号艇 ≥ 1.25": (np.where(okh & (Rh[:, 0] >= 1.25))[0], None), "1号艇 ≥ 1.35": (np.where(okh & (Rh[:, 0] >= 1.35))[0], None)}
    HY = {}
    for lab, (idx, _) in sels.items():
        HY[lab] = hrow(lab, idx, np.ones(len(idx), int))
    rr, bb = np.where(okh[:, None] & (Rh >= 1.25)); HY["どの艇でも ≥ 1.25"] = hrow("x", rr, bb + 1)
    for th in (1.15, 1.25):
        idx = np.where(okh & (RBh[:, 0] >= th))[0]; HY[f"参考B 1号艇 ≥ {th}"] = hrow("x", idx, np.ones(len(idx), int))
    rr, bb = np.where(okh[:, None] & (RBh >= 1.25)); HY["参考B どの艇でも ≥ 1.25"] = hrow("x", rr, bb + 1)
    base_t = HY["全レース（1号艇）"]["単勝"]
    hs = []
    for lab, d in HY.items():
        hs.append({"選び方": lab, "レース（艇）": int(d["n"].sum()), "単勝 通算": d["t"].sum() / (d["n"].sum() * 100), "複勝 通算": d["f"].sum() / (d["n"].sum() * 100),
                   "単勝100%超の年": int((d["単勝"] > 1).sum()), "複勝100%超の年": int((d["複勝"] > 1).sum()),
                   "単勝が全レース1号艇を上回った年": int((d["単勝"] > base_t.reindex(d.index)).sum()), "年数": len(d)})
    HS = pd.DataFrame(hs)
    YT = pd.DataFrame({lab: d["単勝"] for lab, d in HY.items()}); YT.index.name = "年（単勝）"
    YF = pd.DataFrame({lab: d["複勝"] for lab, d in HY.items()}); YF.index.name = "年（複勝）"
    YN = pd.DataFrame({lab: d["n"] for lab, d in HY.items()}); YN.index.name = "年（レース・艇数）"
    # 補正の大きい ペルソナ×艇番（参考）
    RT = pd.DataFrame([{"ペルソナ": c, "区分": l, "艇番": b, "補正": v_} for (c, l, b), v_ in ratio.items()])
    RT["|log|"] = np.log(RT["補正"]).abs(); RT = RT.sort_values("|log|", ascending=False).drop(columns="|log|").head(20)
    md = [f"# IDEA-057 ペルソナの掛け合わせ（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}、{n}レース）", "",
          f"**判定：{judge}**" + (f"（{'、'.join(win)}）" if win else ""), "",
          "## まとめ", "",
          "- 補正：探索期間だけで ペルソナ×艇番 の1着の 実際÷見込み を出し +30 で縮め、1艇の複数のペルソナは対数で足して ±log(1.6) で頭打ち、6艇を正規化し直した。",
          "- 払戻は K ファイルの単勝・複勝の実額（各100円）。性別は公式のファン手帳から（IDEA-054 と同じ）。",
          "- 注意：1艇は10個前後のペルソナ列（年齢帯・級別・年齢帯×級別・体重帯など）に同時に当てはまり、重なった偏りを掛け合わせるため補正が大きく出すぎる（全レースの1号艇で 補正後÷市場 の平均が約1.25、実際÷市場は約1.04）。",
          "  そのため参考に「参考B」（年齢帯×性別・年齢帯×級別・年齢帯×地元の3列だけを、探索期間で条件付きロジットにより同時に学習＝重なりを除いた版）も出した。判定は登録どおりの版で行う。", "",
          "## 結果（探索・評価）", "", T.to_markdown(index=False, floatfmt=".3f"), "",
          "## 12年の参考（2014〜2026、市場の代わりに年ごと・艇番ごとの平均1着率で補正）", "", HS.to_markdown(index=False, floatfmt=".3f"), "",
          "### 年ごとの単勝の回収率", "", YT.reset_index().to_markdown(index=False, floatfmt=".3f"), "",
          "### 年ごとの複勝の回収率", "", YF.reset_index().to_markdown(index=False, floatfmt=".3f"), "",
          "### 年ごとのレース（艇）数", "", YN.reset_index().to_markdown(index=False, floatfmt=".0f"), "",
          "## 補正の大きい ペルソナ×艇番（探索、上位20）", "", RT.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:40]))

if __name__ == "__main__":
    main()
