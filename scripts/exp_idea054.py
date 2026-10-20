"""IDEA-054 選手の「ペルソナ」で市場の見落としを探す（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea054.py DATADIR HISTDS SEXCSV [--out results/idea054.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ、12年の参考用）
- SEXCSV：公式の選手データ（ファン手帳 https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fanYYMM.lzh、
  fan1410・1710・2010・2310・2510・2604 を fetch_url ワークフローで取得）の各行の「級別の直後の 元号+生年月日6桁 の次の1桁」が性別。
  登録番号→性別（1 男子・2 女子）の CSV（列 toban,sex）。載っていない選手（出走の0.04%）は体重50kg未満を女子とみなす
- ペルソナ × 艇番ごとに、1着率・3着以内率 ÷ 市場の見込み（確定3連単オッズから）。探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で確かめる
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm

KEY = ["date", "jcd", "rno"]; EXP = ("20231101", "20251231"); EVL = ("20260101", "20260831")
RNG = np.random.default_rng(0); PSEUDO = 30
VENUE_PREF = {"01": "群馬", "02": "埼玉", "03": "東京", "04": "東京", "05": "東京", "06": "静岡", "07": "愛知", "08": "愛知", "09": "三重", "10": "福井",
              "11": "滋賀", "12": "大阪", "13": "兵庫", "14": "徳島", "15": "香川", "16": "岡山", "17": "広島", "18": "山口", "19": "山口", "20": "福岡",
              "21": "福岡", "22": "福岡", "23": "佐賀", "24": "長崎"}

def personas(e, sex, starts_cut, toban_cut):
    """艇ごとのペルソナ（列名 → 区分のラベル）"""
    p = pd.DataFrame(index=e.index)
    age = pd.to_numeric(e["age"], errors="coerce")
    p["年齢帯"] = pd.cut(age, [0, 22, 25, 29, 39, 49, 99], labels=["〜22", "23〜25", "26〜29", "30〜39", "40〜49", "50〜"]).astype(str)
    s = e["toban"].map(sex)
    fem = np.where(s.notna(), s == 2, pd.to_numeric(e["weight"], errors="coerce") < 50)
    p["性別"] = np.where(fem, "女子", "男子")
    p["登録番号の新しさ"] = pd.cut(e["toban"], toban_cut, labels=["古い1", "2", "3", "4", "新しい5"], include_lowest=True).astype(str)
    p["出走経験"] = pd.cut(e["starts"], starts_cut, labels=["少ない1", "2", "3", "4", "多い5"], include_lowest=True).astype(str)
    p["級別"] = e["cls"].astype(str)
    p["地元"] = np.where(e["branch"].astype(str) == e["jcd"].map(VENUE_PREF), "地元", "地元以外")
    w = pd.to_numeric(e["weight"], errors="coerce")
    p["体重帯"] = pd.cut(w, [0, 49, 51, 53, 55, 99], labels=["〜49", "50〜51", "52〜53", "54〜55", "56〜"]).astype(str)
    p["年齢帯×性別"] = p["年齢帯"] + "・" + p["性別"]
    p["年齢帯×級別"] = p["年齢帯"] + "・" + p["級別"]
    p["年齢帯×地元"] = p["年齢帯"] + "・" + p["地元"]
    return p

def ratio_ci(days, act, exp):
    g = pd.DataFrame({"d": days, "a": act, "e": exp}).groupby("d").sum()
    if len(g) < 2 or g["e"].sum() <= 0:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(2000, len(g))); b = g["a"].values[bi].sum(1) / g["e"].values[bi].sum(1)
    return float(g["a"].sum() / g["e"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def boot_roi(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(2000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(g["rt"].sum() / g["st"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def load_entries(paths):
    cols = KEY + ["boat", "toban", "age", "branch", "weight", "cls", "pos_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in paths], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last").sort_values(["date", "jcd", "rno", "boat"]).reset_index(drop=True)
    # 出走経験：2014年以降、その日より前の出走数
    first = e.groupby(["toban", "date"]).size().rename("n").reset_index().sort_values(["toban", "date"])
    first["starts"] = first.groupby("toban")["n"].cumsum() - first["n"]
    return e.merge(first[["toban", "date", "starts"]], on=["toban", "date"], how="left")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("sexcsv")
    ap.add_argument("--out", default="results/idea054.md"); a = ap.parse_args()
    sex = pd.read_csv(a.sexcsv).set_index("toban")["sex"]
    e = load_entries([a.histds, a.datadir])
    ex = e[(e["date"] >= EXP[0]) & (e["date"] <= EXP[1])]
    starts_cut = np.unique(np.quantile(ex["starts"].dropna(), [0, .2, .4, .6, .8, 1])); starts_cut[0] = -1; starts_cut[-1] = 1e9
    toban_cut = np.unique(np.quantile(ex["toban"], [0, .2, .4, .6, .8, 1])); toban_cut[0] = 0; toban_cut[-1] = 1e5
    P_ = personas(e, sex, starts_cut, toban_cut)
    pcols = list(P_.columns)
    # --- 市場（2023-10 以降、確定3連単オッズ）
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= EXP[0]) & (r["date"] <= EVL[1])]
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
    # 艇ごとの縦持ち
    em = e.join(P_)
    em = m[KEY].reset_index().merge(em[KEY + ["boat"] + pcols], on=KEY, how="inner")
    ri = em["index"].values; bi_ = em["boat"].values.astype(int) - 1
    em["pw"] = pw[ri, bi_]; em["pt3"] = pt3[ri, bi_]; em["aw"] = aw[ri, bi_]; em["at3"] = at3[ri, bi_]; em["per"] = per[ri]
    rows = []
    for col in pcols:
        for (lab, boat), g in em.groupby([col, "boat"]):
            if lab in ("nan", "nan・男子", "nan・女子") or str(lab).startswith("nan"):
                continue
            row = {"ペルソナ": col, "区分": lab, "艇番": int(boat)}
            for pp in ("探索", "評価"):
                gg = g[g["per"] == pp]
                row[f"{pp} 艇数"] = len(gg)
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
        if rw[f"探索 {t}比"] > 1:
            return rw[f"評価 {t}CI下"] > 1
        return rw[f"評価 {t}CI上"] < 1
    S["評価で残った"] = S.apply(survived, axis=1)
    # --- 残った区分（市場より強い向き）の買い方：π×オッズ > 1.0
    pi0 = P ** 1.06; pi0 /= pi0.sum(1, keepdims=True)
    bets = []
    for _, rw in S[S["評価で残った"]].iterrows():
        t = rw["種類"]; up = rw[f"探索 {t}比"] > 1
        rr = (rw[f"探索 {t}実際"] + PSEUDO) / (rw[f"探索 {t}見込み"] + PSEUDO)
        tgt = em[(em[rw["ペルソナ"]] == rw["区分"]) & (em["boat"] == rw["艇番"]) & (em["per"] == "評価")]
        idx = tgt["index"].values; b = int(rw["艇番"])
        if not up or len(idx) == 0:
            bets.append({**{k: rw[k] for k in ("ペルソナ", "区分", "艇番", "種類")}, "補正": rr, "点数": 0, "回収率": np.nan, "CI下": np.nan, "CI上": np.nan}); continue
        mask = (I == b) if t == "1着" else ((I == b) | (J == b) | (K == b))
        pi = pi0[idx].copy(); pi[:, mask] *= rr; pi /= pi.sum(1, keepdims=True)
        buy = v[idx] & mask[None, :] & (pi * np.where(v[idx], O[idx], 0) > 1.0)
        hit = np.zeros_like(buy); hit[np.arange(len(idx)), y[idx]] = True
        st = buy.sum(1) * 100.0; rt = np.where(buy & hit, O[idx] * 100, 0).sum(1)
        roi, lo, hi = boot_roi(days[idx], st, rt)
        # 参考：1番人気1点、その艇が1着の組を全部（確定オッズ）
        fav = np.argmin(np.where(v[idx], O[idx], np.inf), 1)
        fr = np.where(fav == y[idx], O[idx][np.arange(len(idx)), fav] * 100, 0)
        allb = v[idx] & (I[None, :] == b); ar = np.where(allb & hit, O[idx] * 100, 0).sum(1)
        bets.append({**{k: rw[k] for k in ("ペルソナ", "区分", "艇番", "種類")}, "補正": rr, "点数": int(buy.sum()), "回収率": roi, "CI下": lo, "CI上": hi,
                     "レース": len(idx), "参考 1番人気1点": fr.sum() / (100.0 * len(idx)), "参考 その艇1着を全部": ar.sum() / (allb.sum() * 100.0)})
    B = pd.DataFrame(bets)
    win = B[(B["点数"] >= 300) & (B["回収率"] >= 1.1) & (B["CI下"] > 1.0)] if len(B) else B
    judge = "採用候補（前向きの検証へ）" if len(win) else "不採用"
    # --- 12年の参考：その艇が1着の組を全部（20点）の年別回収率
    rh = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["tri_combo", "tri_pay"])) for p in (a.histds, a.datadir)])
    rh = rh.dropna().drop_duplicates(KEY, keep="last"); rh = rh[(rh["date"] >= "20140101") & (rh["date"] <= EVL[1])]
    bad = e[e["pos_raw"].astype(str).str.match(r"^[KFLS]")][KEY].drop_duplicates()
    six = e.groupby(KEY).size(); six = six[six == 6].reset_index()[KEY]
    rh = rh.merge(six, on=KEY).merge(bad.assign(x=1), on=KEY, how="left"); rh = rh[rh["x"].isna()].drop(columns="x")
    rh["w"] = rh["tri_combo"].astype(str).str[0]
    eh = rh.merge(e.join(P_)[KEY + ["boat"] + pcols], on=KEY)
    eh["year"] = eh["date"].str[:4]; eh["ret"] = np.where(eh["w"] == eh["boat"].astype(str), pd.to_numeric(eh["tri_pay"], errors="coerce"), 0)
    base = eh.groupby(["year", "boat"]).agg(ret=("ret", "sum"), n=("ret", "size")); base["roi"] = base["ret"] / (base["n"] * 2000)
    hist = []
    for _, rw in S[S["種類"] == "1着"].drop_duplicates(["ペルソナ", "区分", "艇番"]).iterrows():
        g = eh[(eh[rw["ペルソナ"]] == rw["区分"]) & (eh["boat"] == rw["艇番"])].groupby("year").agg(ret=("ret", "sum"), n=("ret", "size"))
        g["roi"] = g["ret"] / (g["n"] * 2000); bb = base.xs(rw["艇番"], level="boat")["roi"]
        hist.append({"ペルソナ": rw["ペルソナ"], "区分": rw["区分"], "艇番": rw["艇番"], "年数": len(g), "基準を上回った年": int((g["roi"] > bb.reindex(g.index)).sum()),
                     "回収率の平均": g["roi"].mean(), "最高の年": g["roi"].max(), "100%超の年": int((g["roi"] > 1).sum()),
                     **{y_: g["roi"].get(y_, np.nan) for y_ in ("2014", "2017", "2020", "2023", "2025", "2026")}})
    H = pd.DataFrame(hist)
    # --- 出力
    keep = ["ペルソナ", "区分", "艇番", "種類", "探索 艇数", "評価 艇数"]
    def pick(rw, pp, k):
        return rw[f"{pp} {rw['種類']}{k}"]
    S_out = S.copy()
    S_out["探索 比"] = [pick(rw, "探索", "比") for _, rw in S.iterrows()]
    S_out["探索 CI"] = ["%.2f〜%.2f" % (pick(rw, "探索", "CI下"), pick(rw, "探索", "CI上")) for _, rw in S.iterrows()]
    S_out["評価 比"] = [pick(rw, "評価", "比") for _, rw in S.iterrows()]
    S_out["評価 CI"] = ["%.2f〜%.2f" % (pick(rw, "評価", "CI下"), pick(rw, "評価", "CI上")) for _, rw in S.iterrows()]
    S_out = S_out[keep + ["探索 比", "探索 CI", "評価 比", "評価 CI", "評価で残った"]].sort_values(["評価で残った", "評価 比"], ascending=[False, False])
    surv = S_out[S_out["評価で残った"]]
    top = surv.reindex(surv["評価 比"].sub(1).abs().sort_values(ascending=False).index).head(12)
    full = R[["ペルソナ", "区分", "艇番", "探索 艇数", "探索 1着比", "探索 3着内比", "評価 艇数", "評価 1着比", "評価 3着内比"]]
    md = [f"# IDEA-054 選手のペルソナ（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}、{n}レース）", "",
          f"**判定：{judge}**", "",
          "## まとめ", "",
          f"- 性別は公式の選手データ（ファン手帳 fan1410〜fan2604）から。載っていない選手だけ体重50kg未満を女子とみなした。",
          f"- 探索で選ばれた区分 {len(S)} 個のうち、評価でも同じ向きに有意に残ったのは {int(S['評価で残った'].sum())} 個。",
          f"- 残った区分で π×オッズ > 1.0 を買った場合：{'、'.join(f'{x.区分}・{x.艇番}号艇（{x.種類}）{x.点数}点 {x.回収率:.0%}' for x in B.itertuples() if x.点数 > 0) if len(B) else 'なし'}", "",
          "### 評価でも残った偏り（大きい順、上位12）", "", top.to_markdown(index=False, floatfmt=".3f"), "",
          "## 探索で選ばれた区分と評価", "", S_out.to_markdown(index=False, floatfmt=".3f"), "",
          "## 残った区分の買い方（評価、π×オッズ > 1.0、確定オッズで払戻）", "", B.to_markdown(index=False, floatfmt=".3f") if len(B) else "なし", "",
          "## 12年の参考（2014〜2026、その艇が1着の組を全部20点。基準＝同じ艇番の全体）", "", H.to_markdown(index=False, floatfmt=".3f") if len(H) else "なし", "",
          "## 全区分（1着比・3着内比）", "", full.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:30]))

if __name__ == "__main__":
    main()
