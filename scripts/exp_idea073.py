"""IDEA-073 ミッドナイトの日は、1号艇に人気が偏りすぎているか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea073.py DATADIR HISTDS STOREDIR [--histbk /tmp/claude-0/hist] [--out results/idea073.md]
- 開催形態：場・日ごとの最終レースの締切（entries.deadline の最大）が 22:30 以降 → ミッドナイト、20:00〜22:29 → ナイター、それより前 → 昼
- 1. オッズなし：1号艇の1着率・3着内率（形態 × 前半1〜6R／後半7〜12R、ミッドナイトのある場は形態別）
- 2. 市場との比べ：1号艇の1着 実際÷見込み（確定3連単、参考に確定単勝）、1号艇が勝ったときの2着の艇 実際÷見込み、
     1番人気の3連単と 1-x-x のオッズ帯別 実際÷見込み。日単位ブートストラップ
- 3. 買い方（K の実額）：1号艇の単勝・複勝、3連単 1-x-x 全20点、2連単 1-x 全5点。参考：確定1番人気の3連単1点
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL, ratio_ci, boot_roi
from exp_idea072 import kpay
from idea050 import k_pay

FIRST = np.array([int(c[0]) for c in COMBOS]); SECOND = np.array([int(c[2]) for c in COMBOS])
BANDS = [(0, 10, "〜10倍"), (10, 30, "10〜30倍"), (30, 100, "30〜100倍"), (100, 1e9, "100倍〜")]

def fmt(lo, hi):
    return f"{lo:.2f}〜{hi:.2f}" if np.isfinite(lo) else ""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("store")
    ap.add_argument("--histbk", default="/tmp/claude-0/hist"); ap.add_argument("--out", default="results/idea073.md"); a = ap.parse_args()
    cols = KEY + ["boat", "pos_raw", "deadline"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in (a.histds, a.datadir)], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last")
    bad = e[e["pos_raw"].astype(str).str.match(r"^[KFL]")][KEY].drop_duplicates().assign(bad=1)
    six = e.groupby(KEY).size().rename("nb").reset_index(); six = six[six["nb"] == 6][KEY]
    dl = e[e["boat"] == 1][KEY + ["deadline"]].dropna()
    last = dl.groupby(["date", "jcd"])["deadline"].max().rename("last").reset_index()
    last["形態"] = np.where(last["last"] >= "22:30", "ミッドナイト", np.where(last["last"] >= "20:00", "ナイター", "昼"))
    G = dl.merge(last, on=["date", "jcd"]).merge(six, on=KEY).merge(bad, on=KEY, how="left")
    G = G[G["bad"].isna()].drop(columns="bad")
    G["半分"] = np.where(G["rno"] <= 6, "前半1〜6R", "後半7〜12R")
    pos = e[e["boat"] == 1][KEY + ["pos_raw"]].rename(columns={"pos_raw": "pos1"})
    r = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["tri_combo", "tri_pay"])) for p in (a.histds, a.datadir)])
    r = r.dropna(subset=["tri_combo"]).drop_duplicates(KEY, keep="last")
    G = G.merge(r, on=KEY).merge(pos, on=KEY)
    G["win1"] = (G["tri_combo"].str[0] == "1").astype(float)
    G["top3_1"] = G["tri_combo"].str.contains("1").astype(float)
    G["per"] = np.select([(G["date"] >= EXP[0]) & (G["date"] <= EXP[1]), (G["date"] >= EVL[0]) & (G["date"] <= EVL[1])], ["探索", "評価"], "その他")
    G = G[G["date"] <= EVL[1]].reset_index(drop=True)
    print("races", len(G), G["形態"].value_counts().to_dict(), flush=True)
    K = kpay([a.histbk, a.store], "20140101", EVL[1]); G = G.merge(K, on=KEY, how="left")
    K2 = k_pay(a.store, "20231001"); K2 = K2[K2["kind"] == "2t"][KEY + ["combo", "pay"]].rename(columns={"combo": "c2t", "pay": "pay2t"})
    G = G.merge(K2, on=KEY, how="left")
    # 確定3連単の見込み
    o = norm(pd.read_parquet(f"{a.datadir}/odds.parquet")); O = o[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    P = np.where(v, 1 / np.where(v, O, 1), 0); s = P.sum(1, keepdims=True); P = P / np.where(s > 0, s, 1)
    ok = v.sum(1) >= 60
    oo = o[KEY].copy(); oo["p1"] = P[:, FIRST == 1].sum(1)
    for j in range(2, 7):
        oo[f"p1{j}"] = P[:, (FIRST == 1) & (SECOND == j)].sum(1)
    fav = np.argmin(np.where(v, O, np.inf), 1); oo["fav"] = np.array(COMBOS)[fav]; oo["pfav"] = P[np.arange(len(P)), fav]
    oo = oo[ok]; Oi = O[ok]; Pi = P[ok]; okeys = oo[KEY].reset_index(drop=True)
    G = G.merge(oo, on=KEY, how="left")
    tf = pd.concat([pd.read_csv(f, dtype={"jcd": str, "date": str}) for f in sorted(glob.glob(f"{a.store}/oddstf/*.csv.gz"))])
    T = tf[[f"tan{i}" for i in range(1, 7)]].to_numpy(float); tv = np.isfinite(T) & (T > 0)
    Q = np.where(tv, 1 / np.where(tv, T, 1), 0); tf["q1"] = np.where(tv.all(1), Q[:, 0] / Q.sum(1), np.nan)
    tf["jcd"] = tf["jcd"].str.zfill(2); G = G.merge(tf[KEY + ["q1"]].drop_duplicates(KEY), on=KEY, how="left")
    FORMS = ["ミッドナイト", "ナイター", "昼"]
    mvenues = sorted(G.loc[G["形態"] == "ミッドナイト", "jcd"].unique())

    # ---- 1. オッズなし
    rows = []
    for f in FORMS:
        for h in ("前半1〜6R", "後半7〜12R", "全体"):
            for pp in ("探索", "評価", "全期間"):
                g = G[(G["形態"] == f) & ((G["半分"] == h) | (h == "全体")) & ((G["per"] == pp) | (pp == "全期間"))]
                rows.append({"形態": f, "レース": h, "期間": pp, "件数": len(g), "1着率": g["win1"].mean(), "3着内率": g["top3_1"].mean()})
    A1 = pd.DataFrame(rows)
    rows = []
    for j in mvenues:
        for f in FORMS:
            for h in ("前半1〜6R", "後半7〜12R"):
                g = G[(G["jcd"] == j) & (G["形態"] == f) & (G["半分"] == h) & (G["date"] >= "20230101")]
                if len(g) >= 50:
                    rows.append({"場": j, "形態": f, "レース": h, "件数": len(g), "日数": g["date"].nunique(), "1着率": g["win1"].mean(), "3着内率": g["top3_1"].mean()})
    A2 = pd.DataFrame(rows)

    # ---- 2. 市場との比べ
    rows = []
    for f in FORMS:
        for h in ("前半1〜6R", "後半7〜12R", "全体"):
            row = {"形態": f, "レース": h}
            for pp in ("探索", "評価"):
                g = G[(G["形態"] == f) & ((G["半分"] == h) | (h == "全体")) & (G["per"] == pp) & G["p1"].notna()]
                rr, lo, hi = ratio_ci(g["date"].values, g["win1"].values, g["p1"].values) if len(g) >= 100 else (np.nan,) * 3
                row[f"{pp} 件数"] = len(g); row[f"{pp} 1着率"] = g["win1"].mean(); row[f"{pp} 見込み"] = g["p1"].mean()
                row[f"{pp} 比"] = rr; row[f"{pp} CI"] = fmt(lo, hi)
            gq = G[(G["形態"] == f) & ((G["半分"] == h) | (h == "全体")) & (G["per"] == "評価") & G["q1"].notna()]
            rq = ratio_ci(gq["date"].values, gq["win1"].values, gq["q1"].values) if len(gq) >= 100 else (np.nan,) * 3
            row["評価 比（確定単勝）"] = rq[0]; row["評価 単勝CI"] = fmt(rq[1], rq[2]); row["単勝 件数"] = len(gq)
            rows.append(row)
    B1 = pd.DataFrame(rows)
    # 1号艇が勝ったときの2着
    rows = []
    W = G[(G["win1"] == 1) & G["p1"].notna()].copy(); W["sec"] = W["tri_combo"].str[2].astype(int)
    for f in FORMS:
        for pp in ("探索", "評価"):
            g = W[(W["形態"] == f) & (W["per"] == pp)]
            for j in range(2, 7):
                act = (g["sec"] == j).astype(float).values; exp = (g[f"p1{j}"] / g["p1"]).values
                rr, lo, hi = ratio_ci(g["date"].values, act, exp) if len(g) >= 100 else (np.nan,) * 3
                rows.append({"形態": f, "期間": pp, "2着の艇": j, "件数": len(g), "実際": act.mean(), "見込み": exp.mean(), "比": rr, "CI": fmt(lo, hi)})
    B2 = pd.DataFrame(rows)
    # 1番人気の3連単・1-x-x のオッズ帯
    Gm = G.dropna(subset=["p1"]).reset_index(drop=True)
    idx = okeys.reset_index().merge(Gm[KEY + ["形態", "per", "tri_combo"]], on=KEY)
    Oi2 = Oi[idx["index"].values]; Pi2 = Pi[idx["index"].values]
    hitm = (np.array(COMBOS)[None, :] == idx["tri_combo"].astype(str).to_numpy()[:, None])
    rows = []
    for f in FORMS:
        for pp in ("探索", "評価"):
            m = ((idx["形態"] == f) & (idx["per"] == pp)).values
            if m.sum() < 100:
                continue
            d = idx["date"].astype(str).to_numpy()[m]
            fv = np.argmin(np.where(np.isfinite(Oi2[m]) & (Oi2[m] > 0), Oi2[m], np.inf), 1)
            act = hitm[m][np.arange(m.sum()), fv].astype(float); exp = Pi2[m][np.arange(m.sum()), fv]
            rr, lo, hi = ratio_ci(d, act, exp)
            rows.append({"形態": f, "期間": pp, "組": "1番人気の3連単", "点数": int(m.sum()), "実際": act.mean(), "見込み": exp.mean(), "比": rr, "CI": fmt(lo, hi)})
            for lo_, hi_, lab in BANDS:
                sel = (FIRST == 1)[None, :] & (Oi2[m] >= lo_) & (Oi2[m] < hi_)
                act = (hitm[m] & sel).sum(1).astype(float); exp = np.where(sel, Pi2[m], 0).sum(1)
                rr, l2, h2 = ratio_ci(d, act, exp)
                rows.append({"形態": f, "期間": pp, "組": f"1-x-x {lab}", "点数": int(sel.sum()), "実際": act.sum() / max(sel.sum(), 1), "見込み": exp.sum() / max(sel.sum(), 1), "比": rr, "CI": fmt(l2, h2)})
    B3 = pd.DataFrame(rows)

    # ---- 3. 買い方
    def bets(g):
        out = {}
        gk = g[g["tan_boat"].notna()]
        out["1号艇の単勝"] = (gk["date"].values, np.full(len(gk), 100.0), np.where(gk["tan_boat"] == 1, gk["tan_pay"], 0).astype(float))
        out["1号艇の複勝"] = (gk["date"].values, np.full(len(gk), 100.0), gk["fuku1"].fillna(0).values.astype(float))
        out["3連単 1-x-x 全20点"] = (g["date"].values, np.full(len(g), 2000.0), np.where(g["win1"] == 1, g["tri_pay"], 0).astype(float))
        g2 = g[g["c2t"].notna()]
        out["2連単 1-x 全5点"] = (g2["date"].values, np.full(len(g2), 500.0), np.where(g2["c2t"].str[0] == "1", g2["pay2t"], 0).astype(float))
        gf = g[g["fav"].notna()]
        out["参考：確定1番人気の3連単1点"] = (gf["date"].values, np.full(len(gf), 100.0), np.where(gf["fav"] == gf["tri_combo"], gf["tri_pay"], 0).astype(float))
        return out
    rows = []
    for f in FORMS:
        for h in ("全体", "後半7〜12R"):
            for pp in ("探索", "評価", "2023〜2026-08"):
                g = G[(G["形態"] == f) & ((G["半分"] == h) | (h == "全体"))]
                g = g[g["per"] == pp] if pp != "2023〜2026-08" else g[g["date"] >= "20230101"]
                for nm, (d, st, rt) in bets(g).items():
                    roi, lo, hi = boot_roi(d, st, rt) if len(st) else (np.nan,) * 3
                    rows.append({"形態": f, "レース": h, "期間": pp, "買い方": nm, "レース数": len(st), "回収率": roi, "CI下": lo, "CI上": hi})
    C = pd.DataFrame(rows)
    ev = C[(C["形態"] == "ミッドナイト") & (C["期間"] == "評価") & ~C["買い方"].str.startswith("参考")]
    ex = C[(C["形態"] == "ミッドナイト") & (C["期間"] == "探索")]
    win = []
    for _, b in ev.iterrows():
        e1 = ex[(ex["買い方"] == b["買い方"]) & (ex["レース"] == b["レース"])]["回収率"]
        if b["レース数"] >= 300 and b["CI下"] > 1.0 and len(e1) and e1.iloc[0] > 1.0:
            win.append(f"{b['買い方']}（{b['レース']}）")
    judge = ("採用候補（締切前オッズの前向き検証へ）：" + "、".join(win)) if win else "不採用"
    days = G[G["形態"] == "ミッドナイト"].groupby(["jcd", G["date"].str[:4]])["date"].nunique().unstack(fill_value=0)
    show = lambda df: df.to_markdown(index=False, floatfmt=".3f")
    md = [f"# IDEA-073 ミッドナイトの日は1号艇に人気が偏りすぎているか（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}）", "",
          f"**判定：{judge}**", "",
          "- 形態：場・日の最終締切が 22:30 以降 → ミッドナイト、20:00〜22:29 → ナイター、それより前 → 昼。返還のあるレース・6艇そろわないレースは除く。",
          "- 比 = 実際 ÷ 市場の見込み（確定3連単オッズを正規化）。1 より上なら市場が低く見ている（買われ足りない）、1 より下なら買われすぎ。CI は日単位ブートストラップ95%。", "",
          "## ミッドナイトの日数（場 × 年）", "", days.reset_index().to_markdown(index=False), "",
          "## 1. オッズを使わない：1号艇の1着率・3着内率", "", show(A1), "",
          "### ミッドナイトのある場の形態別（2023〜）", "", show(A2) if len(A2) else "なし", "",
          "## 2. 1号艇の1着：実際 ÷ 市場の見込み", "", show(B1), "",
          "### 1号艇が勝ったときの2着の艇：実際 ÷ 見込み（市場の条件付き確率）", "", show(B2), "",
          "### 1番人気の3連単・1-x-x のオッズ帯別：実際の的中 ÷ 見込み", "", show(B3), "",
          "## 3. 買い方（K の実額。オッズで絞らない。参考の1番人気は確定オッズで選ぶので見かけになりうる）", "", show(C), ""]
    open(a.out, "w").write("\n".join(md)); print(judge); print("\n".join(md))

if __name__ == "__main__":
    main()
