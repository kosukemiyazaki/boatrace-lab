"""IDEA-051 実は荒れにくい区分（市場の見込みより1号艇・本命が強いところ）を一通り洗う（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea051.py DATADIR STOREDIR [--hist /tmp/claude-0/ds_hist] [--out results/idea051.md]
- 区分：レース番号、締切時刻の帯、場、グレード（store/grades.csv）、1号艇の級別・年齢帯・F持ち、2号艇のF持ち、節の日目、当地−全国勝率の3分位、1号艇の確率帯、
  掛け合わせ「レース番号×時刻帯」「級別×F持ち」
- 探索 2023-11〜2025-12：(1) 1号艇の1着率÷見込み（日単位CI）、(2) 1番人気1点、(3) 1号艇1着で π×オッズ>0.95（π=P^1.06）。評価 2026-01〜08
- 参考：2014〜2026 の「1号艇1着を全部」の年別回収率（オッズなし）
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm, f_status

KEY = ["date", "jcd", "rno"]; TR = ("20231101", "20251231"); EV = ("20260101", "20260831")
RNG = np.random.default_rng(0)
FIRST = np.array([int(c[0]) for c in COMBOS])

def boot_ratio(days, num, den):
    g = pd.DataFrame({"d": days, "n": num, "e": den}).groupby("d").sum()
    if len(g) < 2 or g["e"].sum() <= 0:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(3000, len(g))); b = g["n"].values[bi].sum(1) / g["e"].values[bi].sum(1)
    return float(g["n"].sum() / g["e"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def band_time(dl):
    h = pd.to_numeric(dl.astype(str).str[:2], errors="coerce")
    return pd.cut(h, [-1, 11, 14, 17, 20, 30], labels=["〜12時", "〜15時", "〜18時", "〜21時", "21時〜"]).astype(str)

def meet_day(df):
    """場ごとに連続開催日の何日目か（間が空いたら1に戻す、7以上は7）"""
    d = df[["date", "jcd"]].drop_duplicates().copy(); d["dt"] = pd.to_datetime(d["date"]); d = d.sort_values(["jcd", "dt"])
    gap = d.groupby("jcd")["dt"].diff().dt.days.fillna(99) > 1
    d["day"] = gap.groupby(d["jcd"]).cumsum(); d["day"] = d.groupby(["jcd", "day"]).cumcount() + 1
    d["day"] = d["day"].clip(upper=7).astype(str)
    return df.merge(d[["date", "jcd", "day"]], on=["date", "jcd"], how="left")["day"].values

def grade_map(store, df):
    g = pd.read_csv(f"{store}/grades.csv", dtype=str)
    g["g"] = g["grade"].str.replace(r"[ab]$", "", regex=True).replace({"ippan": "一般"})
    out = pd.Series("一般", index=df.index)
    for r in g.itertuples():
        m = (df["jcd"] == r.jcd.zfill(2)) & (df["date"] >= r.start) & (df["date"] <= r.end)
        out[m] = r.g
    return out.values

def boat_feats(e, boats=(1, 2)):
    fs = f_status(e); ef = e.merge(fs, on=["toban", "date"], how="left")
    out = None
    for b in boats:
        x = ef[ef["boat"] == b][KEY + ["cls", "age", "nat_win", "loc_win", "deadline", "fh"]].drop_duplicates(KEY)
        x = x.rename(columns={c: f"{c}{b}" for c in ["cls", "age", "nat_win", "loc_win", "deadline", "fh"]})
        out = x if out is None else out.merge(x, on=KEY, how="outer")
    return out

def segments(m, tr_mask=None, with_odds=True, store=None):
    S = {}
    S["レース番号"] = m["rno"].astype(int).astype(str).values
    S["締切時刻の帯"] = band_time(m["deadline1"])
    S["場"] = m["jcd"].values
    if store is not None:
        S["グレード"] = grade_map(store, m)
    S["1号艇の級別"] = m["cls1"].astype(str).values
    S["1号艇の年齢帯"] = pd.cut(pd.to_numeric(m["age1"], errors="coerce"), [0, 29, 39, 49, 99], labels=["〜29", "30〜39", "40〜49", "50〜"]).astype(str)
    S["1号艇のF持ち"] = np.where(m["fh1"].fillna(False).astype(bool), "F持ち", "なし")
    S["2号艇のF持ち"] = np.where(m["fh2"].fillna(False).astype(bool), "F持ち", "なし")
    S["節の日目"] = meet_day(m)
    diff = pd.to_numeric(m["loc_win1"], errors="coerce") - pd.to_numeric(m["nat_win1"], errors="coerce")
    ref = diff[tr_mask] if tr_mask is not None else diff
    q = np.nanquantile(ref, [1 / 3, 2 / 3]); S["当地−全国勝率（1号艇）"] = np.where(diff.isna(), "nan", np.array(["低", "中", "高"])[np.digitize(diff.fillna(0), q)])
    if with_odds:
        S["1号艇の確率帯"] = np.array(["〜40", "40〜60", "60〜"])[np.digitize(m["p1"].values, [0.4, 0.6])]
    S["レース番号×時刻帯"] = np.char.add(np.char.add(S["レース番号"].astype(str), "R・"), S["締切時刻の帯"].astype(str))
    S["級別×F持ち"] = np.char.add(np.char.add(S["1号艇の級別"].astype(str), "・"), S["1号艇のF持ち"].astype(str))
    return S

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--hist", default="/tmp/claude-0/ds_hist")
    ap.add_argument("--out", default="results/idea051.md"); a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo", "tri_pay"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]))
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(boat_feats(e), on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int); n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); m["p1"] = P[:, FIRST == 1].sum(1)
    win1 = (FIRST[y] == 1).astype(float); hit = np.zeros_like(v); hit[np.arange(n), y] = True
    days = m["date"].values; tr = (days >= TR[0]) & (days <= TR[1]); ev = days >= EV[0]
    fav = np.argmin(np.where(v, O, np.inf), 1); rtA = np.where(hit[np.arange(n), fav], O[np.arange(n), fav] * 100, 0); stA = np.full(n, 100.0)
    pi = P ** 1.06; pi /= pi.sum(1, keepdims=True); bC = v & (FIRST[None, :] == 1) & (pi * np.where(v, O, 0) > 0.95)
    stC = bC.sum(1) * 100.0; rtC = np.where(bC & hit, O * 100, 0).sum(1)
    S = segments(m, tr, True, a.store)
    rows = []
    for sname, sv in S.items():
        for lab in pd.unique(sv):
            if lab in ("nan", "None"):
                continue
            row = {"区分": sname, "値": lab}
            for per, msk in (("探索", tr), ("評価", ev)):
                s = msk & (sv == lab); row[f"{per} レース"] = int(s.sum())
                if s.sum() < 30:
                    continue
                ratio, lo, hi = boot_ratio(days[s], win1[s], m["p1"].values[s])
                row[f"{per} 1号艇 実際÷見込み"] = ratio; row[f"{per} CI下"] = lo; row[f"{per} CI上"] = hi
                row[f"{per} 1番人気1点"] = rtA[s].sum() / stA[s].sum()
                row[f"{per} π×オッズ>0.95 点数"] = int(bC[s].sum()); row[f"{per} π×オッズ>0.95"] = rtC[s].sum() / stC[s].sum() if stC[s].sum() > 0 else np.nan
                if per == "評価":
                    g = pd.DataFrame({"d": days[s], "st": stA[s], "rt": rtA[s]}).groupby("d").sum()
                    bi = RNG.integers(0, len(g), size=(3000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
                    row["評価 1番人気 CI下"] = float(np.quantile(b, .025))
                    g = pd.DataFrame({"d": days[s], "st": stC[s], "rt": rtC[s]}).groupby("d").sum(); g = g[g["st"] > 0]
                    if len(g) >= 2:
                        bi = RNG.integers(0, len(g), size=(3000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
                        row["評価 π CI下"] = float(np.quantile(b, .025))
            rows.append(row)
    R = pd.DataFrame(rows)
    sel = R[(R["探索 レース"] >= 500) & (R["探索 1号艇 実際÷見込み"] >= 1.06) & (R["探索 CI下"] > 1)].copy()
    def passes(x):
        okA = x["評価 レース"] >= 300 and x["評価 1番人気1点"] >= 1.1 and x.get("評価 1番人気 CI下", 0) > 1.0
        okC = x["評価 レース"] >= 300 and x.get("評価 π×オッズ>0.95", 0) >= 1.1 and x.get("評価 π CI下", 0) > 1.0 and x.get("評価 π×オッズ>0.95 点数", 0) >= 300
        return bool(okA or okC)
    sel["採用"] = [passes(x) for _, x in sel.iterrows()] if len(sel) else []
    judge = "採用候補（前向きの検証へ）" if len(sel) and sel["採用"].any() else "不採用"
    keep = R[(R["評価 レース"] >= 300) & (R["評価 CI下"] > 1)].sort_values("評価 1号艇 実際÷見込み", ascending=False)
    # 参考：2014〜2026 の 1号艇1着を全部（年別）
    hist_md = ""
    if os.path.isdir(a.hist):
        rh = norm(pd.read_parquet(f"{a.hist}/races.parquet"))[KEY + ["tri_combo", "tri_pay"]]
        eh = norm(pd.read_parquet(f"{a.hist}/entries.parquet", columns=KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]))
        ra = pd.concat([rh, r[r["date"] <= EV[1]]]).dropna(subset=["tri_combo"]).drop_duplicates(KEY)
        ea = pd.concat([eh, e]); bad = ea[ea["pos_raw"].astype(str).str.match(r"^[KFL]")][KEY].drop_duplicates()
        ra = ra.merge(bad.assign(bad=1), on=KEY, how="left"); ra = ra[ra["bad"].isna()].drop(columns="bad")
        ra = ra.merge(boat_feats(ea), on=KEY, how="left").reset_index(drop=True)
        ra = ra[ra["tri_combo"].isin(COMBOS)].reset_index(drop=True)
        w1 = ra["tri_combo"].str[0] == "1"; pay = np.where(w1, pd.to_numeric(ra["tri_pay"], errors="coerce").fillna(0), 0.0)
        year = ra["date"].str[:4].values; SH = segments(ra, (ra["date"] >= TR[0]).values & (ra["date"] <= TR[1]).values, False, None)
        base = pd.Series(pay).groupby(year).sum() / (pd.Series(np.ones(len(ra))).groupby(year).sum() * 2000)
        hrows = []
        for sname, sv in SH.items():
            for lab in pd.unique(sv):
                if lab in ("nan", "None"):
                    continue
                s = sv == lab
                if s.sum() < 2000:
                    continue
                g = pd.DataFrame({"y": year[s], "pay": pay[s]}).groupby("y").agg(pay=("pay", "sum"), n=("pay", "size")); g["roi"] = g["pay"] / (g["n"] * 2000)
                g = g[g["n"] >= 200]
                hrows.append({"区分": sname, "値": lab, "レース": int(s.sum()), "回収率（全期間）": pay[s].sum() / (s.sum() * 2000), "年数": int(len(g)),
                              "基準以上の年数": int((g["roi"] >= base.reindex(g.index).values).sum()), "最低": g["roi"].min(), "最高": g["roi"].max()})
        H = pd.DataFrame(hrows).sort_values("基準以上の年数", ascending=False)
        hist_md = "\n".join(["## 参考：2014〜2026 の「1号艇1着を全部」の年別回収率（オッズなし、返還のあるレースを除く、2,000レース以上の区分）", "",
                             "全レースの年別回収率（基準）：" + "、".join(f"{k} {v:.2f}" for k, v in base.items()), "",
                             H.head(25).to_markdown(index=False, floatfmt=".3f"), "", "全区分は results/idea051_hist.md"])
        open(a.out.replace(".md", "_hist.md"), "w", encoding="utf-8").write(H.to_markdown(index=False, floatfmt=".3f") + "\n")
    md = [f"# IDEA-051 荒れにくい区分を一通り洗う（探索 {TR[0]}〜{TR[1]}／評価 {EV[0]}〜{EV[1]}、{n}レース）", "",
          f"**判定：{judge}**", "",
          f"探索で選ばれた区分（レース数500以上・1号艇 実際÷見込み 1.06以上・CI下限>1）：{len(sel)}個", "",
          sel.to_markdown(index=False, floatfmt=".3f") if len(sel) else "（なし）", "",
          "## 評価でも 1号艇 実際÷見込み が有意に 1 を超える区分（300レース以上、記録用）", "",
          keep[["区分", "値", "評価 レース", "評価 1号艇 実際÷見込み", "評価 CI下", "評価 CI上", "評価 1番人気1点", "評価 π×オッズ>0.95"]].to_markdown(index=False, floatfmt=".3f") if len(keep) else "（なし）", "",
          "## 全区分", "", R.to_markdown(index=False, floatfmt=".3f"), "", hist_md]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:8])); print(keep[["区分", "値", "評価 レース", "評価 1号艇 実際÷見込み", "評価 CI下", "評価 1番人気1点"]].to_string(index=False) if len(keep) else "keep: none")
    print(hist_md[:3000])

if __name__ == "__main__":
    main()
