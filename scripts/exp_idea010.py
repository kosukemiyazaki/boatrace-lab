"""IDEA-010: 市場の値付けの偏りを直接測る（docs/ideas.md の登録どおり。採用判定ではなく地図）
usage: python scripts/exp_idea010.py DATADIR STOREDIR [--out results/idea010.md]
- 2023-10〜2026-08 の全レース、締切時オッズの全組番を各100円買った回収率を区分ごとに出す（F/L/欠場艇を含む組番は返還）
- 区分: オッズ帯 / 人気順位帯 / 場 / 締切時刻帯 / レース種別 / グレード / 年四半期
- 候補の基準: 回収率 90% 以上 かつ 日単位ブートストラップ95%CIの下限 85% 超
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, parse_b
from exp_idea005 import grade_race, is_lower_race

START, END = "20231001", "20260831"
RNG = np.random.default_rng(0)
ODDS_BANDS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 300), (300, 1e9)]
RANK_BANDS = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 40), (41, 120)]

def roi_ci(days, st, rt, n_boot=3000):
    df = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum()
    s, r = df["st"].values, df["rt"].values
    if s.sum() == 0:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(s), size=(n_boot, len(s)))
    b = r[bi].sum(1) / np.clip(s[bi].sum(1), 1, None)
    return r.sum() / s.sum(), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def table(name, groups, days, stake, ret):
    """groups: dict 区分名 -> (レース×組番の bool マスク)。回収率を区分ごとに"""
    rows = []
    for g, m in groups.items():
        st = (m * 100.0).sum(1); rt = (ret * m).sum(1)
        roi, lo, hi = roi_ci(days, st, rt)
        rows.append({name: g, "点数": int(m.sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi,
                     "候補": "◯" if (roi >= 0.9 and lo > 0.85) else ""})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea010.md")
    a = ap.parse_args()
    key = ["date", "jcd", "rno"]
    races = pd.read_parquet(f"{a.datadir}/races.parquet").drop_duplicates(key)
    races = races[(races["date"] >= START) & (races["date"] <= END) & races["tri_combo"].notna()]
    odds = pd.read_parquet(f"{a.datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    e = pd.read_parquet(f"{a.datadir}/entries.parquet", columns=key + ["boat", "pos_raw", "deadline"])
    e = e[(e["date"] >= START) & (e["date"] <= END)]
    dl = e[e["boat"] == 1][key + ["deadline"]].drop_duplicates(key)
    r = races[key + ["tri_combo", "tri_pay", "race_name"]].merge(odds, on=key).merge(dl, on=key, how="left").reset_index(drop=True)
    O = np.array(r[COMBOS].to_numpy(float), copy=True); O[~(O > 0)] = np.nan
    ok = np.isfinite(O).sum(1) >= 60
    r, O = r[ok].reset_index(drop=True), O[ok]
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    rf = e.assign(rf=e["pos_raw"].astype(str).str.match(r"^(F|L|K)")).pivot_table(index=key, columns="boat", values="rf", aggfunc="max")
    rf = rf.reindex(pd.MultiIndex.from_frame(r[key])).fillna(False).to_numpy(bool)
    ref = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref[:, ci] |= rf[:, int(b) - 1]
    win = (np.arange(120)[None, :] == y[:, None]) & ~ref
    ret = np.where(win, r["tri_pay"].to_numpy(float)[:, None], 0.0) + np.where(ref, 100.0, 0.0)
    valid = np.isfinite(O)
    rank = (-np.where(valid, 1 / np.where(valid, O, 1), 0)).argsort(1, kind="stable").argsort(1) + 1
    days = r["date"].values
    out = [f"# IDEA-010 市場の値付けの偏り（締切時オッズ・全組番各100円、{START}〜{END}、{len(r)}レース・{len(set(days))}日）", "",
           "候補 = 回収率 90% 以上 かつ 日単位ブートストラップ95%CIの下限 85% 超。区分の総数は末尾に記す。", ""]
    n_seg = 0
    # 1. オッズ帯
    g = {f"{lo}〜{hi}倍" if hi < 1e9 else f"{lo}倍〜": valid & (O >= lo) & (O < hi) for lo, hi in ODDS_BANDS}
    t = table("オッズ帯", g, days, None, ret); n_seg += len(g)
    out += ["## 1. 締切時オッズ帯", "", t.to_markdown(index=False, floatfmt=".3f"), ""]
    # 2. 人気順位帯
    g = {f"{lo}番人気" if lo == hi else f"{lo}〜{hi}番人気": valid & (rank >= lo) & (rank <= hi) for lo, hi in RANK_BANDS}
    t = table("人気順位", g, days, None, ret); n_seg += len(g)
    out += ["## 2. 人気順位帯", "", t.to_markdown(index=False, floatfmt=".3f"), ""]
    # 3. 場別（全組番 と 30倍以下）
    rows = []
    for j in sorted(r["jcd"].unique()):
        m = (r["jcd"].values == j)[:, None]
        for lab, mm in (("全組番", valid & m), ("30倍以下", valid & (O <= 30) & m)):
            roi, lo, hi = roi_ci(days, (mm * 100.0).sum(1), (ret * mm).sum(1))
            rows.append({"場": j, "対象": lab, "レース": int(m.sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi, "候補": "◯" if (roi >= 0.9 and lo > 0.85) else ""})
            n_seg += 1
    out += ["## 3. 場別", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), ""]
    # 4. 締切時刻帯（番組表の締切時刻）
    hh = r["deadline"].fillna("12:00").str[:2].astype(int)
    hb = pd.cut(hh, [0, 11, 14, 17, 24], labels=["〜12時", "12〜15時", "15〜18時", "18時〜"]).astype(str)
    name4 = "締切時刻帯"
    g = {k: valid & (hb.values == k)[:, None] for k in sorted(hb.unique())}
    t = table(name4, g, days, None, ret); n_seg += len(g)
    out += [f"## 4. {name4}", "", t.to_markdown(index=False, floatfmt=".3f"), ""]
    # 5. レース種別・グレード
    lower = r["race_name"].map(is_lower_race).values
    g = {"予選・一般": valid & lower[:, None], "上位戦": valid & ~lower[:, None]}
    gr = grade_race(r, f"{a.store}/grades.csv")
    g.update({"一般戦（グレードなし）": valid & ~gr[:, None], "グレード戦": valid & gr[:, None]})
    t = table("種別", g, days, None, ret); n_seg += len(g)
    out += ["## 5. レース種別・グレード", "", t.to_markdown(index=False, floatfmt=".3f"), ""]
    # 6. 四半期ごとの オッズ帯回収率（安定性）
    q = pd.to_datetime(r["date"]).dt.to_period("Q").astype(str).values
    rows = []
    for qq in sorted(set(q)):
        m = (q == qq)[:, None]
        row = {"四半期": qq, "レース": int(m.sum())}
        for lo, hi in ODDS_BANDS:
            mm = valid & (O >= lo) & (O < hi) & m
            s = (mm * 100.0).sum(); row[f"{lo}〜{hi}倍" if hi < 1e9 else f"{lo}倍〜"] = (ret * mm).sum() / s if s else np.nan
        rows.append(row)
    out += ["## 6. 四半期ごとのオッズ帯別回収率（安定性）", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), ""]
    out += [f"区分の総数：{n_seg}（表1〜5）。候補（◯）の数と、どの区分かを docs/ideas.md に記録する。"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    main()
