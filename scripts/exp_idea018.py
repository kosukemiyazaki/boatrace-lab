"""IDEA-018: 組番の番号パターンの偏り（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea018.py DATADIR [--out results/idea018.md]
- 2023-10〜2026-08、締切時オッズのある全レースで、120通りそれぞれを各100円買った回収率と日単位ブートストラップ95%CI
- 1番人気（締切時3連単で最も人気の組番の1着艇）が1号艇か否かで2分割したものも出す
- 候補: 回収率 95% 以上 かつ CI 下限 85% 超。120通り×3（全体・1号艇人気・それ以外）を試すので、偶然の候補の期待数を併記
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

START, END = "20231001", "20260831"
RNG = np.random.default_rng(0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea018.md")
    a = ap.parse_args()
    key = ["date", "jcd", "rno"]
    races = pd.read_parquet(f"{a.datadir}/races.parquet").drop_duplicates(key)
    races = races[(races["date"] >= START) & (races["date"] <= END) & races["tri_combo"].notna()]
    odds = pd.read_parquet(f"{a.datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    e = pd.read_parquet(f"{a.datadir}/entries.parquet", columns=key + ["boat", "pos_raw"])
    e = e[(e["date"] >= START) & (e["date"] <= END)]
    r = races[key + ["tri_combo", "tri_pay"]].merge(odds, on=key).reset_index(drop=True)
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
    valid = np.isfinite(O)
    win = (np.arange(120)[None, :] == y[:, None]) & ~ref
    ret = np.where(win, r["tri_pay"].to_numpy(float)[:, None], 0.0) + np.where(ref, 100.0, 0.0)
    ret = np.where(valid, ret, 0.0); stake = valid * 100.0
    fav = np.nanargmin(np.where(valid, O, np.inf), axis=1)
    fav1 = np.array([COMBOS[i][0] == "1" for i in fav])
    days = r["date"].values
    ud, di = np.unique(days, return_inverse=True)
    nd = len(ud)
    bi = RNG.integers(0, nd, size=(2000, nd))
    rows = []
    for seg, m in (("全体", np.ones(len(r), bool)), ("1号艇が人気", fav1), ("1号艇以外が人気", ~fav1)):
        S = np.zeros((nd, 120)); R = np.zeros((nd, 120))
        np.add.at(S, di[m], stake[m]); np.add.at(R, di[m], ret[m])
        bs = S[bi].sum(1); br = R[bi].sum(1)
        boot = br / np.clip(bs, 1, None)
        roi = R.sum(0) / np.clip(S.sum(0), 1, None)
        lo, hi = np.quantile(boot, .025, axis=0), np.quantile(boot, .975, axis=0)
        mo = np.nanmedian(np.where(valid[m], O[m], np.nan), axis=0)
        for ci, c in enumerate(COMBOS):
            rows.append(dict(区分=seg, 組番=c, 点数=int(S[:, ci].sum() / 100), 回収率=roi[ci], CI下=lo[ci], CI上=hi[ci],
                             オッズ中央値=mo[ci], 候補="◯" if (roi[ci] >= 0.95 and lo[ci] > 0.85) else ""))
    df = pd.DataFrame(rows)
    n_cand = int((df["候補"] == "◯").sum())
    # 偶然の候補の期待数: 各組番の回収率の分布を「全体の回収率」に置き換えたときに基準を超える割合で概算
    out = [f"# IDEA-018 組番の番号パターンの偏り（締切時オッズ・各100円、{START}〜{END}、{len(r)}レース・{nd}日）", "",
           f"候補（回収率95%以上かつ95%CI下限85%超）: **{n_cand} 件** / 試した数 {len(df)}（120通り×3区分）", ""]
    for seg in ("全体", "1号艇が人気", "1号艇以外が人気"):
        d = df[df["区分"] == seg].sort_values("回収率", ascending=False)
        out += [f"## {seg}：回収率の上位20と下位10", "", d.head(20).to_markdown(index=False, floatfmt=".3f"), "",
                d.tail(10).to_markdown(index=False, floatfmt=".3f"), ""]
    # 1着艇ごとの回収率（番号パターンの粗い見方）
    df["1着"] = df["組番"].str[0]
    agg = df[df["区分"] == "全体"].groupby("1着").apply(lambda x: pd.Series({"組番数": len(x), "回収率の平均": x["回収率"].mean(),
                                                                       "最高": x["回収率"].max(), "最低": x["回収率"].min()}))
    out += ["## 1着の艇番ごと（全体）", "", agg.to_markdown(floatfmt=".3f")]
    df.drop(columns=["1着"]).to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    main()
