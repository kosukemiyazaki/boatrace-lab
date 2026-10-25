"""IDEA-075 前向きの記録：条件が重なる1号艇の単勝・複勝（docs/ideas.md の登録どおり。判定は 2026-10-25 以降）
usage: python scripts/idea075_fwd.py DATADIR STOREDIR [--start 20261025] [--out results/idea075.md]
- 出走表は DATADIR の entries（B・K から作ったもの）、F持ち・節の日目もそこから（idea075_lib.race_table と同じ計算）
- (b) 市場の1号艇の確率は締切2分前（live_t1）の3連単（逆数の正規化）、重みは models/idea056_weights.json
- (d) は直前情報（live の beforeinfo、締切6分前ごろ）の風向き（models/idea075_frozen.json の場ごとの回転の対応表）で判定（2026-10-25 改定：
  買う時点で見える値にそろえた）。参考に結果ファイル（K）のレース時の風での集計も並べる（判定には使わない）
- 買い方：(1) 1号艇の単勝100円、(2) 1号艇の複勝100円、(3) 締切2分前の単勝 ≥ 1／補正後の1号艇の確率 のときだけ単勝100円。払戻は K の実額、1号艇が欠場・F・L は返還
- 判定（2026-10-25 改定、前向きのレースを1つも数える前に決めた）：
  途中の確認は 100 レースごと。その時点の (1)〜(3) すべてで、回収率の片側80%上限（日単位ブートストラップ）が 100% 未満なら「不採用（見込みなし）」で打ち切る。
  最終は 1,000 レースか 2027-01-31 の早い方。(1)〜(3) のどれかが 回収率110%以上・日単位95%CI下限 > 100% なら採用、そうでなければ不採用。
  あわせて、事前の見込みを「回収率 95%・ばらつき 5%」に置いたベイズの見方で「本当の回収率が100%を超える確率」を毎回出す（判定には使わない）
"""
import argparse, os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY, FIRST
from exp_idea053 import boot
from idea039_stage2 import load
from idea052 import k_pay
from idea075_lib import ECOLS, load_fixed, race_table, ratio, wind_from_code
from scipy.stats import norm as N

RNG = np.random.default_rng(75)
LOOK, NMAX, END = 100, 1000, "20270131"
PRIOR_M, PRIOR_SD = 0.95, 0.05

def bstat(days, pay):
    """日単位ブートストラップ：回収率、95%CI、片側80%上限、標準誤差"""
    g = pd.DataFrame({"d": days, "st": 100.0, "rt": pay}).groupby("d").sum()
    if len(g) < 2:
        roi = g["rt"].sum() / g["st"].sum() if len(g) else np.nan
        return roi, np.nan, np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(g["rt"].sum() / g["st"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975)), float(np.quantile(b, .8)), float(b.std())

def posterior_gt1(roi, se):
    if not np.isfinite(roi) or not np.isfinite(se) or se <= 0:
        return np.nan
    w0, w1 = 1 / PRIOR_SD ** 2, 1 / se ** 2
    m = (PRIOR_M * w0 + roi * w1) / (w0 + w1); sd = (w0 + w1) ** -0.5
    return float(1 - N.cdf((1 - m) / sd))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--start", default="20261025")
    ap.add_argument("--out", default="results/idea075.md"); a = ap.parse_args()
    W, Z = load_fixed()
    kp = k_pay(a.store, a.start); kp = kp[kp["kind"].isin(["tan", "fuku"])]
    haveK = kp[kp["kind"] == "tan"][KEY].drop_duplicates()
    o3 = load(a.store, "live_t1", "odds3t", a.start); tf = load(a.store, "live_t1", "oddstf", a.start); bi = load(a.store, "live", "beforeinfo", a.start)
    eall = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=ECOLS))
    races = norm(pd.read_parquet(f"{a.datadir}/races.parquet", columns=KEY + ["wind_dir"])).drop_duplicates(KEY)
    dates = sorted(set(haveK["date"]) & set(o3["date"] if len(o3) else []))
    rows = []
    for date in dates:
        prog = eall[eall["date"] == date]
        if not len(prog):
            continue
        hist = eall[(eall["date"] < date) & (eall["date"] >= (pd.Timestamp(date) - pd.Timedelta(days=200)).strftime("%Y%m%d"))]
        m = race_table(prog, hist, a.store, W, Z)
        m = m.merge(norm(o3)[KEY + COMBOS], on=KEY).merge(haveK, on=KEY).merge(races, on=KEY, how="left")
        if not len(m):
            continue
        O = m[COMBOS].apply(pd.to_numeric, errors="coerce").to_numpy(float); v = np.isfinite(O) & (O > 0)
        P = np.where(v, 1 / np.where(v, O, 1), 0); P /= np.where(P.sum(1, keepdims=True) > 0, P.sum(1, keepdims=True), 1)
        m["p1"] = P[:, FIRST == 1].sum(1); m = m[(v.sum(1) >= 40) & (m["p1"] > 0) & (m["p1"] < 1)].reset_index(drop=True)
        r, padj = ratio(m["p1"].values, m["s"].values); m["ratio"] = r; m["padj"] = padj
        fav = set(map(tuple, Z["d_fav"]))
        m["b"] = m["ratio"] >= 1.15
        m["d_k"] = [(j, w) in fav for j, w in zip(m["jcd"], m["wind_dir"])]
        bil = norm(bi).sort_values("fetched_at").drop_duplicates(KEY, keep="last")[KEY + ["wind_dir"]] if len(bi) else pd.DataFrame(columns=KEY + ["wind_dir"])
        b2 = m[KEY].merge(bil, on=KEY, how="left")
        m["d"] = [(j, wind_from_code(Z, j, c)) in fav for j, c in zip(m["jcd"], b2["wind_dir"])]
        m["n"] = m["a"].astype(int) + m["b"].astype(int) + m["c"].astype(int) + m["d"].astype(int)
        m["n_k"] = m["a"].astype(int) + m["b"].astype(int) + m["c"].astype(int) + m["d_k"].astype(int)
        t1 = norm(tf)[KEY + ["tan1"]] if len(tf) else pd.DataFrame(columns=KEY + ["tan1"])
        m = m.merge(t1.rename(columns={"tan1": "tan1_t1"}), on=KEY, how="left")
        rows.append(m[KEY + ["a", "b", "c", "d", "d_k", "n", "n_k", "p1", "ratio", "padj", "tan1_t1"]])
    R = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEY + ["n", "n_k"])
    tan = kp[(kp["kind"] == "tan") & (kp["combo"] == "1")].set_index(KEY)["pay"]
    fuku = kp[(kp["kind"] == "fuku") & (kp["combo"] == "1")].set_index(KEY)["pay"]
    b1 = eall[eall["boat"] == 1][KEY + ["pos_raw"]].drop_duplicates(KEY).set_index(KEY)["pos_raw"].astype(str)
    def pays(df):
        ks = list(map(tuple, df[KEY].values))
        ref = np.array([b1.get(k, "")[:1] in ("K", "F", "L") for k in ks], bool)
        t = np.where(ref, 100, [tan.get(k, 0) for k in ks]).astype(float); f = np.where(ref, 100, [fuku.get(k, 0) for k in ks]).astype(float)
        return t, f, ref
    BETS = ("(1) 単勝", "(2) 複勝", "(3) 単勝（損益分岐以上のとき）")
    def bets(df):
        t, f, ref = pays(df)
        be = pd.to_numeric(df["tan1_t1"], errors="coerce").values >= 1 / df["padj"].values
        return {"(1) 単勝": (np.ones(len(df), bool), t), "(2) 複勝": (np.ones(len(df), bool), f), "(3) 単勝（損益分岐以上のとき）": (be, t)}, ref
    def summarize(df, label):
        if not len(df):
            return pd.DataFrame([{"区分": label, "買い方": x, "点数": 0} for x in BETS])
        B, ref = bets(df); d = df["date"].values; out = []
        for name, (msk, pay) in B.items():
            n = int(msk.sum())
            roi, lo, hi, up80, se = bstat(d[msk], pay[msk]) if n else (np.nan,) * 5
            out.append({"区分": label, "買い方": name, "点数": n, "日数": int(pd.Series(d[msk]).nunique()), "的中": int(((pay[msk] > 0) & ~ref[msk]).sum()),
                        "回収率": roi, "95%CI下": lo, "95%CI上": hi, "片側80%上限": up80, "100%超の確率（事前95%）": posterior_gt1(roi, se)})
        return pd.DataFrame(out)
    T = R[R["n"] >= 2].sort_values(KEY).reset_index(drop=True) if len(R) else R
    n1 = int(len(T)); last = T["date"].max() if n1 else ""
    looks, stopped = [], None
    for k in range(LOOK, min(n1, NMAX) + 1, LOOK):
        S = summarize(T.iloc[:k], f"{k}レース時点")
        fut = bool((S["片側80%上限"] < 1.0).all())
        looks.append((k, S, fut))
        if fut and stopped is None:
            stopped = k
    S1 = summarize(T, "判定（(d) は直前情報の風）")
    S2 = summarize(R[R["n_k"] >= 2].sort_values(KEY) if len(R) else R, "参考（(d) はレース時の風、判定に使わない）")
    final = n1 >= NMAX or (last and last > END)
    if stopped is not None:
        status = f"不採用（{stopped}レース時点の途中確認で、3つの買い方すべての片側80%上限が100%未満＝見込みなし）"
    elif final:
        ok = any((x["回収率"] >= 1.10) and (x["95%CI下"] > 1.0) for _, x in S1.iterrows())
        status = "採用（利用者に知らせる）" if ok else "不採用（最終判定で基準に届かない）"
    else:
        status = f"集計中（{n1}／{NMAX}レース、次の途中確認は {(n1 // LOOK + 1) * LOOK} レース）"
    md = [f"# IDEA-075 前向きの記録（{a.start} 以降）", "", f"**状態：{status}**", "",
          "条件 (a) 22歳以下、(b) 補正後÷市場 ≥ 1.15（締切2分前の3連単）、(c) 逃げ上手（IDEA-060 上位10%）、(d) 1コースに有利な風（直前情報の風向き）のうち2つ以上。払戻は K の実額（1号艇の欠場・F・L は返還）。",
          f"判定：{LOOK}レースごとに途中確認（3つの買い方すべての片側80%上限が100%未満なら打ち切り）、最終は {NMAX}レースか {END} の早い方で 回収率110%以上・95%CI下限 > 100%。"
          f"「100%超の確率」は事前の見込みを回収率{PRIOR_M:.0%}・ばらつき{PRIOR_SD:.0%}に置いたベイズの見方（判定には使わない）。", "",
          pd.concat([S1, S2]).to_markdown(index=False, floatfmt=".3f"), "",
          f"対象にしたレース：{n1}（締切2分前のオッズと K がそろうレース {len(R)} のうち）", ""]
    if looks:
        md += ["## 途中確認", ""] + [f"- {k}レース時点：{'打ち切りの基準に当たる' if fut else '続ける'}" for k, _, fut in looks] + [""]
    if n1:
        V = T.sort_values(KEY, ascending=False).head(60)
        t, f, _ = pays(V); V = V.assign(単勝1=t, 複勝1=f)
        md += ["## 対象レース（新しい順）", "", V[KEY + ["a", "b", "c", "d", "d_k", "p1", "ratio", "tan1_t1", "単勝1", "複勝1"]].to_markdown(index=False, floatfmt=".3f"), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:12]))

if __name__ == "__main__":
    main()
