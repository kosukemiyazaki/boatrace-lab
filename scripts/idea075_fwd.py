"""IDEA-075 前向きの記録：条件が重なる1号艇の単勝・複勝（docs/ideas.md の登録どおり。判定は 2026-10-25 以降）
usage: python scripts/idea075_fwd.py DATADIR STOREDIR [--start 20261025] [--out results/idea075.md]
- 出走表は DATADIR の entries（B・K から作ったもの）、F持ち・節の日目もそこから（idea075_lib.race_table と同じ計算）
- (b) 市場の1号艇の確率は締切2分前（live_t1）の3連単（逆数の正規化）、重みは models/idea056_weights.json
- (d) は結果ファイル（K）の風向き（レース時）。参考に直前情報（live の beforeinfo）の風向きから推定した (d) でも集計（判定には使わない）
- 買い方：(1) 1号艇の単勝100円、(2) 1号艇の複勝100円、(3) 締切2分前の単勝 ≥ 1／補正後の1号艇の確率 のときだけ単勝100円。払戻は K の実額、1号艇が欠場・F・L は返還
- 判定：300レースの時点で (1)〜(3) のどれかが 回収率100%以上・日単位95%CI下限 > 100% → 採用。届かなければ記録して終える（最長 2027-04-30）
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
        m["d"] = [(j, w) in fav for j, w in zip(m["jcd"], m["wind_dir"])]
        b2 = m[KEY].merge(norm(bi)[KEY + ["wind_dir"]] if len(bi) else pd.DataFrame(columns=KEY + ["wind_dir"]), on=KEY, how="left")
        m["d_est"] = [(j, wind_from_code(Z, j, c)) in fav for j, c in zip(m["jcd"], b2["wind_dir"])]
        m["n"] = m["a"].astype(int) + m["b"].astype(int) + m["c"].astype(int) + m["d"].astype(int)
        m["n_est"] = m["a"].astype(int) + m["b"].astype(int) + m["c"].astype(int) + m["d_est"].astype(int)
        t1 = norm(tf)[KEY + ["tan1"]] if len(tf) else pd.DataFrame(columns=KEY + ["tan1"])
        m = m.merge(t1.rename(columns={"tan1": "tan1_t1"}), on=KEY, how="left")
        rows.append(m[KEY + ["a", "b", "c", "d", "d_est", "n", "n_est", "p1", "ratio", "padj", "tan1_t1"]])
    R = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEY + ["n", "n_est"])
    tan = kp[(kp["kind"] == "tan") & (kp["combo"] == "1")].set_index(KEY)["pay"]
    fuku = kp[(kp["kind"] == "fuku") & (kp["combo"] == "1")].set_index(KEY)["pay"]
    b1 = eall[eall["boat"] == 1][KEY + ["pos_raw"]].drop_duplicates(KEY).set_index(KEY)["pos_raw"].astype(str)
    def pays(df):
        ks = list(map(tuple, df[KEY].values))
        ref = np.array([b1.get(k, "")[:1] in ("K", "F", "L") for k in ks], bool)
        t = np.where(ref, 100, [tan.get(k, 0) for k in ks]).astype(float); f = np.where(ref, 100, [fuku.get(k, 0) for k in ks]).astype(float)
        return t, f, ref
    def summarize(df, label):
        out = []
        if not len(df):
            return pd.DataFrame([{"区分": label, "買い方": x, "点数": 0} for x in ("(1) 単勝", "(2) 複勝", "(3) 単勝（損益分岐以上のとき）")]), 0
        t, f, ref = pays(df); d = df["date"].values
        be = pd.to_numeric(df["tan1_t1"], errors="coerce").values >= 1 / df["padj"].values
        for name, msk, pay in (("(1) 単勝", np.ones(len(df), bool), t), ("(2) 複勝", np.ones(len(df), bool), f), ("(3) 単勝（損益分岐以上のとき）", be, t)):
            n = int(msk.sum())
            roi, lo, hi = boot(d[msk], np.full(n, 100.0), pay[msk]) if n else (np.nan, np.nan, np.nan)
            out.append({"区分": label, "買い方": name, "点数": n, "日数": int(pd.Series(d[msk]).nunique()), "的中": int(((pay[msk] > 0) & ~ref[msk]).sum()),
                        "回収率": roi, "95%CI下": lo, "95%CI上": hi})
        return pd.DataFrame(out), int(len(df))
    S1, n1 = summarize(R[R["n"] >= 2] if len(R) else R, "判定（(d) はレース時の風）")
    S2, _ = summarize(R[R["n_est"] >= 2] if len(R) else R, "参考（(d) は直前情報の風から推定）")
    ok = any((x["点数"] >= 300) and (x["回収率"] >= 1.0) and (x["95%CI下"] > 1.0) for _, x in S1.iterrows())
    status = ("採用" if ok else "不採用（300レース時点で基準に届かない）") if n1 >= 300 else f"集計中（{n1}／300レース）"
    md = [f"# IDEA-075 前向きの記録（{a.start} 以降）", "", f"**状態：{status}**", "",
          "条件 (a) 22歳以下、(b) 補正後÷市場 ≥ 1.15（締切2分前の3連単）、(c) 逃げ上手（IDEA-060 上位10%）、(d) 1コースに有利な風のうち2つ以上。払戻は K の実額（1号艇の欠場・F・L は返還）。", "",
          pd.concat([S1, S2]).to_markdown(index=False, floatfmt=".3f"), "",
          f"対象にしたレース：{n1}（締切2分前のオッズと K がそろうレース {len(R)} のうち）", ""]
    if len(R) and n1:
        T = R[R["n"] >= 2].sort_values(KEY, ascending=False).head(60)
        t, f, _ = pays(T); T = T.assign(単勝1=t, 複勝1=f)
        md += ["## 対象レース（新しい順）", "", T[KEY + ["a", "b", "c", "d", "p1", "ratio", "tan1_t1", "単勝1", "複勝1"]].to_markdown(index=False, floatfmt=".3f"), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:12]))

if __name__ == "__main__":
    main()
