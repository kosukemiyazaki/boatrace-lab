"""IDEA-064 新モーターの使い始め：2連対率があてにならない時期のずれ（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea064.py DATADIR HISTDS BKDIR [--out results/idea064.md]
- モーターの入れ替え日：場ごとに、その日の出走表のモーター2連対率の中央値が 1 以下で、前の開催日の中央値が 10 より大きい日
  （新モーターの最初の節は全モーターの2連対率が 0 で表示される）。前の入れ替えから60日以内に見つかった日は同じ入れ替えとみなして無視。
- 区分（すべて当日までのデータ）：入れ替えからの経過日数（〜30日／31〜90日／91日〜）、その時点のモーター2連対率のその日・その場の中での3分位
  （全モーター 0 の節は「数字なし」）、入れ替え後のそのモーターの当日より前の出走数（0／1〜15／16〜40／41〜）。
- 区分 × 艇番（1／2〜3／4〜6）ごとに、1着率・3着以内率 ÷ 市場の見込み（確定3連単オッズ）。探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で確かめる（IDEA-063 の core()）
- 参考（判定に使わない）：その時点のモーター2連対率と、そのモーターの入れ替え期間の最後の2連対率（後のデータ）との相関を、経過日数ごとに
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp_idea054 import KEY, EXP
from exp_idea063 import load_entries, core

def replacements(e):
    g = e.groupby(["jcd", "date"])["motor_2r"].median().reset_index().sort_values(["jcd", "date"])
    g["prev"] = g.groupby("jcd")["motor_2r"].shift()
    rep = g[(g["motor_2r"] <= 1) & ((g["prev"] > 10) | g["prev"].isna())][["jcd", "date"]].copy()
    rep["dt"] = pd.to_datetime(rep["date"]); keep = []
    for j, gg in rep.groupby("jcd"):
        last = None
        for d in gg["dt"].sort_values():
            if last is None or (d - last).days > 60:
                keep.append((j, d)); last = d
    return pd.DataFrame(keep, columns=["jcd", "rep_dt"])

def categories(e):
    rep = replacements(e)
    x = e.copy(); x["dt"] = pd.to_datetime(x["date"])
    x = x.sort_values("dt")
    x = pd.merge_asof(x, rep.sort_values("rep_dt"), left_on="dt", right_on="rep_dt", by="jcd", direction="backward")
    x["days"] = (x["dt"] - x["rep_dt"]).dt.days
    # その日・その場の中でのモーター2連対率の3分位（全 0 なら「数字なし」）
    mx = x.groupby(["jcd", "date"])["motor_2r"].transform("max")
    pct = x.groupby(["jcd", "date"])["motor_2r"].rank(pct=True, method="average")
    tert = np.where(pct <= 1 / 3, "下", np.where(pct <= 2 / 3, "中", "上"))
    tert = np.where(mx <= 0, "数字なし", tert)
    # 入れ替え後のそのモーターの当日より前の出走数
    st = x[~x["pos_raw"].astype(str).str.startswith("K")]
    cnt = st.groupby(["jcd", "rep_dt", "motor_no", "date"]).size().rename("n").reset_index().sort_values("date")
    cnt["prior"] = cnt.groupby(["jcd", "rep_dt", "motor_no"])["n"].cumsum() - cnt["n"]
    x = x.merge(cnt[["jcd", "rep_dt", "motor_no", "date", "prior"]], on=["jcd", "rep_dt", "motor_no", "date"], how="left")
    x["prior"] = x["prior"].fillna(0)
    dbin = pd.cut(x["days"], [-1, 30, 90, 1e6], labels=["〜30日", "31〜90日", "91日〜"]).astype(str)
    dbin = np.where(x["rep_dt"].isna(), "nan", dbin)
    x["経過日数"] = dbin
    x["経過×2連率"] = np.where(x["経過日数"] == "nan", "nan", x["経過日数"] + "・" + tert)
    pbin = pd.cut(x["prior"], [-1, 0, 15, 40, 1e6], labels=["0走", "1〜15走", "16〜40走", "41走〜"]).astype(str)
    x["経過×出走数"] = np.where(x["経過日数"] == "nan", "nan", x["経過日数"] + "・" + pbin)
    return x.sort_values(KEY + ["boat"]).reset_index(drop=True), rep

def reliability(x):
    """参考：その時点の2連対率と、同じ入れ替え期間の最後の2連対率の相関（経過日数ごと、モーター×節で1行）"""
    y = x[x["rep_dt"].notna()].groupby(["jcd", "rep_dt", "motor_no", "date"])["motor_2r"].first().reset_index()
    last = y.sort_values("date").groupby(["jcd", "rep_dt", "motor_no"])["motor_2r"].last().rename("final")
    y = y.join(last, on=["jcd", "rep_dt", "motor_no"])
    y["days"] = (pd.to_datetime(y["date"]) - y["rep_dt"]).dt.days
    rows = []
    for lab, lo, hi in (("〜30日", -1, 30), ("31〜90日", 30, 90), ("91〜180日", 90, 180), ("181日〜", 180, 1e6)):
        g = y[(y["days"] > lo) & (y["days"] <= hi) & (y["motor_2r"] > 0)]
        rows.append({"経過日数": lab, "件数": len(g), "その時点と最後の2連対率の相関": g[["motor_2r", "final"]].corr().iloc[0, 1] if len(g) > 10 else np.nan})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("bkdir")
    ap.add_argument("--out", default="results/idea064.md"); a = ap.parse_args()
    e = load_entries([a.histds, a.datadir], ["toban", "pos", "pos_raw", "st", "motor_no", "motor_2r"])
    e["pos"] = pd.to_numeric(e["pos"], errors="coerce"); e["motor_2r"] = pd.to_numeric(e["motor_2r"], errors="coerce")
    x, rep = categories(e)
    rel = reliability(x)
    nrep = rep.assign(y=rep["rep_dt"].dt.year).groupby("y").size().to_dict()
    extra = [f"- モーターの入れ替え日は {len(rep)} 回検出（年ごと {nrep}）。新モーターの最初の節は全モーターの2連対率が 0 で表示されるので「数字なし」。", "",
             "### 参考：モーター2連対率の当てになり具合（その時点の値と、同じモーターの入れ替え期間の最後の値の相関）", "", rel.to_markdown(index=False, floatfmt=".3f"), ""]
    md, judge, S, B, H, R = core(x, ["経過日数", "経過×2連率", "経過×出走数"], a.datadir, a.bkdir, "IDEA-064 新モーターの使い始め", extra)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:45]))

if __name__ == "__main__":
    main()
