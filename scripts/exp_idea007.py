"""IDEA-007: 特定条件での大穴（開発期間のみ。テスト期間は使わない）
usage: python scripts/exp_idea007.py DATADIR [--out results/idea007_longshots.md]
- 期間 2025-10〜2026-08。締切時オッズ100倍以上の全組番を各100円（過去分は締切時オッズでの分類なので甘く出る）
- 条件（4つのみ）: L1 風速5m以上 / L2 波高5cm以上 / L3 1号艇がB級 / L4 1号艇の展示タイムがレース内で最も遅い
  （風・波・展示タイムは直前情報。1号艇の級別は番組表）
- 比較対象: 同じ期間の「100倍以上の全組番」
- 払戻: 公式Kファイルの3連単払戻金。F/L/欠場艇を含む組番は返還
- 判定: 条件の回収率が全体より高く、かつ日単位ブートストラップ95%信頼区間の下限が100%を超える条件だけを候補にする
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

START, END, OMIN = "20251001", "20260831", 100.0
RNG = np.random.default_rng(0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea007_longshots.md")
    a = ap.parse_args()
    key = ["date", "jcd", "rno"]
    races = pd.read_parquet(f"{a.datadir}/races.parquet").drop_duplicates(key)
    races = races[(races["date"] >= START) & (races["date"] <= END) & races["tri_combo"].notna()]
    odds = pd.read_parquet(f"{a.datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    bi = pd.read_parquet(f"{a.datadir}/beforeinfo.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    e = pd.read_parquet(f"{a.datadir}/entries.parquet")
    e = e[(e["date"] >= START) & (e["date"] <= END)]
    b1 = e[e["boat"] == 1][key + ["cls"]]
    refund = e.assign(rf=e["pos_raw"].astype(str).str.match(r"^(F|L|K)"))[key + ["boat", "rf"]]
    r = races[key + ["tri_combo", "tri_pay"]].merge(odds, on=key).merge(bi[key + ["wind", "wave"] + [f"exh{b}" for b in range(1, 7)]], on=key, how="left").merge(b1, on=key, how="left")
    O = r[COMBOS].to_numpy(float)
    ok = np.isfinite(O).sum(1) >= 60
    r, O = r[ok].reset_index(drop=True), O[ok]
    y = r["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    rf = refund.pivot_table(index=key, columns="boat", values="rf", aggfunc="max").reindex(pd.MultiIndex.from_frame(r[key])).fillna(False).to_numpy(bool)
    ref_combo = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref_combo[:, ci] |= rf[:, int(b) - 1] if rf.shape[1] >= int(b) else False
    exh = r[[f"exh{b}" for b in range(1, 7)]].to_numpy(float)
    slowest1 = np.isfinite(exh[:, 0]) & (exh[:, 0] >= np.nanmax(exh, axis=1))
    conds = {
        "全体（100倍以上の全組番）": np.ones(len(r), bool),
        "L1 風速5m以上": r["wind"].to_numpy(float) >= 5,
        "L2 波高5cm以上": r["wave"].to_numpy(float) >= 5,
        "L3 1号艇がB級": r["cls"].isin(["B1", "B2"]).to_numpy(),
        "L4 1号艇の展示タイムが最も遅い": slowest1,
    }
    buy = np.isfinite(O) & (O >= OMIN)
    hit = buy & (np.arange(120)[None, :] == y[:, None]) & ~ref_combo
    stake_r = buy.sum(1) * 100.0
    ret_r = hit.any(1) * r["tri_pay"].to_numpy(float) + (buy & ref_combo).sum(1) * 100.0
    rows = []
    base = None
    for name, m in conds.items():
        df = pd.DataFrame({"day": r["date"][m], "st": stake_r[m], "rt": ret_r[m]}).groupby("day").sum()
        st, rt = df["st"].values, df["rt"].values
        bi_ = RNG.integers(0, len(st), size=(5000, len(st)))
        rois = rt[bi_].sum(1) / st[bi_].sum(1)
        roi = rt.sum() / st.sum()
        if base is None:
            base = roi
        lo = float(np.quantile(rois, .025))
        rows.append({"条件": name, "レース": int(m.sum()), "点数": int(buy[m].sum()), "的中": int(hit[m].sum()),
                     "回収率": roi, "95%CI下": lo, "95%CI上": float(np.quantile(rois, .975)),
                     "判定": "" if name.startswith("全体") else ("候補" if roi > base and lo > 1.0 else "不採用")})
    md = [f"期間 {START}〜{END}（締切時オッズのあるレース {len(r)}）。締切時オッズ{OMIN:.0f}倍以上の全組番を各100円。試した条件4つの全件", "",
          pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
