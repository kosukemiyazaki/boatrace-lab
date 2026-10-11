"""IDEA-022: 単勝の歪みを3連単の市場で見抜く（docs/ideas.md の登録どおり。閾値 {1.2, 1.5, 2.0} の3通りのみ）
usage: python scripts/exp_idea022.py DATADIR STOREDIR [--out results/idea022.md]
- 3連単の締切時オッズの逆数を正規化 → 各艇の1着確率 q_b（3連単の市場）
- 条件: q_b × 単勝の締切時オッズ T_b > 閾値 の艇の単勝を各100円。払戻は T_b × 100円（1着なら）。欠場は返還
- 期間 2025-10〜2026-08（テスト期間は使わない）。締切時オッズで判定・購入できたとしたらの上限
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

THRS = (1.2, 1.5, 2.0)
RNG = np.random.default_rng(0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea022.md")
    a = ap.parse_args()
    key = ["date", "jcd", "rno"]
    tf = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in sorted(glob.glob(f"{a.store}/oddstf/*.csv.gz"))]); tf["jcd"] = tf["jcd"].str.zfill(2)
    o3 = pd.read_parquet(f"{a.datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    e = pd.read_parquet(f"{a.datadir}/entries.parquet", columns=key + ["boat", "pos", "pos_raw"])
    m = tf.merge(o3, on=key)
    m = m[(m["date"] >= "20251001") & (m["date"] < "20260901")].reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); inv = np.where(O > 0, 1 / np.where(O > 0, O, 1), 0)
    ok = (O > 0).sum(1) >= 60
    p3 = inv / np.clip(inv.sum(1, keepdims=True), 1e-12, None)
    first = np.array([int(c[0]) - 1 for c in COMBOS])
    q = np.stack([p3[:, first == b].sum(1) for b in range(6)], 1)
    T = m[[f"tan{b}" for b in range(1, 7)]].to_numpy(float)
    res = e.pivot_table(index=key, columns="boat", values="pos", aggfunc="first").reindex(pd.MultiIndex.from_frame(m[key]))
    absent = e.assign(a=e["pos_raw"].astype(str).str.match(r"^K")).pivot_table(index=key, columns="boat", values="a", aggfunc="max").reindex(pd.MultiIndex.from_frame(m[key])).fillna(False).to_numpy(bool)
    won = (res.to_numpy(float) == 1)
    has_res = np.isfinite(res.to_numpy(float)).any(1)
    valid = ok[:, None] & has_res[:, None] & np.isfinite(T) & (T > 0)
    rows = []
    for thr in THRS:
        buy = valid & (q * np.nan_to_num(T) > thr)
        ret = np.where(buy & absent, 100.0, np.where(buy & won, T * 100, 0.0))
        st = buy.sum(1) * 100.0; rt = ret.sum(1)
        df = pd.DataFrame({"d": m["date"], "st": st, "rt": rt}).groupby("d").sum()
        bi = RNG.integers(0, len(df), size=(5000, len(df)))
        b = df["rt"].values[bi].sum(1) / np.clip(df["st"].values[bi].sum(1), 1, None)
        n = int(buy.sum()); hits = int((buy & won).sum())
        roi = df["rt"].sum() / df["st"].sum() if n else np.nan
        rows.append({"閾値": thr, "点数": n, "1日あたり点数": n / max(len(df), 1), "的中": hits, "的中率": hits / n if n else np.nan,
                     "平均単勝オッズ": float(np.nan_to_num(T)[buy].mean()) if n else np.nan, "3連単市場の平均確率": float(q[buy].mean()) if n else np.nan,
                     "回収率": roi, "95%CI下": float(np.quantile(b, .025)), "95%CI上": float(np.quantile(b, .975)),
                     "判定": "採用候補" if (n >= 200 and roi >= 1.1 and np.quantile(b, .025) > 1.0) else "不採用"})
    t = pd.DataFrame(rows)
    md = [f"### IDEA-022 単勝の歪み（{m['date'].min()}〜{m['date'].max()}、{int(ok.sum())}レース。締切時オッズで判定・購入できたとしたらの上限）", "",
          t.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
