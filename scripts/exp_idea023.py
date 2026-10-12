"""IDEA-023 の 5（docs/ideas.md の登録どおり）：3連単の市場を答えにして、2連単・2連複・3連複のずれを測る（締切時オッズ＝上限）
usage: python scripts/exp_idea023.py DATADIR [--out results/idea023_other.md]
- 3連単の締切時オッズの逆数をレース内で正規化 → p(a-b-c)。2連単 a-b = Σ_c p(a-b-c)、2連複 {a,b} = 2連単 a-b + b-a、3連複 {a,b,c} = 6通りの和
- 条件：その確率 × その券種の締切時オッズ > 閾値 {1.1, 1.2, 1.5} の買い目を各100円。払戻は締切時オッズ×100円（欠場艇を含む買い目は返還＝100円）
- 期間 2025-10〜2026-08（テスト期間は使わない）。判定：点数200以上・回収率110%以上・日単位ブートストラップ95%CI下限100%超 → 採用候補（締切時オッズでの上限として）
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, COMBOS_2F, COMBOS_2T, COMBOS_3F

KEY = ["date", "jcd", "rno"]
THRS = (1.1, 1.2, 1.5)
RNG = np.random.default_rng(0)

def norm(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea023_other.md")
    a = ap.parse_args()
    o3 = norm(pd.read_parquet(f"{a.datadir}/odds.parquet")); f3 = norm(pd.read_parquet(f"{a.datadir}/odds3f.parquet"))
    t2 = norm(pd.read_parquet(f"{a.datadir}/odds2tf.parquet"))
    races = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].drop_duplicates(KEY)
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "pos_raw"]))
    m = races[races["tri_combo"].notna()].merge(o3, on=KEY).merge(f3, on=KEY).merge(t2, on=KEY)
    m = m[(m["date"] >= "20251001") & (m["date"] <= "20260831")].reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); ok = (np.isfinite(O) & (O > 0)).sum(1) >= 60
    m, O = m[ok].reset_index(drop=True), O[ok]
    inv = np.where(np.isfinite(O) & (O > 0), 1 / np.where(O > 0, O, 1), 0); p3 = inv / inv.sum(1, keepdims=True)
    absent = e.assign(x=e["pos_raw"].astype(str).str.match(r"^K")).pivot_table(index=KEY, columns="boat", values="x", aggfunc="max")
    absent = absent.reindex(pd.MultiIndex.from_frame(m[KEY])).fillna(False).to_numpy(bool)
    win = m["tri_combo"].str.split("-", expand=True).astype(int).to_numpy()
    days = m["date"].values
    kinds = {}
    ci = {c: i for i, c in enumerate(COMBOS)}
    q2t = np.stack([sum(p3[:, ci[f"{x}-{y}-{z}"]] for z in range(1, 7) if z not in (x, y)) for x, y in (map(int, c.split("-")) for c in COMBOS_2T)], 1)
    kinds["2連単"] = (COMBOS_2T, q2t, lambda c, w: (int(c[0]), int(c[2])) == (w[0], w[1]))
    i2t = {c: i for i, c in enumerate(COMBOS_2T)}
    q2f = np.stack([q2t[:, i2t[f"{x}-{y}"]] + q2t[:, i2t[f"{y}-{x}"]] for x, y in (map(int, c.split("=")) for c in COMBOS_2F)], 1)
    kinds["2連複"] = (COMBOS_2F, q2f, lambda c, w: {int(c[0]), int(c[2])} == {w[0], w[1]})
    import itertools
    q3f = np.stack([sum(p3[:, ci["-".join(map(str, pp))]] for pp in itertools.permutations(map(int, c.split("=")))) for c in COMBOS_3F], 1)
    kinds["3連複"] = (COMBOS_3F, q3f, lambda c, w: {int(c[0]), int(c[2]), int(c[4])} == set(w[:3]))
    rows = []
    for k, (cols, q, hitf) in kinds.items():
        OK = m[cols].to_numpy(float)
        hit = np.array([[hitf(c, w) for c in cols] for w in win])
        boats = [[int(ch) for ch in c if ch.isdigit()] for c in cols]
        refund = np.stack([absent[:, [b - 1 for b in bs]].any(1) for bs in boats], 1)
        valid = np.isfinite(OK) & (OK > 0)
        for thr in (0.0,) + THRS:
            buy = valid & ((q * np.nan_to_num(OK) > thr) if thr else True)
            ret = np.where(buy & refund, 100.0, np.where(buy & hit, OK * 100, 0.0))
            g = pd.DataFrame({"d": days, "st": buy.sum(1) * 100.0, "rt": np.nan_to_num(ret).sum(1)}).groupby("d").sum()
            n = int(buy.sum())
            bi = RNG.integers(0, len(g), size=(3000, len(g)))
            b = g["rt"].values[bi].sum(1) / np.clip(g["st"].values[bi].sum(1), 1, None)
            roi = g["rt"].sum() / max(g["st"].sum(), 1); lo = float(np.quantile(b, .025))
            rows.append({"券種": k, "閾値": thr or "全買い目", "点数": n, "1日あたり": round(n / len(g), 1), "的中率": round(float((buy & hit).sum() / max(n, 1)), 3),
                         "平均オッズ": round(float(np.nan_to_num(OK)[buy].mean()), 1) if n else np.nan, "3連単市場の確率の平均": round(float(q[buy].mean()), 3) if n else np.nan,
                         "回収率": round(roi, 3), "95%CI下": round(lo, 3), "95%CI上": round(float(np.quantile(b, .975)), 3),
                         "判定": "" if not thr else ("採用候補" if (n >= 200 and roi >= 1.1 and lo > 1.0) else "不採用")})
    md = [f"### IDEA-023 の 5：2連単・2連複・3連複 と3連単の市場のずれ（{m['date'].min()}〜{m['date'].max()}、{len(m)}レース。締切時オッズで判定・購入できたとしたらの上限）", "",
          pd.DataFrame(rows).to_markdown(index=False)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
