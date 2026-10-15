"""IDEA-031 締切2分前に期待値がほどほど（0.7〜1.1）の単勝（docs/ideas.md の登録どおり。2026-10-14 以降のデータだけで判定）
usage: python scripts/exp_idea031.py DATADIR STOREDIR [--out results/idea031.md]
- live_t1（締切2〜0.5分前）の3連単オッズから出した1着確率 × live_t1 の単勝オッズ が 0.7 以上 1.1 以下の艇を各100円
- 払戻は確定の単勝オッズ×100円（K ファイルの単勝払戻と同じ）。判定：1,000点以上の時点で 回収率110%以上・日単位95%CI下限 > 100%
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from live_eval import KEY, _read

START = "20261014"
RNG = np.random.default_rng(0)

def before(o):
    return o[o["fetched_at"] < o["deadline"] + ":00"].drop_duplicates(KEY, keep="last") if len(o) else o

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea031.md")
    a = ap.parse_args()
    t3 = before(_read(f"{a.store}/live_t1/odds3t/*.csv.gz")); tt = before(_read(f"{a.store}/live_t1/oddstf/*.csv.gz"))
    cl = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in glob.glob(f"{a.store}/oddstf/*.csv.gz")])
    cl["jcd"] = cl["jcd"].str.zfill(2); cl = cl.drop_duplicates(KEY, keep="last")
    r = pd.read_parquet(f"{a.datadir}/races.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    r = r[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    m = r[r["date"] >= START].merge(t3[KEY], on=KEY).merge(tt[KEY], on=KEY).merge(cl[KEY], on=KEY)
    if not len(m):
        msg = f"実行前: {START} 以降の対象レース 0"; open(a.out, "w").write(msg + "\n"); print(msg); return
    O3 = m[KEY].merge(t3, on=KEY, how="left")[COMBOS].to_numpy(float)
    inv = np.where(np.isfinite(O3) & (O3 > 0), 1 / np.where(O3 > 0, O3, 1), 0); P = inv / inv.sum(1, keepdims=True)
    first = np.array([int(c[0]) for c in COMBOS]); pw = np.stack([P[:, first == b].sum(1) for b in range(1, 7)], 1)
    W1 = m[KEY].merge(tt, on=KEY, how="left")[[f"tan{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    WC = m[KEY].merge(cl, on=KEY, how="left")[[f"tan{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    win = m["tri_combo"].astype(str).str[0].astype(int).values
    ev = pw * W1
    buy = np.isfinite(ev) & (ev >= 0.7) & (ev <= 1.1) & np.isfinite(WC) & (WC > 0)
    hit = np.zeros_like(buy); hit[np.arange(len(m)), win - 1] = True
    ret = np.where(buy & hit, WC * 100, 0.0)
    g = pd.DataFrame({"d": m["date"].values, "st": buy.sum(1) * 100.0, "rt": ret.sum(1)}).groupby("d").sum()
    n = int(buy.sum()); roi = g["rt"].sum() / max(g["st"].sum(), 1)
    if len(g) >= 2 and n:
        bi = RNG.integers(0, len(g), size=(5000, len(g))); b = g["rt"].values[bi].sum(1) / np.clip(g["st"].values[bi].sum(1), 1, None)
        lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
    else:
        lo = hi = np.nan
    state = ("採用候補" if (roi >= 1.1 and lo > 1.0) else "不採用") if n >= 1000 else f"判定前（{n}/1000点）"
    md = [f"### IDEA-031 締切2分前に期待値 0.7〜1.1 の単勝（{START}〜{m['date'].max()}、{len(m)}レース、{len(g)}日）", "",
          f"点数 {n}、的中 {int((buy & hit).sum())}、回収率 {roi:.3f}（日単位95%CI {lo:.3f}〜{hi:.3f}） → **{state}**", "",
          "日ごと", "", g.assign(回収率=g["rt"] / g["st"]).rename(columns={"st": "購入", "rt": "払戻"}).to_markdown(floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
