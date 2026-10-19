"""IDEA-039 段階2：拡連複の割安は締切直前のオッズでも残るか（docs/ideas.md の登録どおり。2026-10-18 以降で判定）
usage: python scripts/idea039_stage2.py STOREDIR [--start 20261018] [--out results/idea039_stage2.md]
- 判定：締切直前（live_t1）の3連単オッズから出した q（2艇がともに3着以内の確率）× live_t1 の拡連複オッズの下限 > 1.1 の組を各100円
- 払戻は K ファイルの拡連複の実額。点数 1,000 以上の時点で 回収率110%以上・日単位95%CI下限 > 100% → 採用候補
- 参考（判定に使わない）：同じ q × live_t0（締切45〜6秒前）の拡連複オッズの下限 > 1.1、閾値 1.2・1.5
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import exp_idea039 as E
from boatlib.parse import COMBOS, COMBOS_2F

KEY = ["date", "jcd", "rno"]; RNG = np.random.default_rng(0)

def load(store, lead, page, start):
    fs = [f for f in glob.glob(f"{store}/{lead}/{page}/*.csv.gz") if os.path.basename(f)[:8] >= start]
    if not fs:
        return pd.DataFrame(columns=KEY)
    d = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs])
    d["jcd"] = d["jcd"].str.zfill(2); d["date"] = d["date"].astype(str)
    if "fetched_at" in d and "deadline" in d:
        d = d[d["fetched_at"].astype(str) < d["deadline"].astype(str) + ":00"]
    return d.sort_values("fetched_at").drop_duplicates(KEY, keep="last") if "fetched_at" in d else d.drop_duplicates(KEY, keep="last")

def boot(g):
    if len(g) < 2:
        return np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("store"); ap.add_argument("--start", default="20261018")
    ap.add_argument("--out", default="results/idea039_stage2.md"); a = ap.parse_args()
    o3 = load(a.store, "live_t1", "odds3t", a.start)
    E.START, E.END = a.start, "29991231"
    pay = E.wide_pay(a.store); pay["jcd"] = pay["jcd"].astype(str).str.zfill(2)
    done = set(map(tuple, pay[KEY].drop_duplicates().values))
    paymap = {(r.date, r.jcd, r.rno, r.pair): r.pay for r in pay.itertuples()}
    o3 = o3[[k in done for k in map(tuple, o3[KEY].values)]].reset_index(drop=True)
    O = o3[COMBOS].apply(pd.to_numeric, errors="coerce").to_numpy(float); v = np.isfinite(O) & (O > 0); ok = v.sum(1) >= 40
    o3, O, v = o3[ok].reset_index(drop=True), O[ok], v[ok]
    Pm = np.where(v, 1 / np.where(v, O, 1), 0); Pm /= Pm.sum(1, keepdims=True)
    sets = [set(map(int, c.split("-"))) for c in COMBOS]
    rows = []
    for lead in ("live_t1", "live_t0"):
        wk = load(a.store, lead, "oddsk", a.start)
        if not len(wk):
            continue
        m = o3[KEY].reset_index().merge(wk, on=KEY)
        for pr in COMBOS_2F:
            i, j = map(int, pr.split("="))
            q = Pm[m["index"].values][:, [k for k, s in enumerate(sets) if i in s and j in s]].sum(1)
            lo = pd.to_numeric(m.get(f"wide_lo{pr}"), errors="coerce").values
            p = np.array([paymap.get((d, jc, rn, pr), 0) for d, jc, rn in zip(m["date"], m["jcd"], m["rno"])], float)
            rows.append(pd.DataFrame({"lead": lead, "date": m["date"], "q": q, "lo": lo, "pay": p}))
    d = pd.concat(rows, ignore_index=True); d = d[np.isfinite(d["lo"]) & (d["lo"] > 0) & (d["q"] > 0)]
    out = []
    for lead in ("live_t1", "live_t0"):
        for thr in (1.1, 1.2, 1.5):
            s = d[(d["lead"] == lead) & (d["q"] * d["lo"] > thr)]
            g = s.groupby("date").agg(st=("pay", "size"), rt=("pay", "sum")); g["st"] *= 100
            n = len(s); roi = g["rt"].sum() / g["st"].sum() if n else np.nan; lo95, hi95 = boot(g)
            main_rule = lead == "live_t1" and thr == 1.1
            st = ("採用候補" if roi >= 1.1 and lo95 > 1.0 else "不採用") if (main_rule and n >= 1000) else (f"判定前（{n}/1000点）" if main_rule else "参考")
            out.append({"判定時点": {"live_t1": "締切2〜0.5分前", "live_t0": "締切45〜6秒前"}[lead], "閾値": thr, "点数": n,
                        "的中": int((s["pay"] > 0).sum()), "回収率": roi, "95%CI下": lo95, "95%CI上": hi95, "状態": st})
    R = pd.DataFrame(out); nr = len(o3)
    days = sorted(o3["date"].unique())
    md = [f"# IDEA-039 段階2 拡連複：締切直前のオッズでも割安は残るか（{days[0] if days else a.start}〜{days[-1] if days else ''}、{nr}レース）", "",
          "q = 締切直前（live_t1）の3連単オッズから出した「2艇がともに3着以内」の確率。q × 拡連複オッズの下限 > 閾値 の組を各100円、払戻は K ファイルの実額。",
          "判定は 締切2〜0.5分前・閾値1.1 のみ（1,000点で判定）。ほかは参考。", "", R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
