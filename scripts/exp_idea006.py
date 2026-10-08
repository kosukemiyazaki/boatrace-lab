"""IDEA-006 公式コンピュータ予想の逆張り（docs/ideas.md の登録どおり。R1・R2 の2通りのみ）。
usage:
  python scripts/exp_idea006.py r1 DATADIR STOREDIR [--out results/idea006_r1.md]
  python scripts/exp_idea006.py r2 DATADIR STOREDIR [--n 2000] [--out results/idea006_r2.md]
- コンピュータ予想: STOREDIR/pcexpect/*.csv.gz の pc_tri（3連単の予想組番。「5=2-4」などの順不同の表記は展開済み）
- R1（開発期間 2025-10〜2026-08、テスト期間は使わない）: 締切時オッズ帯（10倍未満・10〜30倍・30〜100倍）ごとに、
  予想の組番に「含まれる」/「含まれない」組番を各100円買った場合の回収率。差（含まれない − 含まれる）の95%信頼区間は日単位ブートストラップ。
  払戻は公式Kファイルの3連単払戻金。F/L/欠場艇を含む組番は返還。過去分は締切時オッズでの分類なので甘く出る点に注意。
- R2（撤退判定と同じ評価用レース、scripts/live_eval.py）: π ∝ P_model^a × P_6分前^b × exp(c × 予想の組番に含まれる)
  を前半で推定し、後半の3連単対数損失を M0（c なし）と比べる。
  コンピュータ予想は当日朝に公開され、その日のうちは変わらない前提（daily がレース後に取得したものを使う）。
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea002 import fit_loglin, loglin
from live_eval import KEY, day_boot_ci, head, load, market, nll

START, END = "20251001", "20260831"
BANDS = [(0, 10), (10, 30), (30, 100)]
CI = {c: i for i, c in enumerate(COMBOS)}

def pc_matrix(store, r):
    fs = sorted(glob.glob(f"{store}/pcexpect/*.csv.gz"))
    pc = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str, "pc_tri": str}) for f in fs], ignore_index=True)
    pc["jcd"] = pc["jcd"].str.zfill(2)
    pc = r[KEY].merge(pc.drop_duplicates(KEY, keep="last"), on=KEY, how="left")
    I = np.zeros((len(r), 120), bool)
    for i, s in enumerate(pc["pc_tri"].values):
        if isinstance(s, str):
            for c in s.split(";"):
                if c in CI:
                    I[i, CI[c]] = True
    return I, pc["pc_tri"].notna().values

def r1(a):
    races = pd.read_parquet(f"{a.datadir}/races.parquet").drop_duplicates(KEY)
    races = races[(races["date"] >= START) & (races["date"] <= END) & races["tri_combo"].notna()]
    odds = pd.read_parquet(f"{a.datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    e = pd.read_parquet(f"{a.datadir}/entries.parquet")
    e = e[(e["date"] >= START) & (e["date"] <= END)]
    r = races[KEY + ["tri_combo", "tri_pay"]].merge(odds, on=KEY)
    O = r[COMBOS].to_numpy(float)
    I, has = pc_matrix(a.store, r)
    ok = (np.isfinite(O).sum(1) >= 60) & has
    r, O, I = r[ok].reset_index(drop=True), O[ok], I[ok]
    y = r["tri_combo"].map(CI).values
    rf = e.assign(rf=e["pos_raw"].astype(str).str.match(r"^(F|L|K)")).pivot_table(index=KEY, columns="boat", values="rf", aggfunc="max")
    rf = rf.reindex(pd.MultiIndex.from_frame(r[KEY])).fillna(False).to_numpy(bool)
    ref = np.zeros((len(r), 120), bool)
    for ci, c in enumerate(COMBOS):
        for b in (c[0], c[2], c[4]):
            ref[:, ci] |= rf[:, int(b) - 1]
    win = (np.arange(120)[None, :] == y[:, None]) & ~ref
    pay = r["tri_pay"].to_numpy(float)[:, None]
    ret = np.where(win, pay, 0.0) + np.where(ref, 100.0, 0.0)
    days = r["date"].values
    rows = []
    for lo, hi in BANDS:
        band = np.isfinite(O) & (O >= lo) & (O < hi)
        res = {}
        for name, m in (("含まれる", band & I), ("含まれない", band & ~I)):
            df = pd.DataFrame({"day": days, "st": m.sum(1) * 100.0, "rt": (ret * m).sum(1)}).groupby("day").sum()
            res[name] = df
        dd = res["含まれる"].join(res["含まれない"], lsuffix="_in", rsuffix="_out", how="outer").fillna(0)
        bi = np.random.default_rng(0).integers(0, len(dd), size=(5000, len(dd)))
        roi_in = dd["rt_in"].values[bi].sum(1) / dd["st_in"].values[bi].sum(1)
        roi_out = dd["rt_out"].values[bi].sum(1) / dd["st_out"].values[bi].sum(1)
        diff = roi_out - roi_in
        R_in, R_out = dd["rt_in"].sum() / dd["st_in"].sum(), dd["rt_out"].sum() / dd["st_out"].sum()
        lo_ci, hi_ci = float(np.quantile(diff, .025)), float(np.quantile(diff, .975))
        rows.append({"オッズ帯": f"{lo}〜{hi}倍", "点数（含まれる）": int((band & I).sum()), "回収率（含まれる）": R_in,
                     "点数（含まれない）": int((band & ~I).sum()), "回収率（含まれない）": R_out,
                     "差（含まれない−含まれる）": R_out - R_in, "95%CI下": lo_ci, "95%CI上": hi_ci,
                     "判定": "支持" if (R_out > R_in and lo_ci > 0) else "支持しない"})
    md = [f"### IDEA-006 R1（期間 {START}〜{END}、コンピュータ予想と締切時オッズのあるレース {len(r)}、{len(set(days))}日）",
          "各組番100円。試したオッズ帯3つの全件", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    return md

def r2(a):
    d = load(a.datadir, a.store)
    if len(d["r"]) < a.n:
        return [f"実行前: 対象レース {len(d['r'])} / {a.n}"]
    d = head(d, a.n)
    y, days = d["y"], d["r"]["date"].values
    I, has = pc_matrix(a.store, d["r"])
    lg = lambda Q: np.log(np.clip(Q, 1e-12, None))
    lm, l6 = lg(d["P"]), lg(market(d["L"]["t6"]))
    h = a.n // 2
    fit, ev = np.arange(a.n) < h, np.arange(a.n) >= h
    specs = {"M0": [lm, l6], "M0+予想": [lm, l6, I.astype(float)]}
    res, w = {}, {}
    for k, logs in specs.items():
        ww = fit_loglin([x[fit] for x in logs], y[fit], [0.5, 0.5] + [0.0] * (len(logs) - 2))
        w[k] = [float(v) for v in ww]; res[k] = nll(loglin(logs, ww), y)
    res["market_close"] = nll(market(d["C"]), y)
    diff = res["M0+予想"][ev] - res["M0"][ev]
    ci = day_boot_ci(diff, days[ev])
    out = dict(races=a.n, eval_races=int(ev.sum()), races_with_pc=int(has.sum()), weights=w,
               logloss_eval={k: float(v[ev].mean()) for k, v in res.items()},
               diff_vs_M0=float(diff.mean()), ci95=ci,
               decision="採用候補" if (diff.mean() < 0 and ci[1] < 0) else "不採用")
    return ["### IDEA-006 R2", "```", json.dumps(out, ensure_ascii=False, indent=1), "```"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["r1", "r2"]); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--out")
    a = ap.parse_args()
    md = r1(a) if a.which == "r1" else r2(a)
    out = a.out or f"results/idea006_{a.which}.md"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
