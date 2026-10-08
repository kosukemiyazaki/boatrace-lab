"""IDEA-005 小さい市場への絞り込み（docs/ideas.md の登録どおり。区分は S1・S2・S3 の3通りのみ。賭けの評価はしない）。
usage: python scripts/exp_idea005.py DATADIR STOREDIR [--n 2000] [--out results/idea005.md]
- 評価用レース: 撤退判定と同じ（scripts/live_eval.py）。先頭 n レースの前半で a, b を推定し、後半で評価する。
- レースごとの差 = 「モデル×市場（6分前）」の3連単対数損失 − 「締切時オッズの市場」の3連単対数損失
- 区分ごとに差の平均を出し、「小さい市場」側 − 反対側（差の差）の95%信頼区間を日単位ブートストラップで出す。
- 判定: 差の差が負で、信頼区間が0をまたがない区分だけを、次の段階の買い方の条件の候補にする。
- 区分:
  S1 締切 12:00 より前 / 以降
  S2 予選・一般 / 上位戦（準優・優勝・特選・選抜・ドリーム・特賞）。レース名に「予選」「一般」があれば予選・一般
  S3 一般戦 / グレード戦（STORE/grades.csv で場・日付が開催期間に入り、grade が SG・G1・G2・G3 で始まる）
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.model import blend, fit_blend
from live_eval import head, load, market, nll

UPPER = re.compile("準優|優勝|特選|選抜|ドリーム|特賞")

def is_lower_race(name):
    name = re.sub(r"\s", "", str(name))
    return ("予選" in name) or ("一般" in name) or not UPPER.search(name)

def grade_race(r, grades_path):
    g = pd.read_csv(grades_path, dtype=str)
    g = g[g["grade"].str.match(r"(SG|G1|G2|G3)")]
    out = np.zeros(len(r), bool)
    for _, x in g.iterrows():
        out |= (r["jcd"].values == x["jcd"]) & (r["date"].values >= x["start"]) & (r["date"].values <= x["end"])
    return out

def diff_of_diff_ci(diff, small, days, n_boot=5000, seed=0):
    df = pd.DataFrame(dict(d=diff, s=small, day=days))
    agg = df.groupby(["day", "s"])["d"].agg(["sum", "count"]).unstack(fill_value=0)
    S1, N1 = agg[("sum", True)].values, agg[("count", True)].values
    S0, N0 = agg[("sum", False)].values, agg[("count", False)].values
    bi = np.random.default_rng(seed).integers(0, len(agg), size=(n_boot, len(agg)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = S1[bi].sum(1) / N1[bi].sum(1) - S0[bi].sum(1) / N0[bi].sum(1)
    boot = boot[np.isfinite(boot)]
    return [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--out", default="results/idea005.md")
    a = ap.parse_args()
    d = load(a.datadir, a.store)
    if len(d["r"]) < a.n:
        msg = f"実行前: 対象レース {len(d['r'])} / {a.n}"
        open(a.out, "w").write(msg + "\n"); print(msg); return
    d = head(d, a.n)
    r, y = d["r"], d["y"]
    pl, pc = market(d["L"]["t6"]), market(d["C"])
    h = a.n // 2
    fit, ev = np.arange(a.n) < h, np.arange(a.n) >= h
    (wa, wb), _ = fit_blend(d["P"][fit], pl[fit], y[fit])
    diff = nll(blend(d["P"], pl, wa, wb), y) - nll(pc, y)
    names = pd.read_parquet(f"{a.datadir}/races.parquet", columns=["date", "jcd", "rno", "race_name"]).assign(
        date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    rn = r.merge(names.drop_duplicates(["date", "jcd", "rno"]), on=["date", "jcd", "rno"], how="left")["race_name"]
    seg = {
        "S1 朝（締切12時前）": r["deadline"].values < "12:00",
        "S2 予選・一般": rn.map(is_lower_race).values,
        "S3 一般戦（グレードなし）": ~grade_race(r, f"{a.store}/grades.csv"),
    }
    out = dict(races=a.n, eval_races=int(ev.sum()), period=[str(r["date"].min()), str(r["date"].max())],
               blend_ab=[float(wa), float(wb)], overall_diff=float(diff[ev].mean()), segments={})
    for k, small in seg.items():
        s_ev, days = small[ev], r["date"].values[ev]
        dd = diff[ev]
        ci = diff_of_diff_ci(dd, s_ev, days) if 0 < s_ev.sum() < len(s_ev) else [float("nan")] * 2
        dod = float(dd[s_ev].mean() - dd[~s_ev].mean()) if 0 < s_ev.sum() < len(s_ev) else float("nan")
        out["segments"][k] = dict(n_small=int(s_ev.sum()), n_other=int((~s_ev).sum()),
                                  diff_small=float(dd[s_ev].mean()) if s_ev.any() else None,
                                  diff_other=float(dd[~s_ev].mean()) if (~s_ev).any() else None,
                                  diff_of_diff=dod, ci95=ci,
                                  candidate=bool(dod < 0 and ci[1] < 0))
    md = ["### IDEA-005 小さい市場", "差 = モデル×市場（6分前）− 締切時の市場（3連単対数損失、負ならモデルが良い）", "```",
          json.dumps(out, ensure_ascii=False, indent=1), "```"]
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
