"""IDEA-050 券種どうしの食い違い（締切2分前）：3連複・2連単は3連単より安く売られているか（docs/ideas.md の登録どおり）
usage: python scripts/idea050.py STOREDIR [--start 20261011] [--out results/idea050.md]
- t1 の3連単オッズ → P → π = P^1.06（正規化）。3連複 {a,b,c} の確率 = 6通りの π の合計、2連単 a-b = 4通りの π の合計
- 確率 × t1 の 3連複／2連単オッズ > 1.1（参考 1.0・1.2）の組を各100円、払戻は K ファイルの実額
- 判定：点数 500 以上で 回収率110%以上・日単位95%CI下限 > 100% → 採用候補
- 参考：買った組の「確定払戻/100 ÷ t1 オッズ」の平均と、確定でも 確率×オッズ > 1.1 のままだった割合
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import boatlib.parse as P
from boatlib.parse import COMBOS, COMBOS_3F, COMBOS_2T
from idea039_stage2 import load

KEY = ["date", "jcd", "rno"]; RNG = np.random.default_rng(0); BETA = 1.06

def k_pay(store, start):
    """K ファイルの 3連複・2連単 の払戻（100円あたり）"""
    rows = []
    for f in sorted(glob.glob(f"{store}/bk/*/k*.txt.gz")):
        date = "20" + os.path.basename(f)[1:7]
        if date < start:
            continue
        txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
        for jcd, blk in P._blocks(txt, "K"):
            rno = None
            for line in blk.splitlines():
                l = P.z2h(line); m = P.K_RACE.match(l)
                if m:
                    rno = int(m.group(1)); continue
                if rno is None:
                    continue
                mm = re.match(r"^\s+3連複\s+(\d)-(\d)-(\d)\s+(\d+)", l)
                if mm:
                    rows.append((date, jcd, rno, "3f", "=".join(sorted(mm.group(1, 2, 3))), int(mm.group(4)))); continue
                mm = re.match(r"^\s+2連単\s+(\d)-(\d)\s+(\d+)", l)
                if mm:
                    rows.append((date, jcd, rno, "2t", f"{mm.group(1)}-{mm.group(2)}", int(mm.group(3))))
    d = pd.DataFrame(rows, columns=KEY + ["kind", "combo", "pay"]); d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return d.drop_duplicates(KEY + ["kind"], keep="last")

def boot(g):
    if len(g) < 2:
        return np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("store"); ap.add_argument("--start", default="20261011"); ap.add_argument("--out", default="results/idea050.md")
    a = ap.parse_args()
    o3 = load(a.store, "live_t1", "odds3t", a.start)
    pay = k_pay(a.store, a.start); done = set(map(tuple, pay[KEY].drop_duplicates().values))
    o3 = o3[[k in done for k in map(tuple, o3[KEY].values)]].reset_index(drop=True)
    O = o3[COMBOS].apply(pd.to_numeric, errors="coerce").to_numpy(float); v = np.isfinite(O) & (O > 0); ok = v.sum(1) >= 40
    o3, O, v = o3[ok].reset_index(drop=True), O[ok], v[ok]
    Pm = np.where(v, 1 / np.where(v, O, 1), 0); Pm /= Pm.sum(1, keepdims=True)
    pi = Pm ** BETA; pi /= pi.sum(1, keepdims=True)
    sets3 = [frozenset(map(int, c.split("-"))) for c in COMBOS]; head2 = [c[:3] for c in COMBOS]
    paymap = {(r.date, r.jcd, r.rno, r.kind): (r.combo, r.pay) for r in pay.itertuples()}
    specs = {"3f": ("3連複", "odds3f", COMBOS_3F, lambda c: [k for k, s in enumerate(sets3) if s == frozenset(map(int, c.split("=")))]),
             "2t": ("2連単", "odds2tf", COMBOS_2T, lambda c: [k for k, h in enumerate(head2) if h == c])}
    out, md = [], []
    for kind, (name, page, cols, idx) in specs.items():
        w = load(a.store, "live_t1", page, a.start); m = o3[KEY].reset_index().merge(w, on=KEY)
        if not len(m):
            continue
        rows = []
        for c in cols:
            q = pi[m["index"].values][:, idx(c)].sum(1)
            od = pd.to_numeric(m.get(c), errors="coerce").values
            res = [paymap.get((d, j, r, kind), (None, 0)) for d, j, r in zip(m["date"], m["jcd"], m["rno"])]
            p = np.array([pp if cc == c else 0 for cc, pp in res], float)
            rows.append(pd.DataFrame({"date": m["date"], "q": q, "od": od, "pay": p, "hit": p > 0}))
        d = pd.concat(rows, ignore_index=True); d = d[np.isfinite(d["od"]) & (d["od"] > 0) & (d["q"] > 0)]
        for thr in (1.0, 1.1, 1.2):
            b = d[d["q"] * d["od"] > thr]
            g = b.assign(st=100.0, rt=b["pay"]).groupby("date")[["st", "rt"]].sum()
            n = len(b); roi = g["rt"].sum() / g["st"].sum() if n else np.nan; lo, hi = boot(g)
            h = b[b["hit"]]; ratio = (h["pay"] / 100 / h["od"]).mean() if len(h) else np.nan
            still = ((h["q"] * h["pay"] / 100) > 1.1).mean() if len(h) else np.nan
            out.append({"券種": name, "閾値": thr, "点数": n, "的中": int(b["hit"].sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi,
                        "確定払戻÷t1オッズ（的中した組）": ratio, "確定でも>1.1の割合": still, "レース": int(m.shape[0]), "日数": int(b["date"].nunique())})
    R = pd.DataFrame(out)
    j = R[R["閾値"] == 1.1]
    ready = j["点数"] >= 500
    verdict = []
    for _, r in j.iterrows():
        if r["点数"] < 500:
            verdict.append(f"{r['券種']}：{int(r['点数'])}点（判定は500点）→ 集計中")
        elif r["回収率"] >= 1.1 and r["95%CI下"] > 1.0:
            verdict.append(f"{r['券種']}：採用候補")
        else:
            verdict.append(f"{r['券種']}：不採用（回収率 {r['回収率']:.3f}、CI下 {r['95%CI下']:.3f}）")
    md = [f"# IDEA-050 券種どうしの食い違い（t1、{a.start}〜、払戻は K の実額）", "", "**判定：" + "／".join(verdict) + "**", "",
          R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
