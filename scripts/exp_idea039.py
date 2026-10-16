"""IDEA-039 拡連複（ワイド）：3連単の市場から見て割安か（docs/ideas.md の登録どおり。締切時オッズ＝上限）
usage: python scripts/exp_idea039.py DATADIR STOREDIR [--out results/idea039.md]
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as P
from boatlib.parse import COMBOS, COMBOS_2F

KEY = ["date", "jcd", "rno"]; THRS = (1.1, 1.2, 1.5); START, END = "20260301", "20260831"
RNG = np.random.default_rng(0)

def wide_pay(store):
    rows = []
    for f in sorted(glob.glob(f"{store}/bk/*/k*.txt.gz")):
        date = "20" + os.path.basename(f)[1:7]
        if not (START <= date <= END):
            continue
        txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
        for jcd, blk in P._blocks(txt, "K"):
            rno = None; inw = 0
            for line in blk.splitlines():
                m = P.K_RACE.match(P.z2h(line))
                if m:
                    rno = int(m.group(1)); inw = 0; continue
                if rno is None:
                    continue
                mm = re.match(r"^\s+拡連複\s+(\d)-(\d)\s+(\d+)", line)
                if mm:
                    inw = 1; rows.append((date, jcd, rno, f"{mm.group(1)}={mm.group(2)}", int(mm.group(3)))); continue
                mm = re.match(r"^\s+(\d)-(\d)\s+(\d+)\s+人気", line)
                if inw and mm and inw < 3:
                    inw += 1; rows.append((date, jcd, rno, f"{mm.group(1)}={mm.group(2)}", int(mm.group(3)))); continue
                inw = 0
    return pd.DataFrame(rows, columns=KEY + ["pair", "pay"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea039.md")
    a = ap.parse_args()
    wk = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in glob.glob(f"{a.store}/oddsk/*.csv.gz")])
    wk["jcd"] = wk["jcd"].str.zfill(2); wk = wk.drop_duplicates(KEY, keep="last")
    o3 = pd.read_parquet(f"{a.datadir}/odds.parquet").assign(date=lambda x: x["date"].astype(str), jcd=lambda x: x["jcd"].astype(str).str.zfill(2))
    m = wk.merge(o3, on=KEY); m = m[(m["date"] >= START) & (m["date"] <= END)].reset_index(drop=True)
    pay = wide_pay(a.store); pay["jcd"] = pay["jcd"].astype(str).str.zfill(2)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); ok = v.sum(1) >= 60
    m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]
    Pm = np.where(v, 1 / np.where(v, O, 1), 0); Pm /= Pm.sum(1, keepdims=True)
    sets = [set(map(int, c.split("-"))) for c in COMBOS]
    rows = []
    paymap = {(r.date, r.jcd, r.rno, r.pair): r.pay for r in pay.itertuples()}
    for pr in COMBOS_2F:
        i, j = map(int, pr.split("="))
        q = Pm[:, [k for k, s in enumerate(sets) if i in s and j in s]].sum(1)
        lo = pd.to_numeric(m[f"wide_lo{pr}"], errors="coerce").values; hi = pd.to_numeric(m[f"wide_hi{pr}"], errors="coerce").values
        p = np.array([paymap.get((d, jc, rn, pr), 0) for d, jc, rn in zip(m["date"], m["jcd"], m["rno"])], float)
        rows.append(pd.DataFrame({"date": m["date"], "pair": pr, "q": q, "lo": lo, "hi": hi, "pay": p}))
    d = pd.concat(rows, ignore_index=True); d = d[np.isfinite(d["lo"]) & (d["lo"] > 0) & (d["q"] > 0)]
    out = []
    for thr in (0,) + THRS:
        for basis in ("lo", "hi"):
            if thr == 0 and basis == "hi":
                continue
            s = d[d["q"] * d[basis] > thr] if thr else d
            g = s.groupby("date").agg(st=("pay", "size"), rt=("pay", "sum")); g["st"] *= 100
            n = len(s); roi = g["rt"].sum() / g["st"].sum() if n else np.nan
            if n:
                bi = RNG.integers(0, len(g), size=(3000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
                lo95, hi95 = float(np.quantile(b, .025)), float(np.quantile(b, .975))
            else:
                lo95 = hi95 = np.nan
            out.append({"判定に使うオッズ": {"lo": "下限", "hi": "上限"}[basis], "閾値": thr or "全部", "点数": n, "的中": int((s["pay"] > 0).sum()),
                        "回収率": roi, "95%CI下": lo95, "95%CI上": hi95,
                        "判定": ("採用候補" if (basis == "lo" and thr and n >= 200 and roi >= 1.1 and lo95 > 1.0) else "")})
    R = pd.DataFrame(out)
    nr = m[KEY].drop_duplicates().shape[0]
    md = [f"# IDEA-039 拡連複（ワイド）：3連単の市場から見た割安（{START}〜{END}、{nr}レース、締切時オッズ＝上限）", "",
          "q = 3連単の確定オッズから出した「2艇がともに3着以内」の確率。q × オッズ > 閾値 の組を各100円、払戻は K ファイルの実額。", "",
          R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
