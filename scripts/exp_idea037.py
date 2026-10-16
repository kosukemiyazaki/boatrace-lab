"""IDEA-037 ミッドナイト（締切21:30以降）は市場が甘いか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea037.py DATADIR [--out results/idea037.md]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

KEY = ["date", "jcd", "rno"]
BANDS = [0, 10, 30, 100, 300, np.inf]; NAMES = ["〜10", "10〜30", "30〜100", "100〜300", "300〜"]
RNG = np.random.default_rng(0)

def norm(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2), rno=df["rno"].astype(int))

def beta(L, y):
    def nll(b):
        s = b * L; s = s - s.max(1, keepdims=True); return -(s[np.arange(len(y)), y] - np.log(np.exp(s).sum(1))).mean()
    return minimize_scalar(nll, bounds=(0.5, 1.6), method="bounded").x

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("--out", default="results/idea037.md")
    a = ap.parse_args()
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= "20260831")]
    dl = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["deadline"])).drop_duplicates(KEY)
    m = r.merge(dl, on=KEY).merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY)
    hm = m["deadline"].astype(str).str[:5]
    m["grp"] = np.where(hm >= "21:30", "ミッドナイト", np.where(hm >= "18:00", "ナイター（18:00〜21:29）", ""))
    m = m[m["grp"] != ""].reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); ok = v.sum(1) >= 60
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values; ok &= ~pd.isna(y)
    m, O, v, y = m[ok].reset_index(drop=True), O[ok], v[ok], y[ok].astype(int)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); L = np.log(np.clip(P, 1e-9, None))
    hit = np.zeros_like(v); hit[np.arange(len(y)), y] = True
    band = np.digitize(np.where(v, O, np.nan), BANDS[1:-1])
    out, rows = {}, []
    for g in ("ミッドナイト", "ナイター（18:00〜21:29）"):
        idx = np.where(m["grp"].values == g)[0]; days = m["date"].values[idx]
        b = beta(L[idx], y[idx]); ud = np.unique(days); bs = []
        for _ in range(200):
            pick = RNG.choice(ud, len(ud)); ii = np.concatenate([idx[days == d] for d in pick])
            bs.append(beta(L[ii], y[ii]))
        out[g] = (b, np.quantile(bs, .025), np.quantile(bs, .975), len(idx), len(ud))
        for k in range(5):
            sel = v[idx] & (band[idx] == k)
            st = sel.sum(1) * 100.0; rt = np.where(sel & hit[idx], O[idx] * 100, 0).sum(1)
            gd = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); gd = gd[gd["st"] > 0]
            bi = RNG.integers(0, len(gd), size=(3000, len(gd))); bb = gd["rt"].values[bi].sum(1) / gd["st"].values[bi].sum(1)
            rows.append({"区分": g, "オッズ帯": NAMES[k], "点数": int(sel.sum()), "回収率": gd["rt"].sum() / gd["st"].sum(),
                         "95%CI下": float(np.quantile(bb, .025)), "95%CI上": float(np.quantile(bb, .975))})
    R = pd.DataFrame(rows)
    cand = R[(R["区分"] == "ミッドナイト") & (R["95%CI下"] > 1.0)]
    md = ["# IDEA-037 ミッドナイト（締切21:30以降）の市場", "",
          f"**判定：{'採用候補（新しいデータで再確認）' if len(cand) else '不採用（回収率が100%を超えるオッズ帯なし）'}**", ""]
    for g, (b, lo, hi, n, nd) in out.items():
        md.append(f"- {g}：{n}レース・{nd}日、β = {b:.3f}（95%CI {lo:.3f}〜{hi:.3f}）")
    md += ["", R.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
