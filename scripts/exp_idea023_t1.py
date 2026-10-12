"""IDEA-023（と IDEA-022）の本番：締切直前のオッズで判定したとき、3連単の市場から見たずれが残るか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea023_t1.py DATADIR STOREDIR [--lead t1|t0] [--out results/idea023_t1.md]
- 判定：締切直前（live_t1、2〜0.5分前。--lead t0 なら単勝・複勝だけ live_t0 の45〜6秒前）に取得したオッズのみを使う
  3連単の市場の確率 q（締切直前の3連単オッズの逆数を正規化）× その券種の締切直前オッズ > 閾値 {1.1, 1.2, 1.5}
  券種：単勝、複勝（下限オッズで判定）、2連単、2連複、3連複
- 払戻：単勝・2連単・2連複・3連複は締切時オッズ×100円、複勝は競走成績ファイル（K）の公式払戻。欠場艇を含む買い目は返還（100円）
- 判定基準：点数200以上・回収率110%以上・日単位ブートストラップ95%CI下限100%超 → 採用候補。点数200未満は判断に使わない
- 参考：買った目の「締切時オッズ ÷ 判定時オッズ」の中央値
"""
import argparse, glob, gzip, itertools, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, COMBOS_2F, COMBOS_2T, COMBOS_3F
from live_eval import KEY, _read

THRS = (1.1, 1.2, 1.5)
RNG = np.random.default_rng(0)

def before(o):
    return o[o["fetched_at"] < o["deadline"] + ":00"].drop_duplicates(KEY, keep="last") if len(o) else o

def fuku_pay(store, dates):
    rows = []
    for d in sorted(set(dates)):
        p = f"{store}/bk/{d[:4]}/k{d[2:]}.txt.gz"
        if not os.path.exists(p):
            continue
        jcd = rno = None
        for line in gzip.open(p, "rt", encoding="utf-8"):
            m = re.match(r"^(\d\d)KBGN", line)
            if m: jcd = m.group(1); continue
            m = re.match(r"^\s+(\d{1,2})R\s", line)
            if m and ("H1" in line or "m " in line): rno = int(m.group(1)); continue
            m = re.match(r"^\s+複勝\s+(\d)\s+(\d+)(?:\s+(\d)\s+(\d+))?", line)
            if m and jcd and rno:
                rows.append((d, jcd, rno, int(m.group(1)), int(m.group(2))))
                if m.group(3): rows.append((d, jcd, rno, int(m.group(3)), int(m.group(4))))
    return pd.DataFrame(rows, columns=KEY + ["boat", "fpay"]).drop_duplicates(KEY + ["boat"])

def nz(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--lead", default="t1", choices=["t1", "t0"]); ap.add_argument("--out", default=None)
    a = ap.parse_args(); a.out = a.out or f"results/idea023_{a.lead}.md"
    S = a.store
    o3 = before(_read(f"{S}/live_t1/odds3t/*.csv.gz"))
    tf = before(_read(f"{S}/live_{a.lead}/oddstf/*.csv.gz"))
    t2 = before(_read(f"{S}/live_t1/odds2tf/*.csv.gz")); f3 = before(_read(f"{S}/live_t1/odds3f/*.csv.gz"))
    races = nz(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].drop_duplicates(KEY)
    r = races[races["tri_combo"].notna()].merge(o3[KEY], on=KEY).merge(tf[KEY], on=KEY)
    cl3 = nz(pd.read_parquet(f"{a.datadir}/odds.parquet"))
    r = r.merge(cl3[KEY], on=KEY)
    def get(df, cols):
        if len(df) == 0:
            return np.full((len(r), len(cols)), np.nan)
        return r[KEY].merge(df, on=KEY, how="left").reindex(columns=cols).to_numpy(float)
    O3 = get(o3, COMBOS)
    ok = (np.isfinite(O3) & (O3 > 0)).sum(1) >= 60
    r = r[ok].reset_index(drop=True); O3 = O3[ok]
    if len(r) == 0:
        print("対象レースなし"); open(a.out, "w").write("対象レースなし\n"); return
    inv = np.where(np.isfinite(O3) & (O3 > 0), 1 / np.where(O3 > 0, O3, 1), 0); p3 = inv / inv.sum(1, keepdims=True)
    ci = {c: i for i, c in enumerate(COMBOS)}
    PI = np.array([int(c[0]) for c in COMBOS]); PJ = np.array([int(c[2]) for c in COMBOS])
    qwin = np.stack([p3[:, PI == b].sum(1) for b in range(1, 7)], 1)
    qtop2 = np.stack([p3[:, (PI == b) | (PJ == b)].sum(1) for b in range(1, 7)], 1)
    q2t = np.stack([p3[:, (PI == x) & (PJ == y)].sum(1) for x, y in (map(int, c.split("-")) for c in COMBOS_2T)], 1)
    i2t = {c: i for i, c in enumerate(COMBOS_2T)}
    q2f = np.stack([q2t[:, i2t[f"{x}-{y}"]] + q2t[:, i2t[f"{y}-{x}"]] for x, y in (map(int, c.split("=")) for c in COMBOS_2F)], 1)
    q3f = np.stack([sum(p3[:, ci["-".join(map(str, pp))]] for pp in itertools.permutations(map(int, c.split("=")))) for c in COMBOS_3F], 1)
    # 締切時
    cltf = nz(_read(f"{S}/oddstf/*.csv.gz")).drop_duplicates(KEY, keep="last")
    cl2 = nz(pd.read_parquet(f"{a.datadir}/odds2tf.parquet")); cl3f = nz(pd.read_parquet(f"{a.datadir}/odds3f.parquet"))
    fk = fuku_pay(S, r["date"].values)
    fpay = fk.pivot_table(index=KEY, columns="boat", values="fpay", aggfunc="first").reindex(columns=range(1, 7))
    fpay = fpay.reindex(pd.MultiIndex.from_frame(r[KEY])).to_numpy(float)
    e = nz(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "pos_raw"]))
    absent = e.assign(x=e["pos_raw"].astype(str).str.match(r"^K")).pivot_table(index=KEY, columns="boat", values="x", aggfunc="max")
    absent = absent.reindex(columns=range(1, 7)).reindex(pd.MultiIndex.from_frame(r[KEY])).fillna(False).to_numpy(bool)
    win = r["tri_combo"].str.split("-", expand=True).astype(int).to_numpy()
    days = r["date"].values
    tan_c = [f"tan{b}" for b in range(1, 7)]; fuku_c = [f"fuku_lo{b}" for b in range(1, 7)]
    K = {
        "単勝": (qwin, get(tf, tan_c), get(cltf, tan_c), [[b] for b in range(1, 7)], lambda bs, w: bs[0] == w[0]),
        "複勝": (qtop2, get(tf, fuku_c), None, [[b] for b in range(1, 7)], lambda bs, w: bs[0] in w[:2]),
        "2連単": (q2t, get(t2, COMBOS_2T), get(cl2, COMBOS_2T), [list(map(int, c.split("-"))) for c in COMBOS_2T], lambda bs, w: bs == list(w[:2])),
        "2連複": (q2f, get(t2, COMBOS_2F), get(cl2, COMBOS_2F), [list(map(int, c.split("="))) for c in COMBOS_2F], lambda bs, w: set(bs) == set(w[:2])),
        "3連複": (q3f, get(f3, COMBOS_3F), get(cl3f, COMBOS_3F), [list(map(int, c.split("="))) for c in COMBOS_3F], lambda bs, w: set(bs) == set(w[:3])),
    }
    rows = []
    for k, (q, Od, Oc, boats, hitf) in K.items():
        if a.lead == "t0" and k not in ("単勝", "複勝"):
            continue
        hit = np.array([[hitf(bs, w) for bs in boats] for w in win])
        refund = np.stack([absent[:, [b - 1 for b in bs]].any(1) for bs in boats], 1)
        if k == "複勝":
            pay = np.where(hit, fpay, 0.0); paid_ok = np.isfinite(fpay).any(1)[:, None]
        else:
            pay = np.where(hit, Oc * 100, 0.0); paid_ok = (np.isfinite(Oc) & (Oc > 0)).any(1)[:, None]
        valid = np.isfinite(Od) & (Od > 0) & paid_ok
        for thr in THRS:
            buy = valid & (q * np.nan_to_num(Od) > thr)
            ret = np.where(buy & refund, 100.0, np.where(buy, np.nan_to_num(pay), 0.0))
            n = int(buy.sum())
            g = pd.DataFrame({"d": days, "st": buy.sum(1) * 100.0, "rt": ret.sum(1)}).groupby("d").sum()
            if n:
                bi = RNG.integers(0, len(g), size=(3000, len(g)))
                b = g["rt"].values[bi].sum(1) / np.clip(g["st"].values[bi].sum(1), 1, None)
                roi, lo, hi = g["rt"].sum() / g["st"].sum(), float(np.quantile(b, .025)), float(np.quantile(b, .975))
            else:
                roi = lo = hi = np.nan
            drift = float(np.nanmedian((Oc / Od)[buy])) if (n and Oc is not None) else np.nan
            rows.append({"券種": k, "閾値": thr, "点数": n, "的中": int((buy & hit).sum()), "判定時の平均オッズ": round(float(np.nan_to_num(Od)[buy].mean()), 1) if n else np.nan,
                         "回収率": roi, "95%CI下": lo, "95%CI上": hi, "締切時/判定時オッズ（中央値）": drift,
                         "判定": ("点数不足" if n < 200 else ("採用候補" if (roi >= 1.1 and lo > 1.0) else "不採用"))})
    md = [f"### IDEA-022/023 締切直前（{a.lead}）で判定（{r['date'].min()}〜{r['date'].max()}、{len(r)}レース、{len(set(days))}日）", "",
          pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
