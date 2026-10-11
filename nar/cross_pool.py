"""地方競馬で「3連単の市場を答えにして、他の券種のずれを拾う」（IDEA-024、ボートの IDEA-022/023 と同じ考え方）。**ローカル専用**。
usage: python nar/cross_pool.py [--data nar_data] [--out results/nar_cross_pool.md]
- 3連単の確定オッズの逆数をレースごとに正規化 → 3連単の市場の確率 p(a-b-c)
- そこから各券種の確率を出す：単勝 = a が1着、馬単 = a-b の順で1・2着、馬連 = {a,b} が1・2着、
  複勝 = 3着以内（出走 7頭以下は 2着以内）、ワイド = {a,b} がともに3着以内
- 条件：その確率 × その券種の確定オッズ > 閾値（{1.2, 1.5, 2.0}、事前に固定）の買い目を各100円
- 払戻：単勝・馬単・馬連は 確定オッズ×100円（正確）。複勝・ワイドは幅のあるオッズの「下限」で判定し、
  払戻も下限×100円（＝回収率の下限）と上限×100円（＝回収率の上限）の両方を出す
- 取消・除外の馬を含む買い目は除く。期間は確定オッズのある 2026-03〜2026-08（テスト期間 2026-09 以降は使わない）
- 締切時オッズで判定・購入できたとしたらの上限。締切前のオッズの動きは別に確かめる必要がある
出力は集計値のみ（results/ に置いて共有してよい）
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build import keycols, num, read_zip_csvs

KEY = ["venue", "date", "rno"]
THRS = (1.2, 1.5, 2.0)
START, END = "20260301", "20260831"
KINDS = {"単勝": r"^単勝$", "複勝": r"^複勝$", "馬単": r"^馬単$", "馬連": r"^馬連$|^馬複$|^馬連複$", "ワイド": r"ワイド|拡大馬連"}
RNG = np.random.default_rng(0)

def norm_kind(s):
    s = s.str.strip().str.replace("３", "3").str.replace("三", "3")
    out = pd.Series("", index=s.index)
    out[s.str.contains("3連単")] = "3連単"
    for k, pat in KINDS.items():
        out[(out == "") & s.str.contains(pat)] = k
    return out

def month_tables(df, field):
    """1か月分のオッズ表 → 券種ごとの (KEY + 買い目 + q + オッズ下限・上限) の表"""
    k = norm_kind(df["賭式"])
    base = pd.concat([keycols(df), pd.DataFrame({"kind": k, "n1": num(df["番号1"]), "n2": num(df["番号2"]), "n3": num(df["番号3"]),
                                                  "lo": num(df["オッズ"]), "hi": num(df["オッズ（最大）"]) if "オッズ（最大）" in df else np.nan})], axis=1)
    base = base[(base["kind"] != "") & (base["date"] >= START) & (base["date"] <= END)]
    t = base[base["kind"] == "3連単"].dropna(subset=["n1", "n2", "n3", "lo"])
    t = t[t["lo"] > 0].drop_duplicates(KEY + ["n1", "n2", "n3"], keep="last").copy()
    t["p"] = 1 / t["lo"]; t["p"] /= t.groupby(KEY)["p"].transform("sum")
    t = t.merge(field, on=KEY, how="left")
    out = {}
    win = t.groupby(KEY + ["n1"])["p"].sum().rename("q").reset_index().rename(columns={"n1": "a"})
    ex = t.groupby(KEY + ["n1", "n2"])["p"].sum().rename("q").reset_index().rename(columns={"n1": "a", "n2": "b"})
    qu = ex.assign(a=ex[["a", "b"]].min(1), b=ex[["a", "b"]].max(1)).groupby(KEY + ["a", "b"])["q"].sum().reset_index()
    small = t["field"].fillna(99) <= 7
    pl_parts = [t[[*KEY, c, "p"]].rename(columns={c: "a"}) for c in ("n1", "n2")] + [t.loc[~small, [*KEY, "n3", "p"]].rename(columns={"n3": "a"})]
    pl = pd.concat(pl_parts).groupby(KEY + ["a"])["p"].sum().rename("q").reset_index()
    wd = pd.concat([t[[*KEY, x, y, "p"]].rename(columns={x: "a", y: "b"}) for x, y in (("n1", "n2"), ("n1", "n3"), ("n2", "n3"))])
    wd = wd.assign(a=wd[["a", "b"]].min(1), b=wd[["a", "b"]].max(1)).groupby(KEY + ["a", "b"])["p"].sum().rename("q").reset_index()
    probs = {"単勝": (win, ["a"]), "複勝": (pl, ["a"]), "馬単": (ex, ["a", "b"]), "馬連": (qu, ["a", "b"]), "ワイド": (wd, ["a", "b"])}
    for kind, (q, cols) in probs.items():
        o = base[base["kind"] == kind].dropna(subset=["n1", "lo"]).copy()
        o["a"] = o["n1"]
        if len(cols) == 2:
            o["b"] = o["n2"]
            if kind != "馬単":
                o["a"], o["b"] = o[["n1", "n2"]].min(1), o[["n1", "n2"]].max(1)
        o = o[o["lo"] > 0].drop_duplicates(KEY + cols, keep="last")
        out[kind] = o[KEY + cols + ["lo", "hi"]].merge(q, on=KEY + cols, how="inner")
    return out

def settle(kind, d, res, scr):
    """的中の判定。res: KEY → 1〜3着の馬番と頭数、scr: 取消・除外の馬（KEY + num）"""
    d = d.merge(res, on=KEY, how="inner")
    bad = np.zeros(len(d), bool)
    for c in ("a", "b"):
        if c in d:
            m = d[KEY + [c]].reset_index().merge(scr.rename(columns={"num": c}), on=KEY + [c], how="inner")["index"]
            bad[m.values] = True
    d = d[~bad].reset_index(drop=True)
    a = d["a"].astype(int); b = d["b"].astype(int) if "b" in d else None
    if kind == "単勝":
        hit = a.values == d["p1"].values
    elif kind == "馬単":
        hit = (a.values == d["p1"].values) & (b.values == d["p2"].values)
    elif kind == "馬連":
        hit = (np.minimum(d["p1"], d["p2"]).values == a.values) & (np.maximum(d["p1"], d["p2"]).values == b.values)
    elif kind == "複勝":
        top = np.where((d["field"].fillna(99) <= 7).values[:, None], d[["p1", "p2"]].assign(x=-1).values, d[["p1", "p2", "p3"]].values)
        hit = (top == a.values[:, None]).any(1)
    else:  # ワイド
        top = d[["p1", "p2", "p3"]].values
        hit = (top == a.values[:, None]).any(1) & (top == b.values[:, None]).any(1)
    d["hit"] = hit
    return d

def summarize(kind, d):
    rows = []
    d = d.copy(); d["ev"] = d["q"] * d["lo"]
    hi = d["hi"].where(d["hi"].notna() & (d["hi"] > 0), d["lo"])
    for thr in (0.0,) + THRS:
        s = d[d["ev"] > thr] if thr else d
        n = len(s)
        if n == 0:
            rows.append({"券種": kind, "閾値": thr or "全買い目", "点数": 0}); continue
        pay_lo = np.where(s["hit"], s["lo"] * 100, 0.0); pay_hi = np.where(s["hit"], hi[s.index] * 100, 0.0)
        g = pd.DataFrame({"d": s["date"].values, "st": 100.0, "rt": pay_lo}).groupby("d").sum()
        bi = RNG.integers(0, len(g), size=(3000, len(g)))
        boot = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
        roi = pay_lo.sum() / (n * 100); lo95 = float(np.quantile(boot, .025))
        rows.append({"券種": kind, "閾値": thr or "全買い目", "点数": n, "1日あたり": round(n / max(len(g), 1), 1), "的中率": round(float(s["hit"].mean()), 3),
                     "平均オッズ（下限）": round(float(s["lo"].mean()), 1), "市場の確率の平均": round(float(s["q"].mean()), 3),
                     "回収率": round(roi, 3), "95%CI下": round(lo95, 3), "95%CI上": round(float(np.quantile(boot, .975)), 3),
                     "回収率（オッズ上限で払戻）": round(pay_hi.sum() / (n * 100), 3) if kind in ("複勝", "ワイド") else "",
                     "判定": ("採用候補" if (thr and n >= 200 and roi >= 1.1 and lo95 > 1.0) else ("" if not thr else "不採用"))})
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="nar_data"); ap.add_argument("--out", default="results/nar_cross_pool.md")
    a = ap.parse_args()
    R = pd.read_parquet(f"{a.data}/races.parquet", columns=KEY + ["field"])
    H = pd.read_parquet(f"{a.data}/horses.parquet", columns=KEY + ["num", "pos", "pos_raw"])
    H = H[(H["date"] >= START) & (H["date"] <= END)]
    scr = H[H["pos"].isna() & H["pos_raw"].str.contains("取|除", na=False)][KEY + ["num"]].astype({"num": float}).drop_duplicates()
    t3 = H[H["pos"].between(1, 3)]
    ok = t3.groupby(KEY)["pos"].agg(lambda s: sorted(s.tolist()) == [1, 2, 3])  # 同着などで1〜3着がそろわないレースは除く
    t3 = t3.merge(ok[ok].reset_index()[KEY], on=KEY)
    res = t3.pivot_table(index=KEY, columns="pos", values="num", aggfunc="first")
    res.columns = ["p1", "p2", "p3"]; res = res.reset_index().merge(R, on=KEY, how="left")
    acc = {k: [] for k in KINDS}
    kinds_seen = set()
    for p in sorted(glob.glob(f"{a.data}/odds/*.zip")):
        for n, df in read_zip_csvs(p).items():
            kinds_seen |= set(df["賭式"].str.strip().unique())
            for k, t in month_tables(df, R).items():
                acc[k].append(settle(k, t, res, scr))
        print("done", os.path.basename(p), flush=True)
    rows = []
    for k in KINDS:
        if acc[k]:
            d = pd.concat(acc[k], ignore_index=True)
            print(k, len(d), "買い目", d[KEY].drop_duplicates().shape[0], "レース", flush=True)
            rows += summarize(k, d)
    t = pd.DataFrame(rows)
    md = [f"# IDEA-024 地方競馬：3連単の市場から見た各券種のずれ（{START}〜{END}、確定オッズ）", "",
          "3連単の確定オッズから出した確率 × 各券種の確定オッズ > 閾値 の買い目を各100円。**締切時オッズで判定・購入できたとしたらの上限**。",
          "複勝・ワイドの「回収率」はオッズ下限で払戻（下限）、「回収率（オッズ上限で払戻）」は上限。", "",
          f"オッズファイルの賭式の値: {sorted(kinds_seen)}", "", t.to_markdown(index=False)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main()
