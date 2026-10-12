"""IDEA-027: 締切直前の様子から確定オッズを予測して、単勝・2連複のずれを買う（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea027.py DATADIR STOREDIR [--lead t0] [--n 900] [--out results/idea027.md]
- 判定時点 lead（既定 t0 = 締切45〜6秒前）までに取得したデータのみを使う
  q      : 判定時点の3連単の票数（BOATCAST bc_votes）から出した市場の確率（票の割合）→ 単勝・2連複の確率に集計
  判定時オッズ : 判定時点の票数から 0.75×合計÷票数（単勝 bc_votes1、2連複 bc_votes2）
- 予測するもの: log(確定オッズ ÷ 判定時オッズ)。LightGBM 回帰（既定値のまま）
  特徴量: 判定時の log オッズ、log q、log(q×オッズ)、票の割合、券種の合計票数(log)、3連単の合計票数(log)、券種の合計÷3連単の合計、人気順位、場、レース番号、締切時刻（時）
- 買い方: q × 予測した確定オッズ > 閾値 {1.1, 1.2, 1.5}、各100円。払戻は確定オッズ。ただし自分の1票を足して計算し直す
  （確定票数があれば 0.75×(合計+1)÷(票数+1)、なければ 確定オッズ×票数/(票数+1) の近似）。欠場艇を含む買い目は返還
- 日付・締切時刻の順に並べ先頭 n レース。時系列の交差検証：先頭 300 レースのあと、日ごとに「その日より前の全レース」で学習してその日を評価
  （2026-10-12 変更、データを見る前。以前は前半で学習・後半で評価の1回だけ）
- 判定: 券種・閾値ごとに 点数200以上・回収率110%以上・日単位ブートストラップ95%CI下限100%超 → 採用候補
  比較: 予測なし（判定時オッズのまま q×判定時オッズ > 閾値）の同じ買い方
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, COMBOS_2F
from live_eval import KEY, _read

THRS = (1.1, 1.2, 1.5)
WARM = 300  # 最初の評価日の前に必要なレース数
RNG = np.random.default_rng(0)
PI = np.array([int(c[0]) for c in COMBOS]); PJ = np.array([int(c[2]) for c in COMBOS])
PAIRS = [tuple(map(int, c.split("="))) for c in COMBOS_2F]

def before(o):
    return o[o["fetched_at"] < o["deadline"] + ":00"].drop_duplicates(KEY, keep="last") if len(o) else o

def nz(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2))

def long_table(r, q, Od, Oc, votes, total, ctotal, cvotes, t3, kind, units, hitm, refund):
    n, k = q.shape
    df = pd.DataFrame({
        "ri": np.repeat(np.arange(n), k), "unit": np.tile(np.arange(k), n), "kind": kind,
        "q": q.ravel(), "od": Od.ravel(), "oc": Oc.ravel(), "v": votes.ravel(), "cv": cvotes.ravel(),
        "tot": np.repeat(total, k), "ctot": np.repeat(ctotal, k), "t3": np.repeat(t3, k),
        "hit": hitm.ravel(), "refund": refund.ravel()})
    df["date"] = np.repeat(r["date"].values, k); df["dl"] = np.repeat(r["deadline"].values, k)
    df["jcd"] = np.repeat(r["jcd"].astype(int).values, k); df["rno"] = np.repeat(r["rno"].astype(int).values, k)
    df["rank"] = df.groupby("ri")["od"].rank(method="min")
    return df

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--lead", default="t0"); ap.add_argument("--n", type=int, default=900); ap.add_argument("--warm", type=int, default=WARM); ap.add_argument("--out", default="results/idea027.md")
    a = ap.parse_args(); S = a.store; D = f"live_{a.lead}"
    v3 = before(_read(f"{S}/{D}/bc_votes/*.csv.gz")); v1 = before(_read(f"{S}/{D}/bc_votes1/*.csv.gz")); v2 = before(_read(f"{S}/{D}/bc_votes2/*.csv.gz"))
    if not len(v3) or not len(v1) or not len(v2):
        msg = f"実行前: {D} に bc_votes / bc_votes1 / bc_votes2 がそろっていない"; print(msg); open(a.out, "w").write(msg + "\n"); return
    races = nz(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].drop_duplicates(KEY)
    ctf = nz(_read(f"{S}/oddstf/*.csv.gz")).drop_duplicates(KEY, keep="last")
    c2 = nz(pd.read_parquet(f"{a.datadir}/odds2tf.parquet"))
    cv1 = nz(_read(f"{S}/bc_votes1/*.csv.gz")).drop_duplicates(KEY, keep="last") if glob_exists(f"{S}/bc_votes1") else pd.DataFrame(columns=KEY)
    cv2 = nz(_read(f"{S}/bc_votes2/*.csv.gz")).drop_duplicates(KEY, keep="last") if glob_exists(f"{S}/bc_votes2") else pd.DataFrame(columns=KEY)
    r = races[races["tri_combo"].notna()].merge(v3[KEY + ["deadline", "votes_total"]], on=KEY).merge(v1[KEY], on=KEY).merge(v2[KEY], on=KEY)
    r = r.merge(ctf[KEY], on=KEY).merge(c2[KEY], on=KEY)
    r = r[pd.to_numeric(r["votes_total"], errors="coerce") > 0].reset_index(drop=True)
    r = r.iloc[np.lexsort((r["rno"].values, r["jcd"].values, r["deadline"].values, r["date"].values))].reset_index(drop=True)
    if len(r) < a.n:
        msg = f"実行前: 対象レース {len(r)} / {a.n}"; print(msg); open(a.out, "w").write(msg + "\n"); return
    r = r.iloc[:a.n].reset_index(drop=True)
    get = lambda df, cols: r[KEY].merge(df, on=KEY, how="left").reindex(columns=cols).to_numpy(float)
    V3 = get(v3, [f"v{c}" for c in COMBOS]); T3 = V3.sum(1)
    p3 = V3 / np.clip(T3[:, None], 1, None)
    qwin = np.stack([p3[:, PI == b].sum(1) for b in range(1, 7)], 1)
    q2f = np.stack([p3[:, ((PI == x) & (PJ == y)) | ((PI == y) & (PJ == x))].sum(1) for x, y in PAIRS], 1)
    e = nz(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "pos_raw"]))
    absent = e.assign(x=e["pos_raw"].astype(str).str.match(r"^K")).pivot_table(index=KEY, columns="boat", values="x", aggfunc="max")
    absent = absent.reindex(columns=range(1, 7)).reindex(pd.MultiIndex.from_frame(r[KEY])).fillna(False).to_numpy(bool)
    win = r["tri_combo"].str.split("-", expand=True).astype(int).to_numpy()
    def odds_from(votes, tot):
        return np.where(votes > 0, np.floor(0.75 * tot[:, None] / np.where(votes > 0, votes, 1) * 10) / 10, np.nan)
    # 単勝
    tv = get(v1, [f"tan_v{b}" for b in range(1, 7)]); tt = get(v1, ["tan_total"])[:, 0]
    ctv = get(cv1, [f"tan_v{b}" for b in range(1, 7)]) if len(cv1) else np.full_like(tv, np.nan)
    ctt = get(cv1, ["tan_total"])[:, 0] if len(cv1) else np.full(len(r), np.nan)
    tan = long_table(r, qwin, odds_from(tv, tt), get(ctf, [f"tan{b}" for b in range(1, 7)]), tv, tt, ctt, ctv, T3, "単勝", None,
                     np.stack([win[:, 0] == b for b in range(1, 7)], 1), absent)
    # 2連複
    qv = get(v2, [f"q2f_v{c}" for c in COMBOS_2F]); qt = get(v2, ["q2f_total"])[:, 0]
    cqv = get(cv2, [f"q2f_v{c}" for c in COMBOS_2F]) if len(cv2) else np.full_like(qv, np.nan)
    cqt = get(cv2, ["q2f_total"])[:, 0] if len(cv2) else np.full(len(r), np.nan)
    hit2 = np.stack([(np.minimum(win[:, 0], win[:, 1]) == x) & (np.maximum(win[:, 0], win[:, 1]) == y) for x, y in PAIRS], 1)
    ref2 = np.stack([absent[:, x - 1] | absent[:, y - 1] for x, y in PAIRS], 1)
    qui = long_table(r, q2f, odds_from(qv, qt), get(c2, COMBOS_2F), qv, qt, cqt, cqv, T3, "2連複", None, hit2, ref2)
    df = pd.concat([tan, qui], ignore_index=True)
    df = df[np.isfinite(df["od"]) & (df["od"] > 0) & np.isfinite(df["oc"]) & (df["oc"] > 0) & (df["q"] > 0)].copy()
    df["y"] = np.log(df["oc"] / df["od"])
    df["f_lo"] = np.log(df["od"]); df["f_lq"] = np.log(df["q"]); df["f_ev"] = np.log(df["q"] * df["od"])
    df["f_share"] = df["v"] / df["tot"]; df["f_ltot"] = np.log1p(df["tot"]); df["f_lt3"] = np.log1p(df["t3"])
    df["f_ratio"] = df["tot"] / df["t3"]; df["f_kind"] = (df["kind"] == "2連複").astype(int)
    df["f_hour"] = df["dl"].str[:2].astype(int)
    F = ["f_lo", "f_lq", "f_ev", "f_share", "f_ltot", "f_lt3", "f_ratio", "f_kind", "rank", "jcd", "rno", "f_hour"]
    # 逐次の評価（時系列の交差検証）：先頭 WARM レースのあと、日ごとに「その日より前の全レース」で学習してその日を予測する
    import lightgbm as lgb
    rdays = r["date"].values; parts = []
    for dd in sorted(set(rdays)):
        ridx = np.where(rdays == dd)[0]
        if ridx[0] < a.warm:
            continue
        tr = df[df["ri"] < ridx[0]]; te = df[df["ri"].isin(ridx)].copy()
        if len(te) == 0:
            continue
        te["pred"] = lgb.LGBMRegressor(verbose=-1).fit(tr[F], tr["y"]).predict(te[F]); parts.append(te)
    if not parts:
        msg = f"実行前: 先頭 {WARM} レースのあとに評価できる日がない"; print(msg); open(a.out, "w").write(msg + "\n"); return
    ev = pd.concat(parts); h = int(ev["ri"].min())
    ev["oc_hat"] = ev["od"] * np.exp(ev["pred"])
    # 自分の1票を足した払戻オッズ
    adj = np.where(np.isfinite(ev["cv"]) & np.isfinite(ev["ctot"]) & (ev["cv"] >= 0),
                   np.floor(0.75 * (ev["ctot"] + 1) / (ev["cv"] + 1) * 10) / 10, ev["oc"] * ev["v"].clip(lower=1) / (ev["v"].clip(lower=1) + 1))
    ev["pay"] = np.where(ev["refund"], 100.0, np.where(ev["hit"], adj * 100, 0.0))
    corr = float(np.corrcoef(ev["pred"], ev["y"])[0, 1])
    rows = []
    for kind in ("単勝", "2連複"):
        for name, col in (("予測あり", "oc_hat"), ("予測なし", "od")):
            for thr in THRS:
                s = ev[(ev["kind"] == kind) & (ev["q"] * ev[col] > thr)]
                n = len(s)
                if n:
                    g = s.groupby("date").agg(st=("pay", "size"), rt=("pay", "sum")); g["st"] *= 100.0
                    bi = RNG.integers(0, len(g), size=(3000, len(g)))
                    b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
                    roi, lo, hi = g["rt"].sum() / g["st"].sum(), float(np.quantile(b, .025)), float(np.quantile(b, .975))
                else:
                    roi = lo = hi = np.nan
                rows.append({"券種": kind, "方法": name, "閾値": thr, "点数": n, "的中": int(s["hit"].sum()) if n else 0, "回収率": roi, "95%CI下": lo, "95%CI上": hi,
                             "判定": "点数不足" if n < 200 else ("採用候補" if (roi >= 1.1 and lo > 1.0) else "不採用")})
    md = [f"### IDEA-027 確定オッズの予測（判定 {a.lead}、{a.n}レース、日ごとの逐次評価 {r['date'].iloc[h]}〜{r['date'].iloc[-1]}、評価 {ev['ri'].nunique()}レース）", "",
          f"予測と実際の log(確定÷判定時) の相関（評価）: {corr:.3f}", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

def glob_exists(d):
    import glob
    return bool(glob.glob(f"{d}/*.csv.gz"))

if __name__ == "__main__":
    main()
