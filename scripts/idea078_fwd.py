"""IDEA-078 前向きの記録：堅いレースで、3連単をモデルの確率の上位4点に絞る（docs/ideas.md の登録どおり。前向きは 2026-11-17 から）
usage: python scripts/idea078_fwd.py DATADIR STOREDIR [--start 20261117] [--end 20270131] [--out results/idea078.md]
- 予想：STORE/live_pred の model=frozen_v4・has_bi=1・predicted_at < 締切 の最初の予想（monitor_model.py と同じ選び方）
- 対象：モデルの1号艇の1着確率（3連単120組を正規化して1着が1号艇の組を合計）≥ 0.60 のレース
- 買い方：モデルの確率の上位4組（同じ確率なら COMBOS の順）を各100円。オッズは使わない
- 払戻：DATADIR/races.parquet の tri_combo・tri_pay（確定データの3連単の実額）。tri_combo が無いレース（不成立・中止）は除く。
  欠場（K0/K1）・フライング（F）・出遅れ（L0/L1）の艇を含む組は返還（100円戻す）。判定は DATADIR/entries.parquet の着順欄（pos_raw）
- 参考（判定に使わない）：上位1・2・3・6・12点、全レースで上位4点、同じレースの1号艇の単勝（K の実額、1号艇が欠場・F・L は返還）
- 判定：採用＝回収率110%以上かつ日単位95%CI下限 > 100%。途中確認は500レースごと（日付→締切時刻の順）で、片側80%上限 < 100% なら打ち切り。
  最大 3,000レースか 2027-01-31 の早い方
- 探索期間の再計算は --start 20261011 --end 20261115 --label 探索 で出せる（判定には使わない）
"""
import argparse, glob, os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from idea052 import k_pay

KEY = ["date", "jcd", "rno"]
P1_MIN, NPTS = 0.60, 4
LOOK, NMAX, END = 500, 3000, "20270131"
REFUND = {"F", "L", "L0", "L1", "K", "K0", "K1"}
RNG = np.random.default_rng(78)
CB = np.array([[int(x) for x in c.split("-")] for c in COMBOS])  # 120×3 の艇番
FIRST1 = CB[:, 0] == 1

def read(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs], ignore_index=True)
    df["jcd"] = df["jcd"].str.zfill(2); df["rno"] = df["rno"].astype(int)
    return df

def bstat(days, st, rt):
    """日単位ブートストラップ：回収率、95%CI、片側80%上限"""
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum()
    if len(g) == 0 or g["st"].sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    roi = g["rt"].sum() / g["st"].sum()
    if len(g) < 2:
        return roi, np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(roi), float(np.quantile(b, .025)), float(np.quantile(b, .975)), float(np.quantile(b, .8))

def load(datadir, store, start, end):
    lp = read(f"{store}/live_pred/*.csv.gz")
    if not len(lp):
        return None
    lp = lp[(lp["model"] == "frozen_v4") & (lp["date"] >= start) & (lp["date"] <= end)]
    lp = lp[(lp["has_bi"] == 1) & (lp["predicted_at"] < lp["deadline"] + ":00")].drop_duplicates(KEY, keep="first")
    r = pd.read_parquet(f"{datadir}/races.parquet", columns=KEY + ["tri_combo", "tri_pay"])
    r["date"] = r["date"].astype(str); r["jcd"] = r["jcd"].astype(str).str.zfill(2); r["rno"] = r["rno"].astype(int)
    m = lp.merge(r, on=KEY)
    m = m[m["tri_combo"].isin(COMBOS) & (m["tri_pay"] > 0)].reset_index(drop=True)
    if not len(m):
        return m, None, None
    e = pd.read_parquet(f"{datadir}/entries.parquet", columns=KEY + ["boat", "pos_raw"])
    e["date"] = e["date"].astype(str); e["jcd"] = e["jcd"].astype(str).str.zfill(2); e["rno"] = e["rno"].astype(int)
    e = e[(e["date"] >= start) & (e["date"] <= end)]
    rf = e[e["pos_raw"].astype(str).isin(REFUND)]
    rb = np.zeros((len(m), 7), bool)  # 返還の艇（1〜6）
    if len(rf):
        idx = m[KEY].reset_index().merge(rf, on=KEY)
        rb[idx["index"].values, idx["boat"].astype(int).values] = True
    kp = k_pay(store, start)
    tan1 = kp[(kp["kind"] == "tan") & (kp["combo"] == "1")].groupby(KEY)["pay"].sum()
    haveT = kp[kp["kind"] == "tan"][KEY].drop_duplicates()
    haveT["haveT"] = True
    m = m.merge(haveT, on=KEY, how="left")
    m["tan1"] = m.set_index(KEY).index.map(tan1).fillna(0).values
    return m, rb, None

def evaluate(m, rb):
    """各レースについて、上位k点の賭け金と払戻（円）・的中"""
    P = m[COMBOS].to_numpy(float); P = P / P.sum(1, keepdims=True)
    p1 = P[:, FIRST1].sum(1)
    order = np.argsort(-P, axis=1, kind="stable")  # 同じ確率なら COMBOS の順
    yi = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    pay = m["tri_pay"].values.astype(float)
    # 各組が返還か（買い目の艇のどれかが返還の艇）
    ref = rb[:, CB[:, 0]] | rb[:, CB[:, 1]] | rb[:, CB[:, 2]]  # 行×120
    out = {}
    for k in (1, 2, 3, 4, 6, 12):
        pick = order[:, :k]
        rows = np.arange(len(m))[:, None]
        is_ref = ref[rows, pick]
        is_hit = (pick == yi[:, None]) & ~is_ref
        ret = (is_hit * pay[:, None]).sum(1) + is_ref.sum(1) * 100.0
        out[k] = dict(st=np.full(len(m), 100.0 * k), rt=ret, hit=is_hit.any(1))
    # 1号艇の単勝（K の行があるレースだけ）
    t_ok = m["haveT"].fillna(False).values.astype(bool)
    t_rt = np.where(rb[:, 1], 100.0, m["tan1"].values.astype(float))
    out["tan1"] = dict(st=np.where(t_ok, 100.0, 0.0), rt=np.where(t_ok, t_rt, 0.0), hit=t_ok & (m["tan1"].values > 0) & ~rb[:, 1])
    return p1, out

def row(name, days, d, mask):
    st, rt, hit = d["st"][mask], d["rt"][mask], d["hit"][mask]
    n = int((st > 0).sum())
    roi, lo, hi, up80 = bstat(days[mask], st, rt)
    nh = int(hit.sum())
    avg = (rt[hit].sum() / nh) if nh else np.nan
    return {"買い方": name, "レース": n, "点数": int(st.sum() / 100), "的中レース": nh, "的中率": nh / n if n else np.nan,
            "当たったときの平均払戻": avg, "回収率": roi, "95%CI下": lo, "95%CI上": hi, "片側80%上限": up80}

def section(m, rb, label):
    if m is None or not len(m):
        return [f"## {label}", "", "対象のレース：0（確定データ（K）がまだ無い）", ""], 0
    p1, out = evaluate(m, rb)
    days = m["date"].values
    firm = p1 >= P1_MIN
    allm = np.ones(len(m), bool)
    rows = [row("**判定：堅いレースで上位4点**", days, out[4], firm)]
    for k in (1, 2, 3, 6, 12):
        rows.append(row(f"参考：堅いレースで上位{k}点", days, out[k], firm))
    rows.append(row("参考：全レースで上位4点", days, out[4], allm))
    rows.append(row("参考：堅いレースの1号艇の単勝", days, out["tan1"], firm & (out["tan1"]["st"] > 0)))
    T = pd.DataFrame(rows)
    nd = len(np.unique(days[firm]))
    md = [f"## {label}", "",
          f"確定データのあるレース {len(m)}（{len(np.unique(days))}日）のうち、堅いレース（モデルの1号艇の1着確率 ≥ {P1_MIN:.0%}）{int(firm.sum())}（1日あたり {firm.sum() / max(nd, 1):.1f}）", "",
          T.to_markdown(index=False, floatfmt=".3f"), ""]
    return md, int(firm.sum())

def looks(m, rb):
    """途中確認：日付→締切時刻の順で500レースごと"""
    if m is None or not len(m):
        return ["## 途中確認（500レースごと、日付→締切時刻の順）", "", "（まだ500レースに届いていない）", ""], None
    p1, out = evaluate(m, rb)
    firm = np.where(p1 >= P1_MIN)[0]
    T = m.iloc[firm][KEY + ["deadline"]].assign(i=firm).sort_values(["date", "deadline", "jcd", "rno"]).reset_index(drop=True)
    md = ["## 途中確認（500レースごと、日付→締切時刻の順）", ""]
    stopped = None
    n = len(T)
    if n < LOOK:
        md += [f"（堅いレース {n}／{LOOK}。まだ最初の確認に届いていない）", ""]
    for k in range(LOOK, min(n, NMAX) + 1, LOOK):
        ii = T["i"].values[:k]
        roi, lo, hi, up80 = bstat(m["date"].values[ii], out[4]["st"][ii], out[4]["rt"][ii])
        last = T.iloc[k - 1]
        fut = up80 < 1.0
        md += [f"- {k}レース時点（{k}番目は {last['date']} {last['jcd']} {int(last['rno'])}R）：回収率 {roi:.3f}（95%CI {lo:.3f}〜{hi:.3f}）、片側80%上限 {up80:.3f} → "
               + ("**打ち切りの基準に当たる（不採用・見込みなし）**" if fut else "続ける")]
        if fut and stopped is None:
            stopped = k
    md += [""]
    return md, stopped

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--start", default="20261117"); ap.add_argument("--end", default=END)
    ap.add_argument("--label", default="前向き"); ap.add_argument("--out", default="results/idea078.md")
    ap.add_argument("--no-explore", dest="explore", action="store_false", help="探索期間（2026-10-11〜11-15）の参考を載せない（既定は載せる。毎朝の指示は引数なしで呼ぶ）")
    a = ap.parse_args()
    res = load(a.datadir, a.store, a.start, a.end)
    m, rb = (res[0], res[1]) if res else (None, None)
    sec, nfirm = section(m, rb, f"{a.label}（{a.start}〜{a.end}）")
    lk, stopped = looks(m, rb) if a.label == "前向き" else ([], None)
    md = ["# IDEA-078 堅いレースで、3連単をモデルの確率の上位4点に絞る", "",
          "登録：docs/ideas.md（2026-11-16、前向きの初日 11-17 の結果が出る前）。予想は live の frozen_v4（直前情報入り・締切前の最初の予想）、オッズは使わない。",
          "払戻は確定データの実額。欠場・F・L の艇を含む組は返還（100円）。判定は「堅いレースで上位4点」だけ。ほかの行は参考。", "",
          "判定の基準：採用＝回収率110%以上かつ日単位95%CI下限 > 100%。途中確認は500レースごと（片側80%上限 < 100% で打ち切り）。最大 3,000レースか 2027-01-31。", ""]
    md += sec + lk
    if a.explore:
        r2 = load(a.datadir, a.store, "20261011", "20261115")
        s2, _ = section(r2[0], r2[1], "探索（20261011〜20261115、4点は結果を見て選んだので判定に使わない）")
        md += s2
    status = "打ち切り（不採用・見込みなし）" if stopped else ("前向きの記録中" if nfirm < NMAX else "最大レース数に到達（最終判定）")
    md += [f"状態：{status}", ""]
    txt = "\n".join(md)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)

if __name__ == "__main__":
    main()
