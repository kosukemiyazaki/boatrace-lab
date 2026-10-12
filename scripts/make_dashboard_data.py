"""ボート券種ずれモニター（ダッシュボード）のデータを作る。
usage: python scripts/make_dashboard_data.py DATADIR STOREDIR OUT.json [--days 14]
1行=1買い目（単勝6通り・2連複15通り）× レース。締切2〜0.5分前（live_t1）のオッズと3連単の市場から出した公正オッズ、
締切時オッズと締切時の3連単から見た期待値、的中。締切時オッズがまだないレース（当日）は締切時の列が空。
仮想売買のルール：締切2分前の 期待値（3連単の確率×オッズ）> 1.1 の買い目を各100円、払戻は締切時オッズ（IDEA-023 と同じ）。
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, COMBOS_2F
from live_eval import KEY, _read

VENUE = dict(zip([f"{i:02d}" for i in range(1, 25)], "桐生 戸田 江戸川 平和島 多摩川 浜名湖 蒲郡 常滑 津 三国 びわこ 住之江 尼崎 鳴門 丸亀 児島 宮島 徳山 下関 若松 芦屋 福岡 唐津 大村".split()))
PI = np.array([int(c[0]) for c in COMBOS]); PJ = np.array([int(c[2]) for c in COMBOS])
PAIRS = [tuple(map(int, c.split("="))) for c in COMBOS_2F]

def before(o):
    return o[o["fetched_at"] < o["deadline"] + ":00"].drop_duplicates(KEY, keep="last") if len(o) else o

def probs(O):
    inv = np.where(np.isfinite(O) & (O > 0), 1 / np.where(O > 0, O, 1), 0)
    p = inv / np.clip(inv.sum(1, keepdims=True), 1e-12, None)
    win = np.stack([p[:, PI == b].sum(1) for b in range(1, 7)], 1)
    q2f = np.stack([p[:, ((PI == x) & (PJ == y)) | ((PI == y) & (PJ == x))].sum(1) for x, y in PAIRS], 1)
    return win, q2f

def nz(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("out"); ap.add_argument("--days", type=int, default=14)
    a = ap.parse_args(); S = a.store
    o3 = before(_read(f"{S}/live_t1/odds3t/*.csv.gz")); tf = before(_read(f"{S}/live_t1/oddstf/*.csv.gz")); t2 = before(_read(f"{S}/live_t1/odds2tf/*.csv.gz"))
    r = o3[KEY + ["deadline"]].merge(tf[KEY], on=KEY).merge(t2[KEY], on=KEY)
    days = sorted(r["date"].unique())[-a.days:]
    r = r[r["date"].isin(days)].sort_values(["date", "deadline", "jcd", "rno"]).reset_index(drop=True)
    get = lambda df, cols: r[KEY].merge(df, on=KEY, how="left").reindex(columns=cols).to_numpy(float)
    w1, q1 = probs(get(o3, COMBOS))
    c3 = nz(pd.read_parquet(f"{a.datadir}/odds.parquet")); wc, qc = probs(get(c3, COMBOS))
    ctf = nz(_read(f"{S}/oddstf/*.csv.gz")).drop_duplicates(KEY, keep="last")
    c2 = nz(_read(f"{S}/odds2tf/*.csv.gz")).drop_duplicates(KEY, keep="last")
    T1, T2 = get(tf, [f"tan{b}" for b in range(1, 7)]), get(t2, COMBOS_2F)
    TC, C2 = get(ctf, [f"tan{b}" for b in range(1, 7)]), get(c2, COMBOS_2F)
    races = nz(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].drop_duplicates(KEY)
    res = _read(f"{S}/live_result/*.csv.gz")
    live_top = {}
    if len(res):
        res = res[res["pos"].isin([1.0, 2.0])].drop_duplicates(KEY + ["pos"])
        for k, g in res.groupby(KEY):
            if len(g) == 2:
                live_top[k] = [int(g.loc[g["pos"] == 1.0, "boat"].iloc[0]), int(g.loc[g["pos"] == 2.0, "boat"].iloc[0])]
    tri = dict(zip(zip(races["date"], races["jcd"], races["rno"]), races["tri_combo"]))
    rows = []
    for i, x in r.iterrows():
        k = (x["date"], x["jcd"], int(x["rno"]))
        t = tri.get(k); top = [int(t[0]), int(t[2])] if isinstance(t, str) else live_top.get(k)
        closed = np.isfinite(C2[i]).any() and np.isfinite(qc[i]).any() and qc[i].sum() > 0
        def add(kind, combo, o1, qq1, oc, qqc, hit):
            ev1 = qq1 * o1 if np.isfinite(o1) and o1 > 0 else None
            evc = qqc * oc if (closed and np.isfinite(oc) and oc > 0) else None
            rows.append({"id": f"{x['date']}-{x['jcd']}-{int(x['rno']):02d}-{'W' if kind == '単勝' else 'Q'}-{combo}", "date": f"{x['date'][:4]}-{x['date'][4:6]}-{x['date'][6:]}",
                         "venue": VENUE.get(x["jcd"], x["jcd"]), "race": f"{VENUE.get(x['jcd'], x['jcd'])} {int(x['rno'])}R", "deadline": x["deadline"], "kind": kind, "combo": combo,
                         "odds_t1": round(float(o1), 1) if np.isfinite(o1) and o1 > 0 else None, "fair_t1": round(float(1 / qq1), 1) if qq1 > 0 else None,
                         "ev_t1": round(float(ev1), 3) if ev1 is not None else None,
                         "odds_close": round(float(oc), 1) if (closed and np.isfinite(oc) and oc > 0) else None,
                         "ev_close": round(float(evc), 3) if evc is not None else None,
                         "hit": (None if top is None else int(hit(top))), "settled": bool(closed and top is not None)})
        for b in range(1, 7):
            add("単勝", str(b), T1[i, b - 1], w1[i, b - 1], TC[i, b - 1], wc[i, b - 1], lambda top, b=b: top[0] == b)
        for j, (p, q) in enumerate(PAIRS):
            add("2連複", f"{p}={q}", T2[i, j], q1[i, j], C2[i, j], qc[i, j], lambda top, p=p, q=q: set(top) == {p, q})
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    df = pd.DataFrame(rows)
    print(len(df), "rows", df["date"].nunique(), "days", df[df.settled].shape[0], "settled rows", os.path.getsize(a.out) // 1024, "KB")

if __name__ == "__main__":
    main()
