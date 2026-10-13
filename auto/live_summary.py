"""auto/live.py で取った締切前の時間帯別スナップショットを集計する。**ローカルで実行**、出力は集計値のみ（共有してよい）。
usage: python auto/live_summary.py [--data auto_data] [--out results/auto_live_summary.md]
前提: その日の最終オッズ・払戻を auto/fetch.py --from その日 --to その日 で取ってあること（なければ締切1分後 p01 を最終の代わりに使う）
集計（ボートの live_t1/t0/t00 と見比べるため、同じ考え方）:
1. 見えているお金の割合: 時間帯ごとの券種別売上 ÷ 最終の売上（中央値）
2. 3連単の市場から見た単勝・複勝のずれ: 時間帯ごとに「3連単の確率 × オッズ > 1.1」の買い目について、
   その時点の期待値・最終（3連単も最終）の期待値・オッズの変化（最終 ÷ その時点）・払戻が分かれば回収率（最終オッズで払戻、参考）
"""
import argparse, glob, gzip, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build import KINDS, flat, num

LEAD_ORDER = ["m20", "m15", "m10", "m06", "m04", "m02", "m01", "s30", "s10", "p01"]
SALES = {"rt3": "3連単", "rf3": "3連複", "rtw": "2連単", "rfw": "2連複", "wid": "ワイド", "tns": "単勝", "fns": "複勝"}

def load(p):
    return json.load(gzip.open(p, "rt", encoding="utf-8"))

def odds_of(b, lk, depth):
    return {"-".join(map(str, c)): num(v) for c, v in flat(b.get(lk) or {}, depth)}

def sales_of(b):
    s = b.get("salesInfo") or {}
    out = {k: num((s.get(k) or {}).get("salesCount")) for k in SALES}
    out["計"] = np.nansum(list(out.values()))
    return out

def tri_q(b):
    t = {k: v for k, v in odds_of(b, "rt3OddsList", 3).items() if v and v > 0}
    if len(t) < 30:
        return None
    s = sum(1 / v for v in t.values())
    return {k: (1 / v) / s for k, v in t.items()}

def evs(b, q, kplace):
    """単勝・複勝の (券種, 車番, 3連単から見た確率, オッズ)"""
    out = []
    win, plc = odds_of(b, "tnsOddsList", 1), odds_of(b, "fnsOddsList", 1)
    for c in win:
        qw = sum(p for k, p in q.items() if k.split("-")[0] == c)
        out.append(("単勝", c, qw, win[c]))
        if kplace and c in plc:
            qp = sum(p for k, p in q.items() if c in k.split("-")[:kplace])
            out.append(("複勝", c, qp, plc[c]))
    return out

def own_impact(paid_w, paid_p, pool_w, pool_p, rate):
    """自分が当たり車に100円（1票）を足していたら、払戻（100円あたり）はいくらになったか。
    単勝: 当たり車の票数 v = 払戻に回る額 N ÷ 払戻倍率 r → r' = rate×(P+1) ÷ (v+1)
    複勝: 当たり車それぞれ (r_j − 1)×v_j = (N − W) ÷ k が等しい（W = 当たり車の票数の合計、k = 当たり車の数）ことから v_j を出し、
          1票足したあとの r_j' = 1 + (rate×(P+1) − (W+1)) ÷ (k×(v_j+1))。払戻率 rate は最終の単勝オッズから推定（複勝も同じと仮定）
    """
    ow, op = {}, {}
    if pool_w and np.isfinite(rate):
        N = rate * pool_w
        for c, pay in paid_w.items():
            r = (pay or 0) / 100
            if r > 0:
                v = N / r; ow[c] = max(100.0, 100 * rate * (pool_w + 1) / (v + 1))
    if pool_p and np.isfinite(rate) and paid_p:
        rs = {c: (pay or 0) / 100 for c, pay in paid_p.items()}
        if all(r > 1 for r in rs.values()):
            k = len(rs); N = rate * pool_p
            cc = N / (k + sum(1 / (r - 1) for r in rs.values()))
            v = {c: cc / (r - 1) for c, r in rs.items()}; W = sum(v.values())
            for c in rs:
                op[c] = max(100.0, 100 * (1 + (rate * (pool_p + 1) - (W + 1)) / (k * (v[c] + 1))))
        else:
            op = {c: pay for c, pay in paid_p.items()}  # 元返しなど：調整しない
    return ow, op

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="auto_data"); ap.add_argument("--out", default="results/auto_live_summary.md")
    a = ap.parse_args()
    vis, cand = [], []
    races = sorted(glob.glob(f"{a.data}/live/*/*"))
    n_final = 0
    for rd in races:
        day = os.path.basename(os.path.dirname(rd)); pc, rr = os.path.basename(rd).split("_")
        snaps = {os.path.basename(p)[:-8]: load(p) for p in glob.glob(f"{rd}/*.json.gz")}
        fp = f"{a.data}/odds/{day}_{pc}_{rr}.json.gz"
        final = load(fp) if os.path.exists(fp) else (snaps["p01"]["odds"] if "p01" in snaps else None)
        if final is None:
            continue
        n_final += os.path.exists(fp)
        # 払戻（あれば）：複勝の対象着順の数と的中
        rp = f"{a.data}/refund/{day}_{pc}.json.gz"; paid_w, paid_p = {}, {}
        if os.path.exists(rp):
            for r in load(rp):
                if int(r.get("raceNo", 0)) == int(rr):
                    ri = r.get("refundInfo") or {}
                    paid_w = {str(x.get("carNo")): num(x.get("refund")) for x in ((ri.get("tns") or {}).get("list") or [])}
                    paid_p = {str(x.get("carNo")): num(x.get("refund")) for x in ((ri.get("fns") or {}).get("list") or [])}
        kplace = max(len(paid_p), 2) if paid_p else None
        fs, qf = sales_of(final), tri_q(final)
        rate = 1 / sum(1 / v for v in odds_of(final, "tnsOddsList", 1).values() if v and v > 0) if odds_of(final, "tnsOddsList", 1) else np.nan
        own_w, own_p = own_impact(paid_w, paid_p, fs.get("tns"), fs.get("fns"), rate)
        fo = {kind: odds_of(final, lk, 1) for lk, (kind, d) in KINDS.items() if d == 1}
        for lead, s in snaps.items():
            b = s["odds"]; ss = sales_of(b)
            vis.append({"lead": lead, "sec_left": s.get("sec_left"), **{k: (ss[k] / fs[k] if fs.get(k) else np.nan) for k in ss}})
            q = tri_q(b)
            if q is None or qf is None:
                continue
            for kind, c, qq, o in evs(b, q, kplace):
                if not (o and o > 0) or qq * o <= 1.1:
                    continue
                oc = fo[kind].get(c)
                qc = sum(p for k, p in qf.items() if (k.split("-")[0] == c if kind == "単勝" else c in k.split("-")[:kplace]))
                paid = (paid_w if kind == "単勝" else paid_p)
                cand.append({"lead": lead, "kind": kind, "day": day, "ev_lead": qq * o, "ev_close": qc * oc if oc else np.nan,
                             "drift": oc / o if oc else np.nan, "settled": bool(paid), "ret": (paid.get(c, 0.0) or 0.0) if paid else np.nan,
                             "ret_own": ((own_w if kind == "単勝" else own_p).get(c, 0.0) or 0.0) if paid else np.nan})
    V = pd.DataFrame(vis)
    md = [f"# オートレース：締切前の時間帯別（auto/live.py、{len(races)} レース、最終オッズあり {n_final}）", ""]
    if len(V):
        V["lead"] = pd.Categorical(V["lead"], LEAD_ORDER, ordered=True)
        t = V.groupby("lead", observed=True).agg(レース=("計", "size"), 残り秒中央値=("sec_left", "median"),
                                                  **{k: (k, "median") for k in ["計"] + list(SALES)})
        t = t.rename(columns=SALES)
        md += ["## 1. 見えているお金の割合（その時点の売上 ÷ 最終の売上、中央値）", "", t.to_markdown(floatfmt=".2f"), ""]
    C = pd.DataFrame(cand)
    if len(C):
        rows = []
        for (lead, kind), g in C.groupby(["lead", "kind"]):
            st = g[g["settled"]]
            rows.append({"時間帯": lead, "券種": kind, "点数": len(g), "その時点の期待値": g["ev_lead"].mean(), "最終の期待値": g["ev_close"].mean(),
                         "オッズ 最終÷その時点（中央値）": g["drift"].median(), "払戻あり": len(st),
                         "回収率（最終オッズで払戻・参考）": st["ret"].sum() / (100 * len(st)) if len(st) else np.nan,
                         "回収率（自分の100円込み）": st["ret_own"].sum() / (100 * len(st)) if len(st) else np.nan})
        R = pd.DataFrame(rows); R["o"] = R["時間帯"].map({k: i for i, k in enumerate(LEAD_ORDER)})
        md += ["## 2. 3連単の市場から見た単勝・複勝のずれ（その時点で 確率×オッズ > 1.1、参考）", "",
               R.sort_values(["券種", "o"]).drop(columns="o").to_markdown(index=False, floatfmt=".3f"), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main()
