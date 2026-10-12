"""auto/fetch.py で保存した JSON を表（parquet）にする。**ローカル専用**（出力も auto_data/ の中、コミットしない）。
usage: python auto/build.py [--data auto_data]
出力: races.parquet（1行=1レース：着順上位3台・各券種の払戻・売上）、riders.parquet（1行=1台：ハンデ・試走タイムなど締切前の情報）、
      odds.parquet（1行=1買い目：券種・買い目・最終オッズ）
"""
import argparse, glob, gzip, json, os, re, sys
import numpy as np
import pandas as pd

def num(x):
    """数値（"12.3"、"1,234"、幅のあるオッズ "1.2-1.5" は下限）→ float。数字がなければ NaN"""
    m = re.search(r"\d+(?:\.\d+)?", str(x).replace(",", "")) if x is not None else None
    return float(m.group(0)) if m else np.nan

def flat(d, depth, prefix=()):
    """{a: {b: {c: "12.3"}}} → [((a,b,c), "12.3"), ...]"""
    if depth == 0 or not isinstance(d, dict):
        return [(prefix, d)]
    out = []
    for k, v in d.items():
        out += flat(v, depth - 1, prefix + (int(k),))
    return out

KINDS = {"rt3OddsList": ("3連単", 3), "rf3OddsList": ("3連複", 3), "rtwOddsList": ("2連単", 2), "rfwOddsList": ("2連複", 2),
         "widOddsList": ("ワイド", 2), "tnsOddsList": ("単勝", 1), "fnsOddsList": ("複勝", 1)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="auto_data")
    a = ap.parse_args()
    races, riders, odds, sales_rows = [], [], [], []
    for p in sorted(glob.glob(f"{a.data}/refund/*.json.gz")):
        day, pc = os.path.basename(p)[:-8].split("_")
        for race in json.load(gzip.open(p, "rt", encoding="utf-8")):
            rno = int(race.get("raceNo", 0)); ri = race.get("refundInfo") or {}
            row = {"date": day.replace("-", ""), "pc": int(pc), "rno": rno}
            t3 = ((ri.get("rt3") or {}).get("list") or [])
            if t3:
                row.update({"p1": t3[0].get("1thCarNo"), "p2": t3[0].get("2thCarNo"), "p3": t3[0].get("3thCarNo"), "tri_pay": num(t3[0].get("refund")),
                            "tri_votes": num(t3[0].get("refundVotes"))})
            tn = ((ri.get("tns") or {}).get("list") or [])
            if tn:
                row["win_pay"] = num(tn[0].get("refund"))
            row["fns_paid"] = json.dumps({str(x.get("carNo")): num(x.get("refund")) for x in ((ri.get("fns") or {}).get("list") or [])})
            row["absent"] = json.dumps([x if not isinstance(x, dict) else x.get("carNo") for x in (ri.get("absent") or [])])
            races.append(row)
    R = pd.DataFrame(races)
    for p in sorted(glob.glob(f"{a.data}/odds/*.json.gz")):
        day, pc, rno = os.path.basename(p)[:-8].split("_")
        key = {"date": day.replace("-", ""), "pc": int(pc), "rno": int(rno)}
        b = json.load(gzip.open(p, "rt", encoding="utf-8"))
        for x in b.get("playerList") or []:
            riders.append({**key, "car": x.get("carNo"), "player": x.get("playerCode"), "handicap": num(x.get("handicap")), "trial": num(x.get("trialTime")),
                           "trial_retry": x.get("trialRetryCode"), "absent": x.get("absent") is not None, "ai": x.get("aiExpectCode"),
                           "race_dev": num(x.get("raceDev")), "st_ave": num(x.get("stAve")), "gt_trial_ave": num(x.get("goodTrackTraialAve")),
                           "gt_race_ave": num(x.get("goodTrackRaceAve")), "gt_race_best": num(x.get("goodTrackRaceBest")), "graduation": x.get("graduationCode")})
        sales = b.get("salesInfo") or {}
        srow = dict(key)
        if isinstance(sales, dict):
            for k, v in sales.items():
                srow[f"sales_{k}"] = num(v) if not isinstance(v, dict) else num(next((v[x] for x in ("sales", "amount", "total") if x in v), None))
        sales_rows.append(srow)
        for lk, (kind, depth) in KINDS.items():
            for combo, v in flat(b.get(lk) or {}, depth):
                odds.append({**{k: key[k] for k in ("date", "pc", "rno")}, "kind": kind, "combo": "-".join(map(str, combo)), "odds": num(v)})
    Rd, Od = pd.DataFrame(riders), pd.DataFrame(odds)
    if sales_rows:
        R = R.merge(pd.DataFrame(sales_rows), on=["date", "pc", "rno"], how="left")
    R.to_parquet(f"{a.data}/races.parquet"); Rd.to_parquet(f"{a.data}/riders.parquet"); Od.to_parquet(f"{a.data}/odds.parquet")
    print(f"races {len(R)}（{R['date'].min()}〜{R['date'].max()}）, riders {len(Rd)}, odds {len(Od)}")
    # 確認：3連単の最終オッズ×100 と公式の払戻が一致するか（最終オッズかどうかの確認）
    t = Od[Od["kind"] == "3連単"].copy()
    m = R.dropna(subset=["p1"]).assign(combo=lambda x: x["p1"].astype(int).astype(str) + "-" + x["p2"].astype(int).astype(str) + "-" + x["p3"].astype(int).astype(str))
    m = m.merge(t, on=["date", "pc", "rno", "combo"], how="inner")
    if len(m):
        ok = (np.abs(m["odds"] * 100 - m["tri_pay"]) <= 10).mean()
        print(f"  3連単の最終オッズ×100 と払戻の一致率（±10円）: {ok:.3f}（{len(m)} レース）")

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main()
