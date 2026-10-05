"""出走表データ -> 予測確率 -> 買い目（1レース上限1000円）
モデル: コース別基礎勝率 × 能力補正(勝率・当地・モーター・ST・F持ち) を正規化し、
Plackett-Luce で3連単確率を出す。上位10点×100円。カバー率が低い荒れ予想は見送り。
"""
import json, glob, itertools, math, datetime, sys

BASE = [0.55, 0.14, 0.12, 0.11, 0.06, 0.02]  # 全国平均のコース別1着率（概算）
LOCK = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))

def strength(r):
    s = 0.0
    s += 0.55 * (r.get("nat_win") or 0)
    lw = r.get("loc_win") or 0
    s += 0.15 * (lw if lw > 0 else (r.get("nat_win") or 0))
    s += 0.02 * (r.get("motor_2r") or 30)
    st = r.get("st") or 0.18
    s += -8.0 * (st - 0.17)
    if (r.get("f") or 0) > 0:
        s -= 0.4  # F持ちはST慎重
    return s

def probs(racers):
    act = [r for r in racers if not r.get("absent")]
    ss = [strength(r) for r in act]
    m = sum(ss) / len(ss)
    w = {}
    for r, s in zip(act, ss):
        w[r["boat"]] = BASE[r["boat"] - 1] * math.exp(0.9 * (s - m))
    t = sum(w.values())
    return {b: v / t for b, v in w.items()}

def trifecta(p):
    out = {}
    for a, b, c in itertools.permutations(p.keys(), 3):
        pa = p[a]; pb = p[b] / (1 - p[a]); pc = p[c] / (1 - p[a] - p[b])
        out[f"{a}-{b}-{c}"] = pa * pb * pc
    return sorted(out.items(), key=lambda x: -x[1])

def bets_for(racers):
    p = probs(racers)
    tri = trifecta(p)
    top = tri[:10]
    cover = sum(x[1] for x in top)
    if cover < 0.25:
        return p, [], cover
    return p, [{"type": "3連単", "combo": k, "yen": 100, "prob": round(v, 4)} for k, v in top], cover

def main():
    out = []
    lock_hm = LOCK.strftime("%H:%M")
    for f in sorted(glob.glob("/home/claude/boat/data/*.json")):
        d = json.load(open(f))
        for race in d.get("races", []):
            if not race.get("racers") or race.get("deadline", "00:00") <= lock_hm:
                continue  # 予測確定時刻より前に締切のレースは対象外
            p, bets, cover = bets_for(race["racers"])
            out.append({
                "jcd": d["jcd"], "venue": d["venue"], "rno": race["rno"],
                "deadline": race["deadline"], "win_prob": {str(k): round(v, 3) for k, v in p.items()},
                "cover": round(cover, 3), "bets": bets,
                "stake": sum(b["yen"] for b in bets),
            })
    rec = {"locked_at": LOCK.isoformat(), "races": out}
    json.dump(rec, open("/home/claude/boat/predictions.json", "w"), ensure_ascii=False, indent=1)
    print(f"locked {LOCK.isoformat()} races={len(out)} bet_races={sum(1 for r in out if r['bets'])} "
          f"stake={sum(r['stake'] for r in out)}")

if __name__ == "__main__":
    main()
