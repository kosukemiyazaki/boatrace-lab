"""auto/build.py と auto/run.py の動作確認用に、公式と同じ形の「作り物の」JSON を作る（中身は乱数。本物のデータではない）。
usage: python auto/selftest.py OUTDIR && python auto/build.py --data OUTDIR && python auto/run.py --data OUTDIR --out OUTDIR/summary.md
"""
import gzip, itertools, json, os, sys
import numpy as np
import pandas as pd

def main(out):
    rng = np.random.default_rng(1)
    ab = {i: rng.normal(0, 1) for i in range(300)}
    for day in pd.date_range("2025-01-01", "2026-08-31", freq="5D").strftime("%Y-%m-%d"):
        for pc in (2, 5):
            races = []
            for rno in range(1, 13):
                ids = rng.choice(300, 8, replace=False)
                hc = rng.choice([0, 10, 20, 30], 8)
                trial = 3.35 - 0.03 * np.array([ab[i] for i in ids]) + rng.normal(0, 0.02, 8)
                perf = np.array([ab[i] for i in ids]) - hc / 30 + rng.normal(0, 1, 8)
                order = np.argsort(-perf)
                p = np.exp(np.array([ab[i] for i in ids]) - hc / 30); p /= p.sum()
                perms = list(itertools.permutations(range(8), 3))
                t = np.array([p[a] * p[b] / (1 - p[a]) * p[c] / (1 - p[a] - p[b]) for a, b, c in perms]); t /= t.sum()
                rt3 = {}
                for (a, b, c), v in zip(perms, t):
                    rt3.setdefault(str(a + 1), {}).setdefault(str(b + 1), {})[str(c + 1)] = f"{0.7 / v:.1f}"
                tns = {str(i + 1): f"{0.7 / p[i]:.1f}" for i in range(8)}
                fns = {str(i + 1): f"{max(1.0, 0.7 / min(0.99, 3 * p[i])):.1f}-{max(1.1, 0.9 / min(0.99, 3 * p[i])):.1f}" for i in range(8)}
                w = [int(x) + 1 for x in order[:3]]
                odds = {"playerList": [{"carNo": i + 1, "playerCode": str(ids[i]), "handicap": int(hc[i]), "trialTime": f"{trial[i]:.2f}", "trialRetryCode": None,
                                        "aiExpectCode": str(rng.integers(1, 4)), "absent": None, "raceDev": f"{3.4 - 0.03 * ab[ids[i]]:.3f}", "stAve": "0.15",
                                        "goodTrackTraialAve": f"{3.36 - 0.03 * ab[ids[i]]:.3f}", "goodTrackRaceAve": "3.40", "goodTrackRaceBest": "3.35", "graduationCode": "30"} for i in range(8)],
                        "rt3OddsList": rt3, "tnsOddsList": tns, "fnsOddsList": fns, "salesInfo": {"rt3": "1234567", "tns": "12345"}}
                p_ = f"{out}/odds/{day}_{pc}_{rno:02d}.json.gz"; os.makedirs(os.path.dirname(p_), exist_ok=True)
                json.dump(odds, gzip.open(p_, "wt", encoding="utf-8"))
                tri_odds = float(rt3[str(w[0])][str(w[1])][str(w[2])])
                races.append({"raceNo": rno, "refundInfo": {
                    "rt3": {"list": [{"1thCarNo": w[0], "2thCarNo": w[1], "3thCarNo": w[2], "refund": int(tri_odds * 100), "pop": 1, "refundVotes": "100"}]},
                    "tns": {"list": [{"carNo": w[0], "refund": int(float(tns[str(w[0])]) * 100), "pop": 1, "refundVotes": "10"}]},
                    "fns": {"list": [{"carNo": c, "order": j + 1, "refund": 150, "pop": 1, "refundVotes": "5"} for j, c in enumerate(w)]}, "absent": []}})
            p_ = f"{out}/refund/{day}_{pc}.json.gz"; os.makedirs(os.path.dirname(p_), exist_ok=True)
            json.dump(races, gzip.open(p_, "wt", encoding="utf-8"))
    print("作り物のデータを作成:", out)

if __name__ == "__main__":
    main(sys.argv[1])
