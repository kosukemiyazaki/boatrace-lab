"""nar/build.py と nar/run.py の動作確認用に、公式と同じ列の形の「作り物の」ZIP を作る（中身は乱数。本物のデータではない）。
usage: python nar/selftest.py OUTDIR && python nar/build.py --data OUTDIR && python nar/run.py --data OUTDIR --out OUTDIR/out
"""
import io, itertools, os, sys, zipfile
import numpy as np
import pandas as pd

RACE_COLS = ['競馬場', '競走年月日', 'レース番号', '発走時刻', '競走種類名称', 'レース名'] + [f'副賞名{i}' for i in range(1, 16)] + \
    ['芝ダート区分', '回り', '距離', '天候', '馬場', '頭数', '条件', '1着賞金(円)', '2着賞金(円)', '3着賞金(円)', '4着賞金(円)', '5着賞金(円)', '上がり4F', '上がり3F'] + \
    [f'ハロンタイム{i}' for i in range(1, 16)] + [f'コーナー名称{i}' for i in range(1, 9)] + [f'コーナー通過順{i}' for i in range(1, 9)]
HORSE_COLS = ['競馬場', '競走年月日', 'レース番号', '枠番', '帽色', '馬番', '馬名', '性', '齢', '毛色', '生年月日', '父馬名', '母馬名', '母父馬名', '騎手名', '騎手所属',
              '負担重量', '騎手成績', '調教師', '調教師所属', '馬主氏名', '生産牧場名', '馬体重', '馬体重増減', '全成績', 'ダート左成績', 'ダート右成績', '当競馬場成績',
              'うち当距離成績', '最高タイム', '最高タイム良馬場', '着順', 'タイム', '着差', '上がり3F', '人気']
PAY_COLS = ['競馬場', '競走年月日', 'レース番号', 'レース名'] + [f'x{i}' for i in range(44)] + ['３連単組番馬番1', '３連単組番馬番2', '３連単組番馬番3', '３連単払戻金（円）', '３連単人気']
ODDS_COLS = ['競馬場', '競走年月日', 'レース番号', '賭式', '番号1', '番号2', '番号3', 'オッズ', 'オッズ（最大）', '人気']

def main(out):
    rng = np.random.default_rng(1)
    horses = [(f"ウマ{i}", f"2020/0{1 + i % 9}/1{i % 9}", rng.normal(0, 1)) for i in range(400)]
    jockeys = [f"騎手{i}" for i in range(40)]
    months = pd.period_range("2025-01", "2026-08", freq="M")
    for pm in months:
        R, Hh, Pb, Od = [], [], [], []
        for day in (3, 10, 17, 24):
            date = f"{pm.year}/{pm.month:02d}/{day:02d}"
            for venue in ("大井", "船橋"):
                for rno in range(1, 7):
                    n = int(rng.integers(8, 13))
                    idx = rng.choice(len(horses), n, replace=False)
                    abil = np.array([horses[i][2] for i in idx]) + rng.normal(0, 1, n)
                    order = np.argsort(-abil)
                    pos = np.empty(n, int); pos[order] = np.arange(1, n + 1)
                    R.append([venue, date, str(rno), f"{12 + rno}:00", "一般", "テスト"] + [""] * 15 + ["ダ", "右", "1200", "晴", "良", str(n), "C1", "500000"] + [""] * 4 + ["", ""] + [""] * 31)
                    p = np.exp(np.array([horses[i][2] for i in idx]) * 0.9); p /= p.sum()
                    pop = (-p).argsort().argsort() + 1
                    for b in range(n):
                        h = horses[idx[b]]
                        pr = "取消" if rng.random() < 0.01 else str(pos[b])
                        Hh.append([venue, date, str(rno), str(b // 2 + 1), "", str(b + 1), h[0], "牡", "5", "鹿毛", h[1], "", "", "", jockeys[(idx[b] + day) % 40], "大井",
                                   "54.0", "10-8-7-50", "調教師", "大井", "", "", "460", "+2", "3-2-1-10", "", "", "1-0-0-2", "0-0-0-1", "1:12.3", "1:12.5",
                                   pr, f"1:{12 + pos[b] * 0.1:.1f}", "", f"{37 + pos[b] * 0.1:.1f}", str(pop[b])])
                    w = [int(x) + 1 for x in order[:3]]
                    perms = list(itertools.permutations(range(n), 3))
                    t = np.array([p[a] * p[b_] / (1 - p[a]) * p[c] / (1 - p[a] - p[b_]) for a, b_, c in perms]); t /= t.sum()
                    odds = np.round(0.75 / t, 1)
                    pay = int(odds[perms.index(tuple(order[:3]))] * 100)
                    Pb.append([venue, date, str(rno), "テスト"] + [""] * 44 + [str(w[0]), str(w[1]), str(w[2]), str(pay), "1"])
                    if pm >= pd.Period("2026-03", "M"):
                        for (a, b_, c), o in zip(perms, odds):
                            Od.append([venue, date, str(rno), "3連単", str(a + 1), str(b_ + 1), str(c + 1), f"{o:.1f}", "", "1"])
                        # 他の券種（3連単の確率に雑音を掛けた作り物のオッズ。IDEA-024 の動作確認用）
                        P3 = dict(zip(perms, t))
                        def agg(f):
                            d = {}
                            for k3, v in P3.items():
                                for kk in f(k3):
                                    d[kk] = d.get(kk, 0) + v
                            return d
                        noisy = lambda q: max(1.0, round(0.8 / q * float(np.exp(rng.normal(0, 0.3))), 1))
                        for (x,), q in agg(lambda k3: [(k3[0],)]).items():
                            Od.append([venue, date, str(rno), "単勝", str(x + 1), "", "", f"{noisy(q):.1f}", "", "1"])
                        for (x,), q in agg(lambda k3: [(k3[0],), (k3[1],), (k3[2],)]).items():
                            o = noisy(q); Od.append([venue, date, str(rno), "複勝", str(x + 1), "", "", f"{o:.1f}", f"{o * 1.5:.1f}", "1"])
                        for (x, y), q in agg(lambda k3: [(k3[0], k3[1])]).items():
                            Od.append([venue, date, str(rno), "馬単", str(x + 1), str(y + 1), "", f"{noisy(q):.1f}", "", "1"])
                        for (x, y), q in agg(lambda k3: [tuple(sorted(k3[:2]))]).items():
                            Od.append([venue, date, str(rno), "馬連", str(x + 1), str(y + 1), "", f"{noisy(q):.1f}", "", "1"])
                        for (x, y), q in agg(lambda k3: [tuple(sorted(c)) for c in itertools.combinations(k3, 2)]).items():
                            o = noisy(q); Od.append([venue, date, str(rno), "ワイド", str(x + 1), str(y + 1), "", f"{o:.1f}", f"{o * 1.5:.1f}", "1"])
        ym = f"{pm.year}{pm.month:02d}"
        def zipit(path, files):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with zipfile.ZipFile(path, "w") as z:
                for name, cols, rows in files:
                    b = io.StringIO(); pd.DataFrame(rows, columns=cols).to_csv(b, index=False); z.writestr(name, b.getvalue().encode("utf-8-sig"))
        zipit(f"{out}/race/{ym}.zip", [(f"{ym}_racelist.csv", RACE_COLS, R), (f"{ym}_horselist.csv", HORSE_COLS, Hh), (f"{ym}_payback.csv", PAY_COLS, Pb)])
        if Od:
            zipit(f"{out}/odds/{ym}.zip", [(f"{ym}_01_odds.csv", ODDS_COLS, Od)])
    print("作り物のデータを作成:", out)

if __name__ == "__main__":
    # Windows のコンソール（cp932）で表示できない文字があっても止まらないようにする
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main(sys.argv[1])
