"""ライブの参考予想（scratchpad の pred_ls.py と同じ計算）＋番組表から F 持ち・級別・年齢を自動で入れる＋ IDEA-075 の条件
usage: python scripts/pred_live.py JCD RNO [--date YYYYMMDD] [--f 1,3,5] [--src DIR] [--store /tmp/claude-0/store] [--datadir /tmp/claude-0/ds] [--no-refresh]
- オッズ：data ブランチの live_t15・live_t10・live（6分前）・live_t1 の3連単のうち一番新しいもの。既定では origin/data から当日分を SRC に取り直す
- 計算（変えていない）：市場の確率 → 本命・穴の補正（β=1.055）→ F 持ちの補正（IDEA-036 の係数）→ 展開の補正（IDEA-035 の c2・c3、models/idea035_c35.npz）
- F 持ち：番組表の当日の出走選手について、審査期間の開始から前日までに F があった艇（IDEA-036 の定義）。--f を渡すとそちらを使う（画面の表示と違うとき）
- IDEA-075：(a)(c) は番組表、(b) は判定に使ったオッズの時点の市場の1号艇の確率、(d) は直前情報の風向き（場ごとの対応表による推定）
- 買い目は補正後の確率の高い順に選ぶ（IDEA-074：締切前の期待値で組を外さない）。自動で投票はしない
"""
import argparse, glob, os, subprocess, sys, tempfile, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from idea075_lib import ROOT, load_fixed, load_program, hist_entries, race_table, ratio, wind_from_code, fav_dirs, today_jst
from upset_days import NAMES

LEADS = ["live_t15", "live_t10", "live", "live_t1"]
warnings.filterwarnings("ignore")

def refresh(src, date):
    subprocess.run(["git", "-C", ROOT, "fetch", "-q", "origin", "data"], check=True)
    files = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "--name-only", "origin/data"], capture_output=True, text=True, check=True).stdout.split()
    want = [f for f in files if f"/{date}_" in f and (f.split("/")[0] in LEADS) and f.split("/")[1] in ("odds3t", "oddstf", "beforeinfo")]
    if not want:
        return
    os.makedirs(src, exist_ok=True)
    tar = subprocess.run(["git", "-C", ROOT, "archive", "origin/data"] + want, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", src], input=tar, check=True)

def latest(src, page, date, jcd, rno, leads=LEADS):
    rows = []
    for lead in leads:
        for f in glob.glob(f"{src}/{lead}/{page}/{date}_*.csv.gz"):
            d = pd.read_csv(f, dtype={"jcd": str, "date": str}); d = d[(d.jcd.str.zfill(2) == jcd) & (d.rno == rno)]
            if len(d):
                rows.append(d.assign(lead=lead))
    return pd.concat(rows).sort_values("fetched_at").iloc[-1] if rows else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("jcd"); ap.add_argument("rno", type=int)
    ap.add_argument("--date", default=None); ap.add_argument("--f", default=None, help="F 持ちの艇（例 1,3,5）。省くと番組表から自動")
    ap.add_argument("--src", default=os.path.join(tempfile.gettempdir(), "pred_live_src")); ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--store", default="/tmp/claude-0/store"); ap.add_argument("--datadir", default="/tmp/claude-0/ds")
    ap.add_argument("--c35", default=os.path.join(ROOT, "models", "idea035_c35.npz"))
    a = ap.parse_args()
    date = a.date or today_jst(); jcd = a.jcd.zfill(2); rno = a.rno
    if not a.no_refresh:
        refresh(a.src, date)
    # 番組表：級別・年齢・F・IDEA-075 の条件
    W, Z = load_fixed(); info = None
    prog = load_program(a.store, date)
    if prog is not None and ((prog["jcd"] == jcd) & (prog["rno"] == rno)).any():
        m = race_table(prog[(prog["jcd"] == jcd)], hist_entries(a.datadir, date), a.store, W, Z)
        info = m[m["rno"] == rno].iloc[0]
    if a.f is not None:
        FH = [int(x) for x in a.f.split(",") if x]; fsrc = "手入力"
    elif info is not None:
        FH = [b for b in range(1, 7) if info.get(f"fn_{b}", 0) and info[f"fn_{b}"] > 0]; fsrc = "番組表から自動"
    else:
        FH = []; fsrc = "番組表なし（--f で入れる）"
    print(f"{NAMES.get(jcd, jcd)} {rno}R（{date}）")
    if info is not None:
        print("出走：" + "　".join(f"{b}号艇 {int(info[f'toban_{b}'])}・{info[f'cls_{b}']}・{int(info[f'age_{b}'])}歳・F{int(info[f'fn_{b}'])}" for b in range(1, 7) if pd.notna(info.get(f"toban_{b}"))))
    row = latest(a.src, "odds3t", date, jcd, rno)
    if row is None:
        print("まだオッズなし"); return
    print("使うオッズ:", row.lead, "取得", row.fetched_at, "締切", row.deadline)
    cz = np.load(a.c35)
    # ---- ここから pred_ls.py と同じ計算 ----
    O = pd.to_numeric(row[COMBOS], errors="coerce").to_numpy(float); v = np.isfinite(O) & (O > 0)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(); Pb = P ** 1.055; Pb /= Pb.sum()
    I = np.array([int(c[0]) for c in COMBOS]) - 1; J = np.array([int(c[2]) for c in COMBOS]) - 1
    adj = np.zeros(6)
    for b in FH:
        adj[b - 1] += 0.035 if b == 1 else (-0.022 if b <= 3 else -0.099)
        if b < 6: adj[b] += -0.056
    print(f"F持ち（{fsrc}）:", FH if FH else "なし", ("1着の補正倍率: " + " ".join(f"{i+1}:{np.exp(adj[i]):.3f}" for i in range(6))) if FH else "")
    Pb = Pb * np.exp(adj[I]); Pb /= Pb.sum()
    P1 = np.zeros(6); P12 = np.zeros((6, 6))
    for c in range(120): P1[I[c]] += Pb[c]; P12[I[c], J[c]] += Pb[c]
    q2 = P12 / np.where(P1[:, None] > 0, P1[:, None], 1); band = int(np.digitize(P1[0], [0.4, 0.6]))
    q2a = q2 * cz["c2"][:, :, band]; np.fill_diagonal(q2a, 0); q2a /= np.where(q2a.sum(1, keepdims=True) > 0, q2a.sum(1, keepdims=True), 1)
    q3 = Pb / np.where(P12[I, J] > 0, P12[I, J], 1) * cz["c3"] * (Pb > 0); pair = I * 6 + J
    den = np.bincount(pair, weights=q3, minlength=36); q3 = q3 / np.where(den[pair] > 0, den[pair], 1)
    PI = P1[I] * q2a[I, J] * q3; PI /= PI.sum()
    t = pd.DataFrame({"組番": COMBOS, "オッズ": O, "市場の確率": P, "補正後の確率": PI}); t["期待値"] = t["補正後の確率"] * t["オッズ"]
    print("1号艇の1着確率（市場）: %.1f%%" % (100 * P1[0]))
    print("1着確率（補正後）:", " ".join(f"{i+1}号艇 {100*PI[I==i].sum():.1f}%" for i in range(6)))
    print(t.sort_values("補正後の確率", ascending=False).head(15).to_string(index=False, float_format="%.3f"))
    print("--- 穴（100倍以上）の確率の高い順")
    print(t[t["オッズ"] >= 100].sort_values("補正後の確率", ascending=False).head(10).to_string(index=False, float_format="%.3f"))
    # ---- IDEA-075 ----
    pm1 = float(P[np.array([int(c[0]) for c in COMBOS]) == 1].sum())
    tf = latest(a.src, "oddstf", date, jcd, rno)
    tan1 = pd.to_numeric(tf["tan1"], errors="coerce") if tf is not None else np.nan
    if info is not None:
        r, padj = ratio(pm1, info["s"]); r, padj = float(r), float(padj)
        bi = latest(a.src, "beforeinfo", date, jcd, rno, leads=["live"])
        wd = wind_from_code(Z, jcd, bi["wind_dir"]) if bi is not None else None
        fav = fav_dirs(Z, jcd)
        cond = {"(a) 22歳以下": bool(info["a"]), "(b) 補正後÷市場≥1.15": r >= 1.15, "(c) 逃げ上手": bool(info["c"]),
                "(d) 有利な風（推定）": (wd in fav) if wd else False}
        n = sum(cond.values())
        print(f"--- IDEA-075（1号艇の単勝・複勝の条件）：{n}つ成立 → {'対象' if n >= 2 else '対象外'}")
        print("   " + "　".join(f"{k}:{'○' if x else '×'}" for k, x in cond.items()))
        print(f"   市場の1号艇 {pm1:.1%}（{row.lead}）、補正後 {padj:.1%}（÷市場 {r:.3f}）、直前情報の風 {wd or '不明'}（有利な向き：{'・'.join(fav) or 'なし'}）")
        print(f"   単勝の損益分岐 {1/padj:.2f}倍、取れている単勝 {tan1 if pd.notna(tan1) else '—'}倍（{tf.lead if tf is not None else '—'}、締切前の単勝は売上が少なく当てにならない）")
    else:
        print(f"--- 番組表がないので IDEA-075 の条件は出せない（市場の1号艇 {pm1:.1%}）")

if __name__ == "__main__":
    main()
