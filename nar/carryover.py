"""トリプル馬単（SPAT4 LOTO、南関東4場・門別）のキャリーオーバーの履歴を作り、期待値がプラスの回の頻度と大きさを出す（IDEA-021）。**ローカル専用**。
usage:
  python nar/carryover.py fetch [--data nar_data] [--from 2018-01-01] [--to 2026-08-31] [--interval 5]
  python nar/carryover.py report [--data nar_data] [--out results/nar_carryover.md]
- fetch: 地方競馬情報サイトの当日払戻金ページ（/KeibaWeb/TodayRaceInfo/RefundMoneyList?k_raceDate=YYYY/MM/DD&k_babaCode=NN）を、
  nar_data/races.parquet にある「その場の開催日」だけ取得し、トリプル馬単の欄（前回からの繰越・発売金額・返還金額・次回への繰越・的中口数・払戻）を
  nar_data/carryover.csv に1行=1回で保存する。HTML は保存しない。既に取った日は取り直さない。
- report: 回ごとに 期待回収率 ≒ (0.70 × 売上 + 前回からの繰越) / 売上 を計算し、100% を超える回の頻度・大きさ、実際の払戻を集計する。
  出力は集計値のみ（results/ に置いて共有してよい）。
場コード: 浦和 18、船橋 19、大井 20、川崎 21、門別 36
"""
import argparse, csv, datetime as dt, os, re, sys, time, urllib.request

BABA = {"浦和": "18", "船橋": "19", "大井": "20", "川崎": "21", "門別": "36"}
URL = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RefundMoneyList?k_raceDate={d}&k_babaCode={c}"
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research; personal analysis)"}
COLS = ["date", "venue", "carry_in", "sales", "refund", "carry_out", "hits", "payout_per_unit", "raw_labels"]
P = 0.70  # トリプル馬単の払戻率（公式Q&A）

def yen(s):
    m = re.search(r"([\d,]+)\s*円", s)
    return int(m.group(1).replace(",", "")) if m else None

def parse(html):
    """トリプル馬単の欄 -> dict。欄がなければ None"""
    i = html.find("トリプル馬単")
    if i < 0:
        return None
    sec = html[i:]
    j = sec.find('class="attention"')
    sec = sec[:j] if j > 0 else sec
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", sec))
    out = {"carry_in": None, "sales": None, "refund": None, "carry_out": None, "hits": None, "payout_per_unit": None}
    pairs = re.findall(r'<td class="a">(.*?)</td>\s*<td class="b">(.*?)</td>', sec, re.S)
    labels = []
    for lab, val in pairs:
        lab = re.sub(r"<[^>]+>|\s+", "", lab); val = re.sub(r"<[^>]+>|\s+", "", val)
        labels.append(lab)
        if "前回" in lab and "キャリー" in lab:
            out["carry_in"] = yen(val)
        elif "次回" in lab and "キャリー" in lab:
            out["carry_out"] = yen(val)
        elif lab.startswith("発売"):
            out["sales"] = yen(val)
        elif lab.startswith("返還"):
            out["refund"] = yen(val)
    # 的中口数と1口あたりの払戻（表の形が分からない部分は文字列から拾う。拾えなければ空のまま）
    m = re.search(r"([\d,]+)\s*円\s*\|?[^|]*\|?\s*([\d,]+)\s*口", text)
    if m:
        out["payout_per_unit"] = int(m.group(1).replace(",", "")); out["hits"] = int(m.group(2).replace(",", ""))
    elif re.search(r"的中(?:者|票)?なし|該当なし", text):
        out["hits"] = 0
    out["raw_labels"] = "/".join(labels)[:200]
    return out

def fetch(a):
    import pandas as pd
    R = pd.read_parquet(f"{a.data}/races.parquet", columns=["venue", "date"])
    R = R[R["venue"].isin(BABA) & (R["date"] >= a.start.replace("-", "")) & (R["date"] <= a.to.replace("-", ""))].drop_duplicates()
    path = f"{a.data}/carryover.csv"
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["date"], r["venue"]) for r in csv.DictReader(f)}
    todo = [(r.date, r.venue) for r in R.sort_values(["date", "venue"]).itertuples() if (r.date, r.venue) not in done]
    print(f"対象 {len(R)} 開催日、未取得 {len(todo)}（間隔 {a.interval} 秒、目安 {len(todo) * (a.interval + 2) / 3600:.1f} 時間）", flush=True)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for n, (d, v) in enumerate(todo, 1):
            u = URL.format(d=f"{d[:4]}/{d[4:6]}/{d[6:]}", c=BABA[v])
            try:
                h = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read().decode("utf-8", "replace")
                o = parse(h)
            except Exception as e:
                print("NG", d, v, e, flush=True); time.sleep(30); continue
            row = {"date": d, "venue": v, **(o or {})}
            w.writerow({c: row.get(c, "") for c in COLS}); f.flush()
            if n == 1 or n % 50 == 0:
                print(n, d, v, {k: row.get(k) for k in ("carry_in", "sales", "carry_out", "hits", "raw_labels")}, flush=True)
            time.sleep(a.interval)

def report(a):
    import numpy as np
    import pandas as pd
    c = pd.read_csv(f"{a.data}/carryover.csv", dtype={"date": str})
    c = c.dropna(subset=["sales"]).query("sales > 0").copy()
    c["carry_in"] = c["carry_in"].fillna(0)
    c["cs"] = c["carry_in"] / c["sales"]
    c["ev"] = (P * c["sales"] + c["carry_in"]) / c["sales"]
    c["year"] = c["date"].str[:4]
    pos = c[c["ev"] > 1.0]
    lines = [f"# IDEA-021 トリプル馬単のキャリーオーバー（{c['date'].min()}〜{c['date'].max()}、{len(c)} 回、{c['venue'].nunique()} 場）", "",
             "期待回収率 ≒ (0.70 × 売上 + 前回からの繰越) / 売上。的中者が出ない回はまた繰り越されるため、実際の期待値はこれより低い。", "",
             f"**期待回収率が 100% を超えた回：{len(pos)} 回**（全体の {len(pos) / len(c):.1%}）", ""]
    by = c.groupby("year").agg(回数=("ev", "size"), 繰越あり=("carry_in", lambda s: int((s > 0).sum())),
                               期待値プラス=("ev", lambda s: int((s > 1).sum())), C_S最大=("cs", "max"), 期待回収率最大=("ev", "max"))
    lines += ["## 年ごと", "", by.to_markdown(floatfmt=".2f"), ""]
    bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0, 1e9]
    dist = pd.cut(c["cs"], bins, right=False).value_counts().sort_index()
    lines += ["## 繰越 ÷ 売上 の分布", "", dist.to_frame("回数").to_markdown(), ""]
    if len(pos):
        t = pos.assign(売上_億=pos["sales"] / 1e8, 繰越_億=pos["carry_in"] / 1e8, 次回へ_億=pos["carry_out"].fillna(0) / 1e8)
        t = t[["date", "venue", "売上_億", "繰越_億", "cs", "ev", "hits", "次回へ_億"]].sort_values("date")
        lines += ["## 期待値がプラスだった回（日付・場・売上・繰越・C/S・期待回収率・的中口数・次回への繰越）", "", t.to_markdown(index=False, floatfmt=".3f"), ""]
        hit = pos["hits"].fillna(-1)
        lines += [f"そのうち的中者が出た回：{int((hit > 0).sum())}、出なかった回：{int((hit == 0).sum())}、不明：{int((hit < 0).sum())}"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "report"]); ap.add_argument("--data", default="nar_data")
    ap.add_argument("--from", dest="start", default="2018-01-01"); ap.add_argument("--to", default="2026-08-31")
    ap.add_argument("--interval", type=float, default=5.0); ap.add_argument("--out", default="results/nar_carryover.md")
    a = ap.parse_args()
    fetch(a) if a.cmd == "fetch" else report(a)

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main()
