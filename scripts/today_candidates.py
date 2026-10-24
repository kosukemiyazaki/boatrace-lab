"""IDEA-075 当日の候補一覧：番組表から、1号艇の単勝・複勝の条件（IDEA-069/070 の (a)〜(d)、2つ以上で対象）を出す
usage: python scripts/today_candidates.py STOREDIR DATADIR [--date YYYYMMDD] [--after HH:MM] [--out results/today_candidates.md]
- 番組表は STORE/program（.github/workflows/program.yml）か STORE/bk。F持ち・節の日目は DATADIR の entries（前日まで）から
- (a)(c) は朝の時点で確定。(b) は「締切前の市場の1号艇の確率が何%以下なら成り立つか」、(d) は「その場で1コースに有利な風向き」を載せ、締切前に確かめる
- 載せるレース：確定2つ以上／確定1つ（(b) か (d) があと1つ成り立てば対象）／確定0（(b) と (d) の両方が要る。(b) の上限が 45% 以上のものだけ）
- 単勝の損益分岐 = 1／補正後の1号艇の確率（市場の1号艇の確率が 40%・55%・70% のときの目安）
- 選手は登録番号だけを出す（名前・性別は出さない）。買うかどうかは利用者が決める（自動の投票はしない）
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from idea075_lib import load_fixed, load_program, hist_entries, race_table, ratio, p1_threshold, fav_dirs, today_jst
from upset_days import NAMES

def build(store, datadir, date):
    W, Z = load_fixed()
    prog = load_program(store, date)
    if prog is None:
        return None, W, Z
    m = race_table(prog, hist_entries(datadir, date), store, W, Z)
    m["b_thr115"] = [p1_threshold(s, 1.15) for s in m["s"]]
    m["b_thr125"] = [p1_threshold(s, 1.25) for s in m["s"]]
    m["fav"] = [fav_dirs(Z, j) for j in m["jcd"]]
    m["known"] = m["a"].astype(int) + m["c"].astype(int)
    for p in (0.40, 0.55, 0.70):
        m[f"be{int(p*100)}"] = 1 / ratio(p, m["s"].values)[1]
    return m, W, Z

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("store"); ap.add_argument("datadir")
    ap.add_argument("--date", default=None); ap.add_argument("--after", default=None)
    ap.add_argument("--out", default="results/today_candidates.md"); ap.add_argument("--json", default="results/today_candidates.json")
    a = ap.parse_args()
    date = a.date or today_jst()
    m, W, Z = build(a.store, a.datadir, date)
    if m is None:
        txt = f"# 当日の候補一覧（{date}）\n\n番組表がまだない（program.yml の取得待ち）。\n"
        open(a.out, "w", encoding="utf-8").write(txt); print(txt); return
    m = m.sort_values(["deadline1", "jcd", "rno"]).reset_index(drop=True)
    if a.after:
        m = m[m["deadline1"].astype(str) >= a.after].reset_index(drop=True)
    def tier(x):
        if x["known"] >= 2:
            return "確定（2つ以上）"
        if x["known"] == 1:
            return "あと1つ（(b)か(d)）"
        if x["b_thr115"] >= 0.45 and len(x["fav"]):
            return "(b)と(d)の両方が要る"
        return None
    m["tier"] = [tier(x) for _, x in m.iterrows()]
    L = m[m["tier"].notna()].copy()
    def row(x):
        known = "・".join([t for t, f in (("(a)22歳以下", x["a"]), ("(c)逃げ上手", x["c"])) if f]) or "なし"
        b = f"市場の1号艇 ≤ {x['b_thr115']:.0%}" if x["b_thr115"] >= 0.10 else ("ほぼ成り立たない" if x["b_thr115"] > 0 else "成り立たない")
        d = "・".join(x["fav"]) if len(x["fav"]) else "（有利な風の群なし）"
        return {"区分": x["tier"], "場": NAMES.get(x["jcd"], x["jcd"]), "R": int(x["rno"]), "締切": x["deadline1"],
                "1号艇（登番・級別・年齢・F）": f"{int(x['toban_1'])}・{x['cls_1']}・{int(x['age_1'])}歳・F{int(x['fn_1'])}",
                "確定の条件": known, "(b) 成り立つ目安": b, "(d) 有利な風向き": d,
                "重みの合計 s": round(float(x["s"]), 3), "(b)≥1.25 の目安": f"≤ {x['b_thr125']:.0%}" if x["b_thr125"] > 0 else "—",
                "単勝の損益分岐（市場40/55/70%）": f"{x['be40']:.2f}／{x['be55']:.2f}／{x['be70']:.2f}倍"}
    T = pd.DataFrame([row(x) for _, x in L.iterrows()])
    order = {"確定（2つ以上）": 0, "あと1つ（(b)か(d)）": 1, "(b)と(d)の両方が要る": 2}
    md = [f"# 当日の候補一覧（{date}{'、' + a.after + ' 以降' if a.after else ''}）", "",
          "IDEA-075（IDEA-069/070 の条件）。(a) 1号艇が22歳以下、(b) 補正後÷市場 ≥ 1.15、(c) 逃げ上手の選手（IDEA-060 上位10%）、(d) 1コースに有利な風。**2つ以上**で対象。",
          "- (a)(c) は番組表で確定。(b) は締切前の3連単から出す市場の1号艇の確率が「目安」以下なら成立。(d) はレース時の風向きが表の向きなら成立（締切前は直前情報で推定）。",
          "- 単勝の損益分岐：締切直前の1号艇の単勝がこの倍率以上なら、補正後の確率で見て損益分岐を超える（市場の1号艇の確率が40/55/70%のときの目安）。",
          "- 過去の成績（参考、これまでの検証）：条件2つ以上の1号艇 単勝 97.5%・複勝 97.6%（評価 2026-01〜08）。100% を超える根拠はまだない。前向きの記録は 2026-10-25 から。", "",
          f"対象レース数：全 {len(m)} レース中、確定2つ以上 {int((m['tier'] == '確定（2つ以上）').sum())}、あと1つ {int((m['tier'] == 'あと1つ（(b)か(d)）').sum())}、(b)(d)両方が要る {int((m['tier'] == '(b)と(d)の両方が要る').sum())}", ""]
    if len(T):
        T = T.sort_values(["区分", "締切"], key=lambda c: c.map(order) if c.name == "区分" else c)
        md += [T.to_markdown(index=False), ""]
    else:
        md += ["（対象なし）", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    keep = ["date", "jcd", "rno", "deadline1", "toban_1", "cls_1", "age_1", "fn_1", "a", "c", "s", "b_thr115", "b_thr125", "fav", "known", "tier", "be40", "be55", "be70"]
    recs = m[keep].rename(columns={"deadline1": "deadline"}).to_dict("records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, (np.integer,)): r[k] = int(v)
            elif isinstance(v, (np.floating,)): r[k] = float(v)
            elif isinstance(v, (np.bool_,)): r[k] = bool(v)
    json.dump({"date": date, "races": recs}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("\n".join(md))

if __name__ == "__main__":
    main()
