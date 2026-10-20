"""IDEA-056 段階2 (B) 前向きの検証：偏りを重ねて1号艇が強いレースで 1号艇の単勝・複勝（docs/ideas.md の登録どおり。判定は 2026-10-21 以降）
usage: python scripts/idea056_fwd.py DATADIR STOREDIR [--start 20261021] [--weights models/idea056_weights.json] [--out results/idea056_stage2.md]
- 締切2分前（live_t1）の3連単から市場の1号艇の確率 p1（逆数の正規化。重みの学習と同じ定義）、IDEA-053 の21区分（出走表は DATADIR の entries）と
  固定した重み（models/idea056_weights.json）で 補正後÷市場 を出し、1.25 以上のレースで 1号艇の単勝・複勝を各100円
- 払戻は K ファイルの実額。1号艇が欠場・フライング・出遅れなら返還（100円戻る）。K がまだ無いレースは数えない
- 判定：300レース以上の時点で、単勝か複勝が 回収率110%以上・日単位95%CI下限 > 100% → 採用候補（最長 2027-03-31）
- 出力は OUT の「## (B) 前向きの検証」以降を書き換える
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import KEY, FIRST, boat_feats, segments, meet_day
from exp_idea053 import SEL, boot
from idea039_stage2 import load
from idea052 import k_pay

HEAD = "## (B) 前向きの検証"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--start", default="20261021")
    ap.add_argument("--weights", default="models/idea056_weights.json"); ap.add_argument("--out", default="results/idea056_stage2.md")
    a = ap.parse_args()
    W = json.load(open(a.weights, encoding="utf-8")); thr = W["ratio_threshold"]
    sel = [(s, v) for s, v, _ in W["segments"]]; w = np.array([x for _, _, x in W["segments"]])
    assert sel == list(SEL), "重みの区分が IDEA-053 と一致しない"
    o3 = load(a.store, "live_t1", "odds3t", a.start)
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]))
    e = e[e["date"] >= "20260101"]
    if not len(o3):
        print("no live_t1 odds"); return
    m = norm(o3)[KEY + COMBOS].merge(boat_feats(e), on=KEY, how="inner").reset_index(drop=True)
    O = m[COMBOS].apply(pd.to_numeric, errors="coerce").to_numpy(float); v = np.isfinite(O) & (O > 0)
    m = m[v.sum(1) >= 40].reset_index(drop=True); O = O[v.sum(1) >= 40]; v = v[v.sum(1) >= 40]
    Pm = np.where(v, 1 / np.where(v, O, 1), 0); Pm /= Pm.sum(1, keepdims=True); p1 = Pm[:, FIRST == 1].sum(1); m["p1"] = p1
    S = segments(m, None, True, a.store)
    # 節の日目は、開始日より前の開催日も数えるため出走表の全日付から出し直す
    dj = e[["date", "jcd"]].drop_duplicates().reset_index(drop=True); dj["day"] = meet_day(dj)
    S["節の日目"] = m[["date", "jcd"]].merge(dj, on=["date", "jcd"], how="left")["day"].astype(str).values
    X = np.zeros((len(m), len(sel)))
    for k, (s, val) in enumerate(sel):
        if s in S:
            X[:, k] = (np.asarray(S[s]).astype(str) == val)
    good = (p1 > 0) & (p1 < 1)
    off = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1)); padj = 1 / (1 + np.exp(-(off + X @ w)))
    ratio = np.where(good, padj / np.where(good, p1, 1), np.nan)
    pick = good & (ratio >= thr)
    # 払戻（K の実額）
    kp = k_pay(a.store, a.start); kp = kp[kp["kind"].isin(["tan", "fuku"])]
    haveK = set(map(tuple, kp[kp["kind"] == "tan"][KEY].values))
    tan = kp[(kp["kind"] == "tan") & (kp["combo"] == "1")].set_index(KEY)["pay"]
    fuku = kp[(kp["kind"] == "fuku") & (kp["combo"] == "1")].set_index(KEY)["pay"]
    b1 = e[e["boat"] == 1][KEY + ["pos_raw"]].drop_duplicates(KEY).set_index(KEY)["pos_raw"].astype(str)
    rows = []
    for i in np.where(pick)[0]:
        k = (m.at[i, "date"], m.at[i, "jcd"], int(m.at[i, "rno"]))
        if k not in haveK:
            continue
        refund = b1.get(k, "")[:1] in ("K", "F", "L")
        rows.append({"date": k[0], "jcd": k[1], "rno": k[2], "市場の1号艇": p1[i], "補正後÷市場": ratio[i],
                     "単勝1": 100 if refund else tan.get(k, 0), "複勝1": 100 if refund else fuku.get(k, 0), "返還": refund})
    R = pd.DataFrame(rows, columns=["date", "jcd", "rno", "市場の1号艇", "補正後÷市場", "単勝1", "複勝1", "返還"])
    n = len(R); ndays = R["date"].nunique() if n else 0
    lines = [HEAD, "", f"対象：{a.start} 以降、締切2分前の3連単と K の払戻がそろうレース。補正後÷市場 ≥ {thr} のレースで 1号艇の単勝・複勝を各100円（重みは `{a.weights}` で固定）。", ""]
    summ = []
    for col in ("単勝1", "複勝1"):
        if n:
            roi, lo, hi = boot(R["date"].values, np.full(n, 100.0), R[col].values.astype(float))
        else:
            roi = lo = hi = np.nan
        summ.append({"券種": col, "レース": n, "日数": ndays, "的中": int((R[col] > 0).sum() - R["返還"].sum()) if n else 0,
                     "回収率": roi, "95%CI下": lo, "95%CI上": hi})
    Sm = pd.DataFrame(summ)
    ok = any((x["レース"] >= 300) and (x["回収率"] >= 1.1) and (x["95%CI下"] > 1.0) for x in summ)
    status = ("採用候補" if ok else "不採用（300レース時点で基準に届かない）") if n >= 300 else f"集計中（{n}／300レース）"
    lines += [f"**状態：{status}**", "", Sm.to_markdown(index=False, floatfmt=".3f"), "",
              f"候補になったレース（全レース中の割合）：{n}／{int((good & np.array([(d, j, int(r)) in haveK for d, j, r in zip(m['date'], m['jcd'], m['rno'])])).sum())}", ""]
    if n:
        lines += ["### 対象レース（新しい順）", "", R.sort_values(KEY, ascending=False).head(60).to_markdown(index=False, floatfmt=".3f"), ""]
    txt = open(a.out, encoding="utf-8").read() if os.path.exists(a.out) else "# IDEA-056 段階2\n\n"
    txt = (txt.split(HEAD)[0] if HEAD in txt else txt.rstrip("\n") + "\n\n") + "\n".join(lines) + "\n"
    open(a.out, "w", encoding="utf-8").write(txt); print("\n".join(lines[:12]))

if __name__ == "__main__":
    main()
