"""IDEA-059 段階2 締切前のオッズで判定する前向きの検証（docs/ideas.md の登録どおり）
usage:
  python scripts/idea059_fwd.py fit  DATADIR HISTDS SEXCSV                 # 学習（2023-11〜2025-12）して models/idea059_* に保存
  python scripts/idea059_fwd.py eval DATADIR HISTDS SEXCSV STORE [--start 20261021] [--out results/idea059_fwd.md]
- モデル：exp_idea059.py と同じ LightGBM（艇ごとの1着、init_score = 市場の1着確率の logit）。学習後は変更しない
- SEXCSV：登録番号→性別（公式のファン手帳から作ったもの。個人の属性なのでコミットしない。実行時に読むだけ）
- 判定の時点：6分前（live/odds3t の3連単）・締切2分前（live_t1/odds3t）・45〜6秒前（live_t0/bc_votes の3連単の票数を正規化）
  単勝・複勝のオッズも同じ時点のもの（live・live_t1・live_t0 の oddstf、締切前の最後の記録）
- 買い方：補正後の1着確率 × 単勝オッズ > 1.05・1.15・1.25 の艇の単勝を100円。
  複勝（参考）：補正後の2着以内の確率（3連単の確率を1着の補正の比で直して足し上げ）× 複勝の下限オッズ > 同じ閾値
- 払戻は K ファイルの実額（全艇の単勝・複勝）。判定は t1 か t0 で 1,000点以上の時点
"""
import argparse, glob, gzip, json, os, re, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, boot_roi
from exp_idea059 import FEATS, PARAMS, TR, boat_table, build_X, softmax_rows, fit_predict

MODEL = "models/idea059_lgb.txt"; META = "models/idea059_meta.json"
LEADS = (("6分前", "live", "odds3t"), ("締切2分前", "live_t1", "odds3t"), ("45〜6秒前", "live_t0", "bc_votes"))
THRS = (1.05, 1.15, 1.25)
I = np.array([int(c[0]) for c in COMBOS]); J = np.array([int(c[2]) for c in COMBOS])

def k_tanfuku_all(bkdir, start, end):
    """K ファイルの単勝・複勝の払戻（全艇）：date, jcd, rno, tan1..6, fuku1..6（100円あたり、払戻のない艇は0）"""
    rows = []
    for fn in sorted(glob.glob(f"{bkdir}/*/k*.txt.gz")):
        date = "20" + os.path.basename(fn)[1:7]
        if not (start <= date <= end):
            continue
        txt = gzip.open(fn, "rt", encoding="utf-8", errors="replace").read()
        for jcd, blk in BP._blocks(txt, "K"):
            rno = None
            for line in blk.splitlines():
                z = BP.z2h(line); m = BP.K_RACE.match(z)
                if m:
                    rno = int(m.group(1)); tan = [0] * 6; continue
                if rno is None:
                    continue
                mm = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", z)
                if mm:
                    tan[int(mm.group(1)) - 1] = int(mm.group(2)); continue
                mm = re.match(r"^\s+複勝\s+(.*)$", z)
                if mm:
                    fuku = [0] * 6
                    for b, p in re.findall(r"(\d)\s+(\d+)", mm.group(1)):
                        fuku[int(b) - 1] += int(p)
                    rows.append((date, jcd, rno, *tan, *fuku)); rno = None
    cols = KEY + [f"tan{i}" for i in range(1, 7)] + [f"fuku{i}" for i in range(1, 7)]
    d = pd.DataFrame(rows, columns=cols); d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return d.drop_duplicates(KEY, keep="last")

def last_before(dirpath, start):
    """締切前の最後の記録（race ごと）"""
    fs = [f for f in glob.glob(f"{dirpath}/*.csv.gz") if os.path.basename(f)[:8] >= start]
    if not fs:
        return pd.DataFrame(columns=KEY)
    d = norm(pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs], ignore_index=True))
    d = d[d["fetched_at"].astype(str) < d["deadline"].astype(str) + ":00"].sort_values("fetched_at")
    return d.drop_duplicates(KEY, keep="last").reset_index(drop=True)

def market_P(d, kind):
    """3連単の確率（オッズの逆数か票数を正規化）"""
    if kind == "bc_votes":
        A = d[[f"v{c}" for c in COMBOS]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        v = np.isfinite(A) & (A > 0); P = np.where(v, A, 0)
    else:
        O = d[COMBOS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        v = np.isfinite(O) & (O > 0); P = np.where(v, 1 / np.where(v, O, 1), 0)
    s = P.sum(1, keepdims=True); return P / np.where(s > 0, s, 1), v.sum(1)

def cmd_fit(a):
    """exp_idea059.py と同じ作り方で学習して保存"""
    sex = pd.read_csv(a.sexcsv).set_index("toban")["sex"]
    e = boat_table([a.histds, a.datadir], sex)
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= TR[0]) & (r["date"] <= TR[1])]
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].astype(str).str[0]
    ok = (v.sum(1) >= 60) & y.str.isdigit().values
    m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = m["tri_combo"].astype(str).str[0].astype(int).values - 1
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    pw = np.stack([P[:, I == b].sum(1) for b in range(1, 7)], 1)
    ok2 = (pw > 0).all(1); m, pw, y = m[ok2].reset_index(drop=True), pw[ok2], y[ok2]; n = len(m)
    X = build_X(m, e); yb = np.zeros((n, 6)); yb[np.arange(n), y] = 1
    mdl, _ = fit_predict(X.reshape(-1, len(FEATS)), yb.ravel(), np.log(np.clip(pw, 1e-9, None)).ravel(), X[:1].reshape(-1, len(FEATS)))
    os.makedirs("models", exist_ok=True); mdl.save_model(MODEL)
    json.dump({"feats": FEATS, "params": PARAMS, "train": list(TR), "races": int(n), "num_boost_round": 300,
               "note": "exp_idea059.py と同じ学習。性別は実行時に SEXCSV から読む（コミットしない）"}, open(META, "w"), ensure_ascii=False, indent=1)
    print("saved", MODEL, "races", n)

def cmd_eval(a):
    sex = pd.read_csv(a.sexcsv).set_index("toban")["sex"]
    mdl = lgb.Booster(model_file=MODEL)
    e = boat_table([a.histds, a.datadir], sex)
    k = k_tanfuku_all(f"{a.store}/bk", a.start, "29991231")
    if not len(k):
        print("K がまだない"); return
    end = k["date"].max(); PAYT = k[[f"tan{i}" for i in range(1, 7)]].to_numpy(float); PAYF = k[[f"fuku{i}" for i in range(1, 7)]].to_numpy(float)
    rows = []; nraces = {}
    for name, lead, kind in LEADS:
        d = last_before(f"{a.store}/{lead}/{kind}", a.start)
        if not len(d):
            continue
        d = d[d["date"] <= end].reset_index(drop=True)
        P, nv = market_P(d, kind)
        pw = np.stack([P[:, I == b].sum(1) for b in range(1, 7)], 1)
        ok = (nv >= 60) & (pw > 0).all(1); d, P, pw = d[ok].reset_index(drop=True), P[ok], pw[ok]
        tf = last_before(f"{a.store}/{lead}/oddstf", a.start)
        mm = d[KEY].reset_index().merge(tf[KEY + [f"tan{i}" for i in range(1, 7)] + [f"fuku_lo{i}" for i in range(1, 7)]], on=KEY)
        mm = mm.merge(k[KEY].reset_index().rename(columns={"index": "ki"}), on=KEY)
        # 出走表（その日の B ファイル）がまだデータに入っていないレースは除く
        has = mm[KEY].merge(e[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)["_merge"].eq("both").values
        mm = mm[has].reset_index(drop=True)
        if not len(mm):
            continue
        ii = mm["index"].values; ki = mm["ki"].values; nraces[name] = len(mm)
        X = build_X(mm, e); pl = pw[ii]
        q = softmax_rows(np.log(pl) + mdl.predict(X.reshape(-1, len(FEATS)), raw_score=True).reshape(len(mm), 6))
        # 2着以内：3連単の確率を1着の補正の比（q/p）で直して正規化 → 1着か2着にその艇がいる組の合計
        Pa = P[ii] * (q / pl)[:, I - 1]; Pa /= Pa.sum(1, keepdims=True)
        top2 = np.stack([Pa[:, (I == b) | (J == b)].sum(1) for b in range(1, 7)], 1)
        top2m = np.stack([P[ii][:, (I == b) | (J == b)].sum(1) for b in range(1, 7)], 1)
        TO = mm[[f"tan{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        FO = mm[[f"fuku_lo{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        pt, pf = PAYT[ki], PAYF[ki]; days = mm["date"].values
        for ken, prob, probm, Od, pay in (("単勝", q, pl, TO, pt), ("複勝（参考）", top2, top2m, FO, pf)):
            for thr in THRS:
                for nm, pr in (("モデル", prob), ("市場だけ", probm)):
                    bb = np.isfinite(Od) & (Od > 0) & (pr * np.nan_to_num(Od) > thr)
                    roi, lo, hi = boot_roi(days, bb.sum(1) * 100.0, np.where(bb, pay, 0).sum(1))
                    hit = bb & (pay > 0)
                    rows.append({"判定の時点": name, "券種": ken, "確率": nm, "閾値": thr, "レース": len(mm), "点数": int(bb.sum()),
                                 "的中": int(hit.sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi,
                                 "当たった艇の 確定÷判定時オッズ": float(np.mean(pay[hit] / 100 / Od[hit])) if hit.any() else np.nan})
    R = pd.DataFrame(rows)
    main_ = R[(R["券種"] == "単勝") & (R["確率"] == "モデル") & (R["閾値"] == 1.15) & (R["判定の時点"].isin(["締切2分前", "45〜6秒前"]))]
    ready = main_[main_["点数"] >= 1000]
    if len(ready) and ((ready["回収率"] >= 1.1) & (ready["95%CI下"] > 1.0)).any():
        judge = "採用候補（基準を満たした）"
    elif len(ready):
        judge = "1,000点に達したが基準に届かない → 不採用（最長 2026-12-31 まで見る登録だが、達した時点で判定）"
    else:
        judge = "途中経過（1,000点未満）"
    if a.start < "20261021":
        judge = "試し（10-21 より前を含むので参考。判定に数えない）"
    md = [f"# IDEA-059 段階2 締切前のオッズで判定（{a.start}〜{end}、払戻は K の実額）", "",
          f"**状態：{judge}**（判定に使うのは 単勝・モデル・閾値1.15 の 締切2分前／45〜6秒前）", "",
          "対象レース数：" + "、".join(f"{k_}{v}" for k_, v in nraces.items()), "",
          R.to_markdown(index=False, floatfmt=".3f"), "",
          "- モデルは `models/idea059_lgb.txt`（学習 2023-11〜2025-12、変更しない）。性別は実行時に読むファン手帳の表（コミットしない）。",
          "- 6分前は人が手で買える時間の参考。複勝は参考（判定に使わない）。"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("datadir"); f.add_argument("histds"); f.add_argument("sexcsv")
    v = sp.add_parser("eval"); v.add_argument("datadir"); v.add_argument("histds"); v.add_argument("sexcsv"); v.add_argument("store")
    v.add_argument("--start", default="20261021"); v.add_argument("--out", default="results/idea059_fwd.md")
    a = ap.parse_args()
    cmd_fit(a) if a.cmd == "fit" else cmd_eval(a)

if __name__ == "__main__":
    main()
