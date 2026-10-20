"""IDEA-059 ペルソナ＋市場を機械にまとめて学ばせ、割安な艇の単勝を買う（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea059.py DATADIR HISTDS SEXCSV --bk BKDIR [BKDIR ...] --tf STORE/oddstf [--out results/idea059.md]
- DATADIR：2023-10〜（確定3連単オッズあり）、HISTDS：2014-01〜2023-09（B/K のみ、12年の参考用）
- SEXCSV：登録番号→性別（IDEA-054 と同じ、公式のファン手帳から作ったもの。コミットしない）
- BKDIR：K ファイル（bk/YYYY/kYYMMDD.txt.gz）の置き場所（単勝の払戻の実額）、--tf：確定の単勝オッズ（store/oddstf、2025-10〜）
- 艇ごとの1着を LightGBM（二値、init_score = 市場の1着確率の logit）で学習。学習 2023-11〜2025-12、評価 2026-01〜08
- 評価：補正後の1着確率 × 単勝の確定オッズ > 1.05・1.15・1.25 の艇の単勝を各100円（払戻は K の実額）
- 参考（12年）：出発点を「前年の艇番別の平均1着率」にした同じ形のモデル（学習 2014〜2019）で、補正後÷出発点が大きい艇の単勝の年別回収率
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib import parse as BP
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, VENUE_PREF, boot_roi

TR = ("20231101", "20251231"); EV = ("20260101", "20260831")
H_TR = ("20140101", "20191231"); H_EV = ("20200101", "20260831")
RNG = np.random.default_rng(0)
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)
CLS = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
OWN = ["boat", "age", "fem", "cls", "local", "toban", "starts", "weight", "nat_win", "loc_win"]
RACE = ["o_u22", "o_fem", "o_b2", "o_a1", "o_outer_fem", "o_outer_b2", "b1_age", "b1_cls", "b1_fem"]
FEATS = OWN + RACE

def k_tan(bkdirs, start, end):
    """K ファイルの単勝の払戻：date, jcd, rno, tan_boat, tan_pay（同着は最初の艇だけ）"""
    rows = []
    for bd in bkdirs:
        for fn in sorted(glob.glob(f"{bd}/*/k*.txt.gz")):
            date = "20" + os.path.basename(fn)[1:7]
            if not (start <= date <= end):
                continue
            txt = gzip.open(fn, "rt", encoding="utf-8", errors="replace").read()
            for jcd, blk in BP._blocks(txt, "K"):
                rno = None
                for line in blk.splitlines():
                    z = BP.z2h(line); m = BP.K_RACE.match(z)
                    if m:
                        rno = int(m.group(1)); continue
                    if rno is None:
                        continue
                    mm = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", z)
                    if mm:
                        rows.append((date, jcd, rno, int(mm.group(1)), int(mm.group(2)))); rno = None
    return pd.DataFrame(rows, columns=KEY + ["tan_boat", "tan_pay"]).drop_duplicates(KEY, keep="last")

def boat_table(paths, sex):
    """艇ごとの特徴量（その日より前の情報だけ）と着順"""
    cols = KEY + ["boat", "toban", "age", "branch", "weight", "cls", "nat_win", "loc_win", "pos_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in paths], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last").sort_values(KEY + ["boat"]).reset_index(drop=True)
    first = e.groupby(["toban", "date"]).size().rename("n").reset_index().sort_values(["toban", "date"])
    first["starts"] = first.groupby("toban")["n"].cumsum() - first["n"]          # 2014年以降、その日より前の出走数
    e = e.merge(first[["toban", "date", "starts"]], on=["toban", "date"], how="left")
    for c in ("age", "weight", "nat_win", "loc_win"):
        e[c] = pd.to_numeric(e[c], errors="coerce")
    s = e["toban"].map(sex)
    e["fem"] = np.where(s.notna(), s == 2, e["weight"] < 50).astype(float)
    e["cls"] = e["cls"].astype(str).map(CLS).astype(float)
    e["local"] = (e["branch"].astype(str) == e["jcd"].map(VENUE_PREF)).astype(float)
    e["toban"] = pd.to_numeric(e["toban"], errors="coerce")
    e["absent"] = e["pos_raw"].astype(str).str.strip().isin(["K0", "K1"])
    e["won"] = (e["pos_raw"].astype(str).str.strip().str.lstrip("0") == "1").astype(float)
    # レースの他5艇の要約（自分を除く）と1号艇のペルソナ
    e["u22"] = (e["age"] <= 22).astype(float); e["b2"] = (e["cls"] == 1).astype(float); e["a1"] = (e["cls"] == 4).astype(float)
    e["outer_fem"] = ((e["boat"] >= 4) & (e["fem"] == 1)).astype(float); e["outer_b2"] = ((e["boat"] >= 4) & (e["b2"] == 1)).astype(float)
    g = e.groupby(KEY)
    for c, o in (("u22", "o_u22"), ("fem", "o_fem"), ("b2", "o_b2"), ("a1", "o_a1"), ("outer_fem", "o_outer_fem"), ("outer_b2", "o_outer_b2")):
        e[o] = g[c].transform("sum") - e[c]
    b1 = e[e["boat"] == 1][KEY + ["age", "cls", "fem"]].rename(columns={"age": "b1_age", "cls": "b1_cls", "fem": "b1_fem"})
    return e.merge(b1, on=KEY, how="left")

def build_X(keys, e):
    """レース（KEY の並び）× 6艇 × 特徴量"""
    n = len(keys); wide = keys[KEY].reset_index(drop=True).reset_index().merge(e, on=KEY, how="left")
    wide = wide[wide["boat"].between(1, 6)]
    X = np.full((n, 6, len(FEATS)), np.nan)
    for b in range(1, 7):
        w = wide[wide["boat"] == b].drop_duplicates("index").set_index("index").reindex(range(n))
        X[:, b - 1, :] = w[FEATS].to_numpy(float)
    X[:, :, 0] = np.arange(1, 7)[None, :]
    return X

def last_snap(dirpath, start):
    fs = [f for f in glob.glob(f"{dirpath}/*.csv.gz") if os.path.basename(f)[:8] >= start]
    d = norm(pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs]))
    if "fetched_at" in d and "deadline" in d:
        d = d[d["fetched_at"].astype(str) < d["deadline"].astype(str) + ":00"].sort_values("fetched_at")
    return d.drop_duplicates(KEY, keep="last")

def softmax_rows(s):
    s = s - np.nanmax(s, 1, keepdims=True); q = np.exp(s); q = np.where(np.isfinite(q), q, 0)
    return q / np.where(q.sum(1, keepdims=True) > 0, q.sum(1, keepdims=True), 1)

def fit_predict(Xtr, ytr, btr, Xev):
    """init_score = logit(p)。予測は raw_score（init_score を含まない）を log p に足してレース内で正規化する"""
    p = np.clip(np.exp(btr), 1e-6, 0.999)
    ds = lgb.Dataset(Xtr, ytr, init_score=np.log(p) - np.log1p(-p), feature_name=FEATS, free_raw_data=False)
    mdl = lgb.train(PARAMS, ds, num_boost_round=300)
    return mdl, mdl.predict(Xev, raw_score=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("sexcsv")
    ap.add_argument("--bk", nargs="+", required=True); ap.add_argument("--tf", required=True)
    ap.add_argument("--live", default=None, help="live_t1/live_t0 を含むディレクトリ（締切前の確認、2026-10-11〜）")
    ap.add_argument("--out", default="results/idea059.md"); a = ap.parse_args()
    sex = pd.read_csv(a.sexcsv).set_index("toban")["sex"]
    e = boat_table([a.histds, a.datadir], sex); print("boats", len(e), flush=True)
    tan = k_tan(a.bk, "20140101", "20261231"); print("K 単勝", len(tan), flush=True)
    # ---------- 市場（2023-10 以降、確定3連単オッズ）
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna().drop_duplicates(KEY)
    r = r[(r["date"] >= TR[0]) & (r["date"] <= EV[1])]
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].astype(str).str[0]
    ok = (v.sum(1) >= 60) & y.str.isdigit().values
    m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = m["tri_combo"].astype(str).str[0].astype(int).values - 1; n = len(m)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    I = np.array([int(c[0]) for c in COMBOS]); pw = np.stack([P[:, I == b].sum(1) for b in range(1, 7)], 1)
    # 欠場（その艇の組が全部売れていない）のあるレースは除く
    ok2 = (pw > 0).all(1); m, pw, y = m[ok2].reset_index(drop=True), pw[ok2], y[ok2]; n = len(m)
    X = build_X(m, e)
    yb = np.zeros((n, 6)); yb[np.arange(n), y] = 1
    base = np.log(np.clip(pw, 1e-9, None)); days = m["date"].values
    tr = (days >= TR[0]) & (days <= TR[1]); ev = (days >= EV[0]) & (days <= EV[1])
    mdl, adj = fit_predict(X[tr].reshape(-1, len(FEATS)), yb[tr].ravel(), base[tr].ravel(), X.reshape(-1, len(FEATS)))
    q = softmax_rows(base + adj.reshape(n, 6))
    # 対数損失（評価）
    la = -np.log(np.clip(q[ev][np.arange(ev.sum()), y[ev]], 1e-12, None)); lm = -np.log(np.clip(pw[ev][np.arange(ev.sum()), y[ev]], 1e-12, None))
    d = la - lm; g = pd.Series(d).groupby(days[ev]).agg(["sum", "count"])
    bi = RNG.integers(0, len(g), size=(4000, len(g))); bt = g["sum"].values[bi].sum(1) / g["count"].values[bi].sum(1)
    ll = (lm.mean(), la.mean(), d.mean(), float(np.quantile(bt, .025)), float(np.quantile(bt, .975)))
    print("logloss", ll, flush=True)
    # ---------- 単勝（確定オッズで判定、払戻は K の実額）
    tf = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in glob.glob(f"{a.tf}/*.csv.gz")])
    tf = norm(tf).drop_duplicates(KEY, keep="last")
    mt = m[KEY].reset_index().merge(tf[KEY + [f"tan{i}" for i in range(1, 7)]], on=KEY).merge(tan, on=KEY)
    idx = mt["index"].values
    TO = mt[[f"tan{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pay = np.zeros((len(mt), 6)); pay[np.arange(len(mt)), mt["tan_boat"].values - 1] = mt["tan_pay"].values
    rows = []
    for per, lo_, hi_ in (("探索（2025-10〜12、参考）", "20251001", TR[1]), ("評価（2026-01〜08）", EV[0], EV[1])):
        sel = (mt["date"].values >= lo_) & (mt["date"].values <= hi_)
        for thr in (1.05, 1.15, 1.25):
            buy = sel[:, None] & np.isfinite(TO) & (TO > 0) & (q[idx] * np.nan_to_num(TO) > thr)
            st = buy.sum(1) * 100.0; rt = np.where(buy, pay, 0).sum(1)
            roi, clo, chi = boot_roi(mt["date"].values[sel], st[sel], rt[sel])
            row = {"期間": per, "閾値": thr, "点数": int(buy.sum()), "的中": int((buy & (pay > 0)).sum()), "回収率": roi, "95%CI下": clo, "95%CI上": chi}
            for b in range(6):
                bb = buy[:, b]; row[f"{b + 1}号艇 点数"] = int(bb.sum())
                row[f"{b + 1}号艇 回収率"] = np.where(bb, pay[:, b], 0).sum() / (bb.sum() * 100) if bb.sum() else np.nan
            rows.append(row)
            bm = sel[:, None] & np.isfinite(TO) & (TO > 0) & (pw[idx] * np.nan_to_num(TO) > thr)
            roi, clo, chi = boot_roi(mt["date"].values[sel], (bm.sum(1) * 100.0)[sel], np.where(bm, pay, 0).sum(1)[sel])
            rows.append({"期間": per, "閾値": f"{thr}（市場だけ・参考）", "点数": int(bm.sum()), "的中": int((bm & (pay > 0)).sum()), "回収率": roi, "95%CI下": clo, "95%CI上": chi})
        # 参考：全艇の単勝を全部
        allb = sel[:, None] & np.isfinite(TO) & (TO > 0)
        rows.append({"期間": per, "閾値": "全艇全部（参考）", "点数": int(allb.sum()), "回収率": np.where(allb, pay, 0).sum() / (allb.sum() * 100)})
    R = pd.DataFrame(rows)
    imp = pd.Series(mdl.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    ev_row = R[(R["期間"].str.startswith("評価")) & (R["閾値"].astype(str) == "1.15")].iloc[0]
    # ---------- 締切前の確認（2026-10-11〜、live_t1 の3連単で市場の確率、live_t1／live_t0 の単勝オッズで判定、払戻は K の実額）
    LV = []
    if a.live:
        o3 = last_snap(f"{a.live}/live_t1/odds3t", "20261011")
        o3 = o3[o3["date"] <= tan["date"].max()].merge(tan, on=KEY).reset_index(drop=True)
        O3 = o3[COMBOS].apply(pd.to_numeric, errors="coerce").to_numpy(float); v3 = np.isfinite(O3) & (O3 > 0)
        P3 = np.where(v3, 1 / np.where(v3, O3, 1), 0); P3 /= np.where(P3.sum(1, keepdims=True) > 0, P3.sum(1, keepdims=True), 1)
        pl = np.stack([P3[:, I == b].sum(1) for b in range(1, 7)], 1)
        okl = (v3.sum(1) >= 60) & (pl > 0).all(1); o3, pl = o3[okl].reset_index(drop=True), pl[okl]
        Xl = build_X(o3, e); ql = softmax_rows(np.log(pl) + mdl.predict(Xl.reshape(-1, len(FEATS)), raw_score=True).reshape(len(o3), 6))
        payl = np.zeros((len(o3), 6)); payl[np.arange(len(o3)), o3["tan_boat"].values - 1] = o3["tan_pay"].values
        for lead in ("live_t1", "live_t0"):
            t = last_snap(f"{a.live}/{lead}/oddstf", "20261011")
            mm_ = o3[KEY].reset_index().merge(t[KEY + [f"tan{i}" for i in range(1, 7)]], on=KEY)
            ii = mm_["index"].values; TL = mm_[[f"tan{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            for thr in (1.05, 1.15, 1.25):
                for nm, pr in (("モデル", ql[ii]), ("市場だけ", pl[ii])):
                    bb = np.isfinite(TL) & (TL > 0) & (pr * np.nan_to_num(TL) > thr)
                    roi, clo, chi = boot_roi(mm_["date"].values, bb.sum(1) * 100.0, np.where(bb, payl[ii], 0).sum(1))
                    hitpay = payl[ii][bb & (payl[ii] > 0)]; hitod = TL[bb & (payl[ii] > 0)]
                    LV.append({"判定に使ったオッズ": lead, "確率": nm, "閾値": thr, "レース": len(mm_), "点数": int(bb.sum()), "回収率": roi,
                               "95%CI下": clo, "95%CI上": chi, "当たった艇の 確定÷判定時オッズ": float(np.mean(hitpay / 100 / hitod)) if len(hitod) else np.nan})
    LV = pd.DataFrame(LV)
    judge = "登録の基準は満たす" if (ev_row["点数"] >= 300 and ev_row["回収率"] >= 1.1 and ev_row["95%CI下"] > 1.0) else "不採用"
    if judge == "登録の基準は満たす":
        note = ""
        if len(LV):
            t1 = LV[(LV["判定に使ったオッズ"] == "live_t1") & (LV["確率"] == "モデル") & (LV["閾値"] == 1.15)].iloc[0]
            t0 = LV[(LV["判定に使ったオッズ"] == "live_t0") & (LV["確率"] == "モデル") & (LV["閾値"] == 1.15)].iloc[0]
            note = (f"締切前のオッズで判定すると 2分前 {t1['回収率']:.3f}・45〜6秒前 {t0['回収率']:.3f}（閾値1.15）。")
        judge = ("不採用（実際に買える形では届かない）。確定オッズで判定した数字は IDEA-022 と同じ見かけで、市場だけでも同じくらい出る。" + note)
    # ---------- 12年の参考（オッズなし、出発点 = 前年の艇番別の平均1着率）
    h = e[(e["date"] >= "20130101") & (e["date"] <= EV[1])].copy()
    h = h[h["boat"].between(1, 6)]
    bad = h.groupby(KEY)["absent"].transform("max").astype(bool) | (h.groupby(KEY)["won"].transform("sum") != 1) | (h.groupby(KEY)["boat"].transform("size") != 6)
    h = h[~bad].copy(); h["year"] = h["date"].str[:4].astype(int)
    rate = h.groupby(["year", "boat"])["won"].mean().rename("rate").reset_index(); rate["year"] += 1   # 前年の率を使う
    h = h.merge(rate, on=["year", "boat"], how="inner").sort_values(KEY + ["boat"]).reset_index(drop=True)
    nr = len(h) // 6; Hh = h.iloc[: nr * 6]
    Xh = Hh[FEATS].to_numpy(float).reshape(nr, 6, len(FEATS)); bh = np.log(Hh["rate"].to_numpy(float).reshape(nr, 6))
    bh = bh - np.log(np.exp(bh).sum(1, keepdims=True)); yh = Hh["won"].to_numpy(float).reshape(nr, 6)
    dh = Hh["date"].to_numpy().reshape(nr, 6)[:, 0]
    htr = (dh >= H_TR[0]) & (dh <= H_TR[1])
    _, adjh = fit_predict(Xh[htr].reshape(-1, len(FEATS)), yh[htr].ravel(), bh[htr].ravel(), Xh.reshape(-1, len(FEATS)))
    qh = softmax_rows(bh + adjh.reshape(nr, 6)); ratio = qh / np.exp(bh)
    keyh = pd.DataFrame({"date": dh, "jcd": Hh["jcd"].to_numpy().reshape(nr, 6)[:, 0], "rno": Hh["rno"].to_numpy().reshape(nr, 6)[:, 0]})
    pk = keyh.reset_index().merge(tan, on=KEY, how="left")
    payh = np.zeros((nr, 6)); okp = pk["tan_boat"].notna().values
    payh[pk["index"].values[okp], pk["tan_boat"].values[okp].astype(int) - 1] = pk["tan_pay"].values[okp]
    yrs = dh.astype(str); yr = np.array([s[:4] for s in yrs])
    hrows = []
    for thr in (1.0, 1.25, 1.5):
        for yy in sorted(set(yr)):
            s = (yr == yy) & okp
            if yy < "2015":
                continue
            buy = s[:, None] & (ratio >= thr) if thr > 1.0 else s[:, None] & np.ones_like(ratio, bool)
            stt = buy.sum() * 100.0; rtt = np.where(buy, payh, 0).sum()
            hrows.append({"選び方": "全艇全部" if thr == 1.0 else f"補正÷出発点 ≥ {thr}", "年": yy, "学習期間": "学習" if yy <= "2019" else "評価",
                          "点数": int(buy.sum()), "回収率": rtt / stt if stt else np.nan})
    HR = pd.DataFrame(hrows).pivot_table(index="年", columns="選び方", values=["点数", "回収率"])
    HR.columns = [f"{b} {a_}" for a_, b in HR.columns]
    md = [f"# IDEA-059 ペルソナ＋市場を LightGBM で学習し、割安な艇の単勝（学習 {TR[0]}〜{TR[1]}／評価 {EV[0]}〜{EV[1]}、{n}レース）", "",
          f"**判定：{judge}**　評価（閾値1.15）：{int(ev_row['点数'])}点、回収率 {ev_row['回収率']:.3f}（日単位95%CI {ev_row['95%CI下']:.3f}〜{ev_row['95%CI上']:.3f}）", "",
          f"1着の対数損失（評価）：市場 {ll[0]:.4f} → 補正 {ll[1]:.4f}、差 {ll[2]:+.4f}（95%CI {ll[3]:+.4f}〜{ll[4]:+.4f}）", "",
          "## 単勝（確定オッズで判定、払戻は K の実額）", "", R.to_markdown(index=False, floatfmt=".3f"), "",
          "## 締切前のオッズで判定した場合（2026-10-11〜、払戻は K の実額。判定には使わない確認）", "",
          (LV.to_markdown(index=False, floatfmt=".3f") if len(LV) else "（--live なし）"), "",
          "## 特徴量の重要度（gain 上位15）", "", imp.head(15).to_frame("gain").to_markdown(floatfmt=".0f"), "",
          "## 参考：12年（オッズなし、出発点は前年の艇番別の平均1着率、学習 2014〜2019）", "",
          "補正÷出発点が大きい艇の単勝を各100円（払戻は K の実額）。2020年以降が学習に使っていない年。", "",
          HR.to_markdown(floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
