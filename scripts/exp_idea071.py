"""IDEA-071 1号艇が強いレースで、2着の相方をペルソナ・条件で当てる（2連単 1-X）（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea071.py DATADIR HISTDS SEXCSV STORE HISTSTORE [--out results/idea071.md]
- DATADIR：2023-10〜（確定3連単・2連単オッズ）、HISTDS：2014-01〜2023-09、SEXCSV：登録番号→性別（コミットしない）
- STORE / HISTSTORE：K ファイル（bk/YYYY/k*.txt.gz）の置き場所（2連単の払戻の実額）。STORE には grades.csv も
- 対象レース：IDEA-053/056 の重み（models/idea056_weights.json）で補正した1号艇の1着確率 ≥ 0.6
- 2着モデル：1号艇が勝ったレースだけで、2〜6号艇の「2着か」を LightGBM（二値、init_score = 市場の条件付き2着確率 q2(j|1) の logit）
  q2 は確定3連単から（IDEA-035 の分解：P12[1,j] / P1[1]）。予測は raw_score に log q2 を足して5艇で正規化。学習 2023-11〜2025-12、評価 2026-01〜08
- 本命：補正後÷市場 ≥ 1.15 かつ 補正後の1号艇の1着確率 × 補正後の2着確率 × 確定2連単オッズ(1-j) > 1.0 の 1-j を各100円（払戻は K の実額）
- 参考（オッズを使わない）：補正後÷市場 ≥ 1.15・1.25 の 1-j を各100円
- 参考（12年、オッズなし）：出発点を「前年の艇番別の平均2着率（1号艇1着のとき）」にした同じ形のモデル（学習 2014〜2019）で、
  補正÷出発点 ≥ 1.25 の 1-j の年別回収率
- 進入コース：entries.course はレース後の実際の進入なので使わない（展示の進入は 2025-10〜しかなく、学習期間にない）。艇番で代える
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea051 import FIRST, boat_feats, segments
from exp_idea053 import SEL, design
from exp_idea054 import KEY, boot_roi
from exp_idea059 import PARAMS, TR, EV, H_TR, boat_table, softmax_rows
from exp_idea066 import wind_groups
from idea052 import k_pay

RNG = np.random.default_rng(0)
SECOND = np.array([int(c[2]) for c in COMBOS])
WATER = {"01": 0, "02": 0, "03": 1, "04": 2, "05": 0, "06": 1, "07": 1, "08": 2, "09": 2, "10": 0, "11": 0, "12": 0,
         "13": 0, "14": 2, "15": 2, "16": 2, "17": 2, "18": 2, "19": 2, "20": 2, "21": 0, "22": 1, "23": 0, "24": 2}   # 淡水0・汽水1・海水2
GRP = {"1コース有利": 0, "中間": 1, "1コース不利": 2}
OWN = ["boat", "age", "fem", "cls", "local", "toban", "starts", "weight", "nat_win", "loc_win", "motor_2r", "boat_2r", "exh_rank", "st90"]
RACE = ["jcd_n", "water", "wind", "wave", "wgrp", "b1_age", "b1_cls"]
FEATS = OWN + RACE
NAMES = {"boat": "艇番", "age": "年齢", "fem": "女子", "cls": "級別", "local": "地元", "toban": "登録番号（新しさ）", "starts": "出走数",
         "weight": "体重", "nat_win": "全国勝率", "loc_win": "当地勝率", "motor_2r": "モーター2連率", "boat_2r": "ボート2連率",
         "exh_rank": "展示タイム順位", "st90": "90日平均ST", "jcd_n": "場", "water": "水質", "wind": "風速", "wave": "波高",
         "wgrp": "風向きの群", "b1_age": "1号艇の年齢", "b1_cls": "1号艇の級別"}

def extras(paths):
    """艇ごとの追加特徴量：モーター・ボート2連率、展示タイム順位、90日平均ST（その日より前だけ）"""
    cols = KEY + ["boat", "toban", "motor_2r", "boat_2r", "exh_time", "st"]
    x = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in paths], ignore_index=True)
    x = x.drop_duplicates(KEY + ["boat"], keep="last").reset_index(drop=True)
    for c in ("motor_2r", "boat_2r", "exh_time", "st"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["exh_rank"] = x.groupby(KEY)["exh_time"].rank(method="average")
    s = x[x["st"].between(0.0, 1.0)].groupby(["toban", "date"])["st"].agg(["sum", "count"]).reset_index()
    s["dt"] = pd.to_datetime(s["date"]); s = s.sort_values(["toban", "dt"]).set_index("dt")
    roll = s.groupby("toban")[["sum", "count"]].rolling("90D", closed="left").sum().reset_index()
    roll["date"] = roll["dt"].dt.strftime("%Y%m%d"); roll["st90"] = roll["sum"] / roll["count"].replace(0, np.nan)
    return x.merge(roll[["toban", "date", "st90"]], on=["toban", "date"], how="left")[KEY + ["boat", "motor_2r", "boat_2r", "exh_rank", "st90"]]

def race_conds(paths):
    cols = KEY + ["boat", "course", "pos_raw"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in paths], ignore_index=True).drop_duplicates(KEY + ["boat"], keep="last")
    r = pd.concat([norm(pd.read_parquet(f"{p}/races.parquet", columns=KEY + ["wind", "wind_dir", "wave"])) for p in paths], ignore_index=True)
    r = r.drop_duplicates(KEY, keep="last")
    G = wind_groups(e, r)
    r = r.merge(G[["jcd", "wind_dir", "grp"]], on=["jcd", "wind_dir"], how="left")
    r["wgrp"] = r["grp"].fillna("中間").map(GRP).astype(float)
    r["wind"] = pd.to_numeric(r["wind"], errors="coerce"); r["wave"] = pd.to_numeric(r["wave"], errors="coerce")
    r["jcd_n"] = pd.to_numeric(r["jcd"], errors="coerce"); r["water"] = r["jcd"].map(WATER).astype(float)
    return r[KEY + ["wind", "wave", "wgrp", "jcd_n", "water"]]

def build_X(keys, t):
    """レース（KEY の並び）× 2〜6号艇 × 特徴量"""
    n = len(keys); wide = keys[KEY].reset_index(drop=True).reset_index().merge(t, on=KEY, how="left")
    X = np.full((n, 5, len(FEATS)), np.nan)
    for b in range(2, 7):
        w = wide[wide["boat"] == b].drop_duplicates("index").set_index("index").reindex(range(n))
        X[:, b - 2, :] = w[FEATS].to_numpy(float)
    X[:, :, 0] = np.arange(2, 7)[None, :]
    return X

def fit_predict(Xtr, ytr, btr, Xev):
    """init_score = logit(q2)。予測は raw_score（init_score を含まない）を log q2 に足してレース内で正規化する"""
    p = np.clip(np.exp(btr), 1e-6, 0.999)
    ds = lgb.Dataset(Xtr, ytr, init_score=np.log(p) - np.log1p(-p), feature_name=FEATS, free_raw_data=False)
    mdl = lgb.train(PARAMS, ds, num_boost_round=300)
    return mdl, mdl.predict(Xev, raw_score=True)

def ll_ci(days, la, lm):
    d = la - lm; g = pd.Series(d).groupby(days).agg(["sum", "count"])
    bi = RNG.integers(0, len(g), size=(4000, len(g))); bt = g["sum"].values[bi].sum(1) / g["count"].values[bi].sum(1)
    return lm.mean(), la.mean(), d.mean(), float(np.quantile(bt, .025)), float(np.quantile(bt, .975)), len(d)

def pay2t(pay, keys):
    """レース × 2〜6号艇：2連単 1-j の払戻（100円あたり）。K に2連単の行がないレースは NaN"""
    p = pay[pay["kind"] == "2t"]; p = p[p["combo"].str.match(r"^1-[2-6]$")]
    have = pay[pay["kind"] == "2t"][KEY].drop_duplicates().assign(have=True)
    k = keys[KEY].reset_index(drop=True).reset_index()
    out = np.zeros((len(k), 5))
    h = k.merge(have, on=KEY, how="left")["have"].fillna(False).values.astype(bool)
    q = k.merge(p, on=KEY)
    out[q["index"].values, q["combo"].str[2].astype(int).values - 2] = q["pay"].values
    out[~h] = np.nan
    return out

def bet_rows(label, buy, pay, days):
    st = buy.sum(1) * 100.0; rt = np.where(buy, pay, 0).sum(1)
    roi, lo, hi = boot_roi(days, st, rt)
    row = {"買い方": label, "点数": int(buy.sum()), "的中": int((buy & (pay > 0)).sum()), "回収率": roi, "95%CI下": lo, "95%CI上": hi}
    for j in range(5):
        bb = buy[:, j]; row[f"1-{j + 2} 点数"] = int(bb.sum())
        row[f"1-{j + 2} 回収率"] = np.where(bb, pay[:, j], 0).sum() / (bb.sum() * 100) if bb.sum() else np.nan
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("sexcsv")
    ap.add_argument("store"); ap.add_argument("histstore"); ap.add_argument("--out", default="results/idea071.md"); a = ap.parse_args()
    paths = [a.histds, a.datadir]
    sex = pd.read_csv(a.sexcsv).set_index("toban")["sex"]
    t = boat_table(paths, sex).merge(extras(paths), on=KEY + ["boat"], how="left").merge(race_conds(paths), on=KEY, how="left")
    print("boats", len(t), flush=True)
    # ---- 市場：確定3連単から p1 と q2(j|1)、IDEA-053/056 の補正
    r = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["tri_combo"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    r = r[(r["date"] >= "20231001") & (r["date"] <= EV[1])]
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]))
    m = r.merge(norm(pd.read_parquet(f"{a.datadir}/odds.parquet")), on=KEY).merge(boat_feats(e), on=KEY, how="left").reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0); y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y); m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True); p1 = P[:, FIRST == 1].sum(1)
    P12 = np.stack([P[:, (FIRST == 1) & (SECOND == j)].sum(1) for j in range(2, 7)], 1)
    good = (p1 > 0) & (P12 > 0).all(1)
    m, P12, p1, y = m[good].reset_index(drop=True), P12[good], p1[good], y[good]; n = len(m); m["p1"] = p1
    q2m = P12 / P12.sum(1, keepdims=True)
    days = m["date"].values; tr = (days >= TR[0]) & (days <= TR[1]); ev = (days >= EV[0]) & (days <= EV[1])
    W = json.load(open(os.path.join(os.path.dirname(__file__), "..", "models", "idea056_weights.json"), encoding="utf-8"))
    wmap = {(s, v_): w_ for s, v_, w_ in W["segments"]}; w = np.array([wmap[tuple(x)] for x in SEL])
    X1 = design(segments(m, tr, True, a.store), n)
    off = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1)); padj = 1 / (1 + np.exp(-(off + X1 @ w)))
    win1 = FIRST[y] == 1; sec = SECOND[y] - 2
    # ---- 2着モデル（1号艇が勝ったレースだけで学習）
    X = build_X(m, t); base = np.log(q2m)
    yb = np.zeros((n, 5)); yb[np.arange(n), np.clip(sec, 0, 4)] = 1
    trw = tr & win1
    mdl, adj = fit_predict(X[trw].reshape(-1, len(FEATS)), yb[trw].ravel(), base[trw].ravel(), X.reshape(-1, len(FEATS)))
    q2a = softmax_rows(base + adj.reshape(n, 5)); ratio = q2a / q2m
    print("model done", flush=True)
    LL = []
    for nm, s in (("評価・1号艇1着の全レース", ev & win1), ("評価・対象レース（補正後 ≥0.6）で1号艇1着", ev & win1 & (padj >= 0.6)),
                  ("探索・対象レース（学習に使った期間、参考）", tr & win1 & (padj >= 0.6))):
        ii = np.where(s)[0]
        la = -np.log(np.clip(q2a[ii, sec[ii]], 1e-12, None)); lm = -np.log(np.clip(q2m[ii, sec[ii]], 1e-12, None))
        L = ll_ci(days[ii], la, lm); LL.append({"範囲": nm, "レース": L[5], "市場": L[0], "補正後": L[1], "差": L[2], "95%CI下": L[3], "95%CI上": L[4]})
    LL = pd.DataFrame(LL)
    # ---- 賭け（評価・対象レース）
    pay = pd.concat([k_pay(a.store, "20251001")], ignore_index=True); pay["rno"] = pay["rno"].astype(int)
    o2 = norm(pd.read_parquet(f"{a.datadir}/odds2tf.parquet"))[KEY + [f"1-{j}" for j in range(2, 7)]]
    tg = ev & (padj >= 0.6)
    mt = m[KEY].assign(i=np.arange(n))[tg].merge(o2, on=KEY).reset_index(drop=True)
    idx = mt["i"].values; O2 = mt[[f"1-{j}" for j in range(2, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    PY = pay2t(pay, mt.assign(rno=mt["rno"].astype(int))); hv = np.isfinite(PY).all(1)
    idx, O2, PY, mt = idx[hv], O2[hv], PY[hv], mt[hv].reset_index(drop=True); dd = mt["date"].values
    oko = np.isfinite(O2) & (O2 > 0); O2z = np.nan_to_num(O2)
    pa, qa, qm, rt_ = padj[idx][:, None], q2a[idx], q2m[idx], ratio[idx]
    rows = [bet_rows("本命：補正後÷市場 ≥1.15 かつ 確率×確定2連単オッズ > 1.0", oko & (rt_ >= 1.15) & (pa * qa * O2z > 1.0), PY, dd),
            bet_rows("参考：補正後÷市場 ≥1.15（オッズなし）", oko & (rt_ >= 1.15), PY, dd),
            bet_rows("参考：補正後÷市場 ≥1.25（オッズなし）", oko & (rt_ >= 1.25), PY, dd),
            bet_rows("参考：市場だけ 確率×確定2連単オッズ > 1.0（モデルなし）", oko & (pa * qm * O2z > 1.0), PY, dd),
            bet_rows("参考：補正後の2着確率が最大の1点", oko & (qa == qa.max(1, keepdims=True)), PY, dd),
            bet_rows("参考：市場の2着確率が最大の1点", oko & (qm == qm.max(1, keepdims=True)), PY, dd),
            bet_rows("参考：1-x 5点全部", oko, PY, dd)]
    R = pd.DataFrame(rows)
    hit = PY > 0; ratio_hit = (PY[hit] / 100 / O2[hit])
    imp = pd.Series(mdl.feature_importance("gain"), index=[NAMES[f] for f in FEATS]).sort_values(ascending=False)
    main_ = R.iloc[0]
    ok_ = main_["点数"] >= 300 and main_["回収率"] >= 1.1 and main_["95%CI下"] > 1.0
    judge = "採用候補（締切前オッズでの前向き検証へ）" if ok_ else "不採用"
    print(R[["買い方", "点数", "回収率", "95%CI下", "95%CI上"]], LL, flush=True)
    # ---- 12年の参考（オッズなし）
    h = t[(t["date"] >= "20130101") & (t["date"] <= EV[1]) & t["boat"].between(1, 6)].copy()
    pos = h["pos_raw"].astype(str).str.strip()
    h["sec"] = (pos == "02").astype(float)
    bad = h.groupby(KEY)["absent"].transform("max").astype(bool) | (h.groupby(KEY)["won"].transform("sum") != 1) | \
          (h.groupby(KEY)["sec"].transform("sum") != 1) | (h.groupby(KEY)["boat"].transform("size") != 6)
    h = h[~bad].copy(); h["year"] = h["date"].str[:4].astype(int)
    w1 = h[h["boat"] == 1][KEY + ["won", "cls"]].rename(columns={"won": "w1", "cls": "c1"}); h = h.merge(w1, on=KEY)
    rate = h[(h["w1"] == 1) & (h["boat"] >= 2)].groupby(["year", "boat"])["sec"].mean().rename("rate").reset_index(); rate["year"] += 1   # 前年の率
    h = h[h["boat"] >= 2].merge(rate, on=["year", "boat"], how="inner").sort_values(KEY + ["boat"]).reset_index(drop=True)
    nr = len(h) // 5; Hh = h.iloc[: nr * 5]
    Xh = Hh[FEATS].to_numpy(float).reshape(nr, 5, len(FEATS)); bh = np.log(Hh["rate"].to_numpy(float).reshape(nr, 5))
    bh = bh - np.log(np.exp(bh).sum(1, keepdims=True)); yh = Hh["sec"].to_numpy(float).reshape(nr, 5)
    first = lambda c: Hh[c].to_numpy().reshape(nr, 5)[:, 0]
    dh, w1h, c1h = first("date"), first("w1").astype(bool), first("c1")
    htr = (dh >= H_TR[0]) & (dh <= H_TR[1]) & w1h
    _, adjh = fit_predict(Xh[htr].reshape(-1, len(FEATS)), yh[htr].ravel(), bh[htr].ravel(), Xh.reshape(-1, len(FEATS)))
    qh = softmax_rows(bh + adjh.reshape(nr, 5)); rh = qh / np.exp(bh)
    print("hist model done", flush=True)
    hp = pd.concat([k_pay(a.histstore, "20140101"), k_pay(a.store, "20231001")], ignore_index=True)
    hp["rno"] = hp["rno"].astype(int); hp = hp.drop_duplicates(KEY + ["kind", "combo"], keep="last")
    keyh = pd.DataFrame({"date": dh, "jcd": first("jcd"), "rno": first("rno").astype(int)})
    PH = pay2t(hp, keyh); okh = np.isfinite(PH).all(1); PHz = np.nan_to_num(PH)
    yr = np.array([s[:4] for s in dh.astype(str)]); hrows = []
    for yy in sorted(set(yr)):
        if yy < "2015":
            continue
        for sub, sm in (("全レース", np.ones(nr, bool)), ("1号艇A1", c1h == 4)):
            s = (yr == yy) & okh & sm
            for lab, buy in (("1-x 全部", s[:, None] & np.ones_like(rh, bool)), ("補正÷出発点 ≥1.25", s[:, None] & (rh >= 1.25))):
                hrows.append({"年": yy, "列": f"{sub}・{lab}", "点数": int(buy.sum()),
                              "回収率": np.where(buy, PHz, 0).sum() / (buy.sum() * 100) if buy.sum() else np.nan})
    HR = pd.DataFrame(hrows).pivot_table(index="年", columns="列", values=["点数", "回収率"])
    HR.columns = [f"{b} {a_}" for a_, b in HR.columns]
    l0 = LL.iloc[1]
    md = [f"# IDEA-071 1号艇が強いレースで2着の相方を当てる（2連単 1-X、学習 {TR[0]}〜{TR[1]}／評価 {EV[0]}〜{EV[1]}）", "",
          f"**判定：{judge}**　本命（評価）：{int(main_['点数'])}点、回収率 {main_['回収率']:.3f}（日単位95%CI {main_['95%CI下']:.3f}〜{main_['95%CI上']:.3f}）", "",
          f"2着の対数損失（評価・対象レースで1号艇1着、{int(l0['レース'])}レース）：市場 {l0['市場']:.4f} → 補正 {l0['補正後']:.4f}、差 {l0['差']:+.4f}（95%CI {l0['95%CI下']:+.4f}〜{l0['95%CI上']:+.4f}）", "",
          f"対象：評価期間で補正後の1号艇の1着確率 ≥0.6、確定2連単オッズと K の2連単払戻があるレース（{len(mt)}レース、1号艇の1着率 {win1[idx].mean():.3f}、補正後の見込み {padj[idx].mean():.3f}）。",
          "q2 は確定3連単から、判定の2連単オッズも確定（締切後も動いたあとの値）。確定オッズで選んだ数字は見かけで上ぶれしうる（IDEA-059 の教訓）。", "",
          "## 2着の対数損失（市場の条件付き2着確率 vs 補正後、日単位ブートストラップ）", "", LL.to_markdown(index=False, floatfmt=".4f"), "",
          "## 2連単 1-X（評価・対象レース、払戻は K の実額、相方の艇番別も）", "", R.to_markdown(index=False, floatfmt=".3f"), "",
          f"当たった組の 確定払戻÷確定2連単オッズ の平均：{ratio_hit.mean():.3f}（1 なら確定オッズどおり）", "",
          "## 特徴量の重要度（gain 上位10）", "", imp.head(10).to_frame("gain").to_markdown(floatfmt=".0f"), "",
          "## 参考：12年（オッズなし、出発点は前年の艇番別の平均2着率（1号艇1着のとき）、学習 2014〜2019 の1号艇1着レース）", "",
          "補正÷出発点 ≥1.25 の 1-j を各100円（払戻は K の実額）。1号艇の強さは市場がないので「全レース」と「1号艇A1」で見る。2020年以降が学習に使っていない年。", "",
          HR.to_markdown(floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
