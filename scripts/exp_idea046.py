"""IDEA-046 市場が見落とした「荒れ」を機械に探させる（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea046.py DATADIR STOREDIR [--out results/idea046.md]
- 目的変数：万舟（確定100倍以上）が出たか − 市場の見込み（確定100倍以上の組の確率の合計）。LightGBM の回帰
- 特徴量は締切前に分かるものだけ：場・水質・時間帯・グレード・天候・風・波・水温、6艇の級別・勝率・モーター・展示タイム、
  F持ち・L持ちの数、選手の過去90日の成績（当日より前のみ）、今節のここまでの着順（当日より前）、その日・その場の前のレースまでの荒れ具合、オッズの形
- 天候・風・波・水温は結果ファイル（K）の値で、展示時とほぼ同じとみなす。展示ST・展示の進入は直前情報（2025-10 以降のみ、それ以前は欠損）
- 学習 2023-11〜2025-12、評価 2026-01〜08。評価で予測の上位10%のレースに限って、100倍以上を市場の確率の高い順に10点（各100円）
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm, period_start

KEY = ["date", "jcd", "rno"]; TR = ("20231101", "20251231"); EV = ("20260101", "20260831")
WATER = {"01": 0, "02": 0, "03": 2, "04": 1, "05": 0, "06": 2, "07": 2, "08": 1, "09": 1, "10": 0, "11": 0, "12": 0,
         "13": 0, "14": 1, "15": 1, "16": 1, "17": 1, "18": 1, "19": 1, "20": 1, "21": 0, "22": 2, "23": 0, "24": 1}  # 0 淡水 1 海水 2 汽水
CLS = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
RNG = np.random.default_rng(0)

def market(datadir):
    """レースごとの確定オッズ・市場の確率・万舟・見込み"""
    r = norm(pd.read_parquet(f"{datadir}/races.parquet"))
    r = r.dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    m = r.merge(norm(pd.read_parquet(f"{datadir}/odds.parquet")), on=KEY).reset_index(drop=True)
    O = m[COMBOS].to_numpy(float); v = np.isfinite(O) & (O > 0)
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    ok = (v.sum(1) >= 60) & ~pd.isna(y)
    m, O, v = m[ok].reset_index(drop=True), O[ok], v[ok]; y = y[ok].astype(int)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P /= P.sum(1, keepdims=True)
    big = v & (O >= 100)
    m["manshu"] = (O[np.arange(len(m)), y] >= 100).astype(float)
    m["exp_m"] = np.where(big, P, 0).sum(1)
    return m, O, P, v, y

def fl_status(e, prefix):
    """選手 × 日付 → いまの審査期間で当日より前に着順 prefix（F / L）があるか"""
    f = e.loc[e["pos_raw"].astype(str).str.startswith(prefix), ["toban", "date"]].copy()
    f["fd"] = pd.to_datetime(f["date"]); f = f[["toban", "fd"]]
    q = e[["toban", "date"]].drop_duplicates().copy(); q["d"] = pd.to_datetime(q["date"]); q["ps"] = q["d"].map(period_start)
    mm = q.merge(f, on="toban", how="left")
    mm["hit"] = (mm["fd"] < mm["d"]) & (mm["fd"] >= mm["ps"])
    return mm.groupby(["toban", "date"])["hit"].any().rename(prefix).reset_index()

def player_recent(e):
    """選手 × 日付 → 当日より前の90日間の 出走数・1着率・平均着順・1コースの1着率・平均ST"""
    x = e[["toban", "date", "pos", "course", "st"]].copy(); x["d"] = pd.to_datetime(x["date"])
    x["n"] = x["pos"].notna().astype(float); x["w"] = (x["pos"] == 1).astype(float); x["ps"] = x["pos"].fillna(0)
    x["n1"] = (x["course"] == 1).astype(float); x["w1"] = ((x["course"] == 1) & (x["pos"] == 1)).astype(float)
    x["stn"] = x["st"].notna().astype(float); x["sts"] = x["st"].fillna(0)
    daily = x.groupby(["toban", "d"])[["n", "w", "ps", "n1", "w1", "stn", "sts"]].sum().reset_index().sort_values(["toban", "d"])
    roll = daily.set_index("d").groupby("toban")[["n", "w", "ps", "n1", "w1", "stn", "sts"]].rolling("90D", closed="left").sum().reset_index()
    roll["date"] = roll["d"].dt.strftime("%Y%m%d")
    out = pd.DataFrame({"toban": roll["toban"], "date": roll["date"], "r_n": roll["n"],
                        "r_win": roll["w"] / roll["n"].where(roll["n"] > 0), "r_pos": roll["ps"] / roll["n"].where(roll["n"] > 0),
                        "r_win1": roll["w1"] / roll["n1"].where(roll["n1"] > 0), "r_st": roll["sts"] / roll["stn"].where(roll["stn"] > 0)})
    return out

def meet_so_far(e):
    """選手 × 場 × 日付 → 同じ場で当日より前の6日間の平均着順（今節のここまで）"""
    x = e[["toban", "jcd", "date", "pos"]].copy(); x["d"] = pd.to_datetime(x["date"])
    x["n"] = x["pos"].notna().astype(float); x["ps"] = x["pos"].fillna(0)
    daily = x.groupby(["toban", "jcd", "d"])[["n", "ps"]].sum().reset_index().sort_values(["toban", "jcd", "d"])
    roll = daily.set_index("d").groupby(["toban", "jcd"])[["n", "ps"]].rolling("6D", closed="left").sum().reset_index()
    roll["date"] = roll["d"].dt.strftime("%Y%m%d")
    return pd.DataFrame({"toban": roll["toban"], "jcd": roll["jcd"], "date": roll["date"], "m_n": roll["n"],
                         "m_pos": roll["ps"] / roll["n"].where(roll["n"] > 0)})

def grades(m, store):
    g = pd.read_csv(f"{store}/grades.csv", dtype={"jcd": str, "start": str, "end": str}); g["jcd"] = g["jcd"].str.zfill(2)
    gr = np.zeros(len(m)); lad = np.zeros(len(m))
    for r in g.itertuples():
        sel = ((m["jcd"] == r.jcd) & (m["date"] >= r.start) & (m["date"] <= r.end)).values
        s = str(r.grade); gr[sel] = 4 if s.startswith("SG") else 3 if s.startswith("G1") else 2 if s.startswith("G2") else 1 if s.startswith("G3") else 0
        if any(k in str(r.title) for k in ("レディース", "女子")):
            lad[sel] = 1
    return gr, lad

def boot(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(g["rt"].sum() / g["st"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def build(datadir, store):
    """特徴量を作ってモデルを学習し、全レースの予測を返す（IDEA-047 からも使う）"""
    m, O, P, v, y = market(datadir)
    keep = ((m["date"] >= TR[0]) & (m["date"] <= EV[1])).values
    # その日・その場の前のレースまでの荒れ具合（全期間で計算してから絞る）
    res = (m["manshu"] - m["exp_m"]).values
    tmp = pd.DataFrame({"date": m["date"], "jcd": m["jcd"], "rno": m["rno"], "res": res, "man": m["manshu"]}).sort_values(["date", "jcd", "rno"])
    g = tmp.groupby(["date", "jcd"])
    tmp["day_res"] = g["res"].cumsum() - tmp["res"]; tmp["day_man"] = g["man"].cumsum() - tmp["man"]; tmp["day_n"] = g.cumcount()
    m = m.merge(tmp[["date", "jcd", "rno", "day_res", "day_man", "day_n"]], on=KEY, how="left")
    m, O, P, v, y = m[keep].reset_index(drop=True), O[keep], P[keep], v[keep], y[keep]
    n = len(m); print("races", n, flush=True)
    # 出走表・選手
    e = norm(pd.read_parquet(f"{datadir}/entries.parquet"))
    e = e[e["date"] >= "20230701"]
    rec = player_recent(e); met = meet_so_far(e)
    fs = fl_status(e, "F"); ls_ = fl_status(e, "L")
    ee = e.merge(rec, on=["toban", "date"], how="left").merge(met, on=["toban", "jcd", "date"], how="left")
    ee = ee.merge(fs, on=["toban", "date"], how="left").merge(ls_, on=["toban", "date"], how="left")
    ee["clsn"] = ee["cls"].map(CLS)
    ee = ee[ee["date"] >= TR[0]]
    cols = ["clsn", "age", "weight", "nat_win", "nat_2r", "loc_win", "loc_2r", "motor_2r", "boat_2r", "exh_time",
            "r_n", "r_win", "r_pos", "r_win1", "r_st", "m_n", "m_pos"]
    wide = ee.pivot_table(index=KEY, columns="boat", values=cols, aggfunc="first")
    wide.columns = [f"{c}_{b}" for c, b in wide.columns]
    fl = ee.groupby(KEY).agg(nF=("F", "sum"), nL=("L", "sum"), deadline=("deadline", "first")).reset_index()
    X = m[KEY].merge(wide.reset_index(), on=KEY, how="left").merge(fl, on=KEY, how="left")
    # 展示タイムの順位（1号艇の相対）
    ex = X[[f"exh_time_{b}" for b in range(1, 7)]].to_numpy(float)
    X["exh_rank1"] = (ex < ex[:, [0]]).sum(1).astype(float); X.loc[~np.isfinite(ex[:, 0]), "exh_rank1"] = np.nan
    X["exh_spread"] = np.nanmax(ex, 1) - np.nanmin(ex, 1)
    # 直前情報（2025-10 以降）
    b = norm(pd.read_parquet(f"{datadir}/beforeinfo.parquet", columns=KEY + [f"ex_st{i}" for i in range(1, 7)] + [f"ex_course{i}" for i in range(1, 7)]))
    b = b.drop_duplicates(KEY, keep="last")
    X = X.merge(b, on=KEY, how="left")
    exc = X[[f"ex_course{i}" for i in range(1, 7)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X["not_wakunari"] = np.where(np.isfinite(exc).all(1), (exc != np.arange(1, 7)).any(1).astype(float), np.nan)
    for i in range(1, 7):
        X[f"ex_st{i}"] = pd.to_numeric(X[f"ex_st{i}"], errors="coerce")
    X = X.drop(columns=[f"ex_course{i}" for i in range(1, 7)])
    # 場・天気・時間など
    X["jcd_n"] = m["jcd"].astype(int); X["water"] = m["jcd"].map(WATER); X["rno"] = m["rno"]
    X["hour"] = pd.to_numeric(X["deadline"].astype(str).str[:2], errors="coerce") + pd.to_numeric(X["deadline"].astype(str).str[3:5], errors="coerce") / 60
    X = X.drop(columns=["deadline"])
    X["weather"] = m["weather"].map({"晴": 0, "曇り": 1, "雨": 2, "雪": 3, "霧": 4})
    X["wind"] = pd.to_numeric(m["wind"], errors="coerce"); X["wave"] = pd.to_numeric(m["wave"], errors="coerce")
    X["wind_dir"] = m["wind_dir"].map({"無風": 0, "北": 1, "北東": 2, "東": 3, "南東": 4, "南": 5, "南西": 6, "西": 7, "北西": 8})
    X["dist"] = pd.to_numeric(m["dist"], errors="coerce")
    gr, lad = grades(m, store); X["grade"] = gr; X["ladies"] = lad
    X["day_res"] = m["day_res"]; X["day_man"] = m["day_man"]; X["day_n"] = m["day_n"]
    first = np.array([int(c[0]) for c in COMBOS])
    X["p1"] = P[:, first == 1].sum(1); X["fav"] = P.max(1)
    X["entropy"] = -(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(1)
    X["exp_m"] = m["exp_m"].values
    feats = [c for c in X.columns if c not in KEY]
    F = X[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target = (m["manshu"] - m["exp_m"]).values
    days = m["date"].values; tr = days <= TR[1]; ev = days >= EV[0]
    print("features", len(feats), flush=True)
    ds = lgb.Dataset(F[tr], target[tr], feature_name=feats, free_raw_data=False)
    mdl = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8,
                         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0), ds, num_boost_round=400)
    pred = mdl.predict(F)
    return m, O, P, v, y, pred, mdl, feats, target, days, tr, ev, n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea046.md")
    a = ap.parse_args()
    m, O, P, v, y, pred, mdl, feats, target, days, tr, ev, n = build(a.datadir, a.store)
    # 買い方：100倍以上を市場の確率の高い順に10点
    hit = np.zeros_like(v); hit[np.arange(n), y] = True; big = v & (O >= 100)
    order = np.argsort(-np.where(big, P, -1), 1)
    buy = np.zeros_like(v); buy[np.arange(n)[:, None], order[:, :10]] = True; buy &= big
    st = buy.sum(1) * 100.0; rt = np.where(buy & hit, O * 100, 0).sum(1)
    man = m["manshu"].values; expm = m["exp_m"].values
    rows = []
    for per, msk in (("学習", tr), ("評価", ev)):
        q = pd.qcut(pred[msk], 10, labels=False, duplicates="drop")
        for d in range(10):
            s = np.where(msk)[0][q == d]
            roi, lo, hi = boot(days[s], st[s], rt[s])
            rows.append({"期間": per, "予測の十分位（9=荒れる）": d, "レース": len(s), "万舟率": man[s].mean(), "市場の見込み": expm[s].mean(),
                         "実際÷見込み": man[s].sum() / expm[s].sum(), "穴10点の回収率": roi, "95%CI下": lo, "95%CI上": hi})
    R = pd.DataFrame(rows)
    thr = np.quantile(pred[ev], 0.9); s = ev & (pred >= thr)
    roi, lo, hi = boot(days[s], st[s], rt[s])
    judge = "採用候補（前向きの検証へ）" if (s.sum() >= 300 and roi >= 1.1 and lo > 1.0) else "不採用"
    corr = float(np.corrcoef(pred[ev], target[ev])[0, 1])
    imp = pd.Series(mdl.feature_importance("gain"), index=feats).sort_values(ascending=False).head(15)
    md = [f"# IDEA-046 市場が見落とした荒れを機械に探させる（学習 {TR[0]}〜{TR[1]}／評価 {EV[0]}〜{EV[1]}、{n}レース、特徴量 {len(feats)}）", "",
          f"**判定：{judge}**　評価で予測の上位10%（{int(s.sum())}レース）：万舟の実際÷見込み {man[s].sum() / expm[s].sum():.3f}、"
          f"穴10点の回収率 {roi:.3f}（日単位95%CI {lo:.3f}〜{hi:.3f}）", "",
          f"参考：評価期間の 予測 と 実際の残差（万舟 − 見込み）の相関 {corr:.4f}", "",
          "注：天候・風・波は結果ファイルの値。展示ST・展示の進入は直前情報で 2025-10 以降のみ（それ以前は欠損として学習）。", "",
          "## 予測の十分位ごと", "", R.to_markdown(index=False, floatfmt=".3f"), "",
          "## 特徴量の重要度（gain 上位15）", "", imp.to_frame("gain").to_markdown(floatfmt=".1f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
