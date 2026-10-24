"""IDEA-075 の共通部品：番組表から IDEA-069/070 の条件 (a)〜(d) と IDEA-053/056 の補正（重みの合計）を出す
- 重み：models/idea056_weights.json（IDEA-053 の探索で学習、固定）。21区分はすべて番組表と開催日程で決まる（オッズの項を含まない）
- (c)(d) の固定値：models/idea075_frozen.json（scripts/freeze_idea075.py）
- 補正後÷市場 = sigmoid(logit(p1) + s) / p1（s は重みの合計）。s > 0 なら p1 が小さいほど大きく、ちょうど r になる p1 は (e^s/r − 1)/(e^s − 1)
"""
import datetime as dt, gzip, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import parse_b
from exp_idea036 import norm, period_start
from exp_idea051 import KEY, boat_feats, segments, meet_day
from exp_idea053 import SEL

ROOT = os.path.join(os.path.dirname(__file__), "..")
ECOLS = KEY + ["boat", "toban", "cls", "age", "nat_win", "loc_win", "deadline", "pos_raw"]

def load_fixed(weights=None, frozen=None):
    W = json.load(open(weights or f"{ROOT}/models/idea056_weights.json", encoding="utf-8"))
    assert [(s, v) for s, v, _ in W["segments"]] == list(SEL), "重みの区分が IDEA-053 と一致しない"
    Z = json.load(open(frozen or f"{ROOT}/models/idea075_frozen.json", encoding="utf-8"))
    return W, Z

def program_path(store, date):
    for sub in ("program", "bk"):
        p = f"{store}/{sub}/{date[:4]}/b{date[2:]}.txt.gz"
        if os.path.exists(p):
            return p
    return None

def load_program(store, date):
    """当日の番組表 → 出走表の行（pos_raw は空）。無ければ None"""
    p = program_path(store, date)
    if not p:
        return None
    rows = parse_b(gzip.open(p, "rt", encoding="utf-8").read(), date)
    e = pd.DataFrame(rows)
    e["pos_raw"] = ""
    return norm(e)

def hist_entries(datadir, date):
    """F持ちの判定・節の日目のための、当日より前の出走表（DATADIR の entries）"""
    e = norm(pd.read_parquet(f"{datadir}/entries.parquet", columns=ECOLS))
    d0 = (pd.Timestamp(date) - pd.Timedelta(days=200)).strftime("%Y%m%d")
    return e[(e["date"] >= d0) & (e["date"] < date)]

def f_counts(e_all, date):
    """当日の各選手の F の数（審査期間の開始から前日まで。IDEA-036 の f_status と同じ期間）"""
    ps = period_start(pd.Timestamp(date)).strftime("%Y%m%d")
    f = e_all[e_all["pos_raw"].astype(str).str.startswith("F") & (e_all["date"] >= ps) & (e_all["date"] < date)]
    return f.groupby("toban").size()

def race_table(prog, hist, store, W, Z):
    """1レース1行：s（重みの合計）、(a)(c)、各艇の級別・年齢・F の数"""
    date = prog["date"].iloc[0]
    e_all = pd.concat([hist, prog[ECOLS]], ignore_index=True)
    bf = boat_feats(e_all)
    m = prog[KEY].drop_duplicates().merge(bf, on=KEY, how="left").reset_index(drop=True)
    S = segments(m, None, False, store)
    dj = e_all[["date", "jcd"]].drop_duplicates().reset_index(drop=True); dj["day"] = meet_day(dj)
    S["節の日目"] = m[["date", "jcd"]].merge(dj, on=["date", "jcd"], how="left")["day"].astype(str).values
    w = np.array([x for _, _, x in W["segments"]]); X = np.zeros((len(m), len(w)))
    hits = [[] for _ in range(len(m))]
    for k, (s, val, _) in enumerate(W["segments"]):
        if s in S:
            X[:, k] = (np.asarray(S[s]).astype(str) == val)
            for i in np.where(X[:, k] > 0)[0]:
                hits[i].append(f"{s}={val}")
    m["s"] = X @ w; m["s_hits"] = ["、".join(h) for h in hits]
    t1 = prog[prog["boat"] == 1][KEY + ["toban"]].rename(columns={"toban": "t1"})
    m = m.merge(t1, on=KEY, how="left")
    top10 = set(Z["c_top10_toban"])
    m["a"] = pd.to_numeric(m["age1"], errors="coerce") <= 22
    m["c"] = m["t1"].isin(top10)
    fc = f_counts(e_all, date)
    wide = prog.assign(fn=prog["toban"].map(fc).fillna(0).astype(int))
    for col in ("cls", "age", "fn", "toban"):
        p = wide.pivot_table(index=KEY, columns="boat", values=col, aggfunc="first")
        p.columns = [f"{col}_{b}" for b in p.columns]
        m = m.merge(p.reset_index(), on=KEY, how="left")
    return m

def ratio(p1, s):
    p1 = np.clip(np.asarray(p1, float), 1e-6, 1 - 1e-6)
    padj = 1 / (1 + np.exp(-(np.log(p1 / (1 - p1)) + s)))
    return padj / p1, padj

def p1_threshold(s, r):
    """補正後÷市場 ≥ r になる市場の1号艇の確率の上限（なければ 0）"""
    es = np.exp(s)
    return float(np.clip((es / r - 1) / (es - 1), 0, 1)) if es > r else 0.0

def fav_dirs(Z, jcd):
    return [d for j, d in Z["d_fav"] if j == jcd]

def wind_from_code(Z, jcd, code):
    """直前情報の風向きの番号 → 8方位（場ごとの対応表。推定）"""
    try:
        c = str(int(float(code)))
    except (TypeError, ValueError):
        return None
    return Z["wind_code_map"].get(jcd, {}).get(c)

def today_jst():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d")
