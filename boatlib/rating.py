"""選手・選手×コース・モーターの強さを、レースのたびに Plackett–Luce の対数尤度の勾配で更新するレーティング（IDEA-017）。
- 各レースの前に、その時点のレーティングを特徴量として記録してから、結果で更新する（未来の情報は入らない）
- 艇のスコア s = θ_選手 + φ_選手×コース + β_コース + μ_モーター（場×モーター番号）
  コースは更新では実際の進入コース（競走成績）、記録する特徴量では艇番（＝枠なりの想定コース）を使う
- 勾配: 着順 o_1..o_m（完走艇）について ∂/∂s_i Σ_k [s_{o_k} − log Σ_{j≥k} exp s_{o_j}] = 1[i 完走] − Σ_{k: i ∈ R_k} p_{i,k}
  （失格・転覆などで着順のない艇は最後尾の同順扱いにせず、更新から外す）
- 時間とともに古い情報の重みが下がるよう、選手・モーターのレーティングは更新のたびに少しずつ 0 に縮める
"""
import numpy as np
import pandas as pd

LR = dict(racer=0.06, lane=0.04, course=0.002, motor=0.08)
SHRINK = dict(racer=0.0005, lane=0.0005, motor=0.003)  # 更新1回あたりの縮小率

def compute(entries):
    """entries: date, jcd, rno, boat, toban, motor_no, pos, course を持つ表（全期間、日付順でなくてよい）
    返り値: 同じ行順の DataFrame（rt_racer, rt_lane, rt_motor, rt_sum と、レース内の相対値 rt_sum_rel, rt_sum_rank）"""
    e = entries[["date", "jcd", "rno", "boat", "toban", "motor_no", "pos", "course"]].copy()
    e["_i"] = np.arange(len(e))
    e = e.sort_values(["date", "rno", "jcd", "boat"]).reset_index(drop=True)
    toban = e["toban"].astype(int).values
    motor = (e["jcd"].astype(str) + "_" + e["motor_no"].fillna(-1).astype(int).astype(str)).values
    boat = e["boat"].astype(int).values
    course = e["course"].fillna(e["boat"]).astype(int).clip(1, 6).values
    pos = e["pos"].values
    racer_id, ri = np.unique(toban, return_inverse=True)
    motor_id, mi = np.unique(motor, return_inverse=True)
    th = np.zeros(len(racer_id)); ph = np.zeros((len(racer_id), 6)); be = np.array([1.2, 0.3, 0.1, 0.0, -0.4, -0.9]); mu = np.zeros(len(motor_id))
    out = np.full((len(e), 3), np.nan)
    keys = (e["date"].astype(str) + e["jcd"].astype(str) + e["rno"].astype(str)).values
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(e)]
    for s, t in zip(starts, ends):
        idx = np.arange(s, t)
        r, m, b, c = ri[idx], mi[idx], boat[idx] - 1, course[idx] - 1
        out[idx, 0] = th[r]; out[idx, 1] = ph[r, b]; out[idx, 2] = mu[m]
        p = pos[idx]
        fin = np.isfinite(p)
        if fin.sum() < 2:
            continue
        sc = th[r] + ph[r, c] + be[c] + mu[m]
        order = np.argsort(np.where(fin, p, 99))[: fin.sum()]
        g = np.zeros(len(idx))
        rest = list(order)
        for k in range(len(order) - 1):
            ex = np.exp(sc[rest] - sc[rest].max()); pr = ex / ex.sum()
            g[rest] -= pr
            g[order[k]] += 1.0
            rest = rest[1:]
        g[~fin] = 0.0
        th[r] = th[r] * (1 - SHRINK["racer"]) + LR["racer"] * g
        ph[r, c] = ph[r, c] * (1 - SHRINK["lane"]) + LR["lane"] * g
        mu[m] = mu[m] * (1 - SHRINK["motor"]) + LR["motor"] * g
        np.add.at(be, c, LR["course"] * g)
    res = pd.DataFrame(out, columns=["rt_racer", "rt_lane", "rt_motor"])
    res["rt_sum"] = res.sum(1)
    res["_key"] = keys
    res["rt_sum_rel"] = res["rt_sum"] - res.groupby("_key")["rt_sum"].transform("mean")
    res["rt_sum_rank"] = res.groupby("_key")["rt_sum"].rank(ascending=False, method="min")
    res["_i"] = e["_i"].values
    return res.sort_values("_i").drop(columns=["_key", "_i"]).reset_index(drop=True)

RT_FEATS = ["rt_racer", "rt_lane", "rt_motor", "rt_sum", "rt_sum_rel", "rt_sum_rank"]
