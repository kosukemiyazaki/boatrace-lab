"""着順予測: LightGBM で各艇の1着スコア -> レース内 softmax で1着確率。
3連単確率は Harville 式を「2着・3着はべき乗で平らにした確率」で計算する（γ2, γ3 を検証期間で最尤推定）。
市場ブレンド: π ∝ p_model^a · p_market^b （a, b を検証期間で最尤推定）。p_market は締切オッズの逆数を正規化。
"""
import itertools
import numpy as np
from scipy.optimize import minimize

PERMS = list(itertools.permutations(range(6), 3))  # 0-index, COMBOS と同じ辞書順
PI = np.array([p[0] for p in PERMS]); PJ = np.array([p[1] for p in PERMS]); PK = np.array([p[2] for p in PERMS])

def softmax_rows(s, mask):
    s = np.where(mask, s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

def trifecta(p, g2=1.0, g3=1.0):
    """p: (n,6) 1着確率（欠場は0）-> (n,120)"""
    def norm_pow(g):
        q = np.where(p > 0, p, 0) ** g
        return q / q.sum(axis=1, keepdims=True)
    q2, q3 = norm_pow(g2), norm_pow(g3)
    a = p[:, PI]
    b = q2[:, PJ] / np.clip(1 - q2[:, PI], 1e-9, None)
    c = q3[:, PK] / np.clip(1 - q3[:, PI] - q3[:, PJ], 1e-9, None)
    t = a * b * c
    return t / t.sum(axis=1, keepdims=True)

def fit_gamma(p, y):
    """y: 勝ち組番の列index (n,)"""
    def nll(x):
        t = trifecta(p, *x)
        return -np.log(np.clip(t[np.arange(len(y)), y], 1e-12, None)).mean()
    r = minimize(nll, [0.8, 0.7], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6})
    return tuple(r.x), r.fun

def blend(pm, pk, a, b):
    lg = a * np.log(np.clip(pm, 1e-12, None)) + b * np.log(np.clip(pk, 1e-12, None))
    lg = lg - lg.max(axis=1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(axis=1, keepdims=True)

def fit_blend(pm, pk, y):
    def nll(x):
        t = blend(pm, pk, *x)
        return -np.log(np.clip(t[np.arange(len(y)), y], 1e-12, None)).mean()
    r = minimize(nll, [0.5, 0.5], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6})
    return tuple(r.x), r.fun
