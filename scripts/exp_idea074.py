"""IDEA-074 ライブの「期待値で組を外す」ルールは正しいか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea074.py DATADIR STOREDIR [--c35 models/idea035_c35.npz] [--out results/idea074.md]
- 計算はライブで使っている scratchpad の pred_race.py と同じ（3連単オッズ → 市場の確率、β=1.055、IDEA-035 の c2/c3）。F持ちの補正はなし
- 各レースで補正後の確率の上位15組を候補。期待値 = 補正後の確率 × そのときのオッズ
- (A) ライブ前向き 2026-10-11〜：締切6分前（live）が主、15分前（live_t15）が副。払戻は K の実額（races.parquet の tri_combo / tri_pay）
- (B) 参考：確定オッズ（2023-10〜2026-08）。締切直前のお金が入った後のオッズなので判定には使わない
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS
from exp_idea036 import norm
from exp_idea054 import KEY, boot_roi, ratio_ci

I = np.array([int(c[0]) for c in COMBOS]) - 1; J = np.array([int(c[2]) for c in COMBOS]) - 1
PAIR = I * 6 + J
EV_BANDS = [(0, .72, "〜0.72"), (.72, .76, "0.72〜0.76"), (.76, .80, "0.76〜0.80"), (.80, .84, "0.80〜0.84"), (.84, 9, "0.84〜")]
P_BANDS = [(0, .03, "〜3%"), (.03, .05, "3〜5%"), (.05, .08, "5〜8%"), (.08, 1, "8%〜")]

def model_pi(O, cz):
    """pred_race.py と同じ計算をレースごと（行）にまとめて行う。O: (R,120) のオッズ"""
    v = np.isfinite(O) & (O > 0)
    P = np.where(v, 1 / np.where(v, O, 1), 0); P = P / P.sum(1, keepdims=True)
    Pb = P ** 1.055; Pb = Pb / Pb.sum(1, keepdims=True)
    R = len(O); P1 = np.zeros((R, 6)); P12 = np.zeros((R, 36))
    for c in range(120):
        P1[:, I[c]] += Pb[:, c]; P12[:, PAIR[c]] += Pb[:, c]
    P12 = P12.reshape(R, 6, 6)
    q2 = P12 / np.where(P1[:, :, None] > 0, P1[:, :, None], 1)
    band = np.digitize(P1[:, 0], [0.4, 0.6])
    q2a = q2 * cz["c2"].transpose(2, 0, 1)[band]
    q2a[:, np.arange(6), np.arange(6)] = 0
    s = q2a.sum(2, keepdims=True); q2a = q2a / np.where(s > 0, s, 1)
    p12c = P12[:, I, J]
    q3 = Pb / np.where(p12c > 0, p12c, 1) * cz["c3"][None, :] * (Pb > 0)
    den = np.zeros((R, 36))
    for c in range(120):
        den[:, PAIR[c]] += q3[:, c]
    q3 = q3 / np.where(den[:, PAIR] > 0, den[:, PAIR], 1)
    PI = P1[:, I] * q2a[:, I, J] * q3; PI = PI / PI.sum(1, keepdims=True)
    return PI

def candidates(od, races, cz, top=15):
    """od: KEY + COMBOS のオッズ表。候補（上位15組）を縦持ちで返す"""
    od = od.merge(races, on=KEY, how="inner")
    O = od[COMBOS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = (np.isfinite(O) & (O > 0)).sum(1) >= 100
    od, O = od[ok].reset_index(drop=True), O[ok]
    PI = model_pi(O, cz)
    idx = np.argsort(-PI, 1)[:, :top]
    r = np.repeat(np.arange(len(od)), top); c = idx.ravel()
    out = od.loc[r, KEY + ["tri_combo", "tri_pay"]].reset_index(drop=True)
    out["combo"] = np.array(COMBOS)[c]; out["pi"] = PI[r, c]; out["odds"] = O[r, c]
    out["ev"] = out["pi"] * out["odds"]
    out["hit"] = (out["combo"] == out["tri_combo"]).astype(float)
    out["ret"] = np.where(out["hit"] > 0, out["tri_pay"], 0.0)
    return out

def table(c, col, bands, title):
    rows = [f"#### {title}", "", "| 帯 | 件数 | 的中 | 見込みの的中率 | 実際の的中率 | 実際÷見込み（95%CI） | 回収率（95%CI） |", "|---|---|---|---|---|---|---|"]
    for lo, hi, lab in bands:
        s = c[(c[col] >= lo) & (c[col] < hi)]
        if len(s) == 0:
            continue
        r, rl, rh = ratio_ci(s["date"], s["hit"], s["pi"])
        m, ml, mh = boot_roi(s["date"], np.full(len(s), 100.0), s["ret"])
        rows.append(f"| {lab} | {len(s):,} | {int(s['hit'].sum())} | {100*s['pi'].mean():.2f}% | {100*s['hit'].mean():.2f}% | "
                    f"{r:.2f}（{rl:.2f}〜{rh:.2f}） | {100*m:.1f}%（{100*ml:.0f}〜{100*mh:.0f}%） |")
    return rows + [""]

def groups(c, title):
    """一番の比較と、レース単位の残す組 vs 外す高確率の組"""
    defs = [("確率5%以上・期待値0.78未満（外す側）", (c.pi >= .05) & (c.ev < .78)),
            ("確率5%以上・期待値0.82以上（残す側）", (c.pi >= .05) & (c.ev >= .82)),
            ("確率3%以上・期待値0.78未満（外す側）", (c.pi >= .03) & (c.ev < .78)),
            ("候補のうち期待値0.78以上（プランで残す組）", c.ev >= .78),
            ("候補のうち期待値0.78未満（プランで外す組）", c.ev < .78),
            ("候補15組すべて（確率だけで選ぶ）", c.ev > -1)]
    rows = [f"#### {title}", "", "| 区分 | 件数 | 的中 | 見込みの的中率 | 実際の的中率 | 実際÷見込み（95%CI） | 回収率（95%CI） |", "|---|---|---|---|---|---|---|"]
    res = {}
    for lab, m in defs:
        s = c[m]
        r, rl, rh = ratio_ci(s["date"], s["hit"], s["pi"])
        roi = boot_roi(s["date"], np.full(len(s), 100.0), s["ret"]); res[lab] = roi
        rows.append(f"| {lab} | {len(s):,} | {int(s['hit'].sum())} | {100*s['pi'].mean():.2f}% | {100*s['hit'].mean():.2f}% | "
                    f"{r:.2f}（{rl:.2f}〜{rh:.2f}） | {100*roi[0]:.1f}%（{100*roi[1]:.0f}〜{100*roi[2]:.0f}%） |")
    # 差の CI（日単位、同じ日を抜き出して比べる）
    a = c[(c.pi >= .05) & (c.ev >= .82)]; b = c[(c.pi >= .05) & (c.ev < .78)]
    ga = a.groupby("date").agg(n=("ret", "size"), r=("ret", "sum")); gb = b.groupby("date").agg(n=("ret", "size"), r=("ret", "sum"))
    g = ga.join(gb, lsuffix="a", rsuffix="b", how="outer").fillna(0)
    rng = np.random.default_rng(74); bi = rng.integers(0, len(g), size=(2000, len(g)))
    d = (g["ra"].values[bi].sum(1) / np.maximum(g["na"].values[bi].sum(1), 1) - g["rb"].values[bi].sum(1) / np.maximum(g["nb"].values[bi].sum(1), 1)) / 100
    pt = g["ra"].sum() / max(g["na"].sum(), 1) / 100 - g["rb"].sum() / max(g["nb"].sum(), 1) / 100
    rows += ["", f"- 回収率の差（確率5%以上で 期待値0.82以上 − 期待値0.78未満）：{100*pt:+.1f}ポイント（95%CI {100*np.quantile(d,.025):+.0f}〜{100*np.quantile(d,.975):+.0f}）", ""]
    return rows, res, (pt, np.quantile(d, .025), np.quantile(d, .975))

def load_live(store, lead, start, races):
    fs = sorted(glob.glob(f"{store}/{lead}/odds3t/*.csv.gz"))
    fs = [f for f in fs if os.path.basename(f)[:8] >= start]
    d = pd.concat([pd.read_csv(f, dtype={"jcd": str, "date": str}) for f in fs], ignore_index=True)
    d = norm(d).sort_values("fetched_at").drop_duplicates(KEY, keep="last")
    return d[KEY + COMBOS]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store")
    ap.add_argument("--c35", default="models/idea035_c35.npz"); ap.add_argument("--start", default="20261011")
    ap.add_argument("--out", default="results/idea074.md"); a = ap.parse_args()
    cz = np.load(a.c35)
    races = norm(pd.read_parquet(f"{a.datadir}/races.parquet", columns=KEY + ["tri_combo", "tri_pay"]))
    races = races[races["tri_pay"].notna() & (races["tri_pay"] > 0)]
    out = ["# IDEA-074 ライブの「期待値で組を外す」ルールは正しいか", "",
           "- 計算はライブの pred_race.py と同じ（β=1.055、IDEA-035 の c2/c3、F持ちの補正なし）。各レースで補正後の確率の上位15組を候補とした。",
           "- 回収率は100円ずつ買ったときの K の実額。CI は日単位のブートストラップ（2,000回）。", ""]
    summ = {}
    for lead, lab in [("live", "(A) ライブ前向き・締切6分前（主）"), ("live_t15", "(A') ライブ前向き・締切15分前（副）")]:
        od = load_live(a.store, lead, a.start, races)
        c = candidates(od, races, cz)
        n_r = c[KEY].drop_duplicates().shape[0]; days = c["date"].nunique()
        out += [f"## {lab}：{n_r:,}レース・{days}日（{c['date'].min()}〜{c['date'].max()}）", ""]
        out += table(c, "ev", EV_BANDS, "期待値の帯ごと")
        out += table(c, "pi", P_BANDS, "確率の帯ごと")
        rows, res, diff = groups(c, "一番の比較と、プランで残す組・外す組")
        out += rows; summ[lead] = (res, diff, n_r)
    od = pd.read_parquet(f"{a.datadir}/odds.parquet"); od = norm(od)
    od = od[(od["date"] >= "20231001") & (od["date"] <= "20260831")][KEY + COMBOS]
    c = candidates(od, races, cz)
    out += [f"## (B) 参考：確定オッズ 2023-10〜2026-08（{c[KEY].drop_duplicates().shape[0]:,}レース。締切直前のお金が入った後のオッズなので判定に使わない）", ""]
    out += table(c, "ev", EV_BANDS, "期待値の帯ごと")
    rows, res, diff = groups(c, "一番の比較と、プランで残す組・外す組"); out += rows; summ["close"] = (res, diff, 0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))

if __name__ == "__main__":
    main()
