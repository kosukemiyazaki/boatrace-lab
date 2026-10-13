"""IDEA-033 選んで買う：確定オッズの3連単市場が甘いレースの種類はあるか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea033.py DATADIR STOREDIR [--out results/idea033.md]
- 区分 F1〜F9（締切時刻・レース番号・グレード・レースの種類・A1の人数・1号艇の級別・荒れ・場・女子戦）
- 方法1：区分 × オッズ帯のセルを学習で選ぶ（点数5,000以上・回収率105%以上）→ 評価の回収率
- 方法2：区分ごとに π ∝ p^β の β を学習で推定 → 評価で π×オッズ > 1.1 かつ 300倍以下を買う
- 学習 2023-10〜2025-12 ／ 評価 2026-01〜08。判定：回収率110%以上・点数200以上・日単位95%CI下限 > 100%
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

KEY = ["date", "jcd", "rno"]
TR = ("20231001", "20251231"); EV = ("20260101", "20260831")
BANDS = [0, 10, 30, 100, 300, np.inf]; BAND_NAMES = ["〜10", "10〜30", "30〜100", "100〜300", "300〜"]
RNG = np.random.default_rng(0)

def norm(df):
    return df.assign(date=df["date"].astype(str), jcd=df["jcd"].astype(str).str.zfill(2), rno=df["rno"].astype(int))

def day_ci(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum()
    g = g[g["st"] > 0]
    if len(g) < 2:
        return np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(3000, len(g)))
    b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))

def race_type(s):
    s = str(s)
    if "進入固定" in s: return "進入固定"
    if "準優" in s: return "準優勝戦"
    if "優勝" in s: return "優勝戦"
    if any(k in s for k in ("特選", "選抜", "ドリーム", "特賞")): return "特選・選抜・ドリーム"
    if "予選" in s: return "予選"
    return "一般・その他"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--out", default="results/idea033.md")
    a = ap.parse_args()
    races = norm(pd.read_parquet(f"{a.datadir}/races.parquet"))[KEY + ["race_name", "wind", "wave", "tri_combo"]].dropna(subset=["tri_combo"]).drop_duplicates(KEY)
    races = races[(races["date"] >= TR[0]) & (races["date"] <= EV[1])]
    o3 = norm(pd.read_parquet(f"{a.datadir}/odds.parquet"))
    e = norm(pd.read_parquet(f"{a.datadir}/entries.parquet", columns=KEY + ["boat", "cls", "deadline"]))
    m = races.merge(o3, on=KEY)
    O = m[COMBOS].to_numpy(float); ok = (np.isfinite(O) & (O > 0)).sum(1) >= 60
    m, O = m[ok].reset_index(drop=True), O[ok]
    # 区分の材料
    cls = e.pivot_table(index=KEY, columns="boat", values="cls", aggfunc="first")
    a1n = (cls == "A1").sum(1).rename("a1n"); b1c = cls.get(1).rename("b1cls")
    dl = e.groupby(KEY)["deadline"].first().rename("deadline")
    m = m.merge(pd.concat([a1n, b1c, dl], axis=1).reset_index(), on=KEY, how="left")
    g = pd.read_csv(f"{a.store}/grades.csv", dtype={"jcd": str, "start": str, "end": str}); g["jcd"] = g["jcd"].str.zfill(2)
    m["grade"] = "一般"; m["ladies"] = "それ以外"
    for r in g.itertuples():
        sel = (m["jcd"] == r.jcd) & (m["date"] >= r.start) & (m["date"] <= r.end)
        gr = str(r.grade)
        m.loc[sel, "grade"] = "SG" if gr.startswith("SG") else "G1" if gr.startswith("G1") else "G2" if gr.startswith("G2") else "G3" if gr.startswith("G3") else "一般"
        if any(k in str(r.title) for k in ("レディース", "女子")):
            m.loc[sel, "ladies"] = "女子戦"
    hh = pd.to_numeric(m["deadline"].astype(str).str[:2], errors="coerce")
    fam = {
        "F1 締切時刻": pd.cut(hh, [-1, 11, 15, 17, 30], labels=["〜11:59", "12:00〜15:59", "16:00〜17:59", "18:00〜"]).astype(str),
        "F2 レース番号": pd.cut(m["rno"], [0, 4, 8, 12], labels=["1〜4R", "5〜8R", "9〜12R"]).astype(str),
        "F3 グレード": m["grade"],
        "F4 レースの種類": m["race_name"].map(race_type),
        "F5 A1の人数": m["a1n"].fillna(0).clip(upper=3).astype(int).map({0: "0", 1: "1", 2: "2", 3: "3以上"}),
        "F6 1号艇の級別": m["b1cls"].fillna("?").astype(str),
        "F7 荒れ": np.where((pd.to_numeric(m["wind"], errors="coerce") >= 5) | (pd.to_numeric(m["wave"], errors="coerce") >= 5), "風5m・波5cm以上", "それ以外"),
        "F8 場": m["jcd"],
        "F9 女子戦": m["ladies"],
    }
    inv = np.where(np.isfinite(O) & (O > 0), 1 / np.where(O > 0, O, 1), 0); P = inv / inv.sum(1, keepdims=True)
    L = np.log(np.clip(P, 1e-9, None))
    y = m["tri_combo"].map({c: i for i, c in enumerate(COMBOS)}).values
    okY = ~pd.isna(y); y = np.where(okY, y, 0).astype(int)
    hitO = np.where(okY, O[np.arange(len(O)), y], 0.0)  # 当たり組番の確定オッズ
    valid = np.isfinite(O) & (O > 0)
    band = np.digitize(np.where(valid, O, np.nan), BANDS[1:-1])  # 0..4
    tr = (m["date"] <= TR[1]).values; ev = (m["date"] >= EV[0]).values
    days = m["date"].values
    # 方法1 の材料：レースごと・帯ごとの点数と払戻
    nb = np.stack([(valid & (band == k)).sum(1) for k in range(5)], 1).astype(float)
    hb = np.stack([np.where(okY & (band[np.arange(len(O)), y] == k), hitO * 100, 0.0) for k in range(5)], 1)
    rows1, rows2, cells_out = [], [], []
    for fname, lab in fam.items():
        lab = np.asarray(lab)
        # 方法1
        sel_mask = np.zeros((len(m), 5), bool); n_pick = 0
        for v in sorted(set(lab)):
            iv = lab == v
            for k in range(5):
                n_tr = nb[iv & tr, k].sum(); r_tr = hb[iv & tr, k].sum() / max(100 * n_tr, 1)
                n_ev = nb[iv & ev, k].sum(); r_ev = hb[iv & ev, k].sum() / max(100 * n_ev, 1)
                pick = n_tr >= 5000 and r_tr >= 1.05
                if pick:
                    sel_mask[iv, k] = True; n_pick += 1
                cells_out.append({"区分": fname, "値": v, "オッズ帯": BAND_NAMES[k], "学習点数": int(n_tr), "学習回収率": r_tr, "評価点数": int(n_ev), "評価回収率": r_ev, "選択": "○" if pick else ""})
        st = (nb * sel_mask)[ev].sum(1) * 100; rt = (hb * sel_mask)[ev].sum(1)
        n = int(st.sum() / 100); roi = rt.sum() / st.sum() if st.sum() else np.nan
        lo, hi = day_ci(days[ev], st, rt) if n else (np.nan, np.nan)
        rows1.append({"区分": fname, "選んだセル": n_pick,
                      "評価の点数": n, "評価の回収率": roi, "95%CI下": lo, "95%CI上": hi,
                      "判定": ("採用候補" if (n >= 200 and roi >= 1.1 and lo > 1.0) else "不採用") if n else "選んだセルなし"})
        # 方法2
        st2 = np.zeros(len(m)); rt2 = np.zeros(len(m)); betas = []
        for v in sorted(set(lab)):
            itr = np.where((lab == v) & tr & okY)[0]; iev = np.where((lab == v) & ev)[0]
            if len(itr) < 300 or len(iev) == 0:
                continue
            Lt = L[itr]; yt = y[itr]
            def nll(b):
                s = b * Lt; s = s - s.max(1, keepdims=True); lz = np.log(np.exp(s).sum(1)); return -(s[np.arange(len(yt)), yt] - lz).mean()
            b = minimize_scalar(nll, bounds=(0.5, 1.5), method="bounded").x; betas.append(f"{v}:{b:.3f}")
            s = b * L[iev]; s = s - s.max(1, keepdims=True); pi = np.exp(s); pi /= pi.sum(1, keepdims=True)
            buy = valid[iev] & (O[iev] <= 300) & (pi * np.nan_to_num(O[iev]) > 1.1)
            st2[iev] = buy.sum(1) * 100.0
            rt2[iev] = np.where(okY[iev] & buy[np.arange(len(iev)), y[iev]], hitO[iev] * 100, 0.0)
        n2 = int(st2[ev].sum() / 100); roi2 = rt2[ev].sum() / st2[ev].sum() if n2 else np.nan
        lo2, hi2 = day_ci(days[ev], st2[ev], rt2[ev]) if n2 else (np.nan, np.nan)
        rows2.append({"区分": fname, "β（学習）": " ".join(betas), "評価の点数": n2, "評価の回収率": roi2, "95%CI下": lo2, "95%CI上": hi2,
                      "判定": ("採用候補" if (n2 >= 200 and roi2 >= 1.1 and lo2 > 1.0) else "不採用") if n2 else "買い目なし"})
    C = pd.DataFrame(cells_out)
    corr = C.dropna().pipe(lambda c: c[(c["学習点数"] >= 5000) & (c["評価点数"] >= 2000)])
    md = [f"# IDEA-033 選んで買う：3連単の確定オッズの市場が甘いレースの種類（学習 {TR[0]}〜{TR[1]} {int(tr.sum())}レース／評価 {EV[0]}〜{EV[1]} {int(ev.sum())}レース）", "",
          "## 方法1：区分 × オッズ帯（学習で 点数5,000以上・回収率105%以上 のセルを選び、評価で全部買う）", "", pd.DataFrame(rows1).to_markdown(index=False, floatfmt=".3f"), "",
          f"参考：セルの回収率の学習↔評価の相関（学習5,000点・評価2,000点以上の {len(corr)} セル）: {np.corrcoef(corr['学習回収率'], corr['評価回収率'])[0, 1]:.3f}", "",
          "## 方法2：区分ごとに本命・穴の補正（π ∝ p^β、評価で π×オッズ > 1.1 かつ 300倍以下）", "", pd.DataFrame(rows2).to_markdown(index=False, floatfmt=".3f"), "",
          "## 参考：学習の回収率が高かったセル（上位25、学習5,000点以上）", "",
          C[C["学習点数"] >= 5000].sort_values("学習回収率", ascending=False).head(25).to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
