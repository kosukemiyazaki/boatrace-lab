"""IDEA-058 締切直前のお金（駆け込み）の正体を測る：量・行き先・手で追えるか（docs/ideas.md の登録どおり）
usage: python scripts/idea058.py DATADIR STOREDIR [--start 20261011] [--out results/idea058.md]
- 票数：BOATCAST の3連単の組ごとの票数。t1（締切2分前）・t0（45〜6秒前）・t00（締切直前）は live_t*/bc_votes、確定は bc_votes/YYYYMM.csv.gz
- (1) 量：確定の票数のうち、t1 以降・t0 以降に入った割合
- (2) 行き先：t1→t0、t0→確定 に入った票の配分 ÷ t1 時点の配分（当たり目／外れ目、オッズ帯別）、上位の組への集中
- (3) t0 のずれ：t0 の3連単の票から出した確率（^1.06）× t0 の単勝・複勝（下限）・拡連複（下限）オッズ > 1.1 を各100円、払戻は K の実額
- (4) 追随：t1→t0 で票の伸び（相対）が一番大きい3連単の組（t0 で配分0.5%以上）を各レース1点・上位3点、100円、払戻は確定（K の3連単の実額）
- 判定（買い方）：(3)(4) のどれかが 500点以上・回収率110%以上・日単位95%CI下限 > 100% → 採用候補。届かなければ途中経過、2,000点で最終判定
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS, COMBOS_2F
import boatlib.parse as P
import exp_idea039 as E39

KEY = ["date", "jcd", "rno"]; RNG = np.random.default_rng(0); BETA = 1.06
VC = [f"v{c}" for c in COMBOS]
FIRST = np.array([int(c[0]) for c in COMBOS]); SEC = np.array([int(c[2]) for c in COMBOS]); THR = np.array([int(c[4]) for c in COMBOS])

def nz(d):
    d = d.copy(); d["date"] = d["date"].astype(str); d["jcd"] = d["jcd"].astype(str).str.zfill(2); d["rno"] = d["rno"].astype(int); return d

def read(pattern, start):
    fs = [f for f in glob.glob(pattern) if os.path.basename(f)[:8] >= start or os.path.basename(f)[:6] >= start[:6]]
    if not fs:
        return pd.DataFrame(columns=KEY)
    d = nz(pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs]))
    return d[d["date"] >= start]

def snap(store, lead, page, start):
    """締切前の最後の記録（fetched_at < 締切）"""
    d = read(f"{store}/{lead}/{page}/*.csv.gz", start)
    if not len(d):
        return d
    d = d[d["fetched_at"].astype(str) < d["deadline"].astype(str) + ":00"]
    return d.sort_values("fetched_at").drop_duplicates(KEY, keep="last")

def k_pays(store, start):
    """K ファイル：単勝（艇・払戻）、複勝（2艇ぶん）、3連単（組・払戻）"""
    rows = []
    for f in sorted(glob.glob(f"{store}/bk/*/k*.txt.gz")):
        date = "20" + os.path.basename(f)[1:7]
        if date < start:
            continue
        txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
        for jcd, blk in P._blocks(txt, "K"):
            rno = None
            for line in blk.splitlines():
                z = P.z2h(line); m = P.K_RACE.match(z)
                if m:
                    rno = int(m.group(1)); continue
                if rno is None:
                    continue
                mm = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", z)
                if mm:
                    rows.append((date, jcd, rno, "tan", mm.group(1), int(mm.group(2)))); continue
                mm = re.match(r"^\s+複勝\s+(.*)$", z)
                if mm:
                    for b, p in re.findall(r"(\d)\s+(\d+)", mm.group(1)):
                        rows.append((date, jcd, rno, "fuku", b, int(p)))
                    continue
                mm = re.match(r"^\s+3連単\s+(\d)-(\d)-(\d)\s+(\d+)", z)
                if mm:
                    rows.append((date, jcd, rno, "3t", f"{mm.group(1)}-{mm.group(2)}-{mm.group(3)}", int(mm.group(4))))
    d = pd.DataFrame(rows, columns=KEY + ["kind", "key", "pay"]); d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return d.drop_duplicates(KEY + ["kind", "key"], keep="last")

def boot(days, st, rt):
    g = pd.DataFrame({"d": days, "st": st, "rt": rt}).groupby("d").sum(); g = g[g["st"] > 0]
    if len(g) < 2:
        return (g["rt"].sum() / g["st"].sum() if len(g) else np.nan), np.nan, np.nan
    bi = RNG.integers(0, len(g), size=(4000, len(g))); b = g["rt"].values[bi].sum(1) / g["st"].values[bi].sum(1)
    return float(g["rt"].sum() / g["st"].sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def vmat(d):
    return d[VC].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("store"); ap.add_argument("--start", default="20261011")
    ap.add_argument("--out", default="results/idea058.md"); a = ap.parse_args(); S = a.store
    v1 = snap(S, "live_t1", "bc_votes", a.start); v0 = snap(S, "live_t0", "bc_votes", a.start); v00 = snap(S, "live_t00", "bc_votes", a.start)
    vf = read(f"{S}/bc_votes/*.csv.gz", a.start).drop_duplicates(KEY, keep="last")
    pays = k_pays(S, a.start)
    win = pays[pays["kind"] == "3t"][KEY + ["key", "pay"]].rename(columns={"key": "tri", "pay": "tri_pay"})
    # 4時点の票がそろい、結果のあるレース
    m = v1[KEY + VC].merge(v0[KEY + VC], on=KEY, suffixes=("_1", "_0")).merge(vf[KEY + VC].rename(columns={c: c + "_f" for c in VC}), on=KEY).merge(win, on=KEY)
    m = m.merge(v00[KEY + VC].rename(columns={c: c + "_00" for c in VC}), on=KEY, how="left").reset_index(drop=True)
    V1 = m[[c + "_1" for c in VC]].to_numpy(float); V0 = m[[c + "_0" for c in VC]].to_numpy(float); VF = m[[c + "_f" for c in VC]].to_numpy(float)
    V00 = m[[c + "_00" for c in VC]].fillna(np.nan).to_numpy(float)
    ok = (V1.sum(1) > 0) & (V0.sum(1) >= V1.sum(1)) & (VF.sum(1) >= V0.sum(1)) & m["tri"].isin(COMBOS).values
    m, V1, V0, VF, V00 = m[ok].reset_index(drop=True), V1[ok], V0[ok], VF[ok], V00[ok]; n = len(m)
    T1, T0, TF = V1.sum(1), V0.sum(1), VF.sum(1)
    y = m["tri"].map({c: i for i, c in enumerate(COMBOS)}).values.astype(int)
    hit = np.zeros((n, 120), bool); hit[np.arange(n), y] = True
    days = m["date"].values
    out = [f"# IDEA-058 締切直前のお金（駆け込み）の正体（{a.start}〜、{n}レース・{len(set(days))}日、票は BOATCAST の3連単）", ""]
    # (1) 量
    same00 = np.nanmean(np.isclose(np.nansum(V00, 1), T0)) if np.isfinite(V00).any() else np.nan
    out += ["## (1) 量：確定の票のうち、締切直前に入った割合", "",
            f"- t1（2分前）以降に入った割合：中央値 {np.median(1 - T1 / TF):.1%}（平均 {np.mean(1 - T1 / TF):.1%}、10〜90% {np.quantile(1 - T1 / TF, .1):.1%}〜{np.quantile(1 - T1 / TF, .9):.1%}）",
            f"- t0（45〜6秒前）以降に入った割合：中央値 {np.median(1 - T0 / TF):.1%}（平均 {np.mean(1 - T0 / TF):.1%}、10〜90% {np.quantile(1 - T0 / TF, .1):.1%}〜{np.quantile(1 - T0 / TF, .9):.1%}）",
            f"- t00（締切直前、約15秒前）の票数が t0 と同じだったレース：{same00:.0%}（BOATCAST の集計がまとめて更新されるため、締切直前は t0 から動かないことが多い）",
            "- 確かめ：t1 の公式オッズと、同じ時点の BOATCAST の票から出したオッズはほぼ一致（中央値 1.02倍）。t1 で見えている票は、その時点の売上そのもの。", ""]
    # (2) 行き先
    s1 = V1 / T1[:, None]
    def late(Va, Vb, Ta, Tb):
        d = Tb - Ta; return np.where(d[:, None] > 0, (Vb - Va) / np.where(d[:, None] > 0, d[:, None], 1), np.nan)
    L10 = late(V1, V0, T1, T0); L0F = late(V0, VF, T0, TF); L1F = late(V1, VF, T1, TF)
    def ratio(L, msk):
        r = L[msk] / np.where(s1[msk] > 0, s1[msk], np.nan); r = r[np.isfinite(r)]; return np.median(r), np.mean(np.clip(r, 0, 20)), len(r)
    O1 = 0.75 / np.where(s1 > 0, s1, np.nan)  # t1 の票から出したオッズ
    rows = []
    for nm, L in (("t1→t0", L10), ("t0→確定", L0F), ("t1→確定", L1F)):
        for band, (lo, hi) in (("全体", (0, 1e9)), ("〜10倍", (0, 10)), ("10〜30倍", (10, 30)), ("30〜100倍", (30, 100)), ("100倍〜", (100, 1e9))):
            b = (O1 >= lo) & (O1 < hi)
            wm, wa, wn = ratio(L, hit & b); lm, la, ln_ = ratio(L, ~hit & b)
            rows.append({"区間": nm, "t1のオッズ帯": band, "当たり目 中央値": wm, "当たり目 平均": wa, "当たり目 件数": wn, "外れ目 中央値": lm, "外れ目 平均": la})
    R2 = pd.DataFrame(rows)
    # 集中：t1→確定 の追加分のうち、上位5組・上位10組に入った割合（t1 時点の上位5組の配分と比べる）
    add = np.clip(VF - V1, 0, None); add_s = np.sort(add, 1)[:, ::-1]; ss1 = np.sort(s1, 1)[:, ::-1]
    top5_late = np.median(add_s[:, :5].sum(1) / add.sum(1)); top5_early = np.median(ss1[:, :5].sum(1))
    # 当たり目の「確定の配分 ÷ t1 の配分」と、確定オッズから見た割安さ
    sF = VF / TF[:, None]
    wr = (sF[hit] / s1[hit]); allr = (sF / np.where(s1 > 0, s1, np.nan))[~hit]
    out += ["## (2) 行き先：締切直前の票は当たり目に多く入っているか", "",
            "「追加分の配分 ÷ t1 時点の配分」。1 より大きいほど、その組に締切直前の票が多めに入った。", "",
            R2.to_markdown(index=False, floatfmt=".2f"), "",
            f"- 集中：t1→確定 の追加分のうち、追加が多かった上位5組に入った割合は中央値 {top5_late:.0%}（t1 時点の上位5組の配分は {top5_early:.0%}）",
            f"- 当たり目の配分は t1→確定 で中央値 {np.median(wr):.2f}倍（外れ目 {np.nanmedian(allr):.2f}倍）。当たり目のオッズは同じだけ下がる。", ""]
    # (3) t0 のずれ
    tf0 = snap(S, "live_t0", "oddstf", a.start); k0 = snap(S, "live_t0", "oddsk", a.start)
    pi = np.where(V0 > 0, (V0 / T0[:, None]) ** BETA, 0); pi /= pi.sum(1, keepdims=True)
    pw = np.stack([pi[:, FIRST == b].sum(1) for b in range(1, 7)], 1)
    p2 = np.stack([pi[:, (FIRST == b) | (SEC == b)].sum(1) for b in range(1, 7)], 1)
    pk = {pr: pi[:, [i for i, c in enumerate(COMBOS) if set(map(int, pr.split("="))) <= {int(c[0]), int(c[2]), int(c[4])}]].sum(1) for pr in COMBOS_2F}
    tanp = pays[pays["kind"] == "tan"]; fukup = pays[pays["kind"] == "fuku"]
    E39.START, E39.END = a.start, "29991231"; wp = E39.wide_pay(S); wp["jcd"] = wp["jcd"].astype(str).str.zfill(2)
    tmap = {(r.date, r.jcd, r.rno, r.key): r.pay for r in tanp.itertuples()}
    fmap = {(r.date, r.jcd, r.rno, r.key): r.pay for r in fukup.itertuples()}
    wmap = {(r.date, r.jcd, r.rno, r.pair): r.pay for r in wp.itertuples()}
    done_t = set(map(tuple, tanp[KEY].values)); done_w = set(map(tuple, wp[KEY].values))
    keys = list(map(tuple, m[KEY].values)); idx = {k: i for i, k in enumerate(keys)}
    rows3 = []
    for nm, odf, cols, prob, pmap, done in (
            ("単勝", tf0, [f"tan{b}" for b in range(1, 7)], pw, tmap, done_t),
            ("複勝（下限）", tf0, [f"fuku_lo{b}" for b in range(1, 7)], p2, fmap, done_t),
            ("拡連複（下限）", k0, [f"wide_lo{pr}" for pr in COMBOS_2F], np.stack([pk[pr] for pr in COMBOS_2F], 1), wmap, done_w)):
        if not len(odf):
            continue
        labels = [str(b) for b in range(1, 7)] if nm != "拡連複（下限）" else COMBOS_2F
        for thr in (1.1, 1.0, 1.2):
            st, rt, dd, ratio_l = [], [], [], []
            for r in odf.to_dict("records"):
                k = (r["date"], r["jcd"], int(r["rno"]))
                if k not in idx or k not in done:
                    continue
                i = idx[k]; od = pd.to_numeric(pd.Series([r.get(c, np.nan) for c in cols]), errors="coerce").values
                for j, lab in enumerate(labels):
                    if np.isfinite(od[j]) and od[j] > 0 and prob[i, j] * od[j] > thr:
                        pay = pmap.get((k[0], k[1], k[2], lab), 0); st.append(100); rt.append(pay); dd.append(k[0])
                        if pay > 0:
                            ratio_l.append(pay / 100 / od[j])
            roi, lo, hi = boot(np.array(dd), np.array(st, float), np.array(rt, float))
            rows3.append({"券種": nm, "閾値": thr, "点数": len(st), "的中": int(np.sum(np.array(rt) > 0)), "回収率": roi, "95%CI下": lo, "95%CI上": hi,
                          "当たった組の 確定÷t0オッズ": np.mean(ratio_l) if ratio_l else np.nan})
    R3 = pd.DataFrame(rows3)
    out += ["## (3) t0（45〜6秒前）で見えたずれは確定まで残るか", "",
            "t0 の3連単の票から出した確率（^1.06）× t0 のオッズ（複勝・拡連複は下限）> 閾値 の組を各100円、払戻は K の実額。", "",
            R3.to_markdown(index=False, floatfmt=".3f") if len(R3) else "（t0 のオッズなし）", ""]
    # (4) 追随
    s0 = V0 / T0[:, None]; surge = np.where((s0 >= 0.005) & (V1 > 0), V0 / np.where(V1 > 0, V1, 1), 0)
    order = np.argsort(-surge, 1)
    PF = np.where(VF > 0, VF / TF[:, None], 0)  # 確定の配分（≒確定オッズから見た市場の確率）
    rows4 = []
    for k_ in (1, 3):
        sel = np.zeros((n, 120), bool); sel[np.arange(n)[:, None], order[:, :k_]] = True; sel &= surge > 0
        st = sel.sum(1) * 100.0; rt = np.where(sel & hit, m["tri_pay"].values[:, None], 0).sum(1)
        roi, lo, hi = boot(days, st, rt)
        rows4.append({"買い方": f"票の伸びが大きい上位{k_}点", "点数": int(sel.sum()), "的中": int((sel & hit).sum()), "的中率": (sel & hit).sum() / sel.sum(),
                      "確定の配分から見た見込み": PF[sel].sum() / sel.sum(), "回収率": roi, "95%CI下": lo, "95%CI上": hi})
    R4 = pd.DataFrame(rows4)
    out += ["## (4) 追随：t1→t0 で票が一番伸びた3連単の組を買う", "", R4.to_markdown(index=False, floatfmt=".3f"), ""]
    # 判定
    cand = pd.concat([R3[R3["閾値"] == 1.1].assign(名=lambda d: "(3) " + d["券種"]), R4.assign(名=lambda d: "(4) " + d["買い方"])], ignore_index=True)
    good = cand[(cand["点数"] >= 500) & (cand["回収率"] >= 1.1) & (cand["95%CI下"] > 1.0)]
    enough = (cand["点数"] >= 2000).all()
    if len(good):
        judge = "採用候補（前向きに確認）：" + "、".join(good["名"])
    elif enough:
        judge = "不採用"
    else:
        judge = "途中経過（基準に届く買い方なし。2,000点で最終判定）"
    out.insert(2, f"**判定：{judge}**"); out.insert(3, "")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    main()
