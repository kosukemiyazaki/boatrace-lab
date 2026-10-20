"""IDEA-068 潮位：海水・汽水の場で、満ち引きが1マークの攻防を変えるか（docs/ideas.md の登録どおり）
usage: python scripts/exp_idea068.py DATADIR HISTDS STORE TIDEDIR [--out results/idea068.md]
- 潮位：気象庁の潮位表（天文潮の推算値、毎時、cm）。TIDEDIR/YYYY/STN.txt（collect-tide ワークフローで data-hist ブランチに取り込んだもの）。
  1行 = 1日：1〜72桁目が0〜23時の毎時潮位（3桁ずつ）、73〜78桁目が年月日（YYMMDD、空白詰め）、79〜80桁目が地点記号。
- 場と観測地点の対応（最寄りの地点）は STN を参照。大村は大村湾の中に観測地点がないため佐世保で代用（潮の時刻・幅がずれる点に注意）。
- 締切時刻の潮位（前後の毎時を直線で補間）、潮の向き（締切の1時間後 − 1時間前 > 0 なら上げ潮）、その日の干満の幅の中での高さ（3分位：低い／中くらい／高い）。
- 区分：場 × 潮の向き × 高さ × 進入コース（1／2〜3／4〜6）、あわせて場をまとめた「潮の向き × 高さ × 進入コース」。1着率 ÷ 市場の見込み（確定3連単、2023-10 以降）
- 探索 2023-11〜2025-12 で選び、評価 2026-01〜08 で判定（IDEA-063 と同じ基準）。12年の参考はオッズなし（同じ年・同じコース群の全出走との比）
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp_idea036 import norm
from exp_idea054 import KEY, EXP, EVL
from exp_idea065 import market_boats, attach, evaluate, bgroup

# 場（jcd）→ 気象庁の潮位の観測地点（記号, 名前）
STN = {"03": ("TK", "東京"), "04": ("TK", "東京"), "06": ("MI", "舞阪"), "07": ("G4", "三河"), "08": ("ZD", "鬼崎"), "09": ("G3", "四日市港"),
       "14": ("KM", "小松島"), "15": ("TX", "多度津"), "16": ("UN", "宇野"), "17": ("Q8", "広島"), "18": ("QA", "徳山"), "19": ("CF", "長府"),
       "20": ("N0", "砂津"), "22": ("QF", "博多"), "24": ("QD", "佐世保")}
VENUE = {"03": "江戸川", "04": "平和島", "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "14": "鳴門", "15": "丸亀", "16": "児島",
         "17": "宮島", "18": "徳山", "19": "下関", "20": "若松", "22": "福岡", "24": "大村"}

def load_tide(tidedir):
    """{(地点, YYYYMMDD): 24時間ぶんの潮位(cm)}"""
    out = {}
    for f in glob.glob(f"{tidedir}/*/*.txt"):
        for line in open(f, encoding="ascii", errors="replace"):
            line = line.rstrip("\n")
            if len(line) < 80:
                continue
            try:
                h = [int(line[i * 3:(i + 1) * 3]) for i in range(24)]
                yy, mm, dd = int(line[72:74]), int(line[74:76]), int(line[76:78])
            except ValueError:
                continue
            out[(line[78:80], f"{2000 + yy:04d}{mm:02d}{dd:02d}")] = np.array(h, float)
    return out

def tide_features(r, T):
    """レースごとの 潮位・上げ潮/下げ潮・その日の高さ（3分位）"""
    lev = np.full(len(r), np.nan); dirn = np.full(len(r), np.nan); pos = np.full(len(r), np.nan)
    for i, (jcd, date, dl) in enumerate(zip(r["jcd"].values, r["date"].values, r["deadline"].values)):
        if jcd not in STN or not isinstance(dl, str) or ":" not in dl:
            continue
        a = T.get((STN[jcd][0], date))
        if a is None:
            continue
        hh, mi = map(int, dl.split(":")[:2]); t = hh + mi / 60
        if not (1 <= t <= 22):
            continue
        x = np.arange(24)
        lev[i] = np.interp(t, x, a); dirn[i] = np.interp(t + 1, x, a) - np.interp(t - 1, x, a)
        lo, hi = a.min(), a.max(); pos[i] = (lev[i] - lo) / (hi - lo) if hi > lo else np.nan
    r = r.copy(); r["tide_lv"] = lev; r["tide_dir"] = np.where(np.isfinite(dirn), np.where(dirn > 0, "上げ潮", "下げ潮"), None)
    r["tide_pos"] = np.where(np.isfinite(pos), np.where(pos < 1 / 3, "低い", np.where(pos < 2 / 3, "中くらい", "高い")), None)
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datadir"); ap.add_argument("histds"); ap.add_argument("store"); ap.add_argument("tidedir")
    ap.add_argument("--out", default="results/idea068.md"); a = ap.parse_args()
    T = load_tide(a.tidedir)
    cols = KEY + ["boat", "course", "pos_raw", "deadline"]
    e = pd.concat([norm(pd.read_parquet(f"{p}/entries.parquet", columns=cols)) for p in (a.histds, a.datadir)], ignore_index=True)
    e = e.drop_duplicates(KEY + ["boat"], keep="last")
    e = e[e["jcd"].isin(STN.keys())]
    r = e.groupby(KEY)["deadline"].first().reset_index()
    r = tide_features(r, T)
    cover = r.assign(y=r["date"].str[:4]).groupby("y")["tide_dir"].apply(lambda s: s.notna().mean())
    x = e[~e["pos_raw"].astype(str).str.startswith("K") & e["course"].between(1, 6)].merge(r[KEY + ["tide_dir", "tide_pos"]], on=KEY)
    x = x[x["tide_dir"].notna() & x["tide_pos"].notna()].copy()
    x["cg"] = bgroup(x["course"]) + "コース"
    x["d1"] = x["tide_dir"] + "・" + x["tide_pos"] + "・" + x["cg"]
    x["d2"] = x["jcd"].map(VENUE) + "・" + x["tide_dir"] + "・" + x["tide_pos"] + "・" + x["cg"]
    x["d3"] = x["jcd"].map(VENUE) + "・" + x["tide_dir"] + "・" + x["cg"]
    dims = {"潮の向き×高さ×コース（場をまとめて）": "d1", "場×潮の向き×コース": "d3", "場×潮の向き×高さ×コース": "d2"}
    mk = market_boats(a.datadir)
    L = attach(x, mk)
    R, S, B, pooled, judge = evaluate(L, dims, mk, a.store, metrics=("1着",))
    # 12年の参考（オッズなし）：同じ年・同じコース群の全出走（同じ場の集合）の1着率との比
    H = x.copy(); H["year"] = H["date"].str[:4]; H["won"] = (H["pos_raw"].astype(str) == "01").astype(float)
    base = H.groupby(["year", "cg"])["won"].mean().rename("bw")
    out = []
    for dn, col in dims.items():
        g = H.groupby([col, "year", "cg"])["won"].agg(["sum", "size"]).reset_index()
        g = g.merge(base, left_on=["year", "cg"], right_index=True); g["ew"] = g["size"] * g["bw"]
        yr = g.groupby([col, "year"])[["sum", "ew", "size"]].sum().reset_index(); yr = yr[yr["size"] >= 100]; yr["r"] = yr["sum"] / yr["ew"]
        t = yr.groupby(col).agg(年数=("year", "size"), 出走=("size", "sum"), r=("r", "mean"), u=("r", lambda s: int((s > 1).sum()))).reset_index().rename(columns={col: "値"})
        t.insert(0, "区分", dn); out.append(t)
    Hs = pd.concat(out, ignore_index=True)
    Hs.columns = ["区分", "値", "年数", "出走", "1着率÷同じ年・コース群の全出走（年平均）", "上回った年"]
    Rshow = R[[c for c in R.columns if not c.startswith("_")]]
    stn = pd.DataFrame([{"場": VENUE[j], "jcd": j, "観測地点": f"{s[1]}（{s[0]}）"} for j, s in STN.items()])
    md = [f"# IDEA-068 潮位（探索 {EXP[0]}〜{EXP[1]}／評価 {EVL[0]}〜{EVL[1]}）", "",
          f"**判定：{judge}**", "",
          "出典：気象庁ホームページ「潮位表」（天文潮の推算値）をもとに加工して作成。", "",
          "## 定義", "",
          "- 締切時刻の潮位は、その日の毎時の推算値を直線で補間。潮の向きは 締切1時間後 − 1時間前 の符号（上げ潮／下げ潮）。",
          "- 高さは、その日の最低〜最高の幅の中での位置（3分の1ずつ：低い／中くらい／高い）。推算値なので、気圧・風による実際の潮位のずれは入っていない。",
          "- 進入コースは結果ファイルの実際のコース。市場の見込みは確定3連単オッズから出したその艇の1着確率。", "",
          "## 場と観測地点の対応（最寄り）", "", stn.to_markdown(index=False),
          "", "大村は大村湾の中に観測地点がないため佐世保で代用。鳴門は小松島、若松は砂津（小倉）、津は四日市港で代用（いずれも最寄りで、時刻・幅が少しずれる）。", "",
          "## 潮位が付いたレースの割合（年ごと）", "", cover.to_frame("割合").to_markdown(floatfmt=".3f"), "",
          "## 選ばれた区分（探索で 300艇以上・比 1.08 以上か 0.92 以下・CI が 1 をまたがない）と評価", "",
          (S.to_markdown(index=False, floatfmt=".3f") if len(S) else "なし"), "",
          "## 買い方（評価 2026-01〜08、単勝は K の実額・強い向きのみ、3連単は確定オッズで π×オッズ > 1.0・300倍以下）", "",
          (B[[c for c in B.columns if not c.startswith("_")]].to_markdown(index=False, floatfmt=".3f") if len(B) else "なし"), "", pooled, "",
          "## 全区分（1着の 実際÷市場の見込み）", "", Rshow.to_markdown(index=False, floatfmt=".3f"), "",
          "## 12年の参考（2014〜2026、オッズなし。同じ年・同じコース群の全出走との比、年100出走以上）", "", Hs.to_markdown(index=False, floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(judge); print(cover.to_string()); print(S.to_string() if len(S) else "選択なし")
    print(B[[c for c in B.columns if not c.startswith("_")]].to_string() if len(B) else ""); print(pooled)
    print(Rshow[Rshow["区分"].str.startswith("潮の向き")].to_string())

if __name__ == "__main__":
    main()
