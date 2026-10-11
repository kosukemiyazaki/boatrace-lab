"""live が締切前に記録した予想（モデル単体）の精度を日ごとに見る（運用の異常を早く見つけるため）。
usage: python scripts/monitor_model.py STOREDIR [--model frozen_v4] [--out results/model_monitor.md]
- 対象: live_pred のうち model が指定のもの・直前情報入り（has_bi=1）・締切前に記録したもの。結果は live_result（当日の結果ページ）。
- 見るもの（モデル単体のみ。撤退判定の物差し＝市場との比較は計算しない）:
  - 3連単の対数損失。開発期間の校正月（2026-08）の値 3.782 と、モデル自身が見込む値（予想の確率で計算した期待値）と比べる
  - 1着の対数損失と、モデルの本命（1着確率が最大の艇）の1着率
  - 1着確率の校正（確率帯ごとの予想と実際）
  - 予想の確率の合計が1か、直前情報入りの割合（記録の異常）
- 判定ではなく警告: 累計の3連単対数損失が「見込み + 3×標準誤差」を超えたら異常の疑いとして出す。
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boatlib.parse import COMBOS

DEV_LL = 3.782  # frozen_v4 の校正月（2026-08）の3連単対数損失（SUMMARY.md）
KEY = ["date", "jcd", "rno"]

def read(pattern):
    fs = sorted(glob.glob(pattern))
    df = pd.concat([pd.read_csv(f, dtype={"date": str, "jcd": str}) for f in fs], ignore_index=True)
    df["jcd"] = df["jcd"].str.zfill(2)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store"); ap.add_argument("--model", default="frozen_v4")
    ap.add_argument("--out", default="results/model_monitor.md")
    a = ap.parse_args()
    lp = read(f"{a.store}/live_pred/*.csv.gz")
    lp = lp[lp["model"] == a.model]
    all_pred = lp.drop_duplicates(KEY).groupby("date").size().rename("予想")
    lp = lp[(lp["has_bi"] == 1) & (lp["predicted_at"] < lp["deadline"] + ":00")].drop_duplicates(KEY, keep="first")
    res = read(f"{a.store}/live_result/*.csv.gz").drop_duplicates(KEY + ["boat"], keep="last")
    top3 = res[res["pos"].isin([1, 2, 3])].sort_values("pos").groupby(KEY)["boat"].agg(list)
    top3 = top3[top3.map(len) == 3].map(lambda b: f"{b[0]}-{b[1]}-{b[2]}").rename("tri").reset_index()
    win = res[res["pos"] == 1][KEY + ["boat"]].rename(columns={"boat": "winner"})
    m = lp.merge(top3, on=KEY).merge(win, on=KEY).copy()
    P = m[COMBOS].to_numpy(float)
    psum = P.sum(1)
    P = P / psum[:, None]
    yi = m["tri"].map({c: i for i, c in enumerate(COMBOS)}).values
    W = m[[f"win{b}" for b in range(1, 7)]].to_numpy(float)
    W = W / W.sum(1, keepdims=True)
    wi = m["winner"].astype(int).values - 1
    n = np.arange(len(m))
    m["ll3"] = -np.log(np.clip(P[n, yi], 1e-12, None))
    m["ll3_exp"] = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)  # モデル自身が見込む対数損失
    m["ll1"] = -np.log(np.clip(W[n, wi], 1e-12, None))
    m["fav_hit"] = (W.argmax(1) == wi).astype(float)
    m["fav_p"] = W.max(1)
    d = m.groupby("date").agg(レース=("ll3", "size"), 三連単LL=("ll3", "mean"), 見込みLL=("ll3_exp", "mean"), 一着LL=("ll1", "mean"),
                              本命の1着率=("fav_hit", "mean"), 本命の確率=("fav_p", "mean"))
    d = all_pred.to_frame().join(d, how="left")
    d["直前情報入り・締切前"] = lp.groupby("date").size()
    d = d.astype({"予想": "Int64", "レース": "Int64", "直前情報入り・締切前": "Int64"})
    tot = dict(races=len(m), ll3=m["ll3"].mean(), ll3_exp=m["ll3_exp"].mean(), se=m["ll3"].std() / np.sqrt(max(len(m), 1)),
               ll1=m["ll1"].mean(), fav_hit=m["fav_hit"].mean(), fav_p=m["fav_p"].mean())
    # 1着確率の校正（全艇・全レース）
    pw, ow = W.ravel(), (np.arange(6)[None, :] == wi[:, None]).ravel().astype(float)
    bins = [0, .05, .1, .2, .3, .5, .7, 1.0001]
    cb = pd.DataFrame({"p": pw, "o": ow, "bin": pd.cut(pw, bins, right=False)}).groupby("bin", observed=True).agg(艇数=("p", "size"), 予想=("p", "mean"), 実際=("o", "mean"))
    cb.index = cb.index.astype(str)
    warn = []
    if abs(psum - 1).max() > 0.01:
        warn.append(f"3連単の確率の合計が1から外れた記録がある（最大のずれ {abs(psum - 1).max():.3f}）")
    lim = max(tot["ll3_exp"], DEV_LL) + 3 * tot["se"]
    if tot["ll3"] > lim:
        warn.append(f"累計の3連単対数損失 {tot['ll3']:.3f} が 見込み+3×標準誤差（{lim:.3f}）を超えた")
    md = [f"# モデル単体の精度（{a.model}、live の締切前の記録）", "",
          "撤退判定の物差し（市場との比較）はここでは計算しない。運用の異常（特徴量のずれ・記録の欠け）を早く見つけるためのもの。", "",
          f"累計 {tot['races']} レース：3連単対数損失 **{tot['ll3']:.3f}**（標準誤差 {tot['se']:.3f}）／モデル自身の見込み {tot['ll3_exp']:.3f}／開発期間（2026-08）{DEV_LL}。"
          f"1着の対数損失 {tot['ll1']:.3f}、本命の1着率 {tot['fav_hit']:.1%}（本命の確率の平均 {tot['fav_p']:.1%}）。", "",
          "**警告：" + ("なし**" if not warn else "あり** — " + "；".join(warn)), "",
          "## 日ごと", "", d.to_markdown(floatfmt=".3f"), "",
          "## 1着確率の校正（全艇）", "", cb.to_markdown(floatfmt=".3f")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
