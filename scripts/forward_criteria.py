"""Step 3: 前向きテストの判定基準と、110%を統計的に確認するのに必要な賭けの数
usage: python scripts/forward_criteria.py WALKFORWARD_JSON --omax 30 --thr 1.3 [--days 30] [--out results/forward_criteria.md]

1) 判定基準: バックテストの日別成績（賭け金・払戻）を日単位で復元抽出し、前向きテストと同じ日数 D 日の
   回収率の分布を作る。前向きテストの回収率がこの分布の下位5%点を下回ったら「再現せず」。
2) 必要な賭けの数: 真の回収率が R のとき、片側検定（有意水準5%、検出力80%）で
   「回収率 > 100%」（H0: 回収率 ≤ 100%）を示すのに必要な日数・点数。
   日単位のばらつき（同じ日の賭けは独立でないことを含む）から計算する:
     n_days = ((z_0.95 + z_0.80) / (R - 1))^2 × Var(払戻_d - R×賭け金_d) / 平均(賭け金_d)^2
   参考として、1点ごとに独立とみなした理論値（平均オッズ o の目を買い、的中率 = R/o）も出す:
     Var(1点の払戻/賭け金) = R×o - R^2,  n_tickets = ((z_0.95 + z_0.80) / (R - 1))^2 × (R×o - R^2)
"""
import argparse, json
import numpy as np
import pandas as pd

Z = 1.6449 + 0.8416  # 片側5% + 検出力80%
RNG = np.random.default_rng(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wf"); ap.add_argument("--omax", type=float, required=True); ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--days", type=int, default=30); ap.add_argument("--out", default="results/forward_criteria.md")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    wf = json.load(open(a.wf))
    g = next(x for x in wf["results"] if x["omax"] == a.omax and x["thr"] == a.thr)
    daily = pd.DataFrame(g["daily"], columns=["day", "stake", "ret"])
    st, rt = daily["stake"].to_numpy(float), daily["ret"].to_numpy(float)
    k = len(st)
    # 1) D日分の回収率の分布
    idx = RNG.integers(0, k, size=(20000, a.days))
    rois = rt[idx].sum(1) / st[idx].sum(1)
    q = {p: float(np.quantile(rois, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
    # 2) 必要な日数・点数（日単位）
    tickets_per_day = g["tickets"] / k
    rows = []
    for R in (1.05, 1.10, 1.15, 1.20):
        v = np.var(rt - R * st, ddof=1) / st.mean() ** 2
        n_days = Z ** 2 * v / (R - 1) ** 2
        rows.append({"真の回収率": f"{R:.0%}", "必要日数（日単位のばらつき）": round(n_days), "必要点数": round(n_days * tickets_per_day)})
    theory = []
    for o in (10, 20, 30, 50, 100):
        R = 1.10
        theory.append({"平均オッズ": o, "必要点数（独立とみなした理論値, 真の回収率110%）": round(Z ** 2 * (R * o - R * R) / (R - 1) ** 2)})
    md = [f"### 判定基準の計算{a.label}（オッズ上限 {a.omax:g}倍 × EV閾値 {a.thr}）",
          f"- もとにしたバックテスト: {k}日、{g['tickets']}点、回収率 {g['roi']:.1%}、平均オッズ {g['mean_odds']:.1f}倍、1日 {tickets_per_day:.1f}点",
          f"- 前向きテスト {a.days}日分の回収率の分布（日単位ブートストラップ 20,000回）: "
          f"下位5% {q[0.05]:.1%} / 25% {q[0.25]:.1%} / 中央値 {q[0.5]:.1%} / 75% {q[0.75]:.1%} / 上位5% {q[0.95]:.1%}",
          f"- **判定: 前向きテスト{a.days}日の回収率が {q[0.05]:.1%} を下回ったら「再現せず」**",
          "", "必要な賭けの数（片側5%・検出力80%で「回収率 > 100%」を示す）", "",
          pd.DataFrame(rows).to_markdown(index=False), "", pd.DataFrame(theory).to_markdown(index=False)]
    txt = "\n".join(md)
    open(a.out, "w").write(txt + "\n")
    print(txt)

if __name__ == "__main__":
    main()
