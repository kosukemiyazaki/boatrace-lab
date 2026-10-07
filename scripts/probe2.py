"""データ源の確認2（取得の実装前の調査）。出力は probe_out/ に保存し、probe-out ブランチへコミットする。
- ボートレース: コンピュータ予想（pcexpect）が過去の日付で残っているか / 払戻一覧（pay）に売上が載っているか
- 地方競馬（keiba.go.jp）: トップ・月間開催日程からリンクをたどり、レース一覧・出馬表・オッズ・結果ページの形式と過去分の残り方を見る
"""
import os, re, time, urllib.parse, urllib.request

OUT = "probe_out"; os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}
LOG = open(f"{OUT}/index.tsv", "w")

def get(url, name):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            b = r.read(); code = r.status
    except Exception as e:
        b = b""; code = repr(e)[:80]
    for enc in ("utf-8", "cp932", "euc-jp"):
        try:
            t = b.decode(enc); break
        except Exception:
            t = b.decode("utf-8", "replace")
    open(f"{OUT}/{name}.html", "w").write(t)
    LOG.write(f"{name}\t{code}\t{len(b)}\t{url}\n"); LOG.flush()
    print(name, code, len(b), url, flush=True)
    time.sleep(1)
    return t

B = "https://www.boatrace.jp/owpc/pc/race/"
for hd, j in (("20261004", "23"), ("20251004", "23"), ("20231004", "24")):
    get(f"{B}pcexpect?rno=12&jcd={j}&hd={hd}", f"br_pcexpect_{j}_{hd}")
    get(f"{B}pay?hd={hd}", f"br_pay_{hd}")

K = "https://www.keiba.go.jp"
top = get(f"{K}/KeibaWeb/TodayRaceInfo/TopTodayRaceInfo", "nar_top")
links = sorted(set(re.findall(r'href="(/KeibaWeb/[^"]+)"', top)))
open(f"{OUT}/nar_top_links.txt", "w").write("\n".join(links))
for ym in ("2026/10", "2025/10", "2023/10"):
    y, m = ym.split("/")
    t = get(f"{K}/KeibaWeb/MonthlyConveneInfo/MonthlyConveneInfoTop?k_year={y}&k_month={int(m)}", f"nar_month_{y}{m}")
    lk = sorted(set(re.findall(r'href="([^"]*RaceList[^"]*)"', t)))
    open(f"{OUT}/nar_month_{y}{m}_links.txt", "w").write("\n".join(lk))
    if not lk:
        continue
    u = urllib.parse.urljoin(K + "/KeibaWeb/MonthlyConveneInfo/", lk[0].replace("&amp;", "&"))
    rl = get(u, f"nar_racelist_{y}{m}")
    rlinks = sorted(set(l.replace("&amp;", "&") for l in re.findall(r'href="([^"]*TodayRaceInfo/[^"]+)"', rl)))
    open(f"{OUT}/nar_racelist_{y}{m}_links.txt", "w").write("\n".join(rlinks))
    seen = set()
    for l in rlinks:
        kind = re.search(r'TodayRaceInfo/([A-Za-z]+)', l).group(1)
        if kind in seen or kind in ("RaceList", "TopTodayRaceInfo"):
            continue
        seen.add(kind)
        get(urllib.parse.urljoin(K + "/KeibaWeb/TodayRaceInfo/", l), f"nar_{kind}_{y}{m}")
        if len(seen) >= 8:
            break
