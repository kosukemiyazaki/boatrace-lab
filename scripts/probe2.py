"""データ源の確認2（取得の実装前の調査）。出力は probe_out/ に保存し、probe-out ブランチへコミットする。
- ボートレース: コンピュータ予想（pcexpect）が過去の日付で残っているか / 払戻一覧（pay）に売上が載っているか
- 地方競馬（keiba.go.jp）: トップ・月間開催日程からリンクをたどり、レース一覧・出馬表・オッズ・結果ページの形式と過去分の残り方を見る
"""
import os, re, time, urllib.parse, urllib.request

OUT = "probe_out"; os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}
LOG = open(f"{OUT}/index.tsv", "w")

def get_bin(url, name):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            b = r.read(); code = r.status
    except Exception as e:
        b = b""; code = repr(e)[:80]
    open(f"{OUT}/{name}", "wb").write(b)
    LOG.write(f"{name}\t{code}\t{len(b)}\t{url}\n"); LOG.flush(); print(name, code, len(b), url, flush=True)
    time.sleep(1)

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

K = "https://www.keiba.go.jp"
get_bin(f"{K}/pdf/manual/data_pdf_manual.pdf", "nar_data_manual.pdf")
for ym in ("2026/10", "2025/10", "2023/10", "2020/10"):
    y, m = ym.split("/")
    dl = get(f"{K}/KeibaWeb/DataDownload/RaceDataDownload?type=monthly&k_year={y}&k_month={int(m)}", f"nar_download_{y}{m}")
    open(f"{OUT}/nar_download_{y}{m}_links.txt", "w").write("\n".join(sorted(set(re.findall(r'(?:href|action)="([^"]+)"', dl)))))
    t = get(f"{K}/KeibaWeb/MonthlyConveneInfo/MonthlyConveneInfoTop?k_year={y}&k_month={int(m)}", f"nar_month_{y}{m}")
    lk = sorted(set(l.replace("&amp;", "&") for l in re.findall(r'RaceList\?k_raceDate=[^"\' <>]+', t)))
    if not lk:
        continue
    rl = get(f"{K}/KeibaWeb/TodayRaceInfo/{lk[0]}", f"nar_racelist_{y}{m}")
    rlinks = sorted(set(l.replace("&amp;", "&") for l in re.findall(r'(?:TodayRaceInfo/)?([A-Z][A-Za-z]+\?k_raceDate=[^"\' <>]+)', rl)))
    open(f"{OUT}/nar_racelist_{y}{m}_links.txt", "w").write("\n".join(rlinks))
    seen = set()
    for l in rlinks:
        kind = l.split("?")[0]
        if kind in seen or kind == "RaceList":
            continue
        seen.add(kind)
        get(f"{K}/KeibaWeb/TodayRaceInfo/{l}", f"nar_{kind}_{y}{m}")
