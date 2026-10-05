"""第1段階: データ源の確認用。
公式の番組表(B)・競走成績(K)を数日分ダウンロード・解凍し、
過去レースのオッズページ(3連単/単勝)のHTMLも1レース分保存する。
出力は probe_out/ に置き、Actions の artifact として回収する。
"""
import datetime as dt, os, subprocess, sys, time, urllib.request

OUT = "probe_out"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}

def get(url, path):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
        print("OK ", url, os.path.getsize(path))
        return True
    except Exception as e:
        print("NG ", url, e)
        return False

days = [dt.date(2026, 10, 4), dt.date(2025, 10, 4)]
for d in days:
    for kind in ("B", "K"):
        name = f"{kind.lower()}{d:%y%m%d}"
        url = f"https://www1.mbrace.or.jp/od2/{kind}/{d:%Y%m}/{name}.lzh"
        lzh = f"{OUT}/{name}.lzh"
        if get(url, lzh):
            subprocess.run(["lha", "xqw=" + OUT, lzh], check=False)
        time.sleep(1)

# 過去レースのオッズページ（戸田 2026-10-04 12R）
base = "https://www.boatrace.jp/owpc/pc/race"
for page in ("odds3t", "oddstf", "beforeinfo", "racelist", "raceresult"):
    get(f"{base}/{page}?rno=12&jcd=02&hd=20261004", f"{OUT}/{page}_02_12_20261004.html")
    time.sleep(1)
# 古い日付のオッズが残っているか
get(f"{base}/odds3t?rno=12&jcd=02&hd=20231004", f"{OUT}/odds3t_02_12_20231004.html")

for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(os.path.join(OUT, f)))
