"""第1段階: データ源の確認用。
公式の番組表(B)・競走成績(K)をダウンロード・解凍し、
過去レースのオッズページ(3連単/単勝)のHTMLを日付を変えて保存する（保持期間の確認）。
各日付で実際に開催のあった場(KファイルのxxKBGN行)を使う。
出力は probe_out/ に置き、ワークフローが probe-out ブランチへコミットする。
"""
import datetime as dt, glob, os, re, subprocess, time, urllib.request

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

def lzh(kind, d):
    name = f"{kind.lower()}{d:%y%m%d}"
    url = f"https://www1.mbrace.or.jp/od2/{kind}/{d:%Y%m}/{name}.lzh"
    p = f"{OUT}/{name}.lzh"
    if get(url, p):
        subprocess.run(["lha", "xqw=" + OUT, p], check=False)
        os.remove(p)
    time.sleep(1)
    hits = glob.glob(f"{OUT}/{name}.*") + glob.glob(f"{OUT}/{name.upper()}.*")
    return hits[0] if hits else None

base = "https://www.boatrace.jp/owpc/pc/race"
days = [dt.date(2026, 10, 4), dt.date(2026, 7, 4), dt.date(2026, 4, 4), dt.date(2025, 10, 4),
        dt.date(2025, 4, 4), dt.date(2024, 10, 4), dt.date(2023, 10, 4), dt.date(2022, 10, 4)]
for d in days:
    lzh("B", d)
    k = lzh("K", d)
    if not k:
        continue
    txt = open(k, "rb").read().decode("cp932", "replace")
    jcds = re.findall(r"^(\d\d)KBGN", txt, re.M)
    print(d, "venues", jcds)
    if not jcds:
        continue
    j = jcds[0]
    pages = ("odds3t", "oddstf", "raceresult") if d.year < 2026 or d.month < 10 else \
            ("odds3t", "odds3f", "odds2tf", "oddsk", "oddstf", "beforeinfo", "racelist", "raceresult")
    for page in pages:
        get(f"{base}/{page}?rno=12&jcd={j}&hd={d:%Y%m%d}", f"{OUT}/{page}_{j}_12_{d:%Y%m%d}.html")
        time.sleep(1)

for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(os.path.join(OUT, f)))
