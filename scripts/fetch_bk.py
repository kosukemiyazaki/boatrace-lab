"""番組表(B)・競走成績(K)を日付範囲でダウンロードし、UTF-8 + gzip で保存する。
usage: python scripts/fetch_bk.py START END OUTDIR   (日付は YYYY-MM-DD, 両端含む)
保存先: OUTDIR/bk/YYYY/{b,k}YYMMDD.txt.gz（既存はスキップ）
"""
import datetime as dt, glob, gzip, os, subprocess, sys, tempfile, time, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research)"}

def fetch(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(5 * (i + 1))
        except Exception:
            time.sleep(5 * (i + 1))
    return None

def main(start, end, outdir):
    d = dt.date.fromisoformat(start)
    end = dt.date.fromisoformat(end)
    n_ok = n_miss = 0
    while d <= end:
        for kind in ("B", "K"):
            name = f"{kind.lower()}{d:%y%m%d}"
            dst = f"{outdir}/bk/{d:%Y}/{name}.txt.gz"
            if os.path.exists(dst):
                continue
            raw = fetch(f"https://www1.mbrace.or.jp/od2/{kind}/{d:%Y%m}/{name}.lzh")
            time.sleep(0.5)
            if not raw:
                n_miss += 1
                print("miss", name)
                continue
            with tempfile.TemporaryDirectory() as tmp:
                open(f"{tmp}/a.lzh", "wb").write(raw)
                subprocess.run(["lha", f"xqw={tmp}", f"{tmp}/a.lzh"], check=True)
                txts = [p for p in glob.glob(f"{tmp}/*") if not p.endswith(".lzh")]
                txt = open(txts[0], "rb").read().decode("cp932", "replace").replace("\r\n", "\n")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with gzip.open(dst, "wt", encoding="utf-8") as f:
                f.write(txt)
            n_ok += 1
        d += dt.timedelta(days=1)
    print(f"done ok={n_ok} miss={n_miss}")

if __name__ == "__main__":
    main(*sys.argv[1:4])
