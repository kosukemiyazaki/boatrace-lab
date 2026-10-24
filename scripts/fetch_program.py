"""当日（と翌日）の番組表(B)を取得して STORE/program/YYYY/bYYMMDD.txt.gz に保存する（IDEA-075 の当日の候補一覧・pred_live.py の F 持ち用）
usage: python scripts/fetch_program.py STOREDIR [YYYY-MM-DD ...]   （日付を省くと日本時間の今日と明日）
- 公式（www1.mbrace.or.jp/od2/B）のファイル。fetch_bk.py と同じ取得・展開。既にあればスキップ
- 取得の成否と時刻を STORE/program/fetch_log.csv に追記する（いつ公開されるかを記録するため）
- daily の bk/ とは別の場所に置くので、build_dataset・daily の動きは変わらない
"""
import datetime as dt, glob, gzip, os, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(__file__))
from fetch_bk import fetch

JST = dt.timezone(dt.timedelta(hours=9))

def main(store, dates):
    now = dt.datetime.now(JST)
    if not dates:
        dates = [now.date().isoformat(), (now.date() + dt.timedelta(days=1)).isoformat()]
    os.makedirs(f"{store}/program", exist_ok=True)
    log = open(f"{store}/program/fetch_log.csv", "a", encoding="utf-8")
    if log.tell() == 0:
        log.write("date,checked_at_jst,status\n")
    for ds in dates:
        d = dt.date.fromisoformat(ds); name = f"b{d:%y%m%d}"
        dst = f"{store}/program/{d:%Y}/{name}.txt.gz"
        if os.path.exists(dst):
            print("exists", name); continue
        raw = fetch(f"https://www1.mbrace.or.jp/od2/B/{d:%Y%m}/{name}.lzh")
        time.sleep(0.5)
        stamp = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
        if not raw:
            print("miss", name); log.write(f"{d:%Y%m%d},{stamp},miss\n"); continue
        with tempfile.TemporaryDirectory() as tmp:
            open(f"{tmp}/a.lzh", "wb").write(raw)
            subprocess.run(["lha", f"xqw={tmp}", f"{tmp}/a.lzh"], check=True)
            txts = [p for p in glob.glob(f"{tmp}/*") if not p.endswith(".lzh")]
            txt = open(txts[0], "rb").read().decode("cp932", "replace").replace("\r\n", "\n")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with gzip.open(dst, "wt", encoding="utf-8") as f:
            f.write(txt)
        print("ok", name, len(txt)); log.write(f"{d:%Y%m%d},{stamp},ok\n")
    log.close()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
