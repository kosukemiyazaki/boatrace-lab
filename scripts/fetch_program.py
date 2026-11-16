"""当日（と翌日）の番組表(B)を取得して STORE/program/YYYY/bYYMMDD.txt.gz に保存する（IDEA-075 の当日の候補一覧・pred_live.py の F 持ち用）
usage: python scripts/fetch_program.py STOREDIR [YYYY-MM-DD ...]   （日付を省くと日本時間の今日と明日）
- 公式（www1.mbrace.or.jp/od2/B）のファイル。fetch_bk.py と同じ取得・展開。既にあればスキップ
- ただし保存済みのファイルに「いましばらくお待ちください」（番組がまだ出ていない場。前日17時台はナイターの場がこうなる）が
  残っていれば取り直して上書きする。`--check FILE` はその場の数を出すだけ（live.yml の取り直しの判定用）
- 取得の成否と時刻を STORE/program/fetch_log.csv に追記する（いつ公開されるかを記録するため）
- daily の bk/ とは別の場所に置くので、build_dataset・daily の動きは変わらない
"""
import datetime as dt, glob, gzip, os, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(__file__))
from fetch_bk import fetch

JST = dt.timezone(dt.timedelta(hours=9))
PENDING = "いましばらくお待ちください"

def pending(path):
    """保存済みの番組表のうち、番組がまだ出ていない場の数（無ければ -1）"""
    if not os.path.exists(path):
        return -1
    return gzip.open(path, "rt", encoding="utf-8").read().count(PENDING)

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
        before = pending(dst)
        if before == 0:
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
        n = txt.count(PENDING)
        if before > 0 and n >= before:
            print("still pending", name, n); log.write(f"{d:%Y%m%d},{stamp},pending{n}\n"); continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with gzip.open(dst, "wt", encoding="utf-8") as f:
            f.write(txt)
        st = "ok" if n == 0 else f"partial{n}"
        print(st, name, len(txt)); log.write(f"{d:%Y%m%d},{stamp},{st}\n")
    log.close()

if __name__ == "__main__":
    if sys.argv[1] == "--check":
        print(pending(sys.argv[2])); sys.exit(0)
    main(sys.argv[1], sys.argv[2:])
