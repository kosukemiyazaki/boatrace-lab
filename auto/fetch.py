"""オートレース公式（autorace.jp）のレース情報を手元に保存する。**ローカル専用**。
利用規約：私的利用の範囲に限る、事前の許可なく転載・複製・頒布・公開しない、大量アクセスをしない。
→ 取得したデータは auto_data/ にだけ置き（.gitignore 済み）、公開リポジトリには集計値しか置かない。間隔は既定で2秒以上あける。
usage: python auto/fetch.py [--data auto_data] [--from 2024-09-01] [--to 2026-08-31] [--interval 2.0]
- 日ごと・場ごとに RaceRefund（その日の全レースの払戻・欠車）を取り、開催があればレースごとに Odds（出走表の一部・全券種の最終オッズ・売上）を取る
- 保存: DATA/refund/YYYY-MM-DD_P.json.gz（開催なしは .none）、DATA/odds/YYYY-MM-DD_P_RR.json.gz。既にあるものは取り直さない（途中で止めても再開できる）
- 場コード: 2 川口, 3 伊勢崎, 4 浜松, 5 飯塚, 6 山陽（公式の開催一覧で確認した番号）
"""
import argparse, datetime as dt, gzip, json, os, re, sys, time, urllib.parse, urllib.request, http.cookiejar

B = "https://autorace.jp"
UA = {"User-Agent": "Mozilla/5.0 (boatrace-lab research; personal analysis)"}
PLACES = (2, 3, 4, 5, 6)

class Session:
    def __init__(self, interval):
        self.interval = interval; self.last = 0.0; self.token = None
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def wait(self):
        d = self.interval - (time.time() - self.last)
        if d > 0:
            time.sleep(d)
        self.last = time.time()
    def refresh(self):
        self.wait()
        h = self.op.open(urllib.request.Request(f"{B}/race_info/Live/hamamatsu", headers=UA), timeout=60).read().decode("utf-8", "replace")
        t = re.findall(r'name="csrf-token" content="([^"]+)"', h)
        if not t:
            raise RuntimeError("csrf token が見つからない")
        self.token = t[0]
    def post(self, ep, params):
        for attempt in range(4):
            if self.token is None:
                self.refresh()
            self.wait()
            hdr = dict(UA, **{"X-CSRF-TOKEN": self.token, "X-Requested-With": "XMLHttpRequest", "Content-Type": "application/x-www-form-urlencoded",
                               "Referer": f"{B}/race_info/Live/hamamatsu"})
            try:
                with self.op.open(urllib.request.Request(f"{B}/race_info/{ep}", data=urllib.parse.urlencode(params).encode(), headers=hdr), timeout=60) as r:
                    return json.loads(r.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError as e:
                print(f"  HTTP {e.code} {ep} {params}（再試行 {attempt + 1}）", flush=True)
                self.token = None; time.sleep(30 * (attempt + 1))
            except Exception as e:
                print(f"  NG {ep} {params} {e}（再試行 {attempt + 1}）", flush=True)
                time.sleep(30 * (attempt + 1))
        return None

def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path + ".tmp", "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="auto_data"); ap.add_argument("--from", dest="start", default="2024-09-01")
    ap.add_argument("--to", default="2026-08-31"); ap.add_argument("--interval", type=float, default=2.0)
    a = ap.parse_args()
    a.interval = max(a.interval, 1.5)  # 大量アクセスにならないよう、1.5秒未満にはしない
    s = Session(a.interval)
    d0, d1 = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.to)
    days = [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
    print(f"{len(days)} 日 × {len(PLACES)} 場（間隔 {a.interval} 秒。開催のある日はレースごとに追加で取得）", flush=True)
    n_req = 0
    for day in days:
        for pc in PLACES:
            rp = f"{a.data}/refund/{day}_{pc}.json.gz"
            if os.path.exists(rp[:-8] + ".none"):
                continue
            if os.path.exists(rp):
                body = json.load(gzip.open(rp, "rt", encoding="utf-8"))
            else:
                j = s.post("RaceRefund", {"placeCode": pc, "raceDate": day}); n_req += 1
                if j is None:
                    continue
                if not j.get("body"):
                    codes = [e.get("code") for e in (j.get("errors") or [])]
                    if "4101" in codes:  # 開催なし
                        os.makedirs(os.path.dirname(rp), exist_ok=True); open(rp[:-8] + ".none", "w").close()
                    else:
                        print("  RaceRefund エラー", day, pc, j.get("errors"), flush=True)
                    continue
                body = j["body"]; dump(rp, body)
            for race in body:
                rno = int(race.get("raceNo", 0))
                op_ = f"{a.data}/odds/{day}_{pc}_{rno:02d}.json.gz"
                if rno <= 0 or os.path.exists(op_):
                    continue
                j = s.post("Odds", {"placeCode": pc, "raceDate": day, "raceNo": rno}); n_req += 1
                if j is not None and j.get("body"):
                    dump(op_, j["body"])
            print(f"{day} 場{pc}: {len(body)} レース（累計リクエスト {n_req}）", flush=True)

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main()
