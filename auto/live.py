"""オートレースの締切前のオッズ・売上を時間帯別に保存する（ボートの live_snapshot と見比べるため）。**ローカル専用**。
利用規約：私的利用の範囲に限る、転載・複製・頒布・公開しない、大量アクセスをしない。
→ 保存先は auto_data/live/ だけ（.gitignore 済み）。リクエストの間隔は2秒以上、1台の PC から順番に。
usage: python auto/live.py [--data auto_data] [--interval 2.0] [--until 24:00] [--once]
- 30秒ごと（どこかの場の締切が4分以内なら15秒ごと）に公式の「本日の開催」（race_info/XML/Hold/Today）を1回読み、
  各場の発売中のレース番号と締切時刻（telvoteTime、なければ raceStartTime）を知る。
- 締切までの残りが下の時間帯に入ったら、そのレースの Odds（全券種のオッズ＋券種別の売上 salesInfo）を1回取る。
  時間帯（秒）: 20分 15分 10分 6分 4分 2分 1分 30秒 10秒 / 締切の1分後（締切後の確定に近い値）
  取り損ねた時間帯（次の時間帯に入ってしまった）は飛ばす。実際に取った時刻は fetched_at に残す。
- 保存: DATA/live/YYYY-MM-DD/P_RR/{時間帯}.json.gz = {fetched_at, lead, deadline, start, hold（その場の開催情報）, odds（Odds の body）}
- 最終オッズ・着順・払戻は翌日以降に auto/fetch.py --from その日 --to その日 で取る（締切時との比較に使う）。
- 1日のアクセス：Hold/Today が 1時間に 120〜240回、Odds が 1レース 10回（5場 × 12レースで約600回）。
"""
import argparse, datetime as dt, gzip, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from fetch import B, UA, Session, dump

JST = dt.timezone(dt.timedelta(hours=9))
LEADS = (("m20", 1200), ("m15", 900), ("m10", 600), ("m06", 360), ("m04", 240), ("m02", 120), ("m01", 60), ("s30", 30), ("s10", 10))
AFTER = ("p01", -60)  # 締切の1分後

def now():
    return dt.datetime.now(JST)

def hold_today(s):
    s.wait()
    with s.op.open(urllib.request.Request(f"{B}/race_info/XML/Hold/Today", headers=UA), timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))

def race_day(t):
    """深夜0〜5時は前日の開催とみなす（ミッドナイト）"""
    return (t - dt.timedelta(hours=5)).date()

def at(day, hhmm):
    """開催日 day の HH:MM（24時以降・翌日の早朝にも対応）"""
    try:
        h, m = map(int, str(hhmm).split(":")[:2])
    except Exception:
        return None
    base = dt.datetime.combine(day, dt.time(0, 0), JST)
    t = base + dt.timedelta(hours=h, minutes=m)
    if h < 5:
        t += dt.timedelta(days=1)
    return t

def targets(body, day):
    """Hold/Today の body → [(placeCode, raceNo, 締切, 発走, その場の情報)]"""
    out = []
    for v in (body or {}).get("today") or []:
        if str(v.get("cancelFlg", "0")) not in ("0", "", "None"):
            continue
        rno = int(v.get("oddsRaceNo") or v.get("nowRaceNo") or 0)
        if rno <= 0:
            continue
        start = at(day, v.get("raceStartTime"))
        dl = at(day, v.get("telvoteTime")) or start
        if dl is None:
            continue
        out.append((int(v["placeCode"]), rno, dl, start, v))
    return out

def slot_for(sec, taken):
    """残り sec 秒のときに取るべき時間帯（まだ取っていない中で、いまの残りに一番近いもの）"""
    cur = None
    for name, lim in LEADS:
        if sec <= lim:
            cur = name
    if cur is not None and cur not in taken:
        return cur
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="auto_data"); ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--until", default="24:00", help="この時刻（JST）を過ぎたら終了"); ap.add_argument("--once", action="store_true", help="1回だけ読んで表示（動作確認）")
    a = ap.parse_args()
    a.interval = max(a.interval, 1.5)
    s = Session(a.interval)
    day = race_day(now()); end = at(day, a.until) if a.until != "24:00" else dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 30), JST)
    taken = {}      # (pc, rno) → 取った時間帯の集合
    pending = {}    # (pc, rno) → (締切, 発走, 情報)：締切後の1回を取るまで覚えておく
    n_req = 0; last_print = 0
    print(f"開始 {now():%H:%M:%S}（開催日 {day}、終了 {end:%m-%d %H:%M}、間隔 {a.interval} 秒）", flush=True)
    while now() < end:
        try:
            j = hold_today(s); n_req += 1
        except Exception as e:
            print(f"  Hold/Today 失敗 {e}（60秒後に再試行）", flush=True); time.sleep(60); continue
        body = j.get("body") or {}
        ts = targets(body, day)
        if a.once:
            for pc, rno, dl, st, v in ts:
                print(f"場{pc} {v.get('placeKey')}: 発売中 {rno}R 締切 {dl:%H:%M} 発走 {st:%H:%M}（nowRaceNo={v.get('nowRaceNo')} oddsRaceNo={v.get('oddsRaceNo')} resultRaceNo={v.get('resultRaceNo')}）" if st else f"場{pc}: {rno}R 締切 {dl:%H:%M}")
            if not ts:
                print("いま発売中のレースなし（開催なし・発売前・終了後）")
            return
        t = now()
        for pc, rno, dl, st, v in ts:
            k = (pc, rno); pending[k] = (dl, st, v)
            sec = (dl - t).total_seconds()
            name = slot_for(sec, taken.setdefault(k, set()))
            if name is None or sec < -30:
                continue
            o = s.post("Odds", {"placeCode": pc, "raceDate": day.isoformat(), "raceNo": rno}); n_req += 1
            taken[k].add(name)
            if o and o.get("body"):
                dump(f"{a.data}/live/{day}/{pc}_{rno:02d}/{name}.json.gz",
                     {"fetched_at": now().isoformat(timespec="seconds"), "lead": name, "sec_left": round(sec), "deadline": dl.isoformat(),
                      "start": st.isoformat() if st else None, "hold": v, "odds": o["body"]})
        # 締切の1分後の1回（発売中のレースが次に移ったあとでも取る）
        for k, (dl, st, v) in list(pending.items()):
            if (now() - dl).total_seconds() >= -AFTER[1]:
                pc, rno = k
                o = s.post("Odds", {"placeCode": pc, "raceDate": day.isoformat(), "raceNo": rno}); n_req += 1
                if o and o.get("body"):
                    dump(f"{a.data}/live/{day}/{pc}_{rno:02d}/{AFTER[0]}.json.gz",
                         {"fetched_at": now().isoformat(timespec="seconds"), "lead": AFTER[0], "sec_left": round((dl - now()).total_seconds()),
                          "deadline": dl.isoformat(), "start": st.isoformat() if st else None, "hold": v, "odds": o["body"]})
                del pending[k]
        if time.time() - last_print > 600:
            last_print = time.time()
            print(f"{now():%H:%M} 発売中 {', '.join(f'場{pc}-{r}R' for pc, r, *_ in ts) or 'なし'}／取ったレース {len(taken)}／リクエスト {n_req}", flush=True)
        near = any(0 < (dl - now()).total_seconds() < 240 for _, _, dl, _, _ in ts)
        time.sleep(15 if near else 30)
    print(f"終了 {now():%H:%M:%S}（レース {len(taken)}、リクエスト {n_req}）", flush=True)

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main()
