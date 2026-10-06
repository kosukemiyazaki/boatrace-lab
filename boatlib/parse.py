"""公式データのパーサ。
- 番組表(B)ファイル: 出走表（選手・勝率・モーター等）
- 競走成績(K)ファイル: 着順・展示タイム・ST・払戻
- boatrace.jp の odds3t ページ: 締切時3連単オッズ120通り
B/K は Shift_JIS(cp932) の固定長テキスト。ここでは UTF-8 に変換済みの文字列を受け取る。
"""
import itertools, re, unicodedata

COMBOS = [f"{a}-{b}-{c}" for a, b, c in itertools.permutations(range(1, 7), 3)]  # 辞書順120通り

def z2h(s):
    return unicodedata.normalize("NFKC", s)

def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def _blocks(txt, kind):
    """場ごとのブロック (jcd, text) を返す。kind は 'B' か 'K'。"""
    pat = re.compile(rf"^(\d\d){kind}BGN\s*$(.*?)^\1{kind}END", re.M | re.S)
    return [(m.group(1), m.group(2)) for m in pat.finditer(txt)]

# ---------------- 番組表 ----------------
B_RACE = re.compile(r"^\s*(\d+)Ｒ")
B_ROW = re.compile(r"^([1-6]) (\d{4})")

def parse_b(txt, date):
    """番組表 -> 選手行のリスト(dict)"""
    rows = []
    for jcd, blk in _blocks(txt, "B"):
        rno = None
        dist = deadline = None
        grade = ""
        for line in blk.splitlines():
            h = z2h(line)
            m = re.match(r"^\s*(\d+)R\s+(.*?)\s+H(\d+)m\s+電話投票締切予定(\d\d:\d\d)", h)
            if m:
                rno = int(m.group(1)); grade = m.group(2).strip(); dist = int(m.group(3)); deadline = m.group(4)
                continue
            if rno is None or not B_ROW.match(line):
                continue
            # 固定長: 艇(1) sp 登番(4) 名前(4) 年齢(2) 支部(2) 体重(2) 級別(2)
            boat = int(line[0]); toban = int(line[2:6])
            age = _num(line[10:12]); branch = line[12:14]; weight = _num(line[14:16]); cls = line[16:18]
            # 数値列も固定長（3桁のボート番号が前の列とくっつくため split は不可）
            f = lambda a, b: _num(line[a:b].strip())
            if f(18, 23) is None:
                continue
            rows.append(dict(
                date=date, jcd=jcd, rno=rno, boat=boat, toban=toban, name=line[6:10].replace("　", ""),
                age=age, branch=branch, weight=weight, cls=cls,
                nat_win=f(18, 23), nat_2r=f(23, 29), loc_win=f(29, 34), loc_2r=f(34, 40),
                motor_no=f(40, 43), motor_2r=f(43, 49), boat_no=f(49, 52), boat_2r=f(52, 58),
                race_name=grade, dist=dist, deadline=deadline,
            ))
    return rows

# ---------------- 競走成績 ----------------
K_RACE = re.compile(r"^\s+(\d+)R\s+(\S.*?)\s+H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm")
K_ROW = re.compile(r"^\s+([0-9A-Z]{2})\s+([1-6])\s+(\d{4})\s+(.{8})\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([1-6])?\s+([F L\d.]+?)\s{2,}")
K_PAY = re.compile(r"^\s+(\d+)R\s+(\d-\d-\d)\s+(\d+)\s+")

def parse_k(txt, date):
    """競走成績 -> (選手行のリスト, レース行のリスト)"""
    rows, races = [], []
    for jcd, blk in _blocks(txt, "K"):
        pay = {}
        for line in blk.splitlines():
            m = K_PAY.match(line)
            if m and "[払戻金]" not in line:
                pay[int(m.group(1))] = (m.group(2), int(m.group(3)))
        cur = None
        for line in blk.splitlines():
            m = K_RACE.match(z2h(line))
            if m:
                rno = int(m.group(1))
                combo, p3 = pay.get(rno, (None, None))
                cur = dict(date=date, jcd=jcd, rno=rno, race_name=m.group(2).strip(), dist=int(m.group(3)),
                           weather=m.group(4), wind_dir=m.group(5), wind=int(m.group(6)), wave=int(m.group(7)),
                           tri_combo=combo, tri_pay=p3)
                races.append(cur)
                continue
            if cur is None:
                continue
            m = K_ROW.match(line + "  ")
            if m:
                pos = m.group(1).strip()
                st = m.group(9).strip()
                rows.append(dict(
                    date=date, jcd=jcd, rno=cur["rno"], boat=int(m.group(2)), toban=int(m.group(3)),
                    pos=int(pos) if pos.isdigit() else None, pos_raw=pos,
                    motor_no=int(m.group(5)), boat_no=int(m.group(6)), exh_time=_num(m.group(7)),
                    course=int(m.group(8)) if m.group(8) else None,
                    st=_num(st) if not st.startswith(("F", "L")) else None, st_raw=st,
                ))
            elif re.match(r"^\s+([0-9A-Z]{2})\s+([1-6])\s+(\d{4}) ", line):
                # 欠場等で展示以降が空の行
                mm = re.match(r"^\s+([0-9A-Z]{2})\s+([1-6])\s+(\d{4}) ", line)
                rows.append(dict(date=date, jcd=jcd, rno=cur["rno"], boat=int(mm.group(2)), toban=int(mm.group(3)),
                                 pos=None, pos_raw=mm.group(1), motor_no=None, boat_no=None, exh_time=None,
                                 course=None, st=None, st_raw=""))
    return rows, races

def race_keys_k(txt):
    """Kファイルから (jcd, rno) の一覧"""
    out = []
    for jcd, blk in _blocks(txt, "K"):
        for line in blk.splitlines():
            m = K_RACE.match(z2h(line))
            if m:
                out.append((jcd, int(m.group(1))))
    return out

# ---------------- オッズ ----------------
ODDS_CELL = re.compile(r'<td class="oddsPoint[^"]*">([^<]*)</td>')
_ORDER = []
for i in range(120):
    r, a = divmod(i, 6)
    others = [x for x in range(1, 7) if x != a + 1]
    b, c = list(itertools.permutations(others, 2))[r]
    _ORDER.append(f"{a+1}-{b}-{c}")

def parse_odds3t(html):
    """odds3t ページ -> {combo: odds or None}。表がなければ None。"""
    vals = ODDS_CELL.findall(html)
    if len(vals) != 120:
        return None
    return {k: _num(v.strip()) for k, v in zip(_ORDER, vals)}

# ---------------- 単勝・複勝オッズ ----------------
ODDSTF_COLS = [f"tan{b}" for b in range(1, 7)] + [f"fuku_lo{b}" for b in range(1, 7)] + [f"fuku_hi{b}" for b in range(1, 7)]

def parse_oddstf(html):
    vals = ODDS_CELL.findall(html)
    if len(vals) != 12:
        return None
    tan = [_num(v.strip()) for v in vals[:6]]
    lo, hi = [], []
    for v in vals[6:]:
        a = v.strip().split("-")
        lo.append(_num(a[0]) if len(a) == 2 else None); hi.append(_num(a[1]) if len(a) == 2 else None)
    return dict(zip(ODDSTF_COLS, tan + lo + hi))

# ---------------- 直前情報 ----------------
BI_BOAT = ["weight", "exh", "tilt", "prop_new", "parts", "adj", "ex_course", "ex_st", "ex_f"]
BI_RACE = ["air", "weather", "wind", "wind_dir", "water", "wave"]
BI_COLS = BI_RACE + [f"{c}{b}" for b in range(1, 7) for c in BI_BOAT]
_TAG = re.compile(r"<[^>]+>")

def parse_beforeinfo(html):
    """直前情報: 艇ごとの体重・展示タイム・チルト・プロペラ新品・部品交換数・調整重量・展示進入・展示ST、水面気象"""
    body = re.sub(r"\s+", " ", html)
    tb = re.findall(r'<tbody class="is-fs12 ">(.*?)</tbody>', body)
    if len(tb) != 6:
        return None
    out = {}
    for t in tb:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", t)
        txt = [_TAG.sub("", x).replace("&nbsp;", "").strip() for x in tds]
        b = _num(txt[0])
        if b is None:
            return None
        b = int(b)
        parts = re.findall(r"<li[^>]*>(.*?)</li>", re.search(r'<ul class="labelGroup1">(.*?)</ul>', t).group(1)) if "labelGroup1" in t else []
        out[f"weight{b}"] = _num(txt[3].replace("kg", ""))
        out[f"exh{b}"] = _num(txt[4]); out[f"tilt{b}"] = _num(txt[5])
        out[f"prop_new{b}"] = 1.0 if "新" in txt[6] else 0.0
        out[f"parts{b}"] = float(len(parts))
        # 4行構成の3行目先頭が調整重量
        m = re.search(r'<tr> <td rowspan="2">([^<]*)</td> <td>ST</td>', t)
        out[f"adj{b}"] = _num(m.group(1)) if m else None
    st = re.findall(r'table1_boatImage1Number is-type(\d)">\d</span>.*?table1_boatImage1Time[^>]*>([^<]*)</span>', body)
    for course, (b, v) in enumerate(st, 1):
        b = int(b); v = v.strip()
        out[f"ex_course{b}"] = float(course)
        out[f"ex_f{b}"] = 1.0 if v.startswith("F") else 0.0
        out[f"ex_st{b}"] = _num(v.lstrip("FL") if v.lstrip("FL").startswith(".") else None) if v else None
    def wv(title):
        m = re.search(rf'LabelTitle">{title}</span> <span class="weather1_bodyUnitLabelData">([\d.\-]+)', body)
        return _num(m.group(1)) if m else None
    out["air"] = wv("気温"); out["wind"] = wv("風速"); out["water"] = wv("水温"); out["wave"] = wv("波高")
    m = re.search(r'is-weather(\d+)"', body); out["weather"] = _num(m.group(1)) if m else None
    m = re.search(r'is-wind(\d+)"', body); out["wind_dir"] = _num(m.group(1)) if m else None
    return {c: out.get(c) for c in BI_COLS}
