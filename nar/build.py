"""nar/fetch.py で落とした ZIP を表（parquet）にする。**ローカル専用**（出力も nar_data/ の中に置き、コミットしない）。
usage: python nar/build.py [--data nar_data]
出力: DATA/races.parquet（1行=1レース）, DATA/horses.parquet（1行=1頭）, DATA/tri_odds.parquet（3連単の確定オッズ、1行=1組番）,
      DATA/tri_pay.parquet（3連単の払戻）
- 馬の識別は「馬名＋生年月日」（公式データに馬IDはない）。騎手・調教師は名前＋所属。
- 着順・タイム・着差・上がり3F・人気はレース後の情報。モデルの特徴量には、そのレースより前のレースの値しか使わない（nar/run.py）。
"""
import sys, argparse, glob, io, os, re, zipfile
import numpy as np
import pandas as pd

def read_zip_csvs(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            raw = z.read(n)
            for enc in ("utf-8-sig", "cp932"):
                try:
                    t = raw.decode(enc); break
                except UnicodeDecodeError:
                    continue
            out[n] = pd.read_csv(io.StringIO(t), dtype=str, keep_default_na=False)
    return out

def digits(s):
    return s.str.replace(r"\D", "", regex=True)

def num(s):
    return pd.to_numeric(s.str.replace(",", "").str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")

def record(s):
    """「1-2-3-10」のような成績 → (出走数, 1着数, 3着内数)"""
    p = s.str.extract(r"(\d+)\D+(\d+)\D+(\d+)\D+(\d+)").astype(float)
    n = p.sum(axis=1)
    return n.where(n > 0), p[0], p[0] + p[1] + p[2]

def to_seconds(s):
    """「1:12.3」「72.3」→ 秒"""
    m = s.str.extract(r"(?:(\d+):)?(\d+(?:\.\d+)?)")
    return m[0].astype(float).fillna(0) * 60 + m[1].astype(float)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="nar_data")
    a = ap.parse_args()
    races, horses, pays = [], [], []
    for p in sorted(glob.glob(f"{a.data}/race/*.zip")):
        for n, df in read_zip_csvs(p).items():
            if "racelist" in n:
                races.append(df)
            elif "horselist" in n:
                horses.append(df)
            elif "payback" in n:
                pays.append(df)
    r = pd.concat(races, ignore_index=True); h = pd.concat(horses, ignore_index=True); pb = pd.concat(pays, ignore_index=True)
    key = ["venue", "date", "rno"]
    def keycols(df):
        return pd.DataFrame({"venue": df["競馬場"].str.strip(), "date": digits(df["競走年月日"]).str[:8], "rno": num(df["レース番号"]).astype("Int64")})
    R = pd.concat([keycols(r), pd.DataFrame({
        "post_time": r["発走時刻"].str.strip(), "race_type": r["競走種類名称"].str.strip(), "race_name": r["レース名"].str.strip(),
        "surface": r["芝ダート区分"].str.strip(), "turn": r["回り"].str.strip(), "dist": num(r["距離"]), "weather": r["天候"].str.strip(),
        "going": r["馬場"].str.strip(), "field": num(r["頭数"]), "cond": r["条件"].str.strip(), "prize1": num(r["1着賞金(円)"])})], axis=1)
    R = R.drop_duplicates(key)
    H = pd.concat([keycols(h), pd.DataFrame({
        "waku": num(h["枠番"]), "num": num(h["馬番"]).astype("Int64"), "horse": h["馬名"].str.strip() + "|" + digits(h["生年月日"]),
        "sex": h["性"].str.strip(), "age": num(h["齢"]), "sire": h["父馬名"].str.strip(), "dam_sire": h["母父馬名"].str.strip(),
        "jockey": h["騎手名"].str.strip() + "|" + h["騎手所属"].str.strip(), "weight_carried": num(h["負担重量"]),
        "trainer": h["調教師"].str.strip() + "|" + h["調教師所属"].str.strip(), "owner": h["馬主氏名"].str.strip(),
        "body_weight": num(h["馬体重"]), "body_diff": num(h["馬体重増減"]),
        "best_time": to_seconds(h["最高タイム"]), "best_time_good": to_seconds(h["最高タイム良馬場"]),
        # レース後の情報（特徴量には使わない）
        "pos_raw": h["着順"].str.strip(), "pos": num(h["着順"]), "time": to_seconds(h["タイム"]), "last3f": num(h["上がり3F"]), "pop": num(h["人気"])})], axis=1)
    for col, src in (("rec", "全成績"), ("rec_venue", "当競馬場成績"), ("rec_dist", "うち当距離成績"), ("jrec", "騎手成績")):
        n_, w_, t3_ = record(h[src])
        H[f"{col}_n"], H[f"{col}_win"], H[f"{col}_top3"] = n_, w_ / n_, t3_ / n_
    H = H.dropna(subset=["num"]).drop_duplicates(key + ["num"])
    P = pd.concat([keycols(pb), pd.DataFrame({
        "tri": num(pb["３連単組番馬番1"]).astype("Int64").astype(str) + "-" + num(pb["３連単組番馬番2"]).astype("Int64").astype(str) + "-" + num(pb["３連単組番馬番3"]).astype("Int64").astype(str),
        "tri_pay": num(pb["３連単払戻金（円）"])})], axis=1).drop_duplicates(key)
    os.makedirs(a.data, exist_ok=True)
    R.to_parquet(f"{a.data}/races.parquet"); H.to_parquet(f"{a.data}/horses.parquet"); P.to_parquet(f"{a.data}/tri_pay.parquet")
    print(f"races {len(R)}（{R['date'].min()}〜{R['date'].max()}、{R['venue'].nunique()}場）, horses {len(H)}, payouts {len(P)}")
    print("  着順の値の例（上位10）:", H["pos_raw"].value_counts().head(10).to_dict())
    print("  全成績が読めた割合:", round(float(H["rec_n"].notna().mean()), 3), "／ 騎手成績:", round(float(H["jrec_n"].notna().mean()), 3))
    # 3連単の確定オッズ
    T = []
    for p in sorted(glob.glob(f"{a.data}/odds/*.zip")):
        for n, df in read_zip_csvs(p).items():
            kinds = df["賭式"].str.strip()
            if not T:
                print("  賭式の値:", sorted(kinds.unique())[:20])
            tri = kinds.str.contains("3連単|３連単|三連単") | (kinds == "8")  # 表記ゆれに備える
            d = df[tri]
            if len(d):
                T.append(pd.concat([keycols(d), pd.DataFrame({
                    "tri": num(d["番号1"]).astype("Int64").astype(str) + "-" + num(d["番号2"]).astype("Int64").astype(str) + "-" + num(d["番号3"]).astype("Int64").astype(str),
                    "odds": num(d["オッズ"])})], axis=1))
    if T:
        O = pd.concat(T, ignore_index=True).dropna(subset=["odds"]).drop_duplicates(key + ["tri"], keep="last")
        O.to_parquet(f"{a.data}/tri_odds.parquet")
        print(f"tri_odds {len(O)} 組番、{O.groupby(key).ngroups} レース（{O['date'].min()}〜{O['date'].max()}）")
    else:
        print("3連単の確定オッズが見つからない（賭式の値を確認）")

if __name__ == "__main__":
    # Windows のコンソール（cp932）で表示できない文字があっても止まらないようにする
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    main()
