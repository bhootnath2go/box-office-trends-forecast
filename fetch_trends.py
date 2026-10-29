"""Fetch anchored Google Trends series for every comp film and Ramayana.

Each request pairs the film's query with a constant anchor term so values from
different requests share one scale: value / median(anchor in that window).
Results are cached per (film, geo, gprop) in data/trends_raw/ so reruns only
fetch what's missing.
"""
import sys, time, random
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq

WW = "--ww" in sys.argv           # worldwide interest, for the overseas model
ANCHOR = "sudoku" if WW else "Hanuman Chalisa"   # steady mid-volume term in that geography
PRE_DAYS, POST_DAYS = 60, 3
ROOT = Path(__file__).parent
TOPIC = "--topic" in sys.argv or WW     # query Google's film topic entity instead of a search string
RAW = ROOT / "data" / ("trends_ww_raw" if WW else "trends_topic_raw" if TOPIC else "trends_raw")
RAW.mkdir(parents=True, exist_ok=True)
TODAY = pd.Timestamp("2026-10-28")

# (geo, gprop, label)
CHANNELS = [("", "", "web_ww")] if WW else [("IN", "", "web_in")] if TOPIC else [("IN", "", "web_in"), ("IN", "youtube", "yt_in")]  # geo="" returned India data, so no worldwide channel


def films():
    frames = []
    for f in ["films_india.csv", "films_hollywood.csv"]:
        p = ROOT / "data" / f
        if p.exists():
            frames.append(pd.read_csv(p)[["title", "google_trends_query", "release_date"]])
    df = pd.concat(frames, ignore_index=True)
    ram = ROOT / "data" / "ramayana.csv"
    if ram.exists():
        df = pd.concat([df, pd.read_csv(ram)[["title", "google_trends_query", "release_date"]]])
    return df.dropna(subset=["google_trends_query", "release_date"])


def fetch(pt, query, start, end, geo, gprop):
    for attempt in range(6):
        try:
            pt.build_payload([query, ANCHOR], timeframe=f"{start:%Y-%m-%d} {end:%Y-%m-%d}", geo=geo, gprop=gprop)
            return pt.interest_over_time()
        except Exception as e:  # 429s and transient errors
            wait = 60 * (attempt + 1) + random.uniform(0, 20)
            print(f"  retry {attempt+1} after {wait:.0f}s: {e}", flush=True)
            time.sleep(wait)
    return None


def resolve_topics(pt, df):
    """Map each title to its film topic id; cached in data/topics.csv so picks can be audited/overridden."""
    cache = ROOT / "data" / "topics.csv"
    known = pd.read_csv(cache).set_index("title")["mid"].to_dict() if cache.exists() else {}
    rows = []
    for _, r in df.iterrows():
        if r.title not in known:
            name = r.title.split(" (")[0]
            sug = pt.suggestions(name)
            films = [x for x in sug if "film" in x["type"].lower() or x["type"] == "Movie"]
            pick = (films or sug or [{"mid": None, "title": "", "type": ""}])[0]
            known[r.title] = pick["mid"]
            rows.append({"title": r.title, "mid": pick["mid"], "topic": pick["title"], "type": pick["type"]})
            time.sleep(2)
    if rows:
        old = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).to_csv(cache, index=False, encoding="utf-8")
    df = df.copy()
    df["google_trends_query"] = df["title"].map(known)
    return df.dropna(subset=["google_trends_query"])


def main():
    pt = TrendReq(hl="en-US", tz=-330, timeout=(10, 25))
    failed = []
    df = resolve_topics(pt, films()) if TOPIC else films()
    for _, r in df.iterrows():
        rel = pd.Timestamp(r.release_date)
        start, end = rel - pd.Timedelta(days=PRE_DAYS), min(rel + pd.Timedelta(days=POST_DAYS), TODAY)
        for geo, gprop, label in CHANNELS:
            slug = "".join(c if c.isalnum() else "_" for c in r.title)[:50]
            out = RAW / f"{slug}__{label}.csv"
            if out.exists():
                continue
            print(f"{r.title} [{label}]", flush=True)
            df = fetch(pt, r.google_trends_query, start, end, geo, gprop)
            if df is None or df.empty:
                failed.append((r.title, label))
                continue
            df = df.reset_index().rename(columns={r.google_trends_query: "film", ANCHOR: "anchor"})
            df["title"], df["channel"], df["release_date"] = r.title, label, rel
            df[["title", "channel", "release_date", "date", "film", "anchor"]].to_csv(out, index=False)
            time.sleep(random.uniform(6, 12))
    all_ = pd.concat([pd.read_csv(p) for p in RAW.glob("*.csv")], ignore_index=True)
    all_.to_csv(ROOT / "data" / ("trends_ww.csv" if WW else "trends_topic.csv" if TOPIC else "trends.csv"), index=False)
    print(f"done; {len(all_)} rows; failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
