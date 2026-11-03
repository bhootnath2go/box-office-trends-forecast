"""Fetch daily Wikipedia pageviews (absolute counts, all language editions that have the article)
for every comp film and Ramayana, T-60 to T+3. Unlike Google Trends these are real view counts,
so films are directly comparable without an anchor term.

Article titles are resolved by English Wikipedia search and cached in data/wiki_titles.csv
(check new rows by hand). Language editions come from the article's interlanguage links.
Writes data/wiki_views.csv: title, lang, date, views.
"""
import time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).parent
D = ROOT / "data"
UA = {"User-Agent": "box-office-trends-forecast/1.0 (https://github.com/bhootnath2go/box-office-trends-forecast)"}
LANGS = ["en", "hi", "te", "ta", "kn", "ml", "es", "fr", "de", "pt", "ja", "zh", "ar"]
TODAY = pd.Timestamp("2026-11-03")
PRE_DAYS, POST_DAYS = 60, 3


def films():
    frames = [pd.read_csv(D / f)[["title", "release_date"]] for f in ("films_india.csv", "films_hollywood.csv", "ramayana.csv")]
    return pd.concat(frames, ignore_index=True).dropna().drop_duplicates("title")


def resolve(title, year):
    r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=20, params={
        "action": "query", "list": "search", "srsearch": f"{title} {year} film", "format": "json", "srlimit": 5}).json()
    hits = [h["title"] for h in r["query"]["search"]]
    films_ = [h for h in hits if "film" in h.lower() or title.lower().split(":")[0] in h.lower()]
    return (films_ or hits or [None])[0]


def langlinks(en_title):
    r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=20, params={
        "action": "query", "prop": "langlinks", "titles": en_title, "lllimit": 500, "format": "json", "redirects": 1}).json()
    page = next(iter(r["query"]["pages"].values()))
    links = {"en": page.get("title", en_title)}
    for l in page.get("langlinks", []):
        if l["lang"] in LANGS:
            links[l["lang"]] = l["*"]
    return links


def views(lang, article, start, end):
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/"
           f"{requests.utils.quote(article.replace(' ', '_'), safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}")
    r = requests.get(url, headers=UA, timeout=20)
    if r.status_code != 200:
        return []
    return [(pd.Timestamp(i["timestamp"][:8]), i["views"]) for i in r.json().get("items", [])]


def main():
    cache = D / "wiki_titles.csv"
    known = pd.read_csv(cache).set_index("title")["article"].to_dict() if cache.exists() else {}
    rows, out = [], []
    for _, f in films().iterrows():
        rel = pd.Timestamp(f.release_date)
        if f.title not in known:
            known[f.title] = resolve(f.title, rel.year)
            rows.append({"title": f.title, "article": known[f.title]})
        art = known[f.title]
        if not isinstance(art, str):
            print("no article:", f.title); continue
        start, end = rel - pd.Timedelta(days=PRE_DAYS), min(rel + pd.Timedelta(days=POST_DAYS), TODAY)
        for lang, a in langlinks(art).items():
            for d, v in views(lang, a, start, end):
                out.append({"title": f.title, "release_date": rel, "lang": lang, "date": d, "views": v})
            time.sleep(0.2)
        print(f.title, "->", art, flush=True)
    if rows:
        old = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).to_csv(cache, index=False, encoding="utf-8")
    pd.DataFrame(out).to_csv(D / "wiki_views.csv", index=False)
    print("rows", len(out))


if __name__ == "__main__":
    main()
