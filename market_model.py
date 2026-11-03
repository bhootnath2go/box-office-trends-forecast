"""Market-by-market sentiment model.

Opening gross per day in market m for film f:
    log(open_fm) = a_m + b_m * log(search_fm) + c * log(momentum_fm) + d * log(wiki_f)
- search_fm: Google Trends film topic in that market's country, 28-day mean to T-ASOF, as a multiple of an anchor term
  (Rest of world: Wikipedia views in es/fr/de/pt/ja/zh/ar, since worldwide search for an Indian-audience film
  is mostly searches from India)
- momentum_fm: last-7-day mean / 28-day mean (rising into release vs peaked at the trailer)
- wiki_f: Wikipedia pageviews, all language editions, daily mean T-28..T-ASOF (absolute counts)
- a_m, b_m: conversion level and search response per market (ticket prices, moviegoing habit, anchor-term
  popularity). A per-market slope beat one shared slope in the leave-one-out test (41% vs 54% median error).
Hollywood and Indian films are pooled: films differ only by where their audience searches.
Leave-one-film-out tested. Writes data/market_model.json.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import model as M

D = Path(__file__).parent / "data"
TO_CR = M.USD_INR / 10                     # $M -> Rs crore
MARKETS = {                                # market: (search channel, target column)
    "India": ("web_in", "india_opening_wknd_gross_usd_m"),
    "North America": ("web_us", "na_opening_wknd_usd_m"),
    "UK": ("web_gb", "uk_opening_wknd_usd_m"),
    "Gulf": ("web_ae", "gulf_opening_wknd_usd_m"),
    "Australia/NZ": ("web_au", "aus_nz_opening_wknd_usd_m"),
    "Rest of world": ("web_ww", "row_usd_m"),
}
NAMED = ["na_opening_wknd_usd_m", "uk_opening_wknd_usd_m", "gulf_opening_wknd_usd_m", "aus_nz_opening_wknd_usd_m"]
DAYS = 4                                   # Ramayana: Thu 5 - Sun 8 Nov
Z90 = 1.2816


ROW_LANGS = ["es", "fr", "de", "pt", "ja", "zh", "ar"]


def wiki_feature(langs=None):
    w = pd.read_csv(D / "wiki_views.csv", parse_dates=["date", "release_date"])
    w["t"] = (w["date"] - w["release_date"]).dt.days
    w = w[(w["t"] <= -M.ASOF) & (w["t"] >= -28)]
    if langs:
        w = w[w["lang"].isin(langs)]
    return w.groupby(["title", "date"])["views"].sum().groupby("title").mean().rename("wiki")


def build():
    tr = pd.concat([pd.read_csv(D / f) for f in ("trends_topic.csv", "trends_ww.csv", "trends_geo.csv")])
    tf = M.trend_features(tr)
    wk, row = wiki_feature(), wiki_feature(ROW_LANGS)
    mo = pd.read_csv(D / "market_openings.csv")
    mo["row_usd_m"] = mo["overseas_ex_india_opening_wknd_usd_m"] - mo[NAMED].sum(axis=1, min_count=4)
    rows = []
    films = sorted(set(tf.index) & set(wk.index))
    for f in films:
        o = mo[mo["title"] == f]
        days = float(o["wknd_days"].iloc[0]) if len(o) and pd.notna(o["wknd_days"].iloc[0]) else np.nan
        for m, (ch, col) in MARKETS.items():
            s, l7 = tf.loc[f].get(f"{ch}_mean28", np.nan), tf.loc[f].get(f"{ch}_last7", np.nan)
            mom = l7 / s if s else np.nan
            if m == "Rest of world":
                s = row.get(f, 0) + 1                      # no article in those languages = ~no interest there
            y = o[col].iloc[0] / days if len(o) and pd.notna(o[col].iloc[0]) and o[col].iloc[0] > 0 else np.nan
            rows.append({"title": f, "market": m, "search": s, "momentum": mom, "ww_search": tf.loc[f].get("web_ww_mean28", np.nan),
                         "wiki": wk[f], "y": y})
    return pd.DataFrame(rows)


def design(d, markets):
    s = np.log(d["search"].clip(lower=0.01))
    X = np.column_stack([s * (d["market"] == m) for m in markets]
                        + [np.log(d["momentum"].clip(lower=0.05)), np.log(d["wiki"])]
                        + [(d["market"] == m).astype(float) for m in markets])
    return X


def fit(d, markets, alpha=1.0):
    X, y = design(d, markets), np.log(d["y"].values)
    P = np.diag([alpha] * (len(markets) + 2) + [1e-6] * len(markets))   # shrink slopes only, not the market levels
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def main():
    df = build()
    df = df.dropna(subset=["search", "momentum", "wiki"])
    markets = list(MARKETS)
    train = df[(df["title"] != M.RAMA) & df["y"].notna()]
    films = train["title"].unique()
    # leave one film out: refit without all of that film's markets, predict all its markets
    alpha = 1.0
    pred = []
    for f in films:
        b = fit(train[train["title"] != f], markets, alpha)
        g = df[df["title"] == f]
        pred.append(g.assign(pred=np.exp(design(g, markets) @ b)))
    pr = pd.concat(pred)
    mo = pd.read_csv(D / "market_openings.csv").set_index("title")
    # worldwide per day: actual = India + overseas ex-India; predicted = sum of all six markets
    g = pr.groupby("title")["pred"]
    tot = g.sum()[g.count() == len(markets)].to_frame()        # only films with every market's search data
    tot["actual"] = (mo["india_opening_wknd_gross_usd_m"] + mo["overseas_ex_india_opening_wknd_usd_m"]) / mo["wknd_days"]
    tot = tot.dropna()
    tot["err"] = np.log(tot["actual"] / tot["pred"])
    hol = set(pd.read_csv(D / "films_hollywood.csv")["title"])
    tot["industry"] = np.where(tot.index.isin(hol), "Hollywood", "Indian")
    # baseline on the same films: one worldwide search number -> worldwide opening (the old approach, all comps)
    ww = df[df["market"] == "Rest of world"].set_index("title")["ww_search"].reindex(tot.index)
    x, y = np.log(ww.values), np.log(tot["actual"].values)
    tot["base_pred"] = np.exp([np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i]) for i in range(len(x))])
    tot["base_err"] = np.log(tot["actual"] / tot["base_pred"])
    mk = pr.dropna(subset=["y"]).assign(ape=lambda x: (x["pred"] / x["y"] - 1).abs()).groupby("market")["ape"].median()
    ape = lambda s: float(np.median(np.abs(np.exp(-s) - 1)))   # |pred/actual - 1|
    stats = {"n_films": len(tot)}
    for name, col in (("market", "err"), ("worldwide_baseline", "base_err")):
        stats[name] = {"mdape_all": ape(tot[col]), "mdape_hollywood": ape(tot.loc[tot["industry"] == "Hollywood", col]),
                       "mdape_indian": ape(tot.loc[tot["industry"] == "Indian", col])}
    stats["mdape_by_market"] = mk.to_dict()

    # final fit on all comps, Ramayana per market
    b = fit(train, markets, alpha)
    coefs = dict(zip([f"search_{m}" for m in markets] + ["momentum", "wiki"] + [f"level_{m}" for m in markets], b.tolist()))
    r = df[df["title"] == M.RAMA].copy()
    r["per_day_usd_m"] = np.exp(design(r, markets) @ b)
    # The summed markets came in below actuals on past films (median actual/pred 1.44), so centre the base
    # on the model's own leave-one-out errors and measure the range from there.
    bias = float(np.median(tot["err"]))
    r["per_day_usd_m"] *= np.exp(bias)
    r["wknd_cr"] = r["per_day_usd_m"] * DAYS * TO_CR
    q = np.quantile(tot["err"], [0.1, 0.9]) - bias             # film-level LOO errors on the worldwide total
    base = float(r["wknd_cr"].sum())
    print(r[["market", "search", "momentum", "wiki", "per_day_usd_m", "wknd_cr"]].round(3).to_string())
    print("coefs", {k: round(v, 3) for k, v in coefs.items()})
    print("stats", json.dumps(stats, indent=1))
    print(f"Ramayana WW weekend base {base:.0f} cr, P10 {base*np.exp(q[0]):.0f}, P90 {base*np.exp(q[1]):.0f}")
    out = {"coefs": coefs, "stats": stats, "alpha": alpha, "err_q10_q90": q.tolist(), "bias_correction": float(np.exp(bias)),
           "ramayana": r.drop(columns=["y"]).to_dict("records"),
           "backtest": tot.reset_index().round(3).to_dict("records")}
    (D / "market_model.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    main()
