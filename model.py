"""Forecast Ramayana: Part 1 opening day and opening weekend from Google Trends.

Pipeline: comp box office (data/boxoffice_verified.csv over the raw agent CSVs)
+ anchored Trends series (data/trends.csv, from fetch_trends.py)
-> features measured as of T-8 (the furthest-out point Ramayana is observed at)
-> log-log ridge regression (+ k-NN cross-check), leave-one-out CV
-> Ramayana day-1 forecast, weekend via day-pattern multipliers, worldwide via ratios.
Writes data/model_output.json for the report.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
D = ROOT / "data"
ASOF = 8                     # features use data up to T-8 only
RAMA = "Ramayana: Part 1"
RAMA_BASE = "Ramayana baseline 2024"
USD_INR = 93.0
# Physical Day-1 ceiling in India (screens from Sacnilk; shows/seats/ticket are assumptions)
# Average ticket price from Sacnilk Day-1 advance (28 Oct): gross excl. blocked seats / tickets sold.
ADVANCE = {"gross_cr_excl_blocked": 11.33, "gross_cr_incl_blocked": 14.05, "tickets": 262_000, "shows": 6134}
ATP_ADVANCE = ADVANCE["gross_cr_excl_blocked"] * 1e7 / ADVANCE["tickets"]          # ~Rs 432
ATP = {"advance": ATP_ADVANCE, "base": ATP_ADVANCE * 0.88, "low": ATP_ADVANCE * 0.76}  # walk-ins/tier-2 dilute it
CAPACITY = {"screens": 9000, "shows_per_screen": 5, "seats_per_show": 250, "avg_ticket_gross_inr": ATP_ADVANCE}
# Overseas screens: Sony Pictures' reported ~50,000 international screens (Variety India / Sacnilk, Aug 2026).
# No per-country split is published. Shows, seats and ticket price are assumptions.
OVERSEAS_CAPACITY = {"screens": 50_000, "shows_per_screen": 5, "seats_per_show": 200, "avg_ticket_usd": 11.0}
GROSS_PER_NET = 1.18          # India gross = net + 18% GST on tickets; every output is gross
rng = np.random.default_rng(7)


# ---------- box office ----------
def load_boxoffice():
    ind = pd.read_csv(D / "films_india.csv")
    hol = pd.read_csv(D / "films_hollywood.csv")
    ind["ww_wknd_cr"] = ind["ww_gross_weekend_cr"]
    hol["ww_wknd_cr"] = hol["ww_gross_weekend_usd_m"] * USD_INR / 10
    cols = ["title", "industry", "release_date", "india_net_day1_cr", "india_net_weekend_cr",
            "ww_wknd_cr", "franchise", "top_star", "holiday_release", "verdict"]
    bo = pd.concat([ind[cols], hol[cols]], ignore_index=True)
    bo["wknd_days"] = np.nan
    v = D / "boxoffice_verified.csv"
    if v.exists():
        ver = pd.read_csv(v).set_index("title")
        for c_src, c_dst in [("india_net_d1", "india_net_day1_cr"), ("india_net_opening_wknd", "india_net_weekend_cr"),
                             ("wknd_days", "wknd_days"), ("verdict", "verdict")]:
            m = bo["title"].map(ver[c_src])
            bo[c_dst] = m.where(m.notna(), bo[c_dst])
        usd = bo["title"].map(ver["ww_gross_opening_wknd_usd_m"]) * USD_INR / 10
        inr = bo["title"].map(ver["ww_gross_opening_wknd_inr_cr"])
        bo["ww_wknd_cr"] = inr.fillna(usd).fillna(bo["ww_wknd_cr"])
        days = ver[["india_net_d1", "india_net_d2", "india_net_d3", "india_net_d4", "release_weekday"]]
        bo = bo.merge(days.reset_index(), on="title", how="left")
    bo = bo[bo["title"] != "Hanuman Ansh"]
    # report everything as gross so India and worldwide sit on one basis
    gross = ["india_net_day1_cr", "india_net_weekend_cr", "india_net_d1", "india_net_d2", "india_net_d3", "india_net_d4"]
    bo[gross] = bo[gross] * GROSS_PER_NET
    bo = bo.rename(columns=lambda c: c.replace("india_net_", "india_gross_"))
    bo["release_date"] = pd.to_datetime(bo["release_date"])
    return bo


# ---------- trends features ----------
def trend_features(tr):
    tr = tr.copy()
    tr["date"], tr["release_date"] = pd.to_datetime(tr["date"]), pd.to_datetime(tr["release_date"])
    tr["t"] = (tr["date"] - tr["release_date"]).dt.days
    rows = []
    for (title, ch), g in tr.groupby(["title", "channel"]):
        a = g.loc[g["t"] <= -ASOF, "anchor"].median()
        g = g.assign(x=g["film"].clip(lower=0.5) / max(a, 1))
        pre = g[g["t"] <= -ASOF]
        rows.append({"title": title, "channel": ch,
                     "mean28": pre.loc[pre["t"] >= -28, "x"].mean(),
                     "peak60": pre["x"].max(),
                     "last7": pre.loc[pre["t"] > -ASOF - 7, "x"].mean()})
    f = pd.DataFrame(rows).pivot(index="title", columns="channel")
    f.columns = [f"{c}_{m}" for m, c in f.columns]
    # Ramayana query is a generic word: subtract its 2024 pre-buzz baseline level
    if RAMA_BASE in f.index and RAMA in f.index:
        for c in f.columns:
            f.loc[RAMA, c] = max(f.loc[RAMA, c] - f.loc[RAMA_BASE, c.split("_")[0] + "_" + c.split("_")[1] + "_mean28"], 0.05)
        f = f.drop(RAMA_BASE)
    return f


# No Indian/Hollywood label anywhere: Ramayana is forecast as a global release, from the numbers alone.
FEATS = ["web_in_mean28", "top_star"]
OS_FEATS = ["web_ww_mean28"]


class Ridge:
    """Standardised ridge regression (numpy only; sklearn DLLs are blocked here)."""
    def __init__(self, alpha): self.alpha = alpha
    def fit(self, X, y):
        X = np.asarray(X, float); self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd; self.b0 = y.mean()
        self.coef_ = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y - self.b0))
        return self
    def predict(self, X): return self.b0 + ((np.asarray(X, float) - self.mu) / self.sd) @ self.coef_


class KNN:
    """Cross-check: mean log-target of the k nearest comps in standardised feature space."""
    def __init__(self, k=5): self.k = k
    def fit(self, X, y):
        X = np.asarray(X, float); self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.Z, self.y = (X - self.mu) / self.sd, np.asarray(y); return self
    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        d = ((Z[:, None, :] - self.Z[None]) ** 2).sum(-1)
        return np.array([self.y[np.argsort(r)[:self.k]].mean() for r in d])


def loo(make, X, y):
    X, y = np.asarray(X, float), np.asarray(y)
    return np.array([make().fit(np.delete(X, i, 0), np.delete(y, i)).predict(X[i:i + 1])[0] for i in range(len(y))])


def logx(df, cols):
    X = df[list(cols)].astype(float).copy()
    for c in cols:
        if c.startswith(("web", "yt")):
            X[c] = np.log(X[c].clip(lower=0.01))
    return X


def fit_eval(df, target, feats=None):
    feats = feats or FEATS
    d = df.dropna(subset=[target] + feats)
    d = d[d[target] > 0]
    y, X = np.log(d[target]).values, logx(d, feats)
    sse = lambda p: ((y - p) ** 2).sum()
    alpha = min(np.logspace(-2, 2, 30), key=lambda a: sse(loo(lambda: Ridge(a), X, y)))
    p_r, p_k = loo(lambda: Ridge(alpha), X, y), loo(KNN, X, y)
    ape = lambda p: float(np.median(np.abs(np.exp(p) - np.exp(y)) / np.exp(y)))
    ridge, knn = Ridge(alpha).fit(X, y), KNN().fit(X, y)
    coefs = dict(zip(feats, ridge.coef_ / ridge.sd))
    back = pd.DataFrame({"title": d["title"].values, "actual": np.exp(y), "pred_ridge": np.exp(p_r), "pred_knn": np.exp(p_k)})
    return ridge, knn, feats, y - p_r, {"n": len(d), "alpha": float(alpha), "mdape_ridge": ape(p_r), "mdape_knn": ape(p_k),
                                         "r2_loo": float(1 - sse(p_r) / ((y - y.mean()) ** 2).sum()),
                                         "coefs": coefs}, back


def predict(model, cols, row, resid, n=4000):
    mu = model.predict(logx(pd.DataFrame([row]), cols))[0]
    sims = np.exp(mu + rng.choice(resid, n))
    return float(np.exp(mu)), np.percentile(sims, [10, 50, 90]).tolist()



# Paid previews as a share of the following Day 1 (India net, Sacnilk): Dhurandhar 2 43/102.55,
# Peddi 18.5/51, Mana Shankara Vara Prasad Garu 9.35/32.25, The RajaSaab 9.15/53.75, Welcome to the Jungle 3.75/15.25
PREVIEW_SHARE = [43 / 102.55, 18.5 / 51, 9.35 / 32.25, 9.15 / 53.75, 3.75 / 15.25]

# ---------- advance-booking model ----------
def advance_model():
    """Day-1 gross = Ramayana's T-8 advance x growth to release morning x walk-in multiple.
    Growth: Dhurandhar 2 (only comp with a 2-week booking window) and Ramayana's own momentum.
    Walk-in: log(D1 gross / final advance) regressed on log(final advance), LOO-tested."""
    a = pd.read_csv(D / "advance_bookings.csv")
    a = a[a["blocked_seats"].isin(["incl", "unknown"])]
    ram = pd.read_csv(D / "ramayana_advance.csv").sort_values("days_before_release")
    now = ram.iloc[0]
    # last-day jump (T-1 -> T0), incl. blocked seats, median across comps
    inc = a[a["blocked_seats"] == "incl"]
    jumps = []
    for t, g in inc.groupby("title"):
        g = g.groupby("days_before_release")["advance_gross_cr"].max()
        if 0 in g and 1 in g:
            jumps.append(g[0] / g[1])
    jump = float(np.median(jumps))
    # Dhurandhar 2: T-10 -> T-5 daily rate, T-10 -> T-1 level; rescaled to start at T-8
    d2 = inc[inc["title"].str.startswith("Dhurandhar: The")].groupby("days_before_release")["advance_gross_cr"].max()
    d2_rate = (d2[5] / d2[10]) ** (1 / 5)
    g_d2 = d2[1] / d2[10] / d2_rate ** 2 * jump
    # Ramayana's own momentum T-13 -> T-8, held to T-1, then the last-day jump
    r = ram.set_index("days_before_release")["advance_gross_cr_incl_blocked"]
    r_rate = (r[8] / r[13]) ** (1 / 5)
    g_own = r_rate ** 7 * jump
    # Ramayana's own booking momentum is the base; Dhurandhar 2's pace is the low case.
    jump_hi = float(np.percentile(jumps, 75))
    growth = {"low": float(g_d2), "base": float(g_own), "high": float(r_rate ** 7 * jump_hi)}
    final_adv = {k: now["advance_gross_cr_incl_blocked"] * v for k, v in growth.items()}

    # walk-in multiple from comps' final (release-morning) advance to actual Day-1 gross
    fin = a[(a["days_before_release"] == 0)].groupby("title").agg(adv=("advance_gross_cr", "max"),
                                                                  d1=("actual_day1_india_gross_cr", "first")).dropna()
    fin = fin[~fin.index.str.startswith("Drishyam")]          # blocked-seat basis unknown
    x, y = np.log(fin["adv"].values), np.log(fin["d1"] / fin["adv"]).values
    fit = lambda xs, ys: np.polyfit(xs, ys, 1)
    loo = np.array([np.polyval(fit(np.delete(x, i), np.delete(y, i)), x[i]) for i in range(len(x))])
    resid = (y - loo)
    slope, icpt = fit(x, y)
    wm = fin["d1"] / fin["adv"]                 # comps' walk-in multiples, no size adjustment
    walk = {"low": float(wm.quantile(.25)), "base": float(wm.median()), "high": float(wm.quantile(.75))}
    # 3: Wed 4 Nov paid previews folded into Day 1, as Sacnilk does and as the comps' Day 1 figures are
    prev = pd.Series(PREVIEW_SHARE)
    pv = {"low": float(prev.quantile(.25)), "base": float(prev.median()), "high": float(prev.quantile(.75))}
    d1 = {k: final_adv[k] * walk[k] * (1 + pv[k]) for k in final_adv}
    # log-sd: walk-in LOO error plus growth uncertainty (low-high spread as +-1 sd)
    sd = float(np.sqrt(resid.var() + (np.log(growth["high"] / growth["low"]) / 2) ** 2))
    return {"ramayana_t8_incl": float(now["advance_gross_cr_incl_blocked"]), "last_day_jump": jump,
            "jump_n": len(jumps), "d2_growth_from_t8": float(g_d2), "own_momentum_growth": float(g_own),
            "ramayana_daily_rate": float(r_rate), "growth": growth, "final_advance": final_adv,
            "walkin_fit": {"slope": float(slope), "intercept": float(icpt), "n": len(x),
                           "loo_mdape": float(np.median(np.abs(np.exp(resid) - 1)))},
            "walkin": walk, "preview_share": pv, "d1": d1, "log_sd": sd,
            "comps": fin.assign(walkin=fin["d1"] / fin["adv"]).round(2).reset_index().to_dict("records")}


# ---------- Hollywood tentpole model (the headline forecast) ----------
def hollywood_model(bo):
    """Ramayana forecast the way a Hollywood tentpole is: worldwide opening from worldwide search
    interest, fitted on Hollywood comps only; Day 1 and the domestic/international split from
    Hollywood openings. No Indian comp, booking or walk-in input."""
    gw = trend_features(pd.read_csv(D / "trends_ww.csv"))
    h = bo[bo["industry"] == "Hollywood"].merge(gw, left_on="title", right_index=True)
    h["wknd_days"] = h["wknd_days"].fillna(3)
    h["ww_per_day_cr"] = h["ww_wknd_cr"] / h["wknd_days"]
    ridge, knn, cols, resid, stats, back = fit_eval(h, "ww_per_day_cr", OS_FEATS)
    rama = gw.loc[RAMA].to_dict()
    point, (p10, p50, p90) = predict(ridge, cols, rama, resid)
    per_day = {"low": p10, "base": p50, "high": p90}
    days = 4                                              # Thu 5 Nov - Sun 8 Nov
    # Opening day (previews included) as a share of the Fri-Sun 3-day, Friday-opening Hollywood comps
    od = pd.read_csv(D / "hollywood_opening_days.csv").dropna(subset=["opening_day_usd_m_incl_previews", "dom_3day_usd_m"])
    od = od[od["title"] != "Zootopia 2"]                  # Wednesday opener
    sh = od["opening_day_usd_m_incl_previews"] / od["dom_3day_usd_m"]
    d1_share = {"low": float(sh.quantile(.25)), "base": float(sh.median()), "high": float(sh.quantile(.75))}
    hol = pd.read_csv(D / "films_hollywood.csv")
    dom = (hol["dom_weekend_usd_m"] / hol["ww_gross_weekend_usd_m"]).dropna()
    dom_share = float(dom.median())                       # home market's share of the global opening
    ww_d1 = {k: per_day[k] * 3 * d1_share[k] for k in per_day}
    ww_wknd = {k: per_day[k] * days for k in per_day}
    # India can't sell more seats than its screens hold: cap each day at the physical ceiling
    c = CAPACITY
    ceiling = c["screens"] * c["shows_per_screen"] * c["seats_per_show"] * c["avg_ticket_gross_inr"] / 1e7
    india_d1 = {k: min(v * dom_share, ceiling) for k, v in ww_d1.items()}
    india_wknd = {k: min(v * dom_share, ceiling * days) for k, v in ww_wknd.items()}
    o = OVERSEAS_CAPACITY
    os_ceiling = o["screens"] * o["shows_per_screen"] * o["seats_per_show"] * o["avg_ticket_usd"] * USD_INR / 1e7
    os_d1 = {k: min(v * (1 - dom_share), os_ceiling) for k, v in ww_d1.items()}
    os_wknd = {k: min(v * (1 - dom_share), os_ceiling * days) for k, v in ww_wknd.items()}
    f = {"ww_d1": {k: india_d1[k] + os_d1[k] for k in ww_d1}, "ww_wknd": {k: india_wknd[k] + os_wknd[k] for k in ww_wknd},
         "india_d1": india_d1, "india_wknd": india_wknd, "overseas_d1": os_d1, "overseas_wknd": os_wknd}
    capped = {k: india_d1[k] < ww_d1[k] * dom_share - 1e-9 for k in ww_d1}
    return f, {**stats, "ridge_point_per_day": point, "per_day": per_day, "days": days,
               "india_ceiling_per_day": ceiling, "india_capped": capped, "india_screens": c["screens"],
               "overseas_ceiling_per_day": os_ceiling, "overseas_screens": o["screens"],
               "overseas_capped": {k: os_d1[k] < ww_d1[k] * (1 - dom_share) - 1e-9 for k in ww_d1},
               "d1_share": d1_share, "d1_share_n": len(sh), "d1_share_comps": od.assign(share=sh)[["title", "share"]].round(3).to_dict("records"),
               "dom_share": dom_share, "dom_share_n": len(dom), "ramayana_features": {k: float(v) for k, v in rama.items()},
               "backtest": back.round(2).to_dict("records")}


def main():
    bo = load_boxoffice()
    tf = D / "trends_topic.csv" if (D / "trends_topic.csv").exists() else D / "trends.csv"
    feats = trend_features(pd.read_csv(tf))
    df = bo.merge(feats, left_on="title", right_index=True, how="inner")
    df["is_indian"] = (df["industry"] != "Hollywood").astype(int)
    # Indian comps' 3-day weekend multiple on day 1, for converting day 1 -> weekend
    ind = df[df["is_indian"] == 1].copy()
    out = {"features": FEATS, "asof_days": ASOF}

    rama = {**feats.loc[RAMA].to_dict(), "top_star": 1, "franchise": 1, "is_indian": 1}
    out["ramayana_features"] = {k: float(v) for k, v in rama.items()}

    for target in ["india_gross_day1_cr"]:
        ridge, knn, cols, resid, stats, back = fit_eval(df, target)
        point, (p10, p50, p90) = predict(ridge, cols, rama, resid)
        out[target] = {**stats, "ridge_point": point, "p10": p10, "p50": p50, "p90": p90,
                       "knn_point": float(np.exp(knn.predict(logx(pd.DataFrame([rama]), cols))[0])),
                       "backtest": back.round(2).to_dict("records")}

    # Thu->Sun pattern from Thursday-release event films (Ramayana opens Thu 5 Nov).
    # Sunday 8 Nov is Diwali: Lakshmi Puja evening empties night shows, so Sunday
    # is scaled by DIWALI_SUN (low/base/high) on top of the comps' normal Sunday.
    thu = df[(df["release_weekday"] == "Thursday") & (df["india_gross_d1"] > 5) & (df["title"] != "Peddi")].dropna(subset=["india_gross_d4"])
    r = {f"d{k}": (thu[f"india_gross_d{k}"] / thu["india_gross_d1"]).median() for k in (2, 3, 4)}
    DIWALI_SUN = {"low": 1.0, "base": 1.0, "high": 1.0}
    rows = (thu[[f"india_gross_d{k}" for k in (2, 3, 4)]].div(thu["india_gross_d1"], axis=0)).sum(axis=1) + 1
    mult = {"low": float(rows.min()), "base": float(rows.median()), "high": float(rows.max())}
    out["day_ratios"] = {**r, "comps": thu["title"].tolist(), "diwali_sunday_factor": DIWALI_SUN,
                         "weekend_multiple": mult}

    # Final Day 1 = inverse-variance blend (log space) of the Trends model and the advance model.
    tr = out["india_gross_day1_cr"]
    am = out["advance_model"] = advance_model()
    mu_t, sd_t = np.log(tr["p50"]), (np.log(tr["p90"]) - np.log(tr["p10"])) / (2 * 1.2816)
    mu_a, sd_a = np.log(am["d1"]["base"]), am["log_sd"]
    w_t, w_a = 0.0, 1.0       # India Day 1 from Ramayana's own bookings; search model shown for reference only
    mu, sd = mu_a, sd_a
    out["blend"] = {"trends_weight": w_t / (w_t + w_a), "advance_weight": w_a / (w_t + w_a),
                    "trends_p50": tr["p50"], "advance_base": am["d1"]["base"], "log_sd": sd}
    blended = {"p10": am["d1"]["low"], "p50": am["d1"]["base"], "p90": am["d1"]["high"]}
    c = CAPACITY
    ceiling = c["screens"] * c["shows_per_screen"] * c["seats_per_show"] * c["avg_ticket_gross_inr"] / 1e7
    out["capacity"] = {**c, "ceiling_gross_cr": ceiling, "atp_scenarios": ATP, "advance": ADVANCE}
    d1 = {k: min(v, ceiling) for k, v in blended.items()}
    out["capacity"]["capped"] = blended["p90"] > ceiling
    out["forecast"] = {
        "india_d1": {"low": d1["p10"], "base": d1["p50"], "high": d1["p90"]},
        "india_wknd": {"low": d1["p10"] * mult["low"], "base": d1["p50"] * mult["base"], "high": d1["p90"] * mult["high"]},
    }

    ww = df.dropna(subset=["ww_wknd_cr", "india_gross_weekend_cr"])
    ww = ww[ww["is_indian"] == 1]
    out["ww_ratio"] = {"median": float((ww["ww_wknd_cr"] / ww["india_gross_weekend_cr"]).median()),
                       "p25": float((ww["ww_wknd_cr"] / ww["india_gross_weekend_cr"]).quantile(.25)),
                       "p75": float((ww["ww_wknd_cr"] / ww["india_gross_weekend_cr"]).quantile(.75)),
                       "n": len(ww)}
    # Overseas modelled on its own (Ramayana is a global day-and-date release, not a Bollywood
    # film with a fixed overseas add-on): outside-India opening gross per day vs worldwide Trends.
    f = out["forecast"]
    wwf = D / "trends_ww.csv"
    if wwf.exists():
        gw = trend_features(pd.read_csv(wwf))[["web_ww_mean28", "web_ww_peak60"]]
        dd = df.merge(gw, left_on="title", right_index=True)
        dd["wknd_days"] = dd["wknd_days"].fillna(3)
        dd["overseas_per_day_cr"] = (dd["ww_wknd_cr"] - dd["india_gross_weekend_cr"]) / dd["wknd_days"]
        ridge, knn, cols, resid, stats, back = fit_eval(dd[dd["overseas_per_day_cr"] > 0], "overseas_per_day_cr",
                                                        OS_FEATS)
        rama_ww = gw.loc[RAMA].to_dict()
        point, (p10, p50, p90) = predict(ridge, cols, rama_ww, resid)
        days = 4                                  # Thu 5 Nov - Sun 8 Nov
        D1_SHARE = {"low": 0.30, "base": 0.25, "high": 0.20}   # Thu (+Wed previews) share of the 4 days
        # Overseas pre-sales are hard evidence: Ramayana $2M+ for opening day ~T-10 (Sacnilk, 25-26 Oct).
        # Dhurandhar 2 converted $1.2M at ~T-6 into Rs 52-64 cr overseas Day 1 -> multiple 4.9-6.0x.
        # Measured from T-6, so applied to Ramayana's earlier T-10 figure it understates, never overstates.
        OS_ADV = {"ramayana_usd_m": 0.95, "d2_usd_m": 1.2, "d2_day1_cr": (52.0, 64.0)}
        # Ramayana overseas pre-sales ~$0.95M at ~T-10 (Sacnilk: $400K opening day + $550K NA previews).
        # D2's multiple is from T-6, so first grow Ramayana's figure 4 days at its own booking rate.
        adv_cr = OS_ADV["ramayana_usd_m"] * USD_INR / 10 * am["ramayana_daily_rate"] ** 4
        mult = [d / (OS_ADV["d2_usd_m"] * USD_INR / 10) for d in OS_ADV["d2_day1_cr"]]
        mu_a = np.log(adv_cr * np.sqrt(mult[0] * mult[1]))
        sd_a = np.sqrt((np.log(mult[1] / mult[0]) / 2) ** 2 + 0.30 ** 2)
        mu_t = np.log(p50 * days * D1_SHARE["base"])
        sd_t = (np.log(p90) - np.log(p10)) / (2 * 1.2816)
        w_t, w_a = 0.5, 0.5        # global search and overseas pre-sales count equally
        mu, sd = w_t * mu_t + w_a * mu_a, np.sqrt((w_t * sd_t) ** 2 + (w_a * sd_a) ** 2)
        os_d1 = {"low": float(np.exp(mu - 1.2816 * sd)), "base": float(np.exp(mu)), "high": float(np.exp(mu + 1.2816 * sd))}
        os_wknd = {k: os_d1[k] / D1_SHARE[k] for k in os_d1}
        out["overseas_blend"] = {"advance_inputs": OS_ADV, "advance_cr_at_t6": adv_cr, "d2_multiple": mult, "advance_d1": float(np.exp(mu_a)),
                                 "trends_d1": float(np.exp(mu_t)), "trends_weight": w_t / (w_t + w_a),
                                 "advance_weight": w_a / (w_t + w_a)}
        out["overseas"] = {**stats, "ridge_point_per_day": point, "p10": p10, "p50": p50, "p90": p90,
                           "ramayana_features": {k: float(v) for k, v in rama_ww.items()},
                           "d1_share": D1_SHARE, "backtest": back.round(2).to_dict("records")}
        f["overseas_wknd"] = os_wknd
        f["overseas_d1"] = os_d1
        for k in ("d1", "wknd"):
            f[f"ww_{k}"] = {s: f[f"india_{k}"][s] + f[f"overseas_{k}"][s] for s in ("low", "base", "high")}
        print(f"Overseas/day: n={stats['n']} MdAPE={stats['mdape_ridge']:.0%} R2={stats['r2_loo']:.2f} rama/day={p50:.1f}")
    else:
        wr = out["ww_ratio"]
        for k in ("d1", "wknd"):
            f[f"ww_{k}"] = {"low": f[f"india_{k}"]["low"] * wr["p25"], "base": f[f"india_{k}"]["base"] * wr["median"],
                            "high": f[f"india_{k}"]["high"] * wr["p75"]}
    # Headline = Hollywood tentpole model; the bookings-based forecast above is kept for reference only.
    out["india_style_reference"] = out.pop("forecast")
    out["forecast"], out["hollywood"] = hollywood_model(bo)
    hs = out["hollywood"]
    print(f"Hollywood WW/day: n={hs['n']} MdAPE={hs['mdape_ridge']:.0%} R2={hs['r2_loo']:.2f} "
          f"per-day={hs['per_day']} d1_share={hs['d1_share']} dom_share={hs['dom_share']:.2f}")
    out["comp_table"] = df[["title", "industry", "release_date", "verdict", "india_gross_day1_cr",
                            "india_gross_weekend_cr", "ww_wknd_cr", "web_in_mean28", "web_in_peak60", "web_in_last7"]].assign(
                            release_date=lambda x: x.release_date.dt.strftime("%Y-%m-%d")).round(3).to_dict("records")
    (D / "model_output.json").write_text(json.dumps(out, indent=2, default=float))
    s = out["india_gross_day1_cr"]
    print(f"India D1: n={s['n']} LOO MdAPE ridge={s['mdape_ridge']:.0%} knn={s['mdape_knn']:.0%} R2={s['r2_loo']:.2f}")
    print(f"Ramayana D1 ridge={s['ridge_point']:.1f} knn={s['knn_point']:.1f} P10/50/90={s['p10']:.1f}/{s['p50']:.1f}/{s['p90']:.1f}")
    print("day ratios", out["day_ratios"], "ww ratio", out["ww_ratio"])
    print("advance model", {k: out["advance_model"][k] for k in ("growth", "final_advance", "walkin", "d1", "log_sd", "walkin_fit")})
    print("blend", out["blend"]); print("overseas blend", out.get("overseas_blend"))
    for k, v in out["forecast"].items():
        print(f"  {k:11s} low {v['low']:7.1f}  base {v['base']:7.1f}  high {v['high']:7.1f}")
    print("coefs", {k: round(v, 3) for k, v in s["coefs"].items()})
    print("rama feats", {k: round(v, 3) for k, v in out["ramayana_features"].items()})


if __name__ == "__main__":
    main()
