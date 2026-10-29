"""Write the dashboard's CSV datasets from data/model_output.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model as M
R = ROOT / "report"
o = json.loads((ROOT / "data" / "model_output.json").read_text())
f, am, ob, dr = o["forecast"], o["advance_model"], o["overseas_blend"], o["day_ratios"]
lab = {"ww_d1": ("Worldwide", "Day 1"), "ww_wknd": ("Worldwide", "Opening weekend"),
       "india_d1": ("India", "Day 1"), "india_wknd": ("India", "Opening weekend"),
       "overseas_d1": ("Overseas", "Day 1"), "overseas_wknd": ("Overseas", "Opening weekend")}
pd.DataFrame([{"metric": k, "region": r, "window": w, **{s: round(f[k][s], 1) for s in ("low", "base", "high")}}
              for k, (r, w) in lab.items()]).to_csv(R / "forecast.csv", index=False)
drv = [("india_adv_t8", "India Day-1 advance, 28 Oct (T-8)", am["ramayana_t8_incl"]),
       ("india_daily_growth", "Ramayana's booking growth per day, T-13 to T-8", am["ramayana_daily_rate"] - 1),
       ("india_growth", "Growth to release morning at that pace, plus last-day jump", am["growth"]["base"]),
       ("india_final_adv", "Projected India advance by release morning", am["final_advance"]["base"]),
       ("walkin", "Walk-in multiple, final advance to Day 1 (comp median)", am["walkin"]["base"]),
       ("preview_share", "Wed paid previews as a share of Day 1 (comp median)", am["preview_share"]["base"]),
       ("os_presales", "Overseas pre-sales, ~T-10 (USD million)", ob["advance_inputs"]["ramayana_usd_m"]),
       ("os_adv_t6", "Overseas pre-sales grown to T-6 at Ramayana's pace (cr)", ob["advance_cr_at_t6"]),
       ("d2_multiple", "Dhurandhar 2: overseas pre-sales at T-6 to Day 1", float(np.sqrt(np.prod(ob["d2_multiple"])))),
       ("os_adv_d1", "Overseas Day 1 from pre-sales", ob["advance_d1"]),
       ("os_trends_d1", "Overseas Day 1 from worldwide search", ob["trends_d1"]),
       ("wknd_multiple", "India Thu-Sun weekend as multiple of Day 1", dr["weekend_multiple"]["base"]),
       ("atp", "Average ticket price from advance sales (Rs)", o["capacity"]["avg_ticket_gross_inr"])]
pd.DataFrame(drv, columns=["key", "label", "value"]).round(3).to_csv(R / "drivers.csv", index=False)
print(pd.read_csv(R / "forecast.csv").to_string()); print(pd.read_csv(R / "drivers.csv").to_string())

# comp table: opening grosses plus the search features the model uses
ct = pd.DataFrame(o["comp_table"])
fi = M.trend_features(pd.read_csv(ROOT / "data" / "trends_topic.csv"))["web_in_mean28"]
fw = M.trend_features(pd.read_csv(ROOT / "data" / "trends_ww.csv"))["web_ww_mean28"]
ct = ct.rename(columns={"india_gross_day1_cr": "india_d1", "india_gross_weekend_cr": "india_wknd", "ww_wknd_cr": "ww_wknd"})
ct["market"] = np.where(ct["industry"] == "Hollywood", "Hollywood", "India")
ct["search_in"], ct["search_ww"] = ct["title"].map(fi).round(3), ct["title"].map(fw).round(3)
ram = {"title": M.RAMA, "market": "—", "release_date": "2026-11-05", "verdict": "Upcoming",
       "search_in": round(fi[M.RAMA], 3), "search_ww": round(fw[M.RAMA], 3)}
cols = ["title", "market", "release_date", "verdict", "india_d1", "india_wknd", "ww_wknd", "search_in", "search_ww"]
ct = pd.concat([pd.DataFrame([ram]), ct[cols].round({"india_d1": 1, "india_wknd": 1, "ww_wknd": 1})])[cols]
ct.to_csv(R / "comps.csv", index=False)
