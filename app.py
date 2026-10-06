"""Tonight's front desk: an overbooking game on top of a hotel-cancellation model.

The model sits next to the code, in model/: booster.json (XGBoost) + preprocess.json (scaling and one-hot, as plain
numbers), config.json (features,
versions, metrics), README.md (model card) and test_bookings.parquet (real bookings from months the model never
trained on). To use your own model, replace those files with the ones the session 10 notebook writes.
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from portable import Model

MODEL_DIR = Path(__file__).parent / "model"
NIGHTS_PER_GAME, GUT_CANCEL_RATE, MAX_EXTRA = 5, 0.37, 40
GIF = {k: f"https://media.giphy.com/media/{v}/giphy.gif" for k, v in {
    "welcome": "MCudzuADLuJWw", "thinking": "WRQBXSCnEFJIuxktnw", "fire": "Z1BTGhofioRxK",
    "empty": "3oriff4xQ7Oq2TIgTu", "nailed": "8VrtCswiLDNnO", "cheers": "DfLwM9kttDFEQ",
    "pikachu": "6nWhy3ulBL7GSCvKw6", "win": "3oEduKVQdG4c0JVPSo"}.items()}
LABEL = {"lead_time": "booked {v:.0f} days ahead", "total_nights": "{v:.0f} nights", "adr": "{v:.0f} EUR a night",
         "previous_cancellations": "{v:.0f} earlier cancellations", "previous_bookings_not_canceled": "{v:.0f} earlier stays",
         "is_repeated_guest": "repeat guest: {v:.0f}", "deposit_type": "deposit: {v}", "market_segment": "segment: {v}",
         "customer_type": "customer: {v}", "agent": "agent {v}", "distribution_channel": "channel: {v}",
         "arrival_weekday": "arrival weekday {v:.0f}"}

st.set_page_config(page_title="Tonight's front desk", page_icon="🛎️", layout="wide")


# ---------------------------------------------------------------- the model, loaded once per server
@st.cache_resource
def load():
    get = lambda f: MODEL_DIR / f
    pipe = Model(MODEL_DIR)                 # no pickle: loads with any recent pandas/xgboost
    config = json.load(open(get("config.json")))
    card = open(get("README.md")).read().split("---", 2)[-1].replace("\n# ", "\n#### ")
    bookings = pd.read_parquet(get("test_bookings.parquet"))
    features = config["numeric_features"] + config["categorical_features"]
    bookings["p"] = pipe.predict_proba(bookings[features])
    city = bookings[bookings["hotel"] == "City Hotel"]
    sizes = city.groupby("arrival_date").size()
    nights = sizes[(sizes >= 60) & (sizes <= 180)].index
    return pipe, config, card, city, nights, features


def reasons(pipe, features, rows, top=2):
    """SHAP contributions from XGBoost itself, added back up from one-hot columns to the original features."""
    by_f = pipe.contributions(rows[features]).reset_index(drop=True)
    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        best = by_f.iloc[i].sort_values(ascending=False).head(top)
        out.append(" · ".join(f"{LABEL.get(f, f + ': {v}').format(v=row[f])} ↑" for f, c in best.items() if c > 0))
    return out


def play_night(p, rooms, price, walk_cost, sims=5000, seed=0):
    """Expected profit of selling k extra rooms, from 5,000 simulated nights drawn from the probabilities."""
    shows = (np.random.default_rng(seed).random((sims, len(p))) > p).sum(axis=1)
    k = np.arange(MAX_EXTRA + 1)[:, None]
    profit = price * np.minimum(shows + k, rooms) - walk_cost * np.maximum(shows + k - rooms, 0)
    return profit.mean(axis=1), shows


def outcome(shows, k, rooms, price, walk_cost):
    guests = shows + k
    return price * min(guests, rooms) - walk_cost * max(guests - rooms, 0), max(guests - rooms, 0), max(rooms - guests, 0)


pipe, config, card, city, nights, features = load()

with st.sidebar:
    st.header("🏨 Hotel economics")
    price = st.slider("Price of an extra room (EUR)", 60, 200, 110, 10)
    walk_cost = st.slider("Cost of walking a guest (EUR)", 100, 600, 250, 25,
                          help="Taxi, a room in another hotel, and an angry review.")
    hints = st.toggle("Show the model's forecast", value=True, help="Hard mode: play on gut feeling alone.")
    st.caption(f"Model: `model/booster.json` · version {config['version']} · test AUC {config['metrics_test']['auc']}")

st.title("🛎️ Tonight's front desk")
tab_game, tab_check, tab_hood = st.tabs(["🎲 Play a night", "🔎 Check a booking", "⚙️ Under the hood"])

# ---------------------------------------------------------------- the game
with tab_game:
    if "game" not in st.session_state:
        seed = int(np.random.default_rng().integers(1e9))
        pick = np.random.default_rng(seed).choice(len(nights), NIGHTS_PER_GAME, replace=False)
        st.session_state.game = {"nights": [nights[i] for i in sorted(pick)], "round": 0, "results": [], "revealed": False}
    g = st.session_state.game

    if g["round"] == 0 and not g["results"] and not g["revealed"]:
        left, right = st.columns([2, 1])
        left.markdown(f"""
You run the front desk of a city hotel in Lisbon. Every night some guests cancel, so you can sell a few rooms
twice. Sell too few and rooms stay empty; sell too many and guests arrive to a full hotel and have to be
**walked** to another hotel, at your expense.

Each of the **{NIGHTS_PER_GAME} nights** is a real night from 2017. A tour operator wants extra rooms at
**{price} EUR** each. Walking a guest costs **{walk_cost} EUR**. How many extra rooms do you sell?

You play against two colleagues: **the model**, which knows each booking's cancellation risk, and
**Gut Feeling Gus**, who assumes 37 % of bookings always cancel.
""")
        right.image(GIF["welcome"], caption="via GIPHY")

    if g["round"] < NIGHTS_PER_GAME:
        date = g["nights"][g["round"]]
        night = city[city["arrival_date"] == date].sort_values("p", ascending=False)
        n = len(night); rooms = int(round(0.72 * n))
        exp_profit, sim_shows = play_night(night["p"].values, rooms, price, walk_cost, seed=g["round"])
        k_model = int(np.argmax(exp_profit))
        k_gut = int(max(0, min(MAX_EXTRA, round(rooms - n * (1 - GUT_CANCEL_RATE)))))

        st.subheader(f"Night {g['round'] + 1} of {NIGHTS_PER_GAME} · {pd.Timestamp(date):%A %d %B %Y}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bookings due tonight", n)
        c2.metric("Free rooms", rooms)
        c3.metric("Bookings more than rooms", n - rooms)

        if hints:
            lo, hi = np.percentile(n - sim_shows, [10, 90])
            st.info(f"🤖 **Model forecast:** about **{night['p'].sum():.0f} cancellations** tonight "
                    f"(80 % range {lo:.0f}–{hi:.0f}), so about **{n - night['p'].sum():.0f} guests** will show up "
                    f"for {rooms} rooms.")
            with st.expander("The riskiest bookings tonight, and why"):
                top = night.head(12).copy()
                top["why"] = reasons(pipe, features, top)
                top["risk"] = (100 * top["p"]).round()
                st.dataframe(top[["risk", "lead_time", "deposit_type", "market_segment", "adr", "why"]],
                             column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%d%%"),
                                            "lead_time": "days ahead", "deposit_type": "deposit", "market_segment": "segment",
                                            "adr": st.column_config.NumberColumn("EUR/night", format="%.0f")},
                             hide_index=True, width="stretch")
        else:
            st.warning("🙈 Hard mode: no forecast. Gut feeling only.")

        k = st.slider("Extra rooms to sell tonight", 0, MAX_EXTRA, 0, key=f"k{g['round']}", disabled=g["revealed"])

        if not g["revealed"]:
            if st.button("🚪 Open the doors", type="primary"):
                g["revealed"] = True
                st.rerun()
            st.image(GIF["thinking"], width=260, caption="via GIPHY")
        else:
            shows = int((night["is_canceled"] == 0).sum())
            you, walked, empty = outcome(shows, k, rooms, price, walk_cost)
            model, _, _ = outcome(shows, k_model, rooms, price, walk_cost)
            gut, _, _ = outcome(shows, k_gut, rooms, price, walk_cost)
            if len(g["results"]) == g["round"]:
                g["results"].append({"night": f"{pd.Timestamp(date):%d %b}", "you": you, "model": model, "gut": gut,
                                     "k_you": k, "k_model": k_model, "k_gut": k_gut})
            st.markdown(f"### {shows} guests showed up ({n - shows} cancelled)")
            left, right = st.columns([2, 1])
            with left:
                r1, r2, r3 = st.columns(3)
                r1.metric(f"You: sold {k}", f"{you:,.0f} EUR")
                r2.metric(f"Model: sold {k_model}", f"{model:,.0f} EUR", f"{model - you:+,.0f} vs you", delta_color="inverse")
                r3.metric(f"Gus: sold {k_gut}", f"{gut:,.0f} EUR", f"{gut - you:+,.0f} vs you", delta_color="inverse")
                if walked:
                    st.error(f"😬 {walked} guest{'s' if walked > 1 else ''} arrived to a full hotel: {walked * walk_cost:,} EUR in taxis and apologies.")
                elif empty:
                    st.warning(f"🛏️ {empty} room{'s' if empty > 1 else ''} stayed empty: {empty * price:,} EUR not earned.")
                else:
                    st.success("🎯 Every room full, nobody walked. Perfect night.")
            key = ("fire" if walked >= 5 else "empty" if empty >= 8 else "nailed" if you >= model
                   else "pikachu" if model - you > 3 * price else "cheers")
            right.image(GIF[key], caption="via GIPHY")
            label = "🏁 See the final score" if g["round"] == NIGHTS_PER_GAME - 1 else "➡️ Next night"
            if st.button(label, type="primary"):
                g["round"] += 1; g["revealed"] = False
                st.rerun()

    else:
        res = pd.DataFrame(g["results"])
        tot = res[["you", "model", "gut"]].sum()
        st.subheader("🏁 Final score")
        c1, c2, c3 = st.columns(3)
        c1.metric("You", f"{tot['you']:,.0f} EUR")
        c2.metric("The model", f"{tot['model']:,.0f} EUR")
        c3.metric("Gut Feeling Gus", f"{tot['gut']:,.0f} EUR")
        left, right = st.columns([2, 1])
        left.dataframe(res.rename(columns={"you": "you (EUR)", "model": "model (EUR)", "gut": "Gus (EUR)",
                                           "k_you": "you sold", "k_model": "model sold", "k_gut": "Gus sold"}),
                       hide_index=True, width="stretch")
        left.markdown("""
**What happened?** The model does not know the future. It knows each booking's probability of cancelling,
and because those probabilities are **calibrated**, their sum is a good forecast for the night. From that it
picks the number of extra rooms with the highest *expected* profit. Some nights it loses. Over many nights,
it wins, which is all a model promises.
""")
        right.image(GIF["win"] if tot["you"] >= tot["model"] else GIF["pikachu"], caption="via GIPHY")
        if st.button("🔁 Play five new nights", type="primary"):
            del st.session_state.game
            st.rerun()

# ---------------------------------------------------------------- a single booking
with tab_check:
    st.markdown("Enter a booking as the reservation system sees it **on the day it is made**.")
    cats = config["categories"]
    with st.form("booking"):
        a, b, c = st.columns(3)
        row = {
            "hotel": a.selectbox("Hotel", cats["hotel"], index=cats["hotel"].index("City Hotel")),
            "lead_time": a.number_input("Days until arrival", 0, 700, 90),
            "total_nights": a.number_input("Nights", 1, 30, 3),
            "stays_in_weekend_nights": a.number_input("…of which weekend nights", 0, 10, 1),
            "arrival_weekday": a.selectbox("Arrival day", range(7), 4,
                                           format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]),
            "adults": b.number_input("Adults", 1, 6, 2), "children": b.number_input("Children", 0, 4, 0),
            "babies": b.number_input("Babies", 0, 2, 0),
            "adr": b.number_input("Price per night (EUR)", 0, 600, 100),
            "deposit_type": b.selectbox("Deposit", cats["deposit_type"]),
            "market_segment": c.selectbox("Segment", cats["market_segment"], index=cats["market_segment"].index("Online TA")),
            "distribution_channel": c.selectbox("Channel", cats["distribution_channel"], index=cats["distribution_channel"].index("TA/TO")),
            "customer_type": c.selectbox("Customer", cats["customer_type"], index=cats["customer_type"].index("Transient")),
            "meal": c.selectbox("Meal", cats["meal"]), "reserved_room_type": c.selectbox("Room type", cats["reserved_room_type"]),
            "agent": c.selectbox("Agent", cats["agent"], index=cats["agent"].index("9") if "9" in cats["agent"] else 0),
            "is_repeated_guest": int(a.checkbox("Repeat guest")),
            "previous_cancellations": b.number_input("Earlier cancellations", 0, 20, 0),
            "previous_bookings_not_canceled": b.number_input("Earlier stays", 0, 50, 0),
        }
        go = st.form_submit_button("Predict", type="primary")
    if go:
        row["total_guests"] = row["adults"] + row["children"] + row["babies"]
        one = pd.DataFrame([row])[features]
        p = float(pipe.predict_proba(one)[0])
        t = config["call_threshold"]
        left, right = st.columns(2)
        left.metric("Cancellation risk", f"{p:.0%}")
        left.progress(p)
        right.markdown(f"**Why:** {reasons(pipe, features, one, top=3)[0] or 'nothing pushes the risk up'}")
        right.markdown(("📞 **Call to reconfirm.**" if p >= t else "✅ **No call needed.**")
                       + f" The cut-off is {t:.0%}: a call costs {config['costs_eur']['call']} EUR, a cancellation"
                       f" caught in time is worth {config['costs_eur']['caught_cancellation']} EUR.")

# ---------------------------------------------------------------- how it works
with tab_hood:
    st.markdown("### How this app works")
    st.graphviz_chart(f"""
digraph {{ rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#F2F1F7", color="#211A52", fontname=Helvetica];
  user [label="Your browser"]; app [label="This app\\nStreamlit Community Cloud\\nbuilt from GitHub: app.py + requirements.txt"];
  gh [label="GitHub repo\\napp.py · requirements.txt\\nmodel/: booster.json · preprocess.json · config.json"];
  nb [label="Colab notebook\\ntrain, tune, check\\nexport portable files"];
  nb -> gh [label="commit model files"]; gh -> app [label="build on push"]; user -> app [label="click"]; app -> user [label="page"]; }}
""")
    left, right = st.columns(2)
    with left:
        st.markdown("### Model card")
        st.markdown(card)
    with right:
        st.markdown("### config.json")
        st.json({k: config[k] for k in ["version", "periods", "metrics_test", "call_threshold", "dropped_as_leakage", "versions"]})
        st.markdown("""
### Make it yours
1. **Fork** [aaubs/tonights-front-desk](https://github.com/aaubs/tonights-front-desk) on GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and **Create app** from your fork
   (file `app.py`; any Python version works).
3. Change something small in `app.py` on GitHub: the costs, a meme, the number of nights. Commit, and watch the app update.
4. Optional: train your own model in the notebook, download the `hotel_model` folder, and replace the files in
   your fork's `model/` folder (`booster.json`, `preprocess.json`, `config.json`, `README.md`).
""")
