import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="AI Food Waste Predictor", page_icon="🍲", layout="wide")

DATA_PATH = "data/food_data.csv"
NGO_PATH = "data/ngos.csv"
COST_PER_KG = 60      # assumed cost of food in INR per kg
KG_PER_MEAL = 0.4     # assumed kg of food per meal
SAFETY_BUFFER = 1.05  # cook 5% above predicted demand
WEATHER_MAP = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}
FEATURES = ["day_of_week", "event", "weather_code", "students_present"]
REQUIRED_COLS = ["date", "event", "weather", "students_present",
                 "food_prepared_kg", "food_consumed_kg"]

os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label {
    font-family: 'Poppins', sans-serif !important;
}
.stApp { background: #F3F8F3; }
.block-container { padding-top: 1.6rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }

/* hero banner */
.hero {
    background: linear-gradient(135deg, #1B5E20 0%, #388E3C 55%, #81C784 100%);
    padding: 2rem 2.4rem; border-radius: 22px; margin-bottom: 1.6rem;
    box-shadow: 0 10px 28px rgba(27, 94, 32, 0.25);
}
.hero h1 { color: #fff !important; margin: 0; font-size: 2rem; font-weight: 700; }
.hero p { color: #E8F5E9; margin: .5rem 0 0 0; font-size: 1rem; }
.hero .tag {
    display: inline-block; background: rgba(255,255,255,.18); color: #fff;
    padding: .25rem .8rem; border-radius: 999px; font-size: .75rem; margin-bottom: .7rem;
}

/* KPI cards */
.kpi {
    background: #fff; border-radius: 16px; padding: 1rem 1.2rem;
    box-shadow: 0 4px 16px rgba(0,0,0,.07); border-left: 6px solid #2E7D32;
    display: flex; align-items: center; gap: .9rem; margin-bottom: .6rem;
}
.kpi .ic { font-size: 1.9rem; }
.kpi .l { color: #6B7280; font-size: .78rem; font-weight: 500; }
.kpi .v { color: #1F2937; font-size: 1.55rem; font-weight: 700; line-height: 1.2; }

/* step cards */
.step {
    background: #fff; border-radius: 16px; padding: 1.2rem; height: 100%;
    box-shadow: 0 4px 16px rgba(0,0,0,.07); text-align: center;
}
.step .n { font-size: 1.8rem; }
.step .t { color: #1B5E20; font-weight: 600; margin: .3rem 0; }
.step .d { color: #6B7280; font-size: .82rem; }

/* status banners */
.banner { padding: 1rem 1.2rem; border-radius: 14px; margin: .8rem 0; font-weight: 500; }
.banner.ok   { background: #E8F5E9; color: #1B5E20; border-left: 6px solid #2E7D32; }
.banner.warn { background: #FFF8E1; color: #8D6E00; border-left: 6px solid #FFB300; }
.banner.bad  { background: #FFEBEE; color: #B71C1C; border-left: 6px solid #E53935; }

/* section title */
.sec { color: #1B5E20; font-weight: 600; font-size: 1.15rem; margin: 1.2rem 0 .5rem 0; }

/* sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%); }
[data-testid="stSidebar"] * { color: #fff !important; }
.brand { font-size: 1.25rem; font-weight: 700; margin-bottom: .2rem; }
.brand-sub { font-size: .75rem; opacity: .8; margin-bottom: 1rem; }
.side-kpi {
    background: rgba(255,255,255,.14); border-radius: 12px; padding: .6rem .9rem;
    margin-bottom: .5rem; font-size: .8rem;
}
.side-kpi b { font-size: 1.15rem; display: block; }

/* buttons */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #2E7D32, #43A047); color: #fff !important;
    border: none; border-radius: 12px; padding: .6rem 1.6rem; font-weight: 600;
    box-shadow: 0 4px 12px rgba(46,125,50,.3);
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #1B5E20, #2E7D32); color: #fff !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 16px; }
</style>
""", unsafe_allow_html=True)


def hero(title, sub, tag="SDG 12 · Responsible Consumption"):
    st.markdown(f'<div class="hero"><span class="tag">{tag}</span>'
                f'<h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, icon, color) in zip(cols, items):
        col.markdown(
            f'<div class="kpi" style="border-left-color:{color}"><div class="ic">{icon}</div>'
            f'<div><div class="l">{label}</div><div class="v">{value}</div></div></div>',
            unsafe_allow_html=True)


def banner(text, kind="ok"):
    st.markdown(f'<div class="banner {kind}">{text}</div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def style_fig(fig):
    fig.update_layout(template="plotly_white", font_family="Poppins",
                      margin=dict(l=10, r=10, t=55, b=10), title_font_size=17,
                      legend_title_text="", paper_bgcolor="white", plot_bgcolor="white")
    return fig


# ---------------------------------------------------------------- data
def generate_data(days=365, seed=42):
    """Simulated hostel/canteen data (mention this in your document)."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    rows = []
    for d in pd.date_range(end=end, periods=days):
        dow = d.dayofweek
        event = int(rng.random() < 0.08)
        weather = rng.choice(["Sunny", "Cloudy", "Rainy"], p=[0.55, 0.25, 0.20])
        students = 400 + rng.integers(-30, 30)
        if dow >= 5:
            students *= 0.6
        if event:
            students *= 1.25
        if weather == "Rainy":
            students *= 1.05
        students = int(students)
        consumed = students * 0.45 + rng.normal(0, 8)
        prepared = consumed * rng.uniform(1.10, 1.40)
        rows.append([d.date(), event, weather, students,
                     round(prepared, 1), round(consumed, 1)])
    return pd.DataFrame(rows, columns=REQUIRED_COLS)


def load_data():
    if not os.path.exists(DATA_PATH):
        generate_data().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def load_ngos():
    if not os.path.exists(NGO_PATH):
        pd.DataFrame({
            "ngo_name": ["Annapurna Food Bank", "Hope Shelter", "Seva Kitchen",
                         "City Care Foundation", "Roti Bank"],
            "area": ["Central", "North", "East", "South", "West"],
            "distance_km": [2.1, 3.5, 4.8, 6.2, 7.9],
            "capacity_kg": [60, 40, 80, 50, 100],
            "phone": ["98000 00001", "98000 00002", "98000 00003",
                      "98000 00004", "98000 00005"],
        }).to_csv(NGO_PATH, index=False)
    return pd.read_csv(NGO_PATH)


def prep(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["weather_code"] = df["weather"].map(WEATHER_MAP).fillna(0)
    return df


# ---------------------------------------------------------------- model
@st.cache_resource
def train_model(n_rows, checksum):
    df = prep(load_data())
    X, y = df[FEATURES], df["food_consumed_kg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return model, mean_absolute_error(y_te, pred), r2_score(y_te, pred)


df_raw = load_data()
model, mae, r2 = train_model(len(df_raw), round(df_raw["food_consumed_kg"].sum(), 1))

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown('<div class="brand">🍲 FoodWaste AI</div>'
                    '<div class="brand-sub">Predict · Reduce · Donate</div>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["🏠 Home", "📥 Data Entry", "🔮 Prediction",
                                     "🚨 Surplus Alerts", "📊 Dashboard"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<div class="side-kpi">Model<b>Random Forest</b></div>'
    f'<div class="side-kpi">MAE (kg)<b>{mae:.1f}</b></div>'
    f'<div class="side-kpi">R² score<b>{r2:.3f}</b></div>',
    unsafe_allow_html=True)

# ---------------------------------------------------------------- pages
if page == "🏠 Home":
    hero("AI Food Waste Predictor & Redistribution Platform",
         "Cook what is needed. Donate what is not. Predict tomorrow's demand with "
         "machine learning and connect surplus food with nearby NGOs.")

    avg_waste = (df_raw.food_prepared_kg - df_raw.food_consumed_kg).mean()
    kpi_row([
        ("Records in system", f"{len(df_raw):,}", "🗂️", "#2E7D32"),
        ("Avg. daily waste", f"{avg_waste:.0f} kg", "🗑️", "#E53935"),
        ("Model accuracy (R²)", f"{r2:.2f}", "🎯", "#1E88E5"),
        ("Model error (MAE)", f"{mae:.1f} kg", "📏", "#FB8C00"),
    ])

    section("How it works")
    steps = [("📝", "1. Log", "Record food prepared vs. consumed each day"),
             ("🤖", "2. Predict", "ML forecasts tomorrow's demand"),
             ("🍽️", "3. Cook smart", "Prepare only the recommended amount"),
             ("🤝", "4. Donate", "Alert nearby NGOs about any surplus")]
    for col, (icon, title, desc) in zip(st.columns(4), steps):
        col.markdown(f'<div class="step"><div class="n">{icon}</div>'
                     f'<div class="t">{title}</div><div class="d">{desc}</div></div>',
                     unsafe_allow_html=True)

    section("Last 30 days: prepared vs consumed")
    last = prep(df_raw).sort_values("date").tail(30).rename(
        columns={"food_prepared_kg": "Prepared", "food_consumed_kg": "Consumed"})
    fig = px.area(last, x="date", y=["Prepared", "Consumed"],
                  color_discrete_map={"Prepared": "#EF9A9A", "Consumed": "#43A047"})
    fig.update_layout(xaxis_title="", yaxis_title="kg")
    st.plotly_chart(style_fig(fig))

elif page == "📥 Data Entry":
    hero("Data Entry", "Add daily records manually or upload a CSV. "
                       "The model retrains automatically.", "STEP 1")
    with st.container(border=True):
        section("➕ Add a daily record")
        with st.form("entry"):
            c1, c2 = st.columns(2)
            date = c1.date_input("Date")
            weather = c2.selectbox("Weather", list(WEATHER_MAP))
            event = c1.checkbox("Special event / festival?")
            students = c2.number_input("People present", 1, 10000, 400)
            prepared = c1.number_input("Food prepared (kg)", 0.0, 5000.0, 220.0)
            consumed = c2.number_input("Food consumed (kg)", 0.0, 5000.0, 180.0)
            if st.form_submit_button("💾 Save record"):
                new = pd.DataFrame([[date, int(event), weather, students,
                                     prepared, consumed]], columns=REQUIRED_COLS)
                pd.concat([df_raw, new]).to_csv(DATA_PATH, index=False)
                st.success("Record saved. Model will retrain automatically.")
                st.rerun()

    with st.container(border=True):
        section("📤 Or upload a CSV")
        st.caption("Required columns: " + ", ".join(REQUIRED_COLS))
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            new = pd.read_csv(up)
            if all(c in new.columns for c in REQUIRED_COLS):
                if st.button("Add uploaded data"):
                    pd.concat([df_raw, new[REQUIRED_COLS]]).to_csv(DATA_PATH, index=False)
                    st.success(f"Added {len(new)} rows.")
                    st.rerun()
            else:
                st.error("CSV is missing required columns.")
        if st.button("🔄 Reset to demo data"):
            generate_data().to_csv(DATA_PATH, index=False)
            st.rerun()

    section("Recent records")
    st.dataframe(df_raw.tail(15), hide_index=True)

elif page == "🔮 Prediction":
    hero("Demand Prediction", "Enter tomorrow's details and get the recommended "
                              "quantity to cook.", "STEP 2")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        date = c1.date_input("Date to predict")
        weather = c2.selectbox("Expected weather", list(WEATHER_MAP))
        event = c1.checkbox("Special event / festival?")
        students = c2.number_input("Expected people", 1, 10000, 400)
        planned = st.number_input("Food you plan to cook (kg)", 0.0, 5000.0, 220.0)
        go = st.button("🔮 Predict demand")

    if go:
        x = pd.DataFrame([[pd.Timestamp(date).dayofweek, int(event),
                           WEATHER_MAP[weather], students]], columns=FEATURES)
        demand = float(model.predict(x)[0])
        recommended = demand * SAFETY_BUFFER
        surplus = max(planned - recommended, 0)
        st.session_state["surplus"] = surplus

        kpi_row([
            ("Predicted demand", f"{demand:.0f} kg", "📈", "#1E88E5"),
            ("Recommended to cook", f"{recommended:.0f} kg", "✅", "#2E7D32"),
            ("Predicted surplus", f"{surplus:.0f} kg", "⚠️",
             "#E53935" if surplus > 0 else "#2E7D32"),
        ])

        if surplus > 0:
            banner(f"⚠️ You plan to cook <b>{surplus:.0f} kg</b> more than needed. "
                   "Reduce the quantity or open <b>Surplus Alerts</b> to donate.", "warn")
        elif planned < demand:
            banner(f"🚫 Risk of shortage: planned amount is below predicted demand "
                   f"by <b>{demand - planned:.0f} kg</b>.", "bad")
        else:
            banner("✅ Your planned quantity looks well balanced.", "ok")

elif page == "🚨 Surplus Alerts":
    hero("Surplus Alerts & NGO Matching", "Send surplus food to the nearest NGOs "
                                          "before it goes to waste.", "STEP 3")
    with st.container(border=True):
        surplus = st.number_input("Surplus food to donate (kg)", 0.0, 5000.0,
                                  float(st.session_state.get("surplus", 40.0)))

    ngos = load_ngos().sort_values("distance_km").reset_index(drop=True)
    remaining, plan = surplus, []
    for _, r in ngos.iterrows():
        give = min(remaining, r.capacity_kg)
        plan.append(give)
        remaining -= give
    ngos["suggested_donation_kg"] = plan

    section("Nearby NGOs (closest first)")
    st.dataframe(ngos, hide_index=True)

    if surplus > 0:
        donated = surplus - remaining
        kpi_row([
            ("Food allocated", f"{donated:.0f} kg", "📦", "#2E7D32"),
            ("Meals equivalent", f"{int(donated / KG_PER_MEAL)}", "🍛", "#FB8C00"),
            ("NGOs notified", f"{int((ngos.suggested_donation_kg > 0).sum())}", "🏠", "#1E88E5"),
        ])
        if remaining > 0:
            banner(f"⚠️ {remaining:.0f} kg could not be placed. "
                   "Consider adding more NGO partners.", "warn")
        else:
            banner("✅ All surplus food has been matched to NGOs.", "ok")
        top = ngos.iloc[0]
        section("📲 Sample alert message")
        st.code(f"Alert sent to {top.ngo_name} ({top.phone}): "
                f"{top.suggested_donation_kg:.0f} kg of cooked food "
                f"available for pickup today.")
    else:
        banner("No surplus right now. Nothing to donate. 🎉", "ok")

elif page == "📊 Dashboard":
    hero("Impact Dashboard", "See how much food, money and meals the AI "
                             "recommendations can save.", "STEP 4")
    d = prep(df_raw).sort_values("date")
    d["waste_kg"] = d.food_prepared_kg - d.food_consumed_kg
    d["ai_prepared"] = model.predict(d[FEATURES]) * SAFETY_BUFFER
    d["ai_waste"] = (d.ai_prepared - d.food_consumed_kg).clip(lower=0)

    saved = (d.waste_kg - d.ai_waste).sum()
    kpi_row([
        ("Waste without AI", f"{d.waste_kg.sum():,.0f} kg", "🗑️", "#E53935"),
        ("Food saved with AI", f"{saved:,.0f} kg", "🌱", "#2E7D32"),
        ("Money saved", f"₹{saved * COST_PER_KG:,.0f}", "💰", "#FB8C00"),
        ("Meals equivalent", f"{int(saved / KG_PER_MEAL):,}", "🍛", "#1E88E5"),
    ])
    st.caption("Estimate: assumes the canteen cooks the model's recommended amount. "
               f"Assumed cost ₹{COST_PER_KG}/kg and {KG_PER_MEAL} kg per meal.")

    last = d.tail(60).rename(columns={"food_prepared_kg": "Prepared",
                                      "food_consumed_kg": "Consumed",
                                      "ai_prepared": "AI-recommended"})
    fig1 = px.line(last, x="date", y=["Prepared", "Consumed", "AI-recommended"],
                   title="Prepared vs Consumed vs AI-Recommended (last 60 days)",
                   color_discrete_map={"Prepared": "#E53935", "Consumed": "#1B5E20",
                                       "AI-recommended": "#66BB6A"})
    fig1.update_layout(xaxis_title="", yaxis_title="kg")
    st.plotly_chart(style_fig(fig1))

    c1, c2 = st.columns(2)
    by_day = d.groupby("day_of_week")["waste_kg"].mean().reset_index()
    by_day["day"] = by_day.day_of_week.map(
        dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    fig2 = px.bar(by_day, x="day", y="waste_kg", color="waste_kg",
                  color_continuous_scale=["#A5D6A7", "#E53935"],
                  title="Average waste by day (kg)")
    fig2.update_layout(xaxis_title="", yaxis_title="kg", coloraxis_showscale=False)
    c1.plotly_chart(style_fig(fig2))

    comp = pd.DataFrame({"Scenario": ["Without AI", "With AI"],
                         "Total waste (kg)": [d.waste_kg.sum(), d.ai_waste.sum()]})
    fig3 = px.bar(comp, x="Scenario", y="Total waste (kg)", color="Scenario",
                  color_discrete_map={"Without AI": "#E53935", "With AI": "#43A047"},
                  title="Total waste comparison")
    fig3.update_layout(xaxis_title="", showlegend=False)
    c2.plotly_chart(style_fig(fig3))
