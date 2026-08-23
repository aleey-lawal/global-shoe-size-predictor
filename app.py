# app.py — Global Shoe Size Predictor

import streamlit as st             # streamlit: turns Python scripts into interactive web apps
import pandas as pd                # pandas: DataFrame creation for prediction input
import numpy as np                 # numpy: numerical rounding operations
import joblib                      # joblib: load the saved model from disk
import plotly.graph_objects as go  # plotly graph_objects: low-level chart objects (for the gauge)
import plotly.express as px        # plotly express: high-level quick chart functions (scatter, box, bar)

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Shoe Size Predictor",   # browser tab title
    page_icon="👟",                             # browser tab favicon emoji
    layout="wide",                             # use the full browser width (not narrow centered)
    initial_sidebar_state="expanded"           # open the sidebar by default when the page loads
)

TARGETS = ['Size_EU', 'Size_US', 'Size_UK', 'Size_JP_mm', 'Size_CN']
# list of all model output names — used to unpack predictions into labeled values

REGIONS = {
    'Africa & Middle East': ['Egypt', 'Kenya', 'Nigeria', 'Saudi Arabia', 'South Africa'],
    'Asia':                 ['China', 'India', 'Japan', 'South Korea', 'Thailand', 'Vietnam'],
    'Australia & Oceania':  ['Australia', 'New Zealand'],
    'Europe':               ['France', 'Germany', 'Italy', 'Netherlands', 'Spain', 'UK'],
    'North America':        ['Canada', 'Mexico', 'USA'],
}
# nested dict mapping each region to its list of countries
# used to populate the Country dropdown based on whichever Region the user selects

# ── Load Model ──────────────────────────────────────────────────────────────────
@st.cache_resource
# cache_resource: load the model once and keep it in memory across user interactions
# without this, the model would reload from disk on every button click (very slow)
def load_model():
    return joblib.load('model/shoe_model.pkl')   # load the saved pipeline from the model folder

model = load_model()   # call the function once — result is cached for all future interactions

# ── Helper Functions ────────────────────────────────────────────────────────────
def round_half(x):
    return round(x * 2) / 2
    # rounds a float to the nearest 0.5 (e.g. 42.3 → 42.5, 41.7 → 42.0)
    # multiply by 2, round to integer, divide by 2 — elegant trick for half-step rounding

def get_width_label(width_cm, gender):
    # classify foot width into a standard shoe width category label
    if gender == 'Male':
        if width_cm < 8.8:    return 'Narrow (B)'       # narrower than average male foot
        elif width_cm < 9.5:  return 'Standard (D)'     # typical male width
        elif width_cm < 10.2: return 'Wide (2E)'         # wider than average
        else:                  return 'Extra Wide (4E)'  # very wide
    else:
        if width_cm < 8.0:    return 'Narrow (AA)'      # narrower than average female foot
        elif width_cm < 8.7:  return 'Standard (B)'     # typical female width
        elif width_cm < 9.4:  return 'Wide (D)'          # wider than average
        else:                  return 'Extra Wide (2E)'  # very wide

# ── Page Header ─────────────────────────────────────────────────────────────────
st.title("👟 Global Shoe Size Predictor")   # large H1 title at the top of the page
st.markdown(
    "Enter your measurements to instantly get your shoe size in "
    "**EU · US · UK · JP · CN** standards."
)   # subtitle paragraph with bold text
st.divider()   # draws a horizontal rule line to visually separate the header from content

# ── Sidebar Inputs ──────────────────────────────────────────────────────────────
with st.sidebar:               # everything indented here appears in the left sidebar panel
    st.header("📋 Your Details")   # sidebar section heading

    gender = st.selectbox("Gender", ["Male", "Female"])
    # dropdown widget; user picks Male or Female; result stored in variable 'gender'

    region = st.selectbox("Region", list(REGIONS.keys()))
    # dropdown of region names (the keys of the REGIONS dict)

    country = st.selectbox("Country", REGIONS[region])
    # dropdown that changes dynamically: REGIONS[region] returns the country list for the selected region

    age = st.slider("Age", 16, 65, 28)
    # slider widget: min=16, max=65, default=28; result stored in 'age'

    st.markdown("---")         # horizontal rule to separate sections in the sidebar
    st.subheader("📐 Body Measurements")   # sidebar sub-heading

    height = st.slider("Height (cm)", 140.0, 210.0, 172.0, 0.5)
    # slider: min=140, max=210, default=172, step=0.5 (half-cm increments)

    weight = st.slider("Weight (kg)", 38.0, 160.0, 70.0, 0.5)
    # slider: min=38, max=160, default=70, step=0.5

    st.markdown("---")   # another divider
    st.subheader("📏 Foot Measurements")   # sub-heading

    st.info(
        "**How to measure:** Stand on paper, trace your foot. "
        "Measure heel → longest toe (length) and widest point (width)."
    )   # blue info box with measurement instructions

    foot_length = st.slider("Foot Length (cm)", 18.0, 33.0, 26.0, 0.1)
    # slider: min=18cm (child-like), max=33cm (very large), default=26cm, step=0.1cm

    foot_width = st.slider("Foot Width (cm)", 6.5, 13.5, 9.5, 0.1)
    # slider: min=6.5cm (narrow), max=13.5cm (very wide), default=9.5cm, step=0.1cm

    predict_btn = st.button("🔍 Get My Sizes", use_container_width=True, type="primary")
    # primary action button that spans the full sidebar width; clicking sets predict_btn=True

# ── Tab Layout ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 My Prediction", "📊 Size Explorer", "🌍 Global Insights"])
# create 3 clickable tabs at the top of the main panel; content for each goes in its own 'with' block

# ── TAB 1: PREDICTION ──────────────────────────────────────────────────────────
with tab1:
    if predict_btn:   # only run this block when the user clicks the "Get My Sizes" button

        input_df = pd.DataFrame([{      # create a one-row DataFrame matching the model's expected input format
            'Country':        country,   # from sidebar dropdown
            'Region':         region,    # from sidebar dropdown
            'Gender':         gender,    # from sidebar dropdown
            'Age':            age,       # from sidebar slider
            'Height_cm':      height,    # from sidebar slider
            'Weight_kg':      weight,    # from sidebar slider
            'Foot_Length_cm': foot_length,  # from sidebar slider
            'Foot_Width_cm':  foot_width    # from sidebar slider
        }])

        raw_preds = model.predict(input_df)[0]
        # pass the input through the trained pipeline (preprocessor + model) → get 1 row of 5 predictions
        # [0] extracts the single row as a 1D numpy array: [eu, us, uk, jp, cn]

        eu  = round_half(raw_preds[0])          # EU size → round to nearest 0.5
        us  = round_half(raw_preds[1])          # US size → round to nearest 0.5
        uk  = round_half(raw_preds[2])          # UK size → round to nearest 0.5
        jp  = int(round(raw_preds[3] / 5) * 5) # JP Mondopoint (mm) → round to nearest 5mm, then cast to int
        cn  = round_half(raw_preds[4])          # CN size → round to nearest 0.5
        wid = get_width_label(foot_width, gender)  # classify foot width into a descriptive label

        st.markdown("### 🎯 Your Predicted Shoe Sizes")   # results section heading

        c1, c2, c3, c4, c5 = st.columns(5)   # create 5 equal-width columns side by side
        c1.metric("🇪🇺 EU",      f"{eu}")     # display EU size in a metric card (large number + label)
        c2.metric("🇺🇸 US",      f"{us}")     # US size metric card
        c3.metric("🇬🇧 UK",      f"{uk}")     # UK size metric card
        c4.metric("🇯🇵 JP (mm)", f"{jp}")     # Japan Mondopoint metric card
        c5.metric("🇨🇳 CN",      f"{cn}")     # China size metric card

        st.success(
            f"**Width category:** {wid}  |  "
            f"**Foot length:** {foot_length} cm  |  "
            f"**Region:** {region}"
        )   # green success banner showing additional context below the metrics

        # Gauge chart for EU size
        min_eu = 34 if gender == 'Female' else 39   # set gauge min based on gender
        max_eu = 43 if gender == 'Female' else 48   # set gauge max based on gender

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",              # show both the gauge arc and the numeric value
            value=eu,                         # the predicted EU size is the needle position
            domain={'x': [0, 1], 'y': [0, 1]},  # fill the entire figure area
            title={'text': "EU Size", 'font': {'size': 16}},  # title above the gauge
            gauge={
                'axis': {'range': [min_eu, max_eu], 'tickwidth': 1},  # gauge arc from min to max EU size
                'bar':  {'color': '#3498db'},   # color of the needle/fill bar
                'steps': [
                    {'range': [min_eu, min_eu + (max_eu-min_eu)*0.33], 'color': '#d5e8d4'},  # small sizes: green tint
                    {'range': [min_eu + (max_eu-min_eu)*0.33,
                               min_eu + (max_eu-min_eu)*0.66], 'color': '#fff2cc'},           # medium: yellow tint
                    {'range': [min_eu + (max_eu-min_eu)*0.66, max_eu], 'color': '#f8cecc'},   # large: red tint
                ],
                # three colored background zones dividing the gauge into thirds
                'threshold': {
                    'line':      {'color': 'red', 'width': 3},  # red vertical line at the predicted value
                    'thickness': 0.75,    # line spans 75% of the gauge height
                    'value':     eu       # position the threshold marker at the predicted EU size
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20))
        # set figure height and reduce whitespace margins around the gauge
        st.plotly_chart(fig_gauge, use_container_width=True)
        # render the gauge chart, stretching it to fill the available column width

    else:
        # shown when the page first loads or after a refresh (before the user clicks Predict)
        st.info("👈 Fill in your measurements on the left sidebar, then click **Get My Sizes**.")

        st.markdown("### 📋 International Size Conversion Reference")   # section heading

        conv = pd.DataFrame({
            'EU':    [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],   # EU sizes
            'US ♂':  [4.5, 5, 5.5, 6.5, 7.5, 8, 9, 10, 11, 11.5, 12.5, 13],   # US male equivalents
            'US ♀':  [5.5, 6, 6.5, 7.5, 8.5, 9, 10, 11, 12, 12.5, 13.5, 14],  # US female equivalents
            'UK':    [3.5, 4, 4.5, 5.5, 6.5, 7, 8, 9, 10, 10.5, 11.5, 12],    # UK equivalents
            'JP mm': [225, 230, 240, 245, 255, 260, 265, 275, 280, 285, 295, 300],  # Japan Mondopoint mm
            'CN':    [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],   # China sizing
        })   # static reference table showing standard international size conversions

        st.dataframe(conv, use_container_width=True, hide_index=True)
        # render the table as an interactive dataframe; hide_index=True removes the row numbers

# ── TAB 2: SIZE EXPLORER ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Explore How Size Varies Across Dimensions")   # tab heading

    col_a, col_b = st.columns(2)   # split the tab into two equal-width columns

    with col_a:
        df_viz = pd.read_csv('data/global_shoe_size_dataset.csv')
        # load the full dataset for visualization — done here so the app works on Streamlit Cloud

        fig_scatter = px.scatter(
            df_viz,                    # the full dataset
            x='Foot_Length_cm',        # x-axis: foot length
            y='Size_EU',               # y-axis: EU shoe size
            color='Gender',            # color dots by gender
            symbol='Region',           # use different marker shapes for each region
            opacity=0.55,              # semi-transparent to handle overlapping dots
            size_max=6,                # cap the maximum marker size
            color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'},  # gender colors
            title='Foot Length vs EU Size (by Gender & Region)',   # chart title
            labels={'Foot_Length_cm': 'Foot Length (cm)', 'Size_EU': 'EU Size'}  # axis labels
        )
        fig_scatter.update_layout(height=400)   # fix chart height
        st.plotly_chart(fig_scatter, use_container_width=True)   # render interactive chart

    with col_b:
        fig_box = px.box(
            df_viz,           # the full dataset
            x='Region',       # regions on the x-axis
            y='Size_EU',      # EU size on the y-axis
            color='Gender',   # split each box by gender
            color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'},   # gender colors
            title='EU Size Distribution by Region',   # chart title
            labels={'Size_EU': 'EU Size', 'Region': ''}   # clean axis labels
        )
        fig_box.update_layout(height=400, xaxis_tickangle=-20)  # fix height; angle x labels 20°
        st.plotly_chart(fig_box, use_container_width=True)   # render interactive chart

    # Country-level average bar chart (spans both columns)
    country_avg = df_viz.groupby('Country')['Size_EU'].mean().sort_values(ascending=True)
    # group by country → mean EU size per country → sort smallest to largest

    fig_country = px.bar(
        country_avg.reset_index(),   # convert Series to DataFrame so plotly can read column names
        x='Size_EU',                 # size values on the x-axis
        y='Country',                 # country names on the y-axis
        orientation='h',             # horizontal bars (easier to read country names)
        title='Average EU Shoe Size by Country',   # chart title
        labels={'Size_EU': 'Average EU Size', 'Country': ''},   # clean axis labels
        color='Size_EU',             # color each bar by its size value
        color_continuous_scale='Blues'  # gradient from light blue (small) to dark blue (large)
    )
    fig_country.update_layout(height=500, showlegend=False)  # taller chart; hide redundant color legend
    st.plotly_chart(fig_country, use_container_width=True)   # render interactive chart

# ── TAB 3: GLOBAL INSIGHTS ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🌍 Global Foot Size Research Insights")   # tab heading

    st.markdown("""
    This model is grounded in real-world research data:

    **📚 Data Sources:**
    - **Jurca et al. (2019)** — *Analysis of 1.2 million foot scans from North America,
      Europe and Asia*, Scientific Reports (Volumental 3D scanning study)
    - **Wunderlich & Cavanagh (2001)** — *Gender differences in adult foot shape*,
      Med. Sci. Sports Exerc.
    - **WHO Global Health Observatory** — Regional height and weight averages

    **🔍 Key Findings Embedded in This Model:**
    """)   # multi-line markdown block with research citations

    col1, col2, col3 = st.columns(3)   # three equal columns for metric cards
    col1.metric("Avg Male Foot — Europe",     "27.0 cm", "↑ vs Asia by 1.7 cm")  # metric: value, delta
    col2.metric("Avg Male Foot — Asia",       "25.3 cm", "shortest globally")     # metric with delta label
    col3.metric("Avg Male Foot — N. America", "26.9 cm", "similar to Europe")     # metric with delta label

    col4, col5, col6 = st.columns(3)   # second row of three columns
    col4.metric("Avg Female Foot — Europe",     "24.8 cm")             # metric (no delta)
    col5.metric("Avg Female Foot — Asia",       "23.3 cm", "↓ vs Europe by 1.5 cm")  # negative delta
    col6.metric("Avg Female Foot — N. America", "25.2 cm")             # metric (no delta)

    st.markdown("""
    **📌 Regional Patterns:**
    - Asian customers have significantly **shorter feet** than European and North American
      customers for both sexes — confirmed by Volumental's 1.2M scan analysis
    - Asian feet are also **relatively wider** (higher width-to-length ratio)
    - Northern European and Australian customers tend to have the **largest feet globally**
    - Foot size has **increased over the past 60 years** driven by improved nutrition and
      increased body mass
    """)   # bullet-point summary of research findings shown to the user

    st.info(
        "💡 **Tip:** Because sizing varies significantly by region and brand, always "
        "try shoes on before purchasing when possible. Use this predictor as a starting "
        "point, not a definitive answer."
    )   # blue informational callout box with a usage caveat

# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()   # horizontal rule above the footer
st.markdown(
    "Built by [Lawal Lawal Ali](https://aleey-lawal.github.io) · "
    "[GitHub Repo](https://github.com/aleey-lawal/global-shoe-predictor) · "
    "Data: Jurca et al. (2019) / WHO"
)   # footer text with hyperlinks to portfolio, GitHub, and data citations