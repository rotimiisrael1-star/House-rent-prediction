import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import datetime

# PAGE CONFIGURATION

st.set_page_config(
    page_title="House Rent Prediction System",
    page_icon="🏠 Browse Available Houses",
    layout="wide"
)

# SESSION STATES

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "favorites" not in st.session_state:
    st.session_state.favorites = []

# DARK MODE TOGGLE

dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:

    background_color = "#1A103D"
    card_color = "#1e1e1e"
    text_color = "#ffffff"
    secondary_text = "#cccccc"
    button_color = "#4CAF50"

else:

    background_color = "#f5f7fa"
    card_color = "#ffffff"
    text_color = "#000000"
    secondary_text = "#555555"
    button_color = "#1f4e79"
#000000
    #ffffff

st.markdown(f"""
<style>

.stApp {{
    background-color: {background_color};
    color: {text_color};
}}

[data-testid="stSidebar"] {{
    background-color: {card_color};
}}

[data-testid="stSidebar"] * {{
    color: {text_color};
}}

.main-title {{
    font-size: 48px;
    font-weight: bold;
    color: {button_color};
    text-align: center;
}}

.sub-title {{
    font-size: 20px;
    color: {secondary_text};
    text-align: center;
    margin-bottom: 30px;
}}

.card {{
    background-color: {card_color};
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}}

.house-title {{
    font-size: 24px;
    font-weight: bold;
    color: {button_color};
}}

.house-desc {{
    color: {secondary_text};
    font-size: 15px;
}}

.feature-text {{
    color: {text_color};
    font-size: 15px;
}}

# # /* LABELS ONLY */
# .stSelectbox label,
# .stNumberInput label,
# label {{
#     color: {button_color} !important;
#     font-weight: 700 !important;
#     font-size: 16px !important;
# }}

div.stButton > button {{
    background-color: {button_color};
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    border: none;
    font-size: 16px;
}}

div.stButton > button:hover {{
    opacity: 0.85;
}}

.tabs {{
    text_color: black;
}}
/* Tab headers: black by default, unaffected by hover */
.stTabs [data-baseweb="tab-list"] button {{
    color: #000000 !important;
}}

.stTabs [data-baseweb="tab-list"] button p {{
    color: #000000 !important;
}}

.stTabs [data-baseweb="tab-list"] button:hover {{
    color: #000000 !important;
}}

.stTabs [data-baseweb="tab-list"] button:hover p {{
    color: #000000 !important;
}}
/* Metrics (R², RMSE, MAE) */
[data-testid="stMetricValue"] {{
    color: {text_color} !important;
}}

[data-testid="stMetricLabel"] {{
    color: {text_color} !important;
}}

[data-testid="stMetricDelta"] {{
    color: {text_color} !important;
}}

/* Dataframe / table text */
[data-testid="stDataFrame"] {{
    color: {text_color} !important;
}}

[data-testid="stTable"] {{
    color: {text_color} !important;
}}

/* Bar chart labels/axis (Vega-Lite chart container) */
[data-testid="stVegaLiteChart"] text {{
    fill: {text_color} !important;
}}

/* Subheaders inside the analytics tab */
.stApp h3, .stApp h2 {{
    color: {text_color} !important;
}}


</style>
""", unsafe_allow_html=True)

# LOAD TRAINED FILES

model = joblib.load("rent_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

# LOAD DATASET

try:
    df = pd.read_csv("housing.csv")
except:
    df = None

# HEADER

st.markdown(
    "<div class='main-title'>🏠 House Rent Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Interactive House Rent Prediction </div>",
    unsafe_allow_html=True
)

# PROPERTY FILTER

st.sidebar.title("🏘 Property Filter")

location_filter = st.sidebar.selectbox(
    "Select Location",
    [
        "All",
        "Agbowo",
        "Ajibode",
        "Akobo",
        "Apata",
        "Bodija",
        "Challenge",
        "Mokola",
        "Ologuneru",
        "Oluyole",
        "Ring Road"
    ],
    key="sidebar_location"
)

# SAMPLE HOUSES

houses = [

    {
        "name": "Luxury Duplex",
        "description": "Modern duplex with stable PHCN and borehole water.",
        "location": "Bodija",
        "property_type": "Duplex",
        "bedrooms": 4,
        "bathrooms": 3,
        "house_size_sqm": 250,
        "lot_size_sqm": 400,
        "year_built": 2020,
        "furnishing": "Fully Furnished",
        "water_supply": "Borehole",
        "power_supply": "Stable",
        "image": "images/house1.jpg"
    },

    {
        "name": "Student Mini Flat",
        "description": "Affordable apartment suitable for students.",
        "location": "Agbowo",
        "property_type": "Mini Flat",
        "bedrooms": 1,
        "bathrooms": 1,
        "house_size_sqm": 80,
        "lot_size_sqm": 120,
        "year_built": 2018,
        "furnishing": "Unfurnished",
        "water_supply": "Well",
        "power_supply": "Moderate",
        "image": "images/house2.jpg"
    },

    {
        "name": "Executive Apartment",
        "description": "Luxury apartment with parking and security.",
        "location": "Akobo",
        "property_type": "3 Bedroom Flat",
        "bedrooms": 3,
        "bathrooms": 2,
        "house_size_sqm": 180,
        "lot_size_sqm": 250,
        "year_built": 2021,
        "furnishing": "Semi-Furnished",
        "water_supply": "Borehole",
        "power_supply": "Stable",
        "image": "images/house3.jpg"
    },

    {
        "name": "Self Contained",
        "description": "Afordable apartment fenced with a gate and security.",
        "location": "Ajibode",
        "property_type": "1 Bedroom",
        "bedrooms": 1,
        "bathrooms": 1,
        "house_size_sqm": 100,
        "lot_size_sqm": 250,
        "year_built": 2021,
        "furnishing": "Semi-Furnished",
        "water_supply": "Borehole",
        "power_supply": "Moderate",
        "image": "images/house4.jpg"
    }

]

# FILTER HOUSES

if location_filter != "All":

    houses = [
        house for house in houses
        if house["location"] == location_filter
    ]

# TABS

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

    "🏠 Browse Properties",
    "📝 Manual Prediction",
    "📊 Market Analytics",
    "📜 Prediction History",
    "⚖ Compare Houses",
    "ℹ About System"
])

# PREDICTION FUNCTION

def predict_house_rent(data_dict):

    input_df = pd.DataFrame([data_dict])

    input_encoded = pd.get_dummies(
        input_df,
        drop_first=True
    )

    for col in feature_order:

        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_order]

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)

    return prediction[0]

# RENT CATEGORY

def rent_category(rent):

    if rent < 500000:
        return "Affordable"

    elif rent < 1500000:
        return "Moderate"

    elif rent < 3000000:
        return "Premium"

    else:
        return "Luxury"

# TAB 1 — PROPERTY GALLERY

with tab1:

    st.header("🏠 Browse Available Properties")

    cols = st.columns(2)

    for idx, house in enumerate(houses):

        with cols[idx % 2]:

            st.markdown(
                "<div class='card'>",
                unsafe_allow_html=True
            )

            try:

                image = Image.open(house["image"])

                st.image(
                    image,
                    use_container_width=True
                )

            except:
                st.warning("Image not found")

            st.markdown(
                f"<div class='house-title'>{house['name']}</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div class='house-desc'>{house['description']}</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class='feature-text'>

                 Location: {house['location']} <br>
                 Property Type: {house['property_type']} <br>
                 Bedroom(s): {house['bedrooms']} <br>
                 Bathroom(s): {house['bathrooms']} <br>
                 PHCN Availability: {house['power_supply']} <br>
                 Water Supply: {house['water_supply']}

                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Predict Rent",
                key=f"predict_{idx}"
            ):

                with st.spinner("Predicting rent..."):

                    rent = predict_house_rent(house)

                st.success(
                    f"Estimated Annual Rent: ₦{rent:,.0f}"
                )

                st.info(
                    f"Category: {rent_category(rent)}"
                )

                st.session_state.prediction_history.append({

                    "House": house["name"],
                    "Location": house["location"],
                    "Predicted Rent": rent,
                    "Date": str(datetime.datetime.now())
                })

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# TAB 2 — MANUAL PREDICTION

with tab2:

    st.header("📝 Manual House Rent Prediction")

    col1, col2 = st.columns(2)

    with col1:

        location = st.selectbox(
            "Location",
            [
                "Agbowo",
                "Ajibode",
                "Akobo",
                "Apata",
                "Bodija",
                "Challenge",
                "Mokola",
                "Ologuneru",
                "Oluyole",
                "Ring Road"
            ],
            key="manual_location"
        )

        property_type = st.selectbox(
            "Property Type",
            [
                "2 Bedroom Flat",
                "3 Bedroom Flat",
                "Duplex",
                "Mini Flat",
                "Room",
                "Self Contained"
            ],
            key="manual_property_type"
        )

        furnishing = st.selectbox(
            "Furnishing",
            [
                "Fully Furnished",
                "Semi-Furnished",
                "Unfurnished"
            ],
            key="manual_furnishing"
        )

        water_supply = st.selectbox(
            "Water Supply",
            [
                "Borehole",
                "Public",
                "Well"
            ],
            key="manual_water_supply"
        )

        power_supply = st.selectbox(
            "PHCN Availability",
            [
                "Poor",
                "Moderate",
                "Stable"
            ],
            key="manual_power_supply"
        )

    with col2:

        bedrooms = st.number_input(
            "Bedroom(s)",
            1,
            10,
            2,
            key="manual_bedrooms"
        )

        bathrooms = st.number_input(
            "Bathroom(s)",
            1,
            10,
            1,
            key="manual_bathrooms"
        )

        house_size = st.number_input(
            "House Size (sqm)",
            10,
            5000,
            60,
            key="manual_house_size"
        )

        lot_size = st.number_input(
            "Lot Size (sqm)",
            10,
            10000,
            400,
            key="manual_lot_size"
        )

        year_built = st.number_input(
            "Year Built",
            1950,
            2026,
            2010,
            key="manual_year_built"
        )

    if st.button(
        "Predict Manual Rent",
        key="manual_predict_button"
    ):

        manual_data = {

            "location": location,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "house_size_sqm": house_size,
            "lot_size_sqm": lot_size,
            "year_built": year_built,
            "furnishing": furnishing,
            "water_supply": water_supply,
            "power_supply": power_supply
        }

        with st.spinner("Predicting rent..."):

            rent = predict_house_rent(manual_data)

        st.success(
            f"Estimated Annual Rent: ₦{rent:,.0f}"
        )

        st.info(
            f"Category: {rent_category(rent)}"
        )

        st.session_state.prediction_history.append({

            "House": property_type,
            "Location": location,
            "Predicted Rent": rent,
            "Date": str(datetime.datetime.now())
        })


# TAB 3 — ANALYTICS

with tab3:

    st.header("📊 Housing Market Analytics")

    if df is not None:

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        if "location" in df.columns and "rent" in df.columns:

            avg_rent = df.groupby("location")["rent"].mean()

            st.subheader("Average Rent by Location")

            st.bar_chart(avg_rent)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R² Score", "0.93")

    with col2:
        st.metric("RMSE", "114091.12")

    with col3:
        st.metric("MAE", "71949.19")

# TAB 4 — PREDICTION HISTORY

with tab4:

    st.header("📜 Prediction History")

    if len(st.session_state.prediction_history) > 0:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(history_df)

    else:

        st.info("No prediction history available.")


# TAB 5 — HOUSE COMPARISON

with tab5:

    st.header("⚖ Compare Houses")

    house_names = [house["name"] for house in houses]

    house1 = st.selectbox(
        "Select First House",
        house_names,
        key="compare_house1"
    )

    house2 = st.selectbox(
        "Select Second House",
        house_names,
        key="compare_house2"
    )

    if st.button(
        "Compare Houses",
        key="compare_button"
    ):

        h1 = next(
            h for h in houses if h["name"] == house1
        )

        h2 = next(
            h for h in houses if h["name"] == house2
        )

        rent1 = predict_house_rent(h1)
        rent2 = predict_house_rent(h2)

        comparison_df = pd.DataFrame({

            "Feature": [
                "Location",
                "Bedroom(s)",
                "Bathroom(s)",
                "Predicted Rent"
            ],

            house1: [
                h1["location"],
                h1["bedrooms"],
                h1["bathrooms"],
                f"₦{rent1:,.0f}"
            ],

            house2: [
                h2["location"],
                h2["bedrooms"],
                h2["bathrooms"],
                f"₦{rent2:,.0f}"
            ]
        })

        st.table(comparison_df)

# TAB 6 — ABOUT SYSTEM

with tab6:

    st.header("ℹ About System")

    st.write("""

    This system is a machine learning-based
    application developed for predicting
    house rent prices using Linear Regression.

    Features Include:

    - Property Gallery
    - Manual Prediction
    - Housing Analytics
    - Prediction History
    - House Comparison
    - Dark Mode Toggle

    Technologies Used:

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn

    """)

# FOOTER

st.markdown("---")

current_year = datetime.datetime.now().year

st.markdown(
    f"""
    <center>

    <h4 style='color:{button_color};'>
    Intelligent House Rent Prediction System
    </h4>

    <p style='color:{secondary_text};'>
    © {current_year} Final Year Project
    </p>

    </center>
    """,
    unsafe_allow_html=True
)