import streamlit as st
import pandas as pd
import joblib
import os

BASE = "/content/online_shopping_project"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "purchase_model.pkl")
)

feature_names = joblib.load(
    os.path.join(BASE_DIR, "feature_names.pkl")
)
st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
      radial-gradient(ellipse at 5% 0%, #29205b 0%, transparent 38%),
      radial-gradient(ellipse at 95% 15%, #452052 0%, transparent 35%),
      linear-gradient(135deg, #090d1f 0%, #11152d 50%, #17102b 100%);
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: #f8fafc !important;
}
p, label, .stMarkdown {
    color: #dbe4ff;
}
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(115deg, #6529dc 0%, #4737d5 45%, #087edb 100%);
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 28px;
    padding: 38px 40px;
    box-shadow: 0 18px 55px rgba(88, 55, 220, .3);
    margin-bottom: 26px;
}
.hero h1 {
    font-size: clamp(34px, 5vw, 54px) !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px;
    margin-bottom: 8px;
}
.hero p {
    color: #f0edff !important;
    font-size: 16px;
    margin: 5px 0;
}
.eyebrow {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 99px;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.25);
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 15px;
}
.feature {
    background: linear-gradient(145deg, rgba(35,43,81,.95), rgba(24,27,54,.95));
    border: 1px solid rgba(148,163,255,.22);
    border-radius: 20px;
    padding: 23px;
    min-height: 142px;
    box-shadow: 0 9px 28px rgba(0,0,0,.16);
    margin-bottom: 18px;
}
.feature-icon {
    font-size: 27px;
    margin-bottom: 10px;
}
.feature h3 {
    font-size: 18px !important;
    margin: 0 0 7px 0;
}
.feature p {
    color: #b9c5e8 !important;
    font-size: 13px;
    margin: 0;
}
.section-heading {
    padding: 14px 0 8px 0;
    border-bottom: 1px solid rgba(167,139,250,.25);
    margin-bottom: 18px;
}
div[data-testid="stForm"] {
    background: linear-gradient(145deg, rgba(27,33,66,.98), rgba(20,23,47,.98));
    border: 1px solid rgba(139,132,255,.28);
    border-radius: 24px;
    padding: 27px;
    box-shadow: 0 15px 40px rgba(0,0,0,.22);
}
div[data-testid="stNumberInput"] input {
    background: #111832 !important;
    color: #ffffff !important;
    border: 1px solid #424c79 !important;
    border-radius: 10px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #111832 !important;
    color: #ffffff !important;
    border-color: #424c79 !important;
    border-radius: 10px !important;
}
div[data-testid="stSelectbox"] svg {
    fill: #c4b5fd !important;
}
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(100deg, #9333ea, #6366f1 55%, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 13px !important;
    padding: 15px 20px !important;
    min-height: 52px;
    font-weight: 800 !important;
    font-size: 16px !important;
    box-shadow: 0 8px 24px rgba(99,102,241,.30);
    transition: all .2s ease;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    filter: brightness(1.15);
    transform: translateY(-2px);
}
.result {
    background: linear-gradient(135deg, #202958, #282044);
    border: 1px solid rgba(167,139,250,.4);
    border-radius: 24px;
    padding: 28px;
    margin-top: 24px;
    box-shadow: 0 12px 40px rgba(0,0,0,.2);
}
.result-label {
    color: #c4b5fd;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 800;
}
.result-value {
    color: #ffffff;
    font-family: 'Outfit', sans-serif;
    font-size: 30px;
    font-weight: 800;
    margin: 10px 0;
}
.probability {
    background: linear-gradient(90deg, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
    font-size: 43px;
    font-weight: 800;
}
.footer {
    text-align: center;
    color: #8996bc;
    font-size: 12px;
    padding-top: 25px;
}
div[data-testid="stAlert"] {
    border-radius: 14px;
}
hr {
    border-color: rgba(148,163,255,.2);
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ MACHINE LEARNING • SMART ANALYTICS</div>
  <h1>ShopSense <span style="color:#a5f3fc;">AI.</span></h1>
  <p style="font-size:19px;">Predict the intent. Understand the shopper.</p>
  <p>An intelligent shopping-session analyzer powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Feature cards
a, b, c = st.columns(3)
with a:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">🧠</div>
      <h3>AI Intelligence</h3>
      <p>Machine learning analyzes browsing behaviour.</p>
    </div>""", unsafe_allow_html=True)
with b:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">📈</div>
      <h3>Smart Analytics</h3>
      <p>Explore activity, visitor type and session signals.</p>
    </div>""", unsafe_allow_html=True)
with c:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">⚡</div>
      <h3>Instant Prediction</h3>
      <p>Receive a purchase-intention estimate in seconds.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="section-heading">
  <h2>🎛️ Configure a Shopping Session</h2>
  <p>Adjust the details below to generate a prediction.</p>
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    st.markdown("### 🛒 Browsing Behaviour")
    x1, x2, x3 = st.columns(3)

    with x1:
        administrative = st.number_input("Administrative Pages", min_value=0, value=0)
        administrative_duration = st.number_input("Administrative Duration", min_value=0.0, value=0.0)
        informational = st.number_input("Informational Pages", min_value=0, value=0)

    with x2:
        informational_duration = st.number_input("Informational Duration", min_value=0.0, value=0.0)
        product_related = st.number_input("Product Related Pages", min_value=0, value=1)
        product_duration = st.number_input("Product Related Duration", min_value=0.0, value=0.0)

    with x3:
        bounce_rates = st.number_input("Bounce Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.4f")
        exit_rates = st.number_input("Exit Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.4f")
        page_values = st.number_input("Page Values", min_value=0.0, value=0.0)

    st.markdown("---")
    st.markdown("### 🌐 Visitor & Session Profile")
    y1, y2, y3 = st.columns(3)

    with y1:
        special_day = st.selectbox("Special Day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        month = st.selectbox(
            "Month",
            ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan"]
        )
        visitor_type = st.selectbox(
            "Visitor Type",
            ["Returning_Visitor", "New_Visitor", "Other"]
        )

    with y2:
        operating_system = st.selectbox("Operating System", list(range(1, 9)))
        browser = st.selectbox("Browser", list(range(1, 14)))
        weekend = st.selectbox("Weekend", [False, True])

    with y3:
        region = st.selectbox("Region", list(range(1, 10)))
        traffic_type = st.selectbox("Traffic Type", list(range(1, 21)))

    submitted = st.form_submit_button(
        "✨  PREDICT PURCHASE INTENTION",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_system,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend
    }])

    processed_input = pd.get_dummies(input_data, dtype=int)
    processed_input = processed_input.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(processed_input)[0]
    probability = float(model.predict_proba(processed_input)[0][1])

    st.markdown('<div class="result">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">✦ AI ANALYSIS COMPLETE</div>', unsafe_allow_html=True)
    st.markdown('<h2>🎯 Prediction Result</h2>', unsafe_allow_html=True)

    if prediction == 1:
        st.success("🛍️ Purchase Intention: Likely to Purchase")
        result_text = "Likely to Purchase"
    else:
        st.info("🛒 Purchase Intention: Unlikely to Purchase")
        result_text = "Unlikely to Purchase"

    st.markdown(
        f'<div class="result-value">{result_text}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="result-label">ESTIMATED PURCHASE PROBABILITY</div>'
        f'<div class="probability">{probability:.2%}</div>',
        unsafe_allow_html=True
    )
    st.progress(probability)
    st.caption("This is a model estimate, not a guarantee of actual customer behaviour.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  <p>SHOPSENSE AI • ML MINI PROJECT</p>
  Built with Python · Pandas · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
