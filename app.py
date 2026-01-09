import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("country_wise_latest.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🦠 COVID-19 ML Project")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Country Analysis", "Risk Prediction", "About Project"]
)

# ---------------- RISK LEVEL LOGIC ----------------
def risk_level(row):
    if row["Confirmed"] > 100000 and row["Deaths"] > 2000:
        return "High"
    elif row["Confirmed"] > 20000:
        return "Medium"
    else:
        return "Low"

df["Risk_Level"] = df.apply(risk_level, axis=1)

# Encode target
le = LabelEncoder()
df["Risk_Code"] = le.fit_transform(df["Risk_Level"])

# ---------------- TRAIN MODEL ----------------
X = df[["Confirmed", "Deaths", "Recovered", "Active", "New cases"]]
y = df["Risk_Code"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ================= DASHBOARD =================
if menu == "Dashboard":
    st.title("📊 COVID-19 Global Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌍 Countries", df.shape[0])
    col2.metric("🦠 Total Confirmed", f"{df['Confirmed'].sum():,}")
    col3.metric("❤️ Total Recovered", f"{df['Recovered'].sum():,}")
    col4.metric("⚰️ Total Deaths", f"{df['Deaths'].sum():,}")

    st.markdown("---")

    # -------- Top 10 Countries Bar Chart --------
    st.subheader("🔝 Top 10 Countries by Confirmed Cases")
    top10 = df.sort_values(by="Confirmed", ascending=False).head(10)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(top10["Country/Region"], top10["Confirmed"])
    ax1.set_xlabel("Confirmed Cases")
    ax1.set_ylabel("Country")
    ax1.set_title("Top 10 Countries by Confirmed COVID-19 Cases")

    for i, v in enumerate(top10["Confirmed"]):
        ax1.text(v, i, f"{v:,}", va="center")

    ax1.invert_yaxis()
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    st.pyplot(fig1)

    # -------- Risk Level Distribution Pie Chart --------
    st.subheader("📌 Risk Level Distribution")
    risk_counts = df["Risk_Level"].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(
        risk_counts,
        labels=risk_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax2.set_title("Country-wise COVID Risk Distribution")
    st.pyplot(fig2)

    # -------- WHO Region Bar Chart --------
    st.subheader("🌍 WHO Region-wise Confirmed Cases")
    region_data = df.groupby("WHO Region")["Confirmed"].sum().sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(region_data.index, region_data.values)
    ax3.set_xlabel("WHO Region")
    ax3.set_ylabel("Confirmed Cases")
    ax3.set_title("Confirmed Cases by WHO Region")
    ax3.tick_params(axis="x", rotation=45)
    st.pyplot(fig3)

    # -------- Scatter Plot --------
    st.subheader("📈 Confirmed Cases vs Deaths")
    fig4, ax4 = plt.subplots()
    ax4.scatter(df["Confirmed"], df["Deaths"], alpha=0.6)
    ax4.set_xlabel("Confirmed Cases")
    ax4.set_ylabel("Deaths")
    ax4.set_title("Confirmed Cases vs Deaths")
    ax4.grid(True)
    st.pyplot(fig4)

# ================= COUNTRY ANALYSIS =================
elif menu == "Country Analysis":
    st.title("🌍 Country-wise COVID Analysis")

    country = st.selectbox("Select Country", df["Country/Region"])
    data = df[df["Country/Region"] == country].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Confirmed", f"{data['Confirmed']:,}")
    col2.metric("Recovered", f"{data['Recovered']:,}")
    col3.metric("Deaths", f"{data['Deaths']:,}")

    if data["Risk_Level"] == "High":
        st.error(f"🔴 Risk Level: {data['Risk_Level']}")
    elif data["Risk_Level"] == "Medium":
        st.warning(f"🟡 Risk Level: {data['Risk_Level']}")
    else:
        st.success(f"🟢 Risk Level: {data['Risk_Level']}")

# ================= RISK PREDICTION =================
elif menu == "Risk Prediction":
    st.title("🔮 COVID-19 Risk Prediction")

    c1, c2, c3 = st.columns(3)
    confirmed = c1.number_input("Confirmed Cases", min_value=0)
    deaths = c2.number_input("Deaths", min_value=0)
    recovered = c3.number_input("Recovered", min_value=0)

    c4, c5 = st.columns(2)
    active = c4.number_input("Active Cases", min_value=0)
    new_cases = c5.number_input("New Cases", min_value=0)

    if st.button("🚀 Predict Risk Level"):
        sample = [[confirmed, deaths, recovered, active, new_cases]]
        pred = model.predict(sample)
        result = le.inverse_transform(pred)[0]

        if result == "High":
            st.error("🔴 HIGH RISK")
        elif result == "Medium":
            st.warning("🟡 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")

# ================= ABOUT =================
else:
    st.title("📘 About This Project")
    st.markdown("""
    **COVID-19 Dataset Analysis & Prediction** is a Machine Learning project that:

    - Analyzes global COVID-19 data  
    - Visualizes trends using real graphs  
    - Classifies countries into risk levels  
    - Predicts COVID risk using ML  

    **Technologies Used:**
    - Python  
    - Pandas  
    - Matplotlib  
    - Scikit-learn  
    - Streamlit  

    **Developer:** Vivek Jadhav  
    """)

