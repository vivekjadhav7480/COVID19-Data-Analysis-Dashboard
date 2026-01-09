# 🦠 COVID-19 Data Analysis & Risk Prediction Dashboard

This project is a **Streamlit-based Machine Learning dashboard** that analyzes global COVID-19 data and predicts country risk levels. It helps users **visualize trends, understand the severity of COVID-19 in different countries, and predict risk levels using ML models**.

## 🌐 Project Overview
The COVID-19 pandemic has impacted countries worldwide. Governments, researchers, and citizens need **accurate data insights** to take timely actions. This dashboard provides a **comprehensive analysis of COVID-19 cases globally** and **predicts country-wise risk levels** based on trends and statistical patterns.

## 🚀 Features
- Interactive global COVID-19 statistics dashboard  
- Top 10 affected countries visualization  
- Country-wise detailed analysis (cases, deaths, recoveries)  
- COVID risk level classification: **Low / Medium / High**  
- Machine Learning-based risk prediction for each country  
- Easy-to-use **interactive Streamlit interface**  

## 🛠 Technologies Used
- **Python** – Main programming language  
- **Streamlit** – For building interactive web dashboard  
- **Pandas** – Data cleaning and manipulation  
- **Matplotlib / Seaborn** – Data visualization  
- **Scikit-learn** – Machine Learning for risk prediction  

## 📊 Dataset
- COVID-19 Country-wise Dataset (from [Kaggle](https://www.kaggle.com/))  
- Includes daily updates of confirmed cases, deaths, and recovered cases for all countries  

## 📈 Methodology
1. **Data Collection & Cleaning** – Imported dataset, handled missing values, and formatted data.  
2. **Exploratory Data Analysis (EDA)** – Visualized top affected countries, trends, and daily cases.  
3. **Machine Learning Model** – Built a classification model to predict country risk level (Low / Medium / High).  
4. **Dashboard Deployment** – Integrated visualizations and ML predictions into a **Streamlit dashboard**.  

## 💻 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
