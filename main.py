import pandas as pd
df = pd.read_csv("country_wise_latest.csv")
print(df.head())
print(df.info())
# Basic statistics
print(df.describe())

# Top 5 countries by confirmed cases
top_confirmed = df.sort_values(by="Confirmed", ascending=False).head(5)
print(top_confirmed[["Country/Region", "Confirmed"]])
def risk_level(row):
    if row["Confirmed"] > 100000 and row["Deaths"] > 2000:
        return "High"
    elif row["Confirmed"] > 20000:
        return "Medium"
    else:
        return "Low"

df["Risk_Level"] = df.apply(risk_level, axis=1)

print(df[["Country/Region", "Risk_Level"]].head())
from sklearn.preprocessing import LabelEncoder

# Select features
X = df[["Confirmed", "Deaths", "Recovered", "Active", "New cases"]]
y = df["Risk_Level"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sample = [[50000, 1500, 30000, 18000, 1200]]
prediction = model.predict(sample)

print("Predicted Risk Level:", le.inverse_transform(prediction))
