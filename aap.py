import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -------------------------------
# FUNCTION: GENERATE DATASET
# -------------------------------
def generate_data(rows=150):
    np.random.seed(42)

    data = {
        "Age": np.random.randint(18, 60, rows),
        "Income": np.random.randint(20000, 80000, rows),
        "Purchase_Frequency": np.random.randint(1, 15, rows),
        "Avg_Spending": np.random.randint(500, 5000, rows),
        "Last_Purchase_Days": np.random.randint(1, 120, rows),
        "Satisfaction": np.random.randint(1, 6, rows),
        "Discount_Usage": np.random.randint(0, 2, rows),
        "Online_Purchase_Rate": np.random.randint(10, 100, rows),
        "Loyalty_Points": np.random.randint(50, 500, rows),
        "Store_Visits": np.random.randint(1, 20, rows),
        "Churn": np.random.randint(0, 2, rows)
    }

    return pd.DataFrame(data)

# -------------------------------
# APP TITLE
# -------------------------------
st.title("📊 Customer Churn Prediction System")

# -------------------------------
# DATA SOURCE SELECTION
# -------------------------------
st.sidebar.header("📂 Data Source")
data_option = st.sidebar.radio("Choose Data Option", ["Upload CSV", "Use Sample Data"])

# -------------------------------
# LOAD DATA
# -------------------------------
if data_option == "Upload CSV":
    file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

    if file:
        df = pd.read_csv(file)
    else:
        st.warning("Please upload a dataset")
        st.stop()

else:
    st.info("Using auto-generated sample dataset")
    df = generate_data(150)

# -------------------------------
# PREVIEW DATA
# -------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# VALIDATION
# -------------------------------
if "Churn" not in df.columns:
    st.error("Dataset must contain 'Churn' column")
    st.stop()

# -------------------------------
# SPLIT DATA
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

test_size = st.slider("Test Size (%)", 10, 40, 20) / 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# -------------------------------
# MODEL SELECTION
# -------------------------------
st.sidebar.header("⚙️ Model Selection")

model_name = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"]
)

# -------------------------------
# MODEL INITIALIZATION
# -------------------------------
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()

elif model_name == "Random Forest":
    model = RandomForestClassifier()

elif model_name == "KNN":
    k = st.sidebar.slider("K value", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

elif model_name == "SVM":
    model = SVC(probability=True)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader(f"📈 Model: {model_name}")
st.write(f"Accuracy: {round(accuracy * 100, 2)}%")

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
if model_name in ["Decision Tree", "Random Forest"]:
    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance.set_index("Feature"))

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.subheader("🔮 Predict New Customer")

inputs = []
for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    pred = model.predict([inputs])
    prob = model.predict_proba([inputs])

    if pred[0] == 1:
        st.error("Customer likely to CHURN ⚠️")
    else:
        st.success("Customer likely to STAY ✅")

    st.info(f"Churn Probability: {round(prob[0][1]*100, 2)}%")

# -------------------------------
# MODEL COMPARISON
# -------------------------------
st.subheader("📊 Model Comparison")

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

results = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results[name] = accuracy_score(y_test, pred)

results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
st.dataframe(results_df)

st.bar_chart(results_df.set_index("Model"))

# -------------------------------
# DOWNLOAD DATA
# -------------------------------
df["Prediction"] = model.predict(X)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")