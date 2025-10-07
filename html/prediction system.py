import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Kidneysync CKD Prediction",
    page_icon="🩺",
    layout="centered"
)

# -------------------------
# Load Dataset
# -------------------------
DATA_PATH = r"C:\Users\nicoj\Desktop\Sri Lankan CKD Dataset.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Fill missing numeric values
    imputer_num = SimpleImputer(strategy="mean")
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

    # Fill missing categorical values
    imputer_cat = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # Encode categorical columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


data = load_data(DATA_PATH)

# -------------------------
# Model Training
# -------------------------
X = data.drop(columns=["class"], errors='ignore')
y = data["class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Streamlit UI
# -------------------------
st.title("  Kidney Disease Prediction")
st.markdown("Enter your health parameters below:")

with st.form("kidney_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    bp = st.number_input("Blood Pressure", min_value=50,
                         max_value=200, value=80)
    bgr = st.number_input("Blood Glucose Random",
                          min_value=50, max_value=500, value=120)
    bu = st.number_input("Blood Urea", min_value=5, max_value=200, value=20)

    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
    ba = st.selectbox("Bacteria", ["present", "notpresent"])
    sg = st.number_input("Specific Gravity", min_value=1.0,
                         max_value=1.03, value=1.02)
    su = st.number_input("Sugar", min_value=0, max_value=5, value=0)

    submit_btn = st.form_submit_button("Predict")

# -------------------------
# Prediction Logic
# -------------------------
if submit_btn:
    input_dict = {col: 0 for col in X.columns}

    # Encode categorical inputs manually the same way as training
    cat_map = {"normal": 0, "abnormal": 1, "present": 1, "notpresent": 0}

    input_dict.update({
        "age": age,
        "bp": bp,
        "bgr": bgr,
        "bu": bu,
        "rbc": cat_map[rbc],
        "pc": cat_map[pc],
        "pcc": cat_map[pcc],
        "ba": cat_map[ba],
        "sg": sg,
        "su": su
    })

    input_df = pd.DataFrame([input_dict], columns=X.columns)

    prediction = model.predict(input_df)[0]
    pred_text = "Likely Healthy" if prediction == 0 else "⚠️ Risk of Kidney Disease"

    st.subheader("Prediction Result:")
    st.markdown(
        f"<h2 style='color:#0ea5a4'>{pred_text}</h2>", unsafe_allow_html=True)
