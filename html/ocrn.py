import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import requests
import io
import json
import os

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="KidneySync - OCR & CKD Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 KidneySync Smart Health Platform")
st.markdown(
    """
    Welcome to **KidneySync** — an AI-powered platform for:
    - 🩺 **Kidney Disease Prediction**
    - 📸 **Image-to-Text OCR Extraction**
    
    ---
    """
)

# -------------------------
# OCR SECTION
# -------------------------
st.header("📸 OCR - Image or PDF to Text Extraction (Cloud)")

uploaded_file = st.file_uploader("Upload an Image or PDF to Extract Text", type=[
                                 "png", "jpg", "jpeg", "pdf"])

st.markdown(
    """
    <div style="text-align:center; font-weight:bold; font-size:18px; color:#0ea5a4; margin:15px 0;">OR</div>
    """,
    unsafe_allow_html=True,
)

# Replace this with your OCR.Space API key
OCR_API_KEY = "K84723290588957"

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    st.image(file_bytes, caption="Uploaded File", use_container_width=True)

    with st.spinner("🔍 Extracting text... Please wait..."):
        try:
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files={"file": file_bytes},
                data={
                    "apikey": OCR_API_KEY,
                    "language": "eng",
                    "OCREngine": 2
                },
                timeout=60
            )

            result_json = response.json()

            # ✅ Check for errors safely
            if result_json.get("IsErroredOnProcessing"):
                error_msg = result_json.get(
                    "ErrorMessage", ["Unknown error"])[0]
                st.error(f"OCR failed: {error_msg}")
            elif "ParsedResults" in result_json and len(result_json["ParsedResults"]) > 0:
                extracted_text = result_json["ParsedResults"][0]["ParsedText"]
                st.success("✅ Text Extraction Complete!")
                st.text_area("📜 Extracted Text:", extracted_text, height=200)
            else:
                st.error("⚠️ No text detected in the uploaded image.")

        except Exception as e:
            st.error(f"OCR request failed: {str(e)}")

else:
    st.info("⬆️ Please upload an image or PDF above to extract text.")
# -------------------------
# CKD PREDICTION SECTION
# -------------------------
st.header("🩺 Chronic Kidney Disease (CKD) Prediction")

DATA_PATH = r"C:\Users\nicoj\Desktop\Sri Lankan CKD Dataset.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    imputer_num = SimpleImputer(strategy="mean")
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

    imputer_cat = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


data = load_data(DATA_PATH)

# Train model
X = data.drop(columns=["class"], errors='ignore')
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Prediction Form
# -------------------------
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

    submit_btn = st.form_submit_button("🔮 Predict CKD Risk")

if submit_btn:
    input_dict = {col: 0 for col in X.columns}

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
    pred_text = "🟢 Likely Healthy" if prediction == 0 else "🔴 ⚠️ Risk of Kidney Disease"

    st.subheader("Prediction Result:")
    st.markdown(
        f"<h2 style='color:#0ea5a4'>{pred_text}</h2>", unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 KidneySync | Smart Health with AI 🧠")
