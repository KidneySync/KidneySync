import streamlit as st
import pandas as pd
import numpy as np
import re
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Path to Tesseract executable (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="KidneySync CKD Prediction",
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

# -------------------------
# Model Training
# -------------------------
X = data.drop(columns=["class"], errors='ignore')
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# OCR Function
# -------------------------


def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        pages = convert_from_path(uploaded_file)
        for page in pages:
            text += pytesseract.image_to_string(page)
    else:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
    return text


def extract_medical_values(text):
    # Define regex patterns for medical parameters
    patterns = {
        "age": r"Age[:\s]*([0-9]+)",
        "bp": r"Blood Pressure[:\s]*([0-9]+)",
        "bgr": r"Blood Glucose(?: Random)?[:\s]*([0-9]+)",
        "bu": r"Blood Urea[:\s]*([0-9]+)",
        "sg": r"Specific Gravity[:\s]*([0-9.]+)",
        "su": r"Sugar[:\s]*([0-9]+)",
        "rbc": r"Red Blood Cells[:\s]*(normal|abnormal)",
        "pc": r"Pus Cell[:\s]*(normal|abnormal)",
        "pcc": r"Pus Cell Clumps[:\s]*(present|notpresent)",
        "ba": r"Bacteria[:\s]*(present|notpresent)"
    }

    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1)
    return extracted


# -------------------------
# Streamlit UI
# -------------------------
st.title("🩺 Kidney Disease Prediction (KidneySync)")
st.markdown("Enter your health parameters or upload your medical report:")

st.sidebar.header("📄 Upload Report for OCR")
uploaded_file = st.sidebar.file_uploader("Upload Image or PDF", type=[
                                         "jpg", "png", "jpeg", "pdf"])

ocr_data = {}
if uploaded_file:
    with st.spinner("🔍 Analyzing report..."):
        text_data = extract_text_from_file(uploaded_file)
        ocr_data = extract_medical_values(text_data)
    st.sidebar.success("✅ Report analyzed successfully!")
    st.sidebar.write("**Extracted values:**")
    st.sidebar.json(ocr_data)

# -------------------------
# Form Inputs
# -------------------------
with st.form("kidney_form"):
    age = st.number_input("Age", min_value=1, max_value=120,
                          value=int(ocr_data.get("age", 25)))
    bp = st.number_input("Blood Pressure", min_value=50,
                         max_value=200, value=int(ocr_data.get("bp", 80)))
    bgr = st.number_input("Blood Glucose Random", min_value=50,
                          max_value=500, value=int(ocr_data.get("bgr", 120)))
    bu = st.number_input("Blood Urea", min_value=5,
                         max_value=200, value=int(ocr_data.get("bu", 20)))
    sg = st.number_input("Specific Gravity", min_value=1.0,
                         max_value=1.03, value=float(ocr_data.get("sg", 1.02)))
    su = st.number_input("Sugar", min_value=0,
                         max_value=5, value=int(ocr_data.get("su", 0)))

    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"],
                       index=0 if ocr_data.get("rbc", "normal") == "normal" else 1)
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"],
                      index=0 if ocr_data.get("pc", "normal") == "normal" else 1)
    pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"],
                       index=0 if ocr_data.get("pcc", "notpresent") == "present" else 1)
    ba = st.selectbox("Bacteria", ["present", "notpresent"],
                      index=0 if ocr_data.get("ba", "notpresent") == "present" else 1)

    submit_btn = st.form_submit_button("Predict")

# -------------------------
# Prediction Logic
# -------------------------
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
    pred_text = "✅ Likely Healthy" if prediction == 0 else "⚠️ Risk of Kidney Disease"

    st.subheader("Prediction Result:")
    st.markdown(
        f"<h2 style='color:#0ea5a4'>{pred_text}</h2>",
        unsafe_allow_html=True
    )
