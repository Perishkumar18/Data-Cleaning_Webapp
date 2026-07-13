import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Page config and title
st.set_page_config(page_title="Data Cleaning App", layout="wide")
# Animated gradient background style
# Stylish UI and Sidebar Buttons CSS
# Clean Flat Button Style for Sidebar
st.markdown("""
    <style>
    /* Page background and text */
    body, [data-testid="stAppViewContainer"], [data-testid="stAppBackground"] {
        background-color: #eef4fb !important;
        color: #0f172a;
    }

    /* Main content panel */
    [data-testid="stMain"] .block-container,
    .css-1d391kg,
    .css-1rs6os.edgvbvh3 {
        background-color: #ffffff !important;
        border-radius: 24px !important;
        padding: 1.75rem 1.75rem !important;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08) !important;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f8fbff;
        padding-top: 24px;
        border-right: 1px solid #d6e3f0;
    }

    /* Sidebar buttons */
    div[data-testid="stRadio"] label {
        display: block;
        background-color: #ffffff;
        color: #0f172a !important;
        padding: 14px 16px;
        margin-bottom: 12px;
        border-radius: 16px;
        font-size: 16px;
        font-weight: 600;
        text-align: left;
        transition: all 0.25s ease;
        width: 100%;
        border: 1px solid #d1d5db;
        box-shadow: 0 12px 22px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stRadio"] label * {
        color: #0f172a !important;
    }

    div[data-testid="stRadio"] label:hover {
        background-color: #e4f1ff;
        border-color: #8ab7ff;
        color: #0f172a !important;
    }

    div[data-testid="stRadio"] label[aria-checked="true"] {
        background-color: #0f4c81 !important;
        color: white !important;
        border: 1px solid #0f4c81;
        box-shadow: 0 14px 24px rgba(15, 76, 129, 0.18);
    }

    div[data-testid="stRadio"] label[aria-checked="true"] * {
        color: white !important;
    }

    /* Buttons */
    button, button[kind="primary"], button[data-testid="stButton"], .stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 16px !important;
        border: 1px solid #1e40af !important;
        box-shadow: 0 12px 24px rgba(30, 64, 175, 0.12) !important;
        padding: 0.95rem 1.35rem !important;
        font-weight: 700 !important;
        text-transform: none !important;
        transition: all 0.2s ease !important;
    }

    button:hover, button[kind="primary"]:hover, button[data-testid="stButton"]:hover, .stButton > button:hover {
        background-color: #eff6ff !important;
        border-color: #1e40af !important;
        color: #0f172a !important;
        transform: translateY(-1px) !important;
    }

    button:active, button[kind="primary"]:active, button[data-testid="stButton"]:active, .stButton > button:active {
        background-color: #e0e7ff !important;
        border-color: #1e40af !important;
    }

    /* File uploader browse button */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] .stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #1e40af !important;
        border-radius: 16px !important;
        box-shadow: 0 14px 28px rgba(30, 64, 175, 0.12) !important;
    }

    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] .stButton > button:hover {
        background-color: #eff6ff !important;
        border-color: #1e40af !important;
        color: #0f172a !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar collapse/expand arrow button */
    [data-testid="collapsedControl"] button,
    button[aria-label="Collapse sidebar"],
    button[aria-label="Expand sidebar"],
    button[title="Collapse sidebar"],
    button[title="Expand sidebar"] {
        background-color: #f59e0b !important;
        color: #ffffff !important;
        border: 1px solid #f59e0b !important;
        box-shadow: 0 10px 18px rgba(245, 158, 11, 0.24) !important;
    }

    [data-testid="collapsedControl"] button:hover,
    button[aria-label="Collapse sidebar"]:hover,
    button[aria-label="Expand sidebar"]:hover,
    button[title="Collapse sidebar"]:hover,
    button[title="Expand sidebar"]:hover {
        background-color: #ea9a0a !important;
        border-color: #ea9a0a !important;
    }

    [data-testid="collapsedControl"] svg,
    button[aria-label="Collapse sidebar"] svg,
    button[aria-label="Expand sidebar"] svg,
    button[title="Collapse sidebar"] svg,
    button[title="Expand sidebar"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Success alerts */
    .stAlert,
    div[data-testid="stAlert"] {
        background-color: #dbeafe !important;
        border: 1px solid #0f4c81 !important;
        color: #0f172a !important;
    }

    .stAlert .stAlertContent,
    div[data-testid="stAlert"] .stAlertContent {
        color: #0f172a !important;
    }

    /* Data Info and headings */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6 {
        color: #0f172a !important;
    }

    /* Form inputs */
    textarea, input, select, .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
    }

    /* Section containers */
    .css-1jnxg57, .css-fg4pbf, .css-1os5o2r {
        background-color: #ffffff !important;
        border-radius: 22px !important;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.06) !important;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            padding-top: 16px;
        }
        div[data-testid="stRadio"] label {
            font-size: 15px;
            padding: 12px 14px;
        }
        .css-1d391kg {
            padding: 1rem 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #f8fbff;
        padding-top: 20px;
        border-right: 1px solid #d6e3f0;
    }

    /* Sidebar Title */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #0f4c81 !important;
        text-align: center;
        font-weight: 700;
    }

    /* Sidebar Buttons - Size and Font Style */
    div[data-testid="stRadio"] label {
        display: block;
        background-color: #ffffff;
        color: #0f172a !important;
        padding: 16px;
        margin-bottom: 12px;
        border-radius: 14px;
        font-size: 17px;              /* Font size */
        font-family: 'Segoe UI', sans-serif; /* Font style */
        font-weight: 600;
        text-align: left;
        transition: all 0.25s ease;
        width: 100%;
        border: 1px solid #d1d5db;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
    }

    /* Hover effect */
    div[data-testid="stRadio"] label:hover {
        background-color: #e4f1ff;
        border: 1px solid #8ab7ff;
        color: #0f172a !important;
    }

    /* Selected button */
    div[data-testid="stRadio"] label[aria-checked="true"] {
        background-color: #0f4c81 !important;
        color: white !important;
        border: 1px solid #0f4c81;
    }

    div[data-testid="stRadio"] label[aria-checked="true"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

#

# Render buttons only (no duplicate text divs)
steps = [
    "1. Upload File", "2. View Data", "3. Handle Missing Values",
    "4. Drop Columns", "5. Rename Columns", "6. Change Data Types",
    "7. Remove Duplicates", "8. Encode Categorical Data", "9. Scale Numerical Data",
    "10. Export Cleaned Data", "20. Final Export"
]

if "selected_step" not in st.session_state:
    st.session_state["selected_step"] = steps[0]


st.sidebar.markdown(
    "<h2 style='color: orange; text-align: center;'>Data Cleaning Steps</h2>",
    unsafe_allow_html=True
)
# Custom sidebar menu (Styled)
step = st.sidebar.radio("", steps, index=steps.index(st.session_state["selected_step"]))

st.session_state["selected_step"] = step


# Welcome Lottie animation (only once)
# Welcome Lottie animation (only once)
if "first_load" not in st.session_state:
    st.session_state["first_load"] = True

if st.session_state["first_load"]:
    st.markdown("## **Welcome to Super Data Cleaner!**")
    st.components.v1.iframe(
        "https://lottie.host/embed/f3cdd3e7-00f4-4544-9568-432c391bccee/loRppWNsro.lottie", height=300
    )

    # Stylish welcome text under animation
    st.markdown("""
        <h2 style='text-align: center; color: #0072ff; font-weight: bold; margin-top: -20px;'>
            🦸‍♂️ Welcome  <span style="color:#00c6ff;">Analysisman</span>!
        </h2>
    """, unsafe_allow_html=True)

    st.session_state["first_load"] = False


# Change log
if "change_log" not in st.session_state:
    st.session_state["change_log"] = []

def log_change(text):
    st.session_state["change_log"].append(text)

# Sidebar navigation
# Sidebar Steps (20 buttons, all working)


# Step 1: Upload File
# Step 1: Upload File
if step == "1. Upload File":
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state["df"] = df
            st.success("File uploaded successfully!")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading file: {e}")


# Step 2: View Data
elif step == "2. View Data":
    df = st.session_state.get("df")
    if df is not None:
        st.write("### Data Preview")
        st.dataframe(df)
        st.write("### Data Info")
        buffer = StringIO()
        df.info(buf=buffer)
        st.markdown(
            f"""
            <div style='color: #0f172a; background: #f8fafc; padding: 16px; border-radius: 14px; border: 1px solid #d1d5db; font-family: monospace; white-space: pre-wrap; line-height: 1.5;'>
            {buffer.getvalue()}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("### Summary Statistics")
        st.write(df.describe())
    else:
        st.markdown(
            """
            <div style='color: #0f172a; background-color: #f8fafc; padding: 14px; border-radius: 12px; border: 1px solid #d1d5db;'>
                Please upload a file first.
            </div>
            """,
            unsafe_allow_html=True,
        )

# Step 3: Handle Missing Values
elif step == "3. Handle Missing Values":
    df = st.session_state.get("df")
    if df is not None:
        st.write("Missing Value Counts:")
        st.write(df.isnull().sum())
        method = st.radio("Choose fill method", ["Fill with 0", "Fill with mean", "Fill with median", "Drop rows"])
        if st.button("Apply Missing Value Handling"):
            try:
                if method == "Fill with 0":
                    df.fillna(0, inplace=True)
                elif method == "Fill with mean":
                    df.fillna(df.mean(numeric_only=True), inplace=True)
                elif method == "Fill with median":
                    df.fillna(df.median(numeric_only=True), inplace=True)
                elif method == "Drop rows":
                    df.dropna(inplace=True)
                st.session_state["df"] = df
                st.success("Missing values handled successfully!")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to handle missing values: {e}")
    else:
        st.warning("Please upload a file first.")

# Step 4: Drop Columns
elif step == "4. Drop Columns":
    df = st.session_state.get("df")
    if df is not None:
        columns = df.columns.tolist()
        selected_cols = st.multiselect("Select columns to drop", columns)
        if st.button("Drop Selected Columns"):
            df.drop(columns=selected_cols, inplace=True)
            st.session_state["df"] = df
            st.success("Selected columns dropped.")
            st.dataframe(df)
    else:
        st.warning("Please upload a file first.")

# Step 5: Rename Columns
elif step == "5. Rename Columns":
    df = st.session_state.get("df")
    if df is not None:
        st.write("Current Columns:")
        st.write(df.columns.tolist())
        col_to_rename = st.selectbox("Select column to rename", df.columns)
        new_name = st.text_input("New column name")
        if st.button("Rename Column"):
            df.rename(columns={col_to_rename: new_name}, inplace=True)
            st.session_state["df"] = df
            st.success("Column renamed successfully.")
            st.dataframe(df)
    else:
        st.warning("Please upload a file first.")

# Step 6: Change Data Types
elif step == "6. Change Data Types":
    df = st.session_state.get("df")
    if df is not None:
        col = st.selectbox("Select column to change type", df.columns)
        dtype = st.selectbox("Select new type", ["int", "float", "str"])
        if st.button("Convert Type"):
            try:
                df[col] = df[col].astype(dtype)
                st.session_state["df"] = df
                st.success("Column type converted successfully.")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error converting type: {e}")
    else:
        st.warning("Please upload a file first.")

# Step 7: Remove Duplicates
elif step == "7. Remove Duplicates":
    df = st.session_state.get("df")
    if df is not None:
        if st.button("Remove Duplicates"):
            df.drop_duplicates(inplace=True)
            st.session_state["df"] = df
            st.success("Duplicates removed.")
            st.dataframe(df)
    else:
        st.warning("Please upload a file first.")

# Step 8: Encode Categorical Data
elif step == "8. Encode Categorical Data":
    df = st.session_state.get("df")
    if df is not None:
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        col = st.selectbox("Select categorical column to encode", cat_cols)
        if st.button("Apply Label Encoding"):
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col])
                st.session_state["df"] = df
                st.success("Label encoding applied.")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error encoding column: {e}")
    else:
        st.warning("Please upload a file first.")

# Step 9: Scale Numerical Data
elif step == "9. Scale Numerical Data":
    df = st.session_state.get("df")
    if df is not None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected = st.multiselect("Select columns to scale", num_cols)
        scaler_type = st.selectbox("Select scaler", ["MinMaxScaler", "StandardScaler"])
        if st.button("Scale Selected Columns"):
            try:
                scaler = MinMaxScaler() if scaler_type == "MinMaxScaler" else StandardScaler()
                df[selected] = scaler.fit_transform(df[selected])
                st.session_state["df"] = df
                st.success("Selected columns scaled.")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Scaling failed: {e}")
    else:
        st.warning("Please upload a file first.")

# Step 10: Export Cleaned Data
elif step == "10. Export Cleaned Data":
    df = st.session_state.get("df")
    if df is not None:
        to_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Cleaned CSV", data=to_download, file_name="cleaned_data.csv", mime="text/csv")
    else:
        st.warning("Please upload and clean a file first.")