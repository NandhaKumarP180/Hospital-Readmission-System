import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import shap
from streamlit_lottie import st_lottie

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ----------- CSS for styling -----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #f0f0f5;
    }
    .block-container {
        padding: 2rem 3rem !important;
        max-width: 1100px;
        margin: auto;
    }
    h1, h2, h3, h4 {
        font-weight: 600;
    }
    .card {
        background: rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8.5px);
        -webkit-backdrop-filter: blur(8.5px);
    }
    .stButton>button {
        background-color: #6c63ff;
        border-radius: 12px;
        border: none;
        color: white;
        font-weight: 600;
        padding: 12px 30px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4a42c3;
        cursor: pointer;
    }
    .stCheckbox label {
        color: #ddd;
        font-weight: 500;
    }
    .css-1d391kg {
        font-weight: 600;
        font-size: 18px;
    }
    .css-1lcbmhc {
        background-color: #3a3279 !important;
        color: #ddd !important;
    }
</style>
""", unsafe_allow_html=True)

# --------- Helper functions ---------

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def clean_column_names(columns):
    return [col.replace("[", "_").replace("]", "_").replace("<", "").replace(">", "") for col in columns]

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
        
def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

def reset_models():
    saved_models = load_json("trained_models.json")
    for mid in saved_models.keys():
        model_file = f"{mid}.pkl"
        if os.path.exists(model_file):
            os.remove(model_file)
    if os.path.exists("trained_models.json"):
        os.remove("trained_models.json")

def preprocess_data(df, target_col):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    encoded_df = pd.get_dummies(df.drop(columns=[target_col]))
    encoded_df.columns = clean_column_names(encoded_df.columns)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        if col != target_col and col in encoded_df.columns:
            scaler = StandardScaler()
            encoded_df[col] = scaler.fit_transform(df[[col]])
    target = df[target_col]
    if target.dtype == 'object':
        le = LabelEncoder()
        target = le.fit_transform(target)
    else:
        le = None
    return encoded_df, target, le

def train_and_eval_model(model_name, X_train, y_train, X_test, y_test):
    X_train.columns = clean_column_names(X_train.columns)
    X_test.columns = clean_column_names(X_test.columns)
    if model_name == "Balanced Random Forest":
        model = BalancedRandomForestClassifier(random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    else:
        st.error("Unsupported model")
        return None, None
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds, average='weighted'),
        "Precision": precision_score(y_test, preds, average='weighted'),
        "Recall": recall_score(y_test, preds, average='weighted'),
        "Confusion Matrix": confusion_matrix(y_test, preds).tolist()
    }
    return model, metrics

def display_metrics(metrics):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")
    c4.metric("Recall", f"{metrics['Recall']:.4f}")
    st.markdown("#### Confusion Matrix")
    cm = np.array(metrics['Confusion Matrix'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

def plot_eda(df):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Dataset Statistics")
    st.write(f"Shape: {df.shape}")
    st.write(f"Null Values:\n{df.isnull().sum()}")
    st.write(f"Unique Values per Column:\n{df.nunique()}")
    st.markdown("### Sample Data")
    st.dataframe(df.head())
    st.markdown("### Distribution Plots")
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    st.markdown("### Correlation Heatmap")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)        
        st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# Load cool AI Lottie animation
lottie_ai = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_x62chJ.json")

# Authentication data
admin_username = "admin"
admin_password = "admin123"
user_username = "user"
user_password = "user123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None

def login():
    st.markdown(
    "<h1 style='font-weight:bold;'>Health Radar</h1>"
    "<h3 style='margin-top:-10px;'>Predict.. Prevent.. Protect..</h3>",
    unsafe_allow_html=True
)
    st_lottie(lottie_ai, speed=1, height=150)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == admin_username and password == admin_password:
            st.session_state.logged_in = True
            st.session_state.role = "admin"
        elif username == user_username and password == user_password:
            st.session_state.logged_in = True
            st.session_state.role = "user"
        else:
            st.error("Invalid username or password")

def admin_dashboard():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Admin Dashboard")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    data_changed = False
    if uploaded_file:
        if 'uploaded_name' not in st.session_state or st.session_state.uploaded_name != uploaded_file.name:
            st.session_state.uploaded_name = uploaded_file.name
            data_changed = True
            reset_models()
            st.session_state['dataset'] = None
            st.session_state['X_processed'] = None
            st.session_state['y_processed'] = None
            st.session_state['label_enc'] = None
            st.session_state['target_col'] = None
            st.session_state['feature_descriptions'] = {}
            st.success("New dataset detected. Previous trained models cleared.")
        if st.session_state['dataset'] is None:
            df = pd.read_csv(uploaded_file)
            st.session_state['dataset'] = df
            st.success(f"Dataset uploaded ({uploaded_file.name})")

    if 'dataset' in st.session_state and st.session_state['dataset'] is not None:
        df = st.session_state['dataset']
        tab1, tab2, tab3 = st.tabs(["Data & EDA", "Feature Descriptions", "Train & Manage Models"])
        with tab1:
            target_col = st.selectbox("Select Target Column", df.columns.tolist())
            st.session_state['target_col'] = target_col
            show_eda = st.checkbox("Show Data Exploration (EDA)")
            if show_eda:
                plot_eda(df)

            if st.button("Preprocess Dataset"):
                X_processed, y_processed, label_enc = preprocess_data(df, target_col)
                st.session_state['X_processed'] = X_processed
                st.session_state['y_processed'] = y_processed
                st.session_state['label_enc'] = label_enc
                st.success("Dataset preprocessed")

        with tab2:
            st.markdown("### Feature Descriptions")
            desc_file = "feature_descriptions.json"
            descriptions = load_json(desc_file)
            if "feature_descriptions" not in st.session_state:
                st.session_state['feature_descriptions'] = descriptions if descriptions else {}
            descs = {}
            for col in df.columns:
                val = st.text_input(f"Description for '{col}'", value=st.session_state['feature_descriptions'].get(col, ""))
                descs[col] = val
            if st.button("Save Feature Descriptions"):
                save_json(descs, desc_file)
                st.session_state['feature_descriptions'] = descs
                st.success("Saved feature descriptions")

        with tab3:
            model_choices = ["Balanced Random Forest", "Decision Tree", "XGBoost", "SVM", "Naive Bayes"]
            selected_model = st.selectbox("Select model to train", model_choices)
            if st.session_state.get('X_processed') is not None and st.session_state.get('y_processed') is not None:
                if st.button("Train Model"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state['X_processed'], st.session_state['y_processed'], test_size=0.2, random_state=42)
                    model, metrics = train_and_eval_model(selected_model, X_train, y_train, X_test, y_test)
                    if model:
                        saved_models = load_json("trained_models.json")
                        if saved_models == {}:
                            saved_models = {}
                        model_id = f"{selected_model}_{len(saved_models)+1}"
                        joblib.dump(model, f"{model_id}.pkl")
                        saved_models[model_id] = {
                            "model_name": selected_model,
                            "metrics": metrics,
                            "active": False
                        }
                        save_json(saved_models, "trained_models.json")
                        st.success(f"Model trained and saved as {model_id}")
                        display_metrics(metrics)

            st.markdown("---")
            st.markdown("### Model Manager")
            saved_models = load_json("trained_models.json")
            if saved_models:
                for mid, mval in saved_models.items():
                    st.write(f"**Model ID:** {mid} — {mval['model_name']}")
                    display_metrics(mval['metrics'])
                    if not mval.get('active', False):
                        if st.button(f"Mark {mid} as Active"):
                            for k in saved_models.keys():
                                saved_models[k]['active'] = False
                            saved_models[mid]['active'] = True
                            save_json(saved_models, "trained_models.json")
                            st.success(f"{mid} marked as active model — logging out and redirecting to User Login…")
                            st.session_state.logged_in = False
                            st.session_state.role = None
                            st.session_state["preferred_user_login"] = True
                            st.stop()
            else:
                st.info("No trained models yet.")
    else:
        st.info("Upload a dataset to start.")
    st.markdown("</div>", unsafe_allow_html=True)

def user_dashboard():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("User Dashboard - Prediction & Explanation")
    df = st.session_state.get('dataset', None)
    if df is None:
        st.warning("No dataset found. Please ask admin to upload the dataset.")
        return

    saved_models = load_json("trained_models.json")
    active_model_id = None
    for mid, mval in saved_models.items():
        if mval.get("active", False):
            active_model_id = mid
            break
    if active_model_id is None:
        st.warning("No active model available. Please ask admin to activate a model.")
        return

    model = load_model(f"{active_model_id}.pkl")
    target_col = st.session_state.get('target_col', None)
    if target_col is None:
        st.warning("Target column not defined. Please ask admin to select target column.")
        return

    desc_file = "feature_descriptions.json"
    descriptions = load_json(desc_file)

    st.subheader("Input Features")
    user_input = {}
    for col in df.columns:
        if col == target_col:
            continue
        desc = descriptions.get(col, "")
        dtype = df[col].dtype
        label = f"{col} - {desc}" if desc else col
        if np.issubdtype(dtype, np.number):
            val = st.number_input(label, value=float(df[col].median()))
            user_input[col] = val
        else:
            options = df[col].dropna().unique().tolist()
            val = st.selectbox(label, options)
            user_input[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        X_train_cols = st.session_state['X_processed'].columns
        input_encoded = pd.get_dummies(input_df)
        for col in X_train_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[X_train_cols]
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        for col in numeric_cols:
            if col != target_col and col in input_encoded.columns:
                med = df[col].median()
                std = df[col].std() if df[col].std() > 0 else 1
                input_encoded[col] = (input_encoded[col] - med) / std

        pred = model.predict(input_encoded)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_encoded)[0][1]

        label_enc = st.session_state.get('label_enc', None)
        if label_enc:
            pred_label = label_enc.inverse_transform([pred])[0]
        else:
            pred_label = pred

        st.success(f"Prediction: {target_col} = {pred_label}")
        if proba is not None:
            st.info(f"Probability: {proba*100:.2f}%")


# ------------------ Main app --------------------

st.set_page_config(page_title="Hospital Readmission AI Dashboard", layout="wide", page_icon="🤖")

st.sidebar.title("Navigation")
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    if st.session_state.get("preferred_user_login", False):
        page = "Login"
        st.session_state["preferred_user_login"] = False
    else:
        page = st.sidebar.selectbox("Select Page", ["Login"])
else:
    page = st.sidebar.selectbox("Select Page", ["Admin Dashboard", "User Dashboard", "Logout"])

def logout_and_stop():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.stop()

if page == "Login":
    login()
elif page == "Logout":
    if st.sidebar.button("Confirm Logout"):
        logout_and_stop()
elif page == "Admin Dashboard":
    if st.session_state.logged_in and st.session_state.role == "admin":
        admin_dashboard()
    else:
        st.warning("You need admin login to access this page.")
elif page == "User Dashboard":
    if st.session_state.logged_in and st.session_state.role == "user":
        user_dashboard()
    else:
        st.warning("You need user login to access this page.")
