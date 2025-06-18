# klasifikasi.py

import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model_and_metadata():
    """
    Memuat model XGBoost, MinMaxScaler yang sudah dilatih, dan
    mendefinisikan metadata terkait (nama fitur dan metrik evaluasi).
    """
    model = None
    scaler = None

    # Memuat Model XGBoost
    try:
        with open('best_xgb_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: File model 'best_xgb_model.pkl' tidak ditemukan.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        return None, None, None, None

    # Memuat MinMaxScaler
    try:
        with open('minmaxscaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: File scaler 'minmaxscaler.pkl' tidak ditemukan.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Gagal memuat scaler. Error: {e}")
        return None, None, None, None

    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    evaluation_metrics = {
        "Akurasi": 0.9344, "Presisi": 0.90, "Recall": 0.9643,
        "F1-Score": 0.9310, "AUC-ROC": 0.96
    }

    return model, scaler, feature_names, evaluation_metrics