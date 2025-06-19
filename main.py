# main.py

import streamlit as st
from klasifikasi import load_model_and_metadata  # Kita masih butuh fungsi ini
from main_page import run_main_page        # File baru yang akan kita buat

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi & Asisten Kesehatan Jantung", layout="wide")

# --- CSS (jika Anda masih ingin menggunakannya, bisa dipindahkan ke unified_page.py) ---
# ... (opsional, bisa dipindahkan)

# --- Aplikasi utama ---
def main():
    # Muat model dan metadata SEKALI saja.
    model, scaler, feature_names, evaluation_metrics = load_model_and_metadata()

    # Periksa apakah model berhasil dimuat sebelum melanjutkan
    if model is None or scaler is None:
        st.error("Aplikasi tidak dapat dimulai karena model atau scaler gagal dimuat.")
        st.stop()

    # Jalankan halaman terpadu yang baru
    run_main_page(model, scaler, feature_names, evaluation_metrics)

    st.markdown('<div style="text-align: center; color: black; margin-top: 50px;">Dibuat dengan ❤️ oleh Jati Tepatasa Bagastakwa (dibantu AI)</div>', unsafe_allow_html=True)


# --- Jalankan aplikasi ---
if __name__ == '__main__':
    main()