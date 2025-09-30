# main_page.py

import streamlit as st
import pandas as pd
import os
import base64
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --- BAGIAN 1: KONFIGURASI CHATBOT & FUNGSI BANTU ---

# --- FUNGSI CSS DENGAN STYLE BARU (EFEK KACA) ---
def local_css(file_name):
    """Fungsi untuk memuat gambar background DAN menerapkan style glassmorphism."""
    try:
        with open(file_name, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        
        # Style CSS yang Anda inginkan
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, 0.90), rgba(255, 255, 255, 0.90)),
                                  url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-attachment: fixed;
                font-family: 'Segoe UI', sans-serif;
            }}
            /* Targetkan kontainer utama tempat form dan hasil berada */
            .main .block-container {{
                background: rgba(255, 255, 255, 0.8);
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
            }}
            </style>
            """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Peringatan: File gambar latar belakang '{file_name}' tidak ditemukan.")

# Konfigurasi Model AI
# load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")
try:
    # Saat di-deploy, Streamlit akan mencari secret dengan kunci "GOOGLE_API_KEY"
    google_api_key = st.secrets["GOOGLE_API_KEY"] 
except FileNotFoundError:
    # Ini adalah fallback untuk pengembangan lokal, agar tidak error
    # Pastikan Anda masih memiliki file secrets.toml di lokal
    st.error("File secrets tidak ditemukan. Pastikan Anda sudah menambahkannya di pengaturan Streamlit Cloud.")
    google_api_key = None # Atau muat dari .env jika Anda ingin tetap bisa jalan di lokal
    
try:
    if google_api_key:
        chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.7)
    else:
        chat_model = None
except Exception as e:
    st.error(f"Gagal menginisialisasi model Gemini: {e}")
    chat_model = None

# Template ini digunakan SEBELUM prediksi dibuat
generic_prompt_template = """
Anda adalah chatbot asisten kesehatan jantung yang cerdas dan empatik.
Tugas Anda adalah menjawab pertanyaan umum pengguna tentang kesehatan jantung, gaya hidup sehat (diet, olahraga), dan langkah-langkah preventif.
Saat ini Anda belum memiliki data spesifik dari pengguna.
Jawab pertanyaan pengguna berikut dengan ramah dan informatif.
Selalu ingatkan pengguna bahwa Anda adalah AI dan tidak bisa menggantikan nasihat medis profesional, dan sarankan untuk berkonsultasi dengan dokter untuk diagnosis.

Pertanyaan Pengguna: {question}
"""
GENERIC_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=generic_prompt_template,
)


# Template prompt yang dinamis (digunakan SETELAH prediksi)
contextual_prompt_template = """
Anda adalah chatbot asisten kesehatan jantung yang cerdas dan empatik.
Tugas Anda adalah membantu pengguna memahami hasil prediksi risiko penyakit jantung mereka dan menjawab pertanyaan terkait.

Berikut adalah konteks hasil analisis pengguna yang baru saja dibuat:
--- HASIL ANALISIS PENGGUNA ---
{analysis_result}
---------------------------------

Berdasarkan konteks di atas, jawablah pertanyaan pengguna berikut dengan ramah.
Fokuslah pada saran gaya hidup sehat (diet, olahraga), penjelasan istilah medis yang ada di hasil analisis, dan langkah-langkah preventif.
Selalu ingatkan pengguna bahwa Anda adalah AI dan tidak bisa menggantikan nasihat medis profesional, dan sarankan untuk berkonsultasi dengan dokter.

Pertanyaan Pengguna: {question}
"""
CONTEXTUAL_PROMPT = PromptTemplate(
    input_variables=["analysis_result", "question"],
    template=contextual_prompt_template,
)

def generate_text_summary_for_chatbot(user_df, prediction, prediction_proba, saran_list):
    """Membuat ringkasan teks yang akan menjadi konteks awal untuk chatbot."""
    hasil_prediksi_text = "Risiko Tinggi Penyakit Jantung" if prediction[0] == 1 else "Risiko Rendah Penyakit Jantung"
    probabilitas_text = f"{prediction_proba[0][1]:.2%}"

    summary = f"Hasil Prediksi: {hasil_prediksi_text}\n\n"
    summary += f"Probabilitas: {probabilitas_text}\n\n"
    if saran_list:
        summary += "Faktor Risiko yang Teridentifikasi:\n\n"
        for s in saran_list:
            summary += f"• {s.replace('**', '')}\n\n"
    else:
        summary += "Tidak ada faktor risiko utama yang menonjol.\n\n"
    return summary

def render_metric_bar(label, value, unit, normal_range_str, color, value_percentage):
    """Fungsi untuk membuat visualisasi bar."""
    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <strong>{label}: {value} {unit}</strong>
        <div style="background-color: #e9ecef; border-radius: 5px; height: 18px; width: 100%; margin-top: 5px;">
            <div style="background-color: {color}; width: {value_percentage}%; height: 100%; border-radius: 5px; transition: width 0.5s ease-in-out;"></div>
        </div>
        <div style="font-size: 13px; color: #555; margin-top: 3px;"><i>Pedoman Normal: {normal_range_str}</i></div>
    </div>""", unsafe_allow_html=True)


# --- BAGIAN 2: HALAMAN UTAMA YANG TERPADU ---

def run_main_page(model, scaler, feature_names, eval_metrics):
    # Panggil fungsi CSS baru Anda
    local_css("human-heart-design.jpg")
    
    # Judul utama disesuaikan dengan style baru
    st.markdown(
        '<h1 style="text-align: center; color: #34495e;">🫀 Prediksi Keberadaan Penyakit Jantung &<br>🤖 Asisten Kesehatan Jantung</h1>',
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # Inisialisasi state & chat history di awal
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
    if "initial_context" not in st.session_state:
        st.session_state.initial_context = ""
    # Tambahkan pesan sambutan awal jika riwayat chat kosong
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.session_state.chat_history = [
            AIMessage(content="Halo! Saya adalah Asisten Kesehatan Jantung. Silakan isi data pada formulir untuk mendapatkan analisis risiko, atau Anda bisa langsung mengajukan pertanyaan umum tentang kesehatan jantung di bawah ini.")
        ]


    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h4>Masukkan Data Klinis Pasien untuk Prediksi:</h4>", unsafe_allow_html=True)
        with st.form(key='prediction_form'):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input('**Usia (tahun)**', min_value=1, max_value=100, value=29, help="Masukkan usia pasien dalam tahun.")
                sex_desc = st.selectbox('**Jenis Kelamin**', ('Pria', 'Wanita'), help="Pilih jenis kelamin biologis pasien. Faktor risiko bisa berbeda antara pria dan wanita.")
                cp_desc = st.selectbox('**Tipe Nyeri Dada (cp)**', ('1: Typical Angina', '2: Atypical Angina', '3: Non-anginal Pain', '4: Asymptomatic'), help="Pilih tipe nyeri dada yang dialami. Angina adalah nyeri dada akibat kurangnya aliran darah ke jantung. 'Asymptomatic' berarti tidak ada riwayat nyeri dada.")
                trestbps = st.number_input('**Tekanan Darah Istirahat (trestbps)**', min_value=50, max_value=200, value=120, help="Masukkan nilai tekanan darah sistolik (angka atas pada alat tensi) saat pasien sedang beristirahat, diukur dalam mmHg.")
                chol = st.number_input('**Kolesterol Serum (chol)**', min_value=100, max_value=600, value=200, help="Masukkan kadar kolesterol total dalam darah, diukur dalam mg/dl.")
                fbs_desc = st.selectbox('**Gula Darah Puasa > 120 mg/dl (fbs)**', ('Tidak', 'Ya'), help="Pilih 'Ya' jika kadar gula darah pasien setelah berpuasa melebihi 120 mg/dl.")
            with c2:
                restecg_desc = st.selectbox('**Hasil EKG Istirahat (restecg)**', ('0: Normal', '1: ST-T Wave Abnormality', '2: Probable or Definite LVH'), help="Pilih hasil rekaman elektrokardiogram (EKG) saat istirahat. EKG mengukur aktivitas listrik jantung untuk melihat adanya kelainan.")
                thalach = st.number_input('**Detak Jantung Maksimum (thalach)**', min_value=60, max_value=220, value=150, help="Masukkan detak jantung tertinggi (denyut per menit) yang dicapai pasien selama tes stres (misalnya, saat berlari di treadmill).")
                exang_desc = st.selectbox('**Angina Akibat Olahraga (exang)**', ('Tidak', 'Ya'), help="Pilih 'Ya' jika pasien mengalami nyeri dada (angina) saat atau setelah melakukan aktivitas fisik berat.")
                oldpeak = st.number_input('**Depresi ST akibat olahraga (oldpeak)**', min_value=0.0, max_value=6.2, value=1.0, step=0.1, format="%.1f", help="Nilai 'ST depression' dari EKG saat olahraga. Ini mengukur perubahan listrik jantung saat stres. Nilai lebih tinggi (di atas 1.0) bisa mengindikasikan masalah.")
                slope_desc = st.selectbox('**Kemiringan Segmen ST (slope)**', ('1: Upsloping', '2: Flat', '3: Downsloping'), help="Pilih bentuk kemiringan segmen ST pada EKG saat puncak latihan. Ini adalah salah satu indikator dari tes stres jantung.")
                ca_desc = st.selectbox('**Pembuluh Darah Utama (ca)**', ('0 Pembuluh', '1 Pembuluh', '2 Pembuluh', '3 Pembuluh'), help="Masukkan jumlah pembuluh darah utama (0-3) yang terdeteksi mengalami penyempitan signifikan melalui tes seperti angiografi.")
                thal = st.number_input('**Nilai Tes Thallium (thal)**', min_value=0.0, max_value=10.0, value=3.0, step=0.1, format="%.1f", help="Hasil dari Thallium Stress Test. Nilai umum: 3=Normal, 6=Cacat Tetap (Fixed Defect), 7=Cacat Reversibel (Reversible Defect).")
            
            submit_button = st.form_submit_button(label='**Prediksi Sekarang**', use_container_width=True)

    with col2:
        st.info(f"""
        **Performa Model XGBoost:**
        - Akurasi: **{eval_metrics['Akurasi']:.2%}**
        - Presisi: **{eval_metrics['Presisi']:.2%}**
        - Recall: **{eval_metrics['Recall']:.2%}**
        - AUC-ROC: **{eval_metrics['AUC-ROC']:.2f}**
        """)
        st.warning("**Peringatan:** Aplikasi ini adalah alat bantu dan bukan pengganti diagnosis medis profesional. Hasil prediksi bersifat informasional. Selalu konsultasikan dengan dokter untuk diagnosis dan perawatan.")

    if submit_button:
        sex = 1 if sex_desc == 'Pria' else 0
        cp = int(cp_desc.split(':')[0])
        fbs = 1 if fbs_desc == 'Ya' else 0
        restecg = int(restecg_desc.split(':')[0])
        exang = 1 if exang_desc == 'Ya' else 0
        slope = int(slope_desc.split(':')[0])
        ca = int(ca_desc.split(' ')[0])

        user_data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
        user_df = pd.DataFrame([user_data])[feature_names]

        with st.spinner("Menganalisis data Anda..."):
            scaled_user_data_np = scaler.transform(user_df)
            scaled_user_data = pd.DataFrame(scaled_user_data_np, columns=feature_names)
            prediction = model.predict(scaled_user_data)
            prediction_proba = model.predict_proba(scaled_user_data)
            saran = []
            thal_options_desc = {3.0: 'Normal', 6.0: 'Cacat Tetap (Fixed Defect)', 7.0: 'Cacat Reversibel (Reversible Defect)'}
            if user_df['trestbps'].iloc[0] >= 140: saran.append("Tekanan darah tergolong **Hipertensi Stadium 2**.")
            elif user_df['trestbps'].iloc[0] >= 130: saran.append("Tekanan darah tergolong **Hipertensi Stadium 1**.")
            elif user_df['trestbps'].iloc[0] >= 120: saran.append("Tekanan darah **meningkat (elevated)**.")
            if user_df['chol'].iloc[0] >= 240: saran.append("Kadar kolesterol tergolong **tinggi**.")
            elif user_df['chol'].iloc[0] >= 200: saran.append("Kadar kolesterol di **batas tinggi**.")
            if user_df['fbs'].iloc[0] == 1: saran.append("Gula darah puasa **terindikasi tinggi** (> 120 mg/dl), merupakan faktor risiko diabetes.")
            if user_df['cp'].iloc[0] != 4: saran.append(f"Adanya riwayat **nyeri dada** ({cp_desc.split(': ')[1]}).")
            if user_df['exang'].iloc[0] == 1: saran.append("Mengalami **angina (nyeri dada) saat berolahraga**.")
            if user_df['ca'].iloc[0] > 0: saran.append(f"Terdeteksi **{int(user_df['ca'].iloc[0])} pembuluh darah utama menyempit** berdasarkan hasil tes.")
            thal_val = user_df['thal'].iloc[0]
            if thal_val in [6.0, 7.0]: saran.append(f"Hasil Thallium Stress Test ({thal_val}) terindikasi sebagai **faktor risiko ({thal_options_desc.get(thal_val, 'N/A')})**.")

        st.session_state.prediction_made = True
        st.session_state.saran = saran
        st.session_state.user_df = user_df
        st.session_state.prediction = prediction
        st.session_state.prediction_proba = prediction_proba
        
        initial_context = generate_text_summary_for_chatbot(user_df, prediction, prediction_proba, saran)
        st.session_state.initial_context = initial_context
        st.session_state.chat_history = [AIMessage(content=f"Analisis selesai! Berikut adalah ringkasan hasil Anda. Anda dapat menanyakan apa pun terkait hasil ini.\n\n{initial_context}")]
        st.rerun()


    # Bagian ini akan selalu ditampilkan jika prediksi SUDAH dibuat
    if st.session_state.prediction_made:
        st.markdown("---")
        st.subheader("Hasil Analisis dari Data Anda")

        if st.session_state.prediction[0] == 1:
            st.error(f"**Terdeteksi Risiko Tinggi Penyakit Jantung** (Probabilitas: {st.session_state.prediction_proba[0][1]:.2%})")
        else:
            st.success(f"**Terdeteksi Risiko Rendah Penyakit Jantung** (Probabilitas: {st.session_state.prediction_proba[0][1]:.2%})")
        
        with st.expander("📊 Lihat Analisis Detail dan Visualisasi Input", expanded=True):
            if st.session_state.saran:
                st.markdown("##### Faktor Risiko yang Menonjol dari Input Anda:")
                for s in st.session_state.saran: st.markdown(f"• {s}")
                st.markdown("<br><i>Disarankan untuk mendiskusikan faktor-faktor ini dengan dokter Anda untuk evaluasi lebih lanjut.</i>", unsafe_allow_html=True)
            else:
                st.markdown("Berdasarkan input Anda, tidak ada faktor risiko utama yang menonjol secara spesifik. Tetaplah menjaga gaya hidup sehat dan lakukan pemeriksaan rutin.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Visualisasi Input Dibandingkan Rentang Normal Medis:")
            
            val_trestbps = int(st.session_state.user_df['trestbps'].iloc[0])
            color_trestbps = '#28a745' if val_trestbps < 120 else ('#ffc107' if val_trestbps < 130 else ('#fd7e14' if val_trestbps < 140 else '#dc3545'))
            pct_trestbps = min((val_trestbps - 90) / (200 - 90) * 100, 100)
            render_metric_bar("Tekanan Darah Istirahat", val_trestbps, "mmHg", "< 120 mmHg", color_trestbps, pct_trestbps)

            val_chol = int(st.session_state.user_df['chol'].iloc[0])
            color_chol = '#28a745' if val_chol < 200 else ('#fd7e14' if val_chol < 240 else '#dc3545')
            pct_chol = min((val_chol - 125) / (400 - 125) * 100, 100)
            render_metric_bar("Kolesterol Serum", val_chol, "mg/dl", "< 200 mg/dl", color_chol, pct_chol)
            
            val_thalach = int(st.session_state.user_df['thalach'].iloc[0])
            age_val = int(st.session_state.user_df['age'].iloc[0])
            predicted_max_hr = 220 - age_val if age_val > 0 else 220
            color_thalach = '#dc3545'
            if val_thalach > 0.85 * predicted_max_hr: color_thalach = '#28a745'
            elif val_thalach > 0.75 * predicted_max_hr: color_thalach = '#ffc107'
            pct_thalach = min((val_thalach - 70) / (predicted_max_hr - 60) * 100, 100) if val_thalach > 0 else 0
            target_bpm = int(0.85 * predicted_max_hr)
            render_metric_bar("Detak Jantung Maks. Tercapai", val_thalach, "bpm", f">{target_bpm} bpm (target utk usia {age_val})", color_thalach, pct_thalach)

            val_oldpeak = st.session_state.user_df['oldpeak'].iloc[0]
            color_oldpeak = '#28a745' if val_oldpeak < 1.0 else ('#fd7e14' if val_oldpeak <= 2.0 else '#dc3545')
            pct_oldpeak = min((val_oldpeak / 4.0) * 100, 100)
            render_metric_bar("Depresi ST (oldpeak)", val_oldpeak, "", "Normalnya < 1.0", color_oldpeak, pct_oldpeak)


    # --- PERUBAHAN DIMULAI DI SINI ---
    # Bungkus seluruh antarmuka chat dengan kolom untuk mengontrol lebarnya
    st.markdown("---")
    _, chat_container, _ = st.columns([1, 7, 1]) # [spasi_kiri, konten_tengah, spasi_kanan]

    with chat_container:
        # Judul berubah tergantung konteks
        if st.session_state.prediction_made:
            st.markdown("<h3>💬 Diskusikan Hasil Analisis Data Anda dengan Asisten AI</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3>💬 Tanya Jawab Umum dengan Asisten AI</h3>", unsafe_allow_html=True)

        # Menampilkan riwayat chat
        for message in st.session_state.chat_history:
            role = "human" if isinstance(message, HumanMessage) else "ai"
            with st.chat_message(role):
                st.markdown(message.content)

        # Input chat dari pengguna
        if user_question := st.chat_input("Tanyakan sesuatu..."):
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            with st.chat_message("human"):
                st.markdown(user_question)

            with st.spinner("Asisten sedang berpikir..."):
                if st.session_state.prediction_made:
                    # Jika prediksi sudah ada, gunakan prompt kontekstual
                    prompt_to_use = CONTEXTUAL_PROMPT.format(
                        analysis_result=st.session_state.initial_context, 
                        question=user_question
                    )
                else:
                    # Jika belum, gunakan prompt umum
                    prompt_to_use = GENERIC_PROMPT.format(question=user_question)

                if chat_model:
                    response = chat_model.invoke(prompt_to_use)
                    ai_response = response.content
                else:
                    ai_response = "Maaf, koneksi ke model AI gagal."

                st.session_state.chat_history.append(AIMessage(content=ai_response))
                # Jalankan ulang skrip untuk langsung menampilkan balasan AI tanpa perlu input lagi
                st.rerun() 
    # --- PERUBAHAN SELESAI DI SINI ---