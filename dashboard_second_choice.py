import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf  # Untuk TensorFlow
import torch  # Untuk PyTorch

# Library Needed
import pandas as pd
import numpy as np
import time

import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
# pd.set_option("display.max_row", None)

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ================================================================================================================

st.set_page_config(page_title="Quality Control Kepuasan Layanan Institusi", layout="wide", initial_sidebar_state="auto")


# Mendapatkan ID elemen root dari aplikasi Streamlit
root_container = st.container()

# Mengatur ukuran tampilan dengan CSS
root_container.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
        max-width: 100%;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ================================================================================================================

# Mendefinisikan HTML untuk top bar
top_bar = """
<div style="background-color:#333; padding:10px;">
    <h3 style="color:white; text-align:center;">✨PEGIAT MANPRO - KELOMPOK 2🐲</h3>
</div>
"""

# Menampilkan top bar sebagai komponen HTML
st.markdown(top_bar, unsafe_allow_html=True)


# ================================================================================================================

# Sidebar
st.sidebar.title("Tel-U Surabaya")
# Menampilkan gambar dengan posisi di tengah (center)
st.sidebar.markdown("""
    <style>
        div.stSidebar > div:first-child {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)
st.sidebar.image('Source\TELKOM-UNIVERSITY-LOGO-HORIZONTAL.png', caption='Tel-U Surabaya',width=150 , use_column_width=False)  

# ================================================================================================================



# ================================================================================================================

# Menampilkan judul dengan dekorasi yang terpusat
st.markdown(
    """
    <h1 style='text-align: center;'>
        <span style='display: block;'>
            🚀 <strong>Welcome to Our Awesome Dashboard</strong> 🌟
        </span>
        <span style='display: block; font-size: medium;'>
            *Quality Control Kepuasan Layanan Institusi*
        </span>
    </h1>
    """,
    unsafe_allow_html=True
)

# ================================================================================================================

from transformers import pipeline

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

# Streamlit layout
st.title("Sentiment Analysis")

# Input teks untuk prediksi
text_input = st.text_area("Masukkan teks untuk analisis sentimen:")

# Fungsi untuk melakukan prediksi sentimen
def predict_sentiment(text):
    scores = distilled_student_sentiment_classifier(text)[0]
    predicted_class = max(scores, key=lambda x: x['score'])['label']
    return predicted_class

# Melakukan prediksi dan menampilkan hasil
if st.button("Predict"):
    if text_input:
        prediction = predict_sentiment(text_input)
        st.write(f"Sentimen yang Diprediksi: {prediction}")

# ================================================================================================================

# Fungsi untuk menampilkan tautan ke Google Drive
def show_drive_link():
    # st.markdown("[Link ke Google Drive](https://drive.google.com/drive/folders/1oq3Bnho6qgvOkinyBhvL9eLg9cV9_cZI?usp=drive_link)")
    link = "https://drive.google.com/drive/folders/1oq3Bnho6qgvOkinyBhvL9eLg9cV9_cZI?usp=drive_link"
    # Mengarahkan ke link saat tombol diklik
    js = f"window.open('{link}')"
    
    # CSS for the button and hover effect
    css = """
    <style>
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        transition-duration: 0.4s;
        border-bottom: 1px solid transparent;
    }
    .button:hover {
        text-decoration: none;        
        background-color: #FFFFFF;
        border: 1px solid #45a049;
    }
    </style>
    """
    # HTML button with CSS classes
    html = f"""
    <a href="{link}" target="_blank" class="button">Get Telkom Univeristy Surabaya Data</a>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.title("Download data on G-drive Below")
    st.write("If you have access then you will know the password")
    # Kata sandi yang benar
    correct_password = "Telkom University Surabaya"  # Ganti dengan kata sandi yang diinginkan

    # Meminta kata sandi dari pengguna
    password_attempt = st.text_input("Masukkan kata sandi:", type="password")

    # Cek kata sandi
    if password_attempt == correct_password:
        show_drive_link()
    elif password_attempt != "" and password_attempt != correct_password:
        st.error("Kata sandi salah. Silakan coba lagi.")
    # Jika password_attempt masih kosong, tidak tampilkan pesan kesalahan

if __name__ == "__main__":
    main()

# ================================================================================================================

st.write("Or if you don't have the access, use dummy data below")
# Fungsi untuk mengarahkan ke link
def open_link():
    link = "https://drive.google.com/drive/folders/1Lselx1xTGuZArPmMjnTTcsP1_UneMwN7?usp=sharing"
    # Mengarahkan ke link saat tombol diklik
    js = f"window.open('{link}')"
    
    # CSS for the button and hover effect
    css = """
    <style>
    .button {
        background-color: #4CAF50;
        text-color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        transition-duration: 0.4s;
        border-bottom: 1px solid transparent;
    }
    .button:hover {
        text-decoration: none;        
        background-color: #FFFFFF;
        border: 1px solid #45a049;
    }
    </style>
    """
    # HTML button with CSS classes
    html = f"""
    <a href="{link}" target="_blank" class="button">Get Dummy Data</a>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

if __name__ == "__main__":
    open_link()

# ================================================================================================================

# Streamlit layout
st.title("Data File Reader.")

# Upload file Excel
uploaded_file = st.file_uploader("Upload your file here:", type=(["csv","txt","xlsx","xls"]))

st.warning("Warning: upload the data for give you the result!")
# Jika file sudah diunggah
if uploaded_file is not None:
    # Memuat file Excel menjadi DataFrame
    df = pd.read_excel(uploaded_file)
    
# ================================================================================================================    
    
    # Streamlit layout
    st.title("Progress Bar for Data Preprocessing")

    # Mendapatkan jumlah baris dataframe
    total_rows = len(df)

    # Menggunakan progress bar untuk proses
    progress_bar = st.progress(0)

    # Pemrosesan data yang memerlukan waktu
    for i in range(total_rows):
        # Proses yang memerlukan waktu (simulasi)
        time.sleep(0.001)
        # Update progress bar
        progress_bar.progress((i + 1) / total_rows)

    # Menampilkan pesan ketika selesai
    st.success('Data Preprocessing Completed!')
    
        # Prepocessing df-nya
    df.dropna(inplace = True) # drop saja, karena cuman satu data yang Nan.
    df.reset_index(drop = True, inplace = True)
# ================================================================================================================    

    def main():

        # Multi-select untuk memilih Prodi
        list_prodi = df['Prodi'].unique().tolist()
        global selected_prodi
        selected_prodi = st.sidebar.multiselect("Pilih Prodi:", list_prodi)

        # Multi-select untuk memilih Fakultas
        list_fakultas = df['Fakultas'].unique().tolist()
        global selected_fakultas
        selected_fakultas = st.sidebar.multiselect("Pilih Fakultas:", list_fakultas)

        # Logika untuk menampilkan DataFrame berdasarkan pilihan pengguna
        if not selected_prodi and not selected_fakultas:
            global filtered_df
            filtered_df = df.copy()
        else:
            # Logika filter DataFrame berdasarkan pilihan pengguna
            filtered_df = df[
                ((df['Prodi'].isin(selected_prodi)) | (df['Fakultas'].isin(selected_fakultas))) |
                ((df['Prodi'].isin(selected_prodi)) & (df['Fakultas'].isin(selected_fakultas)))
            ]

    if __name__ == "__main__":
        main()

# ================================================================================================================    
    
    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")
    # Display the dataframe and allow the user to stretch the dataframe
    # across the full width of the container, based on the checkbox value
    st.write("Data from Excel file:") # Menampilkan DataFrame
    st.dataframe(filtered_df, use_container_width=st.session_state.use_container_width)

    # ================================================================================================================

    # Menghitung jumlah total program studi
    programs = filtered_df['Prodi'].unique()
    total_programs = len(programs)

    # Menghitung jumlah total saran yang diterima
    total_suggestions = len(filtered_df)

    # Menghitung jumlah total program studi
    fakultas = filtered_df['Fakultas'].unique()
    total_fakultas = len(fakultas)

    # Membagi layar menjadi dua kolom
    col1, col2, col3 = st.columns(3)

    # Menampilkan informasi jumlah program studi di kolom kiri
    with col1:
        st.metric(label="Total Program Studi:", value=total_programs, delta="🧑‍🚀")

    # Menampilkan informasi jumlah saran di kolom kanan
    with col2:
        st.metric(label="Total Saran Masuk:", value=total_suggestions, delta="🙏")

    # Menampilkan informasi jumlah saran di kolom kanan
    with col3:
        st.metric(label="Total Fakultas:", value=total_fakultas, delta="🏢")
        
    # ================================================================================================================

    # Melakukan operasi pada setiap DataFrame dalam list
    filtered_df['JAWABAN'] = filtered_df['JAWABAN'].apply(lambda x: x.lower() if isinstance(x, str) else x)

    # ================================================================================================================
    filtered_df = filtered_df.reset_index(drop = True)
    # Streamlit layout
    st.title("Select Saran Dari Mahasiswa by Index")

    # Pilihan slider untuk memilih indeks baris
    selected_row = st.slider("Select Row Index", 0, len(filtered_df) - 1, 0)

    # Menampilkan jawaban berdasarkan indeks yang dipilih
    st.write(f"Selected Row ({selected_row}):")
    st.write(filtered_df['JAWABAN'][selected_row])

    # ================================================================================================================
    # Lanjutkan Preprocessing Data
    # -------- Drop "Jawaban" tidak berguna --------
    # Daftar nilai yang ingin dihapus dari kolom 'JAWABAN' (case insensitive)
    values_to_drop = ['tidak ada', 'v', '-', '--', '---', 'belum ada', '.', 'tidak ada saran', 'belum ada saran', 'belom ada', 'n', 'tdk ada', 'nothing']
    # Mengonversi nilai-nilai dalam values_to_drop menjadi lowercase
    values_to_drop_lower = [value.lower() for value in values_to_drop]
    filtered_df = filtered_df[~filtered_df['JAWABAN'].str.lower().isin(values_to_drop_lower)]
    filtered_df.reset_index(drop = True, inplace = True)
    # -------- Menghapus angka --------
    import re
    # Fungsi untuk menghapus angka dari string
    def remove_numbers(text):
        if isinstance(text, str):  # Pastikan teks adalah string
            return re.sub(r'\d+', '', text)  # Menghapus angka dari teks
        else:
            return str(text)  # Mengonversi ke string jika tipe data lainnya
    # -------- Menghapus tanda baca --------
    # Menggunakan apply() pada kolom 'JAWABAN' untuk menghapus angka
    filtered_df['JAWABAN'] = filtered_df['JAWABAN'].apply(remove_numbers)
    # Fungsi untuk menghapus tanda baca dari string
    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", string.punctuation))
    # Menggunakan apply() pada kolom 'JAWABAN' untuk menghapus tanda baca
    filtered_df['JAWABAN'] = filtered_df['JAWABAN'].apply(remove_punctuation)

    # ================================================================================================================

    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    # -------- Mengambil stopwords dari NLTK --------
    stop_words_nltk = set(stopwords.words('indonesian'))
    stemmer = PorterStemmer()

    # -------- Mengambil stopwords dari Sastrawi --------
    stop_factory = StopWordRemoverFactory()
    stopword_remover = stop_factory.create_stop_word_remover()
    more_stopword = ['Kampus', 'surabaya', 'telkom', '...']  # Stopwords tambahan

    # -------- Menggabungkan Ssatrawi dan NLTK stopword --------
    stop_words_sastrawi = stop_factory.get_stop_words() + more_stopword
    stop_words_combined = stop_words_nltk.union(stop_words_sastrawi)
    stopword_id = list(stop_words_combined)

    # -------- Function untuk apply ke dataframe --------
    def preprocess_text(text):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stopword_id]
        # stemmed_words = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

    # -------- Preprocessing ke kolom 'JAWABAN' --------
    filtered_df['JAWABAN'] = filtered_df['JAWABAN'].apply(preprocess_text)

    # ================================================================================================================

    # # -------- Mendapatkan list Prodi unik dan menambahkan opsi "Semua" -------- 
    # list_prodi = ['Semua'] + df['Prodi'].unique().tolist()

    # # -------- Tampilkan radio button untuk memilih Prodi -------- 
    # selected_prodi = st.sidebar.radio("Pilih Prodi:", list_prodi)

    # # -------- Logika untuk menangani pilihan "Semua" atau slicing DataFrame berdasarkan Prodi yang dipilih -------- 
    # if selected_prodi == 'Semua':
    #     filtered_df = df  # -------- Menampilkan keseluruhan DataFrame jika "Semua" dipilih -------- 
    # else:
    #     filtered_df = df[df['Prodi'] == selected_prodi]  # -------- Melakukan slicing DataFrame berdasarkan Prodi yang dipilih  ------

    # # -------- Tampilkan DataFrame hasil slicing atau keseluruhan DataFrame -------- 
    # st.write(f"Dataframe Berdasarkan Prodi '{selected_prodi}':")
    # st.write(filtered_df)

    # ================================================================================================================

    # -------- Menggabungkan semua teks dari kolom "JAWABAN" dalam satu string --------
    text_combined_df = ' '.join(filtered_df['JAWABAN'].astype(str).values)

    # -------- Pisahkan kembali menjadi list setiap kata yang ada --------
    text_separated_df = word_tokenize(text_combined_df)

    # -------- Cek kemunculan setiap kata yang ada --------
    kemunculan_df = nltk.FreqDist(text_separated_df)

    # -------- Mendapatkan 10 kata yang paling umum --------
    common_words = kemunculan_df.most_common(10)

    # -------- Menampilkan hasil dalam DataFrame --------
    df_common_words = pd.DataFrame(common_words, columns=['Kata', 'Kemunculan'])

    # ================================================================================================================

    # -------- Buat WordCloud untuk kemunculan kata-kata --------
    wordcloud_df = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(kemunculan_df)

    # -------- Tampilkan WordCloud di Streamlit --------
    st.title("Word Cloud untuk Kemunculan Kata-Kata")
    st.image(wordcloud_df.to_array(), use_column_width=True)

    # ================================================================================================================

    # -------- Mendapatkan 20 kata yang paling umum dan frekuensinya untuk plotting --------
    kata_df = kemunculan_df.most_common(20)
    kata_umum, frekuensi_umum = zip(*kata_df)

    st.title("Show frequently occurring words")
    # -------- Plot bar chart untuk kata-kata yang umum --------
    plt.figure(figsize=(10, 6))
    bars_umum = plt.bar(kata_umum, frekuensi_umum, color='skyblue')
    plt.xticks(rotation=90)  # -------- Mengatur rotasi label sumbu x agar lebih mudah dibaca --------
    plt.xlabel('Kata')
    plt.ylabel('Frekuensi')
    if selected_prodi and not selected_fakultas:
        plt.title(f'20 Kata Paling Umum saran dari Prodi {selected_prodi}')
    elif selected_prodi and selected_fakultas:
        plt.title(f'20 Kata Paling Umum saran dari {selected_fakultas} Prodi {selected_prodi}')
    elif not selected_prodi and selected_fakultas:
        plt.title(f'20 Kata Paling Umum saran dari seluruh prodi yang ada di {selected_fakultas}')
    plt.grid(True)

    # -------- Menambahkan label ke setiap bar pada kata-kata umum --------
    for bar, freq in zip(bars_umum, frekuensi_umum):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0, str(freq), ha='center', va='bottom')

    plt.tight_layout()

    # -------- Tampilkan grafik di Streamlit --------
    st.pyplot(plt)

    # ================================================================================================================

    # Streamlit layout
    st.title("Check Word in DataFrame")

    # Input teks dari pengguna
    input_word = st.text_input("Enter a word to search in 'JAWABAN':")

    # Memeriksa dan menampilkan kalimat yang mengandung kata tersebut
    result_sentences = filtered_df[filtered_df['JAWABAN'].str.contains(input_word, case=False)]['JAWABAN'].tolist()

    if input_word and result_sentences:
        st.write(f"Sentences containing '{input_word}':")
        for sentence in result_sentences:
            st.write(sentence)
    else:
        st.write(f"No sentences found containing '{input_word}' in 'JAWABAN'.")

    # ================================================================================================================


# Mendefinisikan HTML untuk top bar
top_bar = """
<div style="background-color:#333; padding-bottom:15px; padding-left:30px; padding-right:30px">
    <h3 style="color:white; text-align:center;">🐲PEGIAT MANPRO - KELOMPOK 2🐲</h3>
    
    # PROJECT UAS - 'PROJECT MANAJEMENT'
    Anggota1, Anggota2, Anggota3, Anggota4, Anggota5 = st.Team(5)

    with Anggota1: [1206220014]
    st.metric(Name="Heychanda Elnisa Dyhawa", Jabatan=Project Manager, Icon="🧑‍🚀")

    with Anggota2: [1206220011]
    st.metric(Name="Rendika Nurhartanto Suharto", Jabatan=Data Scientist, Icon="👨‍💻")
    
    with Anggota3: [1206220015]
    st.metric(Name="Miranthy Pramita Ningtyas", Jabatan=Visual Communication, Icon="🧑‍🎨")
    
    with Anggota4: [1206220017]
    st.metric(Name="Shabiha Rahma Fauziah", Jabatan=Research Development, Icon="🧛")
    
    with Anggota5: [1206220009]
    st.metric(Name="Halim Arif Cahyono", Jabatan=Risk Manager, Icon="🧑‍✈️")
</div>
"""

# Menampilkan top bar sebagai komponen HTML
st.markdown(top_bar, unsafe_allow_html=True)