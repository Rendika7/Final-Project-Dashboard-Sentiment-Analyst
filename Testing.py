import streamlit as st

# Fungsi untuk menampilkan tautan ke Google Drive
def show_drive_link():
    st.markdown("[Link ke Google Drive](https://drive.google.com/drive/folders/1oq3Bnho6qgvOkinyBhvL9eLg9cV9_cZI?usp=drive_link)")

def main():
    st.title("Otentikasi untuk Mengakses Google Drive")

    # Kata sandi yang benar
    correct_password = "Rendika"  # Ganti dengan kata sandi yang diinginkan

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
