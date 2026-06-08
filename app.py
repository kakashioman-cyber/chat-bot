import os
import zipfile
import gdown
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# 1. Konfigurasi Halaman Web Streamlit
st.set_page_config(page_title="Chatbot Sejarah Nasional", page_icon="📜", layout="centered")
st.title("📜 Chatbot Sejarah Nasional Indonesia")
st.write("Tanyakan apa saja tentang sejarah Indonesia!")

# 2. Muat API Key dari .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ API Key tidak ditemukan di file .env!")
    st.stop()

# Konfigurasi API Key untuk pustaka google-generativeai
genai.configure(api_key=api_key)

# 3. Kelas Embedding Kustom disesuaikan dengan model lama
class GeminiEmbeddings:
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # 💡 Samakan dengan ingest.py memakai models/gemini-embedding-001
            response = genai.embed_content(model="models/gemini-embedding-001", content=text, task_type="retrieval_document")
            embeddings.append(response['embedding'])
        return embeddings

    def embed_query(self, text):
        # 💡 Samakan dengan ingest.py memakai models/gemini-embedding-001
        response = genai.embed_content(model="models/gemini-embedding-001", content=text, task_type="retrieval_query")
        return response['embedding']

# 4. Inisialisasi Database ChromaDB
@st.cache_resource
def init_services():
    persistent_directory = "./chroma_db"
    embedding_function = GeminiEmbeddings()

    if not os.path.exists(persistent_directory):
        with st.spinner("Sedang mengunduh database sejarah secara aman..."):
            # Mengambil URL rahasia dari Secrets Streamlit Cloud
            db_url = st.secrets["DATABASE_URL"] 

            # 💡 Menggunakan gdown untuk download anti-corrupt dari Google Drive
            gdown.download(db_url, "chroma_db.zip", quiet=False)

            # Proses ekstraksi otomatis
            with zipfile.ZipFile("chroma_db.zip", 'r') as zip_ref:
                zip_ref.extractall(".")

            os.remove("chroma_db.zip") 

    return Chroma(persist_directory=persistent_directory, embedding_function=embedding_function)

db = init_services()

# 5. Kelola Riwayat Obrolan
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Input Chat dari Pengguna
if user_query := st.chat_input("Ketik pertanyaan sejarah di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari di buku sejarah..."):
            try:
                # A. Ambil dokumen relevan dari database
                docs = db.similarity_search(user_query, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # B. Prompt khusus RAG
                prompt = f"""
                Anda adalah seorang pakar Sejarah Nasional Indonesia yang ramah dan edukatif.
                Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan informasi (konteks) yang disediakan di bawah ini.
                Jika informasi tidak ada di dalam konteks, katakan dengan sopan bahwa informasi tersebut tidak ditemukan di dalam buku referensi. Jangan mengarang jawaban.

                KONTEKS SEJARAH:
                {context}

                PERTANYAAN PENGGUNA:
                {user_query}

                JAWABAN:
                """

                # C. Panggil model Gemini 2.5 Flash cara lama
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                
                bot_response = response.text
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
