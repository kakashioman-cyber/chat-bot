import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
import time

# Konfigurasi Halaman Web Streamlit
st.set_page_config(page_title="Chatbot Sejarah Nasional", page_icon="📜", layout="centered")
st.title("📜 Chatbot Sejarah Nasional Indonesia")
st.write("Tanyakan apa saja tentang sejarah Indonesia berdasarkan buku referensi Anda!")

# Muat API Key dari .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ API Key tidak ditemukan di file .env!")
    st.stop()

# Konfigurasi API Key untuk pustaka google-generativeai
genai.configure(api_key=api_key)

# Kelas Embedding
class GeminiEmbeddings:
    def embed_documents(self, texts):
        embeddings = []
        batch_size = 5  # Kirim maksimal 5 potongan teks saja per gelombang
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Mengirim batch kecil ke Google
                response = genai.embed_content(
                    model="models/gemini-embedding-001", 
                    content=batch, 
                    task_type="retrieval_document"
                )
                # Jika batch mengembalikan list isi list
                if isinstance(response['embedding'][0], list):
                    embeddings.extend(response['embedding'])
                else:
                    embeddings.append(response['embedding'])
                    
            except Exception as e:
                # Jika Google memberikan lampu merah, paksa skrip tidur selama 10 detik
                st.warning("⚠️ Server Google sibuk (Rate Limit). Beristirahat selama 10 detik...")
                time.sleep(10)
                # Coba kirim ulang batch yang gagal tadi
                response = genai.embed_content(
                    model="models/gemini-embedding-001", 
                    content=batch, 
                    task_type="retrieval_document"
                )
                if isinstance(response['embedding'][0], list):
                    embeddings.extend(response['embedding'])
                else:
                    embeddings.append(response['embedding'])
            
            # Berikan jeda napas wajib 3 detik di setiap akhir gelombang batch kecil
            time.sleep(3)
            
        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(model="models/gemini-embedding-001", content=text, task_type="retrieval_query")
        return response['embedding']

# Inisialisasi Database ChromaDB
@st.cache_resource
def init_services():
    # Menggunakan model embedding gratis dan tanpa limit dari HuggingFace
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists("./data") or not os.listdir("./data"):
        st.error("❌ Folder './data' tidak ditemukan atau kosong.")
        return None
        
    from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    with st.spinner("⏳ Memproses dokumen sejarah menggunakan HuggingFace (Tanpa Rate Limit)..."):
        all_docs = PyPDFDirectoryLoader("./data").load() + DirectoryLoader("./data", glob="*.txt", loader_cls=TextLoader).load()
        
        # Diperbesar agar chunk lebih ringkas
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        chunks = text_splitter.split_documents(all_docs)
        
        db = Chroma.from_documents(documents=chunks, embedding=embedding_function)
    
    return db

db = init_services()

# Berhenti jika database gagal dimuat
if db is None:
    st.stop()

# Kelola Riwayat Obrolan
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat dari Pengguna
if user_query := st.chat_input("Ketik pertanyaan sejarah di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari di buku sejarah..."):
            try:
                # Ambil dokumen relevan dari database
                docs = db.similarity_search(user_query, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Prompt khusus RAG
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

                # Panggil model Gemini 2.5 Flash
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                
                bot_response = response.text
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
