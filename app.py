import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import UpstashVectorStore
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

# 3. Fungsi Pembuat Vektor untuk Pertanyaan (Query)
def dapatkan_vektor_pertanyaan(text):
    response = genai.embed_content(
        model="models/gemini-embedding-001", 
        content=text, 
        task_type="retrieval_query"
    )
    return response['embedding']

# 4. Inisialisasi Kredensial Upstash Vector 
@st.cache_resource
def init_services():
    upstash_url = st.secrets.get("UPSTASH_VECTOR_REST_URL") or os.getenv("UPSTASH_VECTOR_REST_URL")
    upstash_token = st.secrets.get("UPSTASH_VECTOR_REST_TOKEN") or os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    
    if not upstash_url or not upstash_token:
        st.error("❌ Kredensial Upstash Vector tidak ditemukan!")
        st.stop()
        
    return UpstashVectorStore(
        embedding=False,        # ✨ KUNCI UTAMA: Matikan embedding otomatis Upstash
        text_key="text",
        index_url=upstash_url,
        index_token=upstash_token
    )

db = init_services()

# [BAGIAN 5: KELOLA RIWAYAT OBROLAN - TETAP SAMA]

# 6. Input Chat dari Pengguna (Ada sedikit penyesuaian pada bagian pemanggilan docs)
if user_query := st.chat_input("Ketik pertanyaan sejarah di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari di buku sejarah..."):
            try:
                # A. Ubah pertanyaan menjadi vektor manual dulu
                query_vector = dapatkan_vektor_pertanyaan(user_query)
                
                # B. Cari potongan dokumen berdasarkan koordinat vektor tersebut
                docs = db.similarity_search_by_vector(query_vector, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # C. Prompt khusus RAG (Biarkan ke bawahnya sama seperti kode lama Anda)
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

                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                
                bot_response = response.text
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
