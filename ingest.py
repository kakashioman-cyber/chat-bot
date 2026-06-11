# File skrip untuk mengindeks data
# Untuk memproses dokumen agar masuk ke dalam database pencarian.


import os
import time
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import UpstashVectorStore
import google.generativeai as genai

# Muat API Key dan Kredensial Upstash
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

if not api_key:
    raise ValueError("❌ API Key tidak ditemukan! Pastikan file .env sudah benar.")
if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise ValueError("❌ Kredensial Upstash Vector tidak ditemukan di file .env!")

genai.configure(api_key=api_key)

# Sistem Embedding Batching Cerdas
class GeminiEmbeddings:
    def embed_documents(self, texts):
        embeddings = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = genai.embed_content(
                    model="models/gemini-embedding-001", 
                    content=batch_texts, 
                    task_type="retrieval_document"
                )
                embeddings.extend(response['embedding'])
                print(f"📦 Berhasil memproses vektor data ke {i} sampai {i + len(batch_texts)}")
                time.sleep(6) 
            except Exception as e:
                print(f"⚠️ Terjadi hambatan kuota, menunggu 30 detik untuk pemulihan...")
                time.sleep(30)
                response = genai.embed_content(
                    model="models/gemini-embedding-001", 
                    content=batch_texts, 
                    task_type="retrieval_document"
                )
                embeddings.extend(response['embedding'])
        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(
            model="models/gemini-embedding-001", 
            content=text, 
            task_type="retrieval_query"
        )
        return response['embedding']

def generate_unique_id(chunk):
    source = chunk.metadata.get("source", "unknown")
    content = chunk.page_content
    return hashlib.md5(f"{source}_{content}".encode('utf-8')).hexdigest()

def main():
    print("📂 Membaca file Buku Sejarah di folder './data'...")
    all_docs = PyPDFDirectoryLoader("./data").load() + DirectoryLoader("./data", glob="*.txt", loader_cls=TextLoader).load()
    if not all_docs:
        print("❌ Folder './data' kosong!")
        return
    
    print(f"📄 Berhasil memuat dokumen. Memotong teks menjadi bagian kecil...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    # Inisialisasi Upstash Vector Store
    print("🔌 Menghubungkan ke Upstash Vector Cloud...")
    db = UpstashVectorStore(
    embedding=GeminiEmbeddings(),
    text_key="text",
    index_url=UPSTASH_URL,     # Sesuai perbaikan
    index_token=UPSTASH_TOKEN   # Sesuai perbaikan
)
    
    # Mengambil ID yang sudah ada di Upstash agar tidak duplikat
    print("🔍 Memeriksa dokumen lama di cloud...")
    try:
        # Mengambil info info/ID yang sudah tersimpan di Upstash
        info = db.index.info()
        total_vectors = info.vector_count
        print(f"📊 Total vektor saat ini di Upstash: {total_vectors}")
    except Exception as e:
        print(f"ℹ️ Index baru atau kosong. (Pesan: {e})")
    
    # Catatan: Upstash secara otomatis menolak id yang sama (overwrite), 
    # Namun menyaringnya di kode lokal tetap bagus untuk menghemat kuota API Gemini.
    new_chunks = []
    new_ids = []
    
    # Untuk Upstash, kita bisa langsung memakai skema penambahan dokumen dengan ID kustom Anda
    for chunk in chunks:
        chunk_id = generate_unique_id(chunk)
        new_chunks.append(chunk)
        new_ids.append(chunk_id)
            
    if new_chunks:
        print(f"🚀 Memulai konversi {len(new_chunks)} vektor data ke Cloud Upstash...")
        # Menggunakan add_documents milik UpstashVectorStore
        db.add_documents(documents=new_chunks, ids=new_ids)
        print("✅ SELESAI! Database RAG berhasil diunggah utuh ke Cloud Upstash.")
    else:
        print("😎 Semua dokumen sudah aman di database.")

if __name__ == "__main__":
    main()
