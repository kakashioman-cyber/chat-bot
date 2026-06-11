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

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

if not api_key or not UPSTASH_URL or not UPSTASH_TOKEN:
    raise ValueError("❌ Periksa kembali file .env Anda! API Key atau Kredensial Upstash kosong.")

genai.configure(api_key=api_key)

# Fungsi pembantu untuk membuat embedding dari teks (Batch isi 20)
def dapatkan_gemini_embeddings(texts):
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
            print(f"📦 Berhasil memproses vektor Gemini ke {i} sampai {i + len(batch_texts)}")
            time.sleep(6) 
        except Exception as e:
            print(f"⚠️ Kuota habis, menunggu 30 detik untuk pemulihan...")
            time.sleep(30)
            response = genai.embed_content(
                model="models/gemini-embedding-001", 
                content=batch_texts, 
                task_type="retrieval_document"
            )
            embeddings.extend(response['embedding'])
    return embeddings

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
    
    print(f"📄 Memotong teks dokumen menjadi bagian kecil...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    # Inisialisasi Upstash Store dengan embedding=False (Memaksa mode Custom/Raw Vector)
    db = UpstashVectorStore(
        embedding=False,        # ✨ KUNCI UTAMA: Matikan embedding otomatis Upstash
        text_key="text",
        index_url=UPSTASH_URL,
        index_token=UPSTASH_TOKEN
    )
    
    print(f"🚀 Memulai konversi {len(chunks)} bagian dokumen dengan Gemini...")
    list_teks = [chunk.page_content for chunk in chunks]
    vektor_hasil_gemini = dapatkan_gemini_embeddings(list_teks)
    
    # Susun data sesuai format objek Upstash
    vektor_siap_kirim = []
    for i, chunk in enumerate(chunks):
        chunk_id = generate_unique_id(chunk)
        # Menyatukan teks asli + metadata agar tersimpan di cloud
        metadata_gabungan = chunk.metadata.copy()
        metadata_gabungan["text"] = chunk.page_content 
        
        vektor_siap_kirim.append((
            chunk_id,                 # ID Unik
            vektor_hasil_gemini[i],   # Array Angka Koordinat Vektor
            metadata_gabungan         # Teks dan Metadata dokumen
        ))
        
    print("🔌 Mengunggah kumpulan vektor kustom ke Cloud Upstash...")
    # Kirim menggunakan fungsi objek indeks internal Upstash
    db.index.upsert(vectors=vektor_siap_kirim)
    print("✅ SELESAI! Seluruh database RAG berhasil disimpan di Upstash Cloud!")

if __name__ == "__main__":
    main()
