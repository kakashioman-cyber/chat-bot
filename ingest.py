# File skrip untuk mengindeks data
# Untuk memproses dokumen agar masuk ke dalam database pencarian.


import os
import time
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# Muat API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ API Key tidak ditemukan! Pastikan file .env sudah benar.")

genai.configure(api_key=api_key)

# Sistem Embedding Batching Cerdas
class GeminiEmbeddings:
    def embed_documents(self, texts):
        embeddings = []
        # Membagi ribuan teks menjadi kelompok kecil (1 batch isi 20 teks)
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                # Kirim 20 teks sekaligus dalam 1 kali panggilan API!
                response = genai.embed_content(
                    model="models/gemini-embedding-001", 
                    content=batch_texts, 
                    task_type="retrieval_document"
                )
                embeddings.extend(response['embedding'])
                print(f"📦 Berhasil memproses vektor data ke {i} sampai {i + len(batch_texts)}")
                
                # Jeda 6 detik antar-batch agar server Google bisa "bernapas"
                time.sleep(6) 
            except Exception as e:
                print(f"⚠️ Terjadi hambatan kuota, menunggu 30 detik untuk pemulihan...")
                time.sleep(30)
                # Coba kembali batch yang gagal
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
    
    persistent_directory = "./chroma_db"
    db = Chroma(persist_directory=persistent_directory, embedding_function=GeminiEmbeddings())
    existing_ids = set(db.get()["ids"]) if os.path.exists(persistent_directory) else set()
    
    new_chunks = []
    new_ids = []
    for chunk in chunks:
        chunk_id = generate_unique_id(chunk)
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)
            
    if new_chunks:
        print(f"🚀 Memulai konversi {len(new_chunks)} vektor data dengan metode Batching...")
        db.add_documents(documents=new_chunks, ids=new_ids)
        print("✅ SELESAI! Database RAG berhasil dibuat utuh di folder './chroma_db'.")
    else:
        print("😎 Semua dokumen sudah aman di database.")

if __name__ == "__main__":
    main()