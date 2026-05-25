import os
import hashlib
import logging
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Lutfen 'pip install PyMuPDF' komutunu calistirin.")
    fitz = None

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ayarlar
DOCS_DIR = Path("docs")
DB_PATH = Path("chroma_db")
COLLECTION_NAME = "legal_docs"
EMBED_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def get_md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    if not fitz:
        return ""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        log.error(f"PDF okunurken hata: {pdf_path} - {e}")
    return text

def extract_text_from_txt(txt_path: str) -> str:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        log.error(f"TXT okunurken hata: {txt_path} - {e}")
    return ""

def main():
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir()

    files = list(DOCS_DIR.glob("*.pdf")) + list(DOCS_DIR.glob("*.txt"))
    
    if not files:
        log.info(f"'{DOCS_DIR}' klasorunde islenecek belge bulunamadi.")
        return

    log.info("ChromaDB Yukleniyor...")
    client = chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Veritabanindaki mevcut id'leri al (Tekrar islememek icin)
    existing_ids = set()
    total_docs = collection.count()
    if total_docs > 0:
        # ChromaDB tek seferde hepsini getiremezse limit gerekir, ama kucuk sistemlerde calisir
        res = collection.get(include=[])
        existing_ids = set(res["ids"])

    log.info(f"Mevcut veritabani kayit sayisi: {total_docs}")
    log.info(f"Embedder Yukleniyor: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda", local_files_only=False)

    new_chunks = []
    new_embeddings = []
    new_metadatas = []
    new_ids = []

    for file_path in files:
        log.info(f"Isleniyor: {file_path.name}")
        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(str(file_path))
        else:
            text = extract_text_from_txt(str(file_path))

        if not text.strip():
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            chunk_hash = get_md5(chunk)
            doc_id = f"{file_path.name}_{i}_{chunk_hash[:8]}"

            if doc_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(doc_id)
                new_metadatas.append({"doc_id": file_path.name, "chunk_index": i})

    if new_chunks:
        log.info(f"{len(new_chunks)} yeni chunk bulundu. Vektorler hesaplaniyor...")
        embeddings = embedder.encode(new_chunks, show_progress_bar=True).tolist()
        
        log.info("Veritabanina kaydediliyor...")
        collection.add(
            ids=new_ids,
            documents=new_chunks,
            embeddings=embeddings,
            metadatas=new_metadatas
        )
        log.info("Kayit basarili!")
    else:
        log.info("Eklenecek yeni belge veya degisiklik bulunamadi.")

if __name__ == "__main__":
    main()
