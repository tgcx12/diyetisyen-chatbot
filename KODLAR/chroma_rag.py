# -*- coding: utf-8 -*-
"""
save_to_chromadb_full.py  (istersen dosya adın chroma_rag.py olabilir)

Bu script:
1. 'trmteb' embedding modelini kullanır.
2. JSON'daki meta verileri (doc_id, section, topic_group vb.) korur.
3. Vektörleri 'chroma_db_storage' klasörüne kalıcı olarak kaydeder.
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# --- AYARLAR (Kendi dosya yollarına göre düzenleyebilirsin) ---
JSON_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\merged_all_rag_standardized.json"
CHROMA_STORAGE_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\chroma_db_storage"
COLLECTION_NAME = "diyetisyen_rehberi"
BEST_MODEL_NAME = "trmteb/turkish-embedding-model-fine-tuned"


def main():
    # 1. Veriyi Yükle
    if not os.path.exists(JSON_PATH):
        print(f"Hata: {JSON_PATH} bulunamadı!")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Hata: JSON formatı liste (list) olmalı. Örn: [ {...}, {...} ]")
        return

    print(f"Toplam {len(data)} adet chunk yüklendi.")

    # 2. ChromaDB İstemcisini Yapılandır (Kalıcı Depolama)
    os.makedirs(CHROMA_STORAGE_PATH, exist_ok=True)

    # PersistentClient verilerin klasöre yazılmasını sağlar
    client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)

    # 3. Embedding Fonksiyonunu Tanımla (TRMTEB)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=BEST_MODEL_NAME
    )

    # 4. Koleksiyon Oluştur (Varsa silip temiz bir başlangıç yapıyoruz)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Eski '{COLLECTION_NAME}' koleksiyonu temizlendi.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}  # Benzerlik ölçütü: Kosinüs
    )

    # 5. Verileri Hazırla
    documents = []
    metadatas = []
    ids = []

    print(f"Vektörleştirme ve Kayıt işlemi başlıyor ({BEST_MODEL_NAME})...")

    for entry in tqdm(data):
        if not isinstance(entry, dict):
            continue

        # Vektörlenecek metin (başlık + konu + içerik)
        content_text = (
            f"Başlık: {entry.get('ana_baslik','')} | "
            f"Konu: {entry.get('topic_group','')} | "
            f"İçerik: {entry.get('content','')}"
        )

        # ID: Chroma için string olmalı ve benzersiz olmalı
        _id = entry.get("id", None)
        if _id is None:
            # id yoksa atla (istersen burada kendi id üretimini yazabilirsin)
            continue

        documents.append(content_text)
        ids.append(str(_id))

        # Meta veriler
        metadatas.append({
            "id": str(_id),
            "doc_id": entry.get("doc_id", "Bilinmiyor"),
            "ana_baslik": entry.get("ana_baslik", ""),
            "section": entry.get("section", ""),
            "topic_group": entry.get("topic_group", "")
        })

    if not documents:
        print("Eklenecek döküman bulunamadı (JSON içeriğini kontrol et).")
        return

    # 6. Toplu Ekleme (Batch Insert)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"\n✔ BAŞARILI: {len(documents)} döküman ve meta verileri '{CHROMA_STORAGE_PATH}' klasörüne indekslendi.")
    print("Artık bu veritabanını RAG projenizde kalıcı kaynak olarak kullanabilirsiniz.")


if __name__ == "__main__":
    main()
