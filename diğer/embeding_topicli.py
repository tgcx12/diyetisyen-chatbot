import os, json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

RAG_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\merged_all_rag_standardized.json"
OUT_DIR  = r"C:\Users\user\Desktop\diyetisyen_llm\rag_index_2"
MODEL_NAME = "intfloat/multilingual-e5-base"

def load_chunks(path: str):
    data = json.load(open(path, "r", encoding="utf-8"))
    # Beklenen format: list[dict] veya {"chunks":[...]} gibi olabilir
    if isinstance(data, dict) and "chunks" in data:
        data = data["chunks"]
    if not isinstance(data, list):
        raise ValueError("RAG json formatı list veya {'chunks': list} olmalı")
    # content alanını normalize et
    chunks = []
    for c in data:
        txt = (c.get("content") or c.get("text") or "").strip()
        if not txt:
            continue
        chunks.append({
            "id": c.get("id"),
            "doc_id": c.get("doc_id"),
            "section": c.get("section"),
            "topic_group": c.get("topic_group"),
            "content": txt
        })
    return chunks

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    chunks = load_chunks(RAG_PATH)
    print("Chunks:", len(chunks))

    model = SentenceTransformer(MODEL_NAME)

    # E5 tavsiyesi: passage prefix
    texts = [f"passage: {c['content']}" for c in chunks]

    # embeddings
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine için normalize + inner product
    index.add(emb)

    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    json.dump(chunks, open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("OK ->", OUT_DIR)

if __name__ == "__main__":
    main()
