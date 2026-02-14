# -*- coding: utf-8 -*-
"""
Retriever-Only Benchmark (Chroma, LLM YOK) — TEST SET DEĞİŞTİRMEDEN

Bu sürüm, test_seti.json'da doc_id/chunk_id ground-truth olmadığı için
retriever'ı "topic-based" ölçer:

- HitRate@k: expected_topics'ten en az 1 tanesi top-k sonuçlarda geçiyor mu?
- Recall@k: expected_topics'in kaç tanesi top-k sonuçlarda geçiyor? (oran)

Sonuçlar:
- retriever_test/retriever_details.jsonl
- retriever_test/per_question_retriever.csv
- retriever_test/retriever_summary.csv
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# =========================
# CONFIG (SENİN YOLLARIN)
# =========================
TEST_SET_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\test_seti.json"

CHROMA_STORAGE_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\chroma_db_storage"
COLLECTION_NAME = "diyetisyen_rehberi"
BEST_MODEL_NAME = "trmteb/turkish-embedding-model-fine-tuned"

# K seviyeleri
K_LEVELS = [3, 5, 10]

# output folder
OUT_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\retriever_test"


# =========================
# HELPERS
# =========================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

TR_FOLD_MAP = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "I": "i", "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})

def normalize_tr(s: str) -> str:
    s = (s or "").strip().translate(TR_FOLD_MAP).lower()
    s = re.sub(r"\s+", " ", s)
    return s

def extract_expected_topics(sample: Dict[str, Any]) -> List[str]:
    """
    test_seti.json içinden expected_topics alır.
    Eğer yoksa boş döner.
    """
    meta = sample.get("meta", {}) or {}
    topics = meta.get("expected_topics", [])
    topics = [normalize_tr(str(t)) for t in safe_list(topics) if str(t).strip()]
    # unique (sıra korunarak)
    seen = set()
    out = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# =========================
# CHROMA
# =========================
def build_chroma_collection():
    if not os.path.isdir(CHROMA_STORAGE_PATH):
        raise FileNotFoundError(f"Chroma storage bulunamadı: {CHROMA_STORAGE_PATH}")

    client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=BEST_MODEL_NAME)

    try:
        col = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception as e:
        raise RuntimeError(
            f"Chroma collection bulunamadı: {COLLECTION_NAME}. "
            f"Önce index/ingest çalıştırılmalı. Orijinal hata: {repr(e)}"
        )
    return col


def retrieve(collection, query: str, top_k: int) -> List[Dict[str, Any]]:
    res = collection.query(
        query_texts=[query],
        n_results=int(top_k),
        include=["documents", "metadatas", "distances"]
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(len(docs)):
        meta = metas[i] or {}
        out.append({
            "rank": i + 1,
            "id": str(ids[i]) if i < len(ids) and ids[i] is not None else None,
            "distance": float(dists[i]) if i < len(dists) and dists[i] is not None else None,
            "doc_id": str(meta.get("doc_id")) if meta.get("doc_id") is not None else None,
            "ana_baslik": meta.get("ana_baslik"),
            "section": meta.get("section"),
            "topic_group": meta.get("topic_group"),
            "text": docs[i],
        })
    return out


# =========================
# TOPIC-BASED METRICS
# =========================
def build_retrieved_blob(retrieved: List[Dict[str, Any]]) -> str:
    """
    Top-k sonuçların meta+text birleşimi (arama için).
    """
    parts = []
    for r in retrieved:
        parts.extend([
            r.get("doc_id") or "",
            r.get("ana_baslik") or "",
            r.get("section") or "",
            r.get("topic_group") or "",
            r.get("text") or "",
        ])
    return normalize_tr(" ".join(parts))

def topic_hit_and_recall(retrieved_blob: str, expected_topics: List[str]) -> Tuple[Optional[int], Optional[float], Dict[str, Any]]:
    """
    HitRate@k: expected_topics'ten >=1 tanesi blob içinde geçiyor mu?
    Recall@k: bulunan_topic_sayısı / toplam_topic
    expected_topics yoksa -> None
    """
    if not expected_topics:
        return None, None, {"reason": "no_expected_topics"}

    found = []
    missing = []
    for t in expected_topics:
        if not t:
            continue
        # basit substring; istersen regex word-boundary eklenir
        if t in retrieved_blob:
            found.append(t)
        else:
            missing.append(t)

    hit = 1 if len(found) > 0 else 0
    recall = (len(found) / len(expected_topics)) if expected_topics else 0.0

    dbg = {
        "expected_topics": expected_topics,
        "found_topics": found,
        "missing_topics": missing,
        "n_expected": len(expected_topics),
        "n_found": len(found),
    }
    return hit, float(recall), dbg


# =========================
# RUN
# =========================
def main():
    ensure_dir(OUT_DIR)

    if not os.path.exists(TEST_SET_PATH):
        raise FileNotFoundError(f"TEST_SET_PATH bulunamadı: {TEST_SET_PATH}")

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    collection = build_chroma_collection()

    details_jsonl_path = os.path.join(OUT_DIR, "retriever_details.jsonl")
    per_q_csv_path = os.path.join(OUT_DIR, "per_question_retriever.csv")
    summary_csv_path = os.path.join(OUT_DIR, "retriever_summary.csv")

    per_question_rows: List[Dict[str, Any]] = []

    with open(details_jsonl_path, "w", encoding="utf-8") as jf:
        for sample in test_data:
            qid = sample.get("id")
            query = (sample.get("query") or "").strip()

            expected_topics = extract_expected_topics(sample)

            for k in K_LEVELS:
                retrieved = retrieve(collection, query, top_k=k)
                blob = build_retrieved_blob(retrieved)
                hit, recall, dbg = topic_hit_and_recall(blob, expected_topics)

                per_question_rows.append({
                    "id": qid,
                    "k": int(k),
                    "query": query,
                    "expected_topics": "|".join(expected_topics) if expected_topics else "",
                    "hit_rate@k_item": hit,          # 0/1
                    "recall@k_item": recall,         # 0..1
                    "n_expected_topics": len(expected_topics),
                    "n_retrieved": len(retrieved),
                    "found_topics": "|".join(dbg.get("found_topics", [])) if isinstance(dbg, dict) else "",
                    "missing_topics": "|".join(dbg.get("missing_topics", [])) if isinstance(dbg, dict) else "",
                    "retrieved_doc_ids": "|".join([r["doc_id"] for r in retrieved if r.get("doc_id")]),
                    "retrieved_ids": "|".join([r["id"] for r in retrieved if r.get("id")]),
                })

                jf.write(json.dumps({
                    "id": qid,
                    "k": int(k),
                    "query": query,
                    "expected_topics": expected_topics,
                    "hit": hit,
                    "recall": recall,
                    "debug": dbg,
                    "retrieved": retrieved,
                }, ensure_ascii=False) + "\n")

    per_q_df = pd.DataFrame(per_question_rows)
    per_q_df.to_csv(per_q_csv_path, index=False, encoding="utf-8-sig")

    # summary
    summary_list = []
    for k in K_LEVELS:
        tmp = per_q_df[per_q_df["k"] == k].copy()
        n_all = len(tmp)

        valid = tmp[tmp["hit_rate@k_item"].notna() & tmp["recall@k_item"].notna()]
        n_valid = len(valid)
        n_skipped = n_all - n_valid

        hit_rate = float(valid["hit_rate@k_item"].mean()) if n_valid > 0 else 0.0
        recall_mean = float(valid["recall@k_item"].mean()) if n_valid > 0 else 0.0

        summary_list.append({
            "k": int(k),
            "n_questions_total": int(n_all),
            "n_valid": int(n_valid),
            "n_skipped_no_expected_topics": int(n_skipped),
            "hit_rate@k": hit_rate,
            "recall@k": recall_mean,
        })

    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print("\n✔ Retriever test bitti! (topic-based)")
    print("Detay JSONL:", details_jsonl_path)
    print("Per-question CSV:", per_q_csv_path)
    print("Summary CSV:", summary_csv_path)
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
