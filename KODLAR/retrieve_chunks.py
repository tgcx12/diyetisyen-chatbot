# retrieve_chunks.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG (istersen env/arg ile de alırsın)
# -----------------------------
INDEX_DIR = Path(r"C:\Users\user\Desktop\diyetisyen_llm\rag_index_2")
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "meta.json"

MODEL_NAME = "intfloat/multilingual-e5-base"


# -----------------------------
# LOADERS
# -----------------------------
def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    txt = path.read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        raise ValueError(f"JSON empty: {path}")
    return json.loads(txt)


def _load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    meta = _safe_load_json(meta_path)

    # meta dict formatı ({"0": {...}}) gelirse listeye çevir
    if isinstance(meta, dict):
        # key'ler "0","1"... gibi ise sırala ve liste yap
        items: List[Tuple[int, Dict[str, Any]]] = []
        for k, v in meta.items():
            try:
                ik = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                items.append((ik, v))
        items.sort(key=lambda x: x[0])
        return [v for _, v in items]

    # meta zaten list ise direkt
    if isinstance(meta, list):
        out = []
        for x in meta:
            if isinstance(x, dict):
                out.append(x)
        return out

    raise TypeError(f"Unsupported meta format: {type(meta)}")


def _normalize_query_e5(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return "query: "
    # e5 için query prefix şart
    if not q.lower().startswith("query:"):
        return f"query: {q}"
    return q


# -----------------------------
# SINGLETONS (model/index/meta)
# -----------------------------
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_meta: Optional[List[Dict[str, Any]]] = None


def _ensure_loaded() -> None:
    global _model, _index, _meta

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    if _meta is None:
        _meta = _load_meta(META_PATH)

    if _index is None:
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
        _index = faiss.read_index(str(FAISS_INDEX_PATH))


# -----------------------------
# PUBLIC API
# -----------------------------
def retrieve(query: str, k: int = 12) -> List[Dict[str, Any]]:
    """
    FAISS tabanlı retrieval.
    meta.json içinden chunk bilgilerini döndürür.

    Return schema:
    [
      {
        "id": "...",
        "doc_id": "...",
        "section": "...",
        "content": "...",
        "score": 0.82
      }, ...
    ]
    """
    _ensure_loaded()
    assert _model is not None
    assert _index is not None
    assert _meta is not None

    qtext = _normalize_query_e5(query)

    # SentenceTransformer -> numpy float32
    q_emb = _model.encode([qtext], normalize_embeddings=True)
    if not isinstance(q_emb, np.ndarray):
        q_emb = np.array(q_emb)
    q_emb = q_emb.astype("float32")

    # Search
    scores, indices = _index.search(q_emb, int(k))

    results: List[Dict[str, Any]] = []

    for rank, idx in enumerate(indices[0].tolist()):
        if idx is None or idx < 0:
            continue
        if idx >= len(_meta):
            # index meta dışına çıktıysa atla (meta/index mismatch)
            continue

        chunk = _meta[idx] or {}
        results.append(
            {
                "id": chunk.get("id"),
                "doc_id": chunk.get("doc_id"),
                "section": chunk.get("section"),
                "content": chunk.get("content"),
                "score": float(scores[0][rank]),
            }
        )

    return results


# -----------------------------
# CLI TEST
# -----------------------------
def _pretty_print(res: List[Dict[str, Any]], max_chars: int = 400) -> None:
    for r in res:
        print("\n---", r.get("id"), "score:", round(float(r.get("score") or 0.0), 4))
        print("doc:", r.get("doc_id"), "| section:", r.get("section"))
        c = (r.get("content") or "").strip().replace("\n", " ")
        print(c[:max_chars])


if __name__ == "__main__":
    q = input("Soru: ").strip()
    out = retrieve(q, k=10)
    _pretty_print(out)
