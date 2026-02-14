# -*- coding: utf-8 -*-
"""
build_foods_chromadb.py

- Reads: C:\\Users\\user\\Desktop\\diyetisyen_llm\\besin_havuzu_eski.normalized.json
  expects: {"foods": [ ... ]} (list of dicts or strings)

- Builds / updates a persistent ChromaDB at:
  C:\\Users\\user\\Desktop\\diyetisyen_llm\\chroma_db_storage

- Stores embeddings using SentenceTransformer (model selectable from CLI).

Install:
  pip install chromadb sentence-transformers

Run:
  python build_foods_chromadb.py ^
    --foods_json "C:\\Users\\user\\Desktop\\diyetisyen_llm\\besin_havuzu_eski.normalized.json" ^
    --persist_dir "C:\\Users\\user\\Desktop\\diyetisyen_llm\\chroma_db_storage" ^
    --collection "foods" ^
    --model "trmteb/turkish-embedding-model-fine-tuned"
"""

import os
import json
import re
import argparse
from typing import Any, Dict, List, Tuple, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_FOODS_JSON = r"C:\Users\user\Desktop\diyetisyen_llm\besin_havuzu_eski.normalized.json"
DEFAULT_PERSIST_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\chroma_db_storage"
DEFAULT_COLLECTION = "foods"

# Default modeli koyuyoruz ama SEN terminalden --model ile veriyorsun.
DEFAULT_MODEL = "trmteb/turkish-embedding-model-fine-tuned"


def norm_plain(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")
    s = re.sub(r"\s+", " ", s)
    return s


def safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_food_id(food: Any, idx: int) -> str:
    """
    Food item dict ise:
      - id / food_id / code vb varsa onu kullan
      - yoksa name üzerinden türet
    String ise:
      - string üzerinden türet
    """
    if isinstance(food, dict):
        for key in ("id", "food_id", "code", "food_code", "uid"):
            v = food.get(key)
            if isinstance(v, (str, int)) and str(v).strip():
                return f"food_{str(v).strip()}"
        name = str(food.get("name") or food.get("turkce") or food.get("title") or "").strip()
        if name:
            return "food_" + re.sub(r"[^a-z0-9_]+", "_", norm_plain(name))[:80]
        return f"food_idx_{idx}"
    if isinstance(food, str):
        return "food_" + re.sub(r"[^a-z0-9_]+", "_", norm_plain(food))[:80]
    return f"food_idx_{idx}"


def build_food_document(food: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Embedding için tek bir metin üret:
    - name
    - synonyms
    - group/category
    - description/notes
    Metadata da sakla.
    """
    if isinstance(food, str):
        name = food.strip()
        doc = name
        meta = {"name": name}
        return doc, meta

    if not isinstance(food, dict):
        doc = str(food)
        meta = {"raw": doc}
        return doc, meta

    name = str(food.get("name") or food.get("turkce") or food.get("title") or "").strip()
    synonyms = food.get("synonyms") or food.get("aliases") or food.get("es_anlamlilar") or []
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    synonyms = [str(x).strip() for x in synonyms if str(x).strip()]

    group_ = str(food.get("group") or food.get("category") or food.get("food_group") or "").strip()
    desc = str(food.get("description") or food.get("note") or food.get("notes") or food.get("content") or "").strip()

    # Bazı havuzlarda nutrient alanları olabilir; text’e hafifçe ekle (çok uzatmadan)
    nutrients = food.get("nutrients") or food.get("macro") or None
    nut_txt = ""
    if isinstance(nutrients, dict) and nutrients:
        # sadece birkaç anahtar
        keys = list(nutrients.keys())[:12]
        parts = []
        for k in keys:
            v = nutrients.get(k)
            if v is None:
                continue
            parts.append(f"{k}:{v}")
        if parts:
            nut_txt = " | nutrients: " + ", ".join(parts)

    pieces = []
    if name:
        pieces.append(f"Besin adı: {name}")
    if synonyms:
        pieces.append("Eş anlamlı/alias: " + ", ".join(synonyms))
    if group_:
        pieces.append(f"Grup: {group_}")
    if desc:
        pieces.append(f"Açıklama: {desc}")
    if nut_txt:
        pieces.append(nut_txt)

    doc = "\n".join(pieces).strip()
    if not doc:
        doc = json.dumps(food, ensure_ascii=False)

    meta: Dict[str, Any] = {
        "name": name,
        "group": group_,
        "synonyms": ", ".join(synonyms) if synonyms else "",
    }
    # metadata boyutu şişmesin diye raw eklemiyoruz
    return doc, meta


def chunked(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foods_json", default=DEFAULT_FOODS_JSON)
    ap.add_argument("--persist_dir", default=DEFAULT_PERSIST_DIR)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)

    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda / cpu / None (auto)")

    ap.add_argument("--recreate", action="store_true", help="Collection'ı silip baştan oluşturur (DİKKAT!)")
    args = ap.parse_args()

    ensure_dir(args.persist_dir)

    print("Foods JSON:", args.foods_json)
    print("Persist dir:", args.persist_dir)
    print("Collection:", args.collection)
    print("Embedding model:", args.model)

    data = safe_load_json(args.foods_json)
    foods = None
    if isinstance(data, dict) and isinstance(data.get("foods"), list):
        foods = data["foods"]
    elif isinstance(data, list):
        # bazı dosyalar direkt liste olabilir
        foods = data
    else:
        raise SystemExit('JSON formatı beklenmeyen. {"foods":[...]} veya direkt [...] olmalı.')

    # --- Chroma client (persistent) ---
    client = chromadb.PersistentClient(
        path=args.persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    # recreate istenirse collection sil
    if args.recreate:
        try:
            client.delete_collection(args.collection)
            print("[OK] Collection silindi:", args.collection)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=args.collection,
        metadata={"source": os.path.basename(args.foods_json), "type": "foods"}
    )

    # --- Embedder ---
    st_kwargs = {}
    if args.device:
        st_kwargs["device"] = args.device
    model = SentenceTransformer(args.model, **st_kwargs)

    # Var olan id’leri çek (update/skip için)
    # Not: Chroma büyükse bu pahalı olabilir; ama foods havuzu genelde manageable.
    existing_ids = set()
    try:
        # get() limit parametresi yok; pagination için include + where kullanmak gerekebilir.
        # Basit yaklaşım: count + iter by offset.
        total = col.count()
        step = 5000
        for offset in range(0, total, step):
            got = col.get(include=[], limit=step, offset=offset)
            ids = got.get("ids") or []
            for _id in ids:
                existing_ids.add(_id)
        print(f"[INFO] Existing docs in collection: {len(existing_ids)}")
    except Exception as e:
        print("[WARN] Existing id scan failed (devam ediyorum):", str(e))

    # hazırlanacak batch'ler
    to_add_ids: List[str] = []
    to_add_docs: List[str] = []
    to_add_metas: List[Dict[str, Any]] = []

    skipped = 0
    for i, food in enumerate(foods):
        fid = get_food_id(food, i)
        if fid in existing_ids:
            skipped += 1
            continue

        doc, meta = build_food_document(food)
        if not doc.strip():
            continue

        # id çakışması olmasın diye garanti
        to_add_ids.append(fid)
        to_add_docs.append(doc)
        to_add_metas.append(meta)

    print(f"[INFO] Total foods: {len(foods)} | new to add: {len(to_add_ids)} | skipped(existing): {skipped}")

    if not to_add_ids:
        print("[OK] Eklenecek yeni veri yok.")
        return

    # batch add
    bs = max(1, int(args.batch_size))
    for b_ids, b_docs, b_metas in zip(
        chunked(to_add_ids, bs),
        chunked(to_add_docs, bs),
        chunked(to_add_metas, bs),
    ):
        embs = model.encode(
            b_docs,
            batch_size=min(64, len(b_docs)),
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        col.add(
            ids=b_ids,
            documents=b_docs,
            metadatas=b_metas,
            embeddings=embs
        )
        print(f"[ADD] {len(b_ids)} docs")

    print("[DONE] Foods collection hazır.")
    print("Persisted at:", args.persist_dir)
    print("Collection:", args.collection)


if __name__ == "__main__":
    main()
