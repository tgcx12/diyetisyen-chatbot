# -*- coding: utf-8 -*-
"""
list_topic_groups_and_doc_ids.py

- merged_all_rag_new.json içindeki tüm chunk'ları okur
- benzersiz topic_group ve doc_id değerlerini çıkarır
- sayıları ve listeleri ekrana basar
- isterse JSON çıktısı da üretir
"""

import json
import argparse
from typing import Any, Dict, List, Set


def safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_chunks(data: Any) -> List[Dict[str, Any]]:
    """
    Olası formatlar:
    - [ {...}, {...} ]
    - { "chunks": [...] }
    - { "rag_chunks": [...] }
    - { "data": [...] }
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for k in ("chunks", "rag_chunks", "data", "items", "documents"):
            if isinstance(data.get(k), list):
                return data[k]

    raise ValueError("Chunk listesi bulunamadı (JSON formatı beklenmeyen)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="merged_all_rag_new.json path")
    ap.add_argument("--out", default=None, help="opsiyonel output json")
    args = ap.parse_args()

    data = safe_load_json(args.inp)
    chunks = iter_chunks(data)

    topic_groups: Set[str] = set()
    doc_ids: Set[str] = set()

    for c in chunks:
        tg = c.get("topic_group")
        di = c.get("doc_id")

        if isinstance(tg, str) and tg.strip():
            topic_groups.add(tg.strip())

        if isinstance(di, str) and di.strip():
            doc_ids.add(di.strip())

    print("\n=== TOPIC GROUPS ===")
    for tg in sorted(topic_groups):
        print("-", tg)

    print("\n=== DOC IDs ===")
    for di in sorted(doc_ids):
        print("-", di)

    print("\n=== COUNTS ===")
    print("Unique topic_group count:", len(topic_groups))
    print("Unique doc_id count     :", len(doc_ids))
    print("Total chunks            :", len(chunks))

    if args.out:
        out_obj = {
            "unique_topic_groups": sorted(topic_groups),
            "unique_doc_ids": sorted(doc_ids),
            "counts": {
                "topic_group": len(topic_groups),
                "doc_id": len(doc_ids),
                "total_chunks": len(chunks),
            },
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print("\nJSON written to:", args.out)


if __name__ == "__main__":
    main()
