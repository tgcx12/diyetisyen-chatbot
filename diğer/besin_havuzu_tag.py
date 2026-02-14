# -*- coding: utf-8 -*-
"""
besin_havuzu_tag.py

- besin_havuzu_eski.json içinden SADECE "foods" içindeki item'ları alır
- food_tags + avoid_tags etiketlerini toplar
- tekilleştirir, alfabetik sıralar
- her etiket için:
    - food_count: kaç farklı besinde geçti
    - occurrence_count: toplam kaç kez geçti (liste tekrarları dahil)
- compact bir JSON üretir:
    {
      "_meta": {...},
      "tags": {
         "gi_dusuk": " | food_count=37 | occurrence_count=37",
         "baklagil_yemegi": " | food_count=4 | occurrence_count=4"
      }
    }

Kullanım (PowerShell tek satır önerilir):
  python besin_havuzu_tag.py --foods "C:\\...\\besin_havuzu_eski.json" --out "C:\\...\\tag_ontology_compact.json"
"""

import os
import json
import argparse
from typing import Any, Dict, List, Set, Tuple


def safe_load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_dump_json(obj: Any, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_food_items(data: Any) -> List[Dict[str, Any]]:
    """
    Beklenen format:
      { "foods": [ ... ] }

    (Opsiyonel) Eğer elinde düz liste varsa da çalışsın diye destekledim:
      [ ... ]
    """
    if isinstance(data, dict):
        foods = data.get("foods")
        if isinstance(foods, list):
            return foods
        raise ValueError("JSON dict ama 'foods' alanı list değil veya yok. Beklenen: {\"foods\": [...]}")

    if isinstance(data, list):
        return data

    raise ValueError("Foods JSON formatı beklenmeyen. Beklenen: dict{foods:[...]} veya list[...]")

def normalize_tag(t: Any) -> str:
    if not isinstance(t, str):
        return ""
    return t.strip()


def collect_tag_stats(foods: List[Dict[str, Any]]) -> Tuple[Set[str], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      - all_tags: set
      - occurrence_count[tag]: total appearances (repeat included)
      - food_count[tag]: number of distinct foods containing tag
    """
    all_tags: Set[str] = set()
    occurrence_count: Dict[str, int] = {}
    food_count: Dict[str, int] = {}

    for it in foods:
        if not isinstance(it, dict):
            continue

        per_food_seen: Set[str] = set()

        for field in ("food_tags", "avoid_tags"):
            arr = it.get(field, [])
            if arr is None:
                arr = []
            if not isinstance(arr, list):
                # bazen yanlışlıkla string gelebilir; onu da yutalım
                continue

            for raw in arr:
                tag = normalize_tag(raw)
                if not tag:
                    continue
                all_tags.add(tag)
                occurrence_count[tag] = occurrence_count.get(tag, 0) + 1
                per_food_seen.add(tag)

        for tag in per_food_seen:
            food_count[tag] = food_count.get(tag, 0) + 1

    return all_tags, occurrence_count, food_count


def build_compact_tags(
    tags_sorted: List[str],
    occurrence_count: Dict[str, int],
    food_count: Dict[str, int],
) -> Dict[str, str]:
    """
    ÇIKTI: tag -> tek satır string
    Örn:
      "gi_dusuk": " | food_count=37 | occurrence_count=37"
    """
    out: Dict[str, str] = {}
    for t in tags_sorted:
        fc = int(food_count.get(t, 0))
        oc = int(occurrence_count.get(t, 0))
        out[t] = f" | food_count={fc} | occurrence_count={oc}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--foods", required=True, help="besin_havuzu_eski.json dosya yolu")
    ap.add_argument("--out", required=True, help="çıktı JSON dosya yolu")
    args = ap.parse_args()

    data = safe_load_json(args.foods)
    foods = iter_food_items(data)

    all_tags, occ_cnt, food_cnt = collect_tag_stats(foods)
    tags_sorted = sorted(all_tags)

    top_by_food = sorted(tags_sorted, key=lambda t: food_cnt.get(t, 0), reverse=True)[:25]
    top_by_occ = sorted(tags_sorted, key=lambda t: occ_cnt.get(t, 0), reverse=True)[:25]

    compact = {
        "_meta": {
            "source": os.path.abspath(args.foods),
            "food_item_count": len(foods),
            "tag_count": len(tags_sorted),
            "format": "tags[tag] = '<aciklama> | food_count=.. | occurrence_count=..'",
            "notes": [
                "Bu dosya besin_havuzu_eski.json içindeki foods[*].food_tags + foods[*].avoid_tags etiketlerinden üretildi.",
                "Her tag değerinin başına açıklama metnini elle yazabilirsin.",
                "food_count = kaç farklı besinde geçti",
                "occurrence_count = toplam geçiş sayısı (tekrarlar dahil)",
            ],
            "top_tags_by_food_count": [{"tag": t, "food_count": int(food_cnt.get(t, 0))} for t in top_by_food],
            "top_tags_by_occurrence_count": [{"tag": t, "occurrence_count": int(occ_cnt.get(t, 0))} for t in top_by_occ],
        },
        "tags": build_compact_tags(tags_sorted, occ_cnt, food_cnt),
    }

    safe_dump_json(compact, args.out)

    print("OK:", os.path.abspath(args.out))
    print("Food item count:", len(foods))
    print("Tag count:", len(tags_sorted))
    print("Top by food_count:", ", ".join([f"{x['tag']}({x['food_count']})" for x in compact["_meta"]["top_tags_by_food_count"][:10]]))
    print("Top by occurrence_count:", ", ".join([f"{x['tag']}({x['occurrence_count']})" for x in compact["_meta"]["top_tags_by_occurrence_count"][:10]]))


if __name__ == "__main__":
    main()
