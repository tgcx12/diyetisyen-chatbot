# -*- coding: utf-8 -*-
"""
extract_food_and_avoid_tag_dict.py

- besin_havuzu_eski.normalized.json içindeki
  foods[*].food_tags ve foods[*].avoid_tags alanlarını okur
- her biri için:
    tag -> kaç farklı besinde geçti
  şeklinde sözlük üretir
"""

import json
import os
import argparse
from typing import Dict, Any


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_tags(foods, field: str) -> Dict[str, int]:
    """
    field = 'food_tags' veya 'avoid_tags'
    """
    counts: Dict[str, int] = {}

    for food in foods:
        if not isinstance(food, dict):
            continue

        tags = food.get(field) or []
        if not isinstance(tags, list):
            continue

        # aynı besinde tekrar sayılmasın
        for tag in set(tags):
            if not isinstance(tag, str):
                continue
            counts[tag] = counts.get(tag, 0) + 1

    return dict(sorted(counts.items()))


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="besin_havuzu_eski.normalized.json path")
    ap.add_argument("--out", dest="out", help="output json path (opsiyonel)")
    args = ap.parse_args()

    data = load_json(args.inp)

    if not (isinstance(data, dict) and isinstance(data.get("foods"), list)):
        raise ValueError('Beklenen format: {"foods": [...]}')

    foods = data["foods"]

    food_tag_dict = count_tags(foods, "food_tags")
    avoid_tag_dict = count_tags(foods, "avoid_tags")

    result = {
        "food_tags": food_tag_dict,
        "avoid_tags": avoid_tag_dict,
    }

    if args.out:
        save_json(result, args.out)
        print("OK:", os.path.abspath(args.out))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print("Food tag count:", len(food_tag_dict))
    print("Avoid tag count:", len(avoid_tag_dict))


if __name__ == "__main__":
    main()
