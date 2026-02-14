# -*- coding: utf-8 -*-
"""
normalize_tags_in_besin_havuzu.py

- besin_havuzu_chat.json ({"foods":[...]}) içindeki food_tags ve avoid_tags'i normalize eder
- Eş anlamlı tag'leri tekilleştirir / map'ler
- İsteğe bağlı gürültü tag'lerini kaldırır
- "yag" tag'ini doymus_yag seviyesine göre otomatik ekler:
    - doymus_yag_orta veya doymus_yag_yuksek varsa -> food_tags'a "yag" ekler
    - doymus_yag_dusuk varsa -> eklemez
"""

import os
import json
import argparse
from typing import Any, Dict, List


# 1) BİRLEŞTİRME / MAP KURALLARI
# (soldaki tag görünürse sağdakine çevrilir)
TAG_MAP = {
    # içecek/sıvı
    "icecek": "sivi",

    # süt grubu
    "sut": "sut_urunu",

    # reflü/asit
    "reflu_gastrit": "reflu_tetikleyici",
    "aci": "asitli",

    # besin sunum/uygulama tag'leri (istersen kaldır)
    # "salata": "sebze",  # istersen aç
}

# 2) GÜRÜLTÜ / CLAIM TAG’LERİ (istersen kaldır)
DROP_TAGS = {
    "anti_inflamatuar",
    "metabolizma_destek",
    "karaciger_dostu",
    "sakinlestirici",
    "potasyum_dostu",
    "magnezyum_kaynak",
    "yan_urun",
    "kan_sekeri_dengeleyici",
}

# 3) AVOID için de aynı map/droplar geçerli olsun istiyorsan kullanacağız.
def norm_one_tag(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    t = tag.strip()
    if not t:
        return ""
    t = TAG_MAP.get(t, t)
    return t


def norm_tag_list(tags: Any, drop_enabled: bool = True) -> List[str]:
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        return []

    out: List[str] = []
    seen = set()
    for raw in tags:
        t = norm_one_tag(raw)
        if not t:
            continue
        if drop_enabled and t in DROP_TAGS:
            continue
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def maybe_add_yag(food_tags: List[str]) -> List[str]:
    """
    yag = "yağlı" anlamında basit bir meta tag olsun.
    Kural:
      - doymus_yag_orta veya doymus_yag_yuksek varsa -> yag ekle
      - doymus_yag_dusuk varsa -> yag ekleme
    """
    s = set(food_tags)
    if "doymus_yag_orta" in s or "doymus_yag_yuksek" in s:
        if "yag" not in s:
            food_tags.append("yag")
    return food_tags


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="besin_havuzu_eski.json path")
    ap.add_argument("--out", dest="out", required=True, help="output json path")
    ap.add_argument("--drop-claims", action="store_true", help="claim/gürültü tag'lerini kaldır")
    ap.add_argument("--add-yag", action="store_true", help="doymus_yag seviyesine göre 'yag' tag'i ekle")
    args = ap.parse_args()

    data = load_json(args.inp)

    if not (isinstance(data, dict) and isinstance(data.get("foods"), list)):
        raise ValueError('Beklenen format: {"foods": [...]}')

    foods = data["foods"]

    changed = 0
    for it in foods:
        if not isinstance(it, dict):
            continue

        old_food = list(it.get("food_tags") or [])
        old_avoid = list(it.get("avoid_tags") or [])

        it["food_tags"] = norm_tag_list(old_food, drop_enabled=args.drop_claims)
        it["avoid_tags"] = norm_tag_list(old_avoid, drop_enabled=args.drop_claims)

        if args.add_yag:
            it["food_tags"] = maybe_add_yag(it["food_tags"])

        if old_food != it["food_tags"] or old_avoid != it["avoid_tags"]:
            changed += 1

    save_json(data, args.out)
    print("OK:", os.path.abspath(args.out))
    print("Foods changed:", changed, "/", len(foods))


if __name__ == "__main__":
    main()
