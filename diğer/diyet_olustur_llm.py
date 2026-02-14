# -*- coding: utf-8 -*-
"""
TEK DOSYA / TEK KAYNAK - DÜZELTİLMİŞ SÜRÜM
- DayState / PatternState tek kez tanımlı (çakışma yok)
- allowed_food / pick_best / add_item imzaları uyumlu (TypeError yok)
- bypass_pattern parametresi GELSE BİLE KURALLAR ASLA BYPASS EDİLMEZ
  (yani "kurallarım her zaman öncelikli" garantili)
- norm_id bozuk satır düzeltildi
- adjust_meal_calories_by_portion içinde yapıştırılmış ölü/tekrar kod temizlendi
- item_line: porsiyon çarpanı ile "6-8 kaşık" gibi aralıkları çarpar ve ekrana yazar
"""

import json
import random
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, Set, Optional, Callable
import traceback
# ============ REPORTLAB (PDF) ============
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    ListFlowable, ListItem, Table, TableStyle, KeepTogether, HRFlowable
)
import os
from typing import Dict, Any, Set, Tuple, List

def _to_set(x) -> Set[str]:
    if isinstance(x, list):
        return {str(t).strip() for t in x if str(t).strip()}
    return set()

def can_pick(food: Dict[str, Any], avoid_set: Set[str]) -> Tuple[bool, List[str]]:
    """
    Her zaman (ok, reasons) döner.
    - food_tags ∪ avoid_tags ile avoid_set kesişirse seçilmez.
    """
    reasons: List[str] = []

    ft = set(food.get("food_tags") or [])
    fa = set(food.get("avoid_tags") or [])
    tags = ft | fa

    hit = tags & set(avoid_set or set())
    if hit:
        reasons.append(f"avoid_tags çakıştı: {sorted(hit)}")
        return False, reasons

    return True, reasons


# Eski hali:
# DEBUG_ENERGY = os.getenv("DEBUG_ENERGY", "0") == "1"

# Yeni hali (Debug'ı zorla açar):
DEBUG_ENERGY = True

# =========================
# DOSYA YOLLARI (KENDİ PC'NE GÖRE)
# =========================
HAVUZ_YOLU = r"C:\Users\user\PycharmProjects\DiyetisyenLLM\.venv\hastalıkların_önlenmesi_rag\besin_havuzu.json"
DIYABET_YOLU = r"C:\Users\user\PycharmProjects\DiyetisyenLLM\.venv\hastalıkların_önlenmesi_rag\diyabet.json"
KOLESTEROL_YOLU = r"C:\Users\user\PycharmProjects\DiyetisyenLLM\.venv\hastalıkların_önlenmesi_rag\kalp_damar.json"
# KARACIGER_YOLU = r"C:\Users\user\PycharmProjects\DiyetisyenLLM\.venv\hastalıkların_önlenmesi_rag\karaciğer.json"
# MENEPOZ_YOLU = r"C:\Users\user\PycharmProjects\DiyetisyenLLM\.venv\hastalıkların_önlenmesi_rag\menepoz.json"
# MIDE_YOLU = r"C:\Users\user\PycharmProjects\DiyetisyenLLM\.venv\hastalıkların_önlenmesi_rag\mide.json"
TANSIYON_YOLU = r"C:\Users\user\PycharmProjects\DiyetisyenLLM\.venv\hastalıkların_önlenmesi_rag\hipertansiyon.json"
SAGLIKLI_YOLU=r"C:\Users\user\PycharmProjects\DiyetisyenLLM\.venv\hastalıkların_önlenmesi_rag\saglıklı.json"
# LLM çıktısı kural dosyaları (öncelik). İstersen env ile değiştir:
#   DISEASE_RULES_DIR="C:\\...\\Outputs"
_RULES_DIR = os.environ.get("DISEASE_RULES_DIR") or os.path.dirname(__file__)
LLM_RULE_FILES = {
    "diyabet": os.path.join(_RULES_DIR, "disease_rules_tip2.json"),
    "tansiyon": os.path.join(_RULES_DIR, "disease_rules_hipertansiyon.json"),
    "mide": os.path.join(_RULES_DIR, "disease_rules_mide.json"),
    "saglikli": os.path.join(_RULES_DIR, "disease_rules_saglikli.json"),
}

def resolve_rule_path(key: str, fallback_path: str) -> str:
    llm_path = LLM_RULE_FILES.get(key)
    if llm_path and os.path.exists(llm_path):
        return llm_path
    return fallback_path



# =========================
# AYARLAR
# =========================
RANDOM_SEED = None  # 42 -> deterministik

MEAL_BAND_LOW = 0.95
MEAL_BAND_HIGH = 1.05

SOUP_STEPS = [0.5, 0.75, 1.0, 1.5]
SULU_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5]
DEFAULT_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# =========================================================
# RENKLER (PDF)
# =========================================================
PRIMARY_COLOR = colors.HexColor("#1A365D")
SECONDARY_COLOR = colors.HexColor("#3182CE")
ACCENT_BG = colors.HexColor("#F8FAFC")
TEXT_DARK = colors.HexColor("#2D3748")
TEXT_LIGHT = colors.HexColor("#718096")
BORDER_COLOR = colors.HexColor("#E2E8F0")


# =========================
# Font
# =========================
# =========================
# Font
# =========================
def _register_turkish_font() -> str:
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        # Windows
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        # local
        "DejaVuSans.ttf",
        "arial.ttf",
        "calibri.ttf",
        "tahoma.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("TRFont", p))
            return "TRFont"
    return "Helvetica"


FONT_NAME = _register_turkish_font()


def _header_footer(canvas: Canvas, doc: SimpleDocTemplate, title: str):
    canvas.saveState()
    canvas.setStrokeColor(SECONDARY_COLOR)
    canvas.setLineWidth(1.8)
    canvas.line(doc.leftMargin, A4[1] - 1.2 * cm, A4[0] - doc.rightMargin, A4[1] - 1.2 * cm)

    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(TEXT_LIGHT)
    canvas.drawString(doc.leftMargin, A4[1] - 0.9 * cm, title.upper())
    canvas.drawRightString(A4[0] - doc.rightMargin, A4[1] - 0.9 * cm, datetime.now().strftime("%d.%m.%Y"))
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.8 * cm, f"Sayfa {canvas.getPageNumber()}")
    canvas.restoreState()


# =========================
# Utils
# =========================
def safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rules_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kural JSON'larında iki farklı şema desteklenir:

    1) Eski şema:
       meal_pattern_rules -> {rule_name: {type:..., ...}, ...}
       energy_rules -> (band/delta_router vs.)

    2) Yeni (LLM) şema:
       meal_pattern_rules -> {logical_rules: {...}, numeric_constraints: [...]}
       energy_rules -> {scale_up_order:[...], scale_down_order:[...], locks: [...]}

    Bu fonksiyon yeni şemayı, motorun beklediği "pool_pattern_rules" (dict) formatına çevirir.
    """
    if not isinstance(doc, dict):
        return {}

    mpr = doc.get("meal_pattern_rules")
    if isinstance(mpr, dict) and ("numeric_constraints" in mpr or "logical_rules" in mpr):
        logical = mpr.get("logical_rules") if isinstance(mpr.get("logical_rules"), dict) else {}
        numeric = mpr.get("numeric_constraints") if isinstance(mpr.get("numeric_constraints"), list) else []
        pool_rules: Dict[str, Any] = {}

        # logical_rules zaten motor şemasında olmalı (opsiyonel)
        for name, rule in logical.items():
            if isinstance(name, str) and isinstance(rule, dict):
                pool_rules[name] = rule

        # numeric_constraints -> auto rules
        for i, it in enumerate(numeric):
            if not isinstance(it, dict):
                continue
            tag = it.get("tag")
            if not isinstance(tag, str) or not tag.strip():
                continue
            tag = tag.strip()

            # min_days => haftalık minimum (en az X gün/kez)
            if isinstance(it.get("min_days"), int):
                n = int(it["min_days"])
                pool_rules[f"_auto_weekly_min_{i}_{tag}"] = {
                    "type": "weekly_min",
                    "target_food_tag": tag,
                    "min_occurrences": n,
                    "always_apply": True,
                    "source": "auto_numeric_constraints"
                }

            # max_count => haftalık maksimum (en fazla X kez) - yeni şemada period bilgisi yok,
            # burada varsayılanı WEEKLY olarak alıyoruz (daha genel).
            if isinstance(it.get("max_count"), int):
                n = int(it["max_count"])
                pool_rules[f"_auto_weekly_max_{i}_{tag}"] = {
                    "type": "weekly_max",
                    "target_food_tag": tag,
                    "max_occurrences": n,
                    "always_apply": True,
                    "source": "auto_numeric_constraints"
                }

            # min_total_portions / max_total_portions gibi günlük kısıtlar motor şemasında yoksa şimdilik atla.
            # (İsterseniz ileride daily_min rule tipi eklenebilir.)

        doc = dict(doc)
        doc["_raw_meal_pattern_rules"] = mpr
        doc["meal_pattern_rules"] = pool_rules

    return doc

def norm_id(x) -> str:
    return str(x or "").strip().upper()


# =========================
# Normalize / Meal match
# =========================
def normalize_meal_token(s: str) -> str:
    s = (s or "").strip().lower()
    s = (s.replace("ö", "o").replace("ğ", "g").replace("ü", "u")
         .replace("ş", "s").replace("ç", "c").replace("ı", "i"))
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s


def meal_key_for_pool(ogun_adi: str) -> str:
    if "Ara Öğün 1" in ogun_adi:
        return normalize_meal_token("ara_1")
    if "Ara Öğün 2" in ogun_adi:
        return normalize_meal_token("ara_2")
    if "Ara" in ogun_adi:
        return normalize_meal_token("ara")
    if "Yatarken" in ogun_adi or "Gece" in ogun_adi:
        return normalize_meal_token("yatarken")
    return normalize_meal_token(ogun_adi)


def food_in_meal(food: Dict[str, Any], ogun_adi: str) -> bool:
    token = meal_key_for_pool(ogun_adi)
    ogun_field = food.get("ogun")

    if isinstance(ogun_field, list):
        hay = " ".join(normalize_meal_token(str(x)) for x in ogun_field)
    else:
        hay = normalize_meal_token(str(ogun_field))

    return token in hay


def is_snack_meal(ogun_adi: str) -> bool:
    return ("Ara" in ogun_adi) or ("Yatarken" in ogun_adi) or ("Gece" in ogun_adi)


# =========================
# Tag helpers
# =========================
def get_food_tags(food: Dict[str, Any]) -> Set[str]:
    return set(food.get("food_tags", []) or [])


def get_food_avoid_tags(food: Dict[str, Any]) -> Set[str]:
    return set(food.get("avoid_tags", []) or [])


def is_soup(food: Dict[str, Any]) -> bool:
    return "corba" in get_food_tags(food)


def is_legume(food: Dict[str, Any]) -> bool:
    return "baklagil_yemegi" in get_food_tags(food)


def is_bread(food: Dict[str, Any]) -> bool:
    return "ekmek" in get_food_tags(food)


def is_salad(food: Dict[str, Any]) -> bool:
    tags = get_food_tags(food)
    name = (food.get("besin_adi") or "").lower()
    return ("salata" in tags) or ("salata" in name)


def is_fish(food: Dict[str, Any]) -> bool:
    tags = get_food_tags(food)
    cat = (food.get("kategori") or "").lower()
    return ("balik" in tags) or ("balık" in cat) or ("balik" in cat)


def is_egg(food: Dict[str, Any]) -> bool:
    return ("yumurta" in get_food_tags(food)) or ("yumurta" in (food.get("besin_adi", "").lower()))


def is_dairy(food: Dict[str, Any]) -> bool:
    t = get_food_tags(food)
    return bool(t.intersection({"sut_urunu", "sut", "yogurt", "kefir", "ayran", "yagsiz_sut_urunleri"}))


def is_nut(food: Dict[str, Any]) -> bool:
    return "kuruyemis" in get_food_tags(food)


def is_fruit(food: Dict[str, Any]) -> bool:
    return "meyve" in get_food_tags(food)


def is_vegetable_like(food: Dict[str, Any]) -> bool:
    tags = get_food_tags(food)
    cat = (food.get("kategori") or "").lower()
    name = (food.get("besin_adi") or "").lower()
    return ("sebze" in tags) or ("sebze" in cat) or ("salata" in tags) or ("yesillik" in name)


def is_meat(food: Dict[str, Any]) -> bool:
    if is_fish(food):
        return False
    tags = get_food_tags(food)
    cat = (food.get("kategori") or "").lower()
    return ("et grubu" in cat) or bool(tags.intersection({"kirmizi_et", "beyaz_et"}))


def is_condiment_like(food: Dict[str, Any]) -> bool:
    tags = get_food_tags(food)
    name = (food.get("besin_adi") or "").lower()
    if "cesni" in tags:
        return True
    if "sarımsak" in name or "sarmisak" in name:
        return True
    return False


def is_disallowed_oil_item(food: Dict[str, Any]) -> bool:
    return "findik_yagi" in get_food_tags(food)


def is_side_item(food: Dict[str, Any]) -> bool:
    cat = (food.get("kategori") or "").lower()
    tags = get_food_tags(food)
    return ("yan ürün" in cat) or ("yan urun" in cat) or ("yan_urun" in tags)


def is_mevsim_salata(food: Dict[str, Any]) -> bool:
    nm = (food.get("besin_adi") or "").lower()
    return is_salad(food) and ("mevsim" in nm)


# --- Haşlanmış sebze / yoğurt / cacık / meyve salatası ---
def is_boiled_veg(food: Dict[str, Any]) -> bool:
    cat = (food.get("kategori") or "").lower()
    methods = [normalize_meal_token(x) for x in (food.get("cooking_method") or [])]
    return ("sebze" in cat) and ("haslama" in methods)


def is_yogurt_plain(food: Dict[str, Any]) -> bool:
    if not is_dairy(food):
        return False
    name = (food.get("besin_adi") or "").lower()
    if "cacık" in name or "cacik" in name:
        return False
    return "yoğurt" in name or "yogurt" in name


def is_cacik(food: Dict[str, Any]) -> bool:
    name = (food.get("besin_adi") or "").lower()
    return ("cacık" in name) or ("cacik" in name)


def is_fruit_salad_like(food: Dict[str, Any]) -> bool:
    tags = get_food_tags(food)
    name = (food.get("besin_adi") or "").lower()
    if ("meyve" in tags and "salata" in tags):
        return True
    if "meyve" in name and "salata" in name:
        return True
    if "meyve salatas" in name:
        return True
    return False


# =========================
# Meal-level uniqueness helper
# =========================
def food_identity(food: Dict[str, Any]) -> str:
    fid = food.get("besin_id")
    if fid is not None and str(fid).strip():
        return f"id:{str(fid).strip()}"
    nm = (food.get("besin_adi") or "").strip().lower()
    return f"name:{nm}"


def has_same_food_in_meal(selected_in_meal: List[Dict[str, Any]], food: Dict[str, Any]) -> bool:
    key = food_identity(food)
    return any(food_identity(it) == key for it in selected_in_meal)


# =========================
# DayState / PatternState (TEK)
# =========================
from typing import Dict, Any, Set


class DayState:
    """
    - çeşitlilik cezası
    - ara öğünlerde üst üste tekrar engeli
    - öğle/akşam ana tip dengeleme takibi
    """

    def __init__(self, *args, **kwargs):
        # üst sınıf varsa çağır
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            pass

        # ✅ Gün içi tekrarlar
        self.used_names_today: Set[str] = set()

        # ✅ Ara öğün tekrar engeli (aynı gün içinde)
        self.last_snack_names: Set[str] = set()

        # ✅ Hafta genelinde çeşitlilik/tekrar cezası (EN KRİTİK: hata buradan)
        # { "Haşlanmış Yumurta (1 adet)": 2, "Kefir": 4, ... }
        self.week_used_counts: Dict[str, int] = {}

        # ✅ Gün içi öğle/akşam ana tip dengeleme için
        self.main_type_by_meal: Dict[str, str] = {}  # {"Öğle": "meat", "Akşam": "legume", ...}
        self.day_main_counts: Dict[str, int] = {
            "meat": 0,
            "fish": 0,
            "legume": 0,
            "veg": 0,
            "other": 0,
        }

        # (Varsa senin başka alanların burada kalabilir)
        # örn: self.day_index = 0

    def new_day(self):
        """Her yeni gün başında çağır."""
        self.used_names_today = set()
        self.last_snack_names = set()

        # ✅ gün içi ana tip sayaçlarını da sıfırla
        self.main_type_by_meal = {}
        self.day_main_counts = {
            "meat": 0,
            "fish": 0,
            "legume": 0,
            "veg": 0,
            "other": 0,
        }

    def new_week(self):
        """Haftaya başlarken (isteğe bağlı) çağır."""
        self.week_used_counts = {}

    def used_penalty(self, name: str) -> int:
        """İsim bazlı tekrar cezası."""
        return int(self.week_used_counts.get(name, 0))

    def mark_used(self, food: Dict[str, Any], is_snack: bool):
        """Bir besin seçildiğinde çağır."""
        nm = (food.get("besin_adi") or "").strip()
        if nm:
            self.used_names_today.add(nm)
            self.week_used_counts[nm] = self.week_used_counts.get(nm, 0) + 1
            if is_snack:
                self.last_snack_names.add(nm)


class PatternState:
    def __init__(self):
        self.weekly_counts: Dict[str, int] = {}
        self.daily_counts: Dict[str, int] = {}
        self.meal_counts: Dict[str, Dict[str, int]] = {}

    def new_day(self):
        self.daily_counts = {}

    def new_meal(self, meal: str):
        self.meal_counts[meal] = {}

    def inc_weekly(self, tag: str, n: int = 1):
        self.weekly_counts[tag] = self.weekly_counts.get(tag, 0) + n

    def inc_daily(self, tag: str, n: int = 1):
        self.daily_counts[tag] = self.daily_counts.get(tag, 0) + n

    def inc_meal(self, meal: str, tag: str, n: int = 1):
        if meal not in self.meal_counts:
            self.meal_counts[meal] = {}
        self.meal_counts[meal][tag] = self.meal_counts[meal].get(tag, 0) + n

    def get_weekly(self, tag: str) -> int:
        return self.weekly_counts.get(tag, 0)

    def get_daily(self, tag: str) -> int:
        return self.daily_counts.get(tag, 0)

    def get_meal(self, meal: str, tag: str) -> int:
        return self.meal_counts.get(meal, {}).get(tag, 0)


# =========================
# Hastalık rules birleşimi
# =========================
def _collect_tags_from_anywhere(obj: Any) -> Set[str]:
    found: Set[str] = set()

    def walk(x: Any):
        if isinstance(x, dict):
            if "meal_pattern_tags" in x:
                mp = x.get("meal_pattern_tags")
                if isinstance(mp, list):
                    for t in mp:
                        if isinstance(t, str) and t.strip():
                            found.add(t.strip())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return found


def merge_rules(rule_docs: List[Any]) -> Dict[str, Any]:
    avoid: Set[str] = set()
    prefer: Set[str] = set()
    meal_patterns: Set[str] = set()
    notes: List[str] = []

    def collect_from_dict(d: Dict[str, Any]):
        avoid.update(d.get("avoid_tags", []) or [])
        prefer.update(d.get("prefer_tags", []) or [])
        n = d.get("notes")
        if isinstance(n, str) and n.strip():
            notes.append(n.strip())
        meal_patterns.update(_collect_tags_from_anywhere(d))

    for doc in rule_docs:
        if doc is None:
            continue
        if isinstance(doc, list):
            for section in doc:
                if isinstance(section, dict):
                    collect_from_dict(section)
                else:
                    meal_patterns.update(_collect_tags_from_anywhere(section))
        elif isinstance(doc, dict):
            collect_from_dict(doc)
        else:
            meal_patterns.update(_collect_tags_from_anywhere(doc))

    return {
        "avoid_tags": sorted(avoid),
        "prefer_tags": sorted(prefer),
        "meal_pattern_tags": sorted(meal_patterns),
        "notes": notes
    }


def should_exclude(food: Dict[str, Any], active_avoid: Set[str]) -> Tuple[bool, str]:
    ft = get_food_tags(food)
    fa = get_food_avoid_tags(food)
    if ft.intersection(active_avoid):
        return True, "etiket kısıtı"
    if fa.intersection(active_avoid):
        return True, "besine özel kısıt"
    return False, ""


# =========================
# Pattern Engine
# =========================
def rule_applies_to_meal(rule: Dict[str, Any], meal_name: str) -> bool:
    applies = rule.get("applies_to_meals")
    if not applies:
        return True
    m = normalize_meal_token(meal_name)
    applies_norm = [normalize_meal_token(x) for x in applies]
    return m in applies_norm


def update_state_after_pick(food: Dict[str, Any], meal_name: str, pattern_rules: Dict[str, Any], state: PatternState):
    tags = get_food_tags(food)

    for rule in pattern_rules.values():
        if not isinstance(rule, dict):
            continue
        rtype = rule.get("type")

        if rtype in {"weekly_min", "weekly_max", "weekly_range"}:
            tt = rule.get("target_food_tag")
            if tt and tt in tags:
                state.inc_weekly(tt, 1)

        if rtype == "daily_max":
            tt = rule.get("target_food_tag")
            mxu = rule.get("max_portion_units")
            if tt and mxu is not None and tt in tags:
                state.inc_daily(tt, 1)

        if rtype == "meal_max":
            tt = rule.get("target_food_tag")
            if tt and tt in tags:
                state.inc_meal(meal_name, tt, 1)


def violates_pattern(
        food: Dict[str, Any],
        meal_name: str,
        selected_in_meal: List[Dict[str, Any]],
        pattern_rules: Dict[str, Any],
        state: PatternState
) -> Tuple[bool, str]:
    ftags = get_food_tags(food)

    chosen_tags: Set[str] = set()
    for it in selected_in_meal:
        chosen_tags |= get_food_tags(it)

    # mutual exclusion
    for rule in pattern_rules.values():
        if not isinstance(rule, dict):
            continue
        if rule.get("type") != "mutual_exclusion_in_meal":
            continue

        group_a = set(rule.get("group_a_tags", []) or [])
        group_b = set(rule.get("group_b_tags", []) or [])

        if (ftags & group_a) and (chosen_tags & group_b):
            return True, "aynı öğünde birlikte önerilmez"
        if (ftags & group_b) and (chosen_tags & group_a):
            return True, "aynı öğünde birlikte önerilmez"

    # weekly max / range
    for rule in pattern_rules.values():
        if not isinstance(rule, dict):
            continue
        if rule.get("type") not in {"weekly_max", "weekly_range"}:
            continue
        if not rule_applies_to_meal(rule, meal_name):
            continue
        tt = rule.get("target_food_tag")
        mx = rule.get("max_count")
        if tt and mx is not None and tt in ftags:
            if state.get_weekly(tt) >= int(mx):
                return True, "haftalık sınır"

    # daily max
    for rule in pattern_rules.values():
        if not isinstance(rule, dict):
            continue
        if rule.get("type") != "daily_max":
            continue
        tt = rule.get("target_food_tag")
        mxu = rule.get("max_portion_units")
        if tt and mxu is not None and tt in ftags:
            if state.get_daily(tt) >= int(mxu):
                return True, "günlük sınır"

    # meal max
    for rule in pattern_rules.values():
        if not isinstance(rule, dict):
            continue
        if rule.get("type") != "meal_max":
            continue
        if not rule_applies_to_meal(rule, meal_name):
            continue
        tt = rule.get("target_food_tag")
        mx = rule.get("max_count")
        if tt and mx is not None and tt in ftags:
            if state.get_meal(meal_name, tt) >= int(mx):
                return True, "öğün sınırı"

    return False, ""


def get_weekly_targets(pattern_rules: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for rule in pattern_rules.values():
        if not isinstance(rule, dict):
            continue
        t = rule.get("type")
        tag = rule.get("target_food_tag")
        if not tag:
            continue

        if t == "weekly_min":
            out[tag] = {"min": int(rule.get("min_count", 0) or 0), "max": 999}

        if t == "weekly_range":
            out[tag] = {"min": int(rule.get("min_count", 0) or 0), "max": int(rule.get("max_count", 0) or 0)}

        if t == "weekly_max":
            mx = rule.get("max_count")
            if mx is not None and tag not in out:
                out[tag] = {"min": 0, "max": int(mx)}

    return out


def get_soft_weekly_targets(pattern_rules: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rule in pattern_rules.values():
        if not isinstance(rule, dict):
            continue
        if rule.get("type") == "weekly_max" and rule.get("soft_target") is True:
            tt = rule.get("target_food_tag")
            mx = rule.get("max_count")
            if tt and mx is not None:
                out[tt] = int(mx)
    return out


# =========================
# Meal helpers
# =========================
def meal_has_fish(items: List[Dict[str, Any]]) -> bool:
    return any(is_fish(x) for x in items)


def meal_has_boiled_veg(items: List[Dict[str, Any]]) -> bool:
    return any(is_boiled_veg(x) for x in items)


def meal_has_yogurt_plain(items: List[Dict[str, Any]]) -> bool:
    return any(is_yogurt_plain(x) for x in items)


def meal_has_cacik(items: List[Dict[str, Any]]) -> bool:
    return any(is_cacik(x) for x in items)


# =========================
# KALORİ / PORSİYON
# =========================
def base_kcal(food: Dict[str, Any]) -> float:
    return float(food.get("kcal", 0) or 0.0)


def item_kcal(food: Dict[str, Any]) -> float:
    mult = float(food.get("_portion_mult", 1.0) or 1.0)
    return base_kcal(food) * mult


def portion_steps_for(food: Dict[str, Any]) -> List[float]:
    if is_soup(food):
        return SOUP_STEPS
    if is_legume(food) or is_vegetable_like(food):
        return SULU_STEPS
    return DEFAULT_STEPS


def set_portion(food: Dict[str, Any], mult: float):
    food["_portion_mult"] = float(mult)


# =========================
# KURALLAR ÖNCELİK: allowed_food / pick_best / add_item
# =========================
from typing import Dict, Any


def classify_main_type(f: Dict[str, Any]) -> str:
    """
    Öğle/Akşam ana protein tipini sınıflandırır.
    Kombin öğünlerde de çalışması için tag'lerden de bakar.
    """
    tags = set(get_food_tags(f))
    name = (f.get("besin_adi") or "").lower()

    # Balık
    if is_fish(f) or ("balik" in tags) or ("somon" in name) or ("levrek" in name) or ("çipura" in name) or (
            "cipura" in name):
        return "fish"

    # Et
    if is_meat(f) or ("et_grubu" in tags) or ("beyaz_et" in tags) or ("kirmizi_et" in tags) or ("tavuk" in name) or (
            "dana" in name) or ("köfte" in name) or ("kofte" in name):
        return "meat"

    # Baklagil / sulu baklagil
    if is_legume(f) or ("baklagil_yemegi" in tags) or ("baklagil" in tags) or ("nohut" in name) or (
            "kuru fasulye" in name) or ("mercimek" in name) or ("barbunya" in name):
        return "legume"

    # Sebze yemeği
    if is_vegetable_like(f) or ("sebze_yemegi" in tags) or ("zeytinyağlı" in name) or ("zeytinyagli" in name):
        return "veg"

    return "other"


from typing import Dict, Any, Set, List, Tuple

from typing import Tuple, Dict, Any, Set, List


def allowed_food(
        f: Dict[str, Any],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        meal_name: str,
        current_meal_items: List[Dict[str, Any]],
        bypass_pattern: bool = False,  # çağrılar bozulmasın diye var
) -> Tuple[bool, str]:
    """
    KURALLAR HER ZAMAN ÖNCELİKLİ:
    - bypass_pattern parametresi GELSE BİLE burada ASLA bypass edilmez.

    ✅ Fix:
    - besin adı alanı robust (besin_adi/ad/isim/name/yemek)
    - snack tekrar kontrolü boş isimle tetiklenmez
    - Öğle/Akşam kontrolleri normalize edilir
    - Snack'te çift süt ürünü / çift kuruyemiş engeli (JSON no_duplicate_group_in_meal uyumu)
    """

    def _food_name(x: Dict[str, Any]) -> str:
        return (
            (x.get("besin_adi") or x.get("ad") or x.get("isim") or x.get("name") or x.get("yemek") or "")
            .strip()
        )

    try:
        nm = _food_name(f)
    except Exception:
        nm = ""
    nm_lc = nm.lower()

    try:
        tags = set(get_food_tags(f) or set())
    except Exception:
        tags = set()

    # aynı öğünde tekrar olmasın
    if has_same_food_in_meal(current_meal_items, f):
        return False, "aynı öğünde tekrar"

    # öğün dışı kalemler
    if is_disallowed_oil_item(f) or is_condiment_like(f):
        return False, "öğün dışı kalem"

    # Balık + süt ürünü olmaz
    if meal_has_fish(current_meal_items) and is_dairy(f):
        return False, "balık yanında süt ürünü olmaz"

    # Haşlanmış sebze varken salata / meyve salatası eklenmesin
    if meal_has_boiled_veg(current_meal_items) and (is_salad(f) or is_fruit_salad_like(f)):
        return False, "haşlanmış sebze varken salata eklenmez"

    # Aynı öğünde 2 haşlanmış sebze olmasın
    if meal_has_boiled_veg(current_meal_items) and is_boiled_veg(f):
        return False, "aynı öğünde 2 haşlanmış sebze olmaz"

    # Balık öğününde haşlanmış sebze olmasın
    if meal_has_fish(current_meal_items) and is_boiled_veg(f):
        return False, "balık öğününde haşlanmış sebze eklenmez"
    # ✅ Yoğurt + cacık aynı öğünde olmasın (özellikle yancı seçiminde çakışmayı önler)
    if is_cacik(f) and any(is_yogurt_plain(x) for x in current_meal_items):
        return False, "yoğurt varken cacık eklenmez"
    if is_yogurt_plain(f) and any(is_cacik(x) for x in current_meal_items):
        return False, "cacık varken yoğurt eklenmez"

    # Çift çorba / çift meyve (genel)
    if is_soup(f) and any(is_soup(x) for x in current_meal_items):
        return False, "çift çorba yasak"
    if is_fruit(f) and any(is_fruit(x) for x in current_meal_items):
        return False, "çift meyve yasak"

    # ✅ Snack'te (JSON no_duplicate_group_in_meal uyumu) çift süt ürünü / çift kuruyemiş engeli
    if is_snack_meal(meal_name):
        # 1) Toplam kalem sayısı kontrolü (En fazla 2 ürün olabilir)
        if len(current_meal_items) >= 2:
            return False, "ara öğünde en fazla 2 ürün olabilir"

        # 2) Çift süt ürünü engeli
        if is_dairy(f) and any(is_dairy(x) for x in current_meal_items):
            return False, "ara öğünde çift süt ürünü yasak"

        # 3) Çift kuruyemiş engeli
        if ("kuruyemis" in tags) and any(("kuruyemis" in (get_food_tags(x) or set())) for x in current_meal_items):
            return False, "ara öğünde çift kuruyemiş yasak"

    # Hastalık avoid_tags kontrolü
    ex, _ = should_exclude(f, avoid_tags)
    if ex:
        return False, "kısıtlı"

    # Ara öğünlerde üst üste aynı şey gelmesin (✅ boş isimle tetikleme)
    if is_snack_meal(meal_name):
        if not hasattr(day, "last_snack_names") or getattr(day, "last_snack_names") is None:
            day.last_snack_names = set()

        if nm_lc and nm_lc in {str(x).lower() for x in day.last_snack_names}:
            return False, "ara öğün tekrar"

    # -----------------------------
    # ✅ GÜN İÇİ ÖĞLE/AKŞAM DENGE (hard)
    # -----------------------------
    try:
        meal_key = normalize_meal_key(meal_name)
    except Exception:
        meal_key = meal_name.lower()

    is_main_meal = meal_key in {"öğle", "aksam", "akşam"}

    if is_main_meal:
        # DayState'te yoksa oluştur
        if not hasattr(day, "main_type_by_meal"):
            day.main_type_by_meal = {}
        if not hasattr(day, "day_main_counts"):
            day.day_main_counts = {"meat": 0, "fish": 0, "legume": 0, "veg": 0, "other": 0}

        t = classify_main_type(f)

        # sadece "ana yemek" gibi adaylarda çalışsın
        is_mainish_candidate = (
                t in {"meat", "fish", "legume", "veg"}
                and ("corba" not in tags)
                and ("salata" not in tags)
                and ("yan_urun" not in tags)
                and (not is_dairy(f))
        )

        if is_mainish_candidate:
            # meal_name stringine değil normalize'a göre diğer öğünü seç
            other_meal = "Akşam" if meal_key == "öğle" else "Öğle"
            other_type = day.main_type_by_meal.get(other_meal)

            # 1) Öğle ve akşam aynı ana tip olmasın
            if other_type and other_type == t:
                return False, f"öğle/akşam aynı ana tip olmasın ({t})"

            # 2) Gün içinde en fazla 1 et
            if t == "meat" and day.day_main_counts.get("meat", 0) >= 1:
                return False, "günde 1 öğünden fazla et olmasın"

            # 3) Gün içinde en fazla 1 balık
            if t == "fish" and day.day_main_counts.get("fish", 0) >= 1:
                return False, "günde 1 öğünden fazla balık olmasın"

            # 4) Eğer diğer öğünde et/balık varsa, bu öğünde et/balık seçme
            if other_type in {"meat", "fish"} and t in {"meat", "fish"}:
                return False, "et/balık + et/balık aynı güne yığılmasın (diğer öğün baklagil/sebze olsun)"

            # 5) (Opsiyonel - daha sert)
            if meal_key in {"aksam", "akşam"}:
                lunch_type = day.main_type_by_meal.get("Öğle")
                if lunch_type in {"meat", "fish"} and t not in {"legume", "veg"}:
                    return False, "öğle et/balık ise akşam baklagil/sebze olmalı"

    # ✅ pattern motoru HER ZAMAN çalışır
    viol, _ = violates_pattern(f, meal_name, current_meal_items, pattern_rules, state)
    if viol:
        return False, "pattern kuralı"

    return True, ""
def eligible_for_meal(food: Dict[str, Any], meal: str) -> bool:
    """
    food['ogun'] alanına göre öğün uygunluğu.
    meal: "Sabah" / "Öğle" / "Akşam" / "Ara_1" gibi gelebilir.
    """
    ogun_list = food.get("ogun") or []
    if not isinstance(ogun_list, list):
        return False

    meal_norm = str(meal).strip().lower()
    # Türkçe normalize (öğle/akşam vs)
    meal_norm = (meal_norm
                 .replace("ö", "o").replace("ğ", "g").replace("ü", "u")
                 .replace("ş", "s").replace("ı", "i").replace("ç", "c"))

    ogun_norm = []
    for x in ogun_list:
        s = str(x).strip().lower()
        s = (s.replace("ö", "o").replace("ğ", "g").replace("ü", "u")
             .replace("ş", "s").replace("ı", "i").replace("ç", "c"))
        ogun_norm.append(s)

    # "Sabah" -> "sabah", "Ara_1" -> "ara_1"
    return meal_norm in ogun_norm
def score_food(food: Dict[str, Any], prefer_set: Set[str]) -> int:
    """
    Basit skor: food_tags içinde prefer tag varsa +puan.
    """
    ft = set(food.get("food_tags") or [])
    ps = set(prefer_set or set())

    # eşleşen prefer sayısı
    hits = ft & ps
    score = len(hits) * 3  # her eşleşme +3

    # küçük bonuslar (opsiyonel)
    if "lif_yuksek" in ft:
        score += 1
    if "posa_yuksek" in ft:
        score += 1

    return score


def pick_best(
        candidates: List[Dict[str, Any]],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        meal_name: str,
        current_meal_items: List[Dict[str, Any]],
        extra_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        bypass_pattern: bool = False,  # çağrılar bozulmasın diye var
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    # çeşitlilik: haftalık tekrar cezası + random
    random.shuffle(candidates)
    candidates.sort(
        key=lambda x: (day.used_penalty((x.get("besin_adi") or "").strip()) * 25) + random.random()
    )

    for f in candidates:
        if extra_filter and not extra_filter(f):
            continue

        ok, _ = allowed_food(
            f, avoid_tags, day, pattern_rules, state,
            meal_name, current_meal_items,
            bypass_pattern=bypass_pattern  # ilet ama allowed_food yine de bypass etmez
        )
        if ok:
            return f

    return None


from typing import List, Dict, Any, Set, Optional, Tuple


def add_item(
        items: List[Dict[str, Any]],
        f: Dict[str, Any],
        day: "DayState",
        pattern_rules: Dict[str, Any],
        state: "PatternState",
        meal_name: str,
        avoid_tags: Set[str],
        out_reason: Optional[List[str]] = None,  # ✅ reason debug için
) -> bool:
    """
    Öğüne bir besin ekler.

    ✅ Düzeltmeler:
    - allowed_food reason'ını dışarı verebilir (out_reason)
    - meal_name ana öğün kontrolü normalize edilir (Öğle/Akşam kaçmasın)
    - get_food_tags None güvenli
    """
    # 1) allowed_food kontrolü (reason'ı kaybetme)
    ok, reason = allowed_food(f, avoid_tags, day, pattern_rules, state, meal_name, items)
    if not ok:
        if out_reason is not None:
            out_reason.append(str(reason) if reason is not None else "allowed_food rejected (no reason)")
        return False

    # 2) Kopya ekle (yan etkileri önlemek için)
    new_item = dict(f)

    # 3) Varsayılan portion çarpanı
    if "_portion_mult" not in new_item:
        new_item["_portion_mult"] = 1.0

    items.append(new_item)

    # 4) Gün içi tekrar/çeşitlilik işaretlemesi
    try:
        day.mark_used(new_item, is_snack_meal(meal_name))
    except Exception:
        pass

    try:
        state.observe_food(meal_name, new_item)
    except Exception:
        pass

    # 5) ✅ Gün içi öğle/akşam ana tip takibi (normalize)
    try:
        meal_key = normalize_meal_key(meal_name)
    except Exception:
        meal_key = meal_name

    is_main_meal = meal_key in {"öğle", "aksam", "akşam"}  # aksan/normalize farkları için

    if is_main_meal:
        try:
            tags = set(get_food_tags(new_item) or set())
        except Exception:
            tags = set()

        t = classify_main_type(new_item)

        # ana yemek sayılacak mı?
        is_mainish = (
                t in {"meat", "fish", "legume", "veg"}
                and ("corba" not in tags)
                and ("salata" not in tags)
                and ("yan_urun" not in tags)
                and (not is_dairy(new_item))
        )

        # Sadece öğünde ilk ana tip yazılsın
        if is_mainish and meal_name not in getattr(day, "main_type_by_meal", {}):
            if not hasattr(day, "main_type_by_meal"):
                day.main_type_by_meal = {}
            if not hasattr(day, "day_main_counts"):
                day.day_main_counts = {"meat": 0, "fish": 0, "legume": 0, "veg": 0, "other": 0}

            day.main_type_by_meal[meal_name] = t
            day.day_main_counts[t] = day.day_main_counts.get(t, 0) + 1

    return True


# =========================
# Yan ürün seçimi
# =========================
def side_candidates_for_meal(meal_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for f in meal_pool:
        if is_side_item(f):
            out.append(f)
            continue
        if is_dairy(f):
            out.append(f)
            continue
    return out


def pick_side_for_meal(
        meal_pool: List[Dict[str, Any]],
        all_foods: List[Dict[str, Any]],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        meal_name: str,
        current_items: List[Dict[str, Any]],
        force_salad: bool
) -> Optional[Dict[str, Any]]:
    """
    ✅ Yan ürün seçiminde çakışma çözümü (salata / yoğurt / cacık):
    - Balıkta: sadece salata (allowed_food zaten süt ürünü engeller ama burada da güvenlik).
    - Haşlanmış sebze varsa: burada side seçilmez; enforce_boiled_veg_bundle yönetir.
    - Varsayılan: Tek bir "yan" seç (salata OR cacık). Yoğurt yalnızca sebze/baklagil-benzeri öğünlerde tercih edilir.
    - Eğer özel bir kural bu öğüne zaten yan eklediyse (salata/yoğurt/cacık), ikinci bir yan ekleme.
    - Eğer öğünde yoğurt + salata birlikte varsa: cacık ASLA eklenmez (kullanıcı isteği).
    """
    sides = side_candidates_for_meal(meal_pool)

    # Haşlama paketi ayrı yönetiliyor
    if meal_has_boiled_veg(current_items):
        return None

    # Zaten yan varsa ikinciyi ekleme (özel kural vs.)
    if any(is_salad(x) or is_yogurt_plain(x) or is_cacik(x) for x in current_items):
        return None

    # Eğer özel kural yoğurt+salata koyduysa cacık eklemeyelim (sadece güvenlik)
    if any(is_yogurt_plain(x) for x in current_items) and any(is_salad(x) for x in current_items):
        return None

    # Balık -> salata zorunlu
    if force_salad or meal_has_fish(current_items):
        cand = pick_best(
            sides, avoid_tags, day, pattern_rules, state, meal_name, current_items,
            extra_filter=is_mevsim_salata
        )
        if cand:
            return cand
        cand = pick_best(sides, avoid_tags, day, pattern_rules, state, meal_name, current_items, extra_filter=is_salad)
        if cand:
            return cand
        return pick_best(
            all_foods, avoid_tags, day, pattern_rules, state, meal_name, current_items,
            extra_filter=lambda x: (is_side_item(x) or is_dairy(x)) and is_salad(x)
        )

    # Sebze/baklagil ağırlıklı öğünlerde yoğurt (bulunamazsa cacık) tercih et
    veg_like = meal_has_any_tag(current_items, ["sebze_yemegi", "baklagil", "baklagil_yemegi", "haslama"])
    if veg_like:
        cand = pick_best(sides, avoid_tags, day, pattern_rules, state, meal_name, current_items,
                         extra_filter=is_yogurt_plain)
        if cand:
            return cand
        cand = pick_best(sides, avoid_tags, day, pattern_rules, state, meal_name, current_items, extra_filter=is_cacik)
        if cand:
            return cand
        # son çare: herhangi bir süt ürünü (balık yok zaten)
        cand = pick_best(sides, avoid_tags, day, pattern_rules, state, meal_name, current_items, extra_filter=is_dairy)
        if cand:
            return cand

    # Diğer öğünler: salata / cacık (tek yan)
    prefer_salad = (random.random() < 0.60)

    def _pick_salad() -> Optional[Dict[str, Any]]:
        c = pick_best(sides, avoid_tags, day, pattern_rules, state, meal_name, current_items, extra_filter=is_salad)
        if c:
            return c
        return pick_best(
            all_foods, avoid_tags, day, pattern_rules, state, meal_name, current_items,
            extra_filter=lambda x: (is_side_item(x) or is_dairy(x)) and is_salad(x)
        )

    def _pick_cacik() -> Optional[Dict[str, Any]]:
        # yoğurt yokken cacık seçilebilir (yoğurt varsa zaten erken return)
        c = pick_best(sides, avoid_tags, day, pattern_rules, state, meal_name, current_items, extra_filter=is_cacik)
        if c:
            return c
        # son çare: süt ürünü (yoğurt/cacık çıkabilir)
        return pick_best(sides, avoid_tags, day, pattern_rules, state, meal_name, current_items, extra_filter=is_dairy)

    if prefer_salad:
        cand = _pick_salad()
        if cand:
            return cand
        return _pick_cacik()
    else:
        cand = _pick_cacik()
        if cand:
            return cand
        return _pick_salad()


def enforce_boiled_veg_bundle(
        meal_name: str,
        items: List[Dict[str, Any]],
        meal_pool: List[Dict[str, Any]],
        all_foods: List[Dict[str, Any]],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        notes: List[str]
):
    """
    Haşlanmış sebze özel kuralı:
    - Salata / meyve salatası bu öğünde bulunamaz (kaldırılır).
    - Yan olarak YOĞURT tercih edilir; yoksa CACIK.
    - Aynı anda yoğurt + cacık eklenmez.
    """
    if not meal_has_boiled_veg(items):
        return

    # 1) Salata/meyve salatası varsa kaldır
    removed_any = False
    kept: List[Dict[str, Any]] = []
    for it in items:
        if is_salad(it) or is_fruit_salad_like(it):
            removed_any = True
            continue
        kept.append(it)
    if removed_any:
        items[:] = kept
        notes.append("Haşlanmış sebze olduğu için salata/meyve salatası kaldırıldı.")

    # 2) Zaten yoğurt/cacık varsa tekrar ekleme
    if any(is_yogurt_plain(x) or is_cacik(x) for x in items):
        return

    # 3) Yoğurt kesin olsun (yoksa ekle). Bulunamazsa cacık.
    # Önce pool (öğün uyumlu), sonra tüm yiyecekler fallback
    yogurt = pick_best(
        side_candidates_for_meal(meal_pool),
        avoid_tags, day, pattern_rules, state, meal_name, items,
        extra_filter=is_yogurt_plain
    )
    if yogurt is None:
        yogurt = pick_best(
            all_foods,
            avoid_tags, day, pattern_rules, state, meal_name, items,
            extra_filter=is_yogurt_plain
        )
    if yogurt:
        ok = add_item(items, yogurt, day, pattern_rules, state, meal_name, avoid_tags)
        if ok:
            notes.append("Haşlanmış sebze yanında yoğurt eklendi.")
        return

    cacik = pick_best(
        side_candidates_for_meal(meal_pool),
        avoid_tags, day, pattern_rules, state, meal_name, items,
        extra_filter=is_cacik
    )
    if cacik is None:
        cacik = pick_best(
            all_foods,
            avoid_tags, day, pattern_rules, state, meal_name, items,
            extra_filter=is_cacik
        )
    if cacik:
        ok = add_item(items, cacik, day, pattern_rules, state, meal_name, avoid_tags)
        if ok:
            notes.append("Haşlanmış sebze yanında cacık eklendi.")


def select_active_pattern_rules(pool_pattern_rules: Dict[str, Any], active_pattern_tags: List[str]) -> Dict[str, Any]:
    """
    Aktif pattern kural setini seçer.

    Eski şema: meal_pattern_tags listesinde adı geçen kuralları aktif eder.
    Yeni (LLM) şema: meal_pattern_tags boş olabilir; numeric_constraints'ten türetilen "_auto_*" kuralları
    her zaman uygulanmalıdır.

    Kural objesi içinde `always_apply: True` varsa, active_pattern_tags'e bakmadan dahil eder.
    """
    if not isinstance(pool_pattern_rules, dict) or not pool_pattern_rules:
        return {}

    active_set = set([t.strip() for t in active_pattern_tags if isinstance(t, str) and t.strip()])

    out: Dict[str, Any] = {}
    for k, rule in pool_pattern_rules.items():
        if not isinstance(rule, dict):
            continue
        if rule.get("always_apply") is True:
            out[k] = rule
            continue
        # meal_pattern_tags boşsa: eğer kural adı tag gibi yönetilmiyorsa ve auto değilse,
        # eski davranışta hiçbir şey aktif olmazdı. Burada daha güvenli davranıyoruz:
        # active_pattern_tags boşsa tüm kuralları aktif ETMEYİZ; sadece always_apply kalsın.
        if active_set and k in active_set:
            out[k] = rule

    return out
def sor_hastaliklar() -> List[str]:
    hastalik_menu = {
        "1": "diyabet",
        "2": "kolesterol",
        "3": "karaciger",
        "4": "menepoz",
        "5": "mide",
        "6": "tansiyon",
        "0": "yok"
    }

    print("\nVarsa sağlık durumunuzu seçiniz (birden fazla olabilir):")
    print("1 - Diyabet")
    print("2 - Kolesterol / Kalp-Damar")
    print("3 - Karaciğer")
    print("4 - Menepoz")
    print("5 - Mide")
    print("6 - Tansiyon")
    print("0 - Yok")

    raw = input("Seçiminiz (örn: 1,5): ").strip()
    if not raw:
        return []

    secimler = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]

    bulunanlar = set()
    for s in secimler:
        if s in hastalik_menu:
            hastalik = hastalik_menu[s]
            if hastalik != "yok":
                bulunanlar.add(hastalik)

    return sorted(bulunanlar)


def load_data() -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    havuz_data = safe_load_json(HAVUZ_YOLU)
    foods = havuz_data.get("foods", []) or []

    meta = havuz_data.get("meta", {}) or {}
    tag_desc = (meta.get("tag_descriptions", {}) or {})

    # Havuz açıklamaları (genel etiket sözlüğü)
    pool_pattern_desc = (tag_desc.get("meal_pattern_tags", {}) or {})
    avoid_desc_map = (tag_desc.get("avoid_tags", {}) or {})
    meal_level_hints = meta.get("meal_level_rules_hints", []) or []
    pool_pattern_rules = meta.get("meal_pattern_rules", {}) or {}

    hastaliklar = sor_hastaliklar()
    rule_docs: List[Dict[str, Any]] = []
    used_rule_files: List[str] = []

    file_map = {
        "diyabet": (DIYABET_YOLU, "diyabet.json"),
        "kolesterol": (KOLESTEROL_YOLU, "kalp_damar.json"),
        # "karaciger": (KARACIGER_YOLU, "karaciğer.json"),
        # "menepoz": (MENEPOZ_YOLU, "menepoz.json"),
        # "mide": (MIDE_YOLU, "mide.json"),
        "tansiyon": (TANSIYON_YOLU, "hipertansiyon.json"),
        "sağlıklı": (SAGLIKLI_YOLU, "saglıklı.json")
    }

    for key, (path, filename) in file_map.items():
        if key in hastaliklar and os.path.exists(path):
            doc = normalize_rules_doc(safe_load_json(path))
            rule_docs.append(doc)
            used_rule_files.append(filename)

    # Hastalık JSON'larını birleştir
    rules = merge_rules(rule_docs)

    # 1) Aktif meal_pattern_tags
    active_tags = rules.get("meal_pattern_tags", []) or []

    # Fallback: bazı hastalık JSON'larında meal_pattern_tags alanı olmayabilir.
    # Bu durumda meal_pattern_rules ve/veya meal_pattern_tag_descriptions üzerinden aktif tag'leri türet.
    if not active_tags:
        derived: set[str] = set()
        for d in rule_docs:
            if not isinstance(d, dict):
                continue
            mpr = d.get("meal_pattern_rules")
            if isinstance(mpr, dict):
                derived.update([k for k in mpr.keys() if isinstance(k, str) and k.strip()])
            mpd = d.get("meal_pattern_tag_descriptions")
            if isinstance(mpd, dict):
                rto = mpd.get("rule_tags_only")
                if isinstance(rto, dict):
                    derived.update([k for k in rto.keys() if isinstance(k, str) and k.strip()])
        active_tags = sorted(derived)

    # 2) SADECE aktif tag'lerde tanımlı kuralları uygulamaya sok
    # Havuz + hastalık meal_pattern_rules birleşimi (hastalık override/extend)
    merged_pattern_rules: Dict[str, Any] = dict(pool_pattern_rules or {})
    for d in rule_docs:
        if isinstance(d, dict):
            mpr = d.get("meal_pattern_rules")
            if isinstance(mpr, dict):
                merged_pattern_rules.update(mpr)

    active_pattern_rules = select_active_pattern_rules(
        pool_pattern_rules=merged_pattern_rules,
        active_pattern_tags=active_tags
    )

    # 3) Hastalık JSON'undan (varsa) tag/kuralların kullanıcıya açıklaması
    disease_pattern_desc: Dict[str, Any] = {}
    disease_info_desc: Dict[str, Any] = {}
    disease_food_tag_desc: Dict[str, Any] = {}

    for d in rule_docs:
        mpd = (d.get("meal_pattern_tag_descriptions") or {})
        if isinstance(mpd, dict):
            rto = mpd.get("rule_tags_only") or {}
            ito = mpd.get("info_only_tags") or {}
            if isinstance(rto, dict):
                disease_pattern_desc.update(rto)
            if isinstance(ito, dict):
                disease_info_desc.update(ito)

        dtd = (d.get("disease_tag_descriptions") or {})
        if isinstance(dtd, dict):
            disease_food_tag_desc.update(dtd)

    # 4) Enerji scaling: ana motorda mantık yok.
    energy_rules_active: Dict[str, Any] = {}
    if len(rule_docs) == 1 and isinstance(rule_docs[0], dict):
        # Diyabet JSON'unda anahtar "energy_rules" olarak gelebilir (eski: "energy_scaling_rules")
        er = (rule_docs[0].get("energy_scaling_rules") or rule_docs[0].get("energy_rules"))
        if isinstance(er, dict):
            energy_rules_active = er

    pool_meta = {
        "avoid_desc_map": avoid_desc_map,
        "meal_pattern_desc_pool": pool_pattern_desc,
        "meal_pattern_desc_disease_rule": disease_pattern_desc,
        "meal_pattern_desc_disease_info": disease_info_desc,
        "disease_food_tag_desc": disease_food_tag_desc,
        "meal_pattern_rules_active": active_pattern_rules,
        "meal_pattern_tags_active": active_tags,
        "meal_level_hints": meal_level_hints,
        "energy_scaling_rules_active": energy_rules_active,
    }

    return foods, rules, used_rule_files, hastaliklar, pool_meta


def pick_activity_factor() -> float:
    allowed = [1.2, 1.375, 1.55, 1.725]
    print("\nAktivite Düzeyi:\n1.2 (Hareketsiz)\n1.375 (Hafif)\n1.55 (Orta)\n1.725 (Çok Aktif)")
    raw = input("Seçiminiz: ").strip().replace(",", ".")
    try:
        val = float(raw)
    except:
        val = 1.2
    return min(allowed, key=lambda x: abs(x - val))


def hesapla_gereksinim() -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("      AKILLI DİYETİSYEN: KALORİ VE BKİ ANALİZİ")
    print("=" * 60)

    ad_soyad = input("Ad Soyad: ").strip()
    cinsiyet = input("Cinsiyet (E/K): ").strip().upper()
    kilo = float(input("Kilo (kg): ").strip().replace(",", "."))
    boy = float(input("Boy (cm): ").strip().replace(",", "."))
    yas = int(input("Yaş: ").strip())
    aktivite = pick_activity_factor()

    bmr = (10 * kilo) + (6.25 * boy) - (5 * yas) + (5 if cinsiyet == "E" else -161)
    tdee = bmr * aktivite
    bki = kilo / ((boy / 100) ** 2)

    hedef_kalori = tdee
    acik_kural = "kalori dengesi"
    if bki >= 25:
        hedef_kalori = tdee * 0.85
        acik_kural = "%15 kalori açığı"

    return {
        "ad_soyad": ad_soyad,
        "cinsiyet": cinsiyet,
        "kilo": kilo,
        "boy": boy,
        "yas": yas,
        "aktivite": aktivite,
        "bmr": bmr,
        "tdee": tdee,
        "bki": bki,
        "hedef_kalori": round(hedef_kalori),
        "acik_kural": acik_kural
    }


def pick_band_id(abs_delta_kcal: float, energy_rules: Dict[str, Any]) -> str:
    router = (energy_rules.get("delta_router") or {})
    bands = router.get("bands") or []
    tol = float(router.get("tolerance_kcal", 0) or 0)

    x = max(0.0, float(abs_delta_kcal) - tol)

    # default band
    chosen = "B1"
    for b in bands:
        mn_inc = b.get("min_inclusive", None)
        mn_exc = b.get("min_exclusive", None)
        mx_inc = b.get("max_inclusive", None)
        mx_exc = b.get("max_exclusive", None)

        ok = True
        if mn_inc is not None and x < float(mn_inc):
            ok = False
        if mn_exc is not None and x <= float(mn_exc):
            ok = False
        if mx_inc is not None and x > float(mx_inc):
            ok = False
        if mx_exc is not None and x >= float(mx_exc):
            ok = False

        if ok:
            chosen = b.get("band_id") or chosen
            break

    return chosen


def normalize_meal_key(meal_name: str) -> str:
    m = (meal_name or "").strip().lower()

    # yaygın TR isimler
    if m == "sabah":
        return "sabah"
    if "ara" in m and "1" in m:
        return "ara_1"
    if "ara" in m and "2" in m:
        return "ara_2"
    if m == "öğle" or m == "ogle":
        return "öğle"
    if m == "akşam" or m == "aksam":
        return "akşam"
    if m == "yatarken":
        return "yatarken"

    # fallback: boşlukları "_" yap
    return m.replace(" ", "_")


def is_meal_locked_by_energy_rules(meal_name: str, items: List[Dict[str, Any]], energy_rules: Dict[str, Any]) -> bool:
    meal_key = normalize_meal_key(meal_name)
    locks = energy_rules.get("locks") or []

    def meal_has_any_tag(tag_list: List[str]) -> bool:
        for it in items:
            tags = set(get_food_tags(it))
            if any(t in tags for t in (tag_list or [])):
                return True
        return False

    for lk in locks:
        if not isinstance(lk, dict):
            continue
        if meal_key not in set(lk.get("applies_to_meals") or []):
            continue

        typ = lk.get("type")

        if typ == "lock_meal":
            return True

        if typ == "lock_meal_if_contains":
            cond = (lk.get("if") or {}).get("any_food_has", {}) or {}
            tags_any = cond.get("food_tags_any") or []
            if meal_has_any_tag(tags_any):
                return True

    return False


def energy_rules_to_priority_order(energy_rules: Dict[str, Any], direction: str, band_id: str) -> List[Dict[str, Any]]:
    """
    Convert disease-specific energy_rules (DM) into a linear list of executable step dicts.

    Important:
    - meal_pattern_rules are enforced elsewhere and remain higher priority.
    - This function must not "understand" nutrition; it only normalizes the JSON contract
      into the engine's internal step format.
    """
    if not energy_rules:
        return []

        # direction hem "up"/"down" hem de eski çağrılar için "scale_up"/"scale_down" gelebilir
    if direction in ("up", "scale_up"):
        section_key = "scale_up"
    else:
        section_key = "scale_down"
    section = energy_rules.get(section_key) or {}
    if not section:
        return []

    # Pick band strategy
    band_strategy = section.get("band_strategy") or []
    strat = next((b for b in band_strategy if b.get("band_id") == band_id), None)
    if not strat:
        return []

    # Index templates by id
    templates = section.get("step_templates") or []
    step_by_id = {t.get("step_id"): t for t in templates if t.get("step_id")}

    # Normalize action names (support both legacy and new DM contract)
    action_map = {
        # legacy -> internal
        "increase_spoons": "increase_portion",
        "decrease_spoons": "decrease_portion",
        "increase_pieces": "increase_portion",
        "decrease_pieces": "decrease_portion",
        "increase_grams": "increase_portion",
        "decrease_grams": "decrease_portion",
        "add_food": "add",
        "remove_or_reduce": "remove_or_reduce",
        "replace_with": "replace_with",
        # new DM contract -> internal (identity)
        "increase_portion": "increase_portion",
        "decrease_portion": "decrease_portion",
        "add": "add",
        "add_or_increase": "add_or_increase",

    }

    priority_order: List[Dict[str, Any]] = []
    for step_id in (strat.get("steps_in_order") or []):
        tpl = step_by_id.get(step_id) or {}
        if not tpl:
            continue

        action_raw = tpl.get("action")
        action = action_map.get(action_raw)
        if not action:
            continue

        step: Dict[str, Any] = {"step_id": step_id, "action": action}

        # Pass-through common selectors / constraints
        for k in ("applies_to_meals", "limits", "preconditions", "candidate_filter"):
            if k in tpl:
                step[k] = tpl[k]

        # For the newer contract: portion rules are referenced by name in energy_rules
        if tpl.get("uses_rule_ref"):
            step["uses_rule_ref"] = tpl["uses_rule_ref"]

        # Backward compatible fields (some older templates may still provide these)
        if tpl.get("target_food_tags_any"):
            step["target_food_tags_any"] = tpl["target_food_tags_any"]
        if tpl.get("step"):
            step["step"] = tpl["step"]
        if tpl.get("max_step"):
            step["max_step"] = tpl["max_step"]

        # If an "add" step didn't specify target tags explicitly, derive from candidate_filter
        if action == "add" and not step.get("target_food_tags_any"):
            cf = step.get("candidate_filter") or {}
            tags = cf.get("must_have_food_tags_all") or cf.get("must_have_food_tags_any") or []
            if tags:
                step["target_food_tags_any"] = tags

        priority_order.append(step)

    return priority_order


def pick_band_id_and_max_steps(abs_delta_kcal: float, energy_rules: Dict[str, Any]) -> Tuple[str, int]:
    """
    delta_router: abs_delta_band
    tolerance_kcal düşülür, sonra band bulunur.
    """
    router = (energy_rules.get("delta_router") or {})
    bands = list(router.get("bands") or [])
    tol = float(router.get("tolerance_kcal", 0) or 0)

    x = max(0.0, float(abs_delta_kcal) - tol)

    # bands'i yaklaşık min değerine göre sırala (sağlamlık)
    def band_sort_key(b: Dict[str, Any]) -> float:
        if not isinstance(b, dict):
            return 1e18
        if b.get("min_inclusive") is not None:
            return float(b["min_inclusive"])
        if b.get("min_exclusive") is not None:
            # exclusive ise +epsilon gibi düşün
            return float(b["min_exclusive"]) + 1e-6
        return 0.0

    bands = [b for b in bands if isinstance(b, dict)]
    bands.sort(key=band_sort_key)

    chosen_id = "B1"
    chosen_max_steps = 1

    for b in bands:
        mn_inc = b.get("min_inclusive", None)
        mn_exc = b.get("min_exclusive", None)
        mx_inc = b.get("max_inclusive", None)
        mx_exc = b.get("max_exclusive", None)

        ok = True
        if mn_inc is not None and x < float(mn_inc):
            ok = False
        if mn_exc is not None and x <= float(mn_exc):
            ok = False
        if mx_inc is not None and x > float(mx_inc):
            ok = False
        if mx_exc is not None and x >= float(mx_exc):
            ok = False

        if ok:
            chosen_id = (b.get("band_id") or chosen_id)
            chosen_max_steps = int(b.get("max_steps_total", chosen_max_steps) or chosen_max_steps)
            break

    return chosen_id, chosen_max_steps


def compute_meal_energy_kcal(items: List[Dict[str, Any]]) -> int:
    """Sum current (possibly scaled) kcal for a meal."""
    total = 0
    for it in items:
        if "kcal_scaled" in it and isinstance(it.get("kcal_scaled"), (int, float)):
            total += int(round(float(it["kcal_scaled"])))
        else:
            base = float(it.get("kcal", 0) or 0)
            mult = float(it.get("_portion_mult", 1.0) or 1.0)
            total += int(round(base * mult))
    return int(total)


def _regex_any_match(text: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    for p in patterns:
        try:
            if re.search(p, text, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _has_tags(item: Dict[str, Any], must_have: List[str] = None, must_not: List[str] = None) -> bool:
    """Tag matching helper with a small synonym bridge (most importantly: et_grubu)."""
    must_have = must_have or []
    must_not = must_not or []

    try:
        tags = set(get_food_tags(item) or [])
    except Exception:
        tags = set(item.get("food_tags") or item.get("tags") or [])

    # --- Synonym / fallback bridges ---
    # Some pools tag proteins as "beyaz_et"/"kirmizi_et" but rules may require "et_grubu".
    if "et_grubu" in must_have and "et_grubu" not in tags:
        cat = (item.get("kategori") or "").lower()
        name = (item.get("besin_adi") or item.get("ad") or "").lower()
        if tags.intersection({"beyaz_et", "kirmizi_et"}) or ("et" in cat) or ("tavuk" in name) or ("dana" in name) or (
                "kofte" in name) or ("köfte" in name):
            # treat as if et_grubu present
            tags.add("et_grubu")

    if must_have and not any(t in tags for t in must_have):
        return False
    if must_not and any(t in tags for t in must_not):
        return False
    return True

import re
from typing import List, Dict, Any, Optional, Tuple, Callable


_HYPHENS = r"[-–—]"  # normal '-' + en dash + em dash

import re
from typing import List, Dict, Any, Tuple, Optional, Callable


def apply_portion_rule_ref(
    *,
    items: List[Dict[str, Any]],
    energy_rules: Dict[str, Any],
    rule_ref: str,
    direction: str,
    dlog: Callable,
) -> bool:
    """
    GENEL / TEK FONKSİYON (tüm hastalıklarda ortak):
    - '-' yerine '–' '—' destekler.
    - portion_step_by_grams: multiplier + ladder
    - portion_step_by_spoons: tek sayı + aralık
    - Candidate seçimi: gerçekten değişebilecek olanı seçer.

    ✅ KRİTİK DÜZELTME:
    - DOWN micro-backstep VAR ama:
        (1) SADECE daha önce artırılmış (_portion_touched==True) ve mult>1 olanlarda çalışır
        (2) SADECE bu rule_ref'in eligible filtresinden geçenlerde çalışır
        (3) ÇORBA (tag corba/çorba) ve/veya miktar_str içinde kepçe geçenler micro-backstep'e ASLA girmez
        (4) micro-backstep adımı sabit 0.25 (istersen config’e alırsın)

    Böylece:
    - Çorba asla 2.00->1.75->1.50 düşmez
    - Çorba sadece soup_ladle_portion_rule ile 2->1 kepçe düşer
    """

    if direction not in ("up", "down"):
        dlog(f"[ENERGY] portion_rule_ref invalid direction={direction!r}")
        return False

    rule = energy_rules.get(rule_ref) or {}
    if not rule:
        dlog(f"[ENERGY] portion_rule_ref missing: {rule_ref}")
        return False

    ef = (rule.get("eligible_food_filter") or {})
    must_have = ef.get("must_have_food_tags_any") or ef.get("must_have_food_tags_all") or []
    must_not = ef.get("must_not_have_food_tags_any") or []
    patterns = ef.get("must_match_miktar_regex_any") or []

    rtype = (rule.get("type") or "").strip()

    # dash variations
    _HYPHENS = r"(?:-|–|—)"

    def item_name(it: Dict[str, Any]) -> str:
        return str(it.get("besin_adi") or it.get("ad") or it.get("isim") or it.get("name") or it.get("yemek") or "?")

    def current_amount(it: Dict[str, Any]) -> str:
        return (it.get("miktar_str") or it.get("miktar") or "").strip()

    def base_mult_of(it: Dict[str, Any]) -> float:
        try:
            return float(it.get("_portion_mult", 1.0) or 1.0)
        except Exception:
            return 1.0

    def base_kcal_of(it: Dict[str, Any]) -> float:
        try:
            return float(it.get("kcal", 0) or 0.0)
        except Exception:
            return 0.0

    def has_soup_semantics(it: Dict[str, Any]) -> bool:
        """
        Çorba micro-backstep'e girmesin:
        - tag: corba/çorba
        - miktar: kepçe/kepce
        """
        amt = (current_amount(it) or "").lower()
        if "kepçe" in amt or "kepce" in amt:
            return True
        try:
            tags = set(it.get("food_tags") or it.get("tags") or [])
        except Exception:
            tags = set()
        # projende get_food_tags varsa ve istiyorsan burayı ona bağlayabilirsin
        # ama generic kalsın diye alanlardan okuyoruz; _has_tags zaten var.
        if "corba" in tags or "çorba" in tags:
            return True
        # Tag'leri direct okumadık diyelim: _has_tags üzerinden de kontrol edelim
        try:
            if _has_tags(it, must_have=["corba"], must_not=[]):
                return True
        except Exception:
            pass
        try:
            if _has_tags(it, must_have=["çorba"], must_not=[]):
                return True
        except Exception:
            pass
        return False

    def set_scaled(candidate: Dict[str, Any], new_amt: str, ratio: float) -> bool:
        """
        ratio: "bu item'in mevcut miktarına göre" oran.
        _portion_mult = base_mult * ratio
        kcal_scaled = kcal * _portion_mult
        """
        if ratio <= 0:
            return False

        # orig sakla
        if "_miktar_str_orig" not in candidate:
            candidate["_miktar_str_orig"] = current_amount(candidate)
        if "_portion_mult_orig" not in candidate:
            candidate["_portion_mult_orig"] = float(candidate.get("_portion_mult", 1.0) or 1.0)

        base_mult = base_mult_of(candidate)
        new_mult = round(base_mult * float(ratio), 4)

        # değişiklik yoksa
        if abs(new_mult - base_mult) < 1e-9 and (new_amt or "").strip() == current_amount(candidate):
            return False

        candidate["_portion_mult"] = new_mult
        candidate["kcal_scaled"] = int(round(base_kcal_of(candidate) * new_mult))
        candidate["miktar_str"] = new_amt

        # işaretle
        candidate["_changeable"] = True
        candidate["_portion_rule_ref_last"] = rule_ref
        candidate["_portion_direction_last"] = direction
        candidate["_portion_touched"] = True

        return True

    # -----------------------------
    # PARSERS
    # -----------------------------
    def parse_int(s: str) -> Optional[int]:
        try:
            return int(s)
        except Exception:
            return None

    def parse_range_g(amt: str) -> Optional[Tuple[int, int]]:
        m = re.search(rf"(\d+)\s*{_HYPHENS}\s*(\d+)\s*g\b", amt, flags=re.IGNORECASE)
        if not m:
            return None
        a = parse_int(m.group(1))
        b = parse_int(m.group(2))
        if a is None or b is None:
            return None
        if b < a:
            a, b = b, a
        return a, b

    def parse_single_g(amt: str) -> Optional[int]:
        m = re.search(r"(\d+)\s*g\b", amt, flags=re.IGNORECASE)
        if not m:
            return None
        return parse_int(m.group(1))

    def parse_range_spoons(amt: str) -> Optional[Tuple[int, int, str]]:
        m = re.search(
            rf"(\d+)\s*{_HYPHENS}\s*(\d+)\s*(yemek\s*kaşığı|yemek\s*kaşı|yk)\b",
            amt,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        a = parse_int(m.group(1))
        b = parse_int(m.group(2))
        unit = m.group(3)
        if a is None or b is None:
            return None
        if b < a:
            a, b = b, a
        return a, b, unit

    def parse_single_spoons(amt: str) -> Optional[Tuple[int, str]]:
        m = re.search(r"(?:^|\b)(\d+)\s*(yemek\s*kaşığı|yemek\s*kaşı|yk)\b", amt, flags=re.IGNORECASE)
        if not m:
            return None
        a = parse_int(m.group(1))
        unit = m.group(2)
        if a is None:
            return None
        return a, unit

    def parse_generic_units(amt: str) -> Optional[Tuple[int, str]]:
        m = re.search(r"\b(\d+)\s*(porsiyon|kase|adet|kepçe|kepce|dilim|tam)\b", amt, flags=re.IGNORECASE)
        if not m:
            return None
        cur = parse_int(m.group(1))
        unit = m.group(2)
        if cur is None:
            return None
        return cur, unit

    # -----------------------------
    # LADDER HELPERS
    # -----------------------------
    def next_ladder_up(cur: int, ladder: List[int]) -> Optional[int]:
        for v in ladder:
            if v > cur:
                return v
        return None

    def next_ladder_down(cur: int, ladder: List[int]) -> Optional[int]:
        for v in ladder:
            if v < cur:
                return v
        return None

    def clamp_int(v: int, mn: int, mx: int) -> int:
        if v < mn:
            return mn
        if v > mx:
            return mx
        return v

    # -----------------------------
    # ELIGIBLE CANDIDATES
    # -----------------------------
    candidates: List[Dict[str, Any]] = []
    for it in items:
        amt = current_amount(it)
        if not amt:
            continue
        if not _has_tags(it, must_have=must_have, must_not=must_not):
            continue
        if patterns and (not _regex_any_match(amt, patterns)):
            continue
        candidates.append(it)

    if not candidates:
        dlog(f"[ENERGY] portion_rule_ref no eligible item for {rule_ref}")
        return False

    # ----------------------------------------------------------
    # ✅ DOWN MICRO-BACKSTEP (ÇORBA HARİÇ, SADECE TOUCHED, SADECE ELIGIBLE)
    # ----------------------------------------------------------
    if direction == "down":
        # sadece eligible + daha önce touched + mult>1
        touched_bumped = []
        for it in candidates:
            m = base_mult_of(it)
            if m > 1.0 + 1e-9 and bool(it.get("_portion_touched")):
                # çorba/kepçe micro-backstep'e girmez
                if has_soup_semantics(it):
                    continue
                touched_bumped.append(it)

        if touched_bumped:
            it0 = max(touched_bumped, key=lambda x: base_mult_of(x))
            curm = base_mult_of(it0)

            step_mult = 0.25
            newm = max(1.0, round(curm - step_mult, 4))

            if newm < curm - 1e-9:
                it0["_portion_mult"] = newm
                it0["kcal_scaled"] = int(round(base_kcal_of(it0) * newm))
                it0["_changeable"] = True
                it0["_portion_touched"] = True
                it0["_portion_rule_ref_last"] = rule_ref
                it0["_portion_direction_last"] = "down_micro"

                dlog(f"[ENERGY] micro-backstep: {item_name(it0)} mult {curm:.2f}->{newm:.2f}")
                return True
        # micro-backstep yoksa normal rule akışına devam

    # “değişebilir mi?” kontrolü
    def can_change(it: Dict[str, Any]) -> bool:
        amt = current_amount(it)

        if rtype in ("portion_step_generic", "portion_step_by_pieces", "portion_step_by_ladles", "portion_step_by_slices"):
            pr = parse_generic_units(amt)
            if not pr:
                return False
            cur, _ = pr
            cfg = rule.get("steps") or rule.get("pieces") or rule.get("slices") or rule.get("ladles") or {}
            step = int(cfg.get("step_up" if direction == "up" else "step_down", 1) or 1)
            mn = int(cfg.get("min", 1) or 1)
            mx = int(cfg.get("max", 2) or 2)
            newv = cur + step if direction == "up" else cur - step
            newv = clamp_int(newv, mn, mx)
            return newv != cur

        if rtype == "portion_step_by_spoons":
            spo = rule.get("spoons") or {}
            step = int(spo.get("step_up" if direction == "up" else "step_down", 2) or 2)
            mn = int(spo.get("min", 1) or 1)
            mx = int(spo.get("max", 10) or 10)

            rr = parse_range_spoons(amt)
            if rr:
                a, b, _unit = rr
                na, nb = (a + step, b + step) if direction == "up" else (a - step, b - step)
                if direction == "up":
                    na, nb = clamp_int(na, mn, mx), clamp_int(nb, mn, mx)
                else:
                    na, nb = max(mn, na), max(mn, nb)
                return (na, nb) != (a, b)

            rs = parse_single_spoons(amt)
            if rs:
                a, _unit = rs
                na = a + step if direction == "up" else a - step
                na = clamp_int(na, mn, mx) if direction == "up" else max(mn, na)
                return na != a

            return False

        if rtype == "portion_step_by_grams":
            gcfg = rule.get("grams") or {}
            strategy = (gcfg.get("strategy") or "multiplier").lower()

            mn_abs = int(gcfg.get("min_grams_absolute", 0) or 0)
            mx_abs = int(gcfg.get("max_grams_absolute", 10**9) or 10**9)

            gr = parse_range_g(amt)
            gs = parse_single_g(amt)

            # ladder
            if strategy == "ladder":
                up_list = [int(x) for x in (gcfg.get("ladder_up_grams") or [])]
                down_list = [int(x) for x in (gcfg.get("ladder_down_grams") or [])]
                if not up_list and not down_list:
                    strategy = "multiplier"
                else:
                    cur = None
                    if gr:
                        _a, b = gr
                        cur = b
                    elif gs is not None:
                        cur = gs
                    else:
                        return False

                    if direction == "down" and cur > mx_abs:
                        return True

                    nxt = next_ladder_up(cur, up_list) if direction == "up" else next_ladder_down(cur, down_list)
                    if nxt is None:
                        if direction == "up" and cur < mx_abs:
                            return True
                        if direction == "down" and cur > mn_abs:
                            return True
                        return False

                    nxt = clamp_int(int(nxt), mn_abs, mx_abs)
                    return nxt != cur

            # multiplier
            cur = None
            if gr:
                _a, b = gr
                cur = b
            elif gs is not None:
                cur = gs
            else:
                return False

            if direction == "down" and cur > mx_abs:
                return True

            mult_up = float(gcfg.get("max_multiplier", 1.5) or 1.5)
            mult_down = float(gcfg.get("min_multiplier", 1.5) or 1.5)
            mult = mult_up if direction == "up" else (1.0 / mult_down if mult_down else 1.0)
            new_cur = int(round(cur * mult))
            if direction == "up":
                new_cur = clamp_int(new_cur, mn_abs, mx_abs)
            else:
                new_cur = max(mn_abs, new_cur)
            return new_cur != cur

        return False

    changeable = [it for it in candidates if can_change(it)]
    if not changeable:
        dlog(f"[ENERGY] portion_rule_ref no CHANGEABLE item for {rule_ref} (cands={len(candidates)})")
        return False

    def pick_candidate(changeable_: List[Dict[str, Any]]) -> Dict[str, Any]:
        if direction == "down":
            touched = [
                x for x in changeable_
                if x.get("_portion_touched") and base_mult_of(x) > 1.0 + 1e-9
            ]
            if touched:
                return max(touched, key=lambda x: base_mult_of(x))

            bumped = [x for x in changeable_ if base_mult_of(x) > 1.0 + 1e-9]
            if bumped:
                return max(bumped, key=lambda x: base_mult_of(x))

            return max(changeable_, key=lambda x: base_kcal_of(x))

        return max(changeable_, key=lambda x: base_kcal_of(x))

    candidate = pick_candidate(changeable)
    amt = current_amount(candidate)

    # ==========================================================
    # A) GENERIC (adet/kepçe/dilim/porsiyon/kase/tam)
    # ==========================================================
    if rtype in ("portion_step_generic", "portion_step_by_pieces", "portion_step_by_ladles", "portion_step_by_slices"):
        pr = parse_generic_units(amt)
        if not pr:
            dlog(f"[ENERGY] amount parse failed (generic): {amt}")
            return False
        cur, unit = pr
        cfg = rule.get("steps") or rule.get("pieces") or rule.get("slices") or rule.get("ladles") or {}

        step = int(cfg.get("step_up" if direction == "up" else "step_down", 1) or 1)
        mn = int(cfg.get("min", 1) or 1)
        mx = int(cfg.get("max", 2) or 2)

        newv = cur + step if direction == "up" else cur - step
        newv = clamp_int(newv, mn, mx)
        if newv == cur:
            return False

        new_amt = re.sub(rf"\b{cur}\b(?=\s*{unit}\b)", str(newv), amt, count=1, flags=re.IGNORECASE)
        if new_amt == amt:
            new_amt = re.sub(r"\b" + str(cur) + r"\b", str(newv), amt, count=1)

        return set_scaled(candidate, new_amt, ratio=newv / float(cur))

    # ==========================================================
    # B) SPOONS (yk)
    # ==========================================================
    if rtype == "portion_step_by_spoons":
        spo = rule.get("spoons") or {}
        step = int(spo.get("step_up" if direction == "up" else "step_down", 2) or 2)
        mn = int(spo.get("min", 1) or 1)
        mx = int(spo.get("max", 10) or 10)

        rr = parse_range_spoons(amt)
        if rr:
            a, b, unit = rr
            na, nb = (a + step, b + step) if direction == "up" else (a - step, b - step)

            if direction == "up":
                na, nb = clamp_int(na, mn, mx), clamp_int(nb, mn, mx)
            else:
                na, nb = max(mn, na), max(mn, nb)

            if (na, nb) == (a, b):
                return False

            new_amt = re.sub(
                rf"(\d+)\s*{_HYPHENS}\s*(\d+)\s*({re.escape(unit)})",
                f"{na}-{nb} {unit}",
                amt,
                count=1,
                flags=re.IGNORECASE,
            )

            return set_scaled(candidate, new_amt, ratio=na / float(a))

        rs = parse_single_spoons(amt)
        if rs:
            a, unit = rs
            na = a + step if direction == "up" else a - step
            na = clamp_int(na, mn, mx) if direction == "up" else max(mn, na)
            if na == a:
                return False

            new_amt = re.sub(
                rf"\b{a}\b(?=\s*{re.escape(unit)}\b)",
                str(na),
                amt,
                count=1,
                flags=re.IGNORECASE,
            )
            if new_amt == amt:
                new_amt = re.sub(r"\b" + str(a) + r"\b", str(na), amt, count=1)

            return set_scaled(candidate, new_amt, ratio=na / float(a))

        dlog(f"[ENERGY] spoons parse failed: {amt}")
        return False

    # ==========================================================
    # C) GRAMS (multiplier OR ladder)
    # ==========================================================
    if rtype == "portion_step_by_grams":
        gcfg = rule.get("grams") or {}
        strategy = (gcfg.get("strategy") or "multiplier").lower()

        mn_abs = int(gcfg.get("min_grams_absolute", 0) or 0)
        mx_abs = int(gcfg.get("max_grams_absolute", 10**9) or 10**9)
        clamp_always = bool(gcfg.get("clamp_always", False))

        gr = parse_range_g(amt)
        gs = parse_single_g(amt)

        def apply_clamps(v: int, *, is_up: bool) -> int:
            if is_up:
                return clamp_int(v, mn_abs, mx_abs)
            v2 = max(mn_abs, v)
            if clamp_always:
                v2 = min(mx_abs, v2)
            return v2

        # ---- LADDER ----
        if strategy == "ladder":
            up_list = [int(x) for x in (gcfg.get("ladder_up_grams") or [])]
            down_list = [int(x) for x in (gcfg.get("ladder_down_grams") or [])]

            if up_list or down_list:
                if gr:
                    a, b = gr
                    cur_ref = b
                elif gs is not None:
                    cur_ref = gs
                else:
                    dlog(f"[ENERGY] grams parse failed: {amt}")
                    return False

                if direction == "down" and cur_ref > mx_abs:
                    nxt = mx_abs
                else:
                    nxt = next_ladder_up(cur_ref, up_list) if direction == "up" else next_ladder_down(cur_ref, down_list)
                    if nxt is None:
                        if direction == "up" and cur_ref < mx_abs:
                            nxt = mx_abs
                        elif direction == "down" and cur_ref > mn_abs:
                            nxt = mn_abs
                        else:
                            return False

                nxt = apply_clamps(int(nxt), is_up=(direction == "up"))
                if nxt == cur_ref:
                    return False

                if gr:
                    a, b = gr
                    width = max(0, b - a)
                    nb = nxt
                    na = max(mn_abs, nb - width)
                    na = apply_clamps(int(na), is_up=(direction == "up"))

                    new_amt = re.sub(
                        rf"(\d+)\s*{_HYPHENS}\s*(\d+)\s*g\b",
                        f"{na}-{nb} g",
                        amt,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    return set_scaled(candidate, new_amt, ratio=na / float(a))

                # single
                a = int(cur_ref)
                na = int(nxt)
                new_amt = re.sub(rf"\b{a}\b(?=\s*g\b)", str(na), amt, count=1, flags=re.IGNORECASE)
                if new_amt == amt:
                    new_amt = re.sub(r"(\d+)\s*g\b", f"{na} g", amt, count=1, flags=re.IGNORECASE)
                return set_scaled(candidate, new_amt, ratio=na / float(a))

            # ladder yoksa multiplier'a düş

        # ---- MULTIPLIER ----
        mult_up = float(gcfg.get("max_multiplier", 1.5) or 1.5)
        mult_down = float(gcfg.get("min_multiplier", 1.5) or 1.5)
        mult = mult_up if direction == "up" else (1.0 / mult_down if mult_down else 1.0)

        if gr:
            a, b = gr
            na = int(round(a * mult))
            nb = int(round(b * mult))
            na = apply_clamps(na, is_up=(direction == "up"))
            nb = apply_clamps(nb, is_up=(direction == "up"))
            if (na, nb) == (a, b):
                return False
            new_amt = re.sub(
                rf"(\d+)\s*{_HYPHENS}\s*(\d+)\s*g\b",
                f"{na}-{nb} g",
                amt,
                count=1,
                flags=re.IGNORECASE,
            )
            return set_scaled(candidate, new_amt, ratio=na / float(a))

        if gs is not None:
            a = gs
            na = int(round(a * mult))
            na = apply_clamps(na, is_up=(direction == "up"))
            if na == a:
                return False
            new_amt = re.sub(rf"\b{a}\b(?=\s*g\b)", str(na), amt, count=1, flags=re.IGNORECASE)
            if new_amt == amt:
                new_amt = re.sub(r"(\d+)\s*g\b", f"{na} g", amt, count=1, flags=re.IGNORECASE)
            return set_scaled(candidate, new_amt, ratio=na / float(a))

        dlog(f"[ENERGY] grams parse failed: {amt}")
        return False

    dlog(f"[ENERGY] Unsupported rule type: {rtype}")
    return False


def apply_energy_scaling_rules(
    items: List[Dict[str, Any]],
    hedef_kcal: float,
    meal_name: str,
    pool: List[Dict[str, Any]],
    pattern_rules: Dict[str, Any],
    avoid_tags: Set[str],
    state: "PatternState",
    day: "DayState",
    energy_rules: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """
    ✅ Bu sürüm "genel / tüm hastalıklar" için güvenli ve deterministik bir energy scaling motorudur.

    Düzeltmeler / Ekler:
    1) ✅ Zikzak master loop: her tur now>hedef => down, now<hedef => up
    2) ✅ Tolerance penceresi: abs(now-hedef) <= tolerance => DUR
    3) ✅ Overshoot kontrolü:
       - ADD: zaten güvenli-kcal filtreli
       - INCREASE_PORTION / add_or_increase(rule_ref): olası overshoot olursa geri alır (rollback)
    4) ✅ Step limits: JSON step_templates.limits:
       - max_applications_per_meal
       - max_applications_per_day
       - max_additions_in_meal (add / add_or_increase-add için)
    5) ✅ Daily counters:
       - add/remove/replace zaten günceller
       - rule_ref ile porsiyon art/azalt (ekmek vb.) olduğunda da counter +/- yapar
    6) ✅ no_duplicate_group_in_meal: snack add öncesi
    7) ✅ add_if_missing_daily action desteği (kuruyemiş gibi):
       - gün içinde hiç yoksa ve günlük limit uygunsa 1 kez ekler
    """

    notes: List[str] = []

    # -------------------------
    # DEBUG
    # -------------------------
    def dlog(*args) -> None:
        if globals().get("DEBUG_ENERGY", False):
            print("[ENERGY]", *args)

    def item_name(it: Dict[str, Any]) -> str:
        return (it.get("ad") or it.get("isim") or it.get("name") or it.get("yemek") or "?")

    def item_tags(it: Dict[str, Any]) -> Set[str]:
        try:
            return set(get_food_tags(it) or set())
        except Exception:
            return set()

    def item_tags_str(it: Dict[str, Any], max_n: int = 10) -> str:
        try:
            tags = list(item_tags(it))
            if len(tags) > max_n:
                tags = tags[:max_n] + ["..."]
            return ",".join(tags)
        except Exception:
            return ""

    def summarize_items(max_n: int = 8) -> str:
        out = []
        for it in items[:max_n]:
            mult = it.get("_portion_mult", 1.0)
            out.append(f"{item_name(it)} x{mult}")
        if len(items) > max_n:
            out.append("...")
        return " | ".join(out)

    def total_now() -> float:
        # item_kcal zaten _portion_mult vs dikkate alıyor olmalı
        return float(sum(item_kcal(x) for x in items))

    def clamp(v: float, lo: float, hi: float) -> float:
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def normalize_key(s: str) -> str:
        try:
            return normalize_meal_key(s)
        except Exception:
            return (s or "").strip().lower()

    # -------------------------
    # PORTION SANITY
    # -------------------------
    ps = (energy_rules.get("portion_sanity") or {})
    min_fraction = float(ps.get("min_fraction", 0.5) or 0.5)
    max_fraction = float(ps.get("max_fraction", 2.0) or 2.0)
    allowed_steps = ps.get("allowed_steps") or ["0.5_porsiyon", "1_porsiyon"]

    max_grams_default = ps.get("max_grams_default")  # optional
    max_grams_by_tag = ps.get("max_grams_by_tag") or {}

    def _get_item_grams(it: Dict[str, Any]) -> Optional[float]:
        for k in ("gram", "grams", "miktar_g", "miktarGram", "quantity_g", "portion_g"):
            if k in it and it.get(k) is not None:
                try:
                    return float(it.get(k))
                except Exception:
                    pass
        return None

    def _set_item_grams(it: Dict[str, Any], new_g: float) -> None:
        for k in ("gram", "grams", "miktar_g", "miktarGram", "quantity_g", "portion_g"):
            if k in it:
                it[k] = float(new_g)
                return

    def _max_grams_for_item(it: Dict[str, Any]) -> Optional[float]:
        tags = item_tags(it)
        lims = []
        for t, v in (max_grams_by_tag or {}).items():
            try:
                if t in tags:
                    lims.append(float(v))
            except Exception:
                continue
        if lims:
            return min(lims)
        try:
            if max_grams_default is not None:
                return float(max_grams_default)
        except Exception:
            return None
        return None

    def enforce_portion_sanity_all() -> None:
        for it in items:
            now_mult = float(it.get("_portion_mult", 1.0) or 1.0)
            clamped = clamp(now_mult, min_fraction, max_fraction)
            if abs(clamped - now_mult) > 1e-9:
                dlog("SANITY clamp _portion_mult:", item_name(it), "from", now_mult, "to", clamped)
                it["_portion_mult"] = clamped

            lim_g = _max_grams_for_item(it)
            if lim_g is None:
                continue
            g = _get_item_grams(it)
            if g is None:
                continue
            if g > float(lim_g) + 1e-9:
                dlog("SANITY clamp grams:", item_name(it), "from", g, "to", lim_g, "tags=", item_tags_str(it))
                _set_item_grams(it, float(lim_g))

    def parse_step_val(step_name: str) -> float:
        step_name = step_name or "0.5_porsiyon"
        if step_name not in allowed_steps:
            dlog("STEP invalid:", step_name, "-> fallback 0.5_porsiyon")
            step_name = "0.5_porsiyon"
        return 0.5 if "0.5" in step_name else 1.0

    # -------------------------
    # LOCKS (meal_pattern + energy_rules)
    # -------------------------
    if normalize_key(meal_name) == normalize_key("yatarken") and ("yatarken_ara_ogun" in (pattern_rules or {})):
        dlog("LOCK pattern_rules: yatarken_ara_ogun -> scaling skipped")
        return items, int(round(total_now())), notes

    if is_meal_locked_by_energy_rules(meal_name, items, energy_rules):
        dlog("LOCK energy_rules.locks -> scaling skipped")
        return items, int(round(total_now())), notes

    # -------------------------
    # GLOBAL EXCLUSIONS
    # -------------------------
    gex = energy_rules.get("global_exclusions") or {}
    hard_exclude_tags_any = set(gex.get("hard_exclude_tags_any") or [])
    hard_exclude_targets_any = [str(x).lower() for x in (gex.get("hard_exclude_targets_any") or [])]

    def food_name_lc(f: Dict[str, Any]) -> str:
        return (f.get("ad") or f.get("isim") or f.get("name") or f.get("yemek") or "").lower()

    def has_any_tag(food: Dict[str, Any], tags_any: List[str]) -> bool:
        ft = item_tags(food)
        return any(t in ft for t in (tags_any or []))

    def is_globally_excluded(food: Dict[str, Any]) -> bool:
        tags = item_tags(food)
        if hard_exclude_tags_any and any(t in tags for t in hard_exclude_tags_any):
            return True
        nm = food_name_lc(food)
        if hard_exclude_targets_any and any(t in nm for t in hard_exclude_targets_any):
            return True
        return False

    # -------------------------
    # PICK HELPERS
    # -------------------------
    def pick_by_tags(tags_any: List[str], local_avoid: Set[str]) -> Optional[Dict[str, Any]]:
        cands = [f for f in pool if has_any_tag(f, tags_any) and (not is_globally_excluded(f))]
        best = pick_best(cands, local_avoid, day, pattern_rules, state, meal_name, items)
        if best is not None:
            return best
        return cands[0] if cands else None

    def bump_portion_generic(tags_any: List[str], step: float, cap: float) -> bool:
        cap = min(float(cap), float(max_fraction))
        for it in items:
            if has_any_tag(it, tags_any):
                now = float(it.get("_portion_mult", 1.0) or 1.0)
                if now + 1e-9 < cap:
                    it["_portion_mult"] = clamp(now + step, min_fraction, cap)
                    dlog("BUMP_PORTION OK:", item_name(it), "from", now, "to", it["_portion_mult"], "tags_any=", tags_any)
                    return True
        return False

    def dec_portion_generic(tags_any: List[str], step: float) -> bool:
        for it in items:
            if has_any_tag(it, tags_any):
                now_mult = float(it.get("_portion_mult", 1.0) or 1.0)
                if now_mult - 1e-9 > min_fraction:
                    it["_portion_mult"] = clamp(now_mult - step, min_fraction, max_fraction)
                    dlog("DEC_PORTION OK:", item_name(it), "from", now_mult, "to", it["_portion_mult"])
                    return True
        return False

    # ------------------------------------------------------------------
    # DAILY COUNTERS
    # ------------------------------------------------------------------
    def _ensure_day_counters() -> Dict[str, int]:
        if not hasattr(day, "counters") or getattr(day, "counters") is None:
            try:
                setattr(day, "counters", {})
            except Exception:
                pass
        try:
            return getattr(day, "counters")
        except Exception:
            return {}

    def get_daily_counter_limit(counter_key: str) -> Optional[int]:
        sc = energy_rules.get("state_counters") or {}
        for c in (sc.get("counters") or []):
            if c.get("key") == counter_key:
                try:
                    return int(c.get("max"))
                except Exception:
                    return None
        return None

    def get_daily_counter_value(counter_key: str) -> int:
        counters = _ensure_day_counters()
        try:
            return int(counters.get(counter_key, 0))
        except Exception:
            return 0

    def inc_daily_counter(counter_key: str, inc: int = 1) -> None:
        counters = _ensure_day_counters()
        try:
            newv = int(counters.get(counter_key, 0)) + int(inc)
            if newv < 0:
                newv = 0
            counters[counter_key] = newv
        except Exception:
            pass

    def bump_counter_for_food(food: Dict[str, Any], inc: int) -> None:
        tags = item_tags(food)
        if "meyve" in tags:
            inc_daily_counter("fruit_portions_used", inc)
        elif "kuruyemis" in tags or "kuruyemiş" in tags:
            inc_daily_counter("nuts_portions_used", inc)
        elif "ekmek" in tags:
            inc_daily_counter("bread_portions_used", inc)

    def bump_counter_for_rule_ref(rule_ref: str, direction: str) -> None:
        """
        rule_ref ile porsiyon artır/azalt olduğunda günlük sayaçları güncellemek için.
        Bu, özellikle ekmek 1<->2 dilim gibi kurallarda gerekli.
        """
        rule = energy_rules.get(rule_ref) or {}
        ef = (rule.get("eligible_food_filter") or {})
        must_have = ef.get("must_have_food_tags_any") or ef.get("must_have_food_tags_all") or []

        # Hangi sayacı artıracağız?
        key = None
        if any(t in ("ekmek", "tam_tahilli_ekmek", "tam_tahil", "bulgur") for t in must_have):
            key = "bread_portions_used"
        elif any(t in ("meyve",) for t in must_have):
            key = "fruit_portions_used"
        elif any(t in ("kuruyemis", "kuruyemiş", "cerez", "yag_tohum") for t in must_have):
            key = "nuts_portions_used"

        if not key:
            return

        inc = +1 if direction == "up" else -1
        inc_daily_counter(key, inc)

    # ------------------------------------------------------------------
    # PRECONDITIONS
    # ------------------------------------------------------------------
    def meal_contains_any_tags(tag_list: List[str]) -> bool:
        if not tag_list:
            return False
        for it in items:
            it_tags = item_tags(it)
            for t in tag_list:
                if t in it_tags:
                    return True
        return False

    def check_preconditions(stepdef: Dict[str, Any]) -> Tuple[bool, str]:
        pre = stepdef.get("preconditions") or []
        for p in pre:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "meal_must_not_contain_tags_any":
                tags_any = p.get("food_tags_any") or []
                if meal_contains_any_tags(tags_any):
                    return False, f"meal contains tags_any={tags_any}"
            elif ptype == "respect_daily_counter":
                key = p.get("counter_key")
                if key:
                    lim = get_daily_counter_limit(key)
                    if lim is not None:
                        used = get_daily_counter_value(key)
                        if used >= lim:
                            return False, f"daily counter {key} used={used} >= lim={lim}"
        return True, ""

    # ------------------------------------------------------------------
    # no_duplicate_group_in_meal (snack)
    # ------------------------------------------------------------------
    def violates_no_duplicate_group_in_meal(target_tags_any: List[str]) -> Tuple[bool, str]:
        nd = energy_rules.get("no_duplicate_group_in_meal") or {}
        applies = {normalize_key(x) for x in (nd.get("applies_to_meals") or [])}
        if normalize_key(meal_name) not in applies:
            return False, ""
        groups = nd.get("groups") or []
        for g in groups:
            if not isinstance(g, dict):
                continue
            gtags = g.get("food_tags_any") or []
            try:
                max_count = int(g.get("max_count", 999))
            except Exception:
                max_count = 999
            if not gtags:
                continue
            if any(t in (target_tags_any or []) for t in gtags):
                cnt = 0
                for it in items:
                    it_tags = item_tags(it)
                    if any(t in it_tags for t in gtags):
                        cnt += 1
                if cnt >= max_count:
                    return True, f"group {g.get('group_id')} already cnt={cnt} max={max_count}"
        return False, ""

    # ------------------------------------------------------------------
    # step limit trackers
    # ------------------------------------------------------------------
    # gün boyunca birden çok öğün çağrılabileceği için day üzerinde tutmak en güvenlisi
    if not hasattr(day, "energy_step_counts") or getattr(day, "energy_step_counts") is None:
        try:
            setattr(day, "energy_step_counts", {})
        except Exception:
            pass

    def _get_day_step_counts() -> Dict[str, int]:
        try:
            return getattr(day, "energy_step_counts") or {}
        except Exception:
            return {}

    def _inc_day_step(step_id: str, inc: int = 1) -> None:
        dct = _get_day_step_counts()
        dct[step_id] = int(dct.get(step_id, 0)) + int(inc)
        if dct[step_id] < 0:
            dct[step_id] = 0
        try:
            setattr(day, "energy_step_counts", dct)
        except Exception:
            pass

    # öğün bazlı sayım: bu fonksiyon çağrısı boyunca yeterli
    meal_step_counts: Dict[str, int] = {}
    meal_additions_count = 0

    def _limits_allow(stepdef: Dict[str, Any]) -> Tuple[bool, str]:
        step_id = stepdef.get("step_id") or ""
        lim = (stepdef.get("limits") or {})

        # per meal
        if lim.get("max_applications_per_meal") is not None:
            try:
                mmax = int(lim.get("max_applications_per_meal"))
                if meal_step_counts.get(step_id, 0) >= mmax:
                    return False, f"max_applications_per_meal reached for {step_id}"
            except Exception:
                pass

        # per day
        if lim.get("max_applications_per_day") is not None:
            try:
                dmax = int(lim.get("max_applications_per_day"))
                used = _get_day_step_counts().get(step_id, 0)
                if used >= dmax:
                    return False, f"max_applications_per_day reached for {step_id}"
            except Exception:
                pass

        # add count per meal
        if lim.get("max_additions_in_meal") is not None:
            try:
                amax = int(lim.get("max_additions_in_meal"))
                if meal_additions_count >= amax:
                    return False, f"max_additions_in_meal reached (count={meal_additions_count})"
            except Exception:
                pass

        return True, ""

    def _mark_step_used(stepdef: Dict[str, Any], inc_additions: bool = False) -> None:
        nonlocal meal_additions_count
        step_id = stepdef.get("step_id") or ""
        meal_step_counts[step_id] = int(meal_step_counts.get(step_id, 0)) + 1
        _inc_day_step(step_id, 1)
        if inc_additions:
            meal_additions_count += 1

    # ------------------------------------------------------------------
    # add_if_missing_daily helpers
    # ------------------------------------------------------------------
    def day_has_any_tag(tags_any: List[str]) -> bool:
        """
        Gün içinde bu öğeye gelmeden önce eklenmişler dahil:
        day üzerinde day.all_items gibi bir yapı yoksa en azından mevcut meal items'ta bakar.
        Eğer sende günün tüm öğünleri listesi varsa, bunu o liste ile değiştirebilirsin.
        """
        # Eğer DayState'de günün tüm öğünleri tutuluyorsa burada genişletebilirsin.
        for it in items:
            if has_any_tag(it, tags_any):
                return True
        return False

    # -------------------------
    # START sanity
    # -------------------------
    enforce_portion_sanity_all()
    start_kcal = total_now()
    dlog(f"START meal={meal_name!r} hedef={float(hedef_kcal):.0f} start={start_kcal:.0f} items={len(items)}")
    dlog("START items:", summarize_items())

    # ------------------------------------------------------------------
    # ACTION RUNNER
    # ------------------------------------------------------------------
    def run_actions(
        block: Dict[str, Any],
        direction: str,
        max_steps_total: int,
        hedef_kcal: float,
        tolerance_kcal: float,
    ) -> bool:
        po = (block.get("priority_order") or [])
        if not isinstance(po, list) or not po:
            dlog(f"RUN_ACTIONS dir={direction}: priority_order EMPTY -> no-op")
            return False

        hard_avoid = set(block.get("hard_avoid") or [])
        local_avoid = set(avoid_tags) | hard_avoid

        upper_ok = float(hedef_kcal) + float(tolerance_kcal)
        lower_ok = float(hedef_kcal) - float(tolerance_kcal)

        steps_done = 0
        progressed_any = False

        for loop_i in range(24):
            now = total_now()
            dlog(f"LOOP#{loop_i} dir={direction} now={now:.0f} steps_done={steps_done}/{max_steps_total}")
            dlog("  items:", summarize_items())

            # Stop by tolerance window
            if direction == "up" and now >= lower_ok:
                dlog("STOP: reached lower tolerance bound (up)")
                break
            if direction == "down" and now <= upper_ok:
                dlog("STOP: reached upper tolerance bound (down)")
                break
            if steps_done >= max_steps_total:
                dlog("STOP: reached max_steps_total")
                break

            progressed = False

            for step_i, stepdef in enumerate(po):
                if not isinstance(stepdef, dict):
                    continue

                # meal filter
                meal_types = stepdef.get("meal_types") or stepdef.get("applies_to_meals") or []
                if meal_types:
                    my_key = normalize_key(meal_name)
                    mt_norm = {normalize_key(x) for x in meal_types}
                    if my_key not in mt_norm:
                        continue

                ok_pre, why = check_preconditions(stepdef)
                if not ok_pre:
                    continue

                ok_lim, why_lim = _limits_allow(stepdef)
                if not ok_lim:
                    continue

                act = stepdef.get("action")

                # -------------------------
                # ADD
                # -------------------------
                if act == "add":
                    if direction != "up":
                        continue

                    tags_any = stepdef.get("target_food_tags_any") or []
                    if not tags_any:
                        continue

                    vio, _whyv = violates_no_duplicate_group_in_meal(tags_any)
                    if vio:
                        continue

                    now_before = total_now()
                    max_add_kcal = (upper_ok - now_before)
                    if max_add_kcal <= 1e-9:
                        continue

                    safe_cands = [
                        f for f in pool
                        if has_any_tag(f, tags_any)
                        and (not is_globally_excluded(f))
                        and float(item_kcal(f) or 0.0) <= (max_add_kcal + 1e-9)
                    ]
                    if not safe_cands:
                        continue

                    cand = pick_best(safe_cands, local_avoid, day, pattern_rules, state, meal_name, items)
                    if cand is None:
                        cand = min(safe_cands, key=lambda x: float(item_kcal(x) or 0.0))

                    ok = add_item(items, cand, day, pattern_rules, state, meal_name, local_avoid)
                    if ok:
                        bump_counter_for_food(cand, +1)
                        enforce_portion_sanity_all()
                        progressed = True
                        progressed_any = True
                        steps_done += 1
                        _mark_step_used(stepdef, inc_additions=True)
                        notes.append(stepdef.get("note") or "Kalori ayarı: ekleme yapıldı.")
                        break

                # -------------------------
                # ADD_IF_MISSING_DAILY  (kuruyemiş vb.)
                # -------------------------
                elif act == "add_if_missing_daily":
                    if direction != "up":
                        continue

                    # uses_rule_ref beklenir
                    ref = stepdef.get("uses_rule_ref")
                    if not ref:
                        continue

                    rr = energy_rules.get(ref) or {}
                    tags_any = rr.get("target_food_tags_any") or stepdef.get("target_food_tags_any") or []
                    if not tags_any:
                        continue

                    # gün içinde var mı?
                    if day_has_any_tag(tags_any):
                        continue

                    # günlük limit
                    counter_key = rr.get("daily_counter_key") or "nuts_portions_used"
                    daily_max = rr.get("daily_max")
                    if daily_max is None:
                        daily_max = get_daily_counter_limit(counter_key)
                    try:
                        daily_max = int(daily_max) if daily_max is not None else None
                    except Exception:
                        daily_max = None

                    if daily_max is not None and get_daily_counter_value(counter_key) >= daily_max:
                        continue

                    # snack duplicate group
                    vio, _whyv = violates_no_duplicate_group_in_meal(tags_any)
                    if vio:
                        continue

                    # kcal güvenliği (overshoot istemiyoruz)
                    now_before = total_now()
                    max_add_kcal = (upper_ok - now_before)
                    if max_add_kcal <= 1e-9:
                        continue

                    safe_cands = [
                        f for f in pool
                        if has_any_tag(f, tags_any)
                        and (not is_globally_excluded(f))
                        and float(item_kcal(f) or 0.0) <= (max_add_kcal + 1e-9)
                    ]
                    if not safe_cands:
                        continue

                    cand = pick_best(safe_cands, local_avoid, day, pattern_rules, state, meal_name, items)
                    if cand is None:
                        cand = min(safe_cands, key=lambda x: float(item_kcal(x) or 0.0))

                    ok = add_item(items, cand, day, pattern_rules, state, meal_name, local_avoid)
                    if ok:
                        # günlük sayaçları doğru artır
                        # burada counter_key'e göre artıralım
                        inc_daily_counter(counter_key, +1)
                        enforce_portion_sanity_all()
                        progressed = True
                        progressed_any = True
                        steps_done += 1
                        _mark_step_used(stepdef, inc_additions=True)
                        notes.append(stepdef.get("note") or rr.get("note") or "Kalori ayarı: eksik günlük ekleme yapıldı.")
                        break

                # -------------------------
                # ADD_OR_INCREASE
                # -------------------------
                elif act == "add_or_increase":
                    ref = stepdef.get("uses_rule_ref")

                    # (A) önce portion rule_ref (bu artış/azalış olabilir)
                    if ref:
                        before_snapshot = [dict(x) for x in items]
                        before_kcal = total_now()

                        ok = apply_portion_rule_ref(
                            items=items,
                            energy_rules=energy_rules,
                            rule_ref=ref,
                            direction=("up" if direction == "up" else "down"),
                            dlog=dlog,
                        )
                        enforce_portion_sanity_all()

                        if ok:
                            after_kcal = total_now()
                            # ✅ Overshoot rollback (sadece up tarafında anlamlı)
                            if direction == "up" and after_kcal > upper_ok + 1e-9:
                                dlog("ROLLBACK: portion increase overshot upper_ok", after_kcal, ">", upper_ok)
                                items[:] = before_snapshot
                                enforce_portion_sanity_all()
                                ok = False
                            elif direction == "down" and after_kcal < lower_ok - 1e-9:
                                # down'da aşırı azaltım istemezsen bunu da koru
                                dlog("ROLLBACK: portion decrease overshot below lower_ok", after_kcal, "<", lower_ok)
                                items[:] = before_snapshot
                                enforce_portion_sanity_all()
                                ok = False

                        if ok:
                            bump_counter_for_rule_ref(ref, direction)
                            progressed = True
                            progressed_any = True
                            steps_done += 1
                            _mark_step_used(stepdef, inc_additions=False)
                            notes.append(stepdef.get("note") or "Kalori ayarı: porsiyon artır/azalt yapıldı.")
                            break

                    # (B) add'e düş (sadece up)
                    if direction != "up":
                        continue

                    tags_any = stepdef.get("target_food_tags_any") or []
                    if (not tags_any) and ref:
                        r0 = energy_rules.get(ref) or {}
                        ef0 = r0.get("eligible_food_filter") or {}
                        tags_any = ef0.get("must_have_food_tags_any") or ef0.get("must_have_food_tags_all") or []

                    if not tags_any:
                        continue

                    vio, _whyv = violates_no_duplicate_group_in_meal(tags_any)
                    if vio:
                        continue

                    now_before = total_now()
                    max_add_kcal = upper_ok - now_before
                    if max_add_kcal <= 1e-9:
                        continue

                    safe_cands = [
                        f for f in pool
                        if has_any_tag(f, tags_any)
                        and (not is_globally_excluded(f))
                        and float(item_kcal(f) or 0.0) <= (max_add_kcal + 1e-9)
                    ]
                    if not safe_cands:
                        continue

                    cand = pick_best(safe_cands, local_avoid, day, pattern_rules, state, meal_name, items)
                    if cand is None:
                        cand = min(safe_cands, key=lambda x: float(item_kcal(x) or 0.0))

                    ok2 = add_item(items, cand, day, pattern_rules, state, meal_name, local_avoid)
                    if ok2:
                        bump_counter_for_food(cand, +1)
                        enforce_portion_sanity_all()
                        progressed = True
                        progressed_any = True
                        steps_done += 1
                        _mark_step_used(stepdef, inc_additions=True)
                        notes.append(stepdef.get("note") or "Kalori ayarı: ekleme yapıldı.")
                        break

                # -------------------------
                # INCREASE_PORTION
                # -------------------------
                elif act == "increase_portion":
                    if direction != "up":
                        continue

                    if stepdef.get("uses_rule_ref"):
                        ref = stepdef["uses_rule_ref"]

                        before_snapshot = [dict(x) for x in items]
                        before_kcal = total_now()

                        ok = apply_portion_rule_ref(
                            items=items,
                            energy_rules=energy_rules,
                            rule_ref=ref,
                            direction="up",
                            dlog=dlog,
                        )
                        enforce_portion_sanity_all()

                        if ok:
                            after_kcal = total_now()
                            if after_kcal > upper_ok + 1e-9:
                                dlog("ROLLBACK: increase_portion overshot upper_ok", after_kcal, ">", upper_ok)
                                items[:] = before_snapshot
                                enforce_portion_sanity_all()
                                ok = False

                        if ok:
                            bump_counter_for_rule_ref(ref, "up")
                            progressed = True
                            progressed_any = True
                            steps_done += 1
                            _mark_step_used(stepdef, inc_additions=False)
                            notes.append(stepdef.get("note") or "Kalori ayarı: porsiyon artırıldı.")
                            break
                        continue

                    tags_any = stepdef.get("target_food_tags_any") or ["scalable"]
                    step_val = parse_step_val(stepdef.get("step") or "0.5_porsiyon")

                    before_snapshot = [dict(x) for x in items]
                    ok = bump_portion_generic(tags_any, step=step_val, cap=max_fraction)
                    enforce_portion_sanity_all()

                    if ok:
                        after_kcal = total_now()
                        if after_kcal > upper_ok + 1e-9:
                            dlog("ROLLBACK: generic increase overshot upper_ok", after_kcal, ">", upper_ok)
                            items[:] = before_snapshot
                            enforce_portion_sanity_all()
                            ok = False

                    if ok:
                        progressed = True
                        progressed_any = True
                        steps_done += 1
                        _mark_step_used(stepdef, inc_additions=False)
                        notes.append(stepdef.get("note") or "Kalori ayarı: porsiyon artırıldı.")
                        break

                # -------------------------
                # REMOVE_OR_REDUCE
                # -------------------------
                elif act == "remove_or_reduce":
                    if direction != "down":
                        continue

                    tags_any = stepdef.get("target_food_tags_any") or []
                    if not tags_any:
                        continue

                    removable_idx = [i for i, it in enumerate(items) if has_any_tag(it, tags_any)]
                    if not removable_idx:
                        continue

                    def safe_kcal(i: int) -> float:
                        try:
                            return float(item_kcal(items[i]) or 0.0)
                        except Exception:
                            return 0.0

                    best_i = max(removable_idx, key=safe_kcal)
                    removed_item = items.pop(best_i)

                    bump_counter_for_food(removed_item, -1)
                    enforce_portion_sanity_all()

                    progressed = True
                    progressed_any = True
                    steps_done += 1
                    _mark_step_used(stepdef, inc_additions=False)
                    notes.append(stepdef.get("note") or "Kalori ayarı: 1 kalem çıkarıldı.")
                    break

                # -------------------------
                # DECREASE_PORTION
                # -------------------------
                elif act == "decrease_portion":
                    if direction != "down":
                        continue

                    if stepdef.get("uses_rule_ref"):
                        ref = stepdef["uses_rule_ref"]

                        before_snapshot = [dict(x) for x in items]
                        ok = apply_portion_rule_ref(
                            items=items,
                            energy_rules=energy_rules,
                            rule_ref=ref,
                            direction="down",
                            dlog=dlog,
                        )
                        enforce_portion_sanity_all()

                        if ok:
                            after_kcal = total_now()
                            if after_kcal < lower_ok - 1e-9:
                                dlog("ROLLBACK: decrease_portion undershot lower_ok", after_kcal, "<", lower_ok)
                                items[:] = before_snapshot
                                enforce_portion_sanity_all()
                                ok = False

                        if ok:
                            bump_counter_for_rule_ref(ref, "down")
                            progressed = True
                            progressed_any = True
                            steps_done += 1
                            _mark_step_used(stepdef, inc_additions=False)
                            notes.append(stepdef.get("note") or "Kalori ayarı: porsiyon azaltıldı.")
                            break
                        continue

                    tags_any = stepdef.get("target_food_tags_any") or ["scalable"]
                    step_val = parse_step_val(stepdef.get("step") or "0.5_porsiyon")

                    before_snapshot = [dict(x) for x in items]
                    ok = dec_portion_generic(tags_any, step=step_val)
                    enforce_portion_sanity_all()

                    if ok:
                        after_kcal = total_now()
                        if after_kcal < lower_ok - 1e-9:
                            dlog("ROLLBACK: generic decrease undershot lower_ok", after_kcal, "<", lower_ok)
                            items[:] = before_snapshot
                            enforce_portion_sanity_all()
                            ok = False

                    if ok:
                        progressed = True
                        progressed_any = True
                        steps_done += 1
                        _mark_step_used(stepdef, inc_additions=False)
                        notes.append(stepdef.get("note") or "Kalori ayarı: porsiyon azaltıldı.")
                        break

                # -------------------------
                # REPLACE_WITH
                # -------------------------
                elif act == "replace_with":
                    # Bu action hem up hem down olabilir ama genelde down için kullanılır.
                    rep_targets = stepdef.get("replace_target_tags_any") or []
                    repl_tags = stepdef.get("replacement_tags_any") or []
                    if not rep_targets or not repl_tags:
                        continue

                    idx = None
                    for i, it in enumerate(items):
                        if has_any_tag(it, rep_targets):
                            idx = i
                            break
                    if idx is None:
                        continue

                    removed = items.pop(idx)
                    cand = pick_by_tags(repl_tags, local_avoid)
                    if cand and add_item(items, cand, day, pattern_rules, state, meal_name, local_avoid):
                        bump_counter_for_food(removed, -1)
                        bump_counter_for_food(cand, +1)
                        enforce_portion_sanity_all()

                        # overshoot güvenliği (up ise)
                        after_kcal = total_now()
                        if direction == "up" and after_kcal > upper_ok + 1e-9:
                            # revert
                            items.pop()  # remove added (son eklenen)
                            items.insert(idx, removed)
                            enforce_portion_sanity_all()
                            continue
                        if direction == "down" and after_kcal < lower_ok - 1e-9:
                            items.pop()
                            items.insert(idx, removed)
                            enforce_portion_sanity_all()
                            continue

                        progressed = True
                        progressed_any = True
                        steps_done += 1
                        _mark_step_used(stepdef, inc_additions=False)
                        notes.append(stepdef.get("note") or "Kalori ayarı: değişim yapıldı.")
                        break
                    else:
                        items.insert(idx, removed)

            if not progressed:
                dlog("STOP: no progressed in this loop")
                break

        return progressed_any

    # ==========================================================
    # ✅ ZİKZAK MASTER LOOP
    # ==========================================================
    MAX_MASTER_LOOPS = int(energy_rules.get("max_master_loops", 6) or 6)
    TOLERANCE_KCAL = float(energy_rules.get("tolerance_kcal", energy_rules.get("overshoot_tolerance_kcal", 50)) or 50)

    for master_i in range(MAX_MASTER_LOOPS):
        now_kcal = total_now()
        diff = float(now_kcal) - float(hedef_kcal)

        dlog(f"MASTER#{master_i} now={now_kcal:.0f} hedef={float(hedef_kcal):.0f} diff={diff:+.0f}")

        if abs(diff) <= TOLERANCE_KCAL:
            dlog("STOP: within ±tolerance")
            break

        direction = "down" if diff > 0 else "up"

        abs_delta = abs(diff)
        band_id, band_max_steps = pick_band_id_and_max_steps(abs_delta, energy_rules)
        dlog(f"BAND_SELECT abs_delta={abs_delta:.0f} -> band_id={band_id} band_max_steps={band_max_steps} dir={direction}")

        progressed = False

        if direction == "up" and isinstance(energy_rules.get("scale_up"), dict):
            up_block = dict(energy_rules["scale_up"])
            up_po = energy_rules_to_priority_order(energy_rules, "up", band_id)
            up_block["priority_order"] = up_po
            progressed = run_actions(
                up_block,
                direction="up",
                max_steps_total=band_max_steps,
                hedef_kcal=hedef_kcal,
                tolerance_kcal=TOLERANCE_KCAL,
            )

        elif direction == "down" and isinstance(energy_rules.get("scale_down"), dict):
            down_block = dict(energy_rules["scale_down"])
            down_po = energy_rules_to_priority_order(energy_rules, "down", band_id)
            down_block["priority_order"] = down_po
            progressed = run_actions(
                down_block,
                direction="down",
                max_steps_total=band_max_steps,
                hedef_kcal=hedef_kcal,
                tolerance_kcal=TOLERANCE_KCAL,
            )
        else:
            dlog("STOP: no applicable scale block for direction=", direction)
            break

        enforce_portion_sanity_all()

        if not progressed:
            dlog("STOP: master loop no progress")
            break

        now2 = total_now()
        if abs(now2 - float(hedef_kcal)) <= TOLERANCE_KCAL:
            dlog("STOP: reached tolerance after action")
            break

    end_kcal = total_now()
    dlog(f"END meal={meal_name!r} end={end_kcal:.0f} diff={end_kcal - start_kcal:+.0f} notes={len(notes)}")
    dlog("END items:", summarize_items())

    return items, int(round(end_kcal)), notes

# Merkezi kalori ayarlayıcı
# - ANA MOTOR mantık içermez
# - Sadece hastalık JSON'undan gelen energy_scaling_rules varsa uygular
# =========================

# Merkezi kalori ayarlayıcı
# - ANA MOTOR mantık içermez
# - Sadece hastalık JSON'undan gelen energy_scaling_rules varsa uygular
# =========================
def adjust_meal_calories_by_portion(
    items: List[Dict[str, Any]],
    hedef_kcal: int,
    meal_name: str,
    pool: List[Dict[str, Any]],
    pattern_rules: Dict[str, Any],
    state: Dict[str, Any],
    day_index: int,
    avoid_set: Set[str],
    energy_rules: Optional[Dict[str, Any]],
    max_iter: int = 20,
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """
    Öğün kalorisini hedefe yaklaştırır.

    - Eski energy_rules şeması: apply_energy_scaling_rules(...) kullanılır.
    - Yeni (LLM) şeması: energy_rules = {scale_up_order, scale_down_order, locks}
      Bu modda basit bir "portion artır/azalt" + gerekirse "tag'e göre yeni item ekle/çıkar" yaklaşımı kullanılır.
    """
    logs: List[str] = []
    total_kcal = sum(item_kcal(it) for it in items)

    if not energy_rules or not isinstance(energy_rules, dict) or not energy_rules:
        return items, total_kcal, logs

    # --- Yeni (LLM) şema: scale_up_order / scale_down_order
    if isinstance(energy_rules.get("scale_up_order"), list) or isinstance(energy_rules.get("scale_down_order"), list):
        scale_up = [t.strip() for t in energy_rules.get("scale_up_order", []) if isinstance(t, str) and t.strip()]
        scale_down = [t.strip() for t in energy_rules.get("scale_down_order", []) if isinstance(t, str) and t.strip()]
        locks = set([t.strip() for t in energy_rules.get("locks", []) if isinstance(t, str) and t.strip()])

        def _increase_portion(it: Dict[str, Any], step: float = 0.25, max_mult: float = 2.0) -> bool:
            cur = float(it.get("portion_multiplier", 1.0) or 1.0)
            if cur >= max_mult:
                return False
            it["portion_multiplier"] = round(min(max_mult, cur + step), 2)
            return True

        def _decrease_portion(it: Dict[str, Any], step: float = 0.25, min_mult: float = 0.5) -> bool:
            cur = float(it.get("portion_multiplier", 1.0) or 1.0)
            if cur <= min_mult:
                return False
            it["portion_multiplier"] = round(max(min_mult, cur - step), 2)
            return True

        def _pick_from_pool_by_tag(tag: str) -> Optional[Dict[str, Any]]:
            # avoid_set filtre + pattern ihlali yok
            for f in pool:
                if not isinstance(f, dict):
                    continue
                tags = f.get("tags") or []
                if not (isinstance(tags, list) and tag in tags):
                    continue
                if not can_pick(f, avoid_set):
                    continue
                # pattern ihlali kontrolü (tag bazlı)
                if violates_pattern(f, pattern_rules, state, day_index, meal_name):
                    continue
                return dict(f)  # copy
            return None

        tol = max(25, int(hedef_kcal * 0.05))

        for _ in range(max_iter):
            total_kcal = sum(item_kcal(it) for it in items)
            if abs(total_kcal - hedef_kcal) <= tol:
                break

            if total_kcal < hedef_kcal:
                did = False

                # 1) mevcut öğünde scale_up tag'li item varsa porsiyon artır
                for tag in scale_up:
                    if tag in locks:
                        continue
                    for it in items:
                        tags = it.get("tags") or []
                        if isinstance(tags, list) and tag in tags:
                            if _increase_portion(it):
                                logs.append(f"[{meal_name}] +portion {it.get('name','?')} ({tag})")
                                did = True
                                break
                    if did:
                        break
                if did:
                    continue

                # 2) yoksa havuzdan item ekle
                for tag in scale_up:
                    if tag in locks:
                        continue
                    cand = _pick_from_pool_by_tag(tag)
                    if cand:
                        cand["portion_multiplier"] = 1.0
                        items.append(cand)
                        logs.append(f"[{meal_name}] +add {cand.get('name','?')} ({tag})")
                        did = True
                        break

                if not did:
                    break

            else:  # total_kcal > hedef_kcal
                did = False

                # 1) scale_down tag'li item varsa porsiyon azalt
                for tag in scale_down:
                    if tag in locks:
                        continue
                    for it in items:
                        tags = it.get("tags") or []
                        if isinstance(tags, list) and tag in tags:
                            if _decrease_portion(it):
                                logs.append(f"[{meal_name}] -portion {it.get('name','?')} ({tag})")
                                did = True
                                break
                    if did:
                        break
                if did:
                    continue

                # 2) yoksa en yüksek kalorili (lock olmayan) item çıkar
                removable = []
                for it in items:
                    tags = it.get("tags") or []
                    if isinstance(tags, list) and any(t in locks for t in tags):
                        continue
                    removable.append(it)
                if removable:
                    # en yüksek kcal
                    victim = max(removable, key=lambda x: item_kcal(x))
                    items.remove(victim)
                    logs.append(f"[{meal_name}] -remove {victim.get('name','?')}")
                    did = True

                if not did:
                    break

        total_kcal = sum(item_kcal(it) for it in items)
        return items, total_kcal, logs

    # --- Eski enerji motoru
    return apply_energy_scaling_rules(
        items=items,
        hedef_kcal=hedef_kcal,
        meal_name=meal_name,
        pool=pool,
        pattern_rules=pattern_rules,
        state=state,
        day_index=day_index,
        avoid_set=avoid_set,
        energy_rules=energy_rules,
        max_iter=max_iter,
    )
import random
from typing import Optional

def pick_best(candidates: List[Dict[str, Any]],
              prefer_set: Set[str],
              avoid_set: Set[str],
              meal: str,
              category_whitelist: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -999

    for f in candidates:
        if not eligible_for_meal(f, meal):
            continue

        if category_whitelist and (f.get("kategori") not in category_whitelist):
            continue

        ok, _ = can_pick(f, avoid_set)
        if not ok:
            continue

        s = score_food(f, prefer_set)
        # kcal çok yüksekse hafif ceza (opsiyonel)
        kcal = float(f.get("kcal") or 0)
        s2 = s - (1 if kcal >= 450 else 0)

        if s2 > best_score:
            best_score = s2
            best = f

    return best

def build_breakfast(food_db: List[Dict[str, Any]],
                    prefer_set: Set[str],
                    avoid_set: Set[str],
                    target_kcal: int = 450) -> List[Dict[str, Any]]:
    """
    Parça parça kahvaltı: tahıl + protein/süt + sebze (+opsiyonel süt/yoğurt)
    """
    meal = "sabah"
    chosen: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()

    def add(food: Optional[Dict[str, Any]]):
        if not food:
            return
        fid = food.get("besin_id")
        if fid and fid in used_ids:
            return
        chosen.append(food)
        if fid:
            used_ids.add(fid)

    # 1) Tahıl seç (ekmek veya yulaf)
    tahil = pick_best(
        food_db, prefer_set, avoid_set, meal,
        category_whitelist={"Ekmek", "Tahıl", "Hamur İşi"}  # Hamur işi avoid_set’e takılır zaten
    )
    add(tahil)

    # 2) Protein / süt ürünü seç (yumurta veya peynir/yoğurt)
    protein = pick_best(
        food_db, prefer_set, avoid_set, meal,
        category_whitelist={"Protein", "Süt Ürünü", "Et Ürünü"}  # Et ürünü genelde islenmis_et -> avoid
    )
    add(protein)

    # 3) 1-2 sebze
    for _ in range(2):
        sebze = pick_best(
            food_db, prefer_set, avoid_set, meal,
            category_whitelist={"Sebze", "Sebze / Yeşillik"}
        )
        add(sebze)

    # 4) Kalori çok düşük kaldıysa: ikinci bir tahıl veya süt ürünü ekle
    total = sum(int(x.get("kcal") or 0) for x in chosen)
    if total < target_kcal - 120:
        extra = pick_best(
            food_db, prefer_set, avoid_set, meal,
            category_whitelist={"Tahıl", "Süt Ürünü"}
        )
        add(extra)

    return chosen


def build_snack(
        hedef: float,
        foods: List[Dict[str, Any]],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        meal: str,
        energy_rules: Optional[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    notes: List[str] = []
    items: List[Dict[str, Any]] = []
    state.new_meal(meal)

    pool = [f for f in foods if food_in_meal(f, meal)]

    def is_combo(f: Dict[str, Any]) -> bool:
        tags = set(get_food_tags(f))
        cat = (f.get("kategori") or "").lower().strip()
        return (cat == "kombin") or ("kombin" in tags) or ("kombinasyon" in tags) or ("combo" in tags)

    meal_key = (meal or "").lower()
    is_snack_meal = ("ara" in meal_key)

    def safe_mark_used(chosen_item: Dict[str, Any]) -> None:
        # add_item içinde zaten eklemen en doğrusu; burada da garantiye alıyoruz
        try:
            day.mark_used(chosen_item, is_snack_meal)
        except Exception:
            pass

    # 1) Yatarken: sadece süt/süt ürünü (tek kalem) - kilit
    if meal == "Yatarken" and "yatarken_ara_ogun" in pattern_rules:
        dairy_pool = [f for f in pool if ("sut" in get_food_tags(f)) or is_dairy(f)]
        base = pick_best(dairy_pool, avoid_tags, day, pattern_rules, state, meal, items)
        if base and add_item(items, base, day, pattern_rules, state, meal, avoid_tags):
            safe_mark_used(base)
        kcal = int(round(sum(item_kcal(x) for x in items)))
        return items, kcal, notes

    # 2) Ara öğün: TEK SEÇİM (kombin + normal item aynı havuzda yarışır)
    if is_snack_meal:
        # Kombin adayları (tek kalem)
        snack_combo_pool = [
            f for f in pool
            if is_combo(f) and (
                    "ara_ogun" in get_food_tags(f)
                    or "meyve" in get_food_tags(f)
                    or "kuruyemis" in get_food_tags(f)
                    or "sut_urunu" in get_food_tags(f)
                    or "sut" in get_food_tags(f)
            )
        ]

        # Normal tek kalem adaylar
        fruit_pool = [f for f in pool if "meyve" in get_food_tags(f)]
        nut_pool = [f for f in pool if "kuruyemis" in get_food_tags(f)]
        dairy_pool = [f for f in pool if
                      is_dairy(f) or ("sut" in get_food_tags(f)) or ("sut_urunu" in get_food_tags(f))]

        # Hepsini tek listeye koy: kombin zorunlu değil, skorla seçilecek
        candidates: List[Dict[str, Any]] = []
        candidates.extend(snack_combo_pool)
        candidates.extend(fruit_pool)
        candidates.extend(dairy_pool)
        candidates.extend(nut_pool)

        # Eğer hiçbir şey çıkmazsa fallback
        if not candidates:
            candidates = pool

        chosen = pick_best(candidates, avoid_tags, day, pattern_rules, state, meal, items)
        if chosen and add_item(items, chosen, day, pattern_rules, state, meal, avoid_tags):
            safe_mark_used(chosen)
        # (GÜN SONU ÖLÇEKLEME AKTİF) Öğün içi kalori ayarı kapalı.
        kcal = int(round(sum(item_kcal(x) for x in items)))

        # Diyabette meyve porsiyonunu kilitle
        if "meyve_siniri" in pattern_rules:
            for it in items:
                if "meyve" in get_food_tags(it):
                    it["_portion_mult"] = 1.0
                    it["_portion_lock"] = True
            kcal = int(round(sum(item_kcal(x) for x in items)))

        return items, kcal, notes[:6]

    # 3) Snack değilse: eski davranış (tek kalem + porsiyon ayarı)
    base = pick_best(pool, avoid_tags, day, pattern_rules, state, meal, items)
    if base and add_item(items, base, day, pattern_rules, state, meal, avoid_tags):
        safe_mark_used(base)
    # (GÜN SONU ÖLÇEKLEME AKTİF) Öğün içi kalori ayarı kapalı.
    kcal = int(round(sum(item_kcal(x) for x in items)))
    return items, kcal, notes[:6]


def meal_key(meal: str) -> str:
    # her yerde aynı anahtar
    return (meal or "").strip().lower()


def meal_has_tag(items, tag: str) -> bool:
    t = tag.strip().lower()
    return any(t in [x.lower() for x in get_food_tags(i)] for i in items)


def meal_has_any_tag(items, tags) -> bool:
    tags_l = [t.strip().lower() for t in tags]
    for i in items:
        itags = [x.lower() for x in get_food_tags(i)]
        if any(t in itags for t in tags_l):
            return True
    return False


def maybe_add_wholegrain_bread(
        meal: str,
        items: list,
        pool: list,
        avoid_tags: set,
        day,
        pattern_rules,
        state,
        *,
        require_veg_like: bool = True,
        block_if_soup_and_legume: bool = True
):
    # Öğün anahtarını normalize et
    mk = meal_key(meal)

    # 1) İsteğe bağlı: sadece sebze/haşlama/baklagil varsa ekmek ekle
    if require_veg_like:
        veg_like = meal_has_any_tag(items, ["sebze_yemegi", "haslama", "baklagil", "baklagil_yemegi"])
        if not veg_like:
            return

    # 2) Zaten ekmek varsa tekrar ekleme
    if meal_has_any_tag(items, ["ekmek", "tam_tahilli_ekmek"]):
        return

    # 3) İsteğe bağlı: çorba + baklagil varsa ekmek ekleme
    if block_if_soup_and_legume:
        if meal_has_any_tag(items, ["corba", "çorba"]) and meal_has_any_tag(items, ["baklagil", "baklagil_yemegi"]):
            return

    # 4) aday havuz: tam tahıllı ekmek
    bread_pool = [
        f for f in pool
        if meal_key(meal) in [meal_key(x) for x in (f.get("ogun") or [])]  # güvenlik
           and any(t in [x.lower() for x in get_food_tags(f)] for t in ["ekmek", "tam_tahilli_ekmek"])
           and not any(t in [x.lower() for x in get_food_tags(f)] for t in ["beyaz_ekmek", "rafine_tahil", "kombin"])
    ]

    bread = pick_best(bread_pool, avoid_tags, day, pattern_rules, state, meal, items)
    if bread:
        add_item(items, bread, day, pattern_rules, state, meal, avoid_tags)


def build_main_meal_template(
        *,
        hedef: float,
        foods: List[Dict[str, Any]],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        meal: str,
        force_fish_today: bool = False,
        enable_bread: bool = True
) -> Tuple[List[Dict[str, Any]], int, List[str], bool]:
    notes: List[str] = []
    items: List[Dict[str, Any]] = []
    did_fish = False

    # ✅ Tek tip meal ismi kullan
    meal = meal_key(meal)  # "öğle" / "akşam"
    state.new_meal(meal)

    pool = [f for f in foods if food_in_meal(f, meal)]

    # 1) Balık zorunluluğu (isteğe bağlı)
    if force_fish_today:
        fish_pool = [f for f in pool if "balik" in get_food_tags(f) or "balık" in get_food_tags(f)]
        fish = pick_best(fish_pool, avoid_tags, day, pattern_rules, state, meal, items)
        if fish and add_item(items, fish, day, pattern_rules, state, meal, avoid_tags):
            did_fish = True
            notes.append("Balık zorunluluğu uygulandı.")

    # 2) Ana yemek (balık yoksa)
    if not did_fish:
        main_candidates = [
            f for f in pool
            if "kombin" not in get_food_tags(f)
               and any(t in get_food_tags(f) for t in [
                "balik", "balık",
                "beyaz_et",
                "kirmizi_et",
                "baklagil",
                "baklagil_yemegi",
                "sebze_yemegi"
            ])
        ]

        main = pick_best(main_candidates, avoid_tags, day, pattern_rules, state, meal, items)

        # Baklagil gaz hassasiyeti fallback
        if not main:
            AVOID_SOFT = set(avoid_tags) - {"baklagil_gaz_yapici"}
            legume_pool = [f for f in pool if any(t in get_food_tags(f) for t in ["baklagil", "baklagil_yemegi"])]
            main = pick_best(legume_pool, AVOID_SOFT, day, pattern_rules, state, meal, items)
            if main:
                notes.append("Not: Baklagil gaz etiketi yumuşatılarak ana yemek seçildi.")

        if main:
            add_item(items, main, day, pattern_rules, state, meal, avoid_tags)
        else:
            notes.append("UYARI: Ana yemek seçilemedi.")

    # 3) Çorba (varsa ekle)
    if not meal_has_any_tag(items, ["corba", "çorba"]):
        soup_pool = [f for f in pool if any(t in get_food_tags(f) for t in ["corba", "çorba"])]
        soup = pick_best(soup_pool, avoid_tags, day, pattern_rules, state, meal, items)
        if soup:
            add_item(items, soup, day, pattern_rules, state, meal, avoid_tags)

    # 4) Yan ürün (ÇAKIŞMA ÇÖZÜMÜ)
    # - Varsayılan: tek yan (salata OR cacık)
    # - Sebze/baklagil ağırlığında: yoğurt (yoksa cacık) tercih edilir
    # - Balıkta: sadece salata
    # - Özel kural (pattern engine) zaten yan eklediyse ikinci yan eklenmez
    # - Yoğurt + salata birlikte varsa: cacık ASLA eklenmez
    if not meal_has_boiled_veg(items):
        force_salad = bool(meal_has_fish(items))
        side = pick_side_for_meal(pool, foods, avoid_tags, day, pattern_rules, state, meal, items,
                                  force_salad=force_salad)
        if side:
            add_item(items, side, day, pattern_rules, state, meal, avoid_tags)

    # 5) (KALDIRILDI) ayrı ayrı "salata" ve "süt ürünü" ekleme burada yapılmıyor
    # çünkü kullanıcı isteği: varsayılan durumda tek yan ürün gelsin.
    # (Haşlama sebze paketi enforce_boiled_veg_bundle içinde yönetiliyor.)

    # 6) Haşlama sebze paket kuralı (senin fonksiyonun)
    enforce_boiled_veg_bundle(meal, items, pool, foods, avoid_tags, day, pattern_rules, state, notes)

    # 7) Ekmek kuralı (en sonda)
    if enable_bread:
        maybe_add_wholegrain_bread(
            meal, items, pool, avoid_tags, day, pattern_rules, state,
            require_veg_like=True,
            block_if_soup_and_legume=True
        )

    kcal = int(round(sum(item_kcal(x) for x in items)))
    return items, kcal, notes[:6], did_fish


def build_lunch(
        hedef: float,
        foods: List[Dict[str, Any]],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        energy_rules: Optional[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    items, kcal, notes, _ = build_main_meal_template(
        hedef=hedef,
        foods=foods,
        avoid_tags=avoid_tags,
        day=day,
        pattern_rules=pattern_rules,
        state=state,
        meal="öğle",
        force_fish_today=False,
        enable_bread=True
    )
    return items, kcal, notes


def build_dinner(
        hedef: float,
        foods: List[Dict[str, Any]],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        force_fish_today: bool,
        energy_rules: Optional[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, List[str], bool]:
    return build_main_meal_template(
        hedef=hedef,
        foods=foods,
        avoid_tags=avoid_tags,
        day=day,
        pattern_rules=pattern_rules,
        state=state,
        meal="akşam",
        force_fish_today=force_fish_today,
        enable_bread=True
    )


# =========================
# Haftalık plan
# =========================


# =========================
# Gün sonu toplam kalori ölçekleme (TÜM GÜN)
# - Öğün bazlı art/az yapmaz
# - Tüm gün tek seferde ortak bir oranla _portion_mult uygular
# - _portion_lock=True olan item'lere dokunmaz (örn: diyabet meyve kilidi)
# =========================
def scale_whole_day_to_target(
        gun_menu: Dict[str, Any],
        gunluk_hedef: float,
        *,
        tolerance_kcal: float = 50.0,
        min_ratio: float = 0.75,
        max_ratio: float = 1.25,
) -> List[str]:
    notes: List[str] = []
    meal_order = ["Sabah", "Ara Öğün 1", "Öğle", "Ara Öğün 2", "Akşam", "Yatarken"]

    debug = (os.getenv("DEBUG_REGEX", "0") == "1") or (os.getenv("DEBUG_ENERGY", "0") == "1")

    def dlog(*args) -> None:
        if debug:
            print("[DAY_SCALE]", *args)

    def day_total() -> float:
        return sum(float((gun_menu.get(m) or {}).get("kcal", 0) or 0) for m in meal_order)

    mevcut = day_total()
    if mevcut <= 0:
        return ["Gün toplamı 0 olduğu için gün sonu ölçekleme yapılmadı."]

    fark = float(gunluk_hedef) - float(mevcut)
    if abs(fark) <= float(tolerance_kcal):
        return [
            f"Gün sonu ölçekleme gerekmedi (toplam={int(round(mevcut))} kcal, hedef={int(round(gunluk_hedef))} kcal, fark={int(round(fark))} kcal)."]

    oran_raw = float(gunluk_hedef) / float(mevcut)
    oran = max(min_ratio, min(max_ratio, oran_raw))

    dlog(
        f"START toplam={mevcut:.0f} hedef={float(gunluk_hedef):.0f} fark={fark:+.0f} oran_raw={oran_raw:.3f} oran={oran:.3f}")

    for m in meal_order:
        det = gun_menu.get(m) or {}
        items = det.get("items") or []
        if not isinstance(items, list) or not items:
            continue

        # item bazında çarpan uygula
        for it in items:
            try:
                if it.get("_portion_lock") is True:
                    continue
                base_mult = float(it.get("_portion_mult", 1.0) or 1.0)
                it["_portion_mult"] = round(base_mult * oran, 4)
                # (opsiyonel) kcal_scaled alanı
                try:
                    base_kcal = float(it.get("kcal", 0) or 0)
                    it["kcal_scaled"] = int(round(base_kcal * float(it["_portion_mult"])))
                except Exception:
                    pass
            except Exception:
                continue

        det["items"] = items
        det["kcal"] = int(round(sum(item_kcal(x) for x in items)))
        gun_menu[m] = det

    yeni = day_total()
    notes.append(
        f"Gün sonu ölçekleme uygulandı (oran={oran:.3f}). Toplam: {int(round(mevcut))} → {int(round(yeni))} kcal (hedef={int(round(gunluk_hedef))} kcal)."
    )
    dlog(f"END toplam={yeni:.0f}")
    return notes


def day_end_adjust_by_energy_rules(
        gun_menu: Dict[str, Any],
        gunluk_hedef: int,
        foods: List[Dict[str, Any]],
        avoid_tags: Set[str],
        day: DayState,
        pattern_rules: Dict[str, Any],
        state: PatternState,
        energy_rules: Dict[str, Any],
        *,
        debug: bool = True
) -> List[str]:
    notes: List[str] = []
    tol = int((energy_rules.get("delta_router", {}) or {}).get("tolerance_kcal", 25))

    def dlog(*a):
        if debug:
            print(*a)

    meal_order = ["Sabah", "Ara Öğün 1", "Öğle", "Ara Öğün 2", "Akşam", "Yatarken"]
    meal_key_map = {
        "Sabah": "sabah",
        "Ara Öğün 1": "ara_1",
        "Öğle": "öğle",
        "Ara Öğün 2": "ara_2",
        "Akşam": "akşam",
        "Yatarken": "yatarken",
    }

    def day_total() -> int:
        return int(round(sum(int((gun_menu.get(m) or {}).get("kcal", 0) or 0) for m in meal_order)))

    def is_meal_locked(meal_display: str) -> bool:
        mk = meal_key_map.get(meal_display, meal_display.lower())

        # lock sabah/yatarken
        if mk in ("sabah", "yatarken"):
            return True

        # lock_meal_if_contains kombİN
        det = gun_menu.get(meal_display) or {}
        items = det.get("items") or []
        for it in items:
            if "kombin" in set(get_food_tags(it)):
                return True
        return False

    def pick_candidates_for_add(meal_display: str, must_have_all: List[str]) -> List[Dict[str, Any]]:
        pool = [f for f in foods if food_in_meal(f, meal_display)]
        out = []
        for f in pool:
            tags = set(get_food_tags(f))
            ok = True
            for t in must_have_all:
                if t not in tags:
                    ok = False
                    break
            if ok:
                out.append(f)
        return out

    def recalc_meal_kcal(meal_display: str) -> None:
        det = gun_menu.get(meal_display) or {}
        items = det.get("items") or []
        det["kcal"] = int(round(sum(item_kcal(x) for x in items)))
        gun_menu[meal_display] = det

    # ---------------------------------------------------------
    # ✅ Dinamik yön/band/step seçimi (delta'ya göre)
    # ---------------------------------------------------------
    def choose_context(delta_val: int) -> Dict[str, Any]:
        direction_key = "scale_up" if delta_val > 0 else "scale_down"
        scale_cfg = energy_rules.get(direction_key, {}) or {}
        band_list = (scale_cfg.get("band_strategy") or [])
        step_templates = {s.get("step_id"): s for s in (scale_cfg.get("step_templates") or [])}

        # band seçimi (router)
        abs_delta = abs(int(delta_val))
        router = energy_rules.get("delta_router", {}) or {}
        bands = router.get("bands", []) or []

        chosen_band_id = None
        max_steps_total = 3

        def band_match(band_cfg: Dict[str, Any], value: int) -> bool:
            if "min_inclusive" in band_cfg and band_cfg["min_inclusive"] is not None:
                if value < int(band_cfg["min_inclusive"]):
                    return False
            if "min_exclusive" in band_cfg and band_cfg["min_exclusive"] is not None:
                if value <= int(band_cfg["min_exclusive"]):
                    return False

            if "max_inclusive" in band_cfg and band_cfg["max_inclusive"] is not None:
                if value > int(band_cfg["max_inclusive"]):
                    return False
            if "max_exclusive" in band_cfg and band_cfg["max_exclusive"] is not None:
                if value >= int(band_cfg["max_exclusive"]):
                    return False

            return True

        for b in bands:
            if band_match(b, abs_delta):
                chosen_band_id = b.get("band_id")
                max_steps_total = int(b.get("max_steps_total", max_steps_total))
                break

        if not chosen_band_id:
            chosen_band_id = "B6"

        band_cfg = next((x for x in band_list if x.get("band_id") == chosen_band_id), None) or {}
        steps_in_order = band_cfg.get("steps_in_order", []) or []
        loop_steps = bool(band_cfg.get("loop_steps", True))

        return {
            "direction_key": direction_key,
            "scale_cfg": scale_cfg,
            "chosen_band_id": chosen_band_id,
            "max_steps_total": max_steps_total,
            "steps_in_order": steps_in_order,
            "loop_steps": loop_steps,
            "step_templates": step_templates,
        }

    # START
    total0 = day_total()
    delta0 = int(gunluk_hedef - total0)
    dlog(f"[DAY_RULES] START toplam={total0} hedef={gunluk_hedef} delta={delta0:+d}")

    if abs(delta0) <= tol:
        notes.append(f"Gün sonu: delta tolerans içinde ({delta0:+d} kcal).")
        return notes

    # Global limit: yön değişse bile sonsuz döngü olmasın
    global_max_steps = 10

    # İlk context
    ctx = choose_context(delta0)
    direction_key = ctx["direction_key"]
    notes.append(f"Gün sonu band: {ctx['chosen_band_id']} (abs_delta={abs(delta0)}), mode={direction_key}.")

    steps_in_order = ctx["steps_in_order"]
    step_templates = ctx["step_templates"]
    loop_steps = ctx["loop_steps"]
    max_steps_total = ctx["max_steps_total"]

    if not steps_in_order:
        notes.append(
            f"Gün sonu: band '{ctx['chosen_band_id']}' için steps_in_order boş. "
            f"scale_cfg.band_strategy içinde bu band yok veya yanlış band seçildi. Plan rebalance gerekli."
        )
        total_end = day_total()
        delta_end = int(gunluk_hedef - total_end)
        dlog(f"[DAY_RULES] END toplam={total_end} hedef={gunluk_hedef} delta={delta_end:+d} steps_done=0/{max_steps_total}")
        notes.append(
            f"Gün sonu sonuç: toplam={total_end} kcal, hedef={gunluk_hedef} kcal, delta={delta_end:+d} kcal, adım=0/{max_steps_total}"
        )
        return notes

    # uygulama döngüsü
    steps_done = 0
    idx = 0

    no_progress_cycles = 0
    progress_in_cycle = False
    steps_seen_in_cycle = 0
    cycle_len = len(steps_in_order)

    last_direction_key = direction_key

    while steps_done < min(global_max_steps, max_steps_total):
        total = day_total()
        delta = int(gunluk_hedef - total)

        if abs(delta) <= tol:
            break

        # ✅ Her adımda yönü yeniden belirle (artı/eksiye göre)
        direction_key_now = "scale_up" if delta > 0 else "scale_down"

        # ✅ yön değiştiyse context’i yeniden seç
        if direction_key_now != last_direction_key:
            ctx = choose_context(delta)
            direction_key_now = ctx["direction_key"]
            last_direction_key = direction_key_now

            notes.append(f"Gün sonu mode değişti: {direction_key_now}, band={ctx['chosen_band_id']} (abs_delta={abs(delta)}).")

            steps_in_order = ctx["steps_in_order"]
            step_templates = ctx["step_templates"]
            loop_steps = ctx["loop_steps"]
            max_steps_total = ctx["max_steps_total"]

            if not steps_in_order:
                notes.append(
                    f"Gün sonu: band '{ctx['chosen_band_id']}' için steps_in_order boş. "
                    f"Plan rebalance gerekli."
                )
                break

            # cycle state reset
            idx = 0
            no_progress_cycles = 0
            progress_in_cycle = False
            steps_seen_in_cycle = 0
            cycle_len = len(steps_in_order)

        # idx listeyi aştıysa: loop ise yeni cycle, değilse çık
        if idx >= cycle_len:
            if loop_steps:
                idx = 0
                if not progress_in_cycle:
                    no_progress_cycles += 1
                else:
                    no_progress_cycles = 0

                progress_in_cycle = False
                steps_seen_in_cycle = 0

                if no_progress_cycles >= 1:
                    notes.append("Gün sonu: uygun adım bulunamadı (eligible item yok veya öğünler kilitli). Plan rebalance gerekli.")
                    break
            else:
                break

        step_id = steps_in_order[idx]
        idx += 1
        steps_seen_in_cycle += 1

        tmpl = step_templates.get(step_id)
        if not tmpl:
            continue

        applies_to = tmpl.get("applies_to_meals", []) or []
        action = tmpl.get("action")

        # ✅ delta işaretine göre sadece uygun aksiyonlara izin ver
        if delta > 0:
            # artırma modu: increase_portion + add
            if action not in ("increase_portion", "add"):
                continue
        else:
            # azaltma modu: sadece decrease_portion
            if action not in ("decrease_portion",):
                continue

        candidate_meals = []
        for m in ["Öğle", "Akşam", "Ara Öğün 1", "Ara Öğün 2", "Sabah", "Yatarken"]:
            mk = meal_key_map.get(m, m.lower())
            if mk in applies_to:
                candidate_meals.append(m)

        did_any = False

        for meal_display in candidate_meals:
            if is_meal_locked(meal_display):
                continue

            det = gun_menu.get(meal_display) or {}
            items = det.get("items") or []

            if action in ("increase_portion", "decrease_portion"):
                rule_ref = tmpl.get("uses_rule_ref")
                if not rule_ref:
                    continue

                ok = apply_portion_rule_ref(
                    items=items,
                    energy_rules=energy_rules,
                    rule_ref=rule_ref,
                    direction=("up" if action == "increase_portion" else "down"),
                    dlog=(lambda *a: dlog("[DAY_RULES]", *a)),
                )
                if ok:
                    recalc_meal_kcal(meal_display)
                    det = gun_menu.get(meal_display) or {}
                    det.setdefault("kural_notlari", [])
                    det["kural_notlari"].append(f"{direction_key_now}:{step_id} ({meal_display})")
                    gun_menu[meal_display] = det
                    did_any = True
                    break

            elif action == "add":
                cand_filter = tmpl.get("candidate_filter", {}) or {}
                must_all = cand_filter.get("must_have_food_tags_all", []) or []
                candidates = pick_candidates_for_add(meal_display, must_all)
                chosen = pick_best(candidates, avoid_tags, day, pattern_rules, state, meal_display, items)
                if chosen and add_item(items, chosen, day, pattern_rules, state, meal_display, avoid_tags):
                    recalc_meal_kcal(meal_display)
                    det = gun_menu.get(meal_display) or {}
                    det.setdefault("kural_notlari", [])
                    det["kural_notlari"].append(f"{direction_key_now}:{step_id} ({meal_display})")
                    gun_menu[meal_display] = det
                    did_any = True
                    break

        if did_any:
            steps_done += 1
            progress_in_cycle = True

    total_end = day_total()
    delta_end = int(gunluk_hedef - total_end)
    dlog(f"[DAY_RULES] END toplam={total_end} hedef={gunluk_hedef} delta={delta_end:+d} steps_done={steps_done}/{min(global_max_steps, max_steps_total)}")
    notes.append(f"Gün sonu sonuç: toplam={total_end} kcal, hedef={gunluk_hedef} kcal, delta={delta_end:+d} kcal, adım={steps_done}/{min(global_max_steps, max_steps_total)}")
    return notes

def build_week_plan(
        gunluk_hedef: int,
        foods: List[Dict[str, Any]],
        avoid_tags: Set[str],
        pool_meta: Dict[str, Any]
) -> Dict[str, Any]:
    gunler = ["PAZARTESİ", "SALI", "ÇARŞAMBA", "PERŞEMBE", "CUMA", "CUMARTESİ", "PAZAR"]

    # Öğünlerin gün içindeki ağırlığı (taslak oluştururken referans)
    dagilim = {
        "Sabah": 0.25,
        "Ara Öğün 1": 0.10,
        "Öğle": 0.25,
        "Ara Öğün 2": 0.10,
        "Akşam": 0.20,
        "Yatarken": 0.10
    }

    # ✅ MUTLAKA burada tanımlı olmalı
    pattern_rules = pool_meta.get("meal_pattern_rules_active", {}) or {}
    energy_rules = pool_meta.get("energy_scaling_rules_active", {}) or {}

    weekly_targets = get_weekly_targets(pattern_rules)
    soft_targets = get_soft_weekly_targets(pattern_rules)

    day = DayState()
    state = PatternState()

    week: Dict[str, Any] = {}
    fish_min = weekly_targets.get("balik", {}).get("min", 0)
    fish_done_for_report = 0

    for day_idx, gun in enumerate(gunler):
        if RANDOM_SEED is not None:
            random.seed((RANDOM_SEED + day_idx) * 101)

        day.new_day()
        state.new_day()

        days_left = 7 - day_idx
        need_fish = max(0, fish_min - state.get_weekly("balik"))
        force_fish_today = (need_fish > 0) and (days_left <= need_fish + 1 or random.random() < 0.35)

        # ---------------------------------------------------------
        # ADIM 1: TASLAK ÖĞÜNLERİ OLUŞTUR (ÖĞÜN İÇİ SCALING KAPALI)
        # ---------------------------------------------------------
        gun_menu: Dict[str, Any] = {}

        # Taslak Sabah
        h_sabah = gunluk_hedef * dagilim["Sabah"]
        items_s, kcal_s, notes_s = build_breakfast(
            h_sabah, foods, avoid_tags, day, pattern_rules, state,
            soft_targets, days_left, energy_rules=None
        )
        gun_menu["Sabah"] = {"items": items_s, "kcal": kcal_s, "hedef": int(round(h_sabah)), "kural_notlari": notes_s}

        # Taslak Ara 1
        h_ara1 = gunluk_hedef * dagilim["Ara Öğün 1"]
        items_a1, kcal_a1, notes_a1 = build_snack(
            h_ara1, foods, avoid_tags, day, pattern_rules, state,
            "Ara Öğün 1", energy_rules=None
        )
        gun_menu["Ara Öğün 1"] = {"items": items_a1, "kcal": kcal_a1, "hedef": int(round(h_ara1)),
                                  "kural_notlari": notes_a1}

        # Taslak Öğle
        h_ogle = gunluk_hedef * dagilim["Öğle"]
        items_o, kcal_o, notes_o = build_lunch(
            h_ogle, foods, avoid_tags, day, pattern_rules, state, energy_rules=None
        )
        gun_menu["Öğle"] = {"items": items_o, "kcal": kcal_o, "hedef": int(round(h_ogle)), "kural_notlari": notes_o}

        # Taslak Ara 2
        h_ara2 = gunluk_hedef * dagilim["Ara Öğün 2"]
        items_a2, kcal_a2, notes_a2 = build_snack(
            h_ara2, foods, avoid_tags, day, pattern_rules, state,
            "Ara Öğün 2", energy_rules=None
        )
        gun_menu["Ara Öğün 2"] = {"items": items_a2, "kcal": kcal_a2, "hedef": int(round(h_ara2)),
                                  "kural_notlari": notes_a2}

        # Taslak Akşam
        h_aksam = gunluk_hedef * dagilim["Akşam"]
        items_ak, kcal_ak, notes_ak, did_fish = build_dinner(
            h_aksam, foods, avoid_tags, day, pattern_rules, state,
            force_fish_today, energy_rules=None
        )
        gun_menu["Akşam"] = {"items": items_ak, "kcal": kcal_ak, "hedef": int(round(h_aksam)),
                             "kural_notlari": notes_ak}
        if did_fish:
            fish_done_for_report += 1

        # Taslak Yatarken
        h_yatar = gunluk_hedef * dagilim["Yatarken"]
        items_y, kcal_y, notes_y = build_snack(
            h_yatar, foods, avoid_tags, day, pattern_rules, state,
            "Yatarken", energy_rules=None
        )
        gun_menu["Yatarken"] = {"items": items_y, "kcal": kcal_y, "hedef": int(round(h_yatar)),
                                "kural_notlari": notes_y}

        # ---------------------------------------------------------
        # ADIM 2: GÜN SONU KURAL MOTORU İLE AŞAMALI KALORİ AYARI
        # ---------------------------------------------------------
        gun_menu.setdefault("_gun_sonu_notlari", [])
        gun_menu["_gun_sonu_notlari"].extend(
            day_end_adjust_by_energy_rules(
                gun_menu=gun_menu,
                gunluk_hedef=gunluk_hedef,
                foods=foods,
                avoid_tags=avoid_tags,
                day=day,
                pattern_rules=pattern_rules,
                state=state,
                energy_rules=energy_rules,
                debug=True
            )
        )

        # ✅ BUNLAR EKSİKTİ: gün menüsünü week’e koy
        week[gun] = gun_menu

    # ✅ haftalık özet
    week["_weekly_summary"] = {"fish_done": fish_done_for_report, "fish_target_min": fish_min}
    return week


# =========================
# PDF item line (porsiyon çarpanı + 6-8 kaşık gibi aralık çarpma)
# =========================
# -*- coding: utf-8 -*-
# ============================================================
#  PDF TASARIMI (Modern / Lacivert + Kırmızı vurgulu)
#  - Strateji başlığı hastalığa göre otomatik
#  - Maddeler düzgün bullet list (ListFlowable)
#  - Öğün içi maddeler de düzgün bullet list
#  - Mevcut item_line korunur (porsiyon çarpanı + 6-8 aralık)
# ============================================================

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import traceback

from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
    PageBreak,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os

from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
    PageBreak,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================
# 0) GÜVENLİ METİN (TEK KAYNAK)
# =========================
def safe_text(x: Any) -> str:
    """Düz metin temizliği (Paragraph'a ham vermeden önce)."""
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        return ""
    s = s.replace("\u00a0", " ")
    s = " ".join(s.split())
    return s.strip()


def safe_para(x: Any) -> str:
    """ReportLab Paragraph için: mini-XML parser kırılmasın diye escape."""
    return xml_escape(safe_text(x))


def maybe_kv(label: str, value: Any) -> Optional[str]:
    v = safe_text(value)
    if not v:
        return None
    return f"{safe_para(label)}: {safe_para(v)}"


# =========================
# 1) TR FONT REGISTER (PROJE YOKSA WINDOWS'TAN BUL)
# =========================
def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def register_tr_fonts_or_die(
    base_dir: str,
    regular_rel: str = "assets/fonts/DejaVuSans.ttf",
    bold_rel: str = "assets/fonts/DejaVuSans-Bold.ttf",
    regular_name: str = "TRFont",
    bold_name: str = "TRFont-Bold",
) -> Tuple[str, str]:
    """
    1) Önce projede verilen relative fontları dener.
    2) Bulamazsa Windows Fonts klasöründen (arial/calibri/tahoma) seçer.
    """
    # --- Proje içi dene ---
    regular_path = os.path.join(base_dir, regular_rel) if regular_rel else ""
    bold_path = os.path.join(base_dir, bold_rel) if bold_rel else ""

    has_regular = bool(regular_path and os.path.exists(regular_path))
    has_bold = bool(bold_path and os.path.exists(bold_path))

    # --- Windows fallback ---
    if not has_regular or not has_bold:
        win = os.environ.get("WINDIR", r"C:\Windows")
        fonts_dir = os.path.join(win, "Fonts")

        arial = os.path.join(fonts_dir, "arial.ttf")
        arial_b = os.path.join(fonts_dir, "arialbd.ttf")

        calibri = os.path.join(fonts_dir, "calibri.ttf")
        calibri_b = os.path.join(fonts_dir, "calibrib.ttf")

        tahoma = os.path.join(fonts_dir, "tahoma.ttf")
        tahoma_b = os.path.join(fonts_dir, "tahomabd.ttf")

        # Regular seç
        if not has_regular:
            regular_path = _first_existing(arial, calibri, tahoma)

        # Bold seç
        if not has_bold:
            bold_path = _first_existing(arial_b, calibri_b, tahoma_b)

        if not regular_path:
            raise FileNotFoundError(
                "TR font bulunamadı. Projede font yok ve Windows Fonts içinde arial/calibri/tahoma da bulunamadı."
            )

        # Bold bulunmazsa regular ile devam
        if not bold_path:
            bold_path = regular_path
            bold_name = regular_name

    # --- Register ---
    pdfmetrics.registerFont(TTFont(regular_name, regular_path))
    if bold_path == regular_path:
        bold_name = regular_name
    else:
        pdfmetrics.registerFont(TTFont(bold_name, bold_path))

    # family mapping (bazı durumlarda işe yarar)
    try:
        pdfmetrics.registerFontFamily(
            regular_name,
            normal=regular_name,
            bold=bold_name,
            italic=regular_name,
            boldItalic=bold_name,
        )
    except Exception:
        pass

    return regular_name, bold_name


# =========================
# 2) STYLES
# =========================
def build_styles_pro(font_reg: str, font_bold: str):
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "title",
        parent=base["Title"],
        fontName=font_bold,
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#0B2A4A"),
        alignment=1,
        spaceAfter=8,
    )

    subtitle = ParagraphStyle(
        "subtitle",
        parent=base["Normal"],
        fontName=font_reg,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        alignment=1,
        spaceAfter=10,
    )

    h1 = ParagraphStyle(
        "h1",
        parent=base["Heading1"],
        fontName=font_bold,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0B2A4A"),
        spaceBefore=6,
        spaceAfter=6,
    )

    meal_header = ParagraphStyle(
        "meal_header",
        parent=base["Normal"],
        fontName=font_bold,
        fontSize=10,
        leading=14,
        textColor=colors.white,
    )

    body = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontName=font_reg,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )

    small = ParagraphStyle(
        "small",
        parent=base["Normal"],
        fontName=font_reg,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4B5563"),
    )

    note_red = ParagraphStyle(
        "note_red",
        parent=base["Normal"],
        fontName=font_bold,
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#B91C1C"),
        spaceAfter=6,
    )

    bullet = ParagraphStyle(
        "bullet",
        parent=base["Normal"],
        fontName=font_reg,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )

    return title, subtitle, h1, meal_header, body, small, note_red, bullet


# =========================
# 3) HEADER / FOOTER
# =========================
def header_footer(canvas, doc, title: str, font_reg: str, font_bold: str):
    canvas.saveState()
    width, height = doc.pagesize

    canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
    canvas.setLineWidth(0.6)
    canvas.line(1.5 * cm, height - 1.55 * cm, width - 1.5 * cm, height - 1.55 * cm)

    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.setFont(font_bold, 9)
    canvas.drawString(1.5 * cm, height - 1.25 * cm, safe_text(title))

    canvas.setFont(font_reg, 8)
    canvas.drawRightString(width - 1.5 * cm, 1.15 * cm, f"Sayfa {doc.page}")

    canvas.restoreState()


# =========================
# 4) create_pdf (TR OK / WINDOWS FONT OK)
# =========================
def create_pdf(
    filename: str,
    user_profile: Dict[str, Any],
    hastaliklar: List[str],
    used_rule_files: List[str],
    rules: Dict[str, Any],
    week_plan: Dict[str, Any],
    pool_meta: Dict[str, Any],
    *,
    font_regular_path: str = "assets/fonts/DejaVuSans.ttf",
    font_bold_path: str = "assets/fonts/DejaVuSans-Bold.ttf",
) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # relative gönderiyoruz (projede yoksa Windows fallback yapacak)
    rel_reg = font_regular_path if not os.path.isabs(font_regular_path) else os.path.relpath(font_regular_path, base_dir)
    rel_bold = font_bold_path if not os.path.isabs(font_bold_path) else os.path.relpath(font_bold_path, base_dir)

    font_reg, font_bold = register_tr_fonts_or_die(
        base_dir=base_dir,
        regular_rel=rel_reg,
        bold_rel=rel_bold,
        regular_name="TRFont",
        bold_name="TRFont-Bold",
    )

    title_style, subtitle_style, h1_style, meal_h_style, body_style, small_style, note_red_style, bullet_style = \
        build_styles_pro(font_reg, font_bold)

    NAVY = colors.HexColor("#0B2A4A")
    GRID = colors.HexColor("#CBD5E1")
    CARD_BG = colors.HexColor("#F8FAFC")

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
    )

    story: List[Any] = []

    # ---- KAPAK ----
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph(safe_para("BESLENME VE DİYET PROGRAMI"), title_style))

    ad_soyad = (safe_text(user_profile.get("ad_soyad")) or "DANIŞAN").upper()
    profil_txt = (", ".join([safe_text(x) for x in (hastaliklar or []) if safe_text(x)]) or "GENEL SAĞLIK").upper()
    dt = datetime.now().strftime("%d.%m.%Y")

    story.append(Paragraph(safe_para(f"{ad_soyad} • {profil_txt} • {dt}"), subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=GRID, spaceAfter=12))

    left_col: List[str] = []
    right_col: List[str] = []

    for s in [
        maybe_kv("Danışan", ad_soyad),
        maybe_kv("Profil", profil_txt),
        maybe_kv("Kural", user_profile.get("acik_kural")),
    ]:
        if s:
            left_col.append(s)

    hedef = safe_text(user_profile.get("hedef_kalori"))
    kilo = safe_text(user_profile.get("kilo"))
    boy = safe_text(user_profile.get("boy"))

    for s in [
        maybe_kv("Hedef Kalori", f"{hedef} kcal" if hedef else ""),
        maybe_kv("Kilo", f"{kilo} kg" if kilo else ""),
        maybe_kv("Boy", f"{boy} cm" if boy else ""),
    ]:
        if s:
            right_col.append(s)

    rows = max(len(left_col), len(right_col), 1)
    while len(left_col) < rows:
        left_col.append("")
    while len(right_col) < rows:
        right_col.append("")

    info_data = [
        [Paragraph(left_col[i], body_style), Paragraph(right_col[i], body_style)]
        for i in range(rows)
    ]
    info_table = Table(info_data, colWidths=[8.7 * cm, 8.7 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.9, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.35 * cm))

    # ---- KURALLAR / STRATEJİ ----
    sec_title = "PROGRAM STRATEJİSİ • " + profil_txt if profil_txt else "PROGRAM STRATEJİSİ"

    active_tags = pool_meta.get("meal_pattern_tags_active", []) or []
    disease_rule_desc = pool_meta.get("meal_pattern_desc_disease_rule", {}) or {}
    pool_desc = pool_meta.get("meal_pattern_desc_pool", {}) or {}

    bullets: List[str] = []
    for tag in active_tags:
        item = disease_rule_desc.get(tag) or pool_desc.get(tag)
        desc = item.get("description") if isinstance(item, dict) else safe_text(item)
        if safe_text(desc):
            bullets.append(safe_text(desc))

    red_note = "Bu kurallar programın etkinliği için kritiktir. Medikal takip yerine geçmez."
    story.append(Paragraph(safe_para(sec_title), h1_style))
    story.append(Paragraph(safe_para(red_note), note_red_style))

    if bullets:
        lf = ListFlowable(
            [ListItem(Paragraph(safe_para(x), bullet_style), value="•") for x in bullets],
            bulletType="bullet",
            leftIndent=12,
            bulletOffsetY=2,
        )
        story.append(lf)

    used_rule_files_clean = [safe_text(x) for x in (used_rule_files or []) if safe_text(x)]
    if used_rule_files_clean:
        story.append(Spacer(1, 0.25 * cm))
        story.append(Paragraph(safe_para("Kullanılan Kural Dosyaları"), body_style))
        mini_list = ListFlowable(
            [ListItem(Paragraph(safe_para(x), small_style), value="•") for x in used_rule_files_clean[:20]],
            bulletType="bullet",
            leftIndent=12,
        )
        story.append(mini_list)

    story.append(PageBreak())

    # ---- HAFTALIK PLAN ----
    meal_order = ["Sabah", "Ara Öğün 1", "Öğle", "Ara Öğün 2", "Akşam", "Yatarken"]

    for gun, ogunler in (week_plan or {}).items():
        if str(gun).startswith("_"):
            continue
        if not isinstance(ogunler, dict):
            continue

        day_total = 0
        for m in meal_order:
            try:
                day_total += int(ogunler.get(m, {}).get("kcal", 0) or 0)
            except Exception:
                pass

        story.append(Paragraph(safe_para(str(gun).upper()), h1_style))
        story.append(Paragraph(safe_para(f"Gün Toplam: {day_total} kcal"), body_style))
        story.append(HRFlowable(width="100%", thickness=0.8, color=GRID, spaceAfter=10))

        for ogun_adi in meal_order:
            det = ogunler.get(ogun_adi)
            if not isinstance(det, dict) or not (det.get("items") or []):
                continue

            kcal_val = det.get("kcal", 0) or 0
            try:
                kcal_txt = f"{int(kcal_val)} kcal"
            except Exception:
                kcal_txt = safe_text(kcal_val)

            header_bar = Table(
                [[Paragraph(safe_para(f"{ogun_adi.upper()} • {kcal_txt}"), meal_h_style)]],
                colWidths=[17.4 * cm],
            )
            header_bar.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(header_bar)

            # ÖĞÜN SATIRLARI
            meal_lines: List[str] = []
            for it in (det.get("items") or []):
                name = safe_text(it.get("ad") or it.get("isim") or it.get("name") or it.get("besin_adi"))
                miktar = safe_text(it.get("miktar_str") or it.get("miktar"))
                kcal = it.get("kcal_scaled", None)
                if kcal is None:
                    kcal = it.get("kcal", None)

                line_parts = []
                if name:
                    line_parts.append(name)
                if miktar:
                    line_parts.append(f"— {miktar}")
                if kcal is not None and safe_text(kcal):
                    try:
                        line_parts.append(f"({int(float(kcal))} kcal)")
                    except Exception:
                        line_parts.append(f"({safe_text(kcal)} kcal)")

                line = " ".join(line_parts).strip()
                if line:
                    meal_lines.append(line)

            if meal_lines:
                meal_list = ListFlowable(
                    [ListItem(Paragraph(safe_para(x), bullet_style), value="•") for x in meal_lines],
                    bulletType="bullet",
                    leftIndent=12,
                    bulletOffsetY=2,
                )

                body_box = Table([[meal_list]], colWidths=[17.4 * cm])
                body_box.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]))
                story.append(body_box)
                story.append(Spacer(1, 0.35 * cm))

        story.append(PageBreak())

    doc.build(
        story,
        onFirstPage=lambda c, d: header_footer(c, d, "DİYET REHBERİ", font_reg, font_bold),
        onLaterPages=lambda c, d: header_footer(c, d, "DİYET REHBERİ", font_reg, font_bold),
    )

# =========================
# 5) main (EN SONA KOY)
# =========================
def main():
    try:
        # SENİN load_data / hesapla_gereksinim / build_week_plan fonksiyonların
        foods, rules, used_rule_files, hastaliklar, pool_meta = load_data()
        user_profile = hesapla_gereksinim()

        avoid_set = set(rules.get("avoid_tags", []) or [])
        week_plan = build_week_plan(user_profile["hedef_kalori"], foods, avoid_set, pool_meta)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "diyet_listeleri")
        os.makedirs(out_dir, exist_ok=True)

        ad_temiz = (safe_text(user_profile.get("ad_soyad")) or "danisan").strip().replace(" ", "_").lower()
        hastalik_metni = "_".join([safe_text(x) for x in (hastaliklar or []) if safe_text(x)]).lower().replace(" ", "_") or "genel"

        out_name = f"program_{ad_temiz}_{hastalik_metni}.pdf"
        full_path = os.path.join(out_dir, out_name)

        create_pdf(full_path, user_profile, hastaliklar, used_rule_files, rules, week_plan, pool_meta)

        print("\n" + "=" * 60)
        print("BAŞARILI: PDF başarıyla oluşturuldu.")
        print(f"DOSYA ADI: {out_name}")
        print(f"TAM YOL  : {full_path}")
        print("=" * 60)

    except Exception as e:
        print("\n" + "!" * 30)
        print(f"KRİTİK HATA: {e}")
        traceback.print_exc()
        print("!" * 30)


if __name__ == "__main__":
    main()

