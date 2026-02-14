# -*- coding: utf-8 -*-
"""
Diet AI - Weekly / Monthly Planner + LLM Audit Patch (QA Cross-Check) + (Opsiyonel) PDF
======================================================================================
TEK DOSYA - DÜZELTİLMİŞ SÜRÜM

ÖNE ÇIKANLAR:
✅ NLP labels.json ÜSTÜNE YAZMAZ (baz olarak kalır)
✅ LLM audit "PATCH" üretir -> sadece eksik numeric_constraints ve gerekirse küçük eklemeler
✅ Revize labels kaydı:
   C:\\Users\\user\\Desktop\\diyetisyen_llm\\llm_labels_revize\\<condition>\\<model_key>\\labels.json
✅ Planlama revize labels üzerinden devam eder
✅ Türkçe kullanıcı bilgisi alınır
✅ Terminale önce LLM açıklaması (qa_summary bazlı) sonra plan basılır
✅ PDF sorar; E ise üretir
✅ PDF: Türkçe font fix + her gün 1 sayfa
✅ PDF’de prefer/avoid/limit listesi basılmaz -> sadece LLM açıklaması + plan
✅ Weekly numeric constraints ENFORCE edilir (period_days=7, count-unit: kez/adet/porsiyon vb.)
✅ Öğün sırası: sabah -> ara_1 -> öğle -> ara_2 -> ara_3 -> akşam -> yatarken

EK DÜZELTME (Numeric constraint tag-fix):
✅ numeric_constraints.description metni tag_dicts.json ile eşleştirilir
✅ yanlış tag varsa düzeltilir
✅ duplicate numeric constraints tekilleştirilir
✅ "350-500 gram" gibi gramajlar yanlışlıkla min_count/max_count içine yazıldıysa gram alanlarına taşınır
✅ period_days/unit description’dan normalize edilir

EK:
✅ Her model için otomatik test raporu JSON kaydı:
   C:\\Users\\user\\Desktop\\diyetisyen_llm\\diet_test_report\\<condition>\\<model_key>\\report_<timestamp>.json
"""

import os
import re
import json
import random
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

# PDF (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================================================
# MODELS
# =========================================================
MODELS: Dict[str, Dict[str, Any]] = {
    "llama_3_1_8b_instruct": {"label": "Llama-3.1:8B Instruct", "ollama": "llama3.1:8b", "num_ctx": 8192},
    "gemma_3_4b": {"label": "Gemma-3:4B", "ollama": "gemma3:4b", "num_ctx": 8192},
    "qwen2_5_3b": {"label": "Qwen-2.5:3B", "ollama": "qwen2.5:3b", "num_ctx": 8192},
    "gemma_2_2b": {"label": "Gemma-2:2B", "ollama": "gemma2:2b", "num_ctx": 8192},

    "qwen2_5_7b_instruct": {"label": "Qwen-2.5:7B Instruct (extra)", "ollama": "qwen2.5:7b-instruct", "num_ctx": 8192},
    "mistral_7b_instruct": {"label": "Mistral:7B Instruct (extra)", "ollama": "mistral:7b-instruct", "num_ctx": 8192},

    "llama_3_2_3b_instruct": {"label": "Llama-3.2:3B Instruct (extra2)", "ollama": "llama3.2:3b-instruct", "num_ctx": 8192},
}

DEFAULT_MODEL_ORDER = ["gemma_2_2b", "qwen2_5_3b", "gemma_3_4b", "llama_3_1_8b_instruct"]
EXTRA_MODEL_ORDER = ["mistral_7b_instruct", "qwen2_5_7b_instruct", "llama_3_2_3b_instruct"]


# =========================================================
# DEFAULT PATHS
# =========================================================
DEFAULT_PROJECT_ROOT = r"C:\Users\user\Desktop\diyetisyen_llm"
DEFAULT_QA_DEMO_ROOT = os.path.join(DEFAULT_PROJECT_ROOT, "deneme_cache", "qa_demo")
DEFAULT_LABELS_ROOT = os.path.join(DEFAULT_PROJECT_ROOT, "labels")
DEFAULT_FOODS_PATH = os.path.join(DEFAULT_PROJECT_ROOT, "besin_havuzu_eski.normalized.json")
DEFAULT_TAG_DICTS_PATH = os.path.join(DEFAULT_PROJECT_ROOT, "tag_dicts.json")
DEFAULT_LLM_REVISED_ROOT = os.path.join(DEFAULT_PROJECT_ROOT, "llm_labels_revize")
DEFAULT_OUT_ROOT = os.path.join(DEFAULT_PROJECT_ROOT, "diet_out")
DEFAULT_TEST_REPORT_ROOT = os.path.join(DEFAULT_PROJECT_ROOT, "diet_test_report")
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_CONDITION = "kolesterol"
QA_SUMMARY_FIXED_FOLDER = "gemma_3_4b"


# =========================================================
# Helpers
# =========================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s).strip())

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_optional_text_or_json(path: str) -> Any:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        c = content.strip()
        if c.startswith("{") or c.startswith("["):
            try:
                return json.loads(c)
            except Exception:
                return content
        return content
    except Exception:
        return ""

def extract_json_obj(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty LLM output.")

    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    if s.startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            pass

    start = s.find("{")
    if start == -1:
        raise ValueError("JSON not found in LLM output.")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(s[start:i+1])

    raise ValueError("Unbalanced JSON braces in LLM output.")

def prompt_str(msg: str, default: str = "") -> str:
    s = input(f"{msg} [Varsayılan: {default}]: ").strip()
    return s if s else default

def prompt_int(msg: str, default: int, min_v: int = 1, max_v: int = 52) -> int:
    s = input(f"{msg} [Varsayılan: {default}]: ").strip()
    if not s:
        return default
    try:
        v = int(s)
        v = max(min_v, min(max_v, v))
        return v
    except:
        return default

def prompt_float(msg: str, default: float) -> float:
    s = input(f"{msg} [Varsayılan: {default}]: ").strip().replace(",", ".")
    if not s:
        return default
    try:
        return float(s)
    except:
        return default

def prompt_choice(msg: str, choices: List[str], default: str) -> str:
    s = input(f"{msg} {choices} [Varsayılan: {default}]: ").strip()
    if not s:
        return default
    s2 = s.lower()
    for c in choices:
        if c.lower() == s2:
            return c
    return default

def prompt_yes_no(msg: str, default_yes: bool = False) -> bool:
    default = "E" if default_yes else "H"
    s = input(f"{msg} (E/H) [Varsayılan: {default}]: ").strip().lower()
    if not s:
        return default_yes
    return s.startswith("e")


# =========================================================
# Ollama client
# =========================================================
class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout_sec: int = 240):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def chat(self, model: str, system: str, user: str, temperature: float = 0.0) -> str:
        chat_body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature},
            "stream": False,
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=chat_body, timeout=self.timeout_sec)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception:
            # fallback
            prompt = f"{system}\n\n{user}"
            gen_body = {"model": model, "prompt": prompt, "options": {"temperature": temperature}, "stream": False}
            r2 = requests.post(f"{self.base_url}/api/generate", json=gen_body, timeout=self.timeout_sec)
            r2.raise_for_status()
            return r2.json().get("response", "")


# =========================================================
# Path resolvers
# =========================================================
def list_conditions(labels_root: str) -> List[str]:
    if not labels_root or not os.path.isdir(labels_root):
        return []
    idx_path = os.path.join(labels_root, "_index.json")
    conditions = []
    if os.path.exists(idx_path):
        try:
            idx = load_json(idx_path)
            if isinstance(idx, dict):
                if "conditions" in idx and isinstance(idx["conditions"], list):
                    conditions = [str(x) for x in idx["conditions"]]
                elif "folders" in idx and isinstance(idx["folders"], list):
                    conditions = [str(x) for x in idx["folders"]]
                elif "domains" in idx and isinstance(idx["domains"], list):
                    conditions = [str(x) for x in idx["domains"]]
                else:
                    conditions = [str(k) for k in idx.keys() if not str(k).startswith("_")]
        except Exception:
            conditions = []
    if not conditions:
        for name in os.listdir(labels_root):
            full = os.path.join(labels_root, name)
            if os.path.isdir(full) and not name.startswith("_"):
                conditions.append(name)
    return sorted(list(set(conditions)))

def resolve_labels_path(labels_root: str, condition: str) -> Optional[str]:
    if not labels_root or not os.path.isdir(labels_root) or not condition:
        return None
    cond_low = condition.strip().lower()
    for name in os.listdir(labels_root):
        full = os.path.join(labels_root, name)
        if os.path.isdir(full) and name.lower() == cond_low:
            cand = os.path.join(full, "python_only", "labels.json")
            if os.path.exists(cand):
                return cand
    cand = os.path.join(labels_root, condition.strip(), "python_only", "labels.json")
    if os.path.exists(cand):
        return cand
    return None

def resolve_qa_summary_path_fixed(qa_demo_root: str, condition: str, profile_fallback: Optional[str] = None) -> Optional[str]:
    p1 = os.path.join(qa_demo_root, condition, QA_SUMMARY_FIXED_FOLDER, "qa_summary.json")
    if os.path.exists(p1):
        return p1
    if profile_fallback:
        p2 = os.path.join(qa_demo_root, profile_fallback, QA_SUMMARY_FIXED_FOLDER, "qa_summary.json")
        if os.path.exists(p2):
            return p2
    cond_low = condition.strip().lower()
    if os.path.isdir(qa_demo_root):
        for name in os.listdir(qa_demo_root):
            full = os.path.join(qa_demo_root, name)
            if os.path.isdir(full) and name.lower() == cond_low:
                p3 = os.path.join(full, QA_SUMMARY_FIXED_FOLDER, "qa_summary.json")
                if os.path.exists(p3):
                    return p3
    return None


# =========================================================
# Nutrition math
# =========================================================
@dataclass
class UserProfile:
    ad_soyad: str
    sex: str
    age: int
    height_cm: float
    weight_kg: float
    activity: str
    deficit_pct: float

ACTIVITY_FACTORS = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}

def bmi(weight_kg: float, height_cm: float) -> float:
    h_m = height_cm / 100.0
    return 0.0 if h_m <= 0 else weight_kg / (h_m * h_m)

def mifflin_st_jeor_bmr(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    s = 5 if sex.lower().startswith("m") else -161
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + s

def tdee_from_bmr(bmr_v: float, activity: str) -> float:
    return bmr_v * ACTIVITY_FACTORS.get(activity, 1.55)

def target_kcal_from_tdee(tdee_v: float, deficit_pct: float) -> float:
    deficit_pct = max(0.0, min(deficit_pct, 0.6))
    return tdee_v * (1.0 - deficit_pct)


# =========================================================
# Foods loading
# =========================================================
def normalize_foods_any(food_json: Any) -> List[Dict[str, Any]]:
    if isinstance(food_json, list):
        return [x for x in food_json if isinstance(x, dict)]
    if isinstance(food_json, dict):
        for k in ["foods", "items", "data", "records", "rows", "besinler"]:
            v = food_json.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        for v in food_json.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return [x for x in v if isinstance(x, dict)]
    raise ValueError("Food pool JSON format not recognized.")

def load_foods(foods_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(foods_path):
        return []
    try:
        data = load_json(foods_path)
        foods = normalize_foods_any(data)
    except Exception:
        foods = []
        try:
            with open(foods_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            foods.append(obj)
                    except:
                        continue
        except:
            return []

    for f in foods:
        if "besin_id" not in f and "id" in f:
            f["besin_id"] = f["id"]
        try:
            f["kcal"] = float(f.get("kcal", 0) or 0)
        except:
            f["kcal"] = 0.0
        if not isinstance(f.get("ogun"), list):
            f["ogun"] = []
        if not isinstance(f.get("food_tags"), list):
            f["food_tags"] = []
        if not isinstance(f.get("avoid_tags"), list):
            f["avoid_tags"] = []
        if "miktar" not in f:
            f["miktar"] = ""
        if "besin_adi" not in f and "name" in f:
            f["besin_adi"] = f.get("name")
    return foods


# =========================================================
# Meal order (SABİT - senin istediğin)
# =========================================================
MEALS_FIXED = ["sabah", "ara_1", "öğle", "ara_2", "ara_3", "akşam", "yatarken"]
MEAL_ORDER_MAP = {m: i for i, m in enumerate(MEALS_FIXED)}

def sort_meals(meals: List[str]) -> List[str]:
    return sorted(meals, key=lambda m: MEAL_ORDER_MAP.get(m, 99))

def derive_meals_from_foods(_: List[Dict[str, Any]]) -> List[str]:
    return list(MEALS_FIXED)


# =========================================================
# Tag vocab (load full) + Numeric constraint tag-fix engine
# =========================================================
def load_tag_dicts(tag_dicts_path: str) -> Dict[str, Any]:
    td = load_json(tag_dicts_path)
    if not isinstance(td, dict) or "tags" not in td or not isinstance(td["tags"], dict):
        raise ValueError("tag_dicts.json format hatası: top-level 'tags' object olmalı.")
    return td

def _norm_tr(s: str) -> str:
    s = (s or "").lower().strip()
    s = (s.replace("ı", "i").replace("ş", "s").replace("ç", "c")
           .replace("ğ", "g").replace("ö", "o").replace("ü", "u"))
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize_tr(s: str) -> List[str]:
    s = _norm_tr(s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    toks = [t for t in s.split() if t]
    stop = {
        "icin","olan","olarak","onerilir","oneri","onerisi","tuketimi","tuketiminin",
        "sinirlanmasi","haftada","haftalik","gun","gunde","gunluk","en","az","enaz",
        "kadar","yetiskinlerde","yetiskin","cocuk","kisi","miktari","miktar","tuketim",
        "tavsiye","edilir","edilmelidir","olmali","olmalidir","sinir","sinirlandirilmasi",
        "uc","iki","bir","dort","bes","alti","yedi","sekiz","dokuz","on"
    }
    return [t for t in toks if t not in stop and len(t) > 1]

def _walk_collect_strings(obj: Any, out: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
        return
    if isinstance(obj, (int, float, bool)):
        return
    if isinstance(obj, list):
        for it in obj:
            _walk_collect_strings(it, out)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in {"example", "examples", "desc", "description", "name", "label", "labels", "synonym", "synonyms", "keywords", "keyword"}:
                _walk_collect_strings(v, out)
            else:
                _walk_collect_strings(v, out)

def build_tag_text_index(tag_dicts: Dict[str, Any]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    tags = tag_dicts.get("tags") or {}
    for tag, meta in tags.items():
        bag: List[str] = [str(tag)]
        _walk_collect_strings(meta, bag)
        idx[tag] = " ".join(bag)
    return idx

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / max(1, uni)

def _extract_nums(desc: str) -> List[float]:
    if not desc:
        return []
    d = desc.replace(",", ".")
    nums = re.findall(r"(\d+(?:\.\d+)?)", d)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except:
            pass
    return out

def _looks_like_grams(x: Any) -> bool:
    try:
        v = float(x)
        return v >= 30.0
    except:
        return False

def _guess_period_days(desc_norm: str, fallback: Optional[int]) -> Optional[int]:
    if "hafta" in desc_norm or "haftada" in desc_norm or "haftalik" in desc_norm:
        return 7
    if "gun" in desc_norm or "gunde" in desc_norm or "gunluk" in desc_norm:
        return 1
    return fallback

def _guess_unit(desc_norm: str, fallback: str) -> str:
    if "gram" in desc_norm or re.search(r"\b\d+\s*g\b", desc_norm):
        return "gram"
    if "porsiyon" in desc_norm:
        return "porsiyon"
    if "adet" in desc_norm:
        return "adet"
    if "kez" in desc_norm:
        return "kez"
    return fallback or "unknown"

DESC_HINT_TAG = {
    "posa": ["lif_orta", "lif_yuksek", "lif_dusuk"],
    "lif": ["lif_orta", "lif_yuksek", "lif_dusuk"],
    "fiber": ["lif_orta", "lif_yuksek", "lif_dusuk"],
    "baklagil": ["baklagil", "baklagil_yemegi", "baklagil_gaz_yapici"],
    "kirmizi et": ["kirmizi_et", "et_grubu"],
    "beyaz et": ["beyaz_et", "et_grubu"],
    "sebze": ["sebze", "sebze_meyve", "sebzemeyve"],
    "meyve": ["meyve", "sebze_meyve", "sebzemeyve"],
}

def _score_tag_for_description(tag: str, desc: str, tag_text: str) -> float:
    dn = _norm_tr(desc)
    tt = _norm_tr(tag_text)

    desc_tokens = set(_tokenize_tr(dn))
    tag_tokens = set(_tokenize_tr(tt))

    score = 0.0
    score += 10.0 * _jaccard(desc_tokens, tag_tokens)

    if tag in dn:
        score += 2.0

    for hint, candidates in DESC_HINT_TAG.items():
        if hint in dn:
            if tag in candidates:
                score += 1.5

    if ("posa" in dn or "lif" in dn) and ("lif" in tt or "posa" in tt):
        score += 1.0
    if ("baklagil" in dn) and ("baklagil" in tt):
        score += 1.0
    if ("kirmizi" in dn and "et" in dn) and ("kirmizi" in tt or "kirmizi_et" in tt):
        score += 1.0
    if ("beyaz" in dn and "et" in dn) and ("beyaz" in tt or "beyaz_et" in tt):
        score += 1.0

    return score

def guess_best_tag_from_description(
    desc: str,
    tag_text_index: Dict[str, str],
    vocab_keys_set: set,
    current_tag: Optional[str] = None
) -> Optional[str]:
    desc = (desc or "").strip()
    if not desc:
        return current_tag if (isinstance(current_tag, str) and current_tag in vocab_keys_set) else None

    best_tag = None
    best_score = -1e9
    for tag, tag_text in tag_text_index.items():
        if tag not in vocab_keys_set:
            continue
        s = _score_tag_for_description(tag, desc, tag_text)
        if current_tag and tag == current_tag:
            s += 0.2
        if s > best_score:
            best_score = s
            best_tag = tag

    if best_tag is None:
        return None
    if best_score < 0.25:
        return current_tag if (isinstance(current_tag, str) and current_tag in vocab_keys_set) else best_tag
    return best_tag

def _normalize_numeric_constraint_fields(nc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(nc)
    desc = str(out.get("description") or "").strip()
    dn = _norm_tr(desc)

    pd = out.get("period_days", None)
    try:
        pd_i = int(pd) if pd is not None else None
    except:
        pd_i = None
    out["period_days"] = _guess_period_days(dn, pd_i)

    unit = out.get("unit") or "unknown"
    unit = str(unit) if not isinstance(unit, str) else unit
    out["unit"] = _guess_unit(dn, unit)

    nums = _extract_nums(desc)

    if out["unit"] == "gram":
        for fld in ["min_grams", "max_grams"]:
            if fld in out and out[fld] is not None:
                try:
                    out[fld] = float(out[fld])
                except:
                    out[fld] = None

        # gram yanlışlıkla min_count/max_count ise taşı
        if _looks_like_grams(out.get("min_count")) or _looks_like_grams(out.get("max_count")):
            try:
                mc1 = float(out.get("min_count")) if out.get("min_count") is not None else None
            except:
                mc1 = None
            try:
                mc2 = float(out.get("max_count")) if out.get("max_count") is not None else None
            except:
                mc2 = None

            cands = [x for x in [mc1, mc2] if x is not None]
            if cands:
                mn = min(cands)
                mx = max(cands)
                if len(nums) >= 2:
                    mn = min(nums[0], nums[1])
                    mx = max(nums[0], nums[1])
                elif len(nums) == 1:
                    mx = nums[0]
                    mn = out.get("min_grams") or mn

                out["min_grams"] = float(mn) if mn is not None else out.get("min_grams")
                out["max_grams"] = float(mx) if mx is not None else out.get("max_grams")
                out["min_count"] = None
                out["max_count"] = None

        if (out.get("min_grams") is None or out.get("max_grams") is None) and len(nums) >= 1:
            if len(nums) >= 2:
                out["min_grams"] = float(min(nums[0], nums[1]))
                out["max_grams"] = float(max(nums[0], nums[1]))
            else:
                out["max_grams"] = float(nums[0])

        return out

    for fld in ["min_count", "max_count"]:
        if fld in out and out[fld] is not None:
            try:
                out[fld] = float(out[fld])
                if abs(out[fld] - round(out[fld])) < 1e-6:
                    out[fld] = int(round(out[fld]))
            except:
                out[fld] = None

    return out

def _nc_semantic_key(nc: Dict[str, Any]) -> Tuple:
    tag = str(nc.get("tag") or "")
    pd = nc.get("period_days", None)
    try:
        pd = int(pd) if pd is not None else None
    except:
        pd = None
    unit = str(nc.get("unit") or "")

    mn_c = nc.get("min_count", None)
    mx_c = nc.get("max_count", None)
    mn_g = nc.get("min_grams", None)
    mx_g = nc.get("max_grams", None)

    def _n(x):
        if x is None: return None
        try:
            v = float(x)
            return round(v, 4)
        except:
            return None

    desc = str(nc.get("description") or "")
    toks = tuple(sorted(set(_tokenize_tr(desc))))

    return (tag, pd, unit, _n(mn_c), _n(mx_c), _n(mn_g), _n(mx_g), toks)

def _prefer_more_complete(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    def score(x: Dict[str, Any]) -> int:
        s = 0
        for k in ["min_count", "max_count", "min_grams", "max_grams", "period_days", "unit"]:
            v = x.get(k, None)
            if v is not None and str(v).strip() != "":
                s += 1
        s += min(2, len(str(x.get("description") or "")) // 40)
        return s
    return a if score(a) >= score(b) else b

def fix_numeric_constraints_tags_and_dedupe(
    labels_json: Dict[str, Any],
    tag_dicts: Dict[str, Any],
    vocab_keys_set: set
) -> Dict[str, Any]:
    out = dict(labels_json)
    out.setdefault("meal_pattern_rules", {})
    if not isinstance(out["meal_pattern_rules"], dict):
        out["meal_pattern_rules"] = {}
    mpr = out["meal_pattern_rules"]
    mpr.setdefault("numeric_constraints", [])
    if not isinstance(mpr["numeric_constraints"], list):
        mpr["numeric_constraints"] = []

    tag_text_index = build_tag_text_index(tag_dicts)
    fixed: List[Dict[str, Any]] = []

    for nc in mpr["numeric_constraints"]:
        if not isinstance(nc, dict):
            continue
        desc = str(nc.get("description") or "").strip()
        cur_tag = nc.get("tag") if isinstance(nc.get("tag"), str) else None

        best_tag = guess_best_tag_from_description(desc, tag_text_index, vocab_keys_set, current_tag=cur_tag)
        if not best_tag and cur_tag and cur_tag in vocab_keys_set:
            best_tag = cur_tag

        nnc = dict(nc)
        if best_tag:
            nnc["tag"] = best_tag

        nnc = _normalize_numeric_constraint_fields(nnc)

        if nnc.get("unit") == "gram":
            for a, b in [("min_value", "min_grams"), ("max_value", "max_grams")]:
                if nnc.get(b) is None and nnc.get(a) is not None:
                    try:
                        nnc[b] = float(nnc[a])
                    except:
                        pass
            if _looks_like_grams(nnc.get("min_count")) or _looks_like_grams(nnc.get("max_count")):
                nnc["min_count"] = None
                nnc["max_count"] = None

        if not isinstance(nnc.get("tag"), str) or nnc["tag"] not in vocab_keys_set:
            continue

        fixed.append(nnc)

    # exact-ish dedupe
    dedup_map: Dict[Tuple, Dict[str, Any]] = {}
    for nc in fixed:
        key = _nc_semantic_key(nc)
        if key in dedup_map:
            dedup_map[key] = _prefer_more_complete(dedup_map[key], nc)
        else:
            dedup_map[key] = nc

    items = list(dedup_map.values())
    merged: List[Dict[str, Any]] = []
    used = [False] * len(items)

    def get_num_sig(x: Dict[str, Any]) -> Tuple:
        return (
            x.get("tag"),
            x.get("period_days"),
            x.get("unit"),
            x.get("min_count"), x.get("max_count"),
            x.get("min_grams"), x.get("max_grams"),
        )

    # fuzzy dedupe
    for i in range(len(items)):
        if used[i]:
            continue
        a = items[i]
        used[i] = True
        best = a
        ta = set(_tokenize_tr(str(a.get("description") or "")))

        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            b = items[j]
            if (a.get("tag"), a.get("period_days"), a.get("unit")) != (b.get("tag"), b.get("period_days"), b.get("unit")):
                continue
            if get_num_sig(a) != get_num_sig(b):
                continue

            tb = set(_tokenize_tr(str(b.get("description") or "")))
            sim = _jaccard(ta, tb)
            if sim >= 0.85:
                best = _prefer_more_complete(best, b)
                used[j] = True

        merged.append(best)

    def _ord(x: Dict[str, Any]) -> Tuple:
        tag = str(x.get("tag") or "")
        pd = x.get("period_days", 0)
        try:
            pd = int(pd) if pd is not None else 0
        except:
            pd = 0
        unit = str(x.get("unit") or "")
        return (tag, pd, unit, str(x.get("description") or ""))

    merged.sort(key=_ord)
    mpr["numeric_constraints"] = merged

    out.setdefault("audit_log", [])
    if isinstance(out["audit_log"], list):
        out["audit_log"].append({
            "ts": now_ts(),
            "type": "numeric_constraints_tagfix_dedupe",
            "before": len(fixed),
            "after": len(merged),
        })

    return out


# =========================================================
# Rules + Plan utilities
# =========================================================
FISH_TAG = "balik"
DAIRY_TAGS = {"sut_urunu", "peynir", "tam_yagli_sut_urunleri", "az_yagli_sut_urunleri", "yagsiz_sut_urunleri", "yogurt", "kefir"}
MANDATORY_SNACKS = ["ara_1", "ara_2", "ara_3"]
COUNT_UNITS = {"kez", "adet", "porsiyon", "bardak", "dilim", "cay_kasigi", "tatli_kasigi", "yemek_kasigi"}

def food_meals(food: Dict[str, Any]) -> set:
    og = food.get("ogun") or []
    if not isinstance(og, list):
        og = []
    return set([x for x in og if isinstance(x, str)])

def food_tags_union(food: Dict[str, Any]) -> set:
    ft = food.get("food_tags") or []
    at = food.get("avoid_tags") or []
    if not isinstance(ft, list): ft = []
    if not isinstance(at, list): at = []
    return set([x for x in ft if isinstance(x, str)]) | set([x for x in at if isinstance(x, str)])

def food_id(food: Dict[str, Any]) -> str:
    return str(food.get("besin_id") or food.get("id") or "")

def empty_day_plan(meals: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    return {m: [] for m in meals}

def meal_has_fish(meal_items: List[Dict[str, Any]]) -> bool:
    tags = set()
    for it in meal_items:
        tags |= set(it.get("food_tags") or [])
        tags |= set(it.get("avoid_tags") or [])
    return FISH_TAG in tags

def violates_meal_combo(meal_foods: List[Dict[str, Any]]) -> bool:
    tags = set()
    for f in meal_foods:
        tags |= set(f.get("food_tags") or [])
        tags |= set(f.get("avoid_tags") or [])
    return (FISH_TAG in tags) and bool(tags & DAIRY_TAGS)

def read_rules(labels_json: Dict[str, Any]) -> Dict[str, Any]:
    prefer = labels_json.get("prefer_tags") or []
    limit = labels_json.get("limit_tags") or []
    avoid = labels_json.get("avoid_tags") or []
    mpr = labels_json.get("meal_pattern_rules") or {}
    ncs = mpr.get("numeric_constraints") or []
    if not isinstance(ncs, list):
        ncs = []
    norm_ncs = []
    for nc in ncs:
        if not isinstance(nc, dict):
            continue
        tag = nc.get("tag")
        unit = (nc.get("unit") or "unknown")
        if not isinstance(tag, str) or not isinstance(unit, str):
            continue
        norm_ncs.append(nc)
    return {
        "prefer_tags": [t for t in prefer if isinstance(t, str)],
        "limit_tags": [t for t in limit if isinstance(t, str)],
        "avoid_tags": [t for t in avoid if isinstance(t, str)],
        "numeric_constraints": norm_ncs,
        "raw": labels_json,
    }

def is_food_allowed(food: Dict[str, Any], rules: Dict[str, Any], meal_name: Optional[str] = None) -> bool:
    """
    - avoid_tags normalde engel.
    - Ancak sabah öğününde 'ekmek' avoid ise bile sabah serbest kuralı (senin isteğin).
    """
    avoid_set = set(rules.get("avoid_tags") or [])
    tags = food_tags_union(food)
    if meal_name == "sabah" and "ekmek" in tags:
        return True
    return not bool(tags & avoid_set)

def collect_used_food_ids(day_plan: Dict[str, List[Dict[str, Any]]]) -> set:
    used = set()
    for items in day_plan.values():
        for it in items:
            fid = food_id(it)
            if fid:
                used.add(fid)
    return used


# =========================================================
# Öğün-kural yardımcıları (eksik olanlar eklendi)
# =========================================================
def snack_has_kombin(snack_items: List[Dict[str, Any]]) -> bool:
    for it in snack_items:
        if "kombin" in food_tags_union(it):
            return True
    return False

def enforce_fish_meal_only_salad(day_plan: Dict[str, List[Dict[str, Any]]], meal_name: str) -> None:
    """
    Eğer öğünde balık varsa, sadece:
    - balık içeren itemlar + salata/ sebze (salata tag'i) kalsın.
    Ayrıca süt ürünlerini komple çıkar (balık+süt çakışması).
    """
    items = day_plan.get(meal_name, [])
    if not items:
        return
    if not meal_has_fish(items):
        return

    kept = []
    for it in items:
        tags = food_tags_union(it)
        if FISH_TAG in tags:
            kept.append(it)
            continue
        if "salata" in tags or "sebze" in tags:
            if not (tags & DAIRY_TAGS):
                kept.append(it)
            continue
        # diğerlerini at
    day_plan[meal_name] = kept

def enforce_single_main_dish(day_plan: Dict[str, List[Dict[str, Any]]], meal_name: str) -> None:
    """
    Öğle/akşam için "tek ana yemek" kuralı.
    Ana yemek sayılabilecek tag'ler:
      - et_grubu, kirmizi_et, beyaz_et, balik, baklagil_yemegi, baklagil, sebze_yemegi, tavuk, protein
    Eğer 2+ ana yemek varsa, en yüksek kalorili olanı tut, diğerlerini at.
    """
    MAIN_TAGS = {
        "et_grubu", "kirmizi_et", "beyaz_et", "balik",
        "baklagil", "baklagil_yemegi", "sebze_yemegi", "protein"
    }
    items = day_plan.get(meal_name, [])
    if not items or len(items) <= 1:
        return

    main_idxs = []
    for i, it in enumerate(items):
        if food_tags_union(it) & MAIN_TAGS:
            main_idxs.append(i)

    if len(main_idxs) <= 1:
        return

    # en kalorili ana yemeği tut
    def _k(it):
        try:
            return float(it.get("kcal_scaled") or it.get("kcal") or 0)
        except:
            return 0.0

    best_i = max(main_idxs, key=lambda i: _k(items[i]))
    new_items = []
    for i, it in enumerate(items):
        if i == best_i:
            new_items.append(it)
        else:
            # ana yemekse çıkar; değilse kalsın (salata/çorba gibi)
            if i in main_idxs:
                continue
            new_items.append(it)

    day_plan[meal_name] = new_items

def _find_food_candidates_by_tag(foods: List[Dict[str, Any]], meal: str, rules: Dict[str, Any], tag_any: List[str], used_today: set) -> List[Dict[str, Any]]:
    out = []
    tag_any_set = set(tag_any or [])
    for f in foods:
        fid = food_id(f)
        if not fid or fid in used_today:
            continue
        if meal not in food_meals(f):
            continue
        if not is_food_allowed(f, rules, meal_name=meal):
            continue
        if tag_any_set and not (food_tags_union(f) & tag_any_set):
            continue
        out.append(f)
    return out

def force_turkish_breakfast(
    foods: List[Dict[str, Any]],
    rules: Dict[str, Any],
    plan: Dict[str, List[Dict[str, Any]]],
    used_today: set,
    week_constraints: List[Dict[str, Any]],
    week_counts: Dict[str, int],
) -> None:
    """
    Sabah için: ekmek + (peynir/yumurta) + sebze
    - Ekmek avoid olsa bile sabah serbest.
    - Bulamazsa en yakın alternatifleri dener.
    """
    def add_one(tag_any: List[str], kcal_max: float = 450.0) -> bool:
        cands = []
        for f in foods:
            fid = food_id(f)
            if not fid or fid in used_today:
                continue
            if "sabah" not in food_meals(f):
                continue
            # sabah için ekmek istisnası var
            if not is_food_allowed(f, rules, meal_name="sabah"):
                # ekmek dışında avoid varsa engel
                tags = food_tags_union(f)
                if not ("ekmek" in tags):
                    continue
            tags = food_tags_union(f)
            if tag_any and not (set(tag_any) & tags):
                continue
            kcal = float(f.get("kcal", 0) or 0)
            if kcal <= 0 or kcal > kcal_max:
                continue
            cands.append(f)
        if not cands:
            return False
        cands.sort(key=lambda x: float(x.get("kcal", 0) or 0))
        pick = random.choice(cands[:min(10, len(cands))])
        item = make_plan_item_from_food(pick)
        plan["sabah"].append(item)
        used_today.add(food_id(item))
        for t in food_tags_union(item):
            week_counts[t] = week_counts.get(t, 0) + 1
        return True

    # ekmek
    add_one(["ekmek", "tahil"], 400)
    # protein
    if not add_one(["yumurta", "peynir", "sut_urunu", "yogurt"], 450):
        add_one(["kuruyemis", "yagli_tohum"], 350)
    # sebze
    add_one(["sebze", "salata"], 250)


# =========================================================
# Portion inference + scaling (NO FOOD POOL CHANGE)
# =========================================================
UNIT_ALIASES = {
    "g": "gram", "gr": "gram", "gram": "gram",
    "ml": "ml",
    "adet": "adet",
    "dilim": "dilim",
    "yk": "yemek_kasigi",
    "yemek kaşığı": "yemek_kasigi",
    "yemek kasigi": "yemek_kasigi",
    "yemek_kasigi": "yemek_kasigi",
    "kepçe": "kepce", "kepce": "kepce",
    "bardak": "bardak",
    "çay kaşığı": "cay_kasigi", "cay kaşığı": "cay_kasigi", "cay kasigi": "cay_kasigi", "cay_kasigi": "cay_kasigi",
    "tatlı kaşığı": "tatli_kasigi", "tatli kaşığı": "tatli_kasigi", "tatli kasigi": "tatli_kasigi", "tatli_kasigi": "tatli_kasigi",
}

def _norm_unit(u: str) -> str:
    u = (u or "").strip().lower()
    u = u.replace("ı","i").replace("ş","s").replace("ç","c").replace("ğ","g").replace("ö","o").replace("ü","u")
    return UNIT_ALIASES.get(u, u)

def _extract_numbers(s: str) -> List[float]:
    if not s:
        return []
    s2 = s.replace(",", ".")
    nums = re.findall(r"(\d+(?:\.\d+)?)", s2)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except:
            pass
    return out

def _has_tag(food: Dict[str, Any], tag: str) -> bool:
    tags = set(food.get("food_tags") or []) | set(food.get("avoid_tags") or [])
    return tag in tags

def infer_portion_model(food: Dict[str, Any]) -> Dict[str, Any]:
    miktar = str(food.get("miktar") or "").strip()
    kcal = float(food.get("kcal", 0) or 0)

    model = {
        "unit": "porsiyon",
        "qty": 1.0,
        "min": 1.0,
        "max": 1.0,
        "step": 1.0,
        "base_qty": 1.0,
        "base_kcal": kcal,
        "scalable": False
    }
    scalable_flag = _has_tag(food, "scalable")

    low = miktar.lower()

    if "dilim" in low:
        nums = _extract_numbers(miktar)
        base = nums[0] if nums else 1.0
        model.update({"unit": "dilim", "qty": base, "base_qty": base, "min": 1.0, "max": 2.0, "step": 1.0, "scalable": True})
        return model

    if "adet" in low:
        nums = _extract_numbers(miktar)
        base = nums[0] if nums else 1.0
        is_egg = _has_tag(food, "yumurta")
        model.update({"unit": "adet", "qty": base, "base_qty": base, "min": 1.0, "max": 2.0 if is_egg else 3.0, "step": 1.0, "scalable": bool(scalable_flag or is_egg)})
        return model

    if ("yemek kaşığı" in low) or ("yemek kasigi" in low) or re.search(r"\byk\b", low):
        nums = _extract_numbers(miktar)
        if len(nums) >= 2:
            base = float(nums[0]) if nums[1] >= 20 else float(max(nums[0], nums[1]))
        elif len(nums) == 1:
            base = float(nums[0])
        else:
            base = 4.0
        base = max(1.0, base)
        model.update({"unit": "yemek_kasigi", "qty": base, "base_qty": base, "min": 4.0, "max": 10.0, "step": 1.0, "scalable": True})
        return model

    if "kepçe" in low or "kepce" in low:
        nums = _extract_numbers(miktar)
        base = nums[0] if nums else 1.0
        model.update({"unit": "kepce", "qty": base, "base_qty": base, "min": 1.0, "max": 2.0, "step": 1.0, "scalable": True})
        return model

    if "g" in low or "gr" in low:
        nums = _extract_numbers(miktar)
        if len(nums) >= 2:
            base = float(max(nums[0], nums[1]))
        elif len(nums) == 1:
            base = float(nums[0])
        else:
            base = 100.0

        is_red_meat = _has_tag(food, "kirmizi_et")
        is_white_meat = _has_tag(food, "beyaz_et")
        is_meat = _has_tag(food, "et_grubu") or is_red_meat or is_white_meat

        if is_meat:
            if is_red_meat:
                mn = 60.0
                mx = min(200.0, max(base, mn))
            else:
                mn = 100.0
                mx = min(200.0, max(base, mn))
            model.update({"unit": "gram", "qty": base, "base_qty": base, "min": mn, "max": mx, "step": 10.0, "scalable": True})
            return model

        model.update({"unit": "gram", "qty": base, "base_qty": base, "min": base, "max": base, "step": 10.0, "scalable": False})
        return model

    if "ml" in low:
        nums = _extract_numbers(miktar)
        base = nums[0] if nums else 200.0
        is_milk = _has_tag(food, "sut_urunu") and ("süt" in (food.get("besin_adi","").lower()) or "sut" in (food.get("besin_adi","").lower()))
        if is_milk:
            model.update({"unit": "ml", "qty": base, "base_qty": base, "min": 200.0, "max": 250.0, "step": 50.0, "scalable": bool(scalable_flag)})
            return model
        model.update({"unit": "ml", "qty": base, "base_qty": base, "min": base, "max": base, "step": 50.0, "scalable": False})
        return model

    model["scalable"] = bool(scalable_flag)
    return model

def scaled_kcal(base_kcal: float, qty: float, base_qty: float) -> float:
    base_qty = float(base_qty or 1.0)
    if base_qty <= 0:
        base_qty = 1.0
    return float(base_kcal) * (float(qty) / base_qty)

def format_scaled_miktar(portion: Dict[str, Any]) -> str:
    unit = portion.get("unit", "porsiyon")
    qty = portion.get("qty", 1.0)
    if unit == "yemek_kasigi":
        return f"{int(round(qty))} yemek kaşığı"
    if unit == "kepce":
        return f"{int(round(qty))} kepçe"
    if unit == "dilim":
        return f"{int(round(qty))} dilim"
    if unit == "adet":
        return f"{int(round(qty))} adet"
    if unit == "gram":
        return f"{int(round(qty))} g"
    if unit == "ml":
        return f"{int(round(qty))} ml"
    return f"{qty} {unit}"

def make_plan_item_from_food(food: Dict[str, Any]) -> Dict[str, Any]:
    it = dict(food)
    p = infer_portion_model(food)
    it["portion"] = p
    it["miktar_scaled"] = format_scaled_miktar(p)
    it["kcal_scaled"] = scaled_kcal(float(food.get("kcal",0) or 0), float(p["qty"]), float(p["base_qty"]))
    return it

def _item_kcal(it: Dict[str, Any]) -> float:
    if isinstance(it, dict) and "kcal_scaled" in it:
        try:
            return float(it.get("kcal_scaled") or 0)
        except:
            return 0.0
    try:
        return float(it.get("kcal", 0) or 0)
    except:
        return 0.0

def plan_kcal(plan: Dict[str, List[Dict[str, Any]]]) -> float:
    return sum(_item_kcal(it) for items in plan.values() for it in items)

def meal_kcal(plan: Dict[str, List[Dict[str, Any]]], meal: str) -> float:
    return sum(_item_kcal(it) for it in plan.get(meal, []))


# =========================================================
# QA summary -> text
# =========================================================
def _qa_to_text(raw_answers: Any) -> str:
    if raw_answers is None:
        return ""
    if isinstance(raw_answers, str):
        return raw_answers.strip()
    if isinstance(raw_answers, dict):
        for k0 in ["quote", "answer", "text"]:
            if isinstance(raw_answers.get(k0), str) and raw_answers.get(k0).strip():
                return raw_answers[k0].strip()
        for k in ["answers", "items", "qa_pairs", "data", "records", "rows"]:
            v = raw_answers.get(k)
            if isinstance(v, list) and v:
                parts = []
                for i, it in enumerate(v, 1):
                    if isinstance(it, str):
                        parts.append(f"- {it}")
                    elif isinstance(it, dict):
                        q = it.get("question") or it.get("q") or ""
                        a = it.get("answer") or it.get("a") or it.get("quote") or ""
                        qq = q.strip() if isinstance(q, str) else ""
                        aa = a.strip() if isinstance(a, str) else ""
                        if qq or aa:
                            parts.append(f"[{i}] S: {qq}\n    C: {aa}")
                        else:
                            parts.append(json.dumps(it, ensure_ascii=False))
                    else:
                        parts.append(str(it))
                return "\n".join(parts).strip()
        return json.dumps(raw_answers, ensure_ascii=False, indent=2)
    if isinstance(raw_answers, list):
        return "\n".join([str(x) for x in raw_answers]).strip()
    return str(raw_answers).strip()


# =========================================================
# Weekly numeric constraints (period_days=7, count units)
# =========================================================
def extract_weekly_count_constraints(rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for nc in rules.get("numeric_constraints", []) or []:
        if not isinstance(nc, dict):
            continue
        tag = nc.get("tag")
        unit = (nc.get("unit") or "unknown")
        pd = nc.get("period_days", None)

        if not isinstance(tag, str) or not isinstance(unit, str):
            continue
        if pd is None:
            continue
        try:
            pd_i = int(pd)
        except:
            continue
        if pd_i != 7:
            continue
        if unit not in COUNT_UNITS:
            continue

        min_c = nc.get("min_count", None)
        max_c = nc.get("max_count", None)
        if min_c is not None:
            try: min_c = float(min_c)
            except: min_c = None
        if max_c is not None:
            try: max_c = float(max_c)
            except: max_c = None
        if min_c is None and max_c is None:
            continue

        out.append({
            "tag": tag, "min_count": min_c, "max_count": max_c,
            "unit": unit, "description": (nc.get("description") or "").strip(),
        })
    return out

def week_tag_counts(week_days: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in week_days:
        plan = d.get("meals") or {}
        if not isinstance(plan, dict):
            continue
        for items in plan.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                tags = food_tags_union(it)
                for t in tags:
                    counts[t] = counts.get(t, 0) + 1
    return counts

def _max_banned_tags_from_week(week_constraints: List[Dict[str, Any]], week_counts: Dict[str, int]) -> set:
    max_banned = set()
    for c in week_constraints:
        t = c.get("tag")
        mx = c.get("max_count", None)
        if not isinstance(t, str) or mx is None:
            continue
        try:
            mx_i = int(mx)
        except:
            continue
        if week_counts.get(t, 0) >= mx_i:
            max_banned.add(t)
    return max_banned


# =========================================================
# Draft plan (weekly constraints aware)
# =========================================================
def pick_best_food(
    foods: List[Dict[str, Any]],
    meal: str,
    rules: Dict[str, Any],
    prefer_tags: List[str],
    used_today: set,
    week_constraints: List[Dict[str, Any]],
    week_counts: Dict[str, int],
    banned_tags_any: Optional[List[str]] = None,
    must_have_any: Optional[List[str]] = None,
    kcal_max: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    banned_tags_any = banned_tags_any or []
    must_have_any = must_have_any or []
    banned = set(banned_tags_any) | set(rules["avoid_tags"])
    max_banned_tags = _max_banned_tags_from_week(week_constraints, week_counts)

    candidates = []
    for f in foods:
        fid = food_id(f)
        if not fid or fid in used_today:
            continue
        if meal not in food_meals(f):
            continue
        if not is_food_allowed(f, rules, meal_name=meal):
            continue

        tags = food_tags_union(f)
        if tags & banned:
            continue
        if tags & max_banned_tags:
            continue
        if must_have_any and not (set(must_have_any) & tags):
            continue

        kcal = float(f.get("kcal", 0) or 0)
        if kcal_max is not None and kcal > kcal_max:
            continue

        prefer_matches = len(set(prefer_tags) & tags)
        limit_matches = len(set(rules["limit_tags"]) & tags)

        weekly_bonus = 0.0
        for c in week_constraints:
            t = c.get("tag")
            mn = c.get("min_count", None)
            if not isinstance(t, str) or mn is None:
                continue
            try:
                mn_i = int(mn)
            except:
                continue
            if t in tags and week_counts.get(t, 0) < mn_i:
                weekly_bonus += 5.0

        score = prefer_matches * 3 + weekly_bonus - limit_matches * 2 - (kcal / 400.0)
        candidates.append((score, f))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)

    k = min(6, len(candidates))
    top = candidates[:k]
    scores = [s for s, _ in top]
    mx = max(scores)
    weights = [math.exp((s - mx) / 1.5) for s in scores]
    total = sum(weights) or 1.0
    r = random.random() * total
    acc = 0.0
    for w, (_, f) in zip(weights, top):
        acc += w
        if r <= acc:
            return f
    return top[0][1]


def enforce_weekly_min_constraints(
    days: List[Dict[str, Any]],
    foods: List[Dict[str, Any]],
    meals: List[str],
    rules: Dict[str, Any]
) -> List[Dict[str, Any]]:
    constraints = extract_weekly_count_constraints(rules)
    if not constraints:
        return days

    for week_idx in range((len(days) + 6) // 7):
        start = week_idx * 7
        end = min(len(days), start + 7)
        week_slice = days[start:end]
        counts = week_tag_counts(week_slice)

        for c in constraints:
            tag = c["tag"]
            mn = c.get("min_count", None)
            mx = c.get("max_count", None)
            if mn is None:
                continue
            try:
                need = int(mn) - counts.get(tag, 0)
            except:
                continue
            if need <= 0:
                continue

            for _ in range(need):
                placed = False
                if mx is not None:
                    try:
                        if counts.get(tag, 0) >= int(mx):
                            break
                    except:
                        pass

                for di in range(start, end):
                    day_plan = days[di]["meals"]
                    used_today = collect_used_food_ids(day_plan)

                    # öncelik: öğle/akşam
                    for meal_try in ["öğle", "akşam", "sabah", "ara_1", "ara_2", "ara_3", "yatarken"]:
                        if meal_try not in day_plan:
                            continue

                        item_food = pick_best_food(
                            foods=foods, meal=meal_try, rules=rules,
                            prefer_tags=rules.get("prefer_tags", []),
                            used_today=used_today,
                            week_constraints=constraints, week_counts=counts,
                            must_have_any=[tag], kcal_max=950
                        )
                        if item_food:
                            item = make_plan_item_from_food(item_food)
                            tmp = list(day_plan[meal_try]) + [item]
                            if violates_meal_combo(tmp):
                                continue
                            day_plan[meal_try].append(item)
                            used_today.add(food_id(item))
                            for t in food_tags_union(item):
                                counts[t] = counts.get(t, 0) + 1
                            placed = True
                            break

                    if placed:
                        break

                if not placed:
                    break

    return days

def build_initial_plan_days(
    foods: List[Dict[str, Any]],
    meals: List[str],
    rules: Dict[str, Any],
    day_count: int,
    seed: int = 42
) -> Dict[str, Any]:
    days = []
    prefer = rules["prefer_tags"]
    week_constraints = extract_weekly_count_constraints(rules)

    DINNER_PROTEIN_TAGS = ["balik", "et_grubu", "kirmizi_et", "beyaz_et", "protein"]

    for d in range(1, day_count + 1):
        random.seed(seed + d)
        plan = empty_day_plan(meals)
        used_today = collect_used_food_ids(plan)

        week_idx = (d - 1) // 7
        week_start = week_idx * 7 + 1
        week_days_so_far = [x for x in days if week_start <= int(x.get("day", 0)) <= d - 1]
        week_counts = week_tag_counts(week_days_so_far)

        def try_add(meal_name: str, must_any: List[str], kcal_max: float):
            if meal_name not in plan:
                return
            item_food = pick_best_food(
                foods=foods,
                meal=meal_name,
                rules=rules,
                prefer_tags=prefer,
                used_today=used_today,
                week_constraints=week_constraints,
                week_counts=week_counts,
                must_have_any=must_any,
                kcal_max=kcal_max
            )
            if item_food:
                item = make_plan_item_from_food(item_food)
                tmp = list(plan[meal_name]) + [item]
                if violates_meal_combo(tmp):
                    return
                plan[meal_name].append(item)
                fid = food_id(item)
                if fid:
                    used_today.add(fid)
                for t in food_tags_union(item):
                    week_counts[t] = week_counts.get(t, 0) + 1

        # SABAH
        if "sabah" in plan:
            force_turkish_breakfast(
                foods=foods,
                rules=rules,
                plan=plan,
                used_today=used_today,
                week_constraints=week_constraints,
                week_counts=week_counts,
            )

        # ARA_1
        if "ara_1" in plan:
            try_add("ara_1", ["kombin"], 450)
            if snack_has_kombin(plan.get("ara_1", [])):
                plan["ara_1"] = [it for it in plan["ara_1"] if "kombin" in food_tags_union(it)]
            else:
                r = random.random()
                if r < 0.45:
                    try_add("ara_1", ["meyve"], 240)
                elif r < 0.75:
                    try_add("ara_1", ["kuruyemis", "yagli_tohum"], 280)
                else:
                    try_add("ara_1", ["sut_urunu", "yogurt", "kefir"], 260)
                if len(plan["ara_1"]) == 0:
                    try_add("ara_1", ["meyve"], 240)

        # ÖĞLE
        if "öğle" in plan:
            try_add("öğle", ["corba"], 320)
            try_add("öğle", ["sebze_yemegi", "baklagil", "baklagil_yemegi", "protein"], 950)
            if random.random() < 0.20:
                try_add("öğle", DINNER_PROTEIN_TAGS, 950)
            try_add("öğle", ["salata", "sebze"], 320)
            if random.random() < 0.50:
                try_add("öğle", ["ekmek", "tahil"], 260)

            enforce_fish_meal_only_salad(plan, "öğle")
            enforce_single_main_dish(plan, "öğle")

        # ARA_2
        if "ara_2" in plan:
            try_add("ara_2", ["kombin"], 450)
            if snack_has_kombin(plan.get("ara_2", [])):
                plan["ara_2"] = [it for it in plan["ara_2"] if "kombin" in food_tags_union(it)]
            else:
                r = random.random()
                if r < 0.40:
                    try_add("ara_2", ["meyve"], 240)
                elif r < 0.70:
                    try_add("ara_2", ["sut_urunu", "yogurt", "kefir"], 260)
                else:
                    try_add("ara_2", ["kuruyemis", "yagli_tohum"], 280)
                if len(plan["ara_2"]) == 0:
                    try_add("ara_2", ["meyve"], 240)

        # ARA_3 (zorunlu)
        if "ara_3" in plan:
            r = random.random()
            if r < 0.50:
                try_add("ara_3", ["meyve"], 240)
            else:
                try_add("ara_3", ["sut_urunu", "yogurt", "kefir"], 260)
            if len(plan["ara_3"]) == 0:
                try_add("ara_3", ["meyve"], 240)

        # AKŞAM
        if "akşam" in plan:
            try_add("akşam", ["corba"], 320)
            try_add("akşam", DINNER_PROTEIN_TAGS, 1100)
            if not any(food_tags_union(it) & set(DINNER_PROTEIN_TAGS) for it in plan["akşam"]):
                try_add("akşam", ["sebze_yemegi", "baklagil", "baklagil_yemegi", "protein"], 1100)
            try_add("akşam", ["salata", "sebze"], 320)
            if random.random() < 0.65:
                try_add("akşam", ["ekmek"], 260)

            enforce_fish_meal_only_salad(plan, "akşam")
            enforce_single_main_dish(plan, "akşam")

        # YATARKEN (opsiyonel)
        if "yatarken" in plan and random.random() < 0.60:
            try_add("yatarken", ["sut_urunu", "yogurt", "kefir"], 220)

        # Balık+süt çakışması son güvenlik
        for meal_name, items in list(plan.items()):
            if violates_meal_combo(items):
                plan[meal_name] = [it for it in items if not (food_tags_union(it) & DAIRY_TAGS)]

        # final enforce
        for mm in ["öğle", "akşam"]:
            if mm in plan:
                enforce_fish_meal_only_salad(plan, mm)
                enforce_single_main_dish(plan, mm)

        days.append({"day": d, "meals": plan})

        if week_constraints:
            days = enforce_weekly_min_constraints(days, foods, meals, rules)

    return {"days": days, "notes": [f"draft_generated_days={day_count}"]}


# =========================================================
# LLM Audit: PATCH (silme yok, sadece ekleme)
# =========================================================
def build_audit_system_prompt_patch() -> str:
    return (
        "Sen bir 'Tıbbi Kural Tamamlayıcı'sın.\n"
        "Amaç: labels_json zaten büyük ölçüde doğru. ONLARI SİLME / BOZMA.\n"
        "Sadece qa_summary_text içinde olup labels_json'da eksik kalan kural ve özellikle sayısal kısıtları PATCH olarak üret.\n\n"
        "KURALLAR:\n"
        "- ASLA mevcut prefer/limit/avoid listelerini boşaltma veya silme önerme.\n"
        "- Sadece 'ekleme' patch'i döndür.\n"
        "- Yeni tag uydurma. Sadece tag_vocab_keys içinden tag kullan.\n"
        "- Sayısal kısıtları numeric_constraints_add içine yaz.\n"
        "- Örnek: \"haftada 2-3 kez 1 adet yumurta\" -> tag:'yumurta', period_days:7, unit:'adet', min_count:2, max_count:3, description:'...'\n\n"
        "ÇIKTI SADECE JSON OLACAK (markdown yok, açıklama yok).\n"
        "ŞEMA:\n"
        "{\n"
        "  \"numeric_constraints_add\": [\n"
        "     {\"tag\":\"...\",\"period_days\":7,\"unit\":\"adet\",\"min_count\":2,\"max_count\":3,\"description\":\"...\"}\n"
        "  ],\n"
        "  \"prefer_add\": [\"tag\"],\n"
        "  \"limit_add\": [\"tag\"],\n"
        "  \"avoid_add\": [\"tag\"],\n"
        "  \"notes\": [\"...\"]\n"
        "}\n"
    )

def build_audit_user_prompt_patch(labels_json: Dict[str, Any], raw_answers: Any, tag_vocab_keys: List[str], condition: str) -> str:
    qa_text = _qa_to_text(raw_answers)
    payload = {
        "task": "qa_crosscheck_patch_only",
        "condition": condition,
        "labels_json": labels_json,
        "qa_summary_text": qa_text,
        "tag_vocab_keys": tag_vocab_keys[:1200],
        "instruction": "Return ONLY PATCH JSON. Do NOT remove anything from labels_json."
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def _nc_key(nc: Dict[str, Any]) -> Tuple:
    return (
        str(nc.get("tag") or ""),
        int(nc.get("period_days") or 0) if str(nc.get("period_days") or "").isdigit() else str(nc.get("period_days") or ""),
        str(nc.get("unit") or ""),
        str(nc.get("min_count") or ""),
        str(nc.get("max_count") or ""),
        str(nc.get("min_value") or ""),
        str(nc.get("max_value") or ""),
        str(nc.get("min_grams") or ""),
        str(nc.get("max_grams") or ""),
    )

def apply_audit_patch(base_labels: Dict[str, Any], patch: Dict[str, Any], vocab_keys_set: set) -> Dict[str, Any]:
    out = dict(base_labels)

    out.setdefault("prefer_tags", [])
    out.setdefault("limit_tags", [])
    out.setdefault("avoid_tags", [])
    out.setdefault("meal_pattern_rules", {})
    if not isinstance(out["meal_pattern_rules"], dict):
        out["meal_pattern_rules"] = {}
    out["meal_pattern_rules"].setdefault("numeric_constraints", [])
    if not isinstance(out["meal_pattern_rules"]["numeric_constraints"], list):
        out["meal_pattern_rules"]["numeric_constraints"] = []

    prefer = set([t for t in out.get("prefer_tags", []) if isinstance(t, str)])
    limit = set([t for t in out.get("limit_tags", []) if isinstance(t, str)])
    avoid = set([t for t in out.get("avoid_tags", []) if isinstance(t, str)])

    for k, target_set in [("prefer_add", prefer), ("limit_add", limit), ("avoid_add", avoid)]:
        arr = patch.get(k) or []
        if isinstance(arr, list):
            for t in arr:
                if isinstance(t, str) and t in vocab_keys_set:
                    target_set.add(t)

    existing = out["meal_pattern_rules"]["numeric_constraints"]
    existing_keys = set()
    for nc in existing:
        if isinstance(nc, dict):
            existing_keys.add(_nc_key(nc))

    add_list = patch.get("numeric_constraints_add") or []
    if isinstance(add_list, list):
        for nc in add_list:
            if not isinstance(nc, dict):
                continue
            tag = nc.get("tag")
            if not isinstance(tag, str) or tag not in vocab_keys_set:
                continue

            pd = nc.get("period_days", None)
            try:
                pd_i = int(pd)
            except:
                pd_i = None
            if pd_i is None or pd_i <= 0:
                continue
            nc["period_days"] = pd_i

            unit = nc.get("unit") or "unknown"
            if not isinstance(unit, str):
                unit = "unknown"
            nc["unit"] = unit

            for fld in ["min_count", "max_count", "min_value", "max_value", "min_grams", "max_grams"]:
                if fld in nc and nc[fld] is not None:
                    try:
                        nc[fld] = float(nc[fld])
                        if fld in ["min_count", "max_count"] and abs(nc[fld] - round(nc[fld])) < 1e-6:
                            nc[fld] = int(round(nc[fld]))
                    except:
                        nc[fld] = None

            if not isinstance(nc.get("description"), str):
                nc["description"] = ""

            key = _nc_key(nc)
            if key in existing_keys:
                continue
            existing.append(nc)
            existing_keys.add(key)

    prefer = prefer - avoid - limit
    limit = limit - avoid

    out["prefer_tags"] = sorted(prefer)
    out["limit_tags"] = sorted(limit)
    out["avoid_tags"] = sorted(avoid)

    out.setdefault("audit_log", [])
    if isinstance(out["audit_log"], list):
        out["audit_log"].append({
            "ts": now_ts(),
            "type": "patch_applied",
            "added_numeric_constraints": len(patch.get("numeric_constraints_add") or []),
            "added_prefer": len(patch.get("prefer_add") or []),
            "added_limit": len(patch.get("limit_add") or []),
            "added_avoid": len(patch.get("avoid_add") or []),
        })

    return out

def save_revised_labels(revised_root: str, condition: str, model_key: str, audited_labels: Dict[str, Any]) -> str:
    out_dir = os.path.join(revised_root, safe_name(condition), safe_name(model_key))
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "labels.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audited_labels, f, ensure_ascii=False, indent=2)
    return out_path

def llm_audit_patch_only(
    llm: OllamaClient,
    model_ollama: str,
    labels_json_base: Dict[str, Any],
    raw_answers: Any,
    tag_vocab_keys: List[str],
    vocab_keys_set: set,
    condition: str,
    debug_dir: str,
    model_key: str,
    tag_dicts: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    system = build_audit_system_prompt_patch()
    user = build_audit_user_prompt_patch(labels_json_base, raw_answers, tag_vocab_keys, condition)
    out = llm.chat(model=model_ollama, system=system, user=user, temperature=0.0)

    print("\n================ LLM AUDIT PATCH (HAM ÇIKTI) ================\n")
    print(out if out else "(boş)")
    print("\n=============================================================\n")
    write_text(os.path.join(debug_dir, "audit_patch_raw.txt"), out or "")

    patch: Dict[str, Any] = {}
    try:
        patch = extract_json_obj(out)
        if not isinstance(patch, dict):
            patch = {}
        print("✅ LLM patch JSON parse edildi.\n")
    except Exception as e:
        print("❌ LLM patch JSON parse edilemedi -> patch boş sayılacak.")
        print("Hata:", str(e))
        patch = {}

    patch.setdefault("numeric_constraints_add", [])
    patch.setdefault("prefer_add", [])
    patch.setdefault("limit_add", [])
    patch.setdefault("avoid_add", [])
    patch.setdefault("notes", [])

    revised = apply_audit_patch(labels_json_base, patch, vocab_keys_set=vocab_keys_set)
    revised = fix_numeric_constraints_tags_and_dedupe(revised, tag_dicts=tag_dicts, vocab_keys_set=vocab_keys_set)

    revised_path = save_revised_labels(DEFAULT_LLM_REVISED_ROOT, condition, model_key, revised)
    revised = load_json(revised_path)

    print("✅ Revize labels kaydedildi:", revised_path)
    return revised, revised_path, patch


# =========================================================
# LLM: QA SUMMARY'den Türkçe açıklama üret (terminal + PDF için)
# =========================================================
def build_explain_system_prompt_tr() -> str:
    return (
        "Sen bir diyetisyen asistanısın.\n"
        "Sana bir 'qa_summary_text' ve hastalık/condition adı verilecek.\n"
        "Görev:\n"
        "1) QA metnindeki en önemli beslenme kurallarını Türkçe, maddeli ve anlaşılır şekilde özetle.\n"
        "2) Haftalık sayısal sınırlar varsa (örn: haftada 2-3 kez) mutlaka belirt.\n"
        "3) Tıbbi iddia ekleme, sadece verilen metni düzenle/özetle.\n"
        "ÇIKTI: düz metin (JSON değil)."
    )

def build_explain_user_prompt_tr(condition: str, raw_answers: Any) -> str:
    qa_text = _qa_to_text(raw_answers)
    payload = {
        "condition": condition,
        "qa_summary_text": qa_text
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def llm_generate_explanation_tr(llm: OllamaClient, model_ollama: str, condition: str, raw_answers: Any, debug_dir: str) -> str:
    if not raw_answers:
        return "⚠️ QA_SUMMARY bulunamadı veya boş. Bu nedenle açıklama üretilemedi."
    system = build_explain_system_prompt_tr()
    user = build_explain_user_prompt_tr(condition, raw_answers)
    out = llm.chat(model=model_ollama, system=system, user=user, temperature=0.2)
    out = (out or "").strip()
    write_text(os.path.join(debug_dir, "qa_explanation_tr.txt"), out)
    return out or "⚠️ LLM açıklama üretmedi (boş çıktı)."


# =========================================================
# Terminal yazdırma (düzgün liste)
# =========================================================
def item_name(it: Dict[str, Any]) -> str:
    for k in ["name", "besin_adi", "food_name", "title", "ad", "isim"]:
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    fid = food_id(it)
    return f"(besin_id={fid})"

def print_explanation_terminal(model_label: str, condition: str, explanation_text: str) -> None:
    print("\n==================== LLM AÇIKLAMA (QA SUMMARY) ====================")
    print(f"Model: {model_label}")
    print(f"Profil/Hastalık: {condition}")
    print("---------------------------------------------------------------")
    print(explanation_text.strip() if explanation_text else "(boş)")
    print("===================================================================\n")

def print_day_plan(day_index: int, day_plan: Dict[str, List[Dict[str, Any]]], meals: List[str]) -> None:
    print(f"\n==================== GÜN {day_index} ====================")
    print(f"Toplam kcal: {round(plan_kcal(day_plan), 1)}")
    for meal in meals:
        items = day_plan.get(meal, [])
        print(f"\n[{meal}]")
        if not items:
            print("  - (boş)")
            continue
        for it in items:
            mk = it.get("miktar_scaled") or it.get("miktar") or ""
            print(f"  - {item_name(it)} | miktar={mk} | kcal={_item_kcal(it):.0f}")
    print("\n=========================================================\n")


# =========================================================
# PDF (Türkçe font fix + her gün 1 sayfa) - SADECE AÇIKLAMA + PLAN
# =========================================================
def _try_register_font_pair(name_reg: str, path_reg: str, name_bold: str, path_bold: str) -> Optional[Tuple[str, str]]:
    try:
        pdfmetrics.registerFont(TTFont(name_reg, path_reg))
        if os.path.exists(path_bold):
            pdfmetrics.registerFont(TTFont(name_bold, path_bold))
            return name_reg, name_bold
        return name_reg, name_reg
    except Exception:
        return None

def register_font_if_possible() -> Tuple[str, str]:
    candidates: List[Tuple[str, str, str, str]] = []
    candidates += [
        ("TRFont", os.path.join(DEFAULT_PROJECT_ROOT, "assets", "fonts", "DejaVuSans.ttf"),
         "TRFont-Bold", os.path.join(DEFAULT_PROJECT_ROOT, "assets", "fonts", "DejaVuSans-Bold.ttf")),
        ("TRFont", os.path.join(DEFAULT_PROJECT_ROOT, "assets", "fonts", "NotoSans-Regular.ttf"),
         "TRFont-Bold", os.path.join(DEFAULT_PROJECT_ROOT, "assets", "fonts", "NotoSans-Bold.ttf")),
    ]
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates += [
            ("TRFont", os.path.join(base_dir, "assets", "fonts", "DejaVuSans.ttf"),
             "TRFont-Bold", os.path.join(base_dir, "assets", "fonts", "DejaVuSans-Bold.ttf")),
            ("TRFont", os.path.join(base_dir, "assets", "fonts", "NotoSans-Regular.ttf"),
             "TRFont-Bold", os.path.join(base_dir, "assets", "fonts", "NotoSans-Bold.ttf")),
        ]
    except Exception:
        pass

    win = r"C:\Windows\Fonts"
    candidates += [
        ("TRFont", os.path.join(win, "DejaVuSans.ttf"), "TRFont-Bold", os.path.join(win, "DejaVuSans-Bold.ttf")),
        ("TRFont", os.path.join(win, "arial.ttf"), "TRFont-Bold", os.path.join(win, "arialbd.ttf")),
        ("TRFont", os.path.join(win, "calibri.ttf"), "TRFont-Bold", os.path.join(win, "calibrib.ttf")),
        ("TRFont", os.path.join(win, "seguisym.ttf"), "TRFont-Bold", os.path.join(win, "seguisym.ttf")),
    ]
    for (nreg, preg, nbold, pbold) in candidates:
        if os.path.exists(preg):
            pair = _try_register_font_pair(nreg, preg, nbold, pbold)
            if pair:
                return pair
    return "Helvetica", "Helvetica-Bold"

def create_pdf_plan(
    filename: str,
    user_profile: UserProfile,
    condition: str,
    model_label: str,
    explanation_text: str,
    meals: List[str],
    scaled_days: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> None:
    font_reg, font_bold = register_font_if_possible()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName=font_bold, fontSize=18, leading=22, textColor=colors.HexColor("#0B2A4A"))
    h1 = ParagraphStyle("h1", parent=styles["Heading2"], fontName=font_bold, fontSize=13, leading=16, textColor=colors.HexColor("#0B2A4A"), spaceBefore=10)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font_reg, fontSize=10.5, leading=13)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontName=font_reg, fontSize=9.5, leading=12, textColor=colors.HexColor("#334155"))

    NAVY = colors.HexColor("#0B2A4A")
    GRID = colors.HexColor("#CBD5E1")
    CARD_BG = colors.HexColor("#F8FAFC")

    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
    )

    story: List[Any] = []
    story.append(Paragraph("BESLENME PROGRAMI", title))
    story.append(Spacer(1, 0.25 * cm))

    dt = time.strftime("%d.%m.%Y")
    story.append(Paragraph(f"<b>Danışan:</b> {user_profile.ad_soyad} &nbsp;&nbsp; <b>Profil:</b> {condition} &nbsp;&nbsp; <b>Tarih:</b> {dt}", body))
    story.append(Paragraph(f"<b>Model:</b> {model_label}", body))
    story.append(Spacer(1, 0.2 * cm))

    info_left = [f"<b>Cinsiyet:</b> {user_profile.sex}", f"<b>Yaş:</b> {user_profile.age}", f"<b>Boy:</b> {user_profile.height_cm} cm"]
    info_right = [f"<b>Kilo:</b> {user_profile.weight_kg} kg", f"<b>Aktivite:</b> {user_profile.activity}", f"<b>Hedef Kalori:</b> {int(metrics.get('target_kcal', 0))} kcal"]
    rows = max(len(info_left), len(info_right))
    data = []
    for i in range(rows):
        l = info_left[i] if i < len(info_left) else ""
        r = info_right[i] if i < len(info_right) else ""
        data.append([Paragraph(l, body), Paragraph(r, body)])
    t = Table(data, colWidths=[8.7 * cm, 8.7 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("QA SUMMARY'YE GÖRE AÇIKLAMA", h1))
    story.append(Spacer(1, 0.1 * cm))
    # açıklamayı paragraflara böl
    exp = (explanation_text or "").strip()
    if not exp:
        exp = "QA_SUMMARY bulunamadığı için açıklama üretilemedi."
    for para in exp.split("\n"):
        p = para.strip()
        if not p:
            story.append(Spacer(1, 0.1 * cm))
            continue
        story.append(Paragraph(p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), small))

    story.append(PageBreak())

    for idx, day in enumerate(scaled_days):
        d = int(day.get("day", 0))
        plan = day.get("meals") or {}

        story.append(Paragraph(f"PLAN - GÜN {d}", h1))
        story.append(Paragraph(f"Gün toplam: {int(plan_kcal(plan))} kcal", body))
        story.append(Spacer(1, 0.2 * cm))

        for meal in meals:
            items = plan.get(meal, [])
            if not items:
                continue
            story.append(Paragraph(f"<b>{meal.upper()}</b> (≈ {int(meal_kcal(plan, meal))} kcal)", body))
            lines = []
            for it in items:
                nm = item_name(it)
                kcalv = _item_kcal(it)
                mk = it.get("miktar_scaled") or it.get("miktar") or ""
                lines.append(f"{nm} - {mk} ({int(kcalv)} kcal)")
            lf2 = ListFlowable([ListItem(Paragraph(x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), small), value="•") for x in lines],
                               bulletType="bullet", leftIndent=14)
            story.append(lf2)
            story.append(Spacer(1, 0.12 * cm))

        if idx != len(scaled_days) - 1:
            story.append(PageBreak())

    doc.build(story)


# =========================================================
# Basit değerlendirme (model kıyası)
# =========================================================
def evaluate_plan_against_rules(scaled_days: List[Dict[str, Any]], rules: Dict[str, Any], target_kcal: float) -> Dict[str, Any]:
    avoid_set = set(rules.get("avoid_tags") or [])
    weekly_constraints = extract_weekly_count_constraints(rules)

    kcal_abs = []
    kcal_pct_abs = []
    day_within_10pct = 0
    avoid_viol = 0
    fish_dairy_viol = 0
    snack_empty_days = 0
    total_items = 0

    for day in scaled_days:
        plan = day.get("meals") or {}
        day_kcal = float(plan_kcal(plan))
        delta = abs(day_kcal - float(target_kcal))
        kcal_abs.append(delta)
        pct = (delta / max(1.0, float(target_kcal)))
        kcal_pct_abs.append(pct)
        if pct <= 0.10:
            day_within_10pct += 1

        empty_snack = False
        for s in MANDATORY_SNACKS:
            if s in plan and len(plan.get(s, [])) == 0:
                empty_snack = True
        if empty_snack:
            snack_empty_days += 1

        for items in plan.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                total_items += 1
                tags = food_tags_union(it)
                if tags & avoid_set:
                    avoid_viol += 1

        for meal_items in plan.values():
            if isinstance(meal_items, list) and violates_meal_combo(meal_items):
                fish_dairy_viol += 1

    weekly_results = []
    for week_idx in range((len(scaled_days) + 6) // 7):
        start = week_idx * 7
        end = min(len(scaled_days), start + 7)
        slice_days = scaled_days[start:end]
        counts = week_tag_counts(slice_days)
        week_ok = True
        details = []
        for c in weekly_constraints:
            t = c["tag"]
            mn = c.get("min_count", None)
            mx = c.get("max_count", None)
            val = counts.get(t, 0)
            ok = True
            if mn is not None and val < int(mn):
                ok = False
            if mx is not None and val > int(mx):
                ok = False
            details.append({"tag": t, "count": val, "min": mn, "max": mx, "ok": ok, "desc": c.get("description", "")})
            if not ok:
                week_ok = False
        weekly_results.append({"week": week_idx + 1, "ok": week_ok, "details": details})

    avg_abs = sum(kcal_abs) / max(1, len(kcal_abs))
    avg_pct = sum(kcal_pct_abs) / max(1, len(kcal_pct_abs))
    within_10pct_rate = day_within_10pct / max(1, len(scaled_days))

    violation_units = float(avoid_viol + fish_dairy_viol + snack_empty_days)
    denom = max(1.0, float(total_items) + float(len(scaled_days)))
    constraint_adherence = max(0.0, 1.0 - (violation_units / denom))

    weekly_ok_rate = (sum(1 for w in weekly_results if w.get("ok")) / max(1, len(weekly_results))) if weekly_results else 1.0

    return {
        "avg_abs_kcal_delta": round(avg_abs, 2),
        "avg_pct_kcal_delta": round(avg_pct, 4),
        "within_10pct_days_rate": round(within_10pct_rate, 4),
        "avoid_violations": int(avoid_viol),
        "fish_dairy_violations": int(fish_dairy_viol),
        "mandatory_snack_empty_days": int(snack_empty_days),
        "total_items": int(total_items),
        "constraint_adherence_est": round(constraint_adherence, 4),
        "weekly_numeric_ok_rate": round(weekly_ok_rate, 4),
        "weekly_numeric": weekly_results,
    }


# =========================================================
# TEST REPORT WRITER (per model)
# =========================================================
def save_model_test_report_json(
    report_root: str,
    condition: str,
    model_key: str,
    report_obj: Dict[str, Any],
) -> str:
    out_dir = os.path.join(report_root, safe_name(condition), safe_name(model_key))
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"report_{now_ts()}.json")
    write_json(out_path, report_obj)
    return out_path


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n=== Diet AI (LLM PATCH AUDIT + QA AÇIKLAMA + OPSİYONEL PDF) ===\n")

    # Dosya yollarını da Türkçe sor (isterse default)
    foods_path = prompt_str("Besin havuzu JSON yolu", DEFAULT_FOODS_PATH)
    tag_dicts_path = prompt_str("tag_dicts.json yolu", DEFAULT_TAG_DICTS_PATH)
    labels_root = prompt_str("labels klasörü yolu", DEFAULT_LABELS_ROOT)
    qa_demo_root = prompt_str("qa_demo (deneme_cache/qa_demo) yolu", DEFAULT_QA_DEMO_ROOT)

    if not os.path.exists(foods_path):
        print("❌ Foods dosyası yok:", foods_path)
        return
    if not os.path.exists(tag_dicts_path):
        print("❌ tag_dicts.json yok:", tag_dicts_path)
        return

    ad_soyad = prompt_str("Ad Soyad", "Danışan")
    condition = prompt_str("Hangi hastalık / condition?", DEFAULT_CONDITION)

    weeks = prompt_int("Kaç haftalık plan? (1=7 gün, 4=28 gün)", 1, min_v=1, max_v=12)
    day_count = weeks * 7

    print("\nKullanılacak ana modeller (4):")
    for k in DEFAULT_MODEL_ORDER:
        print(f" - {k}: {MODELS[k]['label']} ({MODELS[k]['ollama']})")

    use_all = prompt_yes_no("Tüm 4 modeli çalıştırayım mı?", default_yes=True)
    also_extras = prompt_yes_no("Ek modelleri de deneyeyim mi? (mistral/qwen7b/llama3.2)", default_yes=False)

    model_keys: List[str] = []
    if use_all:
        model_keys += list(DEFAULT_MODEL_ORDER)
    else:
        chosen = prompt_str("Model key yaz (örn: gemma_3_4b)", "gemma_3_4b").strip()
        model_keys += [chosen] if chosen in MODELS else [DEFAULT_MODEL_ORDER[0]]

    if also_extras:
        model_keys += [k for k in EXTRA_MODEL_ORDER if k in MODELS]

    base_url = prompt_str("Ollama base_url", DEFAULT_BASE_URL)

    sex = prompt_choice("Cinsiyet", ["male", "female"], "female")
    age = prompt_int("Yaş", 30, min_v=10, max_v=90)
    height_cm = prompt_float("Boy (cm)", 165.0)
    weight_kg = prompt_float("Kilo (kg)", 70.0)
    activity = prompt_choice("Aktivite", list(ACTIVITY_FACTORS.keys()), "light")
    deficit_pct = prompt_float("Defisit (0.15 = %15)", 0.15)

    seed = prompt_int("Random seed", 42, min_v=0, max_v=10_000)
    want_pdf = prompt_yes_no("Planı PDF olarak da oluşturmak ister misiniz?", default_yes=True)

    foods = load_foods(foods_path)
    print(f"\n✅ FOODS LOADED: {len(foods)}  |  {foods_path}")
    if not foods:
        print("❌ Foods=0 -> besin JSON formatı bozuk veya okunamadı.")
        return

    meals = derive_meals_from_foods(foods)
    print("✅ ÖĞÜN SIRASI:", " -> ".join(meals))

    tag_dicts = load_tag_dicts(tag_dicts_path)
    tag_vocab_keys = sorted(list((tag_dicts.get("tags") or {}).keys()))
    vocab_keys_set = set(tag_vocab_keys)

    labels_path = resolve_labels_path(labels_root, condition)
    if not labels_path:
        print("❌ NLP labels bulunamadı:", os.path.join(labels_root, condition, "python_only", "labels.json"))
        print("Mevcut conditionlar:", ", ".join(list_conditions(labels_root)))
        return

    labels_json_base = load_json(labels_path)
    print("\n✅ NLP LABELS PATH:", labels_path)

    labels_json_base = fix_numeric_constraints_tags_and_dedupe(labels_json_base, tag_dicts=tag_dicts, vocab_keys_set=vocab_keys_set)

    qa_path = resolve_qa_summary_path_fixed(qa_demo_root, condition, profile_fallback=condition)
    raw_answers = load_optional_text_or_json(qa_path) if qa_path else ""
    if qa_path:
        print("✅ QA_SUMMARY PATH:", qa_path)
    else:
        print(f"⚠️ QA_SUMMARY bulunamadı (opsiyonel). Beklenen: qa_demo/<condition>/{QA_SUMMARY_FIXED_FOLDER}/qa_summary.json")

    profile = UserProfile(ad_soyad=ad_soyad, sex=sex, age=age, height_cm=height_cm, weight_kg=weight_kg, activity=activity, deficit_pct=deficit_pct)
    bmr_v = mifflin_st_jeor_bmr(profile.sex, profile.weight_kg, profile.height_cm, profile.age)
    tdee_v = tdee_from_bmr(bmr_v, profile.activity)
    target = target_kcal_from_tdee(tdee_v, profile.deficit_pct)

    metrics = {
        "bmi": round(bmi(profile.weight_kg, profile.height_cm), 2),
        "bmr_mifflin": round(bmr_v, 0),
        "tdee": round(tdee_v, 0),
        "deficit_pct": float(profile.deficit_pct),
        "target_kcal": round(float(target), 0),
        "meals_detected": meals,
        "day_count": day_count,
        "weeks": weeks,
    }
    print("\n✅ METRICS:", metrics)

    llm = OllamaClient(base_url=base_url)
    all_model_scores: Dict[str, Any] = {}

    for model_key in model_keys:
        if model_key not in MODELS:
            print("⚠️ Model key yok, atlanıyor:", model_key)
            continue

        mi = MODELS[model_key]
        model_label = mi["label"]
        model_ollama = mi["ollama"]

        print(f"\n\n==================== MODEL ÇALIŞIYOR: {model_key} ====================")
        print(f"Label: {model_label} | Ollama: {model_ollama}")

        debug_dir = os.path.join(DEFAULT_OUT_ROOT, safe_name(model_key), safe_name(condition), f"run_{now_ts()}")
        ensure_dir(debug_dir)

        revised_labels = dict(labels_json_base)
        revised_path = ""
        patch_obj: Dict[str, Any] = {}

        # 1) Audit patch
        try:
            revised_labels, revised_path, patch_obj = llm_audit_patch_only(
                llm=llm,
                model_ollama=model_ollama,
                labels_json_base=labels_json_base,
                raw_answers=raw_answers,
                tag_vocab_keys=tag_vocab_keys,
                vocab_keys_set=vocab_keys_set,
                condition=condition,
                debug_dir=debug_dir,
                model_key=model_key,
                tag_dicts=tag_dicts,
            )
        except Exception as e:
            print("⚠️ Audit aşamasında model hata verdi, bu model atlanacak.")
            print("Hata:", str(e))
            continue

        revised_labels = fix_numeric_constraints_tags_and_dedupe(revised_labels, tag_dicts=tag_dicts, vocab_keys_set=vocab_keys_set)

        rules = read_rules(revised_labels)
        # vocab dışı tagleri kırp
        rules["prefer_tags"] = [t for t in rules["prefer_tags"] if t in vocab_keys_set]
        rules["limit_tags"]  = [t for t in rules["limit_tags"]  if t in vocab_keys_set]
        rules["avoid_tags"]  = [t for t in rules["avoid_tags"]  if t in vocab_keys_set]
        rules["numeric_constraints"] = [nc for nc in rules["numeric_constraints"] if isinstance(nc, dict) and isinstance(nc.get("tag"), str) and nc["tag"] in vocab_keys_set]

        # 2) QA açıklaması üret (terminal + pdf)
        explanation_text = llm_generate_explanation_tr(llm, model_ollama, condition, raw_answers, debug_dir=debug_dir)
        print_explanation_terminal(model_label, condition, explanation_text)

        # 3) Plan üret
        draft = build_initial_plan_days(foods=foods, meals=meals, rules=rules, day_count=day_count, seed=seed)
        scaled_days = draft["days"]

        # haftalık minimumları son kez garanti et
        scaled_days = enforce_weekly_min_constraints(scaled_days, foods, meals, rules)

        print("\n==================== DİYET LİSTESİ ====================")
        for day in scaled_days:
            print_day_plan(day["day"], day["meals"], meals)
        print("========================================================\n")

        score = evaluate_plan_against_rules(scaled_days, rules, float(metrics["target_kcal"]))

        thresholds = {
            "constraint_adherence_min": 0.90,
            "within_10pct_days_rate_min": 0.85,
            "weekly_numeric_ok_rate_min": 0.85,
            "factual_accuracy_bertscore_min": 0.85,
            "textual_rougeL_min": 0.80,
            "pilot_likert_min": 3.8,
        }

        pass_flags = {
            "constraint_adherence_pass": (score.get("constraint_adherence_est", 0.0) >= thresholds["constraint_adherence_min"]),
            "numeric_within10pct_pass": (score.get("within_10pct_days_rate", 0.0) >= thresholds["within_10pct_days_rate_min"]),
            "weekly_numeric_pass": (score.get("weekly_numeric_ok_rate", 0.0) >= thresholds["weekly_numeric_ok_rate_min"]),
            "factual_accuracy_bertscore": "not_applicable_without_reference_texts",
            "textual_rougeL": "not_applicable_without_reference_texts",
            "pilot_user_tests": "manual_required",
        }

        report_obj = {
            "ts": now_ts(),
            "condition": condition,
            "model": {
                "model_key": model_key,
                "model_label": model_label,
                "model_ollama": model_ollama,
            },
            "inputs": {
                "profile": {
                    "ad_soyad": profile.ad_soyad,
                    "sex": profile.sex,
                    "age": profile.age,
                    "height_cm": profile.height_cm,
                    "weight_kg": profile.weight_kg,
                    "activity": profile.activity,
                    "deficit_pct": profile.deficit_pct,
                },
                "metrics": metrics,
                "seed": seed,
                "weeks": weeks,
                "day_count": day_count,
                "base_url": base_url,
            },
            "files": {
                "labels_base_path": labels_path,
                "labels_revised_path": revised_path,
                "debug_dir": debug_dir,
            },
            "qa_explanation_text": explanation_text,
            "tests": {
                "constraint_adherence": {
                    "constraint_adherence_est": score.get("constraint_adherence_est"),
                    "avoid_violations": score.get("avoid_violations"),
                    "fish_dairy_violations": score.get("fish_dairy_violations"),
                    "mandatory_snack_empty_days": score.get("mandatory_snack_empty_days"),
                    "target_min": thresholds["constraint_adherence_min"],
                    "pass": pass_flags["constraint_adherence_pass"],
                },
                "numeric_accuracy": {
                    "avg_abs_kcal_delta": score.get("avg_abs_kcal_delta"),
                    "avg_pct_kcal_delta": score.get("avg_pct_kcal_delta"),
                    "within_10pct_days_rate": score.get("within_10pct_days_rate"),
                    "target_within10pct_rate_min": thresholds["within_10pct_days_rate_min"],
                    "pass": pass_flags["numeric_within10pct_pass"],
                },
                "weekly_numeric_constraints": {
                    "weekly_numeric_ok_rate": score.get("weekly_numeric_ok_rate"),
                    "target_min": thresholds["weekly_numeric_ok_rate_min"],
                    "pass": pass_flags["weekly_numeric_pass"],
                    "details": score.get("weekly_numeric"),
                },
                "reference_metrics": {
                    "BERTScore": "not_applicable",
                    "ROUGE-L": "not_applicable",
                }
            },
            "thresholds": thresholds,
            "pass_flags": pass_flags,
            "raw_score": score,
            "patch_used": patch_obj,
        }

        report_path = save_model_test_report_json(DEFAULT_TEST_REPORT_ROOT, condition, model_key, report_obj)
        print("✅ TEST REPORT saved:", report_path)

        all_model_scores[model_key] = {
            "model_label": model_label,
            "model_ollama": model_ollama,
            "labels_base_path": labels_path,
            "labels_revised_path": revised_path,
            "patch_used": patch_obj,
            "debug_dir": debug_dir,
            "score": score,
            "test_report_path": report_path,
        }

        print("\n[MODEL SCORE]")
        print(json.dumps(score, ensure_ascii=False, indent=2))

        # PDF (sadece explanation + plan)
        if want_pdf:
            pdf_dir = os.path.join(DEFAULT_OUT_ROOT, safe_name(model_key), safe_name(condition), "pdf")
            ensure_dir(pdf_dir)
            pdf_name = f"program_{safe_name(profile.ad_soyad)}_{safe_name(condition)}_{safe_name(model_key)}_{now_ts()}.pdf"
            pdf_path = os.path.join(pdf_dir, pdf_name)

            create_pdf_plan(
                filename=pdf_path,
                user_profile=profile,
                condition=condition,
                model_label=model_label,
                explanation_text=explanation_text,
                meals=meals,
                scaled_days=scaled_days,
                metrics=metrics,
            )
            print("✅ PDF oluşturuldu:", pdf_path)

        print(f"\n✅ OK  model={model_key}")

    print("\n\n==================== MODEL KARŞILAŞTIRMA ÖZETİ ====================")

    def weekly_fail_count(score_obj: Dict[str, Any]) -> int:
        ws = score_obj.get("weekly_numeric") or []
        return sum(1 for w in ws if not w.get("ok"))

    rows = []
    for mk, info in all_model_scores.items():
        sc = info["score"]
        rows.append({
            "model_key": mk,
            "avg_abs_kcal_delta": sc.get("avg_abs_kcal_delta"),
            "avg_pct_kcal_delta": sc.get("avg_pct_kcal_delta"),
            "within_10pct_days_rate": sc.get("within_10pct_days_rate"),
            "constraint_adherence_est": sc.get("constraint_adherence_est"),
            "avoid_violations": sc.get("avoid_violations"),
            "fish_dairy_violations": sc.get("fish_dairy_violations"),
            "mandatory_snack_empty_days": sc.get("mandatory_snack_empty_days"),
            "weekly_fail_weeks": weekly_fail_count(sc),
            "test_report_path": info.get("test_report_path", ""),
        })
    rows.sort(key=lambda r: (r["avoid_violations"], r["weekly_fail_weeks"], r["avg_abs_kcal_delta"]))

    print(json.dumps(rows, ensure_ascii=False, indent=2))

    summary_path = os.path.join(DEFAULT_OUT_ROOT, f"model_compare_{safe_name(condition)}_{now_ts()}.json")
    ensure_dir(DEFAULT_OUT_ROOT)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"condition": condition, "rows": rows, "raw": all_model_scores}, f, ensure_ascii=False, indent=2)

    print("\n✅ Karşılaştırma raporu kaydedildi:", summary_path)
    print("\nDONE\n")


if __name__ == "__main__":
    main()
