# -*- coding: utf-8 -*-
"""
generate_disease_rules_local_llama.py

REVIZE-17 (2026-01-15) İSTEKLER:
✅ Cache artık ayrı bir kök dizine yazılır: --cache_root (default: <out_dir>\deneme_cache)
✅ Rules JSON + debug ayrı dizine yazılır: --rules_out_dir (default: <out_dir>\chroma_rules)
✅ Kod değişince cache çakışmasın: cache key'lere otomatik code_fp (script fingerprint) eklendi
✅ Rules üretildiyse Chroma'ya upsert opsiyonel: --store_rules_to_chroma (default OFF)
   - rules koleksiyonu: --rules_collection (default: disease_rules_store)

NOT:
- Mevcut RAG koleksiyonun (diyetisyen_rehberi) aynen kullanılır.
- Rules için ayrı bir Chroma collection kullanılır (KV gibi saklamak için).
"""

import os
import re
import json
import time
import random
import hashlib
import argparse
import shutil
from typing import Any, Dict, List, Set, Optional, Tuple

import requests
import chromadb
from chromadb.utils import embedding_functions


# =============================
# CONFIG (OLLAMA)
# =============================
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
DEFAULT_NUM_CTX = 8192

DEFAULT_TAG_DICTS_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\tag_dicts.json"
DEFAULT_OUT_DIR = r"C:\Users\user\Desktop\diyetisyen_llm"
DEFAULT_CHROMA_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\chroma_db_storage"

# İSTEK: deneme_cache ve chroma_rules klasörü
DEFAULT_CACHE_ROOT = os.path.join(DEFAULT_OUT_DIR, "deneme_cache")
DEFAULT_RULES_OUT_DIR = os.path.join(DEFAULT_OUT_DIR, "chroma_rules")

DEFAULT_RAG_COLLECTION = "diyetisyen_rehberi"
DEFAULT_TENANT = "default_tenant"
DEFAULT_DATABASE = "default_database"

SCRIPT_REV = "REVIZE-17"

# Query içinde kolesterol geçerse domain override
KOLESTEROL_FORCE_RE = re.compile(
    r"\b(kolesterol|kolester|ldl|hdl|trigliserid|dislipidemi|hiperlipidemi)\b",
    re.IGNORECASE
)


# =============================
# SCRIPT FINGERPRINT (cache invalidation)
# =============================
def script_fingerprint() -> str:
    """
    Kod değişince cache otomatik invalid olsun diye dosya SHA1 fingerprint.
    """
    try:
        # __file__ bazı ortamlarda olmayabilir
        p = os.path.abspath(__file__)
        with open(p, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except Exception:
        # Son çare: rev + py ver
        return hashlib.sha1((SCRIPT_REV + "_no_file").encode("utf-8")).hexdigest()[:12]


# =============================
# CACHE
# =============================
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _safe_fs_name(s: str, max_len: int = 60) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s2 = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s2 = s2.strip("._-")
    if not s2:
        s2 = "model"
    if len(s2) > max_len:
        s2 = s2[:max_len].rstrip("._-")
    return s2 or "model"

def build_model_cache_subdir(model: str, num_ctx: int) -> str:
    safe = _safe_fs_name(model, max_len=60)
    h = _sha1(f"{model}__{int(num_ctx)}")[:10]
    return f".cache__{safe}__ctx{int(num_ctx)}__{h}"

class DiskCache:
    def __init__(self, base_dir: str, disabled: bool = False, cache_subdir: str = ".cache"):
        self.base_dir = os.path.join(base_dir, cache_subdir)
        self.disabled = bool(disabled)
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.base_dir, f"{key}.json")

    def clear(self) -> None:
        if os.path.isdir(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)
        os.makedirs(self.base_dir, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        if self.disabled:
            return None
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, obj: Any) -> None:
        if self.disabled:
            return
        p = self._path(key)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


# =============================
# IO / NORMALIZE
# =============================
def norm_plain(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")
    s = re.sub(r"\s+", " ", s)
    return s

def safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_dump_json(obj: Any, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clean_json_text(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(\w+)?", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s

def extract_first_json_object_balanced(text: str) -> Optional[dict]:
    s = clean_json_text(text)
    start = s.find("{")
    if start == -1:
        return None
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
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1].strip()
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


# =============================
# DOMAIN OVERRIDE + CANONICAL QUERY
# =============================
def force_domain_if_needed(domain: str, query: str) -> str:
    if KOLESTEROL_FORCE_RE.search(query or ""):
        return "kolesterol"
    return (domain or "").strip()

def canonicalize_query_for_cache(query: str) -> str:
    q = norm_plain(query)
    q = re.sub(r"[^\w\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


# =============================
# OLLAMA CHAT (LOCAL)
# =============================
def ollama_chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_retries: int = 4,
    num_ctx: int = DEFAULT_NUM_CTX
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_ctx": int(num_ctx),
        }
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=300)
            if r.ok:
                j = r.json()
                return (j.get("message", {}) or {}).get("content", "") or ""
            last_err = f"Ollama HTTP {r.status_code}: {r.text}"
        except Exception as e:
            last_err = str(e)

        sleep_s = min(8, 2 ** attempt) + random.random()
        print(f"[WARN] Ollama çağrısı başarısız: {last_err} | {sleep_s:.1f}s bekleyip retry ({attempt}/{max_retries})")
        time.sleep(sleep_s)
    raise RuntimeError(f"Ollama chat başarısız: {last_err}")


# =============================
# TAG DICTS (RAW) + patch (sebze_meyve kaldırıldı)
# =============================
def load_tag_dicts_raw(tag_dicts_path: str) -> Dict[str, Any]:
    data = safe_load_json(tag_dicts_path)
    if not isinstance(data, dict):
        raise ValueError("tag_dicts.json formatı beklenmeyen. kök dict olmalı.")
    return data

def extract_tags_from_raw(raw: Dict[str, Any]) -> Dict[str, str]:
    tags = raw.get("tags")
    if not isinstance(tags, dict):
        tags = raw  # legacy
    out: Dict[str, str] = {}
    for k, v in tags.items():
        if not isinstance(k, str) or not k.strip():
            continue
        out[k.strip()] = (v if isinstance(v, str) else str(v)).strip()
    return out

def patch_tag_dicts_if_missing(tag_dicts_path: str, disable_patch: bool = False) -> None:
    """
    İSTEK: sebze_meyve patch'i KALDIRILDI.
    Bu fonksiyon artık no-op (mimariyi bozmamak için çağrı kalabilir).
    """
    _ = (tag_dicts_path, disable_patch)
    return


# =============================
# DOMAIN POLICY
# =============================
def norm_domain_key(s: str) -> str:
    return norm_plain(s).replace(" ", "_")

def get_domain_policy(raw_tag_dict: Dict[str, Any], domain: str) -> Dict[str, Any]:
    dp_legacy = raw_tag_dict.get("domain_policies")
    dp_new = raw_tag_dict.get("domains")

    merged: Dict[str, Any] = {}
    if isinstance(dp_new, dict):
        merged.update(dp_new)
    if isinstance(dp_legacy, dict):
        merged.update(dp_legacy)

    if not merged:
        return {}

    dkey = norm_domain_key(domain)
    if dkey in merged and isinstance(merged[dkey], dict):
        return merged[dkey]

    for key, pol in merged.items():
        if not isinstance(pol, dict):
            continue
        if norm_domain_key(str(key)) == dkey:
            return pol
        aliases = pol.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    if norm_plain(a) == norm_plain(domain):
                        return pol
                    if norm_plain(a) in norm_plain(domain) or norm_plain(domain) in norm_plain(a):
                        return pol
    return {}

def policy_bool(policy: Dict[str, Any], key: str, default: bool) -> bool:
    v = policy.get(key, default)
    return bool(v)

def policy_list(policy: Dict[str, Any], key: str) -> List[str]:
    v = policy.get(key)
    if isinstance(v, list):
        return [str(x).strip() for x in v if isinstance(x, str) and x.strip()]
    return []

def policy_required_any(policy: Dict[str, Any]) -> List[str]:
    return policy_list(policy, "required_any")


# =============================
# Heuristics / Filters
# =============================
NOISE_KEYWORDS = [
    "digoksin", "antidepresan", "warfarin", "metformin", "insulin", "insülin",
    "ilac", "ilaç", "farmak", "emilim", "doz", "tablet", "kapsul", "ampul", "enjeksiyon",
    "iv", "intravenoz", "subkutan", "sc", "im", "protokol", "order", "reçete", "recete",
]
CLINICAL_SETTING_KEYWORDS = [
    "yoğun bakım", "yogun bakim", "intensive care", "icu",
    "yanik", "yanık", "burn",
    "kemoterapi", "radyoterapi", "onkoloji",
    "diyaliz", "dializ",
    "parenteral", "enteral", "ng", "peg", "kateter", "ventilat",
]

FOOD_SIGNAL_RE = re.compile(
    r"(ekmek|tam tahil|tam tahıl|bulgur|pirin[cç]|makarna|pilav|sebze|meyve|"
    r"kurubaklagil|kuru baklagil|baklagil|mercimek|nohut|fasulye|"
    r"et\b|tavuk|hindi|bal[iı]k|yumurta|sakatat|"
    r"s[uü]t|yo[gğ]urt|kefir|ayran|peynir|"
    r"ya[gğ]|zeytinyag|margarin|tereya[gğ]|krema|kaymak|"
    r"kuruyemi[sş]|ceviz|badem|f[iı]nd[iı]k|"
    r"tuz|sodyum|şeker|seker|trans|doymu[sş]|"
    r"tam yagli|tam yağlı|az yagli|az yağlı|yagsiz|yağsız|light|"
    r"akdeniz|d(a|â)sh|mediterranean)",
    re.IGNORECASE
)

NUMERIC_HINT_RE = re.compile(
    r"\b(gunde|günde|haftada|ayda)\b|\b(en\s*az|en\s*fazla)\b|\b(\d+)\s*(porsiyon|adet|kez|g|gr|gram|mg)\b",
    re.IGNORECASE
)

HARD_AVOID_CUE_RE = re.compile(
    r"("
    r"tuketilmemeli|tüketilmemeli|yasak|"
    r"uzak\s*dur(ul)?(malı|un|unmalı|unuz|unuz)?|"
    r"kesinlikle\s*.*(tuketme|tüketme)|"
    r"\b(kacin|kaçın)(ın|iniz|ınız|malıdır|malidir|)\b|"
    r"mümkünse\s*tüketmeyin|tüketmeyiniz|"
    r"kaçınılmalıdır|kacinilmalidir|"
    r"sınırlandırılmalıdır|sinirlandirilmalidir|"
    r"tüketimleri\s*sınırlandırılmalıdır|tuketimleri\s*sinirlandirilmalidir"
    r")",
    re.IGNORECASE
)

LIMIT_CUE_RE = re.compile(
    r"(sinirlandir|sınırlandır|kisitla|kısıtla|azalt|en\s*fazla|ölçülü|kısıtlı|sınırlı|yeterlidir|"
    r"sınırlandırılmalıdır|sinirlandirilmalidir)",
    re.IGNORECASE
)
PREFER_CUE_RE = re.compile(
    r"(oner|öner|tercih|tüketin|tuket(in)?|artır|artir|yer ver|daha fazla|seçilmelidir|sec(i|ı)lmelidir)",
    re.IGNORECASE
)

CONDITION_CUE_RE = re.compile(
    r"(derisi\s*ayr[iı]larak|ya[gğ]s[iı]z\s*tavada|ha[sş]lama|f[ıi]r[ıi]nda|k[ıi]zartma\s*yerine|tuz\s*eklemeden|kabuklu\s*olarak|iyi\s*y[ıi]kanmal[ıi])",
    re.IGNORECASE
)

OTHER_DISEASE_TERMS_FALLBACK = [
    "diyabet", "tip 1", "tip1", "tip 2", "tip2", "gestasyonel",
    "hipertansiyon", "tansiyon",
    "bobrek", "böbrek", "nefropati", "diyaliz",
    "karaciger", "karaciğer", "hepatit", "siroz",
    "kanser", "onkoloji", "kemoterapi", "radyoterapi",
    "gut", "hiperürisemi", "hiperurisemi",
    "çölyak", "colyak", "gluten",
    "anemi", "demir eksikligi", "demir eksikliği",
    "pku", "fenilketonuri",
]

def extract_all_domain_terms(raw_tag_dict: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    dp_legacy = raw_tag_dict.get("domain_policies")
    dp_new = raw_tag_dict.get("domains")
    sources: List[Dict[str, Any]] = []
    if isinstance(dp_new, dict):
        sources.append(dp_new)
    if isinstance(dp_legacy, dict):
        sources.append(dp_legacy)
    for dp in sources:
        for key, pol in dp.items():
            if isinstance(key, str) and key.strip():
                out.add(norm_plain(key))
            if not isinstance(pol, dict):
                continue
            aliases = pol.get("aliases") or []
            if isinstance(aliases, list):
                for a in aliases:
                    if isinstance(a, str) and a.strip():
                        out.add(norm_plain(a))
    out = set([t for t in out if len(t) >= 3])
    return out

def build_other_disease_terms_dynamic(raw_tag_dict: Dict[str, Any], current_domain: str, policy: Dict[str, Any]) -> List[str]:
    all_terms = extract_all_domain_terms(raw_tag_dict)
    cur_terms: Set[str] = {norm_plain(current_domain)}
    aliases = policy.get("aliases") or []
    if isinstance(aliases, list):
        for a in aliases:
            if isinstance(a, str) and a.strip():
                cur_terms.add(norm_plain(a))

    allow_related_terms = policy.get("allow_related_terms") or policy.get("allow_related") or []
    allow_set: Set[str] = set()
    if isinstance(allow_related_terms, list):
        for t in allow_related_terms:
            if isinstance(t, str) and t.strip():
                allow_set.add(norm_plain(t))

    other = (all_terms - cur_terms - allow_set)
    if not other:
        return [norm_plain(t) for t in OTHER_DISEASE_TERMS_FALLBACK if isinstance(t, str) and t.strip()]
    return sorted(other, key=len, reverse=True)

def is_noisy_med_text(text: str) -> bool:
    low = norm_plain(text)
    return any(norm_plain(x) in low for x in NOISE_KEYWORDS)

def clinical_setting_hits_text(text: str) -> int:
    low = norm_plain(text)
    hits = 0
    for kw in CLINICAL_SETTING_KEYWORDS:
        if norm_plain(kw) in low:
            hits += 1
    return hits

def is_food_actionable_text(text: str) -> bool:
    return bool(FOOD_SIGNAL_RE.search(text or "")) or bool(NUMERIC_HINT_RE.search(text or ""))

def contains_other_disease_terms(text: str, other_terms: List[str]) -> bool:
    low = norm_plain(text)
    for t in other_terms:
        tt = norm_plain(t)
        if tt and tt in low:
            return True
    return False

def has_constraint_signal(text: str) -> bool:
    t = text or ""
    return bool(NUMERIC_HINT_RE.search(t)) or bool(LIMIT_CUE_RE.search(t))


# =============================
# CHROMA helpers (uyumluluk)
# =============================
def _get_collection_names(client: chromadb.Client) -> List[str]:
    try:
        cols = client.list_collections()
    except Exception:
        return []
    names: List[str] = []
    if isinstance(cols, list):
        for c in cols:
            if isinstance(c, str):
                names.append(c)
            else:
                nm = getattr(c, "name", None)
                if isinstance(nm, str) and nm:
                    names.append(nm)
    return names


# =============================
# CHROMA RETRIEVAL (+ CACHE)
# =============================
def chroma_client(persist_dir: str, tenant: str, database: str) -> chromadb.Client:
    return chromadb.PersistentClient(path=persist_dir, tenant=tenant, database=database)

def chroma_retrieve_chunks(client: chromadb.Client, collection_name: str, query: str, top_k: int, where: Optional[dict] = None) -> List[Dict[str, Any]]:
    existing = _get_collection_names(client)
    if collection_name not in existing:
        raise RuntimeError(
            f"Chroma collection bulunamadı: '{collection_name}'. "
            f"Mevcut koleksiyonlar: {existing}. "
            f"--rag_collection / --chroma_dir / --tenant / --database kontrol et."
        )
    col = client.get_collection(collection_name)
    res = col.query(query_texts=[query], n_results=int(top_k), where=where)

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        md = metas[i] or {}
        out.append({
            "chunk_id": str(ids[i]),
            "topic_group": md.get("topic_group"),
            "ana_baslik": md.get("ana_baslik"),
            "section": md.get("section"),
            "doc_id": md.get("doc_id"),
            "content": docs[i] or "",
            "distance": dists[i] if i < len(dists) else None,
            "retrieval_query": query,
        })
    return out

def cached_chroma_retrieve(
    cache: DiskCache,
    client: chromadb.Client,
    collection_name: str,
    query: str,
    top_k: int,
    where: Optional[dict],
    cache_namespace: str,
    code_fp: str,
    cache_bust: str = ""
) -> List[Dict[str, Any]]:
    key_payload = {
        "ns": cache_namespace,
        "collection": collection_name,
        "query": query,
        "top_k": int(top_k),
        "where": where or None,
        "cache_bust": cache_bust or "",
        "code_fp": code_fp or "",
        "rev": SCRIPT_REV,
    }
    key = "chroma_" + _sha1(json.dumps(key_payload, ensure_ascii=False, sort_keys=True))
    hit = cache.get(key)
    if isinstance(hit, list):
        return hit
    res = chroma_retrieve_chunks(client, collection_name, query, top_k, where)
    cache.set(key, res)
    return res

def dedupe_by_chunk_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        cid = str(it.get("chunk_id") or "")
        if not cid or cid in seen:
            continue
        out.append(it)
        seen.add(cid)
    return out


# =============================
# CHROMA RULES UPSERT (NEW)
# =============================
def upsert_rules_json_to_chroma(
    client: chromadb.Client,
    collection_name: str,
    domain: str,
    rules_obj: Dict[str, Any],
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Rules JSON'u tek doküman olarak Chroma'ya upsert eder.
    """
    ef = embedding_functions.DefaultEmbeddingFunction()
    col = client.get_or_create_collection(name=collection_name, embedding_function=ef)

    doc_id = f"rules::{norm_domain_key(domain)}"
    doc_text = json.dumps(rules_obj, ensure_ascii=False)

    md = {
        "type": "disease_rules",
        "domain": domain,
        "updated_at": int(time.time()),
        "schema_rev": SCRIPT_REV,
    }
    if metadata_extra:
        md.update(metadata_extra)

    col.upsert(
        ids=[doc_id],
        documents=[doc_text],
        metadatas=[md],
    )


# =============================
# DOMAIN UNDERSTANDING (CACHE)
# =============================
def build_domain_understanding_system_prompt() -> str:
    return (
        "Sen klinik-odaklı bir RAG yöneticisisin.\n"
        "Verilen DOMAIN_HINT ve USER_QUERY'e göre domain'i tespit et, eşanlamları/alt başlıkları çıkar ve retrieval için güçlü sorgular üret.\n"
        "SADECE JSON döndür:\n"
        "{\n"
        "  \"domain_canonical\": \"...\",\n"
        "  \"anchors\": [\"...\"],\n"
        "  \"related_conditions\": [\"...\"],\n"
        "  \"biomarkers\": [\"...\"],\n"
        "  \"diet_themes\": [\"...\"],\n"
        "  \"retrieval_queries\": [\"...\"]\n"
        "}\n"
        "Kurallar:\n"
        "- 12-20 kısa sorgu üret (tercihen 14-18).\n"
        "- En az 6 sorgu sayısal tetiklesin: günde/haftada/porsiyon/adet/gram/mg/%.\n"
        "- En az 4 sorgu avoid/limit, en az 4 sorgu prefer dilinde olsun.\n"
        "- HER sorgu DOMAIN_HINT kelimesini içermeli.\n"
        "- DOMAIN_POLICY.required_any varsa, her sorgu required_any'den en az 1 terim içermeli.\n"
        "- DOMAIN_POLICY.exclude_if_only terimleri sorguda geçiyorsa o sorguyu ÜRETME.\n"
        "- Yalnızca tek JSON objesi döndür.\n"
    )

def build_domain_understanding_user_payload(domain_hint: str, user_query: str, policy: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "DOMAIN_HINT": domain_hint or "",
            "USER_QUERY": user_query or "",
            "DOMAIN_POLICY": {
                "required_any": policy_required_any(policy),
                "exclude_if_only": policy.get("exclude_if_only", []),
            },
        },
        ensure_ascii=False
    )

def parse_domain_understanding(text: str) -> Dict[str, Any]:
    obj = extract_first_json_object_balanced(text) or {}
    def _list(x):
        return [a.strip() for a in x if isinstance(a, str) and a.strip()] if isinstance(x, list) else []
    return {
        "domain_canonical": (obj.get("domain_canonical") or "").strip(),
        "anchors": _list(obj.get("anchors")),
        "related_conditions": _list(obj.get("related_conditions")),
        "biomarkers": _list(obj.get("biomarkers")),
        "diet_themes": _list(obj.get("diet_themes")),
        "retrieval_queries": _list(obj.get("retrieval_queries")),
    }

def _ensure_domain(domain: str, q: str) -> str:
    q = (q or "").strip()
    if not q:
        return q
    d = (domain or "").strip()
    if not d:
        return q
    if norm_plain(d) not in norm_plain(q):
        q = f"{d} {q}"
    return q

_RULE_GOOD_CUES = re.compile(
    r"(tüketimi\s+nasıl\s+olmalı\?|tuketimi\s+nasil\s+olmali\?|"
    r"en\s+fazla.*olmalı\?|sinirlandirilmali|sınırlandırılmalı|"
    r"kaçınılmalı|kacinilmali|tüketilmemeli|tuketilmemeli)",
    re.IGNORECASE
)

def _canonicalize_to_rule_style(domain: str, q: str) -> str:
    q = (q or "").strip()
    if not q:
        return q
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"
    if _RULE_GOOD_CUES.search(q):
        return q
    return f"{domain} beslenme genel olarak nasıl olmalı, neler tercih edilmeli?"

def _dedup_preserve_order(lst: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in lst:
        n = norm_plain(x)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(x.strip())
    return out

_STOPWORDS_TR = {
    "mi", "mı", "mu", "mü",
    "icin", "için", "ve", "ile", "ya", "yada", "veya",
    "ne", "kadar", "nasil", "nasıl", "olmali", "olmalı",
    "tuketim", "tüketim", "tuketimi", "tüketimi",
    "onerilir", "önerilir", "tercih", "edilmeli", "edilmelidir",
    "tuketilmeli", "tüketilmeli",
    "en", "fazla", "azaltılmalı", "azaltilmali", "sınırlandırılmalı", "sinirlandirilmali",
    "kaçınılmalı", "kacinilmali", "tüketilmemeli", "tuketilmemeli",
}
_WORD_RE = re.compile(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+")

def _tokenize_for_similarity(q: str) -> Set[str]:
    s = norm_plain(q)
    toks = [t for t in _WORD_RE.findall(s) if t and t not in _STOPWORDS_TR and len(t) >= 3]
    return set(toks)

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter) / float(uni) if uni else 0.0

def dedup_near_duplicates(questions: List[str], threshold: float = 0.86) -> List[str]:
    kept: List[str] = []
    kept_tok: List[Set[str]] = []
    for q in questions:
        qn = q.strip()
        if not qn:
            continue
        tset = _tokenize_for_similarity(qn)
        is_dup = False
        for prev_set in kept_tok:
            if _jaccard(tset, prev_set) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(qn)
            kept_tok.append(tset)
    return kept

def sanitize_du_retrieval_queries(domain: str, queries: List[str], policy: Dict[str, Any]) -> List[str]:
    domain = (domain or "").strip()
    if not domain:
        return _dedup_preserve_order([q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()])

    req_any = [norm_plain(x) for x in policy_required_any(policy)]
    if not req_any:
        req_any = [norm_plain(domain)]

    exclude = policy.get("exclude_if_only", [])
    exclude_norm = []
    if isinstance(exclude, list):
        exclude_norm = [norm_plain(x) for x in exclude if isinstance(x, str) and x.strip()]

    out: List[str] = []
    for q in (queries or []):
        if not isinstance(q, str):
            continue
        q = q.strip()
        if not q:
            continue
        qn = norm_plain(q)
        if any(x and x in qn for x in exclude_norm):
            continue

        q = _ensure_domain(domain, q)
        qn2 = norm_plain(q)

        if req_any and not any(r in qn2 for r in req_any):
            first = policy_required_any(policy)
            if first:
                q = f"{domain} {first[0]} {q}".strip()

        q = _canonicalize_to_rule_style(domain, q)
        out.append(q)

    out = _dedup_preserve_order(out)
    out = dedup_near_duplicates(out, threshold=0.86)
    return out

def get_domain_understanding_cached(
    cache: DiskCache,
    model: str,
    num_ctx: int,
    domain: str,
    query: str,
    policy: Dict[str, Any],
    code_fp: str,
    cache_bust: str = ""
) -> Dict[str, Any]:
    key_payload = {
        "rev": "v17_du_with_policy",
        "script_rev": SCRIPT_REV,
        "code_fp": code_fp,
        "model": model,
        "num_ctx": int(num_ctx),
        "domain": domain,
        "query": query,
        "policy": policy or {},
        "cache_bust": cache_bust or ""
    }
    key = "du_" + _sha1(json.dumps(key_payload, ensure_ascii=False, sort_keys=True))
    hit = cache.get(key)
    if isinstance(hit, dict) and isinstance(hit.get("retrieval_queries"), list) and hit["retrieval_queries"]:
        return hit

    du_system = build_domain_understanding_system_prompt()
    du_user = build_domain_understanding_user_payload(domain, query, policy or {})
    du_resp = ollama_chat(model, du_system, du_user, temperature=0.0, num_ctx=int(num_ctx))
    domain_info = parse_domain_understanding(du_resp)

    du_qs = domain_info.get("retrieval_queries") or []
    if isinstance(du_qs, list):
        domain_info["retrieval_queries"] = sanitize_du_retrieval_queries(domain, du_qs, policy or {})
    else:
        domain_info["retrieval_queries"] = []

    cache.set(key, domain_info)
    return domain_info

def build_anchor_tokens(domain: str, info: Dict[str, Any], policy: Dict[str, Any]) -> List[str]:
    tokens: Set[str] = set()
    if domain:
        tokens.add(domain.lower())

    req_any = policy_required_any(policy)
    for t in req_any:
        tokens.add(t.lower())

    for k in ["anchors", "biomarkers", "diet_themes"]:
        for t in info.get(k, []) or []:
            t = (t or "").strip()
            if t:
                tokens.add(t.lower())

    if policy_bool(policy, "allow_related_conditions", False):
        for t in info.get("related_conditions", []) or []:
            t = (t or "").strip()
            if t:
                tokens.add(t.lower())

    allow_related_terms = policy.get("allow_related_terms") or policy.get("allow_related") or []
    if isinstance(allow_related_terms, list):
        for t in allow_related_terms:
            if isinstance(t, str) and t.strip():
                tokens.add(norm_plain(t))

    extra = ["beslenme", "diyet", "öneri", "tavsiye", "kılavuz", "rehber", "akdeniz", "mediterranean", "dash"]
    for e in extra:
        tokens.add(norm_plain(e))

    cleaned = []
    for t in tokens:
        tt = norm_plain(t)
        if len(tt) >= 3:
            cleaned.append(tt)
    return sorted(set(cleaned), key=len, reverse=True)

def build_domain_regex(anchor_tokens: List[str]) -> Optional[re.Pattern]:
    if not anchor_tokens:
        return None
    pat = r"(" + "|".join(re.escape(t) for t in anchor_tokens) + r")"
    return re.compile(pat, re.IGNORECASE)

def contains_any_anchor(text: str, domain_regex: Optional[re.Pattern], anchor_tokens: List[str]) -> bool:
    if not text:
        return False
    if domain_regex and domain_regex.search(text):
        return True
    low = norm_plain(text)
    return any(t in low for t in anchor_tokens)

def contains_required_any(text: str, required_any: List[str]) -> bool:
    if not required_any:
        return True
    low = norm_plain(text)
    for t in required_any:
        if norm_plain(t) in low:
            return True
    return False

def filter_chroma_candidates_strict(
    items: List[Dict[str, Any]],
    drop_noisy: bool,
    max_clinical_hits: int,
    require_domain_name: bool,
    domain_regex: Optional[re.Pattern],
    anchor_tokens: List[str],
    policy: Dict[str, Any],
    other_terms: List[str],
) -> List[Dict[str, Any]]:
    required_any = policy_required_any(policy)
    allow_other = policy_bool(policy, "allow_other_disease_terms", False)

    out: List[Dict[str, Any]] = []
    for it in items:
        text = f"{it.get('ana_baslik','')} {it.get('topic_group','')} {it.get('section','')} {it.get('content','')}".strip()

        if drop_noisy and is_noisy_med_text(text):
            continue
        if max_clinical_hits >= 0 and clinical_setting_hits_text(text) > max_clinical_hits:
            continue

        if required_any and not contains_required_any(text, required_any):
            continue

        has_anchor = contains_any_anchor(text, domain_regex, anchor_tokens)

        if require_domain_name and not has_anchor:
            continue

        if (not allow_other) and (not has_anchor) and contains_other_disease_terms(text, other_terms):
            continue

        if not is_food_actionable_text(text):
            continue

        out.append(it)
    return out


# =============================
# Questions (sebze_meyve kaldırıldı)
# =============================
GENERIC_CATEGORY_SPECS = [
    ("genel", [
        "{domain} için beslenme genel olarak nasıl olmalı, neler tercih edilmeli",
        "{domain} için genel olarak neler sınırlandırılmalı",
    ]),
    ("protein", [
        "{domain} için kırmızı et tüketimi nasıl olmalı",
        "{domain} için beyaz et (tavuk/hindi) tüketimi nasıl olmalı",
        "{domain} için balık tüketimi nasıl olmalı",
        "{domain} için yumurta tüketimi nasıl olmalı",
    ]),
    ("sut", [
        "{domain} için süt ve süt ürünleri tüketimi nasıl olmalı (az yağlı/yağsız tercihi?)",
    ]),
    ("tahil", [
        "{domain} için ekmek/tahıl/pilav-makarna tüketimi nasıl olmalı (tam tahıl vs rafine?)",
    ]),
    ("baklagil", [
        "{domain} için baklagil tüketimi nasıl olmalı (haftada kaç porsiyon?)",
    ]),
    ("yag", [
        "{domain} için doymuş yağ azaltılmalı mı, hangi yağlar tercih edilmeli",
        "{domain} için trans yağdan kaçınılmalı mı, hangi ürünlerde bulunur",
    ]),
    ("posa_lif", [
        "{domain} için posa/lif tüketimi nasıl olmalı (posa alımını artırma önerileri?)",
    ]),
    ("tuz_seker", [
        "{domain} için tuz/sodyum tüketimi nasıl olmalı",
        "{domain} için rafine şeker/şekerli içecek tüketimi nasıl olmalı",
    ]),
]

def build_fast_queries(domain: str, domain_info: Dict[str, Any], max_total: int) -> List[str]:
    qs: List[str] = []
    for _, templates in GENERIC_CATEGORY_SPECS:
        for t in templates:
            qs.append(t.format(domain=domain))

    du_qs = domain_info.get("retrieval_queries") or []
    if isinstance(du_qs, list):
        for x in du_qs:
            if isinstance(x, str) and x.strip():
                qs.append(x.strip())

    qs = [_canonicalize_to_rule_style(domain, _ensure_domain(domain, q)) for q in qs]
    qs = _dedup_preserve_order(qs)
    qs = dedup_near_duplicates(qs, threshold=0.86)
    return qs[:max_total] if max_total > 0 else qs


# =============================
# Query Boosters
# =============================
PREFER_BOOST = " önerilir tercih edilmeli tüketilmeli artırılmalı tavsiye uygundur"
LIMIT_BOOST = " en fazla sınırlandırılmalı azaltılmalı kısıtlanmalı sınır yeterlidir"
AVOID_BOOST = " kaçınılmalı tüketilmemeli yasak kesinlikle kaçın"

def apply_query_booster(q: str) -> str:
    qn = norm_plain(q)
    if ("kacin" in qn) or ("kaçin" in qn) or ("tuketilmemeli" in qn) or ("yasak" in qn) or ("uzak dur" in qn):
        return q + AVOID_BOOST
    if ("en fazla" in qn) or ("sinir" in qn) or ("sınır" in qn) or ("azalt" in qn) or ("kisit" in qn) or ("kısıt" in qn) or ("sinirlandir" in qn) or ("yeterlidir" in qn):
        return q + LIMIT_BOOST
    if ("oner" in qn) or ("öner" in qn) or ("tercih" in qn) or ("tuketilmeli" in qn) or ("tüketilmeli" in qn) or ("artir" in qn) or ("artır" in qn):
        return q + PREFER_BOOST
    return q + " önerilir"


# =============================
# Atomic snippets (regex splitter)
# =============================
_BULLET_SPLIT_RE = re.compile(r"(?:\n\s*[•\-\*]\s+)")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")

BULLET_PRESENT_RE = re.compile(
    r"(^|\n)\s*(?:[•\-\*]\s+|\d+\s*[\)\.\-]\s+|[a-zA-ZçğıöşüÇĞİÖŞÜ]\s*[\)\.\-]\s+)",
    re.MULTILINE
)

FREQ_CUE_RE = re.compile(r"\b(gunde|günde|haftada|ayda|en\s*az|en\s*fazla)\b", re.IGNORECASE)
_MULTI_RULE_NUM_RE = re.compile(r"(\d+\s*-\s*\d+|\d+)\s*(porsiyon|adet|kez|g|gr|gram|mg)\b", re.IGNORECASE)

def split_to_atomic_snippets(text: str, max_chars: int = 260) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    parts: List[str] = []
    raw_bullets = _BULLET_SPLIT_RE.split("\n" + t)
    raw_bullets = [x.strip() for x in raw_bullets if x and x.strip()]

    for b in raw_bullets:
        sents = _SENT_SPLIT_RE.split(b)
        for s in sents:
            s = s.strip()
            if len(s) < 25:
                continue
            if max_chars and len(s) > max_chars:
                s = s[:max_chars].rstrip()
            parts.append(s)

    seen = set()
    out = []
    for p in parts:
        n = norm_plain(p)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(p)
    return out

def has_bullets_or_lists(text: str) -> bool:
    return bool(BULLET_PRESENT_RE.search(text or ""))

def looks_like_multi_rule_chunk(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 180:
        return False
    if len(_MULTI_RULE_NUM_RE.findall(t)) >= 2:
        return True
    comma_count = t.count(",")
    low = norm_plain(t)
    conj_hits = sum(1 for w in ["ayrica", "ayrıca", "ile", "ve", "haftada", "gunde", "günde", "en az", "en fazla"] if w in low)
    if comma_count >= 4 and conj_hits >= 3:
        return True
    cue_hits = sum(1 for w in ["tuketilmesi", "tüketilmesi", "onerilmektedir", "önerilmektedir", "tercihen", "en az", "en fazla"] if w in low)
    if cue_hits >= 3 and len(t) >= 240:
        return True
    return False

def should_llm_split_when_no_bullet(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if has_bullets_or_lists(t):
        return False
    if len(FREQ_CUE_RE.findall(t)) >= 2:
        return True
    if len(_MULTI_RULE_NUM_RE.findall(t)) >= 2:
        return True
    return looks_like_multi_rule_chunk(t)

def build_chunk_splitter_system_prompt() -> str:
    return (
        "Sen klinik diyetisyen metinlerini ATOMİK KURAL parçalarına bölen bir asistansın.\n"
        "Sana verilen metin içinde birden fazla öneri/kural olabilir.\n"
        "Görev: metni, her biri TEK bir öneri/kural içeren kısa parçalara ayır.\n"
        "ÇIKTI: SADECE JSON döndür: { \"snippets\": [\"...\", ...] }\n"
        "Kurallar:\n"
        "- Her snippet tek bir kural/öneri içersin.\n"
        "- Sayısal hedefleri koru.\n"
        "- Yeni bilgi ekleme.\n"
        "- 80-260 karakter arası.\n"
        "- Metin zaten tek kural ise tek eleman.\n"
    )

def llm_split_chunk_into_snippets_cached(
    cache: DiskCache,
    model: str,
    num_ctx: int,
    chunk_text: str,
    code_fp: str,
    cache_bust: str = "",
    temperature: float = 0.0
) -> List[str]:
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return []

    key_payload = {
        "rev": "v17_splitter_cached",
        "script_rev": SCRIPT_REV,
        "code_fp": code_fp,
        "model": model,
        "num_ctx": int(num_ctx),
        "chunk_sha1": _sha1(chunk_text),
        "cache_bust": cache_bust or "",
    }
    key = "split_" + _sha1(json.dumps(key_payload, ensure_ascii=False, sort_keys=True))
    hit = cache.get(key)
    if isinstance(hit, list) and hit:
        return [str(x).strip() for x in hit if isinstance(x, str) and x.strip()]

    sys = build_chunk_splitter_system_prompt()
    user = json.dumps({"TEXT": chunk_text}, ensure_ascii=False)
    resp = ollama_chat(model, sys, user, temperature=float(temperature), num_ctx=int(num_ctx))
    obj = extract_first_json_object_balanced(resp) or {}
    snippets = obj.get("snippets") if isinstance(obj, dict) else None

    out: List[str] = []
    if isinstance(snippets, list):
        for s in snippets:
            if isinstance(s, str) and s.strip():
                out.append(s.strip())
    if not out:
        out = [chunk_text]

    cleaned = []
    seen = set()
    for s in out:
        s = (s or "").strip()
        if len(s) < 25:
            continue
        n = norm_plain(s)
        if n in seen:
            continue
        seen.add(n)
        cleaned.append(s)
    if not cleaned:
        cleaned = [chunk_text]

    cache.set(key, cleaned)
    return cleaned


# =============================
# LLM RULE EXTRACTION PROMPT
# =============================
def build_system_prompt() -> str:
    return (
        "SEN UZMAN BİR DİYETİSYEN VE BİLGİ ÇIKARIM (INFORMATION EXTRACTION) UZMANISIN.\n"
        "Görevin: RAG_SNIPPETS içinde geçen beslenme ile ilgili TÜM KURALLARI, ayrı ayrı ve eksiksiz şekilde JSON’a dönüştürmektir.\n\n"

        "GENEL KURALLAR:\n"
        "- SADECE JSON döndür (açıklama, markdown, düz metin YOK).\n"
        "- RAG_SNIPPETS’ta AÇIKÇA GEÇMEYEN HİÇBİR bilgiyi ekleme.\n"
        "- Çıkarım, tahmin, normalizasyon, yuvarlama, uydurma YASAK.\n"
        "- Bir tag kullanıyorsan rag_evidence içinde o tag ile eşleşen DOĞRUDAN QUOTE ZORUNLUDUR.\n"
        "- TAG seçimi SADECE TAG_VOCAB içinden yapılır.\n"
        "- TAG_VOCAB dışında tag üretmek KESİNLİKLE YASAKTIR.\n\n"

        "CÜMLE PARÇALAMA (ZORUNLU):\n"
        "- Uzun ve bileşik cümleleri anlamlı parçalara böl.\n"
        "- Tek bir cümleden birden fazla KURAL çıkarabilirsin.\n\n"

        "KURAL TİPLERİ:\n"
        "1) HARD (UYGULANACAK) KURALLAR:\n"
        "- Sayı, miktar, sıklık, oran (%) içeriyorsa HARD kuraldır.\n"
        "- HARD kurallar MUTLAKA numeric_constraints alanına yazılır.\n"
        "- 'en az', 'en fazla', 'üst sınır', 'alt sınır', '%', 'gram', 'mg', 'kez' içeren ifadeler HARD kabul edilir.\n\n"

        "2) SOFT (ÖNERİ / YÖNLENDİRME) KURALLAR:\n"
        "- Sayı içermeyen tercihler ve ikame önerileri recommendations alanına yazılır.\n\n"

        "SINIFLAMA (prefer / limit / avoid) — SADECE TAG ETİKETİ:\n"
        "- 'kaçınılmalı / tüketilmemeli / yasak / uzak durulmalıdır' => avoid\n"
        "- 'sınırlandırılmalı / azaltılmalı / kısıtlanmalı / en fazla' => limit\n"
        "- 'önerilir / tercih edilmeli / tüketilmeli / artırılmalı / yerine seç' => prefer\n\n"

        "ZORUNLU BAĞLANTI KURALI:\n"
        "- Eğer bir tag numeric_constraints içinde yer alıyorsa, o tag MUTLAKA limit_tags içinde de bulunmalıdır.\n\n"

        "ÇAKIŞMA KURALI:\n"
        "- Aynı tag prefer_tags, limit_tags, avoid_tags içinde birden fazla yerde YER ALAMAZ.\n"
        "- Öncelik sırası: avoid > limit > prefer\n\n"

        "HARD POLICY (ASLA İHLAL EDİLEMEZ):\n"
        "- trans_yag_riski, islenmis_et, doymus_yag_yuksek ASLA prefer OLAMAZ.\n"
        "- sebze, meyve, lif_yuksek, posa_yuksek, lifli_beslenme ASLA avoid OLAMAZ.\n\n"

        "NUMERIC_CONSTRAINTS KURALLARI (KRİTİK):\n"
        "- Metinde SAYI veya ORAN (%) varsa numeric_constraints ÜRETMEK ZORUNLUDUR.\n"
        "- Ölçü birimi şemaya uymasa bile numeric_constraints üret.\n"
        "- Ölçü birimi net değilse unit='unknown' kullan.\n"
        "- '%' içeren ifadelerde unit='yuzde' kullan ve sayıyı description alanında AYNEN yaz.\n"
        "- period_days:\n"
        "    - günde / günlük => 1\n"
        "    - haftada / haftalık => 7\n"
        "    - ayda / aylık => 30\n"
        "- Period açıkça belirtilmemişse period_days=null bırak (numeric_constraints YİNE DE ÜRET).\n\n"

        "UNIT DEĞERLERİ (SADECE BUNLAR):\n"
        "- kez | porsiyon | adet | gram | mg | yuzde | cay_kasigi | tatli_kasigi | yemek_kasigi | dilim | bardak | unknown\n\n"

        "RECOMMENDATIONS ŞEMASI:\n"
        "{ \"tag\":\"...\", \"intent\":\"prefer|avoid|choose_instead\", \"description\":\"...\", \"quote\":\"...\", \"chunk_id\":\"...\" }\n\n"

        "LOGICAL_RULES:\n"
        "- prefer_tags, limit_tags, avoid_tags içine aldığın HER tag logical_rules içinde de AYNI sınıfta yer almalıdır.\n\n"

        "KANIT (rag_evidence):\n"
        "- Ürettiğin HER tag için rag_evidence eklemek ZORUNLUDUR.\n"
        "- related_tags tek elemanlı bir LİSTE olmalıdır.\n\n"

        "ÇIKTI ŞEMASI (AYNEN UYGULA):\n"
        "{\n"
        "  \"prefer_tags\": [],\n"
        "  \"limit_tags\": [],\n"
        "  \"avoid_tags\": [],\n"
        "  \"meal_pattern_rules\": {\n"
        "    \"logical_rules\": {\"prefer\": [], \"limit\": [], \"avoid\": []},\n"
        "    \"numeric_constraints\": [\n"
        "      {\n"
        "        \"tag\": \"...\",\n"
        "        \"min_count\": null,\n"
        "        \"max_count\": null,\n"
        "        \"period_days\": null,\n"
        "        \"min_grams\": null,\n"
        "        \"max_grams\": null,\n"
        "        \"unit\": \"...\",\n"
        "        \"description\": \"...\"\n"
        "      }\n"
        "    ]\n"
        "  },\n"
        "  \"energy_rules\": {\"scale_up_order\": [], \"scale_down_order\": [], \"locks\": []},\n"
        "  \"recommendations\": [],\n"
        "  \"rag_evidence\": []\n"
        "}\n\n"

        "ÖRNEKLER (DAVRANIŞ GÖSTERİMİ):\n\n"

        "ÖRNEK 1:\n"
        "RAG_SNIPPET:\n"
        "\"Haftada en fazla 2 kez kırmızı et tüketilmelidir.\"\n\n"
        "ÇIKTI DAVRANIŞI:\n"
        "- kirmizi_et => limit_tags\n"
        "- numeric_constraints ÜRET\n"
        "- max_count=2, period_days=7, unit=kez\n\n"

        "ÖRNEK 2:\n"
        "RAG_SNIPPET:\n"
        "\"Doymuş yağ alımı günlük enerjinin %10’unu geçmemelidir.\"\n\n"
        "ÇIKTI DAVRANIŞI:\n"
        "- doymus_yag_yuksek => limit_tags\n"
        "- numeric_constraints ÜRET\n"
        "- unit=yuzde, description içinde \"%10\" AYNEN yaz\n\n"

        "ÖRNEK 3:\n"
        "RAG_SNIPPET:\n"
        "\"Günde 1 tepeleme çay kaşığından fazla tuz alınmamalıdır.\"\n\n"
        "ÇIKTI DAVRANIŞI:\n"
        "- tuz => limit_tags\n"
        "- numeric_constraints ÜRET\n"
        "- max_count=1, period_days=1, unit=cay_kasigi\n\n"

        "ÖRNEK 4:\n"
        "RAG_SNIPPET:\n"
        "\"Sebze tüketimi artırılmalıdır.\"\n\n"
        "ÇIKTI DAVRANIŞI:\n"
        "- sebze => prefer_tags\n"
        "- numeric_constraints ÜRETME\n"
        "- recommendations içine yaz\n"
    )


def build_user_payload(
    domain: str,
    query: str,
    tag_vocab: List[str],
    tag_meanings: Dict[str, str],
    snippets: List[Dict[str, Any]],
    send_tag_meanings: bool
) -> str:
    payload: Dict[str, Any] = {
        "domain": domain,
        "query": query,
        "TAG_VOCAB": tag_vocab,
        "RAG_SNIPPETS": snippets,
    }
    if send_tag_meanings:
        compact = {k: (v[:140] + "…") if len(v) > 141 else v for k, v in tag_meanings.items()}
        payload["TAG_MEANINGS"] = compact
    return json.dumps(payload, ensure_ascii=False)


# =============================
# TEMPLATE + NORMALIZE
# =============================
ALLOWED_TOP_KEYS = {
    "dataset", "disease",
    "prefer_tags", "limit_tags", "avoid_tags",
    "meal_pattern_tags",
    "disease_food_tag_descriptions",
    "meal_pattern_tag_descriptions",
    "meal_pattern_rules",
    "energy_rules",
    "rag_evidence",
    "new_tags",
    "retrieval_results_raw",
    "retrieval_results_strict",
    "presentable_evidence",
}

def build_empty_template(domain: str) -> Dict[str, Any]:
    return {
        "dataset": f"disease_rules_{domain}",
        "disease": {"disease_id": domain.upper(), "name": domain},
        "prefer_tags": [],
        "limit_tags": [],
        "avoid_tags": [],
        "meal_pattern_tags": [],
        "disease_food_tag_descriptions": {},
        "meal_pattern_tag_descriptions": {},
        "meal_pattern_rules": {"logical_rules": {"prefer": [], "limit": [], "avoid": []}, "numeric_constraints": []},
        "energy_rules": {"scale_up_order": [], "scale_down_order": [], "locks": []},
        "rag_evidence": [],
        "new_tags": [],
        "retrieval_results_raw": [],
        "retrieval_results_strict": [],
        "presentable_evidence": [],
    }

def normalize_string_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for a in x:
        if isinstance(a, str):
            s = a.strip()
            if s:
                out.append(s)
    seen = set()
    res = []
    for t in out:
        if t not in seen:
            seen.add(t)
            res.append(t)
    return res

def prune_to_template_keys(out: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    pruned = {k: out[k] for k in out.keys() if k in ALLOWED_TOP_KEYS}
    for k, v in template.items():
        if k not in pruned:
            pruned[k] = v
    return pruned


# =============================
# TAG VOCAB (sebze_meyve kaldırıldı)
# =============================
RULE_USEFUL_TAGS: Set[str] = {
    "gi_dusuk", "gi_orta", "gi_yuksek",
    "kh_dusuk", "kh_orta", "kh_yuksek",
    "kompleks_kh",
    "lif_yuksek", "lif_orta", "lif_dusuk", "posa_yuksek", "lifli_beslenme",
    "sodyum_dusuk", "sodyum_orta", "sodyum_yuksek",
    "doymus_yag_dusuk", "doymus_yag_orta", "doymus_yag_yuksek", "trans_yag_riski",
    "zeytinyagi", "tekli_doymamis", "coklu_doymamis",
    "balik", "omega3", "beyaz_et", "kirmizi_et",
    "islenmis_et", "et_suyu",
    "tahil", "bulgur", "ekmek", "baklagil", "baklagil_yemegi", "pilav_makarna",
    "sebze", "meyve", "kuruyemis",
    "sut_urunu", "yagsiz_sut_urunleri", "peynir",
    "kolesterol", "yumurta", "sakatat",
    "tam_yagli_sut_urunleri", "az_yagli_sut_urunleri",
}

def load_tag_vocab(tag_meanings_all: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
    vocab_useful = set(tag_meanings_all.keys()).intersection(RULE_USEFUL_TAGS)
    if not vocab_useful:
        vocab_useful = set(tag_meanings_all.keys())
    tag_vocab = list(sorted(vocab_useful))
    tag_meanings = {t: tag_meanings_all.get(t, "") for t in tag_vocab}
    return tag_vocab, tag_meanings


# =============================
# MERGE HELPERS
# =============================
def merge_numeric_constraints(base_list: Any, new_list: Any) -> List[dict]:
    if not isinstance(base_list, list):
        base_list = []
    if not isinstance(new_list, list):
        return base_list
    seen = set()
    out: List[dict] = []

    def key_of(x: dict):
        return (
            x.get("tag"),
            x.get("min_count"),
            x.get("max_count"),
            x.get("period_days"),
            x.get("min_grams"),
            x.get("max_grams"),
            x.get("unit"),
            x.get("description"),
        )

    for x in base_list:
        if isinstance(x, dict):
            k = key_of(x)
            if k not in seen:
                out.append(x)
                seen.add(k)
    for x in new_list:
        if isinstance(x, dict):
            k = key_of(x)
            if k not in seen:
                out.append(x)
                seen.add(k)
    return out

def ensure_energy_rules(out: Dict[str, Any]) -> None:
    if "energy_rules" not in out or not isinstance(out["energy_rules"], dict):
        out["energy_rules"] = {"scale_up_order": [], "scale_down_order": [], "locks": []}
    er = out["energy_rules"]
    er.setdefault("scale_up_order", [])
    er.setdefault("scale_down_order", [])
    er.setdefault("locks", [])

POLICY_TAGS: Set[str] = {"sodyum_dusuk", "sodyum_orta", "sodyum_yuksek"}

def compute_energy_rules(out: Dict[str, Any]) -> None:
    ensure_energy_rules(out)

    prefer = normalize_string_list(out.get("prefer_tags"))
    limit = normalize_string_list(out.get("limit_tags"))
    avoid = normalize_string_list(out.get("avoid_tags"))

    mpr = out.get("meal_pattern_rules") or {}
    lr = (mpr.get("logical_rules") or {}) if isinstance(mpr, dict) else {}

    prefer = normalize_string_list(list(set(prefer) | set(normalize_string_list(lr.get("prefer")))))
    limit = normalize_string_list(list(set(limit) | set(normalize_string_list(lr.get("limit")))))
    avoid = normalize_string_list(list(set(avoid) | set(normalize_string_list(lr.get("avoid")))))

    prefer = [t for t in prefer if t not in POLICY_TAGS]

    out["energy_rules"]["scale_up_order"] = prefer
    out["energy_rules"]["scale_down_order"] = normalize_string_list(avoid + limit)
    out["energy_rules"]["locks"] = []

def merge_llm_into_template(template: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(template)
    if isinstance(decision, dict):
        for k in ["prefer_tags", "limit_tags", "avoid_tags", "meal_pattern_rules", "rag_evidence", "energy_rules"]:
            if k in decision:
                out[k] = decision.get(k)

    if "meal_pattern_rules" not in out or not isinstance(out["meal_pattern_rules"], dict):
        out["meal_pattern_rules"] = {"logical_rules": {"prefer": [], "limit": [], "avoid": []}, "numeric_constraints": []}
    if "rag_evidence" not in out or not isinstance(out["rag_evidence"], list):
        out["rag_evidence"] = []
    ensure_energy_rules(out)
    return out

def merge_decisions(base_out: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(decision, dict):
        return base_out

    for k in ["prefer_tags", "limit_tags", "avoid_tags"]:
        base_out[k] = normalize_string_list(
            list(set(normalize_string_list(base_out.get(k))) | set(normalize_string_list(decision.get(k))))
        )

    d_ev = decision.get("rag_evidence") or []
    if isinstance(d_ev, list):
        b_ev = base_out.setdefault("rag_evidence", [])
        seen = set((e.get("chunk_id"), e.get("quote"), tuple(e.get("related_tags", []))) for e in b_ev if isinstance(e, dict))
        for e in d_ev:
            if not isinstance(e, dict):
                continue
            key = (e.get("chunk_id"), e.get("quote"), tuple(e.get("related_tags", [])))
            if key not in seen:
                b_ev.append(e)
                seen.add(key)

    if isinstance(decision.get("meal_pattern_rules"), dict):
        b = base_out.setdefault("meal_pattern_rules", {"logical_rules": {"prefer": [], "limit": [], "avoid": []}, "numeric_constraints": []})
        d = decision["meal_pattern_rules"]

        if isinstance(d.get("logical_rules"), dict):
            b_lr = b.setdefault("logical_rules", {"prefer": [], "limit": [], "avoid": []})
            d_lr = d["logical_rules"]
            for cls in ["prefer", "limit", "avoid"]:
                merged = set(normalize_string_list(b_lr.get(cls))) | set(normalize_string_list(d_lr.get(cls)))
                b_lr[cls] = normalize_string_list(list(merged))

        b["numeric_constraints"] = merge_numeric_constraints(b.get("numeric_constraints"), d.get("numeric_constraints"))

    if isinstance(decision.get("energy_rules"), dict):
        ensure_energy_rules(base_out)
        er = base_out["energy_rules"]
        der = decision["energy_rules"]
        er["scale_up_order"] = normalize_string_list(list(set(er.get("scale_up_order", [])) | set(normalize_string_list(der.get("scale_up_order")))))
        er["scale_down_order"] = normalize_string_list(list(set(er.get("scale_down_order", [])) | set(normalize_string_list(der.get("scale_down_order")))))
        er["locks"] = normalize_string_list(list(set(er.get("locks", [])) | set(normalize_string_list(der.get("locks")))))
    return base_out


# =============================
# POLICY / VALIDATOR / FIXER
# =============================
NEVER_PREFER_TAGS: Set[str] = {"trans_yag_riski", "islenmis_et", "doymus_yag_yuksek"}
NEVER_AVOID_TAGS: Set[str] = {"sebze", "meyve", "lif_yuksek", "posa_yuksek", "lifli_beslenme"}
AVOID_RESTRICTED_TAGS: Set[str] = {"kirmizi_et", "yumurta", "kuruyemis"}

def enforce_exclusivity(out: Dict[str, Any]) -> None:
    prefer = set(normalize_string_list(out.get("prefer_tags")))
    limit = set(normalize_string_list(out.get("limit_tags")))
    avoid = set(normalize_string_list(out.get("avoid_tags")))

    prefer -= (avoid | limit)
    limit -= avoid

    out["prefer_tags"] = sorted(prefer)
    out["limit_tags"] = sorted(limit)
    out["avoid_tags"] = sorted(avoid)

    mpr = out.get("meal_pattern_rules") or {}
    lr = (mpr.get("logical_rules") or {}) if isinstance(mpr, dict) else {}
    p = set(normalize_string_list(lr.get("prefer")))
    l = set(normalize_string_list(lr.get("limit")))
    a = set(normalize_string_list(lr.get("avoid")))

    p -= (a | l)
    l -= a
    if isinstance(mpr, dict):
        mpr.setdefault("logical_rules", {})
        mpr["logical_rules"]["prefer"] = sorted(p)
        mpr["logical_rules"]["limit"] = sorted(l)
        mpr["logical_rules"]["avoid"] = sorted(a)
        out["meal_pattern_rules"] = mpr

def enforce_hard_policy(out: Dict[str, Any], tag_vocab_set: Set[str], policy: Dict[str, Any]) -> None:
    never_prefer = set(NEVER_PREFER_TAGS) | set(policy_list(policy, "never_prefer"))
    never_avoid = set(NEVER_AVOID_TAGS) | set(policy_list(policy, "never_avoid"))

    prefer = set(normalize_string_list(out.get("prefer_tags")))
    limit = set(normalize_string_list(out.get("limit_tags")))
    avoid = set(normalize_string_list(out.get("avoid_tags")))

    for t in list(prefer):
        if t in never_prefer:
            prefer.remove(t)
            if t in tag_vocab_set:
                limit.add(t)
    for t in list(avoid):
        if t in never_avoid:
            avoid.remove(t)
            if t in tag_vocab_set:
                prefer.add(t)

    out["prefer_tags"] = sorted(prefer)
    out["limit_tags"] = sorted(limit)
    out["avoid_tags"] = sorted(avoid)

    mpr = out.get("meal_pattern_rules") or {}
    lr = (mpr.get("logical_rules") or {}) if isinstance(mpr, dict) else {}
    p = set(normalize_string_list(lr.get("prefer")))
    l = set(normalize_string_list(lr.get("limit")))
    a = set(normalize_string_list(lr.get("avoid")))

    for t in list(p):
        if t in never_prefer:
            p.remove(t)
            l.add(t)
    for t in list(a):
        if t in never_avoid:
            a.remove(t)
            p.add(t)

    if isinstance(mpr, dict):
        mpr.setdefault("logical_rules", {})
        mpr["logical_rules"]["prefer"] = sorted(p)
        mpr["logical_rules"]["limit"] = sorted(l)
        mpr["logical_rules"]["avoid"] = sorted(a)
        out["meal_pattern_rules"] = mpr

def keep_only_vocab_tags(out: Dict[str, Any], tag_vocab_set: Set[str]) -> None:
    out["prefer_tags"] = [t for t in normalize_string_list(out.get("prefer_tags")) if t in tag_vocab_set]
    out["limit_tags"] = [t for t in normalize_string_list(out.get("limit_tags")) if t in tag_vocab_set]
    out["avoid_tags"] = [t for t in normalize_string_list(out.get("avoid_tags")) if t in tag_vocab_set]

    mpr = out.get("meal_pattern_rules") or {}
    lr = (mpr.get("logical_rules") or {}) if isinstance(mpr, dict) else {}
    if isinstance(mpr, dict):
        lr.setdefault("prefer", [])
        lr.setdefault("limit", [])
        lr.setdefault("avoid", [])
        lr["prefer"] = [t for t in normalize_string_list(lr.get("prefer")) if t in tag_vocab_set]
        lr["limit"] = [t for t in normalize_string_list(lr.get("limit")) if t in tag_vocab_set]
        lr["avoid"] = [t for t in normalize_string_list(lr.get("avoid")) if t in tag_vocab_set]
        mpr["logical_rules"] = lr
        out["meal_pattern_rules"] = mpr

def validate_and_fix(out: Dict[str, Any], tag_vocab: List[str], policy: Dict[str, Any]) -> None:
    tag_vocab_set = set(tag_vocab)
    keep_only_vocab_tags(out, tag_vocab_set)
    enforce_exclusivity(out)
    enforce_hard_policy(out, tag_vocab_set, policy)
    enforce_exclusivity(out)
    keep_only_vocab_tags(out, tag_vocab_set)


# =============================
# rag_evidence + numeric sanitize + evidence-backed tags
# (AŞAĞISI: senin orijinal kodunla aynı mantıkta; burada kısaltmadım, eksiksiz kalsın diye)
# =============================
ALLOWED_NUM_UNITS = {"kez", "porsiyon", "adet", "gram", "mg"}

def sanitize_rag_evidence(out: Dict[str, Any], tag_vocab_set: Set[str], valid_chunk_ids: Set[str]) -> None:
    ev = out.get("rag_evidence")
    if not isinstance(ev, list):
        out["rag_evidence"] = []
        return
    cleaned = []
    seen = set()
    for e in ev:
        if not isinstance(e, dict):
            continue
        cid = (e.get("chunk_id") or "").strip()
        quote = (e.get("quote") or "").strip()
        rtags = e.get("related_tags") or []
        if not isinstance(rtags, list):
            rtags = []
        rtags = [t for t in rtags if isinstance(t, str) and t.strip()]
        if not cid or cid == "..." or not quote or quote == "...":
            continue
        if valid_chunk_ids and cid not in valid_chunk_ids:
            continue
        if not rtags:
            continue
        tag = rtags[0].strip()
        if tag not in tag_vocab_set:
            continue
        key = (cid, quote, tag)
        if key in seen:
            continue
        cleaned.append({"chunk_id": cid, "related_tags": [tag], "quote": quote})
        seen.add(key)
    out["rag_evidence"] = cleaned


RE_EVERY_N_DAYS = re.compile(r"(\d+)\s*g(ü|u)n(de)?\s*bir", re.IGNORECASE)
RE_GUN_ASIRI = re.compile(r"g(ü|u)n\s*aş(ı|i)r(ı|i)", re.IGNORECASE)

def _infer_period_days_from_text(desc: str) -> Optional[int]:
    d = norm_plain(desc)
    if RE_GUN_ASIRI.search(desc or ""):
        return 2
    m = RE_EVERY_N_DAYS.search(desc or "")
    if m:
        try:
            n = int(m.group(1))
            if n >= 1:
                return n
        except Exception:
            pass
    if "hafta" in d:
        return 7
    if re.search(r"\bayda\b|\baylik\b|\baylık\b", d):
        return 30
    if "gunde" in d or "günde" in d or "gunluk" in d or "günlük" in d:
        return 1
    return None

RE_GRAM_MIN = re.compile(r"(?:en\s*az)\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gram)\b", re.IGNORECASE)
RE_GRAM_MAX = re.compile(r"(?:en\s*fazla)\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gram)\b", re.IGNORECASE)
RE_GRAM_ANY = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(g|gr|gram)\b", re.IGNORECASE)

def _to_float_safe(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace(",", ".")
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def _to_int_safe(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            if v.is_integer():
                return int(v)
            return None
        s = str(v).strip()
        if not s:
            return None
        return int(float(s.replace(",", ".")))
    except Exception:
        return None

def _extract_grams_from_desc(desc: str) -> Tuple[Optional[float], Optional[float]]:
    if not desc:
        return None, None
    t = desc
    mn = None
    mx = None
    m = RE_GRAM_MIN.search(t)
    if m:
        mn = _to_float_safe(m.group(1))
    m = RE_GRAM_MAX.search(t)
    if m:
        mx = _to_float_safe(m.group(1))
    if mn is None and mx is None:
        m2 = RE_GRAM_ANY.search(t)
        if m2:
            mn = _to_float_safe(m2.group(1))
    return mn, mx

def sanitize_numeric_constraints(out: Dict[str, Any], tag_vocab_set: Set[str]) -> None:
    mpr = out.get("meal_pattern_rules")
    if not isinstance(mpr, dict):
        out["meal_pattern_rules"] = {"logical_rules": {"prefer": [], "limit": [], "avoid": []}, "numeric_constraints": []}
        return

    ncs = mpr.get("numeric_constraints")
    if not isinstance(ncs, list):
        mpr["numeric_constraints"] = []
        out["meal_pattern_rules"] = mpr
        return

    cleaned = []
    seen = set()

    for x in ncs:
        if not isinstance(x, dict):
            continue

        tag = (x.get("tag") or "").strip()
        if tag not in tag_vocab_set:
            continue

        desc = (x.get("description") or "").strip()
        unit = (x.get("unit") or "").strip().lower()

        if "|" in unit:
            unit = ""

        min_count = x.get("min_count", None)
        max_count = x.get("max_count", None)
        period_days = x.get("period_days", None)
        min_grams = x.get("min_grams", None)
        max_grams = x.get("max_grams", None)

        min_count = _to_int_safe(min_count) if min_count is not None else None
        max_count = _to_int_safe(max_count) if max_count is not None else None
        period_days = _to_int_safe(period_days) if period_days is not None else None
        min_grams = _to_float_safe(min_grams) if min_grams is not None else None
        max_grams = _to_float_safe(max_grams) if max_grams is not None else None

        if not desc:
            desc = "Kaynak metindeki sayısal kural."

        if not unit:
            if min_grams is not None or max_grams is not None:
                unit = "gram"
            elif min_count is not None or max_count is not None:
                dl = norm_plain(desc)
                if "porsiyon" in dl:
                    unit = "porsiyon"
                elif "adet" in dl:
                    unit = "adet"
                else:
                    unit = "kez"

        if unit in ["g", "gr"]:
            unit = "gram"

        if unit == "gram":
            dmn, dmx = _extract_grams_from_desc(desc)
            if min_grams is None and dmn is not None:
                min_grams = dmn
            if max_grams is None and dmx is not None:
                max_grams = dmx

            if min_grams is None and min_count is not None:
                min_grams = float(min_count)
            if max_grams is None and max_count is not None:
                max_grams = float(max_count)

            min_count = None
            max_count = None

            if period_days is None:
                period_days = _infer_period_days_from_text(desc)

        if unit not in ALLOWED_NUM_UNITS:
            continue

        if (unit in ["kez", "porsiyon", "adet"]) and (min_count is not None or max_count is not None):
            if period_days is None:
                period_days = _infer_period_days_from_text(desc)
            if period_days is None:
                continue

        if unit == "gram":
            if min_grams is None and max_grams is None:
                continue

        obj = {
            "tag": tag,
            "min_count": min_count,
            "max_count": max_count,
            "period_days": period_days,
            "min_grams": min_grams,
            "max_grams": max_grams,
            "unit": unit,
            "description": desc,
        }

        if not any(v is not None for v in [obj["min_count"], obj["max_count"], obj["min_grams"], obj["max_grams"]]):
            continue

        key = (obj["tag"], obj["min_count"], obj["max_count"], obj["period_days"], obj["min_grams"], obj["max_grams"], obj["unit"], obj["description"])
        if key in seen:
            continue
        cleaned.append(obj)
        seen.add(key)

    mpr["numeric_constraints"] = cleaned
    out["meal_pattern_rules"] = mpr

def resolve_numeric_conflicts(out: Dict[str, Any]) -> None:
    mpr = out.get("meal_pattern_rules") or {}
    ncs = mpr.get("numeric_constraints") or []
    if not isinstance(ncs, list):
        return

    bucket: Dict[Tuple[str, Optional[int], str], Dict[str, Any]] = {}
    for x in ncs:
        if not isinstance(x, dict):
            continue
        tag = x.get("tag")
        per = x.get("period_days")
        unit = x.get("unit")
        if not tag or not unit:
            continue
        key = (tag, per, unit)
        cur = bucket.get(key)
        if cur is None:
            bucket[key] = dict(x)
            continue

        for f in ["min_count", "min_grams"]:
            if x.get(f) is not None:
                cur[f] = max(cur.get(f) or x[f], x[f]) if cur.get(f) is not None else x[f]
        for f in ["max_count", "max_grams"]:
            if x.get(f) is not None:
                cur[f] = min(cur.get(f) or x[f], x[f]) if cur.get(f) is not None else x[f]

        d1 = (cur.get("description") or "").strip()
        d2 = (x.get("description") or "").strip()
        if d2 and d2 not in d1:
            cur["description"] = (d1 + " | " + d2).strip(" |")
        bucket[key] = cur

    mpr["numeric_constraints"] = list(bucket.values())
    out["meal_pattern_rules"] = mpr

def promote_numeric_tags_to_limit(out: Dict[str, Any]) -> None:
    mpr = out.get("meal_pattern_rules") or {}
    ncs = mpr.get("numeric_constraints") or []
    if not isinstance(ncs, list):
        return

    numeric_tags = set()
    for x in ncs:
        if isinstance(x, dict) and isinstance(x.get("tag"), str) and x["tag"].strip():
            numeric_tags.add(x["tag"].strip())

    if not numeric_tags:
        return

    prefer = set(normalize_string_list(out.get("prefer_tags")))
    limit = set(normalize_string_list(out.get("limit_tags")))
    avoid = set(normalize_string_list(out.get("avoid_tags")))

    for t in numeric_tags:
        if t in avoid:
            continue
        prefer.discard(t)
        limit.add(t)

    out["prefer_tags"] = sorted(prefer)
    out["limit_tags"] = sorted(limit)
    out["avoid_tags"] = sorted(avoid)

    lr = ((mpr.get("logical_rules") or {}) if isinstance(mpr, dict) else {})
    p = set(normalize_string_list(lr.get("prefer")))
    l = set(normalize_string_list(lr.get("limit")))
    a = set(normalize_string_list(lr.get("avoid")))

    for t in numeric_tags:
        if t in a:
            continue
        p.discard(t)
        l.add(t)

    if isinstance(mpr, dict):
        mpr["logical_rules"] = {"prefer": sorted(p), "limit": sorted(l), "avoid": sorted(a)}
        out["meal_pattern_rules"] = mpr

def evidence_backed_tags_only(out: Dict[str, Any]) -> None:
    ev = out.get("rag_evidence") or []
    evidence_tags: Set[str] = set()

    if isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict):
                rt = e.get("related_tags") or []
                if isinstance(rt, list) and rt:
                    t = rt[0]
                    if isinstance(t, str) and t.strip():
                        evidence_tags.add(t.strip())

    mpr = out.get("meal_pattern_rules") or {}
    ncs = (mpr.get("numeric_constraints") or []) if isinstance(mpr, dict) else []
    if isinstance(ncs, list):
        for x in ncs:
            if isinstance(x, dict):
                t = x.get("tag")
                if isinstance(t, str) and t.strip():
                    evidence_tags.add(t.strip())

    if not evidence_tags:
        return

    def _filter_list(lst: Any) -> List[str]:
        return [t for t in normalize_string_list(lst) if t in evidence_tags]

    out["prefer_tags"] = _filter_list(out.get("prefer_tags"))
    out["limit_tags"] = _filter_list(out.get("limit_tags"))
    out["avoid_tags"] = _filter_list(out.get("avoid_tags"))

    mpr = out.get("meal_pattern_rules") or {}
    lr = (mpr.get("logical_rules") or {}) if isinstance(mpr, dict) else {}
    if isinstance(mpr, dict):
        mpr["logical_rules"] = {
            "prefer": _filter_list(lr.get("prefer")),
            "limit": _filter_list(lr.get("limit")),
            "avoid": _filter_list(lr.get("avoid")),
        }
        out["meal_pattern_rules"] = mpr


# =============================
# Numeric retag fix (sebze_meyve kaldırıldı)
# =============================
TAG_KEYWORDS: Dict[str, List[str]] = {
    "yumurta": ["yumurta", "yumurtanin", "yumurtanın", "yumurtayi", "yumurtayı", "yumurta sarisi", "yumurta akı", "yumurta aki"],
    "balik": ["balik", "balık", "somon", "uskumru", "hamsi", "sardalya"],
    "baklagil": ["baklagil", "kurubaklagil", "kuru baklagil", "mercimek", "nohut", "fasulye"],
    "baklagil_yemegi": ["baklagil", "kurubaklagil", "kuru baklagil", "mercimek", "nohut", "fasulye"],
    "tahil": ["tahil", "tahıl", "ekmek", "pirinc", "pirinç", "bulgur", "makarna", "pilav", "yulaf", "arpa", "cavdar", "çavdar"],
    "bulgur": ["bulgur"],
    "sebze": ["sebze"],
    "meyve": ["meyve"],
    "posa_yuksek": ["posa", "lif", "kabuklu"],
    "lif_yuksek": ["posa", "lif", "kabuklu"],
    "sodyum_dusuk": ["tuz", "sodyum"],
    "sodyum_yuksek": ["tuz", "sodyum"],
    "tam_yagli_sut_urunleri": ["tam yagli", "tam yağlı", "kaymak", "krema", "tereyağ", "tereyag", "yagli peynir", "tam yağlı peynir"],
    "az_yagli_sut_urunleri": ["az yagli", "az yağlı", "light", "yagsiz", "yağsız"],
    "beyaz_et": ["tavuk", "hindi", "kanatli", "kanatlı"],
    "kirmizi_et": ["kirmizi et", "kırmızı et", "dana", "kuzu", "sığır", "sigir"],
}

def _text_has_any(text: str, keys: List[str]) -> bool:
    low = norm_plain(text)
    for k in keys:
        if norm_plain(k) in low:
            return True
    return False

def retag_numeric_constraints_by_description(out: Dict[str, Any], tag_vocab_set: Set[str]) -> None:
    mpr = out.get("meal_pattern_rules") or {}
    ncs = mpr.get("numeric_constraints") or []
    if not isinstance(mpr, dict) or not isinstance(ncs, list) or not ncs:
        return

    fixed: List[dict] = []
    for x in ncs:
        if not isinstance(x, dict):
            continue
        tag = (x.get("tag") or "").strip()
        if tag not in tag_vocab_set:
            continue
        desc = (x.get("description") or "").strip()
        blob = desc

        cur_keys = TAG_KEYWORDS.get(tag, [])
        cur_hit = _text_has_any(blob, cur_keys) if cur_keys else False

        best_tag = tag
        for cand, keys in TAG_KEYWORDS.items():
            if cand == tag:
                continue
            if cand not in tag_vocab_set:
                continue
            if _text_has_any(blob, keys):
                if not cur_hit:
                    best_tag = cand
                    break

        if best_tag != tag:
            x2 = dict(x)
            x2["tag"] = best_tag
            fixed.append(x2)
        else:
            fixed.append(x)

    mpr["numeric_constraints"] = fixed
    out["meal_pattern_rules"] = mpr


# =============================
# Avoid gate
# =============================
def hard_avoid_gate(out: Dict[str, Any]) -> None:
    avoid = set(normalize_string_list(out.get("avoid_tags")))
    limit = set(normalize_string_list(out.get("limit_tags")))

    ev = out.get("rag_evidence") or []
    by_tag_quotes: Dict[str, List[str]] = {}
    if isinstance(ev, list):
        for e in ev:
            if not isinstance(e, dict):
                continue
            rtags = e.get("related_tags") or []
            if not rtags:
                continue
            t = rtags[0]
            q = e.get("quote") or ""
            by_tag_quotes.setdefault(t, []).append(q)

    for t in list(avoid):
        if t not in AVOID_RESTRICTED_TAGS:
            continue
        quotes = by_tag_quotes.get(t, [])
        ok_hard = any(HARD_AVOID_CUE_RE.search(q or "") for q in quotes)
        if not ok_hard:
            avoid.remove(t)
            limit.add(t)

    out["avoid_tags"] = sorted(avoid)
    out["limit_tags"] = sorted(limit)

    mpr = out.get("meal_pattern_rules") or {}
    lr = (mpr.get("logical_rules") or {}) if isinstance(mpr, dict) else {}
    a = set(normalize_string_list(lr.get("avoid")))
    l = set(normalize_string_list(lr.get("limit")))
    for t in list(a):
        if t in AVOID_RESTRICTED_TAGS:
            quotes = by_tag_quotes.get(t, [])
            ok_hard = any(HARD_AVOID_CUE_RE.search(q or "") for q in quotes)
            if not ok_hard:
                a.remove(t)
                l.add(t)
    if isinstance(mpr, dict):
        lr["avoid"] = sorted(a)
        lr["limit"] = sorted(l)
        mpr["logical_rules"] = lr
        out["meal_pattern_rules"] = mpr


# =============================
# FAIL-SAFE + PRESENTABLE EVIDENCE
# (orijinalindeki gibi bırakıyorum)
# =============================
# ... (BURASI SENİN MEVCUT KODUNDAKİYLE AYNI OLMALI)
# Ancak "eksiksiz" çalıştırabilmen için aşağıda bu bölümleri de korudum.

RE_WEEK_RANGE = re.compile(r"haftada\s*(\d+)\s*-\s*(\d+)\s*(kez|adet|porsiyon)\b", re.IGNORECASE)
RE_WEEK_SINGLE = re.compile(r"haftada\s*(\d+)\s*(kez|adet|porsiyon)\b", re.IGNORECASE)
RE_DAY_SINGLE = re.compile(r"günde\s*(\d+)\s*(kez|adet|porsiyon)\b", re.IGNORECASE)
RE_DAY_ONE = re.compile(r"günde\s*bir\s*(kez|adet|porsiyon)\b", re.IGNORECASE)
RE_WEEK_ONE = re.compile(r"haftada\s*bir\s*(kez|adet|porsiyon)\b", re.IGNORECASE)
RE_GRAMS_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)\s*(g|gr|gram|mg)\b", re.IGNORECASE)
RE_GRAMS_SINGLE = re.compile(r"(\d+)\s*(g|gr|gram|mg)\b", re.IGNORECASE)

def _unit_norm(u: str) -> str:
    u = (u or "").lower().strip()
    if u in ["g", "gr", "gram"]:
        return "gram"
    if u == "mg":
        return "mg"
    if u in ["kez", "adet", "porsiyon"]:
        return u
    return u

def extract_presentable_constraint(sentence: str) -> Optional[Dict[str, Any]]:
    s = sentence or ""
    m = RE_WEEK_RANGE.search(s)
    if m:
        a, b, unit = int(m.group(1)), int(m.group(2)), _unit_norm(m.group(3))
        return {"period_days": 7, "min": min(a, b), "max": max(a, b), "unit": unit}
    m = RE_WEEK_SINGLE.search(s)
    if m:
        a, unit = int(m.group(1)), _unit_norm(m.group(2))
        return {"period_days": 7, "min": None, "max": a, "unit": unit}
    m = RE_WEEK_ONE.search(s)
    if m:
        unit = _unit_norm(m.group(1))
        return {"period_days": 7, "min": None, "max": 1, "unit": unit}
    m = RE_DAY_SINGLE.search(s)
    if m:
        a, unit = int(m.group(1)), _unit_norm(m.group(2))
        return {"period_days": 1, "min": None, "max": a, "unit": unit}
    m = RE_DAY_ONE.search(s)
    if m:
        unit = _unit_norm(m.group(1))
        return {"period_days": 1, "min": None, "max": 1, "unit": unit}

    m = RE_GRAMS_RANGE.search(s)
    if m:
        a, b, unit = int(m.group(1)), int(m.group(2)), _unit_norm(m.group(3))
        return {"period_days": None, "min": min(a, b), "max": max(a, b), "unit": unit}
    m = RE_GRAMS_SINGLE.search(s)
    if m:
        a, unit = int(m.group(1)), _unit_norm(m.group(2))
        return {"period_days": None, "min": None, "max": a, "unit": unit}
    return None

def classify_sentence(sentence: str) -> str:
    s = sentence or ""
    if HARD_AVOID_CUE_RE.search(s):
        return "avoid"
    if CONDITION_CUE_RE.search(s) and not PREFER_CUE_RE.search(s) and not LIMIT_CUE_RE.search(s):
        return "condition"
    if LIMIT_CUE_RE.search(s) or NUMERIC_HINT_RE.search(s):
        return "limit"
    if PREFER_CUE_RE.search(s):
        return "prefer"
    low = norm_plain(s)
    if any(w in low for w in ["saglar", "sağlar", "bulunur", "icerir", "içerir", "etkisi", "az"]):
        return "info"
    return "neutral"

def match_tags_for_sentence(sentence: str, tag_vocab_set: Set[str]) -> List[str]:
    low = norm_plain(sentence)
    hits: List[str] = []

    for tag, keys in TAG_KEYWORDS.items():
        if tag not in tag_vocab_set:
            continue
        if any(norm_plain(k) in low for k in keys):
            hits.append(tag)

    for tag in tag_vocab_set:
        if tag and norm_plain(tag) in low and tag not in hits:
            hits.append(tag)

    return hits[:3]

def build_presentable_evidence_from_raw(raw_chunks: List[Dict[str, Any]], tag_vocab_set: Set[str], snippet_max_chars: int = 260) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in raw_chunks:
        cid = str(ch.get("chunk_id") or "").strip()
        if not cid:
            continue
        full = f"{ch.get('content') or ''}".strip()
        if not full:
            continue

        atoms = split_to_atomic_snippets(full, max_chars=int(snippet_max_chars)) or [full[:snippet_max_chars]]

        sent_objs: List[Dict[str, Any]] = []
        for s in atoms:
            s = (s or "").strip()
            if len(s) < 20:
                continue
            label = classify_sentence(s)
            tags = match_tags_for_sentence(s, tag_vocab_set)
            if not tags:
                tags = ["misc"]
            constraint = extract_presentable_constraint(s)
            sent_objs.append({
                "text": s,
                "label": label,
                "tags": tags,
                "constraint": constraint
            })

        if not sent_objs:
            continue

        out.append({
            "chunk_id": cid,
            "doc_id": ch.get("doc_id"),
            "topic_group": ch.get("topic_group"),
            "ana_baslik": ch.get("ana_baslik"),
            "section": ch.get("section"),
            "distance": ch.get("distance"),
            "retrieval_query": ch.get("retrieval_query"),
            "sentences": sent_objs
        })
    return out

RE_NUMERIC_ANY = re.compile(
    r"\b(gunde|günde|haftada|ayda|en\s*az|en\s*fazla)\b|(\d+)\s*(porsiyon|adet|kez|g|gr|gram|mg)\b",
    re.IGNORECASE
)

def _quote_looks_numeric(quote: str) -> bool:
    return bool(RE_NUMERIC_ANY.search(quote or ""))

def _to_nc_obj_from_constraint(tag: str, quote: str, c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not c or not isinstance(c, dict):
        return None

    unit = _unit_norm(c.get("unit") or "")
    period_days = c.get("period_days", None)
    mn = c.get("min", None)
    mx = c.get("max", None)

    if unit in ["g", "gr"]:
        unit = "gram"
    if unit not in ALLOWED_NUM_UNITS:
        return None

    if unit in ["kez", "porsiyon", "adet"]:
        if period_days is None:
            period_days = _infer_period_days_from_text(quote)
            if period_days is None:
                return None
        return {
            "tag": tag,
            "min_count": int(mn) if mn is not None else None,
            "max_count": int(mx) if mx is not None else None,
            "period_days": int(period_days) if period_days is not None else None,
            "min_grams": None,
            "max_grams": None,
            "unit": unit,
            "description": quote.strip() if quote else "Kaynak metindeki sayısal kural.",
        }

    if unit in ["gram", "mg"]:
        return {
            "tag": tag,
            "min_count": None,
            "max_count": None,
            "period_days": int(period_days) if period_days is not None else None,
            "min_grams": float(mn) if mn is not None else None,
            "max_grams": float(mx) if mx is not None else None,
            "unit": unit,
            "description": quote.strip() if quote else "Kaynak metindeki sayısal kural.",
        }

    return None

def fail_safe_numeric_from_rag_evidence(out: Dict[str, Any], tag_vocab_set: Set[str]) -> None:
    ev = out.get("rag_evidence") or []
    if not isinstance(ev, list) or not ev:
        return

    mpr = out.get("meal_pattern_rules")
    if not isinstance(mpr, dict):
        out["meal_pattern_rules"] = {"logical_rules": {"prefer": [], "limit": [], "avoid": []}, "numeric_constraints": []}
        mpr = out["meal_pattern_rules"]

    ncs = mpr.get("numeric_constraints")
    if not isinstance(ncs, list):
        ncs = []
        mpr["numeric_constraints"] = ncs

    existing_tags = set()
    for x in ncs:
        if isinstance(x, dict):
            t = x.get("tag")
            if isinstance(t, str) and t.strip():
                existing_tags.add(t.strip())

    added: List[Dict[str, Any]] = []

    for e in ev:
        if not isinstance(e, dict):
            continue
        rtags = e.get("related_tags") or []
        if not isinstance(rtags, list) or not rtags:
            continue

        tag = rtags[0].strip() if isinstance(rtags[0], str) else ""
        if not tag or tag not in tag_vocab_set:
            continue

        quote = (e.get("quote") or "").strip()
        if not quote:
            continue

        if not _quote_looks_numeric(quote):
            continue

        if tag in existing_tags:
            continue

        c = extract_presentable_constraint(quote)
        if not c:
            continue

        nc = _to_nc_obj_from_constraint(tag, quote, c)
        if not nc:
            continue

        added.append(nc)
        existing_tags.add(tag)

    if added:
        mpr["numeric_constraints"] = merge_numeric_constraints(mpr.get("numeric_constraints"), added)
        out["meal_pattern_rules"] = mpr


# =============================
# MAIN
# =============================
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--query", required=True)
    ap.add_argument("--domain", required=True)

    ap.add_argument("--tag_dicts", default=DEFAULT_TAG_DICTS_PATH)
    ap.add_argument("--disable_tag_dicts_patch", action="store_true")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    # İSTEK: ayrı dizinler
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--rules_out_dir", default=DEFAULT_RULES_OUT_DIR)

    # Chroma
    ap.add_argument("--chroma_dir", default=DEFAULT_CHROMA_DIR)
    ap.add_argument("--rag_collection", default=DEFAULT_RAG_COLLECTION)
    ap.add_argument("--tenant", default=DEFAULT_TENANT)
    ap.add_argument("--database", default=DEFAULT_DATABASE)

    # Rules store (opsiyonel)
    ap.add_argument("--store_rules_to_chroma", action="store_true")
    ap.add_argument("--rules_collection", default="disease_rules_store")

    # Retrieval
    ap.add_argument("--top_k_per_question", type=int, default=6)

    # Speed / gap-fill
    ap.add_argument("--gap_fill", action="store_true")
    ap.add_argument("--max_retrieval_rounds", type=int, default=3)
    ap.add_argument("--new_queries_per_round", type=int, default=6)
    ap.add_argument("--max_total_queries", type=int, default=40)
    ap.add_argument("--min_unique_chunks_before_llm_questions", type=int, default=10)
    ap.add_argument("--min_constraint_chunks", type=int, default=2)

    # Filters
    ap.add_argument("--drop_noisy", action="store_true")
    ap.add_argument("--max_clinical_hits", type=int, default=3)
    ap.add_argument("--where_json", default="")
    ap.add_argument("--require_domain_name_in_chunks", action="store_true")

    # LLM
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num_ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--llm_batch_size", type=int, default=14)

    ap.add_argument("--no_tag_meanings", action="store_true")

    ap.add_argument("--llm_question_count", type=int, default=12)

    # Truncation
    ap.add_argument("--max_chunk_chars", type=int, default=0)

    # Atomic snippets
    ap.add_argument("--atomic_snippets", action="store_true")
    ap.add_argument("--snippet_max_chars", type=int, default=260)
    ap.add_argument("--use_regex_splitter", action="store_true")

    # Self-check (şimdilik aynı bıraktım)
    ap.add_argument("--self_check", action="store_true")
    ap.add_argument("--max_self_check_tags", type=int, default=8)

    # Cache control
    ap.add_argument("--fresh_run", action="store_true")
    ap.add_argument("--cache_bust", default="")
    ap.add_argument("--clear_cache", action="store_true")

    args = ap.parse_args()

    model = (args.model or DEFAULT_MODEL).strip()
    query = args.query.strip()
    domain_in = (args.domain or "").strip()
    cache_bust = (args.cache_bust or "").strip()

    # 0) fingerprint
    code_fp = script_fingerprint()

    # 1) DOMAIN OVERRIDE (kolesterol)
    domain = force_domain_if_needed(domain_in, query)

    # İSTEK: klasörler
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_root, exist_ok=True)
    os.makedirs(args.rules_out_dir, exist_ok=True)

    # Disk cache (DU / retrieval / splitter) -> cache_root içine
    cache_subdir = build_model_cache_subdir(model, int(args.num_ctx))
    cache = DiskCache(args.cache_root, disabled=bool(args.fresh_run), cache_subdir=cache_subdir)
    if args.clear_cache and not args.fresh_run:
        cache.clear()
        print("[INFO] Cache temizlendi (--clear_cache).")

    patch_tag_dicts_if_missing(args.tag_dicts, disable_patch=bool(args.disable_tag_dicts_patch))

    raw_tag_dict = load_tag_dicts_raw(args.tag_dicts)
    policy = get_domain_policy(raw_tag_dict, domain)

    tag_meanings_all = extract_tags_from_raw(raw_tag_dict)
    tag_vocab, tag_meanings = load_tag_vocab(tag_meanings_all)
    tag_vocab_set = set(tag_vocab)

    where = None
    if args.where_json.strip():
        try:
            where = json.loads(args.where_json)
            if not isinstance(where, dict):
                print("[WARN] where_json dict değil, yok sayıyorum.")
                where = None
        except Exception as e:
            print("[WARN] where_json parse edilemedi, yok sayıyorum:", str(e))
            where = None

    send_meanings = (not bool(args.no_tag_meanings))

    # 2) Chroma client (RAG + optional rules store)
    client = chroma_client(args.chroma_dir, tenant=args.tenant, database=args.database)
    canonical_query = canonicalize_query_for_cache(query)

    # 3) DU
    domain_info = get_domain_understanding_cached(
        cache=cache,
        model=model,
        num_ctx=int(args.num_ctx),
        domain=domain,
        query=query,
        policy=policy,
        code_fp=code_fp,
        cache_bust=cache_bust
    )
    domain_info["retrieval_queries"] = sanitize_du_retrieval_queries(domain, domain_info.get("retrieval_queries") or [], policy)

    anchor_tokens = build_anchor_tokens(domain, domain_info, policy)
    domain_regex = build_domain_regex(anchor_tokens)
    other_terms = build_other_disease_terms_dynamic(raw_tag_dict, domain, policy)

    # 4) FAST retrieval
    all_queries: List[str] = build_fast_queries(domain, domain_info, max_total=int(args.max_total_queries))
    used_norm: Set[str] = set(norm_plain(q) for q in all_queries)

    cache_ns = f"qrag_{domain}_{args.rag_collection}"
    collected: List[Dict[str, Any]] = []

    def run_retrieval_for_queries(queries: List[str]) -> None:
        for q in queries:
            q_for_search = apply_query_booster(q)
            raw = cached_chroma_retrieve(
                cache=cache,
                client=client,
                collection_name=args.rag_collection,
                query=q_for_search,
                top_k=int(args.top_k_per_question),
                where=where,
                cache_namespace=cache_ns,
                code_fp=code_fp,
                cache_bust=cache_bust
            )
            collected.extend(raw)

    run_retrieval_for_queries(all_queries)

    raw_unique_all = dedupe_by_chunk_id(collected)

    def strict_unique_and_constraint_counts(items_in: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        filtered = filter_chroma_candidates_strict(
            items_in,
            drop_noisy=bool(args.drop_noisy),
            max_clinical_hits=int(args.max_clinical_hits),
            require_domain_name=bool(args.require_domain_name_in_chunks),
            domain_regex=domain_regex,
            anchor_tokens=anchor_tokens,
            policy=policy,
            other_terms=other_terms,
        )
        uniq = dedupe_by_chunk_id(filtered)

        if len(uniq) < 6 and bool(args.require_domain_name_in_chunks):
            filtered2 = filter_chroma_candidates_strict(
                items_in,
                drop_noisy=bool(args.drop_noisy),
                max_clinical_hits=int(args.max_clinical_hits),
                require_domain_name=False,
                domain_regex=domain_regex,
                anchor_tokens=anchor_tokens,
                policy=policy,
                other_terms=other_terms,
            )
            uniq2 = dedupe_by_chunk_id(filtered2)
            if len(uniq2) > len(uniq):
                uniq = uniq2

        if len(uniq) < 4:
            uniq = dedupe_by_chunk_id(items_in[:])

        constraint_hits = 0
        for it in uniq:
            if has_constraint_signal(it.get("content") or ""):
                constraint_hits += 1
        return uniq, constraint_hits

    uniq_items, constraint_hits = strict_unique_and_constraint_counts(collected)
    print(f"[Q-RAG] round=0 unique_chunks_strict={len(uniq_items)} constraint_chunks={constraint_hits}")

    # 5) Pack strict chunks
    max_chunk_chars = int(args.max_chunk_chars or 0)
    do_truncate = max_chunk_chars > 0

    def pack_chunk(it: Dict[str, Any]) -> Dict[str, Any]:
        content = it.get("content") or ""
        if do_truncate:
            content = content[:max_chunk_chars]
        return {
            "chunk_id": str(it.get("chunk_id") or "").strip(),
            "topic_group": it.get("topic_group"),
            "ana_baslik": it.get("ana_baslik"),
            "content": content,
        }

    chosen_chunks_strict = [pack_chunk(it) for it in uniq_items]

    # 6) Atomic snippets for LLM extraction (strict only)
    snippets: List[Dict[str, Any]] = []
    if args.atomic_snippets:
        for ch in chosen_chunks_strict:
            cid = ch["chunk_id"]
            txt = (ch.get("content") or "").strip()
            if not txt:
                continue

            if args.use_regex_splitter or has_bullets_or_lists(txt):
                atoms = split_to_atomic_snippets(txt, max_chars=int(args.snippet_max_chars)) or [txt]
                for j, a in enumerate(atoms):
                    a = (a or "").strip()
                    if not a:
                        continue
                    snippets.append({
                        "chunk_id": cid,
                        "snippet_id": f"{cid}::s{j}",
                        "topic_group": ch.get("topic_group"),
                        "ana_baslik": ch.get("ana_baslik"),
                        "content": a,
                    })
            else:
                if should_llm_split_when_no_bullet(txt):
                    atoms = llm_split_chunk_into_snippets_cached(
                        cache=cache,
                        model=model,
                        num_ctx=int(args.num_ctx),
                        chunk_text=txt,
                        code_fp=code_fp,
                        cache_bust=cache_bust,
                        temperature=0.0
                    )
                    for j, a in enumerate(atoms):
                        a = (a or "").strip()
                        if not a:
                            continue
                        snippets.append({
                            "chunk_id": cid,
                            "snippet_id": f"{cid}::llm_s{j}",
                            "topic_group": ch.get("topic_group"),
                            "ana_baslik": ch.get("ana_baslik"),
                            "content": a,
                        })
                else:
                    snippets.append({
                        "chunk_id": cid,
                        "snippet_id": f"{cid}::full",
                        "topic_group": ch.get("topic_group"),
                        "ana_baslik": ch.get("ana_baslik"),
                        "content": txt,
                    })
    else:
        for ch in chosen_chunks_strict:
            txt = (ch.get("content") or "").strip()
            if not txt:
                continue
            snippets.append({
                "chunk_id": ch["chunk_id"],
                "snippet_id": f"{ch['chunk_id']}::full",
                "topic_group": ch.get("topic_group"),
                "ana_baslik": ch.get("ana_baslik"),
                "content": txt,
            })

    print("[EXTRACT] strict_snippets_count:", len(snippets))

    # 7) LLM extraction (strict)
    template = build_empty_template(domain)
    system = build_system_prompt()

    decision_acc: Dict[str, Any] = {
        "prefer_tags": [],
        "limit_tags": [],
        "avoid_tags": [],
        "meal_pattern_rules": {"logical_rules": {"prefer": [], "limit": [], "avoid": []}, "numeric_constraints": []},
        "energy_rules": {"scale_up_order": [], "scale_down_order": [], "locks": []},
        "rag_evidence": [],
    }

    raw_out_first = ""
    bs = max(1, int(args.llm_batch_size))
    for i in range(0, len(snippets), bs):
        batch = snippets[i:i + bs]
        payload = build_user_payload(domain, query, tag_vocab, tag_meanings, batch, send_meanings)
        resp = ollama_chat(model, system, payload, temperature=0.0, num_ctx=int(args.num_ctx))
        if not raw_out_first and resp:
            raw_out_first = resp
        d = extract_first_json_object_balanced(resp) or {}
        if isinstance(d, dict) and d:
            decision_acc = merge_decisions(decision_acc, d)
        else:
            print("[WARN] Batch JSON extract edilemedi.")

    out = merge_llm_into_template(template, decision_acc)
    out = prune_to_template_keys(out, template)

    # 8) Temizleme/policy + numeric sanitize
    validate_and_fix(out, tag_vocab, policy)
    sanitize_numeric_constraints(out, tag_vocab_set)
    resolve_numeric_conflicts(out)
    sanitize_numeric_constraints(out, tag_vocab_set)
    validate_and_fix(out, tag_vocab, policy)

    # 9) Avoid gate
    hard_avoid_gate(out)
    validate_and_fix(out, tag_vocab, policy)

    # 10) Evidence sanitize + fail-safe numeric
    valid_chunk_ids = set([c.get("chunk_id") for c in chosen_chunks_strict if c.get("chunk_id")])
    sanitize_rag_evidence(out, tag_vocab_set, valid_chunk_ids)
    fail_safe_numeric_from_rag_evidence(out, tag_vocab_set)
    sanitize_numeric_constraints(out, tag_vocab_set)
    resolve_numeric_conflicts(out)
    sanitize_numeric_constraints(out, tag_vocab_set)

    evidence_backed_tags_only(out)
    validate_and_fix(out, tag_vocab, policy)

    # 11) Numeric retag fix + numeric=>limit
    retag_numeric_constraints_by_description(out, tag_vocab_set)
    sanitize_numeric_constraints(out, tag_vocab_set)
    resolve_numeric_conflicts(out)
    sanitize_numeric_constraints(out, tag_vocab_set)

    promote_numeric_tags_to_limit(out)
    validate_and_fix(out, tag_vocab, policy)

    evidence_backed_tags_only(out)
    validate_and_fix(out, tag_vocab, policy)

    # 12) Energy rules
    compute_energy_rules(out)

    # ZERO LOSS fields
    out["retrieval_results_raw"] = raw_unique_all
    out["retrieval_results_strict"] = uniq_items
    out["presentable_evidence"] = build_presentable_evidence_from_raw(
        raw_chunks=raw_unique_all,
        tag_vocab_set=tag_vocab_set,
        snippet_max_chars=int(args.snippet_max_chars)
    )

    # 13) Save (rules_out_dir -> chroma_rules)
    out_path = os.path.join(args.rules_out_dir, f"disease_rules_{domain}.json")
    safe_dump_json(out, out_path)

    debug_path = os.path.join(args.rules_out_dir, f"disease_rules_{domain}.debug.json")
    debug_obj = {
        "domain": domain,
        "domain_in": domain_in,
        "query": query,
        "canonical_query": canonical_query,
        "domain_policy": policy,
        "domain_understanding": domain_info,
        "questions": all_queries,
        "params": {
            "model": model,
            "num_ctx": int(args.num_ctx),
            "top_k_per_question": int(args.top_k_per_question),
            "truncate": {"enabled": do_truncate, "max_chunk_chars": max_chunk_chars},
            "atomic_snippets": bool(args.atomic_snippets),
            "use_regex_splitter": bool(args.use_regex_splitter),
            "snippet_max_chars": int(args.snippet_max_chars),
            "gap_fill": bool(args.gap_fill),
            "max_retrieval_rounds": int(args.max_retrieval_rounds),
            "new_queries_per_round": int(args.new_queries_per_round),
            "max_total_queries": int(args.max_total_queries),
            "min_unique_chunks_before_llm_questions": int(args.min_unique_chunks_before_llm_questions),
            "min_constraint_chunks": int(args.min_constraint_chunks),
            "require_domain_name_in_chunks": bool(args.require_domain_name_in_chunks),
            "drop_noisy": bool(args.drop_noisy),
            "max_clinical_hits": int(args.max_clinical_hits),
            "where": where,
            "cache": {
                "fresh_run": bool(args.fresh_run),
                "cache_bust": cache_bust,
                "clear_cache": bool(args.clear_cache),
                "cache_root": args.cache_root,
                "cache_subdir": cache_subdir,
                "code_fp": code_fp,
                "script_rev": SCRIPT_REV,
            },
            "send_tag_meanings": bool(send_meanings),
        },
        "retrieved_raw_unique_chunks": len(raw_unique_all),
        "retrieved_strict_unique_chunks": len(uniq_items),
        "constraint_chunks_strict": constraint_hits,
        "chosen_chunk_ids_preview_strict": [c.get("chunk_id") for c in chosen_chunks_strict[:80]],
        "strict_snippets_preview": snippets[:40],
        "llm_raw_preview": clean_json_text(raw_out_first)[:2500] if raw_out_first else ""
    }
    safe_dump_json(debug_obj, debug_path)

    # 14) OPTIONAL: upsert rules to Chroma
    if args.store_rules_to_chroma:
        try:
            upsert_rules_json_to_chroma(
                client=client,
                collection_name=args.rules_collection,
                domain=domain,
                rules_obj=out,
                metadata_extra={
                    "model": model,
                    "num_ctx": int(args.num_ctx),
                    "code_fp": code_fp,
                    "query": query,
                }
            )
            print(f"[RULES-STORE] OK: collection={args.rules_collection} id=rules::{norm_domain_key(domain)}")
        except Exception as e:
            print("[RULES-STORE] WARN: upsert başarısız:", str(e))

    print("OK:", out_path)
    print("DEBUG:", debug_path)
    print("CACHE_DIR:", cache.base_dir)
    print("RULES_OUT_DIR:", args.rules_out_dir)
    print("CACHE_ROOT:", args.cache_root)
    print("CODE_FP:", code_fp)


if __name__ == "__main__":
    main()