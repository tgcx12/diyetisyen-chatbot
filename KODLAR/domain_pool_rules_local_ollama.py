# -*- coding: utf-8 -*-
"""
domain_pool_rules_local_ollama.py  (2026-01-15)

Amaç (senin istediğin mimari):
✅ Retriever: SADECE domain için geniş bir "chunk havuzu" getirir (tek/az query, yüksek top_k).
✅ LLM: Bu havuzdan kuralları KENDİ çıkarır ve tag'ler (prefer/limit/avoid + numeric_constraints) üretir.
✅ RAG = kaynak havuzu; "cevapları retriever bulsun" değil, "kuralları LLM bulsun".
✅ Cache: retrieval + splitter + extraction cache (disk json).
✅ Store/upsert yok (Chroma’ya yeni şey yazmıyor).

Kullanım örnekleri:
python domain_pool_rules_local_ollama.py --domain kolesterol --query "yumurta tüketimi nasıl olmalı?" --domain_pool_top_k 120
python domain_pool_rules_local_ollama.py --domain diyabet --query "ekmek seçimi nasıl olmalı?" --where_json "{\"topic_group\":\"diyabet\"}"
python domain_pool_rules_local_ollama.py --domain kolesterol --domain_pool_where_domain_key topic_group

Notlar:
- tag_dicts.json içinde "tags" (tag -> anlam) olmalı. Domain policy opsiyonel.
- Chroma collection: --rag_collection (default: diyetisyen_rehberi)
- Ollama: http://localhost:11434/api/chat
"""

import os
import re
import json
import time
import random
import shutil
import hashlib
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import chromadb


# =============================
# CONFIG
# =============================
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
DEFAULT_NUM_CTX = 8192

DEFAULT_TAG_DICTS_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\tag_dicts.json"
DEFAULT_OUT_DIR = r"C:\Users\user\Desktop\diyetisyen_llm"
DEFAULT_CHROMA_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\chroma_db_storage"
DEFAULT_RAG_COLLECTION = "diyetisyen_rehberi"
DEFAULT_TENANT = "default_tenant"
DEFAULT_DATABASE = "default_database"


# =============================
# UTILS
# =============================
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

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


# =============================
# DISK CACHE
# =============================
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
# OLLAMA CHAT
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
        print(f"[WARN] Ollama fail: {last_err} | {sleep_s:.1f}s retry ({attempt}/{max_retries})")
        time.sleep(sleep_s)
    raise RuntimeError(f"Ollama chat başarısız: {last_err}")


# =============================
# TAG DICTS / POLICY
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
                if isinstance(a, str) and a.strip() and norm_plain(a) == norm_plain(domain):
                    return pol
    return {}

def policy_required_any(policy: Dict[str, Any]) -> List[str]:
    v = policy.get("required_any")
    if isinstance(v, list):
        return [str(x).strip() for x in v if isinstance(x, str) and x.strip()]
    return []

def policy_list(policy: Dict[str, Any], key: str) -> List[str]:
    v = policy.get(key)
    if isinstance(v, list):
        return [str(x).strip() for x in v if isinstance(x, str) and x.strip()]
    return []


# =============================
# CHROMA
# =============================
def chroma_client(persist_dir: str, tenant: str, database: str) -> chromadb.Client:
    return chromadb.PersistentClient(path=persist_dir, tenant=tenant, database=database)

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

def chroma_retrieve_chunks(
    client: chromadb.Client,
    collection_name: str,
    query: str,
    top_k: int,
    where: Optional[dict] = None
) -> List[Dict[str, Any]]:
    existing = _get_collection_names(client)
    if collection_name not in existing:
        raise RuntimeError(
            f"Chroma collection bulunamadı: '{collection_name}'. "
            f"Mevcut: {existing} (rag_collection/chroma_dir/tenant/database kontrol et)"
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
    cache_bust: str = ""
) -> List[Dict[str, Any]]:
    key_payload = {
        "ns": cache_namespace,
        "collection": collection_name,
        "query": query,
        "top_k": int(top_k),
        "where": where or None,
        "cache_bust": cache_bust or "",
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
# DOMAIN POOL RETRIEVAL (core)
# =============================
def build_domain_pool_query(domain: str, user_query: str, required_any: List[str]) -> str:
    """
    Tek query ile geniş havuz çekmek için.
    required_any varsa query'e en az 1 tanesini enjekte eder.
    """
    bits = [domain, "beslenme", "diyet", "tercih edilmeli", "sınırlandırılmalı", "kaçınılmalı"]
    if user_query:
        bits.append(user_query.strip())
    # required_any: ilk terimi ekle (çok şişirmeden)
    if required_any:
        bits.append(required_any[0])
    q = " ".join(bits).strip()
    if not q.endswith("?"):
        q = q + "?"
    return q

def try_build_where_domain(domain: str, domain_key: str) -> Optional[dict]:
    """
    Eğer metadata alanı biliyorsan (topic_group vb.) equality filtre uygula.
    """
    k = (domain_key or "").strip()
    if not k:
        return None
    return {k: domain}


# =============================
# ATOMIC SNIPPETS (regex splitter + opsiyonel LLM splitter)
# =============================
_BULLET_SPLIT_RE = re.compile(r"(?:\n\s*[•\-\*]\s+)")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")
BULLET_PRESENT_RE = re.compile(
    r"(^|\n)\s*(?:[•\-\*]\s+|\d+\s*[\)\.\-]\s+|[a-zA-ZçğıöşüÇĞİÖŞÜ]\s*[\)\.\-]\s+)",
    re.MULTILINE
)

def has_bullets_or_lists(text: str) -> bool:
    return bool(BULLET_PRESENT_RE.search(text or ""))

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


# =============================
# EXTRACTION PROMPT (LLM kuralları kendi çıkaracak)
# =============================
ALLOWED_NUM_UNITS = {"kez", "porsiyon", "adet", "gram", "mg"}

def build_extraction_system_prompt() -> str:
    return (
        "SEN UZMAN BİR DİYETİSYEN VE INFORMATION EXTRACTION UZMANISIN.\n"
        "Sana verilen RAG_SNIPPETS bir 'domain havuzudur'.\n"
        "Görev: Bu havuzdaki metinden BESLENME KURALLARINI çıkar ve etiketle.\n\n"
        "ÇOK ÖNEMLİ:\n"
        "- SADECE JSON döndür. Açıklama/markdown yok.\n"
        "- RAG_SNIPPETS içinde açıkça olmayan bilgi ekleme (uydurma yasak).\n"
        "- Tag'ler SADECE TAG_VOCAB içinden seçilecek.\n"
        "- Her tag için rag_evidence içine DOĞRUDAN alıntı (quote) koy.\n"
        "- Aynı tag hem prefer hem limit hem avoid olamaz. Öncelik: avoid > limit > prefer.\n\n"
        "Sınıflama:\n"
        "- 'kaçınılmalı/tüketilmemeli/yasak/uzak dur' => avoid\n"
        "- Sayısal sınır / 'en fazla' + sayı => numeric_constraints + limit\n"
        "- 'sınırlandırılmalı/azaltılmalı' ama sayı yoksa: limit_tags'a alabilirsin (ama evidence şart)\n"
        "- 'önerilir/tercih edilmeli/artırılmalı' => prefer\n\n"
        "numeric_constraints şeması:\n"
        "{\n"
        "  \"tag\":\"...\",\n"
        "  \"min_count\": null,\n"
        "  \"max_count\": null,\n"
        "  \"period_days\": 1|7|30,\n"
        "  \"min_grams\": null,\n"
        "  \"max_grams\": null,\n"
        "  \"unit\": \"kez|porsiyon|adet|gram|mg\",\n"
        "  \"description\": \"...\",\n"
        "  \"quote\": \"...\",\n"
        "  \"chunk_id\": \"...\"\n"
        "}\n\n"
        "ÇIKTI ŞEMASI:\n"
        "{\n"
        "  \"prefer_tags\": [],\n"
        "  \"limit_tags\": [],\n"
        "  \"avoid_tags\": [],\n"
        "  \"meal_pattern_rules\": {\n"
        "    \"logical_rules\": {\"prefer\": [], \"limit\": [], \"avoid\": []},\n"
        "    \"numeric_constraints\": []\n"
        "  },\n"
        "  \"recommendations\": [\n"
        "    {\"tag\":\"...\", \"intent\":\"prefer|avoid|choose_instead\", \"description\":\"...\", \"quote\":\"...\", \"chunk_id\":\"...\"}\n"
        "  ],\n"
        "  \"rag_evidence\": [\n"
        "    {\"chunk_id\":\"...\", \"related_tags\":[\"tag\"], \"quote\":\"...\"}\n"
        "  ]\n"
        "}\n"
    )

def build_extraction_user_payload(
    domain: str,
    user_query: str,
    tag_vocab: List[str],
    tag_meanings: Dict[str, str],
    rag_snippets: List[Dict[str, Any]],
    send_tag_meanings: bool
) -> str:
    payload: Dict[str, Any] = {
        "domain": domain,
        "query": user_query,
        "TAG_VOCAB": tag_vocab,
        "RAG_SNIPPETS": rag_snippets,
    }
    if send_tag_meanings:
        compact = {k: (v[:140] + "…") if len(v) > 141 else v for k, v in tag_meanings.items()}
        payload["TAG_MEANINGS"] = compact
    return json.dumps(payload, ensure_ascii=False)


# =============================
# POST VALIDATION / CLEANUP
# =============================
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
    for x in ncs:
        if not isinstance(x, dict):
            continue
        tag = (x.get("tag") or "").strip()
        if tag not in tag_vocab_set:
            continue
        unit = (x.get("unit") or "").strip().lower()
        if unit in ["g", "gr"]:
            unit = "gram"
        if unit not in ALLOWED_NUM_UNITS:
            continue

        # period_days zorunluluğu (kez/porsiyon/adet için)
        period_days = x.get("period_days", None)
        if unit in ["kez", "porsiyon", "adet"]:
            if period_days not in [1, 7, 30]:
                continue

        # en az bir numeric alan dolu olmalı
        if (x.get("min_count") is None and x.get("max_count") is None and
            x.get("min_grams") is None and x.get("max_grams") is None):
            continue

        cleaned.append(x)

    mpr["numeric_constraints"] = cleaned
    out["meal_pattern_rules"] = mpr

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
        if not isinstance(rtags, list) or not rtags:
            continue
        tag = (rtags[0] or "").strip()
        if not cid or not quote or tag not in tag_vocab_set:
            continue
        if valid_chunk_ids and cid not in valid_chunk_ids:
            continue
        key = (cid, quote, tag)
        if key in seen:
            continue
        cleaned.append({"chunk_id": cid, "related_tags": [tag], "quote": quote})
        seen.add(key)
    out["rag_evidence"] = cleaned

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


# =============================
# MAIN
# =============================
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--query", required=False, default="", help="Kullanıcı sorusu (LLM extraction context).")
    ap.add_argument("--domain", required=True, help="Domain adı (ör: kolesterol, diyabet).")

    ap.add_argument("--tag_dicts", default=DEFAULT_TAG_DICTS_PATH)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    # Chroma
    ap.add_argument("--chroma_dir", default=DEFAULT_CHROMA_DIR)
    ap.add_argument("--rag_collection", default=DEFAULT_RAG_COLLECTION)
    ap.add_argument("--tenant", default=DEFAULT_TENANT)
    ap.add_argument("--database", default=DEFAULT_DATABASE)

    # Domain-pool retrieval
    ap.add_argument("--domain_pool_top_k", type=int, default=100, help="Tek seferde çekilecek chunk sayısı.")
    ap.add_argument("--domain_pool_query", default="", help="Tek query override. Boşsa otomatik.")
    ap.add_argument("--domain_pool_where_domain_key", default="", help="Metadata alan adıyla (eşitlik) filtrele (örn topic_group).")
    ap.add_argument("--where_json", default="", help="Ek where filtresi JSON (dict).")

    # LLM
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num_ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--llm_batch_size", type=int, default=16)
    ap.add_argument("--no_tag_meanings", action="store_true")

    # Snippet
    ap.add_argument("--snippet_max_chars", type=int, default=260)

    # Cache
    ap.add_argument("--fresh_run", action="store_true", help="Cache kapatılır.")
    ap.add_argument("--cache_bust", default="")
    ap.add_argument("--clear_cache", action="store_true")

    args = ap.parse_args()

    domain = (args.domain or "").strip()
    user_query = (args.query or "").strip()
    model = (args.model or DEFAULT_MODEL).strip()
    cache_bust = (args.cache_bust or "").strip()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_subdir = build_model_cache_subdir(model, int(args.num_ctx))
    cache = DiskCache(args.out_dir, disabled=bool(args.fresh_run), cache_subdir=cache_subdir)
    if args.clear_cache and not args.fresh_run:
        cache.clear()
        print("[INFO] Cache temizlendi.")

    raw_tag_dict = load_tag_dicts_raw(args.tag_dicts)
    policy = get_domain_policy(raw_tag_dict, domain)
    required_any = policy_required_any(policy)

    tag_meanings_all = extract_tags_from_raw(raw_tag_dict)
    tag_vocab = sorted(list(tag_meanings_all.keys()))
    tag_meanings = {t: tag_meanings_all.get(t, "") for t in tag_vocab}
    tag_vocab_set = set(tag_vocab)

    send_meanings = (not bool(args.no_tag_meanings))

    # where filtresi (kullanıcı + domain_key)
    where_user = None
    if args.where_json.strip():
        try:
            where_user = json.loads(args.where_json)
            if not isinstance(where_user, dict):
                print("[WARN] where_json dict değil, yok sayıyorum.")
                where_user = None
        except Exception as e:
            print("[WARN] where_json parse edilemedi, yok sayıyorum:", str(e))
            where_user = None

    where_domain = try_build_where_domain(domain, args.domain_pool_where_domain_key)
    # where birleşimi (AND mantığı Chroma'da direkt dict merge gibi çalışmaz; en pratik: kullanıcı verdiyse onu kullan,
    # domain_key kullanmak istiyorsan where_json'a ekle.)
    where = where_user if where_user is not None else where_domain

    # Client
    client = chroma_client(args.chroma_dir, tenant=args.tenant, database=args.database)

    # Tek sorgu: domain pool
    pool_query = (args.domain_pool_query or "").strip()
    if not pool_query:
        pool_query = build_domain_pool_query(domain, user_query, required_any)

    cache_ns = f"domain_pool_{domain}_{args.rag_collection}"
    key = "pool_" + _sha1(json.dumps({
        "ns": cache_ns,
        "pool_query": pool_query,
        "top_k": int(args.domain_pool_top_k),
        "where": where,
        "cache_bust": cache_bust
    }, ensure_ascii=False, sort_keys=True))

    pool_chunks = cache.get(key)
    if not isinstance(pool_chunks, list):
        pool_chunks = cached_chroma_retrieve(
            cache=cache,
            client=client,
            collection_name=args.rag_collection,
            query=pool_query,
            top_k=int(args.domain_pool_top_k),
            where=where,
            cache_namespace=cache_ns,
            cache_bust=cache_bust
        )
        cache.set(key, pool_chunks)

    pool_chunks = dedupe_by_chunk_id(pool_chunks)
    print(f"[POOL] retrieved_unique_chunks={len(pool_chunks)} top_k={int(args.domain_pool_top_k)}")

    # Chunk -> atomic snippets
    snippets: List[Dict[str, Any]] = []
    max_chars = int(args.snippet_max_chars)

    for ch in pool_chunks:
        cid = str(ch.get("chunk_id") or "").strip()
        txt = (ch.get("content") or "").strip()
        if not cid or not txt:
            continue
        atoms = split_to_atomic_snippets(txt, max_chars=max_chars) or [txt[:max_chars]]
        for j, a in enumerate(atoms):
            a = (a or "").strip()
            if not a:
                continue
            snippets.append({
                "chunk_id": cid,
                "snippet_id": f"{cid}::s{j}",
                "topic_group": ch.get("topic_group"),
                "ana_baslik": ch.get("ana_baslik"),
                "content": a
            })

    print(f"[SNIPPETS] count={len(snippets)}")

    # LLM Extraction (batch)
    system = build_extraction_system_prompt()

    # Accumulator
    out: Dict[str, Any] = {
        "dataset": f"disease_rules_{domain}",
        "disease": {"disease_id": domain.upper(), "name": domain},
        "prefer_tags": [],
        "limit_tags": [],
        "avoid_tags": [],
        "meal_pattern_rules": {"logical_rules": {"prefer": [], "limit": [], "avoid": []}, "numeric_constraints": []},
        "recommendations": [],
        "rag_evidence": [],
        "retrieval_pool_query": pool_query,
        "retrieval_pool_where": where,
        "retrieval_results_pool": pool_chunks,
        "snippets_used_count": len(snippets)
    }

    # cache extraction per batch
    bs = max(1, int(args.llm_batch_size))
    for i in range(0, len(snippets), bs):
        batch = snippets[i:i + bs]
        payload = build_extraction_user_payload(domain, user_query, tag_vocab, tag_meanings, batch, send_meanings)
        batch_key = "ext_" + _sha1(json.dumps({
            "model": model,
            "num_ctx": int(args.num_ctx),
            "domain": domain,
            "query": user_query,
            "batch_sha": _sha1(payload),
            "cache_bust": cache_bust
        }, ensure_ascii=False, sort_keys=True))

        resp_obj = cache.get(batch_key)
        if not isinstance(resp_obj, dict):
            resp = ollama_chat(model, system, payload, temperature=0.0, num_ctx=int(args.num_ctx))
            resp_obj = extract_first_json_object_balanced(resp) or {}
            if not isinstance(resp_obj, dict):
                resp_obj = {}
            cache.set(batch_key, resp_obj)

        # Merge (basit union)
        out["prefer_tags"] = sorted(set(normalize_string_list(out["prefer_tags"])) | set(normalize_string_list(resp_obj.get("prefer_tags"))))
        out["limit_tags"]  = sorted(set(normalize_string_list(out["limit_tags"]))  | set(normalize_string_list(resp_obj.get("limit_tags"))))
        out["avoid_tags"]  = sorted(set(normalize_string_list(out["avoid_tags"]))  | set(normalize_string_list(resp_obj.get("avoid_tags"))))

        # meal_pattern_rules merge
        mpr = out.get("meal_pattern_rules") or {}
        d_mpr = resp_obj.get("meal_pattern_rules") or {}
        if isinstance(mpr, dict) and isinstance(d_mpr, dict):
            lr = mpr.get("logical_rules") or {"prefer": [], "limit": [], "avoid": []}
            d_lr = d_mpr.get("logical_rules") or {}
            if isinstance(lr, dict) and isinstance(d_lr, dict):
                for cls in ["prefer", "limit", "avoid"]:
                    lr[cls] = sorted(set(normalize_string_list(lr.get(cls))) | set(normalize_string_list(d_lr.get(cls))))
                mpr["logical_rules"] = lr

            ncs = mpr.get("numeric_constraints") or []
            d_ncs = d_mpr.get("numeric_constraints") or []
            if isinstance(ncs, list) and isinstance(d_ncs, list):
                # naive append + later sanitize
                ncs.extend([x for x in d_ncs if isinstance(x, dict)])
                mpr["numeric_constraints"] = ncs
            out["meal_pattern_rules"] = mpr

        # recs merge
        recs = out.get("recommendations") or []
        d_recs = resp_obj.get("recommendations") or []
        if isinstance(recs, list) and isinstance(d_recs, list):
            recs.extend([x for x in d_recs if isinstance(x, dict)])
            out["recommendations"] = recs

        # evidence merge
        ev = out.get("rag_evidence") or []
        d_ev = resp_obj.get("rag_evidence") or []
        if isinstance(ev, list) and isinstance(d_ev, list):
            ev.extend([x for x in d_ev if isinstance(x, dict)])
            out["rag_evidence"] = ev

    # Post-clean
    keep_only_vocab_tags(out, tag_vocab_set)
    enforce_exclusivity(out)

    # numeric sanitize + numeric tags => limit
    sanitize_numeric_constraints(out, tag_vocab_set)
    promote_numeric_tags_to_limit(out)
    enforce_exclusivity(out)

    # evidence sanitize
    valid_chunk_ids = set([str(c.get("chunk_id")) for c in pool_chunks if c.get("chunk_id")])
    sanitize_rag_evidence(out, tag_vocab_set, valid_chunk_ids)

    # final
    out_path = os.path.join(args.out_dir, f"disease_rules_{domain}.json")
    debug_path = os.path.join(args.out_dir, f"disease_rules_{domain}.debug.json")

    safe_dump_json(out, out_path)
    safe_dump_json({
        "domain": domain,
        "query": user_query,
        "policy": policy,
        "pool_query": pool_query,
        "where": where,
        "retrieved_unique_chunks": len(pool_chunks),
        "snippets_count": len(snippets),
        "params": {
            "model": model,
            "num_ctx": int(args.num_ctx),
            "domain_pool_top_k": int(args.domain_pool_top_k),
            "llm_batch_size": int(args.llm_batch_size),
            "send_tag_meanings": bool(send_meanings),
            "cache_subdir": cache.base_dir
        }
    }, debug_path)

    print("OK:", out_path)
    print("DEBUG:", debug_path)
    print("CACHE_DIR:", cache.base_dir)


if __name__ == "__main__":
    main()
