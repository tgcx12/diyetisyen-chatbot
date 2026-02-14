# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import requests
import chromadb

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS: Dict[str, Dict[str, Any]] = {
    "llama_3_1_8b_instruct": {
        "label": "Llama-3.1:8B Instruct",
        "ollama": "llama3.1:8b-instruct",
        "num_ctx": 8192,
    },
    "gemma_3_4b": {
        "label": "Gemma-3:4B",
        "ollama": "gemma3:4b",
        "num_ctx": 8192,
    },
    "qwen2_5_3b": {
        "label": "Qwen-2.5:3B",
        "ollama": "qwen2.5:3b",
        "num_ctx": 8192,
    },
    "gemma_2_2b": {
        "label": "Gemma-2:2B",
        "ollama": "gemma2:2b",
        "num_ctx": 8192,
    },
}

DEFAULT_MODEL_ORDER = [
    "llama_3_1_8b_instruct",
    "gemma_3_4b",
    "qwen2_5_3b",
    "gemma_2_2b",
]

DEFAULT_NUM_CTX = 8192

CHROMA_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\chroma_db_storage"
RAG_COLLECTION = "diyetisyen_rehberi"

TAG_DICTS_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\tag_dicts.json"
CACHE_ROOT = r"C:\Users\user\Desktop\diyetisyen_llm\deneme_cache\qa_demo"

TOP_K_DEFAULT = 24

ENTITY_EXPAND_TOPK = 350
MAX_EVIDENCE_CHUNKS = 12
MAX_SCAN_CHUNKS = 150

MAX_EVIDENCE_SENTENCES_PER_CHUNK = 8
FALLBACK_FIRST_SENTENCES = 3
MAX_EVIDENCE_CHARS_PER_CHUNK = 1100

RULE_SEPARATOR = " | "
MAX_RULES_OUT = 6

CACHE_VERSION = "v17_strict_models_arg_and_model_summary_json"

REQUIRE_DOMAIN_MARKER_IN_CHUNK = True
DOMAIN_MARKER_META_ONLY = True

DEBUG_PRINT_LLM_PAYLOAD = False


# =========================
# QUESTION SPECS
# =========================
QUESTION_SPECS: List[Dict[str, Any]] = [
    {"id":"genel_1","category":"genel",
     "text":"{domain} için beslenme genel olarak nasıl olmalı, neler tercih edilmeli?",
     "want_numeric":False,"want_qual":True,"entity_terms":[]},

    {"id":"genel_2","category":"genel",
     "text":"{domain} için genel olarak neler sınırlandırılmalı, nelerden kaçınılmalı?",
     "want_numeric":False,"want_qual":True,"entity_terms":[]},

    {"id":"protein_kirmizi_et","category":"protein",
     "text":"{domain} için kırmızı et tüketimi nasıl olmalı ve hangi türlerden/işlenmiş etlerden kaçınılmalı?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["kırmızı et","kirmizi et","dana","kuzu","işlenmiş et","islenmis et","sucuk","salam","sosis","pastırma","pastirma","jambon","kavurma"],
     "exclude_terms":["glisemik","gi","glisemik indeks","karbonhidrat","pirinç","pirinc","ekmek"]},

    {"id":"protein_beyaz_et","category":"protein",
     "text":"{domain} için beyaz et (tavuk/hindi) tüketimi nasıl olmalı ve pişirme yöntemi nasıl olmalı?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["tavuk","hindi","beyaz et"]},

    {"id":"protein_balik","category":"protein",
     "text":"{domain} için balık tüketimi nasıl olmalı ve hangi tür/pişirme tercih edilmeli?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["balık","balik","somon","uskumru","sardalya","hamsi","omega-3","omega3","balık yağı","balik yagi"]},

    {"id":"protein_yumurta","category":"protein",
     "text":"{domain} için yumurta tüketimi nasıl olmalı ve pişirme/yanında tüketim önerisi var mı?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["yumurta","yumurta sarısı","yumurta sarisi"]},

    {"id":"sut_1","category":"sut",
     "text":"{domain} için süt ve süt ürünleri tüketimi nasıl olmalı?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["süt","sut","yoğurt","yogurt","peynir","az yağlı","az yagli","yağsız","yagsiz","tam yağlı","tam yagli","kaymak","krema"]},

    {"id":"tahil_1","category":"tahil",
     "text":"{domain} için ekmek/tahıl/pilav-makarna tüketimi nasıl olmalı (tam tahıl mı rafine mi)?",
     "want_numeric":False,"want_qual":True,
     "entity_terms":["tam tahıl","tam tahil","bulgur","yulaf","beyaz ekmek","rafine","pirinç","pirinc","kepekli"]},

    {"id":"baklagil_1","category":"baklagil",
     "text":"{domain} için baklagil tüketimi nasıl olmalı ve hangi şekilde tercih edilmeli?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["baklagil","kuru baklagil","kurubaklagil","mercimek","nohut","fasulye"],
     "exclude_terms":["sebze","meyve","sebze-meyve"]},

    {"id":"kuruyemis_1","category":"kuruyemis",
     "text":"{domain} için kuruyemiş/kabuklu yemiş tüketimi nasıl olmalı, hangi türler tercih edilmeli ve hangilerinden kaçınılmalı?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":[
        "kuruyemiş","kuruyemis","kabuklu yemiş","kabuklu yemis","yağlı tohum","yagli tohum",
        "badem","ceviz","fındık","findik","fıstık","fistik","antep fıstığı","antep fistigi",
        "kaju","pikan","çekirdek","cekirdek","ay çekirdeği","ay cekirdegi","kabak çekirdeği","kabak cekirdegi",
        "susam","keten","chia","tuzlu","kavrulmuş","kavrulmus","şeker kaplı","seker kapli"
     ],
     "exclude_terms":["kurubaklagil","baklagil","mercimek","nohut","fasulye"]},

    {"id":"yag_doymus","category":"yag",
     "text":"{domain} için doymuş yağ tüketimi nasıl olmalı ve hangi yağlar tercih edilmeli?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["doymuş","doymus","tereyağı","tereyagi","margarin","zeytinyağı","zeytinyagi","fındık yağı","findik yagi","trans"]},

    {"id":"yag_trans","category":"yag",
     "text":"{domain} için trans yağdan kaçınılmalı mı, hangi ürünlerde bulunur?",
     "want_numeric":False,"want_qual":True,
     "entity_terms":["trans","margarin","paketli","hazır","hazir","unlu mamul","fast food","kızart","kizart","gofret","bisküvi","biskuvi","hazır kek","hazir kek"]},

    {"id":"lif_1","category":"posa_lif",
     "text":"{domain} için posa/lif tüketimi nasıl olmalı ve hangi kaynaklar önerilir?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["lif","posa","posalı","posali","tam tahıl","tam tahil","baklagil","sebze","meyve"]},

    {"id":"tuz_1","category":"tuz_seker",
     "text":"{domain} için tuz/sodyum tüketimi nasıl olmalı ve hangi ürünlerden kaçınılmalı?",
     "want_numeric":True,"want_qual":True,
     "entity_terms":["tuz","sodyum","tuzlu","salamura","turşu","tursu","zeytin"]},

    {"id":"seker_1","category":"tuz_seker",
     "text":"{domain} için rafine şeker/şekerli içecek tüketimi nasıl olmalı ve yerine ne tercih edilmeli?",
     "want_numeric":False,"want_qual":True,
     "entity_terms":["şeker","seker","şekerli içecek","sekerli icecek","gazlı","gazli","rafine","eklenmiş şeker","eklenmis seker","reçel","recel","bal","marmelat"]},
]


# =========================
# TEXT HELPERS
# =========================
def norm_plain(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("ı","i").replace("ş","s").replace("ğ","g").replace("ü","u").replace("ö","o").replace("ç","c")
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_query_for_retrieval(q: str) -> str:
    q = (q or "").strip()
    q = q.replace("/", " ").replace("\\", " ")
    q = q.replace("’", "'").replace("“", '"').replace("”", '"')
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def sanitize_dir_name(name: str) -> str:
    """
    Windows klasör adı için riskli karakterleri temizler.
    Domain'lerde ':' vb olmuyor ama garanti olsun.
    """
    n = (name or "").strip()
    n = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", n)
    n = n.rstrip(". ")
    return n or "_"

# =========================
# TAG DICTS: DOMAIN POLICY
# =========================
def get_domains_block(raw: Dict[str, Any]) -> Dict[str, Any]:
    d1 = raw.get("domains")
    d2 = raw.get("domain_policies")
    merged: Dict[str, Any] = {}
    if isinstance(d1, dict):
        merged.update(d1)
    if isinstance(d2, dict):
        merged.update(d2)
    return merged if isinstance(merged, dict) else {}

def get_domain_policy(raw: Dict[str, Any], domain: str) -> Dict[str, Any]:
    domains = get_domains_block(raw)
    if not isinstance(domains, dict):
        return {}
    if domain in domains and isinstance(domains[domain], dict):
        return domains[domain]
    dkey = norm_plain(domain).replace(" ", "_")
    for k, v in domains.items():
        if not isinstance(v, dict):
            continue
        if norm_plain(str(k)).replace(" ", "_") == dkey:
            return v
        aliases = v.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and norm_plain(a) == norm_plain(domain):
                    return v
    return {}

def extract_list(policy: Dict[str, Any], key: str) -> List[str]:
    v = policy.get(key) or []
    if isinstance(v, list):
        return [str(x).strip() for x in v if isinstance(x, str) and x.strip()]
    return []

def get_domain_markers(policy: Dict[str, Any], domain: str) -> List[str]:
    markers: List[str] = []
    markers.extend(extract_list(policy, "anchors"))
    markers.extend(extract_list(policy, "aliases"))
    markers.append(domain)
    seen = set()
    out: List[str] = []
    for m in markers:
        nm = norm_plain(m)
        if nm and nm not in seen:
            seen.add(nm)
            out.append(m)
    return out

def policy_bool(policy: Dict[str, Any], path: Tuple[str, ...], default: bool = False) -> bool:
    cur: Any = policy
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return bool(cur) if cur is not None else default

def extract_competitor_domains(policy: Dict[str, Any]) -> List[str]:
    dis = policy.get("disambiguation")
    if not isinstance(dis, dict):
        return []
    cds = dis.get("competitor_domains") or []
    if isinstance(cds, list):
        return [str(x).strip() for x in cds if isinstance(x, str) and x.strip()]
    return []

def extract_competitor_anchors(tag_dicts_raw: Dict[str, Any], policy: Dict[str, Any]) -> List[str]:
    domains = get_domains_block(tag_dicts_raw)
    out: List[str] = []
    for d in extract_competitor_domains(policy):
        pol = domains.get(d)
        if not isinstance(pol, dict):
            continue
        out.extend(extract_list(pol, "anchors"))
        out.extend(extract_list(pol, "aliases"))
    seen = set()
    res = []
    for x in out:
        nx = norm_plain(x)
        if nx and nx not in seen:
            seen.add(nx)
            res.append(x)
    return res

def is_shared_only_domain(domain_key: str, policy: Dict[str, Any]) -> bool:
    if "_ortak" in norm_plain(domain_key):
        return True
    notes = str(policy.get("notes") or "")
    if "Ortak bağlam" in notes or "Ortak bağlam domain" in notes:
        return True
    return False


# =========================
# CHROMA
# =========================
def chroma_client(persist_dir: str) -> chromadb.Client:
    return chromadb.PersistentClient(path=persist_dir)

def chroma_retrieve(client: chromadb.Client, collection: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    col = client.get_collection(collection)
    res = col.query(query_texts=[query], n_results=int(top_k))
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        md = metas[i] or {}
        out.append({
            "chunk_id": str(ids[i]),
            "content": (docs[i] or ""),
            "ana_baslik": md.get("ana_baslik"),
            "topic_group": md.get("topic_group"),
            "section": md.get("section"),
            "doc_id": md.get("doc_id"),
            "content_type": md.get("content_type"),
        })
    return out

def dedupe_by_chunk_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for it in items:
        cid = it.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(it)
    return out


# =========================
# FILTERING / MATCHING
# =========================
def contains_any(text: str, terms: List[str]) -> bool:
    low = norm_plain(text)
    for t in terms:
        nt = norm_plain(t)
        if nt and nt in low:
            return True
    return False

def build_meta_blob(ch: Dict[str, Any]) -> str:
    return " ".join([
        str(ch.get("chunk_id") or ""),
        str(ch.get("doc_id") or ""),
        str(ch.get("ana_baslik") or ""),
        str(ch.get("section") or ""),
        str(ch.get("topic_group") or ""),
    ])

def has_domain_marker_meta_only(ch: Dict[str, Any], domain_markers: List[str]) -> bool:
    return contains_any(build_meta_blob(ch), domain_markers)

def is_table_chunk(ch: Dict[str, Any]) -> bool:
    ct = str(ch.get("content_type") or "").lower().strip()
    if ct == "table":
        return True
    txt = (ch.get("content") or "")
    t = txt.strip()
    if not t:
        return False
    if "|" in t and re.search(r"\n\s*\|?\s*-{2,}\s*\|", t):
        return True
    if re.search(r"^\s*\|.+\|\s*$", t) and "\n" in t:
        return True
    return False

GLOBAL_BLOCK_TOPIC_GROUPS = {"ilac_besin_etkilesimi"}

def is_blocked_topic_group(ch: Dict[str, Any], domain: str) -> bool:
    tg = norm_plain(str(ch.get("topic_group") or ""))
    if not tg:
        return False
    if tg in GLOBAL_BLOCK_TOPIC_GROUPS and norm_plain(domain) != tg:
        return True
    return False

def filter_chunks_strict(
    chunks: List[Dict[str, Any]],
    domain: str,
    domain_markers: List[str],
    anchors: List[str],
    competitor_anchors: List[str],
    exclude_if_only: List[str],
    require_anchor_in_chunk: bool,
    strict_competitor_exclusion: bool,
    entity_terms: List[str],
    require_domain_marker_in_chunk: bool
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        if is_table_chunk(ch):
            continue
        if is_blocked_topic_group(ch, domain):
            continue

        meta_blob = build_meta_blob(ch)
        full_blob = f"{meta_blob} {ch.get('content','')}"

        has_anchor = contains_any(meta_blob, anchors) if anchors else False
        has_entity = contains_any(full_blob, entity_terms) if entity_terms else False

        has_comp = contains_any(full_blob, competitor_anchors) if competitor_anchors else False
        has_excl = contains_any(full_blob, exclude_if_only) if exclude_if_only else False

        tg = str(ch.get("topic_group") or "").strip()
        if tg:
            if (not contains_any(tg, domain_markers)) and (not contains_any(full_blob, domain_markers)):
                continue

        if require_domain_marker_in_chunk:
            if not has_domain_marker_meta_only(ch, domain_markers):
                continue

        if strict_competitor_exclusion and has_comp and (not has_anchor) and (not has_entity):
            continue

        if (not has_anchor) and has_excl:
            continue

        if require_anchor_in_chunk:
            if not has_anchor and not has_entity:
                continue

        out.append(ch)
    return out

def hard_filter_by_entity(chunks: List[Dict[str, Any]], entity_terms: List[str]) -> List[Dict[str, Any]]:
    if not entity_terms:
        return chunks
    out = []
    for ch in chunks:
        full = f"{build_meta_blob(ch)} {ch.get('content','')}"
        if contains_any(full, entity_terms):
            out.append(ch)
    return out


# =========================
# HINTS / SNIPPET
# =========================
NUMERIC_CUE_RE = re.compile(
    r"(\b(günde|gunde|günlük|gunluk|haftada|ayda)\b|(\d+(?:[.,]\d+)?)\s*(adet|kez|porsiyon|g|gr|gram|mg)\b|(\d+(?:[.,]\d+)?)\s*%|%\s*(\d+(?:[.,]\d+)?))",
    re.IGNORECASE
)

NUMERIC_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:"
    r"g|kg|mg|mcg|µg|ml|l|dl|"
    r"%|kcal|kalori|kkal|"
    r"adet|porsiyon|dilim|"
    r"kez|x|"
    r"gün|gun|hafta|ay|yıl|yil"
    r")\b",
    flags=re.IGNORECASE
)

QUAL_CUE_RE = re.compile(
    r"\b("
    r"tercih|öner|oner|sınır|sinir|kaçın|kacin|azalt|artır|artir|yerine|"
    r"seç|sec|tüket|tuket|paketli|hazır|hazir|işlenmiş|islenmis|fast[- ]?food|"
    r"kızart|kizart|izgara|haşlama|haslama|buhar|buharda|fırın|firin|"
    r"tam tahıl|tam tahil|rafine|omega[- ]?3|omega3|şeker|seker|tuz|sodyum|"
    r"kolesterol|ldl|hdl|trigliserid|trigliserit"
    r")\b",
    re.IGNORECASE
)

SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")
TOPIC_LABEL_RE = re.compile(r"\b[a-z]{2,}_[a-z0-9_]{2,}\b", re.IGNORECASE)
HEADER_FIELD_RE = re.compile(r"(?i)\b(başlık|konu|icerik|içerik)\s*:\s*")

def clean_noise_tokens(text: str) -> str:
    t = (text or "")
    t = HEADER_FIELD_RE.sub("", t)
    t = TOPIC_LABEL_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def normalize_bullets_for_sentence_split(t: str) -> str:
    t = (t or "")
    t = t.replace("•", ". ")
    t = re.sub(r"\s*-\s+", ". ", t)
    t = t.replace(";", ". ")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def split_sentences(text: str) -> List[str]:
    t = normalize_bullets_for_sentence_split(clean_noise_tokens(text))
    sents = [s.strip() for s in SENT_SPLIT_RE.split(t) if s.strip()]
    return sents

def merge_unique(items: List[str], limit: int) -> List[str]:
    seen = set()
    out = []
    for x in items:
        nx = norm_plain(x)
        if nx and nx not in seen:
            seen.add(nx)
            out.append(x)
        if len(out) >= limit:
            break
    return out

def select_evidence_snippet(
    txt: str,
    want_numeric_orig: bool,
    want_qual_orig: bool,
    entity_terms: List[str],
    exclude_terms: List[str],
    max_sentences: int = MAX_EVIDENCE_SENTENCES_PER_CHUNK,
    fallback_first_sentences: int = FALLBACK_FIRST_SENTENCES,
    max_chars: int = MAX_EVIDENCE_CHARS_PER_CHUNK
) -> Tuple[str, List[str], List[str]]:
    txt = clean_noise_tokens((txt or "").strip())
    if not txt:
        return "", [], []

    sents = split_sentences(txt)
    if not sents:
        return "", [], []

    def excluded(s: str) -> bool:
        if not exclude_terms:
            return False
        sn = norm_plain(s)
        return any(norm_plain(et) in sn for et in exclude_terms)

    sents2 = [s for s in sents if not excluded(s)]
    if not sents2:
        sents2 = sents

    numeric_hits: List[str] = []
    qual_hits: List[str] = []

    entity_hits: List[str] = []
    if entity_terms:
        for s in sents2:
            if contains_any(s, entity_terms):
                entity_hits.append(s)

    if want_numeric_orig:
        for s in sents2:
            if NUMERIC_CUE_RE.search(s):
                numeric_hits.append(s)
    if want_qual_orig:
        for s in sents2:
            if QUAL_CUE_RE.search(s):
                qual_hits.append(s)

    merged = []
    merged.extend(entity_hits)
    merged.extend(numeric_hits)
    merged.extend(qual_hits)
    merged = merge_unique(merged, limit=max_sentences)

    if not merged:
        merged = sents2[:fallback_first_sentences]

    snippet = " ".join(merged).strip()
    snippet = clean_noise_tokens(snippet)

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0].strip()

    numeric_hints_clean = merge_unique([h for h in numeric_hits if NUMERIC_UNIT_RE.search(h)], limit=20)
    qual_hints_clean = merge_unique(qual_hits, limit=20)

    return snippet, numeric_hints_clean, qual_hints_clean

def score_chunk(
    ch: Dict[str, Any],
    domain_markers: List[str],
    anchors: List[str],
    want_numeric: bool,
    want_qual: bool,
    entity_terms: List[str]
) -> int:
    meta_blob = build_meta_blob(ch)
    content = str(ch.get("content") or "")
    full_blob = f"{meta_blob} {content}"

    sc = 0
    if entity_terms and contains_any(full_blob, entity_terms):
        sc += 10
    if anchors and contains_any(meta_blob, anchors):
        sc += 2
    if has_domain_marker_meta_only(ch, domain_markers):
        sc += 2
    if want_numeric and NUMERIC_CUE_RE.search(full_blob or ""):
        sc += 2
    if want_qual and QUAL_CUE_RE.search(full_blob or ""):
        sc += 1
    return sc


# =========================
# OUTPUT CLEANING
# =========================
_LEAK_PATTERNS = [
    r"kesin kurallar",
    r"kesintisiz kurallar",
    r"çıktı",
    r"kural\s*\d+",
    r"başlık\s*:",
    r"konu\s*:",
    r"içerik\s*:",
    r"json",
]

def looks_like_instruction_leak(s: str) -> bool:
    low = norm_plain(s)
    if any(re.search(p, low, flags=re.IGNORECASE) for p in _LEAK_PATTERNS):
        return True
    if TOPIC_LABEL_RE.search(s or ""):
        return True
    return False

def strip_leaked_instructions(s: str) -> str:
    s = clean_noise_tokens((s or "").strip())
    s = s.replace("Kaynaklarda bu soru için bilgi bulunamadı.", "")
    s = s.replace("Kaynaklarda bu soru için bilgi bulunamadı", "")
    return s.strip()

def force_one_line_and_separator(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("•", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*-\s*", " ", s)

    parts = [p.strip(" -–—\t ").strip() for p in re.split(r"\s*\|\s*", s) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"[.!?]+", s) if p.strip()]

    cleaned_parts: List[str] = []
    for p in parts:
        pp = clean_noise_tokens(p)
        if not pp:
            continue
        if TOPIC_LABEL_RE.search(pp):
            continue
        if len(pp) < 6:
            continue
        cleaned_parts.append(pp)

    cleaned_parts = merge_unique(cleaned_parts, limit=MAX_RULES_OUT)
    out = f" {RULE_SEPARATOR} ".join(cleaned_parts) if cleaned_parts else s
    out = re.sub(r"\s+\|\s+\|\s+", f" {RULE_SEPARATOR} ", out)
    out = re.sub(r"\s+\|\s*$", "", out).strip()
    return out

def strip_question_echo(ans: str, domain: str, question: str) -> str:
    a = (ans or "").strip()
    if not a:
        return a
    dn = norm_plain(domain)
    qn = norm_plain(question)
    an = norm_plain(a)

    if dn and an.startswith(dn + " icin"):
        a2 = re.sub(r"(?i)^\s*" + re.escape(domain) + r"\s+için\s+", "", a).strip()
        if len(a2) >= 12:
            a = a2

    if qn and (qn in an or an in qn) and len(an) < 120:
        return ""
    return a

def postprocess_answer(s: str, domain: str = "", question: str = "") -> str:
    s = (s or "").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("“") and s.endswith("”")):
        s = s[1:-1].strip()
    s = s.replace('\\"', '"').strip()
    s = re.sub(r"\s+", " ", s).strip()

    s = strip_leaked_instructions(s)
    s = strip_question_echo(s, domain=domain, question=question)
    s = strip_leaked_instructions(s)

    s = force_one_line_and_separator(s)
    return s.strip()


# =========================
# EXTRACTIVE FALLBACK
# =========================
def extractive_fallback_one_line(
    evidence: List[Dict[str, str]],
    want_numeric_orig: bool,
    want_qual_orig: bool,
    max_rules: int = MAX_RULES_OUT
) -> str:
    if not evidence:
        return "Kaynaklarda bu soru için bilgi bulunamadı."

    text = " ".join([e.get("text", "") for e in evidence if (e.get("text") or "").strip()]).strip()
    text = clean_noise_tokens(text)
    if not text:
        return "Kaynaklarda bu soru için bilgi bulunamadı."

    sents = split_sentences(text)
    picked: List[str] = []
    seen = set()

    def add(s: str):
        ss = clean_noise_tokens(s)
        ns = norm_plain(ss)
        if ns and ns not in seen:
            seen.add(ns)
            picked.append(ss)

    if want_numeric_orig:
        for s in sents:
            if NUMERIC_UNIT_RE.search(s):
                add(s)
                if len(picked) >= max_rules:
                    break

    if len(picked) < max_rules and want_qual_orig:
        for s in sents:
            if QUAL_CUE_RE.search(s):
                add(s)
                if len(picked) >= max_rules:
                    break

    if len(picked) < max_rules:
        for s in sents:
            add(s)
            if len(picked) >= max_rules:
                break

    out = f" {RULE_SEPARATOR} ".join(picked[:max_rules]).strip()
    out = force_one_line_and_separator(out)
    return out if out else "Kaynaklarda bu soru için bilgi bulunamadı."


# =========================
# LLM
# =========================
def ollama_chat(model: str, system: str, user: str, num_ctx: int, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": float(temperature), "num_ctx": int(num_ctx)},
    }
    if DEBUG_PRINT_LLM_PAYLOAD:
        print("DEBUG payload model:", model)
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content") or ""

def build_system_prompt(
    want_numeric_orig: bool,
    want_qual_orig: bool,
    allow_numeric: bool,
    must_mention_terms: Optional[List[str]] = None
) -> str:
    must_mention_terms = must_mention_terms or []

    numeric_rule = ""
    if allow_numeric:
        numeric_rule = "Sayı/gram/%/kez gibi sayısal ifadeleri SADECE EVIDENCE/HINTS içinde geçiyorsa kullan; yeni sayı uydurma.\n"
    else:
        if want_numeric_orig:
            numeric_rule = "Bu soru sayısal bilgi isteyebilir; EVIDENCE/HINTS içinde sayı yoksa sayı yazma.\n"
        else:
            numeric_rule = "Sayı/gram/%/kez gibi sayısal ifade yazma.\n"

    qual_rule = ""
    if want_qual_orig:
        qual_rule = "Önerileri SADECE EVIDENCE/HINTS içeriğinden çıkar; yeni bilgi/çıkarım ekleme.\n"
    else:
        qual_rule = "Öneri yazma.\n"

    must_rule = ""
    if must_mention_terms:
        joined = ", ".join(must_mention_terms[:6])
        must_rule = f"Cevapta mümkünse şu terimlerden en az birini geçir: {joined}.\n"

    return (
        "Sen kanıt-temelli klinik diyetisyen asistansın.\n"
        "Sana JSON verilecek: domain, question, HINTS, EVIDENCE.\n"
        "=== KESİN KURALLAR (İHLAL YASAK) ===\n"
        "1) SADECE EVIDENCE ve HINTS içindeki bilgiyle yaz; ASLA yeni bilgi/çıkarım ekleme.\n"
        "2) EVIDENCE boşsa çıktın TAM olarak şu olsun: Kaynaklarda bu soru için bilgi bulunamadı.\n"
        "3) EVIDENCE boş DEĞİLSE 'Kaynaklarda bu soru için bilgi bulunamadı.' yazmak YASAK.\n"
        "4) Çıktı TEK SATIR olacak.\n"
        f"5) En fazla {MAX_RULES_OUT} kısa ifade üret ve '{RULE_SEPARATOR}' ile ayır.\n"
        "6) Liste/bullet/numara/başlık/JSON YAZMA.\n"
        "7) 'KESİN KURALLAR', 'ÇIKTI', 'kural 1' gibi talimatları ASLA cevapta yazma.\n"
        "8) Cevapta teknik domain anahtarlarını ASLA kullanma; doğal bir dil kullan.\n"
        + numeric_rule + qual_rule + must_rule
    )

def validate_llm_answer(
    raw: str,
    processed: str,
    domain: str,
    question: str,
    evidence_present: bool
) -> bool:
    if not processed or len(processed.strip()) < 8:
        return False
    low = processed.strip().lower()

    if evidence_present and ("kaynaklarda bu soru için bilgi bulunamadı" in low):
        return False

    if looks_like_instruction_leak(raw) or looks_like_instruction_leak(processed):
        return False

    if not strip_question_echo(processed, domain, question):
        return False

    if "|" in processed and re.search(r"\|\s*-{2,}\s*\|", processed):
        return False

    return True


# =========================
# RETRIEVAL HELPERS
# =========================
def make_query(question: str) -> str:
    return normalize_query_for_retrieval(question)

def retrieve_strict(
    client: chromadb.Client,
    domain: str,
    question: str,
    anchors: List[str],
    domain_markers: List[str],
    competitor_anchors: List[str],
    exclude_if_only: List[str],
    require_anchor_in_chunk: bool,
    strict_competitor_exclusion: bool,
    entity_terms: List[str],
    top_k: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    q = make_query(question)
    raw = dedupe_by_chunk_id(chroma_retrieve(client, RAG_COLLECTION, q, top_k=top_k))

    strict = filter_chunks_strict(
        raw,
        domain=domain,
        domain_markers=domain_markers,
        anchors=anchors,
        competitor_anchors=competitor_anchors,
        exclude_if_only=exclude_if_only,
        require_anchor_in_chunk=require_anchor_in_chunk,
        strict_competitor_exclusion=strict_competitor_exclusion,
        entity_terms=entity_terms,
        require_domain_marker_in_chunk=REQUIRE_DOMAIN_MARKER_IN_CHUNK
    )
    meta = {"retrieval_attempts": 1, "attempts": [{"retry": False, "query": q}]}
    return strict, meta

def retrieve_entity_aggregate(
    client: chromadb.Client,
    domain: str,
    question: str,
    anchors: List[str],
    domain_markers: List[str],
    competitor_anchors: List[str],
    exclude_if_only: List[str],
    require_anchor_in_chunk: bool,
    strict_competitor_exclusion: bool,
    entity_terms: List[str],
    top_k: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    attempts = []
    merged: List[Dict[str, Any]] = []

    queries = [
        make_query(question),
        make_query(f"{domain} {' '.join(entity_terms[:10])}".strip()),
        make_query(" ".join(entity_terms[:12]).strip())
    ]

    for q in queries:
        if not q:
            continue
        attempts.append({"retry": False, "query": q})
        merged.extend(chroma_retrieve(client, RAG_COLLECTION, q, top_k=top_k))

    merged = dedupe_by_chunk_id(merged)

    strict = filter_chunks_strict(
        merged,
        domain=domain,
        domain_markers=domain_markers,
        anchors=anchors,
        competitor_anchors=competitor_anchors,
        exclude_if_only=exclude_if_only,
        require_anchor_in_chunk=require_anchor_in_chunk,
        strict_competitor_exclusion=strict_competitor_exclusion,
        entity_terms=entity_terms,
        require_domain_marker_in_chunk=REQUIRE_DOMAIN_MARKER_IN_CHUNK
    )
    strict = hard_filter_by_entity(strict, entity_terms)

    meta = {"retrieval_attempts": len(attempts), "attempts": attempts}
    return strict, meta


# =========================
# CACHE (DOMAIN / MODEL)
# =========================
class DiskCache:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def path(self, key: str) -> str:
        return os.path.join(self.base_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Any]:
        p = self.path(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, obj: Any) -> None:
        try:
            with open(self.path(key), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def get_model_cache_dir(domain: str, model_key: str) -> str:
    d_key = sanitize_dir_name((domain or "_unknown_domain").strip() or "_unknown_domain")
    m_key = sanitize_dir_name((model_key or "_unknown_model").strip() or "_unknown_model")
    return os.path.join(CACHE_ROOT, d_key, m_key)

def ensure_domain_model_dirs(domain: str, models_order: List[str]) -> None:
    for mk in models_order:
        os.makedirs(get_model_cache_dir(domain, mk), exist_ok=True)

def cache_key(domain: str, model_key: str, question: str, top_k: int, qspec: Dict[str, Any]) -> str:
    want_numeric_orig = bool(qspec.get("want_numeric"))
    want_qual_orig = bool(qspec.get("want_qual"))
    entity_terms = (qspec.get("entity_terms") or [])[:20]
    exclude_terms = (qspec.get("exclude_terms") or [])[:20]

    return "ans_" + _sha1(json.dumps({
        "ver": CACHE_VERSION,
        "domain": domain,
        "model_key": model_key,  # model bazlı ayrı cache
        "question": question,
        "top_k": int(top_k),
        "wn": want_numeric_orig,
        "wq": want_qual_orig,
        "entity": entity_terms,
        "ex": exclude_terms,
        "req_domain_marker": REQUIRE_DOMAIN_MARKER_IN_CHUNK,
        "meta_only": DOMAIN_MARKER_META_ONLY,
        "max_rules": MAX_RULES_OUT,
    }, ensure_ascii=False, sort_keys=True))

def cache_try_get(domain: str, model_key: str, akey: str) -> Optional[Dict[str, Any]]:
    cd = get_model_cache_dir(domain, model_key)
    hit = DiskCache(cd).get(akey)
    if isinstance(hit, dict) and isinstance(hit.get("answer"), str) and hit["answer"].strip():
        return hit
    return None

def cache_put(domain: str, model_key: str, akey: str, obj: Dict[str, Any]) -> None:
    cd = get_model_cache_dir(domain, model_key)
    DiskCache(cd).set(akey, obj)


# =========================
# QUESTION EXPANSION
# =========================
TOPIC_BY_ID: Dict[str, str] = {
    "protein_kirmizi_et": "kırmızı et",
    "protein_beyaz_et": "beyaz et (tavuk/hindi)",
    "protein_balik": "balık",
    "protein_yumurta": "yumurta",
    "sut_1": "süt ve süt ürünleri",
    "baklagil_1": "baklagil",
    "kuruyemis_1": "kuruyemiş/kabuklu yemiş",
    "yag_doymus": "doymuş yağ",
    "lif_1": "posa/lif",
    "tuz_1": "tuz/sodyum",
}

def infer_topic_from_spec(sp: Dict[str, Any]) -> str:
    sid = str(sp.get("id") or "")
    if sid in TOPIC_BY_ID:
        return TOPIC_BY_ID[sid]
    terms = sp.get("entity_terms") or []
    for cand in terms:
        c = str(cand).strip()
        if c and len(c) >= 4:
            return c
    cat = str(sp.get("category") or "").strip()
    return cat if cat else "bu besin"

def build_question_specs_for_domain(domain_key: str, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    aliases = policy.get("aliases", [])
    display_name = aliases[0] if aliases else domain_key.replace("_", " ")

    specs_out: List[Dict[str, Any]] = []
    for sp in QUESTION_SPECS:
        base = sp.copy()
        base["question"] = sp["text"].format(domain=display_name).strip()
        specs_out.append(base)

        if bool(sp.get("want_numeric")):
            topic = infer_topic_from_spec(sp)

            sp_freq = sp.copy()
            sp_freq["id"] = f"{sp['id']}_num_freq"
            sp_freq["want_numeric"] = True
            sp_freq["want_qual"] = False
            sp_freq["question"] = f"{display_name} için {topic} tüketimi ne sıklıkla olmalı (günde/haftada/ayda kaç kez)?"
            specs_out.append(sp_freq)

            sp_amt = sp.copy()
            sp_amt["id"] = f"{sp['id']}_num_amt"
            sp_amt["want_numeric"] = True
            sp_amt["want_qual"] = False
            sp_amt["question"] = f"{display_name} için {topic} tüketim miktarı ne olmalı (1 porsiyon/adet/gram/ml)?"
            specs_out.append(sp_amt)

    return specs_out


# =========================
# SINGLE MODEL ANSWER (NO CASCADE)
# =========================
def build_evidence_and_hints(
    chunks_sorted: List[Dict[str, Any]],
    want_numeric_orig: bool,
    want_qual_orig: bool,
    entity_terms: List[str],
    exclude_terms: List[str]
) -> Tuple[List[Dict[str, str]], List[str]]:
    evidence: List[Dict[str, str]] = []
    hints: List[str] = []

    for ch in chunks_sorted[:min(len(chunks_sorted), MAX_SCAN_CHUNKS)]:
        txt = (ch.get("content") or "").strip()
        if not txt:
            continue

        snippet, n_h, q_h = select_evidence_snippet(
            txt, want_numeric_orig, want_qual_orig, entity_terms, exclude_terms
        )
        if not snippet:
            continue

        # entity_terms varsa snippet içinde de kalsın (sapmayı azaltır)
        if entity_terms and not contains_any(snippet, entity_terms):
            continue

        evidence.append({"chunk_id": ch.get("chunk_id"), "text": snippet})
        hints.extend(n_h)
        hints.extend(q_h)
        if len(evidence) >= MAX_EVIDENCE_CHUNKS:
            break

    hints = merge_unique(hints, limit=16)
    return evidence, hints

def summarize_with_specific_model(
    model_key: str,
    domain: str,
    question: str,
    want_numeric_orig: bool,
    want_qual_orig: bool,
    hints: List[str],
    evidence: List[Dict[str, str]],
    must_terms: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Tek bir model çalıştırır. Başarısız olursa extractive fallback döndürür.
    """
    if not evidence:
        return "Kaynaklarda bu soru için bilgi bulunamadı.", {"model_used": model_key, "ok": False, "reason": "no_evidence"}

    mi = MODELS.get(model_key) or {}
    model_name = mi.get("ollama") or ""
    num_ctx = int(mi.get("num_ctx") or DEFAULT_NUM_CTX)

    blob = " ".join(hints) + " " + " ".join([e.get("text", "") for e in evidence])
    allow_numeric = bool(NUMERIC_UNIT_RE.search(blob))

    sys_prompt = build_system_prompt(
        want_numeric_orig=want_numeric_orig,
        want_qual_orig=want_qual_orig,
        allow_numeric=allow_numeric and want_numeric_orig,
        must_mention_terms=must_terms
    )

    payload_obj = {"domain": domain, "question": question, "HINTS": hints, "EVIDENCE": evidence}
    payload = json.dumps(payload_obj, ensure_ascii=False)

    if not model_name:
        fb = extractive_fallback_one_line(evidence, want_numeric_orig, want_qual_orig)
        return fb, {"model_used": model_key, "ok": False, "reason": "model_not_configured"}

    try:
        raw = ollama_chat(model_name, sys_prompt, payload, num_ctx=num_ctx, temperature=0.0)
        processed = postprocess_answer(raw, domain=domain, question=question)
        ok = validate_llm_answer(raw, processed, domain, question, evidence_present=True)
        if ok:
            return processed, {"model_used": model_key, "ok": True}
        fb = extractive_fallback_one_line(evidence, want_numeric_orig, want_qual_orig)
        return fb, {"model_used": model_key, "ok": False, "reason": "invalid_output"}
    except Exception as e:
        fb = extractive_fallback_one_line(evidence, want_numeric_orig, want_qual_orig)
        return fb, {"model_used": model_key, "ok": False, "reason": "exception", "error": str(e)}

def answer_one_question_for_model(
    client: chromadb.Client,
    tag_dicts_raw: Dict[str, Any],
    domain: str,
    qspec: Dict[str, Any],
    top_k: int,
    model_key: str,
) -> Dict[str, Any]:
    policy = get_domain_policy(tag_dicts_raw, domain) or {}

    anchors = extract_list(policy, "anchors") or [domain]
    domain_markers = get_domain_markers(policy, domain)

    exclude_if_only = extract_list(policy, "exclude_if_only")
    competitor_anchors = extract_competitor_anchors(tag_dicts_raw, policy)

    require_anchor_for_queries = policy_bool(policy, ("disambiguation", "require_anchor_for_retrieval_queries"), True)
    strict_competitor_excl = policy_bool(policy, ("disambiguation", "strict_competitor_exclusion"), True)
    require_anchor_in_chunk = bool(require_anchor_for_queries)

    question = qspec["question"]
    want_numeric_orig = bool(qspec.get("want_numeric"))
    want_qual_orig = bool(qspec.get("want_qual"))
    entity_terms = qspec.get("entity_terms") or []
    exclude_terms = qspec.get("exclude_terms") or []

    akey = cache_key(domain, model_key, question, top_k, qspec)
    hit = cache_try_get(domain, model_key, akey)
    if hit:
        return hit

    # Retrieval (1 kere)
    if entity_terms:
        chunks, retrieval_meta = retrieve_entity_aggregate(
            client=client,
            domain=domain,
            question=question,
            anchors=anchors,
            domain_markers=domain_markers,
            competitor_anchors=competitor_anchors,
            exclude_if_only=exclude_if_only,
            require_anchor_in_chunk=require_anchor_in_chunk,
            strict_competitor_exclusion=strict_competitor_excl,
            entity_terms=entity_terms,
            top_k=max(int(top_k), ENTITY_EXPAND_TOPK)
        )
    else:
        chunks, retrieval_meta = retrieve_strict(
            client=client,
            domain=domain,
            question=question,
            anchors=anchors,
            domain_markers=domain_markers,
            competitor_anchors=competitor_anchors,
            exclude_if_only=exclude_if_only,
            require_anchor_in_chunk=require_anchor_in_chunk,
            strict_competitor_exclusion=strict_competitor_excl,
            entity_terms=entity_terms,
            top_k=int(top_k)
        )

    if not chunks:
        out = {
            "domain": domain,
            "question": question,
            "answer": "Kaynaklarda bu soru için bilgi bulunamadı.",
            "used_chunks": [],
            "strict_chunks_count": 0,
            "retrieval_meta": retrieval_meta,
            "numeric_in_evidence": False,
            "numeric_in_answer": False,
            "qual_in_evidence": False,
            "qual_in_answer": False,
            "llm_meta": {"model_used": model_key, "ok": False, "reason": "no_chunks"},
            "qspec": {"id": qspec.get("id"), "category": qspec.get("category"),
                      "want_numeric": want_numeric_orig, "want_qual": want_qual_orig},
        }
        cache_put(domain, model_key, akey, out)
        return out

    chunks_sorted = sorted(
        chunks,
        key=lambda c: score_chunk(c, domain_markers, anchors, want_numeric_orig, want_qual_orig, entity_terms),
        reverse=True
    )

    evidence, hints = build_evidence_and_hints(
        chunks_sorted=chunks_sorted,
        want_numeric_orig=want_numeric_orig,
        want_qual_orig=want_qual_orig,
        entity_terms=entity_terms,
        exclude_terms=exclude_terms
    )

    ans, llm_meta = summarize_with_specific_model(
        model_key=model_key,
        domain=domain,
        question=question,
        want_numeric_orig=want_numeric_orig,
        want_qual_orig=want_qual_orig,
        hints=hints,
        evidence=evidence,
        must_terms=entity_terms
    )

    ans = postprocess_answer(ans, domain=domain, question=question)

    ev_text = " ".join([e.get("text", "") for e in evidence]) if evidence else ""
    out = {
        "domain": domain,
        "question": question,
        "answer": ans,
        "used_chunks": [e["chunk_id"] for e in evidence],
        "strict_chunks_count": len(chunks),
        "retrieval_meta": retrieval_meta,
        "numeric_in_evidence": bool(NUMERIC_UNIT_RE.search(ev_text)),
        "numeric_in_answer": bool(NUMERIC_UNIT_RE.search(ans)),
        "qual_in_evidence": bool(QUAL_CUE_RE.search(ev_text)),
        "qual_in_answer": bool(QUAL_CUE_RE.search(ans)),
        "llm_meta": llm_meta,   # model_key kesin
        "qspec": {"id": qspec.get("id"), "category": qspec.get("category"),
                  "want_numeric": want_numeric_orig, "want_qual": want_qual_orig},
    }

    # HER ZAMAN kendi model klasörüne yaz
    cache_put(domain, model_key, akey, out)
    return out


# =========================
# MODEL SUMMARY JSON (Soru + Cevap + Chunk ID)
# =========================
def compact_qa_record(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": str(res.get("question") or "").strip(),
        "answer": str(res.get("answer") or "").strip(),
        "used_chunks": list(res.get("used_chunks") or []),
    }

def write_domain_model_summary(domain: str, model_key: str, items: List[Dict[str, Any]]) -> str:
    out_obj = {
        "domain": domain,
        "model_key": model_key,
        "count": len(items),
        "items": items,
    }
    out_path = os.path.join(get_model_cache_dir(domain, model_key), "qa_summary.json")
    safe_dump_json(out_obj, out_path)
    return out_path


# =========================
# STRICT MODEL ARG PARSING (ASLA sessizce default'a düşme)
# =========================
def _models_help_string() -> str:
    lines = []
    for k, v in MODELS.items():
        lines.append(f"- {k}  (ollama='{v.get('ollama')}', label='{v.get('label')}')")
    return "\n".join(lines)

def parse_models_arg(models_arg: str) -> List[str]:
    """
    Kullanıcı --models ile ne verdiyse SADECE onu çalıştır.
    Kabul ettiklerimiz:
      - model key: qwen2_5_3b
      - ollama adı: qwen2.5:3b
      - label: Qwen-2.5:3B
      - kısa ipucu: 'qwen' => içinde qwen geçen ilk model
    Hiç eşleşme yoksa HATA.
    """
    if not models_arg.strip():
        return DEFAULT_MODEL_ORDER[:]

    tokens = [t.strip() for t in models_arg.split(",") if t.strip()]
    out: List[str] = []
    used = set()

    # map'ler
    key_by_ollama = {str(v.get("ollama") or "").strip(): k for k, v in MODELS.items()}
    key_by_label = {str(v.get("label") or "").strip(): k for k, v in MODELS.items()}

    for tok in tokens:
        if tok in MODELS:
            mk = tok
        elif tok in key_by_ollama:
            mk = key_by_ollama[tok]
        elif tok in key_by_label:
            mk = key_by_label[tok]
        else:
            # gevşek eşleşme
            nt = norm_plain(tok)
            candidates = [k for k, v in MODELS.items()
                          if nt in norm_plain(k) or nt in norm_plain(str(v.get("ollama") or "")) or nt in norm_plain(str(v.get("label") or ""))]
            if len(candidates) == 1:
                mk = candidates[0]
            elif len(candidates) > 1:
                # en deterministik: DEFAULT_MODEL_ORDER'da önce gelen
                ordered = [k for k in DEFAULT_MODEL_ORDER if k in candidates] + [k for k in candidates if k not in DEFAULT_MODEL_ORDER]
                mk = ordered[0]
            else:
                raise ValueError(
                    f"--models içinde tanınmayan model: '{tok}'.\n"
                    f"Kullanılabilir modeller:\n{_models_help_string()}\n\n"
                    f"Örnek: --models qwen2_5_3b  veya  --models qwen2.5:3b"
                )

        if mk not in used:
            used.add(mk)
            out.append(mk)

    if not out:
        raise ValueError(
            f"--models boş/yanlış. Kullanılabilir modeller:\n{_models_help_string()}"
        )

    return out


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default="", help="Sadece bu domain çalışsın.")
    ap.add_argument("--topk", type=int, default=TOP_K_DEFAULT, help="Normal sorular için TOPK.")
    ap.add_argument("--models", type=str, default="", help="Model key/ollama/label (virgülle). Örn: qwen2_5_3b veya qwen2.5:3b")
    args = ap.parse_args()

    models_order = parse_models_arg(args.models)  # ARTIK sessizce default'a düşmez
    tag_dicts_raw = safe_load_json(TAG_DICTS_PATH)
    domains_block = get_domains_block(tag_dicts_raw)

    if args.domain.strip():
        target_domains = [args.domain.strip()]
    else:
        target_domains = [
            str(d) for d, p in domains_block.items()
            if isinstance(p, dict) and not is_shared_only_domain(d, p)
        ]
        target_domains.sort()

    client = chroma_client(CHROMA_DIR)

    for dom in target_domains:
        policy = get_domain_policy(tag_dicts_raw, dom) or {}
        ensure_domain_model_dirs(dom, models_order=models_order)

        qspecs = build_question_specs_for_domain(dom, policy)

        # Her domain için, her modelin özetini biriktir
        summary_by_model: Dict[str, List[Dict[str, Any]]] = {mk: [] for mk in models_order}

        print(f"\n>>> DOMAIN BASLADI: {dom}")
        print(f"    Cache: {os.path.join(CACHE_ROOT, sanitize_dir_name(dom))}")
        print(f"    Models: {', '.join(models_order)}")

        for sp in qspecs:
            for mk in models_order:
                res = answer_one_question_for_model(
                    client=client,
                    tag_dicts_raw=tag_dicts_raw,
                    domain=dom,
                    qspec=sp,
                    top_k=int(args.topk),
                    model_key=mk
                )
                ok = res.get("llm_meta", {}).get("ok", False)
                status = "OK" if ok else "FALLBACK_IN_MODEL"
                print(f"  - {sp.get('id')} -> {dom}/{mk} [{status}]")

                # Sadece soru-cevap-chunk_id özetini topla
                summary_by_model[mk].append(compact_qa_record(res))

        # Domain bitti: her model klasörüne qa_summary.json yaz
        for mk in models_order:
            out_path = write_domain_model_summary(dom, mk, summary_by_model[mk])
            print(f"    [SUMMARY WRITTEN] {dom}/{mk} -> {out_path}")

    print("\n[BİTTİ] Seçtiğin modeller çalıştırıldı ve her model klasörüne kaydedildi.")
    print("Ek olarak: CACHE_ROOT/<domain>/<model_key>/qa_summary.json oluşturuldu (soru+cevap+chunk_id).")


if __name__ == "__main__":
    main()
