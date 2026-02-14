# -*- coding: utf-8 -*-
import os
import re
import json
import time
import argparse
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import requests
import chromadb

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-q4_K_M"
NUM_CTX = 8192

CHROMA_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\chroma_db_storage"
RAG_COLLECTION = "diyetisyen_rehberi"

TAG_DICTS_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\tag_dicts.json"
CACHE_ROOT = r"C:\Users\user\Desktop\diyetisyen_llm\deneme_cache\qa_demo"

TOP_K_DEFAULT = 24

# Entity sorular: “olabildiğince tüm ilgili chunklar”
ENTITY_EXPAND_TOPK = 350
ENTITY_MAX_CHUNKS_FOR_SUMMARY = 160
ENTITY_BATCH_SIZE = 12

# Normal sorular
MAX_EVIDENCE_CHUNKS = 12
MAX_SCAN_CHUNKS = 150

# Snippet / evidence
MAX_EVIDENCE_SENTENCES_PER_CHUNK = 8
FALLBACK_FIRST_SENTENCES = 3
MAX_EVIDENCE_CHARS_PER_CHUNK = 1100

# Output
RULE_SEPARATOR = " | "
MAX_RULES_OUT = 6  # 8 çok şişiriyordu; 6 daha stabil

# Cache
CACHE_VERSION = "v13_kolesterol_rev_topic_table_leakfix_entity_snippet_guard_kuruyemis"

# Domain marker şartı
REQUIRE_DOMAIN_MARKER_IN_CHUNK = True
DOMAIN_MARKER_META_ONLY = True

# Debug
DEBUG_PRINT_LLM_PAYLOAD = False  # True yaparsan console’a evidence snippet’lerini basar


# =========================
# QUESTION SPECS (temiz sorular, ekleme yok)
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
            "content_type": md.get("content_type"),  # <-- table filtre için kritik
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
    return ct == "table"

def filter_chunks_strict(
    chunks: List[Dict[str, Any]],
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
        # 0) TABLE içerikleri asla alma
        if is_table_chunk(ch):
            continue

        meta_blob = build_meta_blob(ch)
        full_blob = f"{meta_blob} {ch.get('content','')}"

        has_anchor = contains_any(meta_blob, anchors) if anchors else False
        has_entity = contains_any(full_blob, entity_terms) if entity_terms else False

        has_comp = contains_any(full_blob, competitor_anchors) if competitor_anchors else False
        has_excl = contains_any(full_blob, exclude_if_only) if exclude_if_only else False

        # 0.5) topic_group doluysa domain marker ile uyumsuzsa ele (karışmaları keser)
        tg = str(ch.get("topic_group") or "").strip()
        if tg:
            # topic_group, domain marker taşımıyorsa ve içerik de domain marker taşımıyorsa ele
            if (not contains_any(tg, domain_markers)) and (not contains_any(full_blob, domain_markers)):
                continue

        # 1) domain marker meta şartı
        if require_domain_marker_in_chunk:
            if not has_domain_marker_meta_only(ch, domain_markers):
                continue

        # 2) competitor dışlama
        if strict_competitor_exclusion and has_comp and (not has_anchor) and (not has_entity):
            continue

        # 3) exclude_if_only
        if (not has_anchor) and has_excl:
            continue

        # 4) anchor şartı (entity varsa anchor zorunlu değil)
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

def normalize_bullets_for_sentence_split(t: str) -> str:
    t = (t or "")
    t = t.replace("•", ". ")
    t = re.sub(r"\s*-\s+", ". ", t)   # "- " maddeleri ayır
    t = t.replace(";", ". ")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def split_sentences(text: str) -> List[str]:
    t = normalize_bullets_for_sentence_split(text)
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
    """
    1) Önce entity içeren cümleleri arar.
    2) Yetmezse numeric/qual cümleleri ekler.
    3) Hâlâ yoksa ilk cümlelerden fallback.
    """
    txt = (txt or "").strip()
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
    return any(re.search(p, low, flags=re.IGNORECASE) for p in _LEAK_PATTERNS)

def strip_leaked_instructions(s: str) -> str:
    s = (s or "").strip()

    # başlık/konu/içerik gibi alanları ve "kesin kurallar" anlatımlarını temizle
    s = re.sub(r"(?i)başlık\s*:\s*", "", s)
    s = re.sub(r"(?i)konu\s*:\s*", "", s)
    s = re.sub(r"(?i)içerik\s*:\s*", "", s)

    # "Kesin kurallar: ...." gibi sızıntıyı kırp
    m = re.search(r"(?i)(\bkesin kurallar\b|\bkesintisiz kurallar\b)", s)
    if m:
        # sızıntı varsa, sonra ilk gerçek maddeyi arayıp oradan itibaren al
        for key in ["|", "•", "-", "1)", "1."]:
            idx = s.find(key, m.start())
            if idx != -1:
                s = s[idx:]
                break
        s = re.sub(r"(?i)\bkesin kurallar\b.*?(?=\||$)", "", s).strip()

    # "Kaynaklarda..." segmentini her zaman temizle (evidence boşken zaten LLM çağırmıyoruz)
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

    # baştaki dash/bullet kırıntıları
    s = re.sub(r"\s*-\s*", " ", s)

    # parçala
    parts = [p.strip(" -–—\t ").strip() for p in re.split(r"\s*\|\s*", s) if p.strip()]

    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"[.!?]+", s) if p.strip()]

    parts = merge_unique(parts, limit=MAX_RULES_OUT)

    out = f" {RULE_SEPARATOR} ".join(parts) if parts else s
    out = re.sub(r"\s+\|\s+\|\s+", f" {RULE_SEPARATOR} ", out)
    out = re.sub(r"\s+\|\s*$", "", out).strip()
    return out

def postprocess_answer(s: str) -> str:
    s = (s or "").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("“") and s.endswith("”")):
        s = s[1:-1].strip()
    s = s.replace('\\"', '"').strip()
    s = re.sub(r"\s+", " ", s).strip()

    s = strip_leaked_instructions(s)
    s = force_one_line_and_separator(s)
    return s


# =========================
# EXTRACTIVE FALLBACK (NO HALLUCINATION)
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
    if not text:
        return "Kaynaklarda bu soru için bilgi bulunamadı."

    sents = split_sentences(text)
    picked: List[str] = []
    seen = set()

    def add(s: str):
        ns = norm_plain(s)
        if ns and ns not in seen:
            seen.add(ns)
            picked.append(s)

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
        + numeric_rule + qual_rule + must_rule
    )

def llm_summarize_from_evidence(
    domain: str,
    question: str,
    want_numeric_orig: bool,
    want_qual_orig: bool,
    hints: List[str],
    evidence: List[Dict[str, str]],
    must_terms: List[str]
) -> str:
    if not evidence:
        return "Kaynaklarda bu soru için bilgi bulunamadı."

    blob = " ".join(hints) + " " + " ".join([e.get("text","") for e in evidence])
    allow_numeric = bool(NUMERIC_UNIT_RE.search(blob))

    sys_prompt = build_system_prompt(
        want_numeric_orig=want_numeric_orig,
        want_qual_orig=want_qual_orig,
        allow_numeric=allow_numeric and want_numeric_orig,
        must_mention_terms=must_terms
    )

    payload_obj = {
        "domain": domain,
        "question": question,
        "HINTS": hints,
        "EVIDENCE": evidence
    }
    payload = json.dumps(payload_obj, ensure_ascii=False)

    if DEBUG_PRINT_LLM_PAYLOAD:
        print("\n=== DEBUG LLM PAYLOAD ===")
        print("Q:", question)
        print("Evidence count:", len(evidence))
        for i, ev in enumerate(evidence[:5]):
            print(f"- ev[{i}] ({ev.get('chunk_id')}): {ev.get('text','')[:180]} ...")
        print("=== END DEBUG ===\n")

    raw = ollama_chat(MODEL, sys_prompt, payload, num_ctx=NUM_CTX, temperature=0.0)
    ans = postprocess_answer(raw)

    # Evidence dolu olduğu halde model kaçış yaptıysa / talimat sızdırdıysa extractive override
    if evidence:
        low = ans.strip().lower()
        if low == "kaynaklarda bu soru için bilgi bulunamadı." or "kaynaklarda bu soru için bilgi bulunamadı" in low:
            return extractive_fallback_one_line(evidence, want_numeric_orig, want_qual_orig)
        if looks_like_instruction_leak(raw) or looks_like_instruction_leak(ans):
            return extractive_fallback_one_line(evidence, want_numeric_orig, want_qual_orig)
        if len(ans.strip()) < 8:
            return extractive_fallback_one_line(evidence, want_numeric_orig, want_qual_orig)

    return ans


# =========================
# QUERY (ekleme yok!)
# =========================
def make_query(question: str) -> str:
    return normalize_query_for_retrieval(question)


# =========================
# RETRIEVAL
# =========================
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
        domain_markers=domain_markers,
        anchors=anchors,
        competitor_anchors=competitor_anchors,
        exclude_if_only=exclude_if_only,
        require_anchor_in_chunk=require_anchor_in_chunk,
        strict_competitor_exclusion=strict_competitor_exclusion,
        entity_terms=entity_terms,
        require_domain_marker_in_chunk=REQUIRE_DOMAIN_MARKER_IN_CHUNK
    )

    # entity hard filter
    strict = hard_filter_by_entity(strict, entity_terms)

    meta = {"retrieval_attempts": len(attempts), "attempts": attempts}
    return strict, meta


# =========================
# MAP-REDUCE (robust)
# =========================
def map_reduce_summary(
    domain: str,
    question: str,
    want_numeric_orig: bool,
    want_qual_orig: bool,
    chunks_sorted: List[Dict[str, Any]],
    entity_terms: List[str],
    exclude_terms: List[str],
    max_chunks: int,
    batch_size: int
) -> Tuple[str, List[Dict[str, str]]]:
    evidence_all: List[Dict[str, str]] = []
    hints_all: List[str] = []

    for ch in chunks_sorted[:max_chunks]:
        txt = (ch.get("content") or "").strip()
        if not txt:
            continue

        snippet, n_h, q_h = select_evidence_snippet(
            txt, want_numeric_orig, want_qual_orig, entity_terms, exclude_terms
        )
        if not snippet:
            continue

        # entity sorularda: snippet entity içermiyorsa at
        if entity_terms and not contains_any(snippet, entity_terms):
            continue

        evidence_all.append({"chunk_id": ch.get("chunk_id"), "text": snippet})
        hints_all.extend(n_h)
        hints_all.extend(q_h)

    if not evidence_all:
        return "Kaynaklarda bu soru için bilgi bulunamadı.", []

    hints_all = merge_unique(hints_all, limit=18)

    batch_summaries: List[str] = []
    for i in range(0, len(evidence_all), batch_size):
        batch_evidence = evidence_all[i:i+batch_size]
        batch_text = " ".join([e["text"] for e in batch_evidence])

        batch_hints = []
        for s in split_sentences(batch_text):
            if NUMERIC_UNIT_RE.search(s):
                batch_hints.append(s)
            elif QUAL_CUE_RE.search(s):
                batch_hints.append(s)
        batch_hints = merge_unique(batch_hints, limit=14)

        bs = llm_summarize_from_evidence(
            domain=domain,
            question=question,
            want_numeric_orig=want_numeric_orig,
            want_qual_orig=want_qual_orig,
            hints=batch_hints,
            evidence=batch_evidence,
            must_terms=entity_terms
        )

        if bs.strip().lower() == "kaynaklarda bu soru için bilgi bulunamadı.":
            continue
        batch_summaries.append(bs)

    if not batch_summaries:
        return extractive_fallback_one_line(evidence_all, want_numeric_orig, want_qual_orig), evidence_all

    reduce_evidence = [{"chunk_id": f"batch_{i}", "text": s} for i, s in enumerate(batch_summaries)]
    reduce_hints = batch_summaries[:14]

    final = llm_summarize_from_evidence(
        domain=domain,
        question=question,
        want_numeric_orig=want_numeric_orig,
        want_qual_orig=want_qual_orig,
        hints=reduce_hints,
        evidence=reduce_evidence,
        must_terms=entity_terms
    )

    if final.strip().lower() == "kaynaklarda bu soru için bilgi bulunamadı.":
        final = extractive_fallback_one_line(evidence_all, want_numeric_orig, want_qual_orig)

    return final, evidence_all


# =========================
# Disk Cache
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


# =========================
# Core Answer
# =========================
def build_question_specs_for_domain(domain: str) -> List[Dict[str, Any]]:
    return [{**sp, "question": sp["text"].format(domain=domain).strip()} for sp in QUESTION_SPECS]

def answer_one_question(
    client: chromadb.Client,
    tag_dicts_raw: Dict[str, Any],
    domain: str,
    qspec: Dict[str, Any],
    top_k: int
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

    domain_cache_dir = os.path.join(CACHE_ROOT, domain)
    os.makedirs(domain_cache_dir, exist_ok=True)
    cache = DiskCache(domain_cache_dir)

    akey = "ans_" + _sha1(json.dumps({
        "ver": CACHE_VERSION,
        "d": domain, "q": question,
        "m": MODEL, "ctx": NUM_CTX,
        "k": top_k,
        "wn": want_numeric_orig, "wq": want_qual_orig,
        "entity": entity_terms[:20], "ex": exclude_terms[:20],
        "req_domain_marker": REQUIRE_DOMAIN_MARKER_IN_CHUNK,
        "meta_only": DOMAIN_MARKER_META_ONLY,
        "max_rules": MAX_RULES_OUT,
    }, ensure_ascii=False, sort_keys=True))

    hit = cache.get(akey)
    if isinstance(hit, dict) and isinstance(hit.get("answer"), str) and hit["answer"].strip():
        return hit

    # Retrieval
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
            "used_retry": False,
            "qspec": {"id": qspec.get("id"), "category": qspec.get("category"),
                      "want_numeric": want_numeric_orig, "want_qual": want_qual_orig},
        }
        cache.set(akey, out)
        return out

    # Sort
    chunks_sorted = sorted(
        chunks,
        key=lambda c: score_chunk(c, domain_markers, anchors, want_numeric_orig, want_qual_orig, entity_terms),
        reverse=True
    )

    evidence: List[Dict[str, str]] = []
    hints: List[str] = []

    # Entity sorularda map-reduce (çok chunk varsa)
    if entity_terms and len(chunks_sorted) > MAX_EVIDENCE_CHUNKS:
        final, evidence_all = map_reduce_summary(
            domain=domain,
            question=question,
            want_numeric_orig=want_numeric_orig,
            want_qual_orig=want_qual_orig,
            chunks_sorted=chunks_sorted,
            entity_terms=entity_terms,
            exclude_terms=exclude_terms,
            max_chunks=min(len(chunks_sorted), ENTITY_MAX_CHUNKS_FOR_SUMMARY),
            batch_size=ENTITY_BATCH_SIZE
        )
        used_evidence = evidence_all
    else:
        # normal akış
        for ch in chunks_sorted[:min(len(chunks_sorted), MAX_SCAN_CHUNKS)]:
            txt = (ch.get("content") or "").strip()
            if not txt:
                continue

            snippet, n_h, q_h = select_evidence_snippet(
                txt, want_numeric_orig, want_qual_orig, entity_terms, exclude_terms
            )
            if not snippet:
                continue

            # entity sorularda: snippet entity içermiyorsa at
            if entity_terms and not contains_any(snippet, entity_terms):
                continue

            evidence.append({"chunk_id": ch.get("chunk_id"), "text": snippet})
            hints.extend(n_h)
            hints.extend(q_h)
            if len(evidence) >= MAX_EVIDENCE_CHUNKS:
                break

        hints = merge_unique(hints, limit=16)
        used_evidence = evidence

        if not used_evidence:
            final = "Kaynaklarda bu soru için bilgi bulunamadı."
        else:
            final = llm_summarize_from_evidence(
                domain=domain,
                question=question,
                want_numeric_orig=want_numeric_orig,
                want_qual_orig=want_qual_orig,
                hints=hints,
                evidence=used_evidence,
                must_terms=entity_terms
            )

    # Final safety: evidence doluysa asla "bulunamadı" bırakma
    if used_evidence:
        low = final.strip().lower()
        if low == "kaynaklarda bu soru için bilgi bulunamadı." or "kaynaklarda bu soru için bilgi bulunamadı" in low:
            final = extractive_fallback_one_line(used_evidence, want_numeric_orig, want_qual_orig)
        if looks_like_instruction_leak(final):
            final = extractive_fallback_one_line(used_evidence, want_numeric_orig, want_qual_orig)

    final = postprocess_answer(final)

    ev_text = " ".join([e.get("text","") for e in used_evidence]) if used_evidence else ""
    numeric_in_evidence = bool(NUMERIC_UNIT_RE.search(ev_text))
    qual_in_evidence = bool(QUAL_CUE_RE.search(ev_text))
    numeric_in_answer = bool(NUMERIC_UNIT_RE.search(final))
    qual_in_answer = bool(QUAL_CUE_RE.search(final))

    out = {
        "domain": domain,
        "question": question,
        "answer": final,
        "used_chunks": [e["chunk_id"] for e in used_evidence],
        "strict_chunks_count": len(chunks),
        "retrieval_meta": retrieval_meta,
        "numeric_in_evidence": numeric_in_evidence,
        "numeric_in_answer": numeric_in_answer,
        "qual_in_evidence": qual_in_evidence,
        "qual_in_answer": qual_in_answer,
        "used_retry": False,
        "qspec": {"id": qspec.get("id"), "category": qspec.get("category"),
                  "want_numeric": want_numeric_orig, "want_qual": want_qual_orig},
    }
    cache.set(akey, out)
    return out


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default="", help="Sadece bu domain çalışsın (örn: kolesterol).")
    ap.add_argument("--topk", type=int, default=TOP_K_DEFAULT, help="Normal sorular için TOPK. Entity sorularda otomatik büyür.")
    args = ap.parse_args()

    tag_dicts_raw = safe_load_json(TAG_DICTS_PATH)
    domains_block = get_domains_block(tag_dicts_raw)

    if not isinstance(domains_block, dict) or not domains_block:
        raise RuntimeError("tag_dicts.json içinde domains/domain_policies bulunamadı veya boş.")

    if args.domain.strip():
        target_domains = [args.domain.strip()]
    else:
        target_domains = []
        for dkey, pol in domains_block.items():
            if not isinstance(pol, dict):
                continue
            if is_shared_only_domain(dkey, pol):
                continue
            target_domains.append(str(dkey))
        target_domains.sort()

    client = chroma_client(CHROMA_DIR)

    all_results: List[Dict[str, Any]] = []
    for dom in target_domains:
        qspecs = build_question_specs_for_domain(dom)
        results: List[Dict[str, Any]] = []
        for sp in qspecs:
            results.append(answer_one_question(client, tag_dicts_raw, dom, sp, top_k=int(args.topk)))

        domain_dir = os.path.join(CACHE_ROOT, dom)
        os.makedirs(domain_dir, exist_ok=True)
        out_path = os.path.join(domain_dir, f"{dom}_answers_{int(time.time())}.json")
        safe_dump_json(results, out_path)

        print(f"\n=== DOMAIN: {dom} ===")
        print("SAVED:", out_path)
        for r in results:
            print(f"- S: {r['question']}")
            print(f"  C: {r['answer']}")
            print(f"  chunks: {r.get('strict_chunks_count')} | attempts: {r.get('retrieval_meta',{}).get('retrieval_attempts')}")
        all_results.extend(results)

    summary_path = os.path.join(CACHE_ROOT, f"_ALL_domains_answers_{int(time.time())}.json")
    safe_dump_json(all_results, summary_path)
    print("\nALL_SAVED:", summary_path)
    print("CACHE_ROOT:", CACHE_ROOT)
    print("CACHE_VERSION:", CACHE_VERSION)
    print("REQUIRE_DOMAIN_MARKER_IN_CHUNK:", REQUIRE_DOMAIN_MARKER_IN_CHUNK)
    print("DOMAIN_MARKER_META_ONLY:", DOMAIN_MARKER_META_ONLY)

if __name__ == "__main__":
    main()
