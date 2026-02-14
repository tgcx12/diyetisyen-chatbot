# -*- coding: utf-8 -*-
import os, re, json, time, hashlib
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

CACHE_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\deneme_cache\qa_demo_kolesterol"
TOP_K = 8

DOMAIN = "kolesterol"

# =========================
# TEXT HELPERS
# =========================
def norm_plain(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("ı","i").replace("ş","s").replace("ğ","g").replace("ü","u").replace("ö","o").replace("ç","c")
    s = re.sub(r"\s+", " ", s)
    return s

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
# DOMAIN POLICY FROM tag_dicts.json
# =========================
def norm_domain_key(s: str) -> str:
    return norm_plain(s).replace(" ", "_")

def get_domains_block(raw: Dict[str, Any]) -> Dict[str, Any]:
    # destek: raw["domains"] veya legacy raw["domain_policies"]
    d1 = raw.get("domains")
    d2 = raw.get("domain_policies")
    merged = {}
    if isinstance(d1, dict):
        merged.update(d1)
    if isinstance(d2, dict):
        merged.update(d2)
    return merged if isinstance(merged, dict) else {}

def get_domain_policy(raw: Dict[str, Any], domain: str) -> Dict[str, Any]:
    domains = get_domains_block(raw)
    if not domains:
        return {}

    dkey = norm_domain_key(domain)
    if dkey in domains and isinstance(domains[dkey], dict):
        return domains[dkey]

    # normalize + alias match
    for k, v in domains.items():
        if not isinstance(v, dict):
            continue
        if norm_domain_key(str(k)) == dkey:
            return v
        aliases = v.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and norm_plain(a) == norm_plain(domain):
                    return v
    return {}

def extract_domain_anchors(policy: Dict[str, Any]) -> List[str]:
    anchors = policy.get("anchors") or []
    if isinstance(anchors, list):
        return [a.strip() for a in anchors if isinstance(a, str) and a.strip()]
    return []

def extract_exclude_if_only(policy: Dict[str, Any]) -> List[str]:
    ex = policy.get("exclude_if_only") or []
    if isinstance(ex, list):
        return [x.strip() for x in ex if isinstance(x, str) and x.strip()]
    return []

def extract_competitor_domains(policy: Dict[str, Any]) -> List[str]:
    dis = policy.get("disambiguation") if isinstance(policy.get("disambiguation"), dict) else {}
    cds = dis.get("competitor_domains") or []
    if isinstance(cds, list):
        return [d.strip() for d in cds if isinstance(d, str) and d.strip()]
    return []

def extract_competitor_anchors(tag_dicts_raw: Dict[str, Any], policy: Dict[str, Any]) -> List[str]:
    """
    policy.disambiguation.competitor_domains -> bu domain'lerin anchors listesini tag_dicts'den toplar
    """
    domains = get_domains_block(tag_dicts_raw)
    out: List[str] = []

    for d in extract_competitor_domains(policy):
        dk = norm_domain_key(d)
        pol = None

        if dk in domains and isinstance(domains[dk], dict):
            pol = domains[dk]
        else:
            for k, v in domains.items():
                if isinstance(v, dict) and norm_domain_key(str(k)) == dk:
                    pol = v
                    break

        if isinstance(pol, dict):
            an = pol.get("anchors") or []
            if isinstance(an, list):
                for a in an:
                    if isinstance(a, str) and a.strip():
                        out.append(a.strip())

    # uniq
    seen = set()
    res = []
    for x in out:
        nx = norm_plain(x)
        if nx and nx not in seen:
            seen.add(nx)
            res.append(x)
    return res

def policy_bool(policy: Dict[str, Any], path: Tuple[str, ...], default: bool = False) -> bool:
    cur: Any = policy
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return bool(cur) if cur is not None else default

# =========================
# QUESTIONS (GENERIC + SAYI KANCALI)
# =========================
GENERIC_CATEGORY_SPECS = [
    ("genel", [
        "{domain} için beslenme genel olarak nasıl olmalı (mümkünse 1 sayısal hedef/üst sınır belirt)",
        "{domain} için genel olarak neler sınırlandırılmalı (günde/haftada/%, gram/mg varsa yaz)",
    ]),
    ("protein", [
        "{domain} için kırmızı et tüketimi en fazla ne olmalı (g/hafta veya porsiyon/hafta) ve hangi türden kaçınılmalı?",
        "{domain} için beyaz et (tavuk/hindi) tüketimi ne olmalı (kez/hafta veya porsiyon/hafta) ve pişirme önerisi var mı?",
        "{domain} için balık tüketimi ne olmalı (kez/hafta veya g/hafta) ve hangi tür/pişirme tercih edilmeli?",
        "{domain} için yumurta tüketimi ne olmalı (adet/hafta veya adet/gün) ve pişirme/yanında tüketim önerisi var mı?",
    ]),
    ("sut", [
        "{domain} için süt ve süt ürünleri tüketimi nasıl olmalı (az yağlı/yağsız; porsiyon/gün varsa)?",
    ]),
    ("tahil", [
        "{domain} için ekmek/tahıl/pilav-makarna tüketimi nasıl olmalı (tam tahıl tercihi + porsiyon/gün varsa)?",
    ]),
    ("baklagil", [
        "{domain} için baklagil tüketimi nasıl olmalı (haftada kaç kez/porsiyon) ve hangi şekilde tercih edilmeli?",
    ]),
    ("yag", [
        "{domain} için doymuş yağ için üst sınır var mı (% enerji gibi) ve hangi yağlar tercih edilmeli?",
        "{domain} için trans yağdan kaçınılmalı mı, hangi ürünlerde bulunur (kanıtta varsa 1-2 örnek)?",
    ]),
    ("posa_lif", [
        "{domain} için posa/lif tüketimi nasıl olmalı (g/gün gibi net sayı varsa) ve hangi kaynaklar önerilir?",
    ]),
    ("tuz_seker", [
        "{domain} için tuz/sodyum tüketimi nasıl olmalı (mg sodyum/g tuz üst sınır varsa) ve hangi ürünlerden kaçınılmalı?",
        "{domain} için rafine şeker/şekerli içecek tüketimi nasıl olmalı (sıklık/üst sınır varsa) ve yerine ne tercih edilmeli?",
    ]),
]

def build_questions(domain: str) -> List[str]:
    out: List[str] = []
    for _, templs in GENERIC_CATEGORY_SPECS:
        for t in templs:
            out.append(t.format(domain=domain).strip())
    return out

# =========================
# CACHE
# =========================
class DiskCache:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.base_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, obj: Any) -> None:
        try:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

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
# STRICT RESULT FILTER
# =========================
def contains_any(text: str, terms: List[str]) -> bool:
    low = norm_plain(text)
    for t in terms:
        nt = norm_plain(t)
        if nt and nt in low:
            return True
    return False

def filter_chunks_strict(
    chunks: List[Dict[str, Any]],
    anchors: List[str],
    competitor_anchors: List[str],
    exclude_if_only: List[str],
    require_anchor_in_chunk: bool,
    strict_competitor_exclusion: bool
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        blob = f"{ch.get('ana_baslik','')} {ch.get('topic_group','')} {ch.get('section','')} {ch.get('content','')}"
        has_anchor = contains_any(blob, anchors) if anchors else False
        has_comp = contains_any(blob, competitor_anchors) if competitor_anchors else False
        has_excl = contains_any(blob, exclude_if_only) if exclude_if_only else False

        if require_anchor_in_chunk and not has_anchor:
            continue
        if strict_competitor_exclusion and has_comp:
            continue
        if (not has_anchor) and has_excl:
            continue

        out.append(ch)
    return out

# =========================
# NUMERIC + QUAL HINTS
# =========================
NUMERIC_CUE_RE = re.compile(
    r"(\b(gunde|günde|gunluk|günlük|haftada|ayda)\b|(\d+(?:[.,]\d+)?)\s*(adet|kez|porsiyon|g|gr|gram|mg)\b|(\d+(?:[.,]\d+)?)\s*%|%\s*(\d+(?:[.,]\d+)?))",
    re.IGNORECASE
)

QUAL_CUE_RE = re.compile(
    r"\b(tercih|oner|öner|sinir|sınır|sinirlandir|sınırlandır|kacin|kaçın|azalt|artir|arttır|yerine|seç|sec|tuket|tüket|hazir|paketli|kizart|kızart|islenmis|işlenmiş|margarin|fast[- ]?food|hamur isi|hamur işi)\b",
    re.IGNORECASE
)

SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")

def extract_sentence_hints(text: str, matcher: re.Pattern, max_hints: int = 10, max_len: int = 260) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    sents = SENT_SPLIT_RE.split(t.replace("\n", " "))
    hints: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if matcher.search(s):
            if len(s) > max_len:
                s = s[:max_len].rstrip()
            hints.append(s)
        if len(hints) >= max_hints:
            break
    # uniq
    seen = set()
    out = []
    for h in hints:
        nh = norm_plain(h)
        if nh and nh not in seen:
            seen.add(nh)
            out.append(h)
    return out[:max_hints]

def score_chunk(ch: Dict[str, Any], anchors: List[str]) -> int:
    blob = f"{ch.get('ana_baslik','')} {ch.get('topic_group','')} {ch.get('section','')} {ch.get('content','')}"
    sc = 0
    if contains_any(blob, anchors):
        sc += 2
    if NUMERIC_CUE_RE.search(blob or ""):
        sc += 3
    if QUAL_CUE_RE.search(blob or ""):
        sc += 1
    return sc

# =========================
# LLM
# =========================
def ollama_chat(model: str, system: str, user: str, num_ctx: int, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": float(temperature), "num_ctx": int(num_ctx)}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content") or ""

def build_system_prompt(competitor_terms: List[str], hard_lock: bool) -> str:
    comp_hint = ", ".join(sorted(set([c for c in competitor_terms if c]))[:40])

    if hard_lock:
        lock = (
            "Eğer NUMERIC_HINTS doluysa cevabında NUMERIC_HINTS içinden EN AZ 1 ifadeyi AYNEN kullanmak ZORUNDASIN.\n"
            "Eğer QUAL_HINTS doluysa cevabında QUAL_HINTS içinden EN AZ 1 ifadeyi (anlamını bozmadan) kullanmak ZORUNDASIN.\n"
        )
    else:
        lock = (
            "Eğer kanıtta sayısal ifade varsa cevabında mutlaka kullan.\n"
            "Eğer kanıtta nitel öneri (tercih/kaçın/limit vb.) varsa cevabında en az 1 nitel öneri kullan.\n"
        )

    return (
        "Sen kanıt-temelli klinik diyetisyen asistansın.\n"
        "SADECE verilen EVIDENCE, NUMERIC_HINTS ve QUAL_HINTS içinden konuş.\n"
        "ÇIKTI: TEK CÜMLE. (İki parçayı noktalı virgül ';' ile bağlayabilirsin.)\n"
        + lock +
        "Soru 'hangi ürünlerde bulunur' diyorsa ve EVIDENCE/QUAL_HINTS içinde örnek ürün geçiyorsa tek cümlede 1-2 örnek ürün yaz.\n"
        "Yeni bilgi uydurma.\n"
        f"ÖNEMLİ: Cevapta başka hastalık isimleri/competitor terimleri ASLA yazma (örn: {comp_hint}).\n"
        "Asla madde işareti kullanma, JSON yazma."
    )

def make_boosted_query(domain: str, q: str, anchors: List[str]) -> str:
    boost = (
        " haftada günde gunluk adet kez porsiyon gram mg yüzde % "
        "en fazla geçmemeli sınır sınırlandırılmalı önerilir tercih edilmeli kaçınılmalı "
        "kızartma paketli işlenmiş margarin fast-food"
    )
    rq = f"{q} {boost}".strip()
    low = norm_plain(rq)
    if anchors and not any(norm_plain(a) in low for a in anchors):
        rq = f"{domain} {rq}"
    return rq

# =========================
# CORE
# =========================
def answer_one_question(
    cache: DiskCache,
    client: chromadb.Client,
    tag_dicts_raw: Dict[str, Any],
    domain: str,
    question: str,
    top_k: int
) -> Dict[str, Any]:
    policy = get_domain_policy(tag_dicts_raw, domain) or {}
    anchors = extract_domain_anchors(policy) or [domain]
    exclude_if_only = extract_exclude_if_only(policy)
    competitor_terms_flat = exclude_if_only[:]  # cevapta yazmasın
    competitor_anchors = extract_competitor_anchors(tag_dicts_raw, policy)

    require_anchor_in_chunk = policy_bool(policy, ("disambiguation", "require_anchor_for_retrieval_queries"), True)
    strict_competitor_excl = policy_bool(policy, ("disambiguation", "strict_competitor_exclusion"), True)

    # 1) retrieval cache
    rkey = "ret_" + _sha1(json.dumps(
        {"d": domain, "q": question, "k": top_k, "anchors": anchors, "strict": True},
        ensure_ascii=False, sort_keys=True
    ))
    chunks = cache.get(rkey)
    if not (isinstance(chunks, list) and chunks):
        rq = make_boosted_query(domain, question, anchors)
        raw_chunks = dedupe_by_chunk_id(chroma_retrieve(client, RAG_COLLECTION, rq, top_k=top_k))
        strict_chunks = filter_chunks_strict(
            raw_chunks,
            anchors=anchors,
            competitor_anchors=competitor_anchors,
            exclude_if_only=exclude_if_only,
            require_anchor_in_chunk=require_anchor_in_chunk,
            strict_competitor_exclusion=strict_competitor_excl
        )
        if not strict_chunks and raw_chunks:
            # fallback: sadece anchor içerenleri tut
            strict_chunks = [ch for ch in raw_chunks if contains_any(
                f"{ch.get('ana_baslik','')} {ch.get('topic_group','')} {ch.get('section','')} {ch.get('content','')}",
                anchors
            )][:max(1, top_k // 2)]

        chunks = strict_chunks
        cache.set(rkey, chunks)

    # 2) answer cache
    akey = "ans_" + _sha1(json.dumps(
        {"d": domain, "q": question, "k": top_k, "m": MODEL, "ctx": NUM_CTX},
        ensure_ascii=False, sort_keys=True
    ))
    hit = cache.get(akey)
    if isinstance(hit, dict) and isinstance(hit.get("answer"), str) and hit["answer"].strip():
        return hit

    # evidence pack (score -> numeric+qual öne)
    chunks_sorted = sorted(chunks, key=lambda c: score_chunk(c, anchors), reverse=True)

    evidence: List[Dict[str, str]] = []
    numeric_hints: List[str] = []
    qual_hints: List[str] = []

    for ch in chunks_sorted[:min(len(chunks_sorted), 12)]:
        txt = (ch.get("content") or "").strip()
        if not txt:
            continue
        numeric_hints.extend(extract_sentence_hints(txt, NUMERIC_CUE_RE, max_hints=8))
        qual_hints.extend(extract_sentence_hints(txt, QUAL_CUE_RE, max_hints=8))
        evidence.append({"chunk_id": ch.get("chunk_id"), "text": txt[:900]})
        if len(evidence) >= 8:
            break

    def uniq_keep(seq: List[str], limit: int) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            nx = norm_plain(x)
            if nx and nx not in seen:
                seen.add(nx)
                out.append(x)
            if len(out) >= limit:
                break
        return out

    numeric_hints = uniq_keep(numeric_hints, 10)
    qual_hints = uniq_keep(qual_hints, 10)

    has_num_in_evidence = bool(numeric_hints) or any(NUMERIC_CUE_RE.search(e["text"] or "") for e in evidence)
    has_qual_in_evidence = bool(qual_hints) or any(QUAL_CUE_RE.search(e["text"] or "") for e in evidence)

    payload = json.dumps({
        "domain": domain,
        "question": question,
        "NUMERIC_HINTS": numeric_hints,
        "QUAL_HINTS": qual_hints,
        "EVIDENCE": evidence
    }, ensure_ascii=False)

    # attempt 1
    sys1 = build_system_prompt(competitor_terms_flat, hard_lock=False)
    resp1 = ollama_chat(MODEL, sys1, payload, num_ctx=NUM_CTX, temperature=0.0).strip()

    def ok_numeric(ans: str) -> bool:
        return bool(NUMERIC_CUE_RE.search(ans or ""))

    def ok_qual(ans: str) -> bool:
        return bool(QUAL_CUE_RE.search(ans or ""))

    need_retry = False
    if has_num_in_evidence and not ok_numeric(resp1):
        need_retry = True
    if has_qual_in_evidence and not ok_qual(resp1):
        need_retry = True

    final = resp1
    used_retry = False

    if need_retry:
        used_retry = True
        sys2 = build_system_prompt(competitor_terms_flat, hard_lock=True)
        resp2 = ollama_chat(MODEL, sys2, payload, num_ctx=NUM_CTX, temperature=0.0).strip()
        if resp2:
            final = resp2

    out = {
        "question": question,
        "answer": final,
        "used_chunks": [e["chunk_id"] for e in evidence],
        "strict_chunks_count": len(chunks),
        "numeric_in_evidence": has_num_in_evidence,
        "numeric_in_answer": ok_numeric(final),
        "qual_in_evidence": has_qual_in_evidence,
        "qual_in_answer": ok_qual(final),
        "numeric_hints": numeric_hints,
        "qual_hints": qual_hints,
        "used_retry": used_retry,
    }

    cache.set(akey, out)
    return out

# =========================
# MAIN
# =========================
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = DiskCache(CACHE_DIR)
    client = chroma_client(CHROMA_DIR)

    tag_dicts_raw = safe_load_json(TAG_DICTS_PATH)

    questions = build_questions(DOMAIN)
    results: List[Dict[str, Any]] = []

    for q in questions:
        results.append(answer_one_question(cache, client, tag_dicts_raw, DOMAIN, q, top_k=TOP_K))

    print("\n=== TEK CÜMLE CEVAPLAR (NUMERIC+QUAL LOCK + STRICT FILTER) ===\n")
    for r in results:
        print(f"- S: {r['question']}")
        print(f"  C: {r['answer']}")
        print(f"  strict_chunks: {r.get('strict_chunks_count')} | retry: {r.get('used_retry')}")
        print(f"  chunks: {', '.join(r['used_chunks'][:4])}{' ...' if len(r['used_chunks'])>4 else ''}")
        print(f"  numeric evidence? {r['numeric_in_evidence']} | numeric answer? {r['numeric_in_answer']}")
        print(f"  qual evidence? {r['qual_in_evidence']} | qual answer? {r['qual_in_answer']}\n")

        if (r["numeric_in_evidence"] and not r["numeric_in_answer"]) or (r["qual_in_evidence"] and not r["qual_in_answer"]):
            print("  [ALERT] Evidence var ama cevap kaçırdı! Hints preview:")
            for h in (r.get("numeric_hints") or [])[:2]:
                print("   - NUM:", h)
            for h in (r.get("qual_hints") or [])[:2]:
                print("   - QUA:", h)
            print()

    out_path = os.path.join(CACHE_DIR, f"{DOMAIN}_answers_{int(time.time())}.json")
    safe_dump_json(results, out_path)
    print("SAVED:", out_path)
    print("CACHE_DIR:", CACHE_DIR)

if __name__ == "__main__":
    main()
