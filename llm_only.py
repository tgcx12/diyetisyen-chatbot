# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
from typing import Any, Dict, List

import httpx
import pandas as pd

# =========================
# CONFIG (SENİN YOLLARIN)
# =========================
TEST_SET_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\test_seti.json"
RULES_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\rules.txt"
OUT_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\benchmark_out_llm_only"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 600

MODELS = [
    "llama3.1:8b",
    "gemma3:4b",
    "qwen2.5:3b",
    "gemma2:2b",
]

# EM modu:
# - "strict": normalize+strip sonrası birebir eşitlik (çoğu generatif senaryoda 0'a yakın çıkar)
# - "relaxed": gold normalize edilmiş metin, pred içinde geçerse (veya tersi) 1 say
# - "token_f1": token-F1, threshold üstündeyse 1 say (daha dengeli)
EM_MODE = "token_f1"     # "strict" | "relaxed" | "token_f1"
EM_F1_THRESHOLD = 0.35   # LLM'lerde 0.90 aşırı serttir


# =========================
# FILE HELPERS
# =========================
def ensure_dir(p: str) -> None:
    if not p:
        return
    os.makedirs(p, exist_ok=True)

def read_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# =========================
# TEXT NORMALIZATION
# =========================
TR_FOLD_MAP = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "I": "i", "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})

def fold_tr(s: str) -> str:
    return (s or "").translate(TR_FOLD_MAP)

def normalize_text(s: str, fold_ascii: bool = True) -> str:
    s = (s or "").strip()
    if fold_ascii:
        s = fold_tr(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s%]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_citations_and_meta(s: str) -> str:
    """
    Model çıktılarından:
    - [KAYNAK 1] / [Kaynak: 2] / (KAYNAK 3) / Kaynak 4 / K1 vb.
    - id=..., doc_id=..., section=..., topic=...
    gibi meta parçalarını temizler.
    """
    if not s:
        return ""
    t = s

    # code blocks
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)

    # kaynak varyantları
    citation_pats = [
        r"\[(?:kayn[aa]k|source|src)\s*:?\s*\d+\]",
        r"\((?:kayn[aa]k|source|src)\s*:?\s*\d+\)",
        r"\b(?:kayn[aa]k)\s*:?\s*\d+\b",
        r"\bk[\s_\-]*\d+\b",
    ]
    for pat in citation_pats:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)

    # meta alanları
    t = re.sub(r"\bid\s*=\s*[^\s|]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdoc_id\s*=\s*[^\s|]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bsection\s*=\s*[^|\n]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\btopic\s*=\s*[^\s|]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(baslik|başlık)\s*:\s*[^.\n]+", " ", t, flags=re.IGNORECASE)

    # "Kaynaklar: ..." satırları
    t = re.sub(r"(?im)^\s*(kaynaklar|references|sources)\s*:.*$", " ", t)

    t = re.sub(r"\s+", " ", t).strip()
    return t


# =========================
# TOKENIZATION
# =========================
def tokenize(s: str) -> List[str]:
    s = normalize_text(s, fold_ascii=True)
    return re.findall(r"[a-z]+|\d+(?:[\.,]\d+)?", s)

TR_STOPWORDS = {
    "ve", "veya", "ile", "da", "de", "ki", "mi", "mı", "mu", "mü",
    "icin", "gibi", "daha", "en", "cok", "az", "bir", "bu", "su", "o",
    "olan", "olarak", "ise", "ama", "fakat", "ancak", "cunku", "diye",
    "nedir", "nelerdir", "nasil", "ne", "neler", "hangi", "kim", "kime",
    "kadar", "sonra", "once", "her", "tum", "butun",
    "yardimci", "olur", "olabilir", "gerekmektedir", "gereklidir",
    "tercih", "edilmelidir", "edilir", "etmelidir", "etmek", "etmeli",
    "onemli", "onemlidir", "mumkun", "mumkunse", "gore", "uzere", "sey",
    "tuketilmesi", "tuketimi", "tuketimin", "tuketiminin",
}

def content_tokens(s: str) -> List[str]:
    toks = tokenize(s)
    out: List[str] = []
    for t in toks:
        if re.fullmatch(r"\d+(?:[\.,]\d+)?", t):
            continue
        if len(t) <= 2:
            continue
        if t in TR_STOPWORDS:
            continue
        out.append(t)
    return out


# =========================
# FUZZY TERM MATCH (TR suffix tolerant)
# =========================
def term_to_fuzzy_regex(term_norm: str) -> str:
    """
    Normalize edilmiş terimi Türkçe ekleri tolere edecek şekilde regex'e çevirir.
    - tek kelime: \\bterm\\w*\\b
    - çok kelime: boşluklar \\s+ toleranslı; son kelime \\w*\\b ile uzatılır
    """
    words = (term_norm or "").split()
    if not words:
        return r"^$"

    if len(words) == 1:
        return r"\b" + re.escape(words[0]) + r"\w*\b"

    parts = [re.escape(w) for w in words[:-1]]
    last = re.escape(words[-1]) + r"\w*\b"
    return r"\b" + r"\s+".join(parts + [last])


# =========================
# METRICS
# =========================
def strict_em(pred: str, gold: str) -> int:
    pred2 = normalize_text(strip_citations_and_meta(pred), fold_ascii=True)
    gold2 = normalize_text(strip_citations_and_meta(gold), fold_ascii=True)
    return int(pred2 == gold2)

def relaxed_em(pred: str, gold: str) -> int:
    pred2 = normalize_text(strip_citations_and_meta(pred), fold_ascii=True)
    gold2 = normalize_text(strip_citations_and_meta(gold), fold_ascii=True)
    if not pred2 and not gold2:
        return 1
    if not pred2 or not gold2:
        return 0
    return int((gold2 in pred2) or (pred2 in gold2))

def content_f1(pred: str, gold: str) -> float:
    pred2 = strip_citations_and_meta(pred)
    gold2 = strip_citations_and_meta(gold)

    p = content_tokens(pred2)
    g = content_tokens(gold2)

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    p_count: Dict[str, int] = {}
    for t in p:
        p_count[t] = p_count.get(t, 0) + 1

    g_count: Dict[str, int] = {}
    for t in g:
        g_count[t] = g_count.get(t, 0) + 1

    overlap = 0
    for t, c in p_count.items():
        overlap += min(c, g_count.get(t, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)

def token_f1_em(pred: str, gold: str, threshold: float = 0.35) -> int:
    return int(content_f1(pred, gold) >= threshold)

def exact_match(pred: str, gold: str) -> int:
    if EM_MODE == "strict":
        return strict_em(pred, gold)
    if EM_MODE == "relaxed":
        return relaxed_em(pred, gold)
    if EM_MODE == "token_f1":
        return token_f1_em(pred, gold, threshold=EM_F1_THRESHOLD)
    return strict_em(pred, gold)

def must_have_score(text: str, must_have: List[str]) -> Dict[str, Any]:
    """
    Türkçe ekleri tolere eden fuzzy regex ile arar.
    """
    t = normalize_text(strip_citations_and_meta(text), fold_ascii=True)

    mh = must_have or []
    if not mh:
        return {"must_have_ok": 1, "must_have_recall": 1.0, "missing": []}

    found: List[str] = []
    missing: List[str] = []

    for x in mh:
        x2 = normalize_text(x, fold_ascii=True)
        if not x2:
            continue
        pat = term_to_fuzzy_regex(x2)
        if re.search(pat, t):
            found.append(x)
        else:
            missing.append(x)

    recall = len(found) / len(mh) if mh else 1.0
    ok = int(len(missing) == 0)
    return {"must_have_ok": ok, "must_have_recall": float(recall), "missing": missing}

NEGATION_CUES = {
    "kacin", "kacinin", "kacinil", "kacinilmalidir", "onerilmez", "onerilmem",
    "tuketme", "tuketmeyin", "sinirla", "sinirlandir", "kisitla", "kisitlayin",
    "uzak", "dur", "sakinin", "sakininiz", "yasak", "olmamal",
    "onerilmemelidir", "tavsiye edilmez", "onermem", "onermiyoruz",
}

def must_not_violation(text: str, must_not_have: List[str]) -> Dict[str, Any]:
    """
    fuzzy regex ile bulur; negation cues varsa ihlal saymaz.
    """
    t = normalize_text(strip_citations_and_meta(text), fold_ascii=True)
    tokens = t.split()

    mn = must_not_have or []
    if not mn:
        return {"must_not_violation": 0, "hits": []}

    hits: List[Dict[str, Any]] = []
    violation = 0

    for term in mn:
        term2 = normalize_text(term, fold_ascii=True)
        if not term2:
            continue

        pat = term_to_fuzzy_regex(term2)
        for m in re.finditer(pat, t):
            before = t[:m.start()].split()
            pos = len(before)

            left = max(0, pos - 6)
            right = min(len(tokens), pos + 6)
            window = tokens[left:right]

            has_neg = any(cue in window for cue in NEGATION_CUES)
            hits.append({"term": term, "pos": pos, "has_negation": bool(has_neg)})

            if not has_neg:
                violation = 1

    return {"must_not_violation": int(violation), "hits": hits[:20]}

def compute_em_llm_only(pred: str, gold: str, mh: Dict[str, Any], mn: Dict[str, Any], f1: float) -> int:
    """
    LLM-only için daha doğru EM:
    - strict/relaxed seçiliyse aynen uygula
    - token_f1 ise: (must_not_violation==0) AND (must_have_recall==1.0 OR f1>=threshold)
    """
    if EM_MODE == "strict":
        return strict_em(pred, gold)
    if EM_MODE == "relaxed":
        return relaxed_em(pred, gold)

    if int(mn.get("must_not_violation", 0)) == 1:
        return 0
    mh_recall = float(mh.get("must_have_recall", 0.0))
    return int((mh_recall >= 1.0) or (f1 >= EM_F1_THRESHOLD))


# =========================
# PROMPT (LLM ONLY - RAG YOK)
# =========================
def load_rules() -> str:
    return read_text(RULES_PATH)

def build_prompt_llm_only(rules: str, query: str) -> str:
    return f"""Sen bir diyetisyen asistanısın. Aşağıdaki kurallara uy.

[KURALLAR]
{rules}

[SORU]
{query}

[ÇIKTI]
- Kısa ve net cevap ver (tercihen 1-2 cümle).
- Uydurma yapma.
- Kesinlikle kaynak/referans yazma: "KAYNAK", "[KAYNAK 1]" vb. HİÇBİR ŞEY ekleme.
- Selamlaşma, açıklama, madde işareti, başlık yazma; sadece cevabı yaz.
"""


# =========================
# OLLAMA CALL
# =========================
async def call_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9},
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()


# =========================
# RUNNER (LLM ONLY)
# =========================
async def run_one_question_llm_only(rules: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    qid = sample.get("id")
    query = sample.get("query", "")

    gold = sample.get("gold", {}) or {}
    gold_answer = gold.get("answer", "")

    must_have = gold.get("must_have", []) or []
    must_not = gold.get("must_not_have", []) or []

    meta = sample.get("meta", {}) or {}

    prompt = build_prompt_llm_only(rules, query)

    tasks = [call_ollama(m, prompt) for m in MODELS]
    answers = await asyncio.gather(*tasks, return_exceptions=True)

    per_model: List[Dict[str, Any]] = []
    for model, ans in zip(MODELS, answers):
        if isinstance(ans, Exception):
            out_text = f"[ERROR] {repr(ans)}"
        else:
            out_text = ans

        f1 = content_f1(out_text, gold_answer)
        mh = must_have_score(out_text, must_have)
        mn = must_not_violation(out_text, must_not)
        em = compute_em_llm_only(out_text, gold_answer, mh=mh, mn=mn, f1=f1)

        per_model.append({
            "model": model,
            "answer": out_text,
            "metrics": {
                "em": int(em),
                "f1": float(f1),

                "must_have_ok": int(mh["must_have_ok"]),
                "must_have_recall": float(mh["must_have_recall"]),

                "must_not_violation": int(mn["must_not_violation"]),
            },
            "debug": {
                "missing_must_have": mh.get("missing", []),
                "must_not_hits": mn.get("hits", []),
            }
        })

    return {
        "id": qid,
        "query": query,
        "gold": gold,
        "meta": meta,
        "results": per_model,
    }


async def main():
    if not os.path.exists(TEST_SET_PATH):
        raise FileNotFoundError(f"TEST_SET_PATH bulunamadı: {TEST_SET_PATH}")

    ensure_dir(os.path.abspath(OUT_DIR))

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    rules = load_rules()

    all_rows: List[Dict[str, Any]] = []
    per_question_rows: List[Dict[str, Any]] = []

    for sample in test_data:
        item = await run_one_question_llm_only(rules, sample)
        all_rows.append(item)

        qid = item["id"]
        for r in item["results"]:
            met = r["metrics"]
            per_question_rows.append({
                "id": qid,
                "model": r["model"],
                **met,
            })

    jsonl_path = os.path.join(OUT_DIR, "results_llm_only.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_q_df = pd.DataFrame(per_question_rows)
    per_q_path = os.path.join(OUT_DIR, "per_question_llm_only.csv")
    per_q_df.to_csv(per_q_path, index=False, encoding="utf-8-sig")

    summary = per_q_df.groupby(["model"]).agg({
        "em": "mean",
        "f1": "mean",
        "must_have_ok": "mean",
        "must_have_recall": "mean",
        "must_not_violation": "mean",
    }).reset_index()

    summary_path = os.path.join(OUT_DIR, "summary_llm_only.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 80)

    print("\n✔ Bitti! (LLM ONLY)")
    print("Detay JSONL        :", jsonl_path)
    print("Soru-model CSV     :", per_q_path)
    print("Özet CSV           :", summary_path)

    print(f"\n[INFO] EM_MODE={EM_MODE} (token_f1 threshold={EM_F1_THRESHOLD if EM_MODE=='token_f1' else 'n/a'})")
    print("\n=== MODEL SUMMARY (best -> worst by F1) ===")
    tmp = summary.sort_values("f1", ascending=False)
    print(tmp.to_string(index=False))

    print("\n=== WORST (id, model) by F1 (first 20) ===")
    worst = (
        per_q_df.sort_values("f1", ascending=True)
        .head(20)[["id", "model", "f1", "must_have_recall", "must_not_violation"]]
    )
    print(worst.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
