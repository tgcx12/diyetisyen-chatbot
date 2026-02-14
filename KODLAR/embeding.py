# -*- coding: utf-8 -*-
"""
retriever_only_eval_faiss_contentgold_v1.py

✅ Retriever-only evaluation (FAISS cosine)
✅ Supports BOTH test set formats:
   (A) Old: {question, gold_answer, expected_topics}
   (B) New: {query, gold:{answer,must_have,must_not_have}, meta:{...}}
✅ Metrics computed by checking RETRIEVED CHUNK CONTENT against gold constraints:
   - HitRate@k: any of top-k chunks satisfies gold constraints (must_have present, must_not_have absent)
   - Recall@k : average must_have coverage within best chunk in top-k
   - AnswerMatch@k (optional): token-overlap score between retrieved content and gold answer (best among top-k)

Outputs:
  - query_based_retrieval_results.txt
  - per_query_debug.txt (optional)
"""

import json
import time
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------
# PATHS (Windows)
# -------------------------
CHUNKS_JSON = r"C:\Users\user\Desktop\diyetisyen_llm\merged_all_rag_standardized.json"
TESTSET_JSON = r"C:\Users\user\Desktop\diyetisyen_llm\test_seti.json"

OUT_TXT_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\query_based_retrieval_results.txt"
DEBUG_TXT_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\per_query_debug.txt"

MODELS = {
    "berturk_nli": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "trmteb": "trmteb/turkish-embedding-model-fine-tuned",
}

K_LIST = [1, 3, 5]
WRITE_DEBUG = True

# If True: require at least one must_have token to appear to count as match (when must_have list exists)
STRICT_MUST_HAVE = True

# -------------------------
# Normalize helpers
# -------------------------
def norm_plain(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")
    s = re.sub(r"\s+", " ", s)
    return s

def safe_list(x):
    return x if isinstance(x, list) else []

def tokenize_simple(s: str):
    s = norm_plain(s)
    s = re.sub(r"[^a-z0-9\s_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split() if t]

# -------------------------
# Load chunks
# -------------------------
def chunk_text_fields(ana: str, grp: str, cont: str) -> str:
    return f"{ana} {grp} {cont}".strip()

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunk_ids = []
    chunk_texts = []
    chunk_texts_norm = []

    for c in data:
        cid = str(c.get("id", "")).strip()
        ana = str(c.get("ana_baslik", "") or "").strip()
        grp = str(c.get("topic_group", "") or "").strip()
        cont = str(c.get("content", "") or "").strip()
        full = chunk_text_fields(ana, grp, cont)

        if not cid or not full:
            continue

        chunk_ids.append(cid)
        chunk_texts.append(full)
        chunk_texts_norm.append(norm_plain(full))

    return chunk_ids, chunk_texts, chunk_texts_norm

# -------------------------
# Load test set (supports old/new formats)
# -------------------------
def load_testset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data

def get_item_query(item: dict) -> str:
    return (item.get("query") or item.get("question") or "").strip()

def get_item_gold_answer(item: dict) -> str:
    if isinstance(item.get("gold"), dict):
        ans = item["gold"].get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
    ga = item.get("gold_answer")
    if isinstance(ga, str) and ga.strip():
        return ga.strip()
    return ""

def get_item_must_have(item: dict):
    if isinstance(item.get("gold"), dict):
        return safe_list(item["gold"].get("must_have"))
    # old format doesn't have must_have
    return []

def get_item_must_not_have(item: dict):
    if isinstance(item.get("gold"), dict):
        return safe_list(item["gold"].get("must_not_have"))
    return []

# -------------------------
# FAISS cosine index
# -------------------------
def build_faiss(emb: np.ndarray):
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine if vectors are normalized
    index.add(emb)
    return index

# -------------------------
# Gold satisfaction checks
# -------------------------
def contains_any(txt_norm: str, terms):
    for t in terms:
        tn = norm_plain(str(t))
        if tn and tn in txt_norm:
            return True
    return False

def must_have_coverage(txt_norm: str, must_have):
    """returns (hits, total, hit_terms[])"""
    mh = [norm_plain(str(t)) for t in must_have if str(t).strip()]
    mh = [t for t in mh if t]
    if not mh:
        return 0, 0, []
    hits = []
    for t in mh:
        if t in txt_norm:
            hits.append(t)
    return len(hits), len(mh), hits

def violates_must_not(txt_norm: str, must_not_have):
    for t in must_not_have:
        tn = norm_plain(str(t))
        if tn and tn in txt_norm:
            return True, tn
    return False, ""

def answer_token_overlap_score(txt: str, gold_answer: str):
    """
    Very simple retrieval-only proxy:
    Jaccard overlap between token sets (no LLM).
    """
    if not gold_answer.strip():
        return 0.0
    a = set(tokenize_simple(gold_answer))
    b = set(tokenize_simple(txt))
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def chunk_satisfies_gold(txt_norm: str, must_have, must_not_have):
    """
    Satisfy if:
      - does NOT violate must_not_have
      - and must_have coverage rule:
          if must_have empty -> ok
          else:
             STRICT_MUST_HAVE=True -> at least 1 must_have present
             STRICT_MUST_HAVE=False -> ok (only penalize by recall)
    """
    bad, _badterm = violates_must_not(txt_norm, must_not_have)
    if bad:
        return False

    hits, total, _ = must_have_coverage(txt_norm, must_have)
    if total == 0:
        return True
    if STRICT_MUST_HAVE:
        return hits >= 1
    return True

# -------------------------
# Content-based evaluation
# -------------------------
def evaluate_content_based(index, chunk_ids, chunk_texts, chunk_texts_norm, query_embs, test_items, k: int):
    """
    Returns:
      hit_rate, recall_avg, ansmatch_avg, valid_count
    where:
      - hit_rate: any chunk in top-k satisfies gold constraints
      - recall_avg: average best must_have coverage among top-k
      - ansmatch_avg: average best answer overlap among top-k (if gold_answer exists)
    """
    hit = 0
    recall_sum = 0.0
    ansmatch_sum = 0.0
    valid = 0

    debug_rows = []

    for qi in range(query_embs.shape[0]):
        item = test_items[qi]
        q = get_item_query(item)

        must_have = get_item_must_have(item)
        must_not = get_item_must_not_have(item)
        gold_answer = get_item_gold_answer(item)

        # If there is literally no gold signal, skip
        if not must_have and not must_not and not gold_answer:
            continue

        valid += 1

        _, idx = index.search(query_embs[qi:qi+1], k)
        top_pos = idx[0].tolist()

        best_satisfy = False
        best_cov = 0.0
        best_ans = 0.0
        best_reason = ""
        best_chunk_id = ""

        # Evaluate each retrieved chunk
        for rank, pos in enumerate(top_pos, start=1):
            if pos < 0 or pos >= len(chunk_ids):
                continue

            cid = chunk_ids[pos]
            txt = chunk_texts[pos]
            txtn = chunk_texts_norm[pos]

            # must_have coverage
            hits, total, hit_terms = must_have_coverage(txtn, must_have)
            cov = (hits / total) if total > 0 else 1.0  # if no must_have, coverage=1 by definition

            # must_not violation
            bad, badterm = violates_must_not(txtn, must_not)

            # satisfy?
            sat = (not bad) and ((total == 0) or (hits >= 1 if STRICT_MUST_HAVE else True))

            # answer overlap proxy
            ans_sc = answer_token_overlap_score(txt, gold_answer) if gold_answer else 0.0

            # pick best chunk by: satisfy > coverage > answer overlap
            # (you can adjust priority, but this is reasonable for retriever-only)
            key = (1 if sat else 0, cov, ans_sc)

            current_best_key = (1 if best_satisfy else 0, best_cov, best_ans)
            if key > current_best_key:
                best_satisfy = sat
                best_cov = cov
                best_ans = ans_sc
                best_chunk_id = cid
                if bad:
                    best_reason = f"violates_must_not='{badterm}'"
                else:
                    best_reason = f"hits={hits}/{total} hit_terms={hit_terms}"

        if best_satisfy:
            hit += 1

        recall_sum += best_cov
        ansmatch_sum += best_ans

        if WRITE_DEBUG:
            debug_rows.append({
                "id": item.get("id", f"q{qi:03d}"),
                "query": q,
                "k": k,
                "best_chunk_id": best_chunk_id,
                "hit": int(best_satisfy),
                "best_must_have_coverage": round(best_cov, 4),
                "best_answer_overlap": round(best_ans, 4),
                "reason": best_reason,
                "must_have": must_have,
                "must_not_have": must_not,
                "gold_answer": gold_answer[:180]
            })

    if valid <= 0:
        return 0.0, 0.0, 0.0, 0, debug_rows

    return hit / valid, recall_sum / valid, ansmatch_sum / valid, valid, debug_rows

# -------------------------
# MAIN
# -------------------------
def main():
    chunk_ids, chunk_texts, chunk_texts_norm = load_chunks(CHUNKS_JSON)
    testset = load_testset(TESTSET_JSON)

    questions = [get_item_query(it) for it in testset]

    lines = []
    lines.append("===== CONTENT-GOLD RETRIEVER-ONLY EVALUATION =====\n")
    lines.append(f"Chunks: {CHUNKS_JSON}\n")
    lines.append(f"Testset: {TESTSET_JSON}\n")
    lines.append(f"Question count: {len(questions)}\n")
    lines.append(f"K values: {K_LIST}\n")
    lines.append(f"STRICT_MUST_HAVE: {STRICT_MUST_HAVE}\n")
    lines.append("Gold checking: must_have/must_not_have + optional gold.answer token overlap\n\n")

    all_debug = []

    for _key, model_name in MODELS.items():
        lines.append(f"--- Model: {model_name} ---\n")

        model = SentenceTransformer(model_name)

        # chunk embeddings
        t0 = time.time()
        chunk_embs = model.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        chunk_time = time.time() - t0

        emb = np.asarray(chunk_embs, dtype="float32")
        index = build_faiss(emb)

        # query embeddings
        t1 = time.time()
        q_embs = model.encode(
            questions,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        q_time = time.time() - t1
        q_embs = np.asarray(q_embs, dtype="float32")

        lines.append(f"Embedding dim: {emb.shape[1]}\n")
        lines.append(f"Chunk embedding time (s): {chunk_time:.2f}\n")
        lines.append(f"Query embedding time (s): {q_time:.2f}\n")

        for k in K_LIST:
            hr, rec, ansm, valid, dbg = evaluate_content_based(
                index=index,
                chunk_ids=chunk_ids,
                chunk_texts=chunk_texts,
                chunk_texts_norm=chunk_texts_norm,
                query_embs=q_embs,
                test_items=testset,
                k=k
            )
            lines.append(f"Valid questions (with gold signal): {valid}/{len(questions)}\n")
            lines.append(f"HitRate@{k}: {hr:.4f}\n")
            lines.append(f"MustHaveRecall@{k}: {rec:.4f}\n")
            lines.append(f"AnswerOverlap@{k}: {ansm:.4f}\n")
            lines.append("-" * 40 + "\n")

            if WRITE_DEBUG:
                # keep only first K list debug, or store all with model+k tags
                for row in dbg:
                    row["model"] = model_name
                    row["metric_k"] = k
                all_debug.extend(dbg)

        lines.append("-" * 60 + "\n\n")

    out_text = "".join(lines)
    print(out_text)

    with open(OUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"📄 Sonuçlar TXT yazıldı: {OUT_TXT_PATH}")

    if WRITE_DEBUG:
        with open(DEBUG_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(all_debug, ensure_ascii=False, indent=2))
        print(f"🧪 Per-query debug yazıldı: {DEBUG_TXT_PATH}")

if __name__ == "__main__":
    main()
