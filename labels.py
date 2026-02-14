import os
import json
import re
from typing import Any, Dict, List, Tuple, Optional

# =========================
# HYBRID NLP (Stanza on-demand)
# =========================
try:
    import stanza
except Exception:
    stanza = None

_STANZA_NLP = None

HEAVY_CLAUSE_RE = re.compile(
    r"\b(yerine|ancak|fakat|ama|yalnız|yalniz|koşuluyla|kosuluyla|şartıyla|sartiyla|"
    r"eğer|eger|ise|aksi halde|aksi takdirde|bununla birlikte|buna rağmen|"
    r"mümkünse|mumkunse|dikkat edil(irse|meli)|"
    r"günlerde|gunlerde)\b",
    re.IGNORECASE
)

def get_stanza_tr():
    """Lazy init: sadece heavy clause gelirse yüklenir."""
    global _STANZA_NLP
    if _STANZA_NLP is not None:
        return _STANZA_NLP
    if stanza is None:
        return None
    _STANZA_NLP = stanza.Pipeline(
        "tr",
        processors="tokenize,pos,lemma,depparse",
        use_gpu=False,
        verbose=False
    )
    return _STANZA_NLP

# =========================
# TEXT NORMALIZATION
# =========================
def turkish_lower(s: str) -> str:
    s = (s or "")
    s = s.replace("İ", "i").replace("I", "ı")
    return s.casefold()

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", turkish_lower(s)).strip()

def normalize_quote_key(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[’'\"“”]", "", s)
    s = re.sub(r"[^\wçğıöşü0-9% ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# PATHS
# =========================
BASE_DIR = r"C:\Users\user\Desktop\diyetisyen_llm\deneme_cache\qa_demo"
TAG_DICTS_PATH = r"C:\Users\user\Desktop\diyetisyen_llm\tag_dicts.json"
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # ...\diyetisyen_llm
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "labels")

# =========================
# UNITS
# =========================
ALLOWED_UNITS = {
    "kez", "porsiyon", "adet", "gram", "mg", "yuzde",
    "cay_kasigi", "tatli_kasigi", "yemek_kasigi",
    "dilim", "bardak"
}

# =========================
# OPTIONAL MANUAL KEYWORD MAP (accuracy booster)
# Not tag list, only alias enrichment. Keep or remove freely.
# (If a tag is not in tag_dicts.json, it is ignored automatically.)
# =========================
TAG_KEYWORD_MAP: Dict[str, List[str]] = {
    "balik": ["balık", "balik", "somon", "uskumru", "hamsi", "lüfer", "lufer", "palamut", "sardalya", "alabalık", "alabalik", "ton balığı", "ton baligi", "tuna", "levrek", "çipura", "cipura"],
    "beyaz_et": ["tavuk", "hindi", "fileto", "göğüs", "gogus"],
    "kirmizi_et": ["kırmızı et", "kirmizi et", "dana", "kuzu", "biftek", "köfte", "kofte"],
    "islenmis_et": ["sucuk", "salam", "sosis", "jambon", "pastırma", "pastirma", "şarküteri", "sarkuteri", "füme", "fume"],
    "sakatat": ["sakatat", "ciğer", "ciger", "yürek", "dalak", "böbrek", "bobrek", "işkembe", "iskembe"],

    "sut_urunu": ["süt", "sut", "yoğurt", "yogurt", "peynir", "kefir", "ayran", "lor", "labne"],
    "az_yagli_sut_urunleri": ["az yağlı", "az yagli", "yarım yağlı", "yarim yagli", "light süt", "light sut", "düşük yağlı süt", "dusuk yagli sut"],
    "yagsiz_sut_urunleri": ["yağsız", "yagsiz", "0 yağ", "0 yag", "yağsız süt", "yagsiz sut", "yağsız yoğurt", "yagsiz yogurt"],

    "sebze": ["sebze", "yeşillik", "yesillik", "ıspanak", "ispanak", "pırasa", "pirasa", "brokoli", "karnabahar", "lahana", "marul", "roka", "maydanoz", "dereotu", "kabak", "patlıcan", "patlican", "biber", "domates", "salatalık", "salatalik", "havuç", "havuc", "turp", "pancar"],
    "meyve": ["meyve", "elma", "armut", "muz", "portakal", "mandalina", "çilek", "cilek", "kiraz", "üzüm", "uzum", "kivi", "nar", "şeftali", "seftali", "kayısı", "kayisi", "incir", "karpuz", "kavun", "meyve suyu", "juice"],
    "salata": ["salata", "mevsim salata", "çoban salata", "coban salata"],

    "tahil": ["tahıl", "tahil", "yulaf", "bulgur", "arpa", "çavdar", "cavdar", "tam tahıl", "tam tahil", "kepek"],
    "ekmek": ["ekmek", "beyaz ekmek", "tam buğday", "tam bugday", "çavdar ekmeği", "cavdar ekmegi", "kepekli ekmek"],
    "pilav_makarna": ["pilav", "pirinç", "pirinc", "makarna", "şehriye", "sehriye", "noodle"],
    "kompleks_kh": ["tam tahıl", "tam tahil", "tam buğday", "tam bugday", "bulgur", "kepekli", "yulaf", "çavdar", "cavdar"],

    "zeytinyagi": ["zeytinyağı", "zeytinyagi", "zeytin yağı", "zeytin yagi"],
    "tekli_doymamis": ["tekli doymamış", "tekli doymamis", "oleik", "zeytin", "avokado", "fındık yağı", "findik yagi"],
    "coklu_doymamis": ["çoklu doymamış", "coklu doymamis", "omega 6", "omega6", "ayçiçek", "aycicek", "mısır özü", "misir ozu"],
    "omega3": ["omega 3", "omega3", "epa", "dha"],

    "kuruyemis": ["kuruyemiş", "kuruyemis", "badem", "ceviz", "fındık", "findik", "yer fıstığı", "yer fistigi", "antep fıstığı", "antep fistigi"],
    "yagli_tohum": ["yağlı tohum", "yagli tohum", "chia", "keten", "keten tohumu", "susam", "tahin", "kabak çekirdeği", "kabak cekirdegi"],

    "sodyum_yuksek": ["tuz", "tuzlu", "salamura", "turşu", "tursu", "sodyum", "cips", "hazır çorba", "hazir corba", "bulyon", "bouillon"],
    "eklenmis_seker": ["eklenmiş şeker", "eklenmis seker", "şeker", "seker", "şerbet", "serbet", "tatlı", "tatli", "gazoz", "kola"],
    "sekerli_icecek": ["şekerli içecek", "sekerli icecek", "şekerle tatlandırılmış", "sekerle tatlandirilmis", "meyve suyu", "enerji içeceği", "enerji icecegi"],
    "su": ["su", "maden suyu"],

    "akrilamid": ["kızartma", "kizartma", "cips", "yanmış", "yanmis", "çok kızarmış", "cok kizarmis"],
    "trans_yag_riski": ["margarin", "paketli", "hazır", "hazir", "fast food", "trans yağ", "trans yag"],
}

# =========================
# "X yerine Y" extraction
# =========================
REPLACEMENT_PAT = re.compile(
    r"(?P<x>[^,.()]{2,60}?)\s+(yerine|yerine\s+de)\s+(?P<y>[^,.()]{2,60}?)\b",
    re.IGNORECASE
)

def stanza_extract_yerine_pairs(text: str) -> List[Tuple[str, str]]:
    """Stanza dependency ile 'X yerine Y' çiftleri (bulamazsa [])."""
    nlp = get_stanza_tr()
    if nlp is None:
        return []

    doc = nlp(text)
    pairs: List[Tuple[str, str]] = []

    for sent in doc.sentences:
        words = sent.words  # 1-based id
        for w in words:
            if (w.text or "").lower() != "yerine":
                continue
            idx = w.id
            left_tokens = [ww.text for ww in words[max(1, idx - 4) - 1: idx - 1] if ww.text]
            right_tokens = [ww.text for ww in words[idx: min(len(words), idx + 4)] if ww.text]
            x = " ".join(left_tokens).strip()
            y = " ".join(right_tokens).strip()
            if len(x) < 2 or len(y) < 2:
                continue
            if len(x) > 80:
                x = " ".join(x.split()[-4:])
            if len(y) > 80:
                y = " ".join(y.split()[:4])
            pairs.append((x, y))

    seen = set()
    out = []
    for x, y in pairs:
        k = (normalize_text(x), normalize_text(y))
        if k in seen:
            continue
        seen.add(k)
        out.append((x, y))
    return out

def extract_replacement_pairs(clause: str) -> List[Tuple[str, str]]:
    t = (clause or "").strip()
    pairs = []
    for m in REPLACEMENT_PAT.finditer(t):
        x = (m.group("x") or "").strip()
        y = (m.group("y") or "").strip()
        if len(x) >= 2 and len(y) >= 2:
            pairs.append((x, y))
    return pairs

def clean_phrase_for_tag_match(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(
        r"\b(tercih|edilmeli|önerilir|onerilir|tüketilmeli|tuketilmeli|kullanılmalı|kullanilmali|"
        r"yenmeli|yenilmeli|eklenmeli|azaltılmalı|azaltilmali|kısıtlanmalı|kisitlanmali)\b",
        " ",
        s
    )
    return re.sub(r"\s+", " ", s).strip()

# =========================
# MEAL CONTEXT
# =========================
MEAL_SIGNALS: Dict[str, List[str]] = {
    "sabah": ["kahvaltı", "kahvalti", "sabah", "uyanış", "uyanis", "ilk öğün", "ilk ogun"],
    "ara": ["ara öğün", "ara ogun", "kuşluk", "kusluk", "ikindi", "gece öğünü", "gece ogunu", "atıştırmalık", "atistirmalik"],
    "ogle": ["öğle", "ogle", "öğlen", "oglen"],
    "aksam": ["akşam", "aksam", "ana yemek", "gece yemeği", "gece yemegi"],
}

def extract_meal_context(text: str) -> List[str]:
    t = normalize_text(text)
    found = []
    for meal, keywords in MEAL_SIGNALS.items():
        for k in keywords:
            if normalize_text(k) in t:
                found.append(meal)
                break
    return found if found else ["genel"]

# =========================
# STEM-LITE (Turkish suffix stripping)
# =========================
COMMON_SUFFIXES = [
    "larımız", "lerimiz", "larınız", "leriniz", "larından", "lerinden",
    "ların", "lerin", "lara", "lere", "ları", "leri", "larınca", "lerince",
    "dan", "den", "tan", "ten",
    "dır", "dir", "dur", "dür", "tir", "tır", "tur", "tür",
    "lı", "li", "lu", "lü",
    "sız", "siz", "suz", "süz",
    "cı", "ci", "cu", "cü",
    "ca", "ce",
    "im", "ım", "um", "üm",
    "in", "ın", "un", "ün",
    "miz", "mız", "muz", "müz",
    "niz", "nız", "nuz", "nüz",
    "si", "sı", "su", "sü",
    "m", "n",
]

def turkish_stem(word: str) -> str:
    w = normalize_text(word)
    w = re.sub(r"[^a-zçğıöşü0-9%]", "", w)
    if len(w) <= 3:
        return w
    for suf in sorted(COMMON_SUFFIXES, key=len, reverse=True):
        if len(w) - len(suf) >= 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w

def tokenize_stems(text: str) -> List[str]:
    t = normalize_text(text)
    toks = re.findall(r"[a-zçğıöşü0-9%]+", t)
    return [turkish_stem(x) for x in toks if x]

# =========================
# AUTO ALIAS INDEX (tag_dicts -> alias)
# =========================
STOPWORDS = {
    "ve", "ile", "vb", "gibi", "genelde", "genel", "olan", "olarak", "icin", "için", "daha",
    "az", "cok", "çok", "orta", "yuksek", "yüksek", "dusuk", "düşük", "uygun", "etiketi",
    "uyarisi", "uyarısı", "destek", "hedef", "kontrol", "grubu", "kategori", "isareti",
    "işareti", "bazi", "bazı", "kisilerde", "kişilerde", "oranı", "oran", "seviyesi",
    "tercih", "edilmeli", "onerilir", "önerilir", "kacinilmali", "kaçınılmalı",
}

# Bu alias'lar çok genel ve sık çakışıyor. (özellikle et/eti yüzünden kirmizi_et -> beyaz_et kaymaları)
GENERIC_SINGLE_ALIASES = {
    "et", "eti", "etin", "etler", "etleri", "yemek", "yemegi", "yemeği", "tuket", "tuketim", "tüketim",
    "grup", "grubu", "gruplari", "oran", "orani", "seviy", "seviyesi"
}

# Regex listesinde mutlaka yer alsın dediğimiz kritik alias'lar (2000 limit vs. yüzünden düşmesin)
PRIORITY_ALIASES = [
    "balık", "balik", "yağ", "yag", "doymuş", "doymus", "trans", "sodyum", "tuz",
    "kırmızı", "kirmizi", "tavuk", "hindi", "yumurta", "zeytin", "lif", "posa",
    "glisemik", "indeks", "karbonhidrat", "kh"
]

def build_alias_index_from_tag_dicts(tag_desc: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Tamamen tag_dicts.json içindeki tag adlarını baz alır.
    Enrichment sadece var olan tag'lar için yapılır.
    """
    out: Dict[str, List[str]] = {}
    for tag, desc in tag_desc.items():
        d = normalize_text(desc or "")
        parens = re.findall(r"\(([^)]{2,200})\)", d)
        toks: List[str] = []

        for p in parens:
            for w in re.split(r"[,/;]", p):
                w = w.strip()
                if not w:
                    continue
                w = re.sub(r"\b(vb|vs)\b\.?", "", w).strip()
                if len(w) >= 3:
                    toks.append(w)

        for w in re.split(r"[^a-zA-Zçğıöşü0-9%]+", d):
            w = w.strip()
            if len(w) < 4:
                continue
            if w in STOPWORDS:
                continue
            if re.fullmatch(r"\d+", w):
                continue
            toks.append(w)

        uniq_list = []
        seen = set()
        for t in toks:
            t = t.strip()
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            uniq_list.append(t)

        augmented = []
        for a in uniq_list:
            aa = normalize_text(a)
            if " " not in aa and len(aa) >= 4:
                augmented.append(aa)
                augmented.append(turkish_stem(aa))
            else:
                augmented.append(aa)

        out[tag] = list(dict.fromkeys([x for x in augmented if x and len(x) >= 3]))[:60]

    # manual keywords -> alias enrichment (only if tag exists in tag_dicts)
    for tag, kws in TAG_KEYWORD_MAP.items():
        if tag not in out:
            continue
        out.setdefault(tag, [])
        for k in kws:
            kk = normalize_text(k)
            if not kk:
                continue
            if " " not in kk:
                out[tag].append(kk)
                out[tag].append(turkish_stem(kk))
            else:
                out[tag].append(kk)
        out[tag] = list(dict.fromkeys(out[tag]))[:140]

    return out

def build_alias_regex_index(alias_index: Dict[str, List[str]]) -> Tuple[re.Pattern, Dict[str, List[str]]]:
    """
    Sağlamlaştırma:
    - alias_to_tags full tutulur.
    - regex'e girecek single-token alias listesi:
        * ambiguous + generic ise (birden fazla tag'a gidiyorsa) EKLENMEZ.
        * PRIORITY_ALIASES her koşulda eklenir.
    """
    alias_to_tags: Dict[str, List[str]] = {}
    for tag, als in alias_index.items():
        for a in als:
            aa = normalize_text(a).strip()
            if len(aa) < 3:
                continue
            if " " in aa:
                continue
            alias_to_tags.setdefault(aa, []).append(tag)

    # Build regex alias list
    regex_aliases: List[str] = []

    # Always include priority aliases if they exist in alias_to_tags (or even if not; safe)
    for pa in PRIORITY_ALIASES:
        paa = normalize_text(pa)
        if len(paa) >= 3:
            regex_aliases.append(re.escape(paa))

    for aa, tags in alias_to_tags.items():
        if len(aa) < 3:
            continue
        # generic ve ambiguous (multi-tag) ise regex'e alma
        if aa in GENERIC_SINGLE_ALIASES and len(tags) > 1:
            continue
        regex_aliases.append(re.escape(aa))

    regex_aliases = sorted(list(set(regex_aliases)), key=len, reverse=True)[:2500]
    if not regex_aliases:
        regex_aliases = [re.escape("sebze"), re.escape("meyve"), re.escape("tuz")]

    pattern = r"(?<!\w)(" + "|".join(regex_aliases) + r")(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE), alias_to_tags

# =========================
# TAG CLASSIFICATION (AUTO from tag_dicts.json)
# =========================
def infer_tag_class(tag: str, desc: str) -> str:
    """
    tag_dicts.json açıklamasından class üretir:
    - risk: kaçın/limit/risk/tetikleyebilir/uyarı/trans/işlenmiş vb.
    - positive: uygun/destekleyici/önerilir/düşük sodyum vb.
    - neutral: kalanlar
    """
    t = normalize_text(tag)
    d = normalize_text(desc or "")

    risk_cues = [
        "kaçın", "kacin", "kaçınıl", "kacinil", "yasak", "uzak dur", "sakın", "sakin",
        "risk", "tetikleyebilir", "uyarı", "uyari", "dikkat", "sınırlan", "sinirlan",
        "trans", "işlenmiş", "islenmis", "yük", "yuk", "ödem", "odem"
    ]
    pos_cues = [
        "uygun", "destek", "destekleyici", "önerilir", "onerilir", "daha uygun",
        "düşük", "dusuk", "hafif", "kontrol", "kardiyo", "kardiyometabolik", "tokluk", "bağırsak", "bagirsak"
    ]

    if "_riski" in t or "risk" in t:
        return "risk"
    if t.endswith("_yuksek") and ("kaçın" in d or "kacin" in d or "sınırl" in d or "sinirl" in d):
        return "risk"
    if t.endswith("_dusuk") and ("uygun" in d or "destek" in d or "daha" in d):
        return "positive"

    if any(c in d for c in risk_cues):
        return "risk"
    if any(c in d for c in pos_cues):
        return "positive"
    return "neutral"

def build_tag_class(tag_desc: Dict[str, str]) -> Dict[str, str]:
    return {tag: infer_tag_class(tag, tag_desc.get(tag, "")) for tag in tag_desc.keys()}

# =========================
# SCHEMA
# =========================
def ensure_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("prefer_tags", [])
    obj.setdefault("limit_tags", [])
    obj.setdefault("avoid_tags", [])
    obj.setdefault("meal_pattern_rules", {})
    obj["meal_pattern_rules"].setdefault("logical_rules", {"prefer": [], "limit": [], "avoid": []})
    obj["meal_pattern_rules"].setdefault("numeric_constraints", [])
    obj["meal_pattern_rules"].setdefault("engine_rules", [])
    obj.setdefault("energy_rules", {"scale_up_order": [], "scale_down_order": [], "locks": []})
    obj.setdefault("recommendations", [])
    obj.setdefault("rag_evidence", [])
    return obj

def uniq(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

# =========================
# INTENT DETECTION
# =========================
PREFER_PAT = re.compile(
    r"\b(önerilir|onerilir|tercih edil|tercih edilmeli|tüketilmeli|tuketilmeli|artır(ıl)?malı|artir(ıl)?mali|eklenmeli|"
    r"seçilmeli|secilmeli|uygun(dur)?|öner(ilir)?|yer verilmeli|kullanılmalı|kullanilmali)\b",
    re.IGNORECASE
)
LIMIT_PAT = re.compile(
    r"\b(sınırla(n)?malı|sinirla(n)?mali|sınırlandır(ıl)?malı|sinirlandir(ıl)?mali|azalt(ıl)?malı|"
    r"kısıtla(n)?malı|kisitla(n)?mali|kontrol altında|ölçülü|olculu|en fazla|aşma(malı)?|asmamali|"
    r"altında tut(ul)?malı|altinda tut(ul)?mali)\b",
    re.IGNORECASE
)
AVOID_PAT = re.compile(
    r"\b(kaçın|kacin|kaçınılmalı|kacinilmali|tüketilmemeli|tuketilmemeli|yasak|uzak dur|sakın|sakin|"
    r"tercih edilmemeli|tercih edilmez|önerilmez|onerilmez)\b",
    re.IGNORECASE
)
NEGATION_PAT = re.compile(r"\b(değil|degil|olmamalı|olmamali)\b", re.IGNORECASE)

def detect_intent(text: str) -> str:
    t = normalize_text(text)
    if AVOID_PAT.search(t):
        return "avoid"
    if LIMIT_PAT.search(t):
        return "limit"
    if PREFER_PAT.search(t):
        if NEGATION_PAT.search(t):
            return "avoid"
        return "prefer"
    return "unknown"

TOKEN_RE = re.compile(r"[a-zçğıöşü0-9%]+", re.IGNORECASE)

def token_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

def intent_in_window(text: str, start: int, end: int, window_tokens: int = 8) -> str:
    toks = token_spans(text)
    if not toks:
        return detect_intent(text)

    idxs = [i for i, (_, s, e) in enumerate(toks) if not (e < start or s > end)]
    if not idxs:
        return detect_intent(text)

    lo = max(0, min(idxs) - window_tokens)
    hi = min(len(toks), max(idxs) + window_tokens + 1)
    snippet = " ".join([toks[i][0] for i in range(lo, hi)])
    return detect_intent(snippet)

# =========================
# CLAUSE SPLIT
# =========================
def preprocess_answer(answer: str, make_bullets: bool = True) -> str:
    if not isinstance(answer, str):
        return ""
    s = answer.strip()
    if make_bullets:
        s = s.replace("|", "\n- ")
        if "\n-" not in s:
            s = "- " + s
        else:
            if not s.lstrip().startswith("-"):
                s = "- " + s
    return s

def split_into_clauses(answer: str) -> List[str]:
    a = (answer or "").strip()
    if not a:
        return []

    raw_lines = []
    for line in a.splitlines():
        line = line.strip(" \t\r\n-•*")
        if line:
            raw_lines.append(line)

    parts: List[str] = []
    for line in raw_lines:
        if len(line) > 160:
            sents = re.split(r"(?<=[\.\!\?\;:])\s+", line)
            for s in sents:
                s = s.strip()
                if len(s.split()) >= 4:
                    parts.append(s)
        else:
            if len(line.split()) >= 4:
                parts.append(line)
    return parts

# =========================
# TAG MATCHING + spans
# =========================
def _substring_span(hay: str, needle: str) -> Optional[Tuple[int, int]]:
    idx = hay.find(needle)
    if idx == -1:
        return None
    start = idx
    end = idx + len(needle)
    if start > 0 and re.match(r"\w", hay[start - 1]):
        return None
    if end < len(hay) and re.match(r"\w", hay[end:end + 1]):
        return None
    return (start, end)

def match_tags_with_spans(
    text: str,
    vocab_set: set,
    alias_index: Dict[str, List[str]],
    alias_re: re.Pattern,
    alias_to_tags: Dict[str, List[str]],
) -> List[Tuple[str, int, int]]:
    """
    Sağlamlaştırma:
    - Multi-word alias'lar önce bulunur (daha spesifik).
    - Single-token alias regex sonra uygulanır.
    - Hepsi toplanır; numeric tarafında artık segment içi filtre yapılacak.
    """
    t = normalize_text(text)
    found: Dict[Tuple[str, int, int], bool] = {}

    # multi-word aliases first
    for tag, als in alias_index.items():
        if tag not in vocab_set:
            continue
        for a in als:
            aa = normalize_text(a)
            if " " not in aa:
                continue
            sp = _substring_span(t, aa)
            if sp:
                found[(tag, sp[0], sp[1])] = True

    # single-token aliases
    for m in alias_re.finditer(t):
        alias = normalize_text(m.group(1))
        a2 = turkish_stem(alias)
        for a in (alias, a2):
            for tag in alias_to_tags.get(a, []):
                if tag in vocab_set:
                    found[(tag, m.start(1), m.end(1))] = True

    return [(tag, s, e) for (tag, s, e) in found.keys()]

# =========================
# NUMERIC PARSING (ROBUST)
# - Numeric tag binding is SEGMENT-LOCAL ONLY.
# - Macro segments prefer macro-hint binding over proximity.
# - If still no tag -> goes to recommendations, NOT numeric_constraints.
# =========================
NUMERIC_SIGNAL_RE = re.compile(
    r"(\bhaftada\b|\bgünde\b|\bayda\b|\bkez\b|%|\bmg\b|\bgram\b|\bgr\b|\bg\b|\badet\b|\bporsiyon\b|\bbardak\b|\b(dilim)\b|"
    r"\ben az\b|\ben fazla\b|\baltında\b|\baltinda\b|\büstünde\b|\bustunde\b|\başmamalı\b|\basmamali\b|"
    r"\bveya daha fazla\b|\bmaksimum\b|\bminimum\b)",
    flags=re.IGNORECASE
)

KCAL_RE = re.compile(r"\b(kcal|kkal|kalori|kilokalori)\b", re.IGNORECASE)

NUM_RE = r"\d{1,6}(?:[.,]\d{1,3})?"
RANGE_DASH_RE = re.compile(rf"(?P<a>{NUM_RE})\s*[-–]\s*(?P<b>{NUM_RE})")
SINGLE_NUM_RE = re.compile(rf"(?P<num>{NUM_RE})")

# handles "2 3 kez", "%20 35" (space-range)
RANGE_SPACE_RE = re.compile(rf"(?P<a>{NUM_RE})\s+(?P<b>{NUM_RE})")
PERCENT_PREFIX_RANGE_RE = re.compile(rf"%\s*(?P<a>{NUM_RE})\s*[-–]?\s*(?P<b>{NUM_RE})", re.IGNORECASE)
PERCENT_PREFIX_SINGLE_RE = re.compile(rf"%\s*(?P<num>{NUM_RE})", re.IGNORECASE)

def parse_num(x: str) -> float:
    return float(x.replace(",", "."))

def _nearest_match(patterns: List[Tuple[str, str]], text: str, anchor_idx: int) -> str:
    best_key = "unknown"
    best_dist = None
    for key, pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            mid = (m.start() + m.end()) // 2
            dist = abs(mid - anchor_idx)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_key = key
    return best_key

def detect_unit_nearest(win: str, anchor_idx: int) -> str:
    w = normalize_text(win)
    patterns = [
        ("yuzde", r"(%|yüzde|yuzde)"),
        ("mg", r"\bmg\b"),
        ("gram", r"\b(gram|gr|g)\b"),
        ("adet", r"\badet\b"),
        ("porsiyon", r"\bporsiyon\b"),
        ("bardak", r"\bbardak\b"),
        ("dilim", r"\bdilim\b"),
        ("kez", r"\bkez\b"),
        ("cay_kasigi", r"(çay kaşığı|cay kasigi)"),
        ("tatli_kasigi", r"(tatlı kaşığı|tatli kasigi)"),
        ("yemek_kasigi", r"(yemek kaşığı|yemek kasigi)"),
    ]
    return _nearest_match(patterns, w, anchor_idx)

def guess_period_days_nearest(win: str, anchor_idx: int) -> Optional[int]:
    w = normalize_text(win)
    key = _nearest_match(
        [
            ("day", r"\b(günde|gunluk|günlük)\b"),
            ("week", r"\bhaftada\b"),
            ("month", r"\bayda\b"),
        ],
        w,
        anchor_idx
    )
    if key == "day":
        return 1
    if key == "week":
        return 7
    if key == "month":
        return 30
    return None

def _wants_min_max(t: str) -> Tuple[bool, bool]:
    tt = normalize_text(t)
    wants_min = bool(re.search(r"\b(en az|minimum|veya daha fazla|üstünde|ustunde|daha fazla|üzeri|uzeri)\b", tt))
    wants_max = bool(re.search(r"\b(en fazla|maksimum|altında|altinda|aşmamalı|asmamali|geçmemeli|gecmemeli|daha az|altı|alti)\b", tt))
    return wants_min, wants_max

def choose_numeric_tag_by_proximity_abs(
    anchor_abs: int,
    matches: List[Tuple[str, int, int]],
) -> Optional[str]:
    if not matches:
        return None

    best = None
    best_dist = None
    for tag, s, e in matches:
        mid = (s + e) // 2
        dist = abs(mid - anchor_abs)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = tag
    return best

def filter_matches_in_span(matches: List[Tuple[str, int, int]], span_start: int, span_end: int) -> List[Tuple[str, int, int]]:
    out = []
    for tag, s, e in matches:
        if s >= span_start and e <= span_end:
            out.append((tag, s, e))
    return out

def is_macro_numeric_segment(seg_norm: str) -> bool:
    """
    Segment içinde makro besin sinyali varsa proximity yerine macro hint kullan.
    """
    t = normalize_text(seg_norm)
    return bool(re.search(
        r"\b(sodyum|tuz|trans|doymuş|doymus|yağ|yag|lif|posa|glisemik|indeks|karbonhidrat|kh)\b",
        t
    ))

def macro_tag_hint_for_segment(seg_norm: str, vocab_set: set) -> Optional[str]:
    t = normalize_text(seg_norm)

    # sodyum/tuz
    if ("sodyum" in t) or ("tuz" in t) or ("salamura" in t):
        if "sodyum_yuksek" in vocab_set and ("yüksek" in t or "yuksek" in t or "kaçın" in t or "kacin" in t):
            return "sodyum_yuksek"
        if "sodyum_dusuk" in vocab_set and ("düşük" in t or "dusuk" in t):
            return "sodyum_dusuk"
        if "sodyum_orta" in vocab_set:
            return "sodyum_orta"

    # trans
    if "trans" in t and "trans_yag_riski" in vocab_set:
        return "trans_yag_riski"

    # doymuş yağ
    if ("doymuş" in t or "doymus" in t):
        if "doymus_yag_kisitlama" in vocab_set:
            return "doymus_yag_kisitlama"
        if "doymus_yag_yuksek" in vocab_set:
            return "doymus_yag_yuksek"
        if "doymus_yag_orta" in vocab_set:
            return "doymus_yag_orta"

    # yağ genel
    if ("yağ" in t or "yag" in t):
        if "yag" in vocab_set:
            return "yag"

    # GI
    if "glisemik" in t or "indeks" in t or re.search(r"\bgi\b", t):
        if "gi_yuksek" in vocab_set and ("yüksek" in t or "yuksek" in t):
            return "gi_yuksek"
        if "gi_dusuk" in vocab_set and ("düşük" in t or "dusuk" in t):
            return "gi_dusuk"
        if "gi_orta" in vocab_set:
            return "gi_orta"

    # KH
    if "karbonhidrat" in t or re.search(r"\bkh\b", t):
        if "kh_yuksek" in vocab_set and ("yüksek" in t or "yuksek" in t):
            return "kh_yuksek"
        if "kh_dusuk" in vocab_set and ("düşük" in t or "dusuk" in t):
            return "kh_dusuk"
        if "kh_orta" in vocab_set:
            return "kh_orta"

    # lif
    if "lif" in t or "posa" in t:
        if "lif_yuksek" in vocab_set and ("yüksek" in t or "yuksek" in t):
            return "lif_yuksek"
        if "lif_dusuk" in vocab_set and ("düşük" in t or "dusuk" in t):
            return "lif_dusuk"
        if "lif_orta" in vocab_set:
            return "lif_orta"
        if "lifli_beslenme" in vocab_set:
            return "lifli_beslenme"

    return None

def fuzzy_tag_hint_from_window(
    window_text: str,
    alias_index: Dict[str, List[str]],
    vocab_set: set,
    min_score: float = 0.84
) -> Optional[str]:
    """
    Segment içinden (window_text) tag tahmini:
    - multiword alias substring: güçlü sinyal
    - single alias stem token overlap: orta sinyal
    Basit skor: (hit_count / possible_hits) değil; pratik:
      multiword hit +2, single hit +1
    En yüksek skor seçilir; skor min_score eşiğiyle normalize edilir.
    """
    w = normalize_text(window_text)
    if not w:
        return None

    stems = set(tokenize_stems(w))
    if not stems:
        stems = set()

    best_tag = None
    best_score = 0.0

    for tag, als in alias_index.items():
        if tag not in vocab_set:
            continue
        score = 0.0
        for a in als:
            aa = normalize_text(a)
            if not aa:
                continue
            if " " in aa:
                if aa in w:
                    score += 2.0
            else:
                st = turkish_stem(aa)
                if len(st) >= 4 and st in stems:
                    score += 1.0

        # normalize: küçük alias listeleri vs diye yumuşat
        if score <= 0:
            continue
        norm = score / 3.0  # 3 puan ~ güçlü
        if norm > best_score:
            best_score = norm
            best_tag = tag

    if best_tag and best_score >= min_score:
        return best_tag
    return None

def split_numeric_segments_with_anchor(clause_norm: str) -> List[Dict[str, Any]]:
    """
    normalize_text(clause) üzerinde çalışır ve indexler normalize edilmiş metinle uyumludur.
    Dönen her item:
      - seg: segment text
      - seg_start, seg_end: clause_norm içindeki index aralığı
      - anchor_in_seg: segment içindeki ilk sayının başlangıcı
      - anchor_abs: clause_norm içindeki mutlak anchor indexi
    """
    t = (clause_norm or "").strip()
    if not t:
        return []

    bases: List[Tuple[str, int]] = [(t, 0)]
    for m in re.finditer(r"\(([^)]{5,200})\)", t):
        inner = (m.group(1) or "").strip()
        if inner:
            bases.append((inner, m.start(1)))

    results: List[Dict[str, Any]] = []
    for base_text, base_start in bases:
        # split by ';' while preserving offsets
        for m1 in re.finditer(r"[^;]+", base_text):
            p1 = (m1.group(0) or "").strip()
            p1_start = base_start + m1.start()
            if not p1:
                continue

            # split by ", " pattern
            for m2 in re.finditer(r".+?(?:,\s+|$)", p1):
                p2_raw = m2.group(0) or ""
                p2 = p2_raw.strip(" ,.;:-")
                p2_start = p1_start + m2.start()
                if not p2:
                    continue

                p2_norm = p2
                if len(p2_norm) < 6:
                    continue
                if not NUMERIC_SIGNAL_RE.search(p2_norm):
                    continue

                # anchor in segment
                mpr = PERCENT_PREFIX_RANGE_RE.search(p2_norm)
                mps = PERCENT_PREFIX_SINGLE_RE.search(p2_norm) if not mpr else None
                mrange = RANGE_DASH_RE.search(p2_norm) if not mpr else None
                msingle = SINGLE_NUM_RE.search(p2_norm) if (not mpr and not mrange) else None
                mspace = None
                if not (mpr or mrange):
                    mspace = RANGE_SPACE_RE.search(p2_norm)

                if mpr:
                    anchor_in_seg = mpr.start()
                elif mps:
                    anchor_in_seg = mps.start()
                elif mrange:
                    anchor_in_seg = mrange.start()
                elif msingle:
                    anchor_in_seg = msingle.start()
                elif mspace:
                    anchor_in_seg = mspace.start()
                else:
                    continue

                seg_end = p2_start + len(p2_norm)

                results.append({
                    "seg": p2_norm,
                    "seg_start": p2_start,
                    "seg_end": seg_end,
                    "anchor_in_seg": anchor_in_seg,
                    "anchor_abs": p2_start + anchor_in_seg,
                })

    seen = set()
    final = []
    for r in results:
        k = normalize_text(r["seg"])
        if k in seen:
            continue
        seen.add(k)
        final.append(r)
    return final

def parse_numeric_constraints_with_unmatched(
    clause: str,
    matches: List[Tuple[str, int, int]],
    vocab_set: set,
    alias_index: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Sağlamlaştırılmış numeric parse:
    - tag seçimi segment içi eşleşmelerle sınırlı
    - macro segment -> macro hint
    - segment içi tag yoksa -> fuzzy window
    - tag hala yoksa numeric'e yazma; unmatched listesine ekle
    - period_days inherit (aynı clause içi)
    """
    if not clause or not clause.strip():
        return [], []

    clause_norm = normalize_text(clause)
    if not NUMERIC_SIGNAL_RE.search(clause_norm):
        return [], []

    constraints: List[Dict[str, Any]] = []
    unmatched: List[str] = []

    last_period_days: Optional[int] = None

    for item in split_numeric_segments_with_anchor(clause_norm):
        seg = item["seg"]
        seg_start = int(item["seg_start"])
        seg_end = int(item["seg_end"])
        anchor_in_seg = int(item["anchor_in_seg"])
        anchor_abs = int(item["anchor_abs"])

        seg_norm = normalize_text(seg)

        # kcal-only segmentlerini atla (kcal var ama başka birim yoksa)
        if KCAL_RE.search(seg_norm) and not re.search(r"\b(mg|gram|gr|g|%|yuzde|yüzde|kez|adet|porsiyon|bardak|dilim)\b", seg_norm):
            continue

        wants_min, wants_max = _wants_min_max(seg_norm)

        # percent special handling
        mpr = PERCENT_PREFIX_RANGE_RE.search(seg_norm)
        mps = PERCENT_PREFIX_SINGLE_RE.search(seg_norm) if not mpr else None

        mrange = RANGE_DASH_RE.search(seg_norm) if not mpr else None
        msingle = SINGLE_NUM_RE.search(seg_norm) if (not mpr and not mrange) else None

        mspace = None
        if not (mpr or mrange):
            mspace = RANGE_SPACE_RE.search(seg_norm)

        unit = detect_unit_nearest(seg_norm, anchor_in_seg)
        if unit not in ALLOWED_UNITS or unit == "unknown":
            if mpr or mps or "%" in seg_norm or "yüzde" in seg_norm or "yuzde" in seg_norm:
                unit = "yuzde"
            else:
                unmatched.append(seg.strip())
                continue

        period_days = guess_period_days_nearest(seg_norm, anchor_in_seg)
        if period_days is None and last_period_days is not None:
            period_days = last_period_days
        if period_days is not None:
            last_period_days = period_days

        min_count = max_count = None
        min_grams = max_grams = None

        # değer çıkarma
        if mpr:
            a = parse_num(mpr.group("a"))
            b = parse_num(mpr.group("b"))
            lo, hi = (a, b) if a <= b else (b, a)
            min_grams = int(lo) if float(lo).is_integer() else lo
            max_grams = int(hi) if float(hi).is_integer() else hi

        elif mrange:
            a = parse_num(mrange.group("a"))
            b = parse_num(mrange.group("b"))
            lo, hi = (a, b) if a <= b else (b, a)
            if unit in ("kez", "adet", "porsiyon", "bardak", "dilim", "cay_kasigi", "tatli_kasigi", "yemek_kasigi"):
                min_count = int(lo) if float(lo).is_integer() else lo
                max_count = int(hi) if float(hi).is_integer() else hi
            else:
                min_grams = int(lo) if float(lo).is_integer() else lo
                max_grams = int(hi) if float(hi).is_integer() else hi

        elif mspace:
            # Ünitenin bulunması şart
            if not re.search(r"\b(kez|adet|porsiyon|bardak|dilim|mg|g|gr|gram|%|yüzde|yuzde)\b", seg_norm, flags=re.IGNORECASE):
                unmatched.append(seg.strip())
                continue
            a = parse_num(mspace.group("a"))
            b = parse_num(mspace.group("b"))
            lo, hi = (a, b) if a <= b else (b, a)
            if unit in ("kez", "adet", "porsiyon", "bardak", "dilim", "cay_kasigi", "tatli_kasigi", "yemek_kasigi"):
                min_count = int(lo) if float(lo).is_integer() else lo
                max_count = int(hi) if float(hi).is_integer() else hi
            else:
                min_grams = int(lo) if float(lo).is_integer() else lo
                max_grams = int(hi) if float(hi).is_integer() else hi

        else:
            if not msingle:
                unmatched.append(seg.strip())
                continue
            val = parse_num(msingle.group("num"))
            if unit in ("kez", "adet", "porsiyon", "bardak", "dilim", "cay_kasigi", "tatli_kasigi", "yemek_kasigi"):
                if wants_min and not wants_max:
                    min_count = int(val) if float(val).is_integer() else val
                elif wants_max and not wants_min:
                    max_count = int(val) if float(val).is_integer() else val
                else:
                    if PREFER_PAT.search(seg_norm):
                        min_count = int(val) if float(val).is_integer() else val
                    else:
                        max_count = int(val) if float(val).is_integer() else val
            else:
                if wants_min and not wants_max:
                    min_grams = int(val) if float(val).is_integer() else val
                elif wants_max and not wants_min:
                    max_grams = int(val) if float(val).is_integer() else val
                else:
                    max_grams = int(val) if float(val).is_integer() else val

        if min_count is None and max_count is None and min_grams is None and max_grams is None:
            unmatched.append(seg.strip())
            continue

        # =========================
        # TAG BINDING (SEGMENT-LOCAL)
        # =========================
        seg_matches = filter_matches_in_span(matches, seg_start, seg_end)

        chosen_tag: Optional[str] = None

        # (1) Macro segment -> macro hint
        if is_macro_numeric_segment(seg_norm):
            chosen_tag = macro_tag_hint_for_segment(seg_norm, vocab_set)

        # (2) Segment içi tag span varsa proximity
        if not chosen_tag and seg_matches:
            chosen_tag = choose_numeric_tag_by_proximity_abs(anchor_abs, seg_matches)

        # (3) Fuzzy window (segment text) fallback
        if not chosen_tag:
            chosen_tag = fuzzy_tag_hint_from_window(seg_norm, alias_index, vocab_set, min_score=0.84)

        if not chosen_tag or chosen_tag not in vocab_set:
            unmatched.append(seg.strip())
            continue

        constraints.append({
            "tag": chosen_tag,
            "min_count": min_count,
            "max_count": max_count,
            "period_days": period_days,
            "min_grams": min_grams,
            "max_grams": max_grams,
            "unit": unit,
            "description": seg.strip()
        })

    return constraints, unmatched

# =========================
# MINER (LLM-free)
# =========================
def _intent_guard(tag: str, proposed_intent: str, tag_class: Dict[str, str]) -> str:
    cls = tag_class.get(tag, "neutral")
    if cls == "risk" and proposed_intent == "prefer":
        return "limit"
    if cls == "positive" and proposed_intent == "avoid":
        return "limit"
    return proposed_intent

def python_mine_clause(
    clause: str,
    used_chunks: List[Any],
    vocab_set: set,
    alias_index: Dict[str, List[str]],
    alias_re: re.Pattern,
    alias_to_tags: Dict[str, List[str]],
    tag_class: Dict[str, str],
) -> Dict[str, Any]:
    obj = ensure_schema({})
    c = (clause or "").strip()
    if not c:
        return obj

    chunk_id = None
    if isinstance(used_chunks, list) and used_chunks and isinstance(used_chunks[0], str):
        chunk_id = used_chunks[0]

    # 1) tag match
    matches = match_tags_with_spans(
        text=c,
        vocab_set=vocab_set,
        alias_index=alias_index,
        alias_re=alias_re,
        alias_to_tags=alias_to_tags,
    )

    # 2) replacement rules: "X yerine Y" (HYBRID)
    rep_pairs: List[Tuple[str, str]] = []
    if HEAVY_CLAUSE_RE.search(normalize_text(c)):
        rep_pairs = stanza_extract_yerine_pairs(c)
    if not rep_pairs:
        rep_pairs = extract_replacement_pairs(c)

    replacement_tags_prefer: List[str] = []
    replacement_tags_limit: List[str] = []
    replacement_tags_avoid: List[str] = []

    for x, y in rep_pairs:
        x_clean = clean_phrase_for_tag_match(x)
        y_clean = clean_phrase_for_tag_match(y)

        mx = match_tags_with_spans(x_clean, vocab_set, alias_index, alias_re, alias_to_tags)
        my = match_tags_with_spans(y_clean, vocab_set, alias_index, alias_re, alias_to_tags)
        x_tags = sorted(list({m[0] for m in mx}))
        y_tags = sorted(list({m[0] for m in my}))

        # "X yerine Y": X genelde azalt/kaçın, Y genelde tercih
        for t in x_tags:
            if tag_class.get(t) == "positive":
                replacement_tags_limit.append(t)
            else:
                replacement_tags_avoid.append(t)
        for t in y_tags:
            if tag_class.get(t) == "risk":
                replacement_tags_limit.append(t)
            else:
                replacement_tags_prefer.append(t)

    # 3) intent scoring
    tag_best: Dict[str, str] = {}

    def better(a: str, b: str) -> str:
        order = {"avoid": 3, "limit": 2, "prefer": 1, "unknown": 0}
        return a if order.get(a, 0) >= order.get(b, 0) else b

    for t in replacement_tags_prefer:
        tag_best[t] = better("prefer", tag_best.get(t, "unknown"))
    for t in replacement_tags_limit:
        tag_best[t] = better("limit", tag_best.get(t, "unknown"))
    for t in replacement_tags_avoid:
        tag_best[t] = better("avoid", tag_best.get(t, "unknown"))

    # window intent from spans
    c_norm = normalize_text(c)
    for (tag, s, e) in matches:
        if tag not in vocab_set:
            continue
        intent = intent_in_window(c_norm, s, e, window_tokens=10) if s >= 0 else detect_intent(c)
        if intent == "unknown":
            continue
        intent = _intent_guard(tag, intent, tag_class)
        tag_best[tag] = better(intent, tag_best.get(tag, "unknown"))

    prefer = [t for t, it in tag_best.items() if it == "prefer"]
    limit = [t for t, it in tag_best.items() if it == "limit"]
    avoid = [t for t, it in tag_best.items() if it == "avoid"]

    obj["prefer_tags"] = sorted(set(prefer))
    obj["limit_tags"] = sorted(set(limit))
    obj["avoid_tags"] = sorted(set(avoid))

    # 4) numeric constraints (ROBUST + unmatched -> recommendations)
    ncs, unmatched = parse_numeric_constraints_with_unmatched(
        c, matches, vocab_set=vocab_set, alias_index=alias_index
    )

    if ncs:
        obj["meal_pattern_rules"]["numeric_constraints"].extend(ncs)
        # numeric geçen tagleri en az limit'e al
        for nc in ncs:
            t = nc.get("tag")
            if isinstance(t, str):
                if t in obj["prefer_tags"]:
                    obj["prefer_tags"].remove(t)
                if t not in obj["limit_tags"] and t not in obj["avoid_tags"]:
                    obj["limit_tags"].append(t)

    # unmatched numeric -> recommendations (numeric'e yazma)
    for seg in unmatched:
        if isinstance(seg, str) and seg.strip():
            obj["recommendations"].append({
                "text": seg.strip(),
                "intent": "unmatched_numeric",
                "chunk_id": chunk_id
            })

    # 5) evidence
    for tag in obj["prefer_tags"] + obj["limit_tags"] + obj["avoid_tags"]:
        obj["rag_evidence"].append({"related_tags": [tag], "quote": c, "chunk_id": chunk_id})

    # 6) engine rules
    meals = extract_meal_context(c)
    engine_rules = []
    for m in meals:
        for tag in obj["prefer_tags"]:
            engine_rules.append({"rule": f"[{m}]=[{tag}]", "meal": m, "tag": tag, "intent": "prefer"})
        for tag in obj["limit_tags"]:
            engine_rules.append({"rule": f"[{m}]=[{tag}]", "meal": m, "tag": tag, "intent": "limit"})
        for tag in obj["avoid_tags"]:
            engine_rules.append({"rule": f"[{m}]=[{tag}]", "meal": m, "tag": tag, "intent": "avoid"})
    obj["meal_pattern_rules"]["engine_rules"] = engine_rules
    obj["meal_pattern_rules"]["logical_rules"]["prefer"] = obj["prefer_tags"]
    obj["meal_pattern_rules"]["logical_rules"]["limit"] = obj["limit_tags"]
    obj["meal_pattern_rules"]["logical_rules"]["avoid"] = obj["avoid_tags"]

    # 7) fallback recommendation (tags yoksa)
    if (not matches) and detect_intent(c) != "unknown":
        obj["recommendations"].append({
            "text": c,
            "intent": detect_intent(c),
            "chunk_id": chunk_id
        })

    return obj

# =========================
# Evidence validation (FAST + cache)
# =========================
_SUPPORTS_CACHE: Dict[Tuple[str, str], bool] = {}

def quote_supports_tag(tag: str, quote: str, alias_index: Dict[str, List[str]]) -> bool:
    qk = normalize_quote_key(quote)
    ck = (tag, qk)
    if ck in _SUPPORTS_CACHE:
        return _SUPPORTS_CACHE[ck]

    q = normalize_text(quote)
    if not q:
        _SUPPORTS_CACHE[ck] = False
        return False

    aliases = alias_index.get(tag) or []
    stems = set(tokenize_stems(q))

    ok = False
    for a in aliases:
        aa = normalize_text(a)
        if not aa:
            continue
        if " " in aa:
            if aa in q:
                ok = True
                break
        else:
            st = turkish_stem(aa)
            if len(st) >= 4 and st in stems:
                ok = True
                break

    _SUPPORTS_CACHE[ck] = ok
    return ok

def normalize_numeric_constraint(nc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    unit = (nc.get("unit") or "unknown")
    if unit not in ALLOWED_UNITS:
        return None

    def to_num_or_none(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            m = re.search(r"-?\d+(?:[.,]\d+)?", v)
            if m:
                return float(m.group(0).replace(",", "."))
        return None

    out = {
        "tag": nc.get("tag"),
        "min_count": to_num_or_none(nc.get("min_count")),
        "max_count": to_num_or_none(nc.get("max_count")),
        "period_days": int(to_num_or_none(nc.get("period_days"))) if nc.get("period_days") is not None else None,
        "min_grams": to_num_or_none(nc.get("min_grams")),
        "max_grams": to_num_or_none(nc.get("max_grams")),
        "unit": unit,
        "description": (nc.get("description") or "").strip(),
    }

    if not out["description"]:
        return None
    if out["min_count"] is None and out["max_count"] is None and out["min_grams"] is None and out["max_grams"] is None:
        return None
    return out

MAX_TAGS_PER_SAME_QUOTE = 2
MAX_EVIDENCE_PER_TAG = 2

def prune_evidence(obj: Dict[str, Any], vocab: set, alias_index: Dict[str, List[str]], report: Dict[str, Any]) -> None:
    evs = obj.get("rag_evidence", [])
    if not isinstance(evs, list):
        obj["rag_evidence"] = []
        return

    cleaned: List[Dict[str, Any]] = []
    for ev in evs:
        if not isinstance(ev, dict):
            continue
        rt = ev.get("related_tags")
        quote = ev.get("quote")

        if not (isinstance(rt, list) and len(rt) == 1 and isinstance(rt[0], str)):
            continue
        tag = rt[0]
        if tag not in vocab:
            report["dropped_unknown_tags"].append({"tag": tag, "class": "rag_evidence"})
            continue
        if not isinstance(quote, str) or not quote.strip():
            report["missing_evidence_tags"].append({"tag": tag, "class": "rag_evidence_empty_quote"})
            continue
        if len(quote.split()) < 4:
            report.setdefault("dropped_short_quote", []).append({"tag": tag, "quote": quote[:80]})
            continue
        if not quote_supports_tag(tag, quote, alias_index):
            report.setdefault("dropped_unsupported_evidence", []).append({"tag": tag, "quote": quote[:120]})
            continue

        chunk_id = ev.get("chunk_id", None)
        if chunk_id is not None and not isinstance(chunk_id, str):
            chunk_id = None

        cleaned.append({"related_tags": [tag], "quote": quote.strip(), "chunk_id": chunk_id})

    by_quote: Dict[str, List[Dict[str, Any]]] = {}
    for ev in cleaned:
        qk = normalize_quote_key(ev["quote"])
        by_quote.setdefault(qk, []).append(ev)

    cleaned2: List[Dict[str, Any]] = []
    for _, group in by_quote.items():
        if len(group) <= MAX_TAGS_PER_SAME_QUOTE:
            cleaned2.extend(group)
        else:
            report.setdefault("quote_reuse_pruned", []).append(
                {"quote": group[0]["quote"][:120], "dropped": len(group) - MAX_TAGS_PER_SAME_QUOTE}
            )
            cleaned2.extend(group[:MAX_TAGS_PER_SAME_QUOTE])

    by_tag: Dict[str, List[Dict[str, Any]]] = {}
    for ev in cleaned2:
        tag = ev["related_tags"][0]
        by_tag.setdefault(tag, []).append(ev)

    final: List[Dict[str, Any]] = []
    for tag, group in by_tag.items():
        if len(group) <= MAX_EVIDENCE_PER_TAG:
            final.extend(group)
        else:
            report.setdefault("evidence_per_tag_pruned", []).append({"tag": tag, "dropped": len(group) - MAX_EVIDENCE_PER_TAG})
            final.extend(group[:MAX_EVIDENCE_PER_TAG])

    obj["rag_evidence"] = final

def rebuild_tags_from_evidence(obj: Dict[str, Any], vocab: set) -> None:
    evidence_tags = set()
    for ev in obj.get("rag_evidence", []):
        if isinstance(ev, dict):
            rt = ev.get("related_tags")
            quote = ev.get("quote")
            if isinstance(rt, list) and len(rt) == 1 and isinstance(rt[0], str) and isinstance(quote, str) and quote.strip():
                if rt[0] in vocab:
                    evidence_tags.add(rt[0])

    def filter_by_evidence(tags: List[str]) -> List[str]:
        return uniq([t for t in tags if t in evidence_tags])

    obj["prefer_tags"] = filter_by_evidence(obj.get("prefer_tags", []))
    obj["limit_tags"] = filter_by_evidence(obj.get("limit_tags", []))
    obj["avoid_tags"] = filter_by_evidence(obj.get("avoid_tags", []))

# =========================
# MUTEX (optional)
# =========================
MUTEX_GROUPS: Dict[str, List[str]] = {
    "sodyum": ["sodyum_dusuk", "sodyum_orta", "sodyum_yuksek"],
    "gi": ["gi_dusuk", "gi_orta", "gi_yuksek"],
    "kh": ["kh_dusuk", "kh_orta", "kh_yuksek"],
    "lif": ["lif_dusuk", "lif_orta", "lif_yuksek"],
    "doymus_yag": ["doymus_yag_dusuk", "doymus_yag_orta", "doymus_yag_yuksek"],
}

MUTEX_RISK_ORDER: Dict[str, List[str]] = {
    "sodyum": ["sodyum_yuksek", "sodyum_orta", "sodyum_dusuk"],
    "gi": ["gi_yuksek", "gi_orta", "gi_dusuk"],
    "kh": ["kh_yuksek", "kh_orta", "kh_dusuk"],
    "lif": ["lif_dusuk", "lif_orta", "lif_yuksek"],
    "doymus_yag": ["doymus_yag_yuksek", "doymus_yag_orta", "doymus_yag_dusuk"],
}

def apply_mutex(tags: List[str], group: List[str], risk_order: List[str]) -> List[str]:
    present = [t for t in tags if t in group]
    if len(present) <= 1:
        return tags
    keep = None
    for r in risk_order:
        if r in present:
            keep = r
            break
    if keep is None:
        keep = present[0]
    return [t for t in tags if t not in group] + [keep]

def enforce_mutex_all(obj: Dict[str, Any]) -> None:
    for grp, members in MUTEX_GROUPS.items():
        risk = MUTEX_RISK_ORDER.get(grp, members)
        obj["prefer_tags"] = apply_mutex(obj["prefer_tags"], members, risk)
        obj["limit_tags"] = apply_mutex(obj["limit_tags"], members, risk)
        obj["avoid_tags"] = apply_mutex(obj["avoid_tags"], members, risk)

# =========================
# ENGINE RULES SANITY
# =========================
def prune_engine_rules(obj: Dict[str, Any]) -> None:
    obj = ensure_schema(obj)
    allowed_tags = set(obj["prefer_tags"] + obj["limit_tags"] + obj["avoid_tags"])
    allowed_intents = {"prefer", "limit", "avoid"}

    out = []
    for r in obj["meal_pattern_rules"].get("engine_rules", []) or []:
        if not isinstance(r, dict):
            continue
        meal = r.get("meal")
        tag = r.get("tag")
        intent = r.get("intent")
        rule = r.get("rule")
        if not (isinstance(meal, str) and isinstance(tag, str) and isinstance(intent, str) and isinstance(rule, str)):
            continue
        if intent not in allowed_intents:
            continue
        if tag not in allowed_tags:
            continue
        if rule != f"[{meal}]=[{tag}]":
            continue
        out.append({"rule": rule, "meal": meal, "tag": tag, "intent": intent})

    seen = set()
    final = []
    for r in out:
        k = (r["meal"], r["tag"], r["intent"])
        if k in seen:
            continue
        seen.add(k)
        final.append(r)

    obj["meal_pattern_rules"]["engine_rules"] = final

# =========================
# VALIDATE + FIX
# =========================
def validate_and_fix(
    obj: Dict[str, Any],
    vocab: set,
    alias_index: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report: Dict[str, Any] = {
        "dropped_unknown_tags": [],
        "missing_evidence_tags": [],
        "numeric_invalid_units": [],
        "numeric_missing_limit_tag": [],
        "dropped_unsupported_evidence": [],
        "quote_reuse_pruned": [],
        "evidence_per_tag_pruned": [],
        "numeric_normalized": 0,
        "dropped_numeric_empty": 0,
    }

    obj = ensure_schema(obj)

    def filter_vocab(tags: Any, cls: str) -> List[str]:
        if not isinstance(tags, list):
            return []
        out = []
        for t in tags:
            if not isinstance(t, str):
                continue
            if t not in vocab:
                report["dropped_unknown_tags"].append({"tag": t, "class": cls})
                continue
            out.append(t)
        return uniq(out)

    obj["prefer_tags"] = filter_vocab(obj.get("prefer_tags"), "prefer")
    obj["limit_tags"] = filter_vocab(obj.get("limit_tags"), "limit")
    obj["avoid_tags"] = filter_vocab(obj.get("avoid_tags"), "avoid")

    prune_evidence(obj, vocab, alias_index, report)

    ncs = obj.get("meal_pattern_rules", {}).get("numeric_constraints", [])
    if not isinstance(ncs, list):
        ncs = []

    fixed_numeric: List[Dict[str, Any]] = []
    for nc in ncs:
        if not isinstance(nc, dict):
            continue
        tag = nc.get("tag")
        if not isinstance(tag, str) or tag not in vocab:
            report["dropped_unknown_tags"].append({"tag": tag, "class": "numeric_constraints"})
            continue

        nnc = normalize_numeric_constraint(nc)
        report["numeric_normalized"] += 1

        if nnc is None:
            report["dropped_numeric_empty"] += 1
            continue

        if nnc["unit"] not in ALLOWED_UNITS:
            report["numeric_invalid_units"].append({"tag": tag, "unit": nnc["unit"]})
            continue

        # numeric => en az limit'e al
        if tag not in obj["limit_tags"] and tag not in obj["avoid_tags"]:
            obj["limit_tags"].append(tag)
            report["numeric_missing_limit_tag"].append(tag)

        fixed_numeric.append(nnc)

    obj["meal_pattern_rules"]["numeric_constraints"] = fixed_numeric
    obj["limit_tags"] = uniq(obj["limit_tags"])

    rebuild_tags_from_evidence(obj, vocab)
    enforce_mutex_all(obj)

    prefer = set(obj["prefer_tags"])
    limit = set(obj["limit_tags"])
    avoid = set(obj["avoid_tags"])
    prefer = prefer - limit - avoid
    limit = limit - avoid

    obj["prefer_tags"] = sorted(prefer)
    obj["limit_tags"] = sorted(limit)
    obj["avoid_tags"] = sorted(avoid)

    obj["meal_pattern_rules"]["logical_rules"]["prefer"] = obj["prefer_tags"]
    obj["meal_pattern_rules"]["logical_rules"]["limit"] = obj["limit_tags"]
    obj["meal_pattern_rules"]["logical_rules"]["avoid"] = obj["avoid_tags"]

    prune_engine_rules(obj)

    # evidence coverage check
    evidence_tags = set()
    for ev in obj.get("rag_evidence", []):
        if isinstance(ev, dict):
            rt = ev.get("related_tags")
            quote = ev.get("quote")
            if isinstance(rt, list) and len(rt) == 1 and isinstance(rt[0], str) and isinstance(quote, str) and quote.strip():
                evidence_tags.add(rt[0])

    for cls, tags in [("prefer", obj["prefer_tags"]), ("limit", obj["limit_tags"]), ("avoid", obj["avoid_tags"])]:
        for t in tags:
            if t not in evidence_tags:
                report["missing_evidence_tags"].append({"tag": t, "class": cls})

    return obj, report

def merge_outputs(base: Dict[str, Any], part: Dict[str, Any]) -> Dict[str, Any]:
    base = ensure_schema(base)
    part = ensure_schema(part)

    base["prefer_tags"] = uniq(base["prefer_tags"] + part["prefer_tags"])
    base["limit_tags"] = uniq(base["limit_tags"] + part["limit_tags"])
    base["avoid_tags"] = uniq(base["avoid_tags"] + part["avoid_tags"])

    base["meal_pattern_rules"]["numeric_constraints"] = base["meal_pattern_rules"]["numeric_constraints"] + part["meal_pattern_rules"]["numeric_constraints"]
    base["meal_pattern_rules"]["engine_rules"] = base["meal_pattern_rules"]["engine_rules"] + part["meal_pattern_rules"]["engine_rules"]
    base["recommendations"] = base["recommendations"] + part["recommendations"]
    base["rag_evidence"] = base["rag_evidence"] + part["rag_evidence"]

    base["meal_pattern_rules"]["logical_rules"]["prefer"] = base["prefer_tags"]
    base["meal_pattern_rules"]["logical_rules"]["limit"] = base["limit_tags"]
    base["meal_pattern_rules"]["logical_rules"]["avoid"] = base["avoid_tags"]
    return base

# =========================
# IO
# =========================
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def discover_domains(base_dir: str) -> List[str]:
    domains = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("."):
            continue
        if name == "labels_output":
            continue
        domains.append(name)
    return sorted(domains)

def input_path_for(domain: str) -> str:
    direct = os.path.join(BASE_DIR, domain, "qa_summary.json")
    if os.path.isfile(direct):
        return direct
    for root, _, files in os.walk(os.path.join(BASE_DIR, domain)):
        if "qa_summary.json" in files:
            return os.path.join(root, "qa_summary.json")
    return direct

# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUTPUT_ROOT)

    tag_dicts = load_json(TAG_DICTS_PATH)
    if "tags" not in tag_dicts or not isinstance(tag_dicts["tags"], dict):
        raise ValueError("tag_dicts.json içinde 'tags' objesi bulunamadı.")

    tag_desc: Dict[str, str] = tag_dicts["tags"]
    vocab_set = set(tag_desc.keys())

    # AUTO: all tags + alias index (only tags from tag_dicts.json)
    alias_index = build_alias_index_from_tag_dicts(tag_desc)
    ALIAS_RE, ALIAS_TO_TAGS = build_alias_regex_index(alias_index)

    # AUTO: tag class from tag_dicts.json
    TAG_CLASS = build_tag_class(tag_desc)

    domains = discover_domains(BASE_DIR)
    print("BASE_DIR:", BASE_DIR)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)
    print("DOMAINS:", domains)

    global_index = {
        "base_dir": BASE_DIR,
        "tag_dicts": TAG_DICTS_PATH,
        "output_root": OUTPUT_ROOT,
        "domains": domains,
        "runs": []
    }

    for domain in domains:
        domain_out_root = os.path.join(OUTPUT_ROOT, domain, "python_only")
        ensure_dir(domain_out_root)

        inp = input_path_for(domain)
        run_record = {"domain": domain, "input": inp, "status": None, "error": None}

        if not os.path.isfile(inp):
            run_record["status"] = "missing_input"
            global_index["runs"].append(run_record)
            print(f"[SKIP missing] {domain} -> {inp}")
            continue

        print(f"\n=== RUN (PYTHON ONLY): domain={domain} ===")

        try:
            qa_summary = load_json(inp)
            items = qa_summary.get("items", [])
            if not isinstance(items, list):
                items = []

            clause_items: List[Dict[str, Any]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                ans = it.get("answer")
                if not isinstance(ans, str) or not ans.strip():
                    continue
                if "Kaynaklarda bu soru için bilgi bulunamadı" in ans:
                    continue

                pre = preprocess_answer(ans, make_bullets=True)
                clauses = split_into_clauses(pre)

                for cl in clauses:
                    clause_items.append({"answer": cl, "used_chunks": it.get("used_chunks", [])})

            final_obj = ensure_schema({
                "prefer_tags": [],
                "limit_tags": [],
                "avoid_tags": [],
                "meal_pattern_rules": {"logical_rules": {"prefer": [], "limit": [], "avoid": []}, "numeric_constraints": [], "engine_rules": []},
                "energy_rules": {"scale_up_order": [], "scale_down_order": [], "locks": []},
                "recommendations": [],
                "rag_evidence": []
            })

            combined_report = {"clauses_total": len(clause_items), "clauses_mined": 0, "final_pass": None}

            for it in clause_items:
                mined = python_mine_clause(
                    clause=it["answer"],
                    used_chunks=it.get("used_chunks", []),
                    vocab_set=vocab_set,
                    alias_index=alias_index,
                    alias_re=ALIAS_RE,
                    alias_to_tags=ALIAS_TO_TAGS,
                    tag_class=TAG_CLASS,
                )

                has_any = bool(
                    mined.get("prefer_tags") or mined.get("limit_tags") or mined.get("avoid_tags") or
                    mined.get("meal_pattern_rules", {}).get("numeric_constraints") or
                    mined.get("meal_pattern_rules", {}).get("engine_rules") or
                    mined.get("rag_evidence") or mined.get("recommendations")
                )
                if not has_any:
                    continue

                combined_report["clauses_mined"] += 1
                fixed_mined, _ = validate_and_fix(mined, vocab_set, alias_index)
                final_obj = merge_outputs(final_obj, fixed_mined)

            final_obj, final_report = validate_and_fix(final_obj, vocab_set, alias_index)
            combined_report["final_pass"] = final_report

            save_json(os.path.join(domain_out_root, "labels.json"), final_obj)
            save_json(os.path.join(domain_out_root, "report.json"), combined_report)

            run_record["status"] = "ok"
            global_index["runs"].append(run_record)
            print(f"Saved: {domain_out_root}\\labels.json")

        except Exception as e:
            save_json(os.path.join(domain_out_root, "error.json"), {"error": str(e)})
            run_record["status"] = "error"
            run_record["error"] = str(e)
            global_index["runs"].append(run_record)
            print("ERROR:", e)

    save_json(os.path.join(OUTPUT_ROOT, "_index.json"), global_index)
    print("\nDONE.")

if __name__ == "__main__":
    main()
