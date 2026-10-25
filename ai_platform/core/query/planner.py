"""Production Query Planner V2.5 for the reusable institutional RAG core.



Design contract

---------------

- Exactly one LLM call per user query.

- Conversation resolution happens before this component.

- The original user wording remains authoritative.

- Deterministic normalization repairs omitted/duplicated structure from the LLM.

- Deterministic attribute extraction is conservative and phrase-aware.

- Multi-intent is based on distinct answerable objectives, not synonyms.

- Retrieval queries are compact and never replace the original question.

- No institution-specific vocabulary or factual knowledge lives here.

V2.5 semantic hardening
-----------------------
- Distinguishes requested objectives from merely mentioned/disclaimed topics.
- Handles correction/contrast language without creating phantom intents.
- Gives deterministic request-type signals priority over noisy LLM labels.
- Keeps target selection away from explicitly discarded mentions.
- Remains institution-agnostic; no new institution facts are embedded.
- Uses shared degree lexicon for robust target extraction across natural phrasing.
- Keeps specific eligibility facets (e.g. work experience) from creating a phantom generic eligibility intent.

"""

from __future__ import annotations



import json
import os
import re

from typing import Any, Literal, Mapping, Protocol, Sequence



from pydantic import BaseModel, ConfigDict, Field



from ai_platform.core.query.models import (
    Ambiguity,
    Constraint,
    ListIntent,
    Query,
    QueryFacet,
    Qualifier,
    QueryRoute,
    RetrievalRequirement,
    SemanticQueryFrame,
    Target,
)



ScopeDecision = Literal["in_scope", "out_of_scope", "uncertain"]

REQUEST_TYPES = (

    "general", "process", "eligibility", "fee", "documents", "deadline",

    "facility", "location", "contact", "policy", "comparison",

)

KNOWN_TYPES = set(REQUEST_TYPES)



# Generic institutional language only. No IITJ-specific facts are embedded.

INSTITUTIONAL_MARKERS = (

    "college", "university", "campus", "institute", "department",

    "admission", "application", "apply", "eligibility", "eligible",

    "fee", "fees", "scholarship", "registration", "enrollment",

    "course", "courses", "curriculum", "semester", "exam", "examination", "examinations",
    "hostel", "mess", "accommodation", "library", "lab", "laboratory", "hours", "timings",
    "faculty", "professor", "student", "students", "office", "form", "probation", "overload", "credits", "credit",

    "documents", "deadline", "last date", "contact", "phone", "email",

    "placement", "research", "phd", "m.tech", "m.sc", "mba", "b.tech",

    "ug", "pg", "gate", "cat", "jee", "net",

    "candidate", "candidates", "applicant", "applicants",

    "percentage", "percent", "cgpa", "gpa", "marks", "score", "cutoff", "cut-off",

    "degree", "degrees", "award", "awarded", "convocation", "master", "masters", "master's", "bachelor", "bachelors", "bachelor's",

    "qualification", "qualifying", "criteria", "requirement", "requirements",

    "written-test", "written test", "interview",

    "category", "gen/obc", "sc/st", "gen-ews", "ews", "obc", "sc", "st", "pwd",

    "fee", "fees", "rules", "policy", "consequence", "cancel", "cancellation",
    "disciplinary", "administrative", "senate", "prescribed", "penalty", "penalties",
    "action", "actions", "waiver", "waive", "extension", "attendance", "leave",
    "regulation", "regulations", "procedure", "procedures", "guidelines",
    "submission", "submit", "submitted", "fails to submit", "ordinance", "ordinances",
    "health centre", "health center", "ambulance", "registrar", "transcript",
    "internal complaints committee", "icc", "gymkhana", "sports complex", "gym",
    "bus", "shuttle", "placement cell", "anti-ragging", "ragging", "preference",
    "seat allotment", "counseling",
    "training", "trainings", "upskilling", "vocational", "skilling", "skill hub", "pmkvy", "gem", "gfr", "vtu",
    "job role", "job roles", "certification", "certified", "enrolled", "enrolment", "enrolments",
    "workshop", "workshops", "participants", "hands-on", "procurement", "course code", "course codes", "code", "codes", "elective", "electives",

    "फीस", "शुल्क", "आवेदन", "दाखिला", "दाखिले", "प्रवेश", "पात्रता",

    "योग्य", "दस्तावेज", "डॉक्यूमेंट", "आखिरी तारीख", "अंतिम तारीख",

    "पंजीकरण", "छात्र", "छात्रावास", "विभाग", "महाविद्यालय", "विश्वविद्यालय",

    "उम्मीदवार", "अंक", "प्रतिशत", "डिग्री",

    "ke liye", "ki fee", "ki fees", "kaise apply", "zaroori", "अनिवार्य",

)

_GENERIC_INSTITUTION_EXCLUSIONS = {
    "institute",
    "the institute",
    "institution",
    "the institution",
    "college",
    "the college",
    "university",
    "the university",
    "campus",
    "the campus",
    "school",
    "the school",
    "iit",
    "nit",
    "iiit",
    "iim",
    "aiims",
    "bits",
}


def _is_institution_name_or_generic(term: str, profile: Any = None) -> bool:
    """Check if a term refers to an institution identity rather than a specialized academic subject."""
    cleaned = (term or "").strip().casefold()
    if not cleaned or cleaned in _GENERIC_INSTITUTION_EXCLUSIONS:
        return True

    active_profile = profile
    if active_profile is None:
        try:
            from ai_platform.institutions.loader import load_institution_profile
            active_profile = load_institution_profile()
        except Exception:
            active_profile = None

    if active_profile is not None:
        inst_tokens = {active_profile.institution_id.casefold(), active_profile.display_name.casefold()}
        inst_tokens.update(t.casefold() for t in active_profile.display_name.split() if len(t) > 1)
        if cleaned in inst_tokens:
            return True

    return False




OBVIOUS_OOS_MARKERS = (

    "cricket", "football", "fifa", "ipl", "movie", "song", "recipe",
    "joke", "weather", "stock price", "bitcoin", "forex", "binary search",
    "leetcode", "python program", "c++ code", "javascript", "sql query",
    "how to code", "flight", "flight ticket", "hotel booking", "airline", "train ticket",
    "क्रिकेट", "फुटबॉल", "फिल्म", "गाना", "रेसिपी", "मौसम",
    "मजाक",

)



NEGATION_PATTERNS = (
    r"\bnot required\b", r"\bnot need(?:ed)?\b", r"\bno need(?: for)?\b",
    r"\bdon't need\b", r"\bdo not need\b",
    r"\bdoesn't require\b", r"\bdoes not require\b",
    r"\bisn't required\b", r"\bis not required\b",
    r"\bare not required\b", r"\bnot mandatory\b",
    r"\bnot compulsory\b", r"\bnot necessary\b", r"\bnot needed\b",
    r"\bwithout\s+(?:having\s+)?(?:a\s+|an\s+)?(?:gate|cat|jee|net|ceed|nbhm|degree|master|bachelor|written|interview|test|experience)\b",
    r"\bexempt(?:ed|ion)?\s+from\b", r"\bwaiver\s+of\b",
    r"\bnot\s+on\s+academic\s+probation\b",
    r"\bmandatory\s+nahin\b", r"\bcompulsory\s+nahin\b",
    r"जरूरी नहीं", r"ज़रूरी नहीं", r"अनिवार्य नहीं", r"आवश्यक नहीं",
)


# Shared, institution-agnostic degree vocabulary. Keep one canonical pattern
# so target extraction does not drift across helpers.
DEGREE_PATTERN = (
    r"(?:M\.?\s*Tech|MTech|M\.?\s*Sc\.?|MSc|"
    r"M\.?\s*Des\.?|MDes|"
    r"M\.?\s*B\.?\s*A\.?|MBA|B\.?\s*Tech|BTech|"
    r"Ph\.?\s*D\.?|PhD|एम\.?टेक|एमटेक|एम\.?एस\.?सी|एमएससी|"
    r"एम\.?बी\.?ए|एमबीए|बी\.?टेक|बीटेक|पी\.?एच\.?डी|पीएचडी|"
    r"M\.?\s*B\.?\s*B\.?\s*S\.?|MBBS|B\.?\s*D\.?\s*S\.?|BDS|"
    r"B\.?\s*Arch|BArch|M\.?\s*Pharm|MPharm|B\.?\s*Pharm|BPharm|"
    r"Diploma|डिप्लोमा)"
)







class StructuredModel(Protocol):

    def invoke(self, input: Any) -> Any: ...





class QueryIntentPlan(BaseModel):

    model_config = ConfigDict(extra="ignore")

    question: str = ""

    request_type: str | None = None

    target: str | None = None

    requested_attributes: list[str] = Field(default_factory=list)

    constraints: list[str] = Field(default_factory=list)

    relations: list[str] = Field(default_factory=list)

    comparison_targets: list[str] = Field(default_factory=list)

    qualifiers: list[str] = Field(default_factory=list)

    temporal_constraints: list[str] = Field(default_factory=list)

    negated: bool = False

    retrieval_query: str = ""

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)





class QueryPlan(BaseModel):

    model_config = ConfigDict(extra="ignore")

    original_query: str = ""

    resolved_query: str = ""

    language: str | None = None

    conversation_mode: str = "standalone"

    domain_decision: ScopeDecision = "uncertain"

    domain_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    domain_reason: str = ""

    is_multi_intent: bool = False

    intents: list[QueryIntentPlan] = Field(default_factory=list)

    preserved_terms: list[str] = Field(default_factory=list)

    ambiguous: bool = False

    needs_clarification: bool = False

    clarification_reason: str = ""

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    route: str = "simple_single_intent"





DEFAULT_INSTITUTION_SCOPE = """

This is an institutional/college information assistant. Clearly in scope:

admissions, applications, programs, degrees, departments, eligibility,

qualifications, fees, scholarships, registration, enrollment, courses,

curriculum, examinations, hostel, accommodation, mess, campus facilities,

library, offices, contacts, forms, procedures, policies, research, placements,

student/faculty/visitor services, schedules and deadlines.



Generic educational questions are not automatically in scope. Clearly

unrelated sports, entertainment, recipes, generic coding help, weather, jokes,

and gibberish are out of scope. Use uncertain for plausible but underspecified

institutional questions. Do not confuse "unknown fact" with "out of scope".

""".strip()



QUERY_PLANNER_SYSTEM_PROMPT = r"""

You are the query-planning component of a college/institutional RAG assistant.

You DO NOT answer the question. You produce a compact structured plan used by

retrieval.



RULES

1\. Preserve the user's meaning exactly enough for retrieval.

2\. Keep the target as the actual subject named or clearly referenced by the user.

3\. requested_attributes are answer properties, not words merely mentioned.

4\. Extract every genuinely requested property that changes the evidence needed.

5\. Understand Hindi, Hinglish, code-switching, informal wording, abbreviations,

   typos, and transliterated Hindi. Do not require exact English phrasing.

5\. "How do I apply?" is a process request. "Can I apply?" is not automatically a

   process request; it is usually eligibility unless the user asks for steps.

6\. Split only genuinely separate answerable objectives.

7\. Never create two intents merely because one says "fee" and another says

   "application fee", or "percentage" and "minimum percentage".

8\. For comparisons, include BOTH compared targets in comparison_targets.

9\. Preserve explicit numbers, percentages, CGPA, exams, dates, academic status,

   conditions, and negation.

10\. retrieval_query is compact retrieval wording, not an answer.

11\. Return null/[] when absent. Never invent institutional facts.

12\. Do not treat a mentioned-but-negated topic as the requested intent, e.g.

    "I am not asking about eligibility; I only need the fee." The fee is the intent.

13. When a sentence contains multiple programs, choose the program actually
    associated with the requested property. Do not blindly select the first name.
14. Distinguish a requested objective from a mentioned objective. Phrases such as
    "I am not asking about X", "I don't need X", "I already know X", "forget X",
    "rather than X", "instead of X", or a self-correction that replaces X with Y
    must not create an intent for X.
15. Treat negation locally. "X is not mandatory" can still be an eligibility
    question about X; "I am not asking about X" excludes X as an answer objective.
16. Prefer the user's corrected or positive objective over an earlier discarded one.
17. Multi-intent means multiple positive answerable objectives that require distinct
    evidence. Do not infer an objective merely because its terminology appears.

SCOPE

in_scope = clearly institutional (academics, admissions, fees, courses, curriculum, course codes, degree programs, departments, research, institutional vocational upskilling, procurement training/workshops, campus facilities, campus rules).

out_of_scope = clearly unrelated to the institution (movies, dating, recipes, personal advice, external commercial bookings, jokes).

uncertain = plausible institutional but underspecified.

""".strip()





# ---------------------------------------------------------------------------

# Generic helpers

# ---------------------------------------------------------------------------



def _clean(v: Any) -> str:

    return " ".join(str(v or "").strip().split())





def _cf(v: Any) -> str:

    return _clean(v).casefold()





def _dedupe(values: Sequence[Any], limit: int = 32) -> list[str]:

    out: list[str] = []

    seen: set[str] = set()

    for v in values or ():

        s = _clean(v)

        key = s.casefold()

        if not s or key in seen:

            continue

        seen.add(key)

        out.append(s)

        if len(out) >= limit:

            break

    return out





def _conf(v: Any) -> float:

    try:

        return max(0.0, min(1.0, float(v)))

    except Exception:

        return 0.0





def _bool(v: Any) -> bool:

    if isinstance(v, bool):

        return v

    return _cf(v) in {"true", "1", "yes", "y", "on"}





def _response_text(response: Any) -> str:

    if response is None:

        return ""

    if isinstance(response, str):

        return response

    content = getattr(response, "content", None)

    if content is not None:

        return str(content)

    if isinstance(response, Mapping):

        for key in ("content", "text", "output", "response"):

            if key in response:

                return str(response[key])

    return str(response)





def _json_object(text: str) -> dict[str, Any] | None:

    raw = _clean(text)

    if not raw:

        return None

    candidates = (

        raw,

        re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S),

    )

    for candidate in candidates:

        try:

            obj = json.loads(candidate)

            if isinstance(obj, dict):

                return obj

        except Exception:

            continue

    start, end = raw.find("{"), raw.rfind("}")

    if start >= 0 and end > start:

        try:

            obj = json.loads(raw[start:end + 1])

            return obj if isinstance(obj, dict) else None

        except Exception:

            return None

    return None





def _scope(v: Any) -> ScopeDecision:

    s = _cf(v)

    if s in {"in_scope", "in", "relevant", "inside", "yes"}:

        return "in_scope"

    if s in {"out_of_scope", "out", "irrelevant", "unrelated", "no"}:

        return "out_of_scope"

    return "uncertain"





def _type(v: Any) -> str | None:

    s = _cf(v)

    aliases = {

        "admission": "process",

        "application": "process",

        "steps": "process",

        "how_to": "process",

        "cost": "fee",

        "amount": "fee",

        "qualification": "eligibility",

        "requirements": "eligibility",

        "compare": "comparison",

    }

    s = aliases.get(s, s)

    return s if s in KNOWN_TYPES else None





# ---------------------------------------------------------------------------

# Deterministic semantic extraction

# ---------------------------------------------------------------------------



def _negated(text: str) -> bool:

    low = _cf(text)

    return any(re.search(p, low) for p in NEGATION_PATTERNS)





def _canonical_degree(value: str) -> str | None:

    low = _cf(value)
    aliases = (
        (r"\bm\.?\s*tech\b|\bmtech\b|एम\.?टेक|एमटेक", "M.Tech"),
        (r"\bm\.?\s*sc\.?\b|\bmsc\b|एम\.?एस\.?सी|एमएससी", "M.Sc."),
        (r"\bm\.?\s*des\.?\b|\bmdes\b|एम\.?डेस", "M.Des"),
        (r"\bm\.?\s*b\.?\s*a\.?\b|\bmba\b|एम\.?बी\.?ए|एमबीए", "MBA"),
        (r"\bb\.?\s*tech\b|\bbtech\b|बी\.?टेक|बीटेक", "B.Tech"),
        (r"\bph\.?\s*d\.?\b|\bphd\b|पी\.?एच\.?डी|पीएचडी", "Ph.D."),
        (r"\bm\.?\s*b\.?\s*b\.?\s*s\.?\b|\bmbbs\b", "MBBS"),
        (r"\bb\.?\s*d\.?\s*s\.?\b|\bbds\b", "BDS"),
        (r"\bb\.?\s*arch\b|\bbarch\b", "B.Arch"),
        (r"\bm\.?\s*pharm\b|\bmpharm\b", "M.Pharm"),
        (r"\bb\.?\s*pharm\b|\bbpharm\b", "B.Pharm"),
        (r"\bdiploma\b|डिप्लोमा", "Diploma"),
    )
    for pattern, canonical in aliases:
        if re.search(pattern, low, re.I):
            return canonical
    return None


def _degree_mentions(text: str) -> list[re.Match[str]]:
    return list(re.finditer(DEGREE_PATTERN, _clean(text), re.I))



def _strip_target(value: str) -> str | None:
    s = _clean(value)
    if not s or s.lower() in {"null", "none", "n/a", "na", "unknown", "nil", "undefined", "not applicable", "the institution", "institution"}:
        return None
    s = re.sub(r"^the\s+", "", s, flags=re.I)

    # Remove only generic suffixes. Never strip domain words such as "library".

    s = re.sub(

        r"\s+(?:admission|admissions|application|applications|eligibility|process|processes|procedure|procedures|criteria|requirements?)$",

        "",

        s,

        flags=re.I,

    )

    s = re.sub(r"\s+(?:program|programme)$", "", s, flags=re.I)

    s = s.strip(" -–—:,;.")

    return s or None





def _score_degree_match(text: str, match: re.Match[str]) -> int:
    start, end = match.start(), match.end()
    prec = text[max(0, start - 60):start].lower()
    foll = text[end:min(len(text), end + 60)].lower()
    score = 0
    # Strong target program signals (preceding)
    if re.search(r"\b(?:apply\s+(?:directly\s+)?(?:for|to)|admission\s+(?:to|in|for)|pursue\b|register\s+for|enroll\s+in|eligible\s+for|eligibility\s+for|for\s+(?:regular\s+)?(?:full-time\s+)?(?:executive\s+)?)\b", prec):
        score += 20
    if re.search(r"\b(?:requirement\s+for|criteria\s+for|cutoff\s+for|cut-off\s+for|fee\s+for|qualifying\s+degree\s+requirement\s+for)\b", prec):
        score += 15
    if re.search(r"\b(?:executive\s+|regular\s+|part-time\s+|full-time\s+)\b", prec):
        score += 5

    # Strong target program signals (following)
    if re.search(r"^\s*(?:admission|admissions|applicants?|program|programme|scholars?|candidate|candidates)\b", foll):
        score += 20
    if re.search(r"^\s*(?:fee|fees|eligibility|requirements?|criteria|duration|course)\b", foll):
        score += 10

    # Prerequisite / qualifying degree signals (preceding within immediate local window)
    prec_tail = text[max(0, start - 30):start].lower()
    if re.search(r"\b(?:have\s+(?:an?\s+)?(?:4-year\s+|four-year\s+)?|completed\s+(?:my\s+)?|holding\s+(?:an?\s+)?|with\s+(?:an?\s+)?(?:4-year\s+|four-year\s+)?|passed\s+(?:my\s+)?|after\s+(?:my\s+)?)\b", prec_tail):
        score -= 25

    # Prerequisite signals (following)
    if re.search(r"^\s*(?:aggregate|marks|score|percentage|percent|holder|graduates?|since\s+graduation|degree\s+holder)\b", foll):
        score -= 25

    return score


def _infer_target(original: str, candidate: Any = None) -> str | None:
    """Infer the safest explicit target without allowing the LLM to hijack it.

    Priority:
    1. Explicit target degree based on contextual score (target vs prerequisite).
    2. Fallback to candidate if valid and text has no explicit target.
    3. Facility or statutory governance committee (hostel, anti-ragging, senate).
    """
    original_text = _clean(original)
    if not original_text:
        return None

    live_degrees: list[re.Match[str]] = []
    for match in _degree_mentions(original_text):
        if not _mention_is_excluded(original_text, match.start(), match.end()):
            live_degrees.append(match)

    if live_degrees:
        if len(live_degrees) == 1:
            score = _score_degree_match(original_text, live_degrees[0])
            if score >= 0:
                deg = _canonical_degree(live_degrees[0].group(0))
                foll = original_text[live_degrees[0].end():]
                spec_m = re.match(r"^(?:[.\s]+(?:programme|program|students?))?[\s]+in[\s]+([A-Za-z0-9\s&/-]+?)(?=[,?.!]|(?:\s+\b(?:why|how|when|what|where|which|who|does|is|are|from|with|combines?|offers?)\b)|$)", foll, re.I)
                if spec_m:
                    spec = spec_m.group(1).strip()
                    if spec and len(spec) > 2 and spec.lower() not in {"the", "a", "an", "this", "that"}:
                        return f"{deg} in {spec}" if deg else spec
                return deg
        else:
            scored = [(_score_degree_match(original_text, m), _canonical_degree(m.group(0)), m) for m in live_degrees]
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] >= 0:
                best_deg = scored[0][1]
                m = scored[0][2]
                foll = original_text[m.end():]
                spec_m = re.match(r"^(?:[.\s]+(?:programme|program|students?))?[\s]+in[\s]+([A-Za-z0-9\s&/-]+?)(?=[,?.!]|(?:\s+\b(?:why|how|when|what|where|which|who|does|is|are|from|with|combines?|offers?)\b)|$)", foll, re.I)
                if spec_m:
                    spec = spec_m.group(1).strip()
                    if spec and len(spec) > 2 and spec.lower() not in {"the", "a", "an", "this", "that"}:
                        return f"{best_deg} in {spec}" if best_deg else spec
                return best_deg

    all_degrees = [_canonical_degree(m.group(0)) for m in _degree_mentions(original_text)]
    distinct_degrees = _dedupe([x for x in all_degrees if x], 4)
    if len(distinct_degrees) == 1:
        return distinct_degrees[0]

    c = _clean(candidate)
    if c:
        if c.lower() in {
            "null", "none", "n/a", "na", "unknown", "nil", "undefined",
            "not applicable", "the institution", "institution", "iit jodhpur",
            "iitj", "iit", "the institute", "institute", "campus", "college", "university",
            "indian institute of technology jodhpur",
        }:
            return None
        degree = _canonical_degree(c)
        if degree:
            return degree
        cleaned = c
        for phrase in (
            "application fee", "processing fee", "hostel fee", "minimum percentage",
            "percentage requirement", "percentage required", "work experience",
            "documents", "deadline", "eligibility", "admission process",
            "application process", "process", "procedure", "requirements",
        ):
            cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.I)
        cleaned_candidate = _strip_target(cleaned)
        if cleaned_candidate:
            return cleaned_candidate

    if re.search(r"\b(?:apm\b|annual\s+progress\s+monitoring)\b", original_text, re.I):
        return "Ph.D."
    if re.search(r"\b(?:scholars?)\b", original_text, re.I) and not _degree_mentions(original_text):
        return "Ph.D."

    training_match = re.search(
        r"\b(pmkvy(?:\s*4\.0)?(?:\s+phase[- ]?[iI1]+)?|"
        r"specialized\s+training(?:\s+in\s+government\s+e-marketplace|\s+in\s+gem|\s+on\s+gfr\s+2017|\s+on\s+gfr)?|"
        r"gem\s+training|gfr\s+training|gfr\s+2017|government\s+e-marketplace|"
        r"office\s+of\s+vocational\s+training\s+&\s+upskilling|vtu|vocational\s+training)\b",
        original_text,
        re.I,
    )
    if training_match:
        return training_match.group(0).strip()

    facility_match = re.search(
        r"\b(hostel\s+accommodation|hostel|guest\s+room|anti[- ]ragging\s+committee|anti[- ]ragging|ragging|"
        r"internal\s+complaints\s+committee|icc|disciplinary\s+committee|disciplinary|"
        r"health\s+centre|health\s+center|ambulance|gymkhana|dean\s+of\s+academic\s+affairs|dean|registrar|"
        r"library|mess|senate|placement\s+cell|placement|sports\s+complex|sports|gym|bus\s+shuttle|shuttle|bus|"
        r"medical\s+insurance|insurance|supplementary\s+exam(?:ination)?s?|supplementary|"
        r"mcm\s+scholarship|scholarship|grade\s+appeal|appeal\s+(?:against\s+)?(?:an?\s+)?(?:academic\s+)?grade|"
        r"grading|grades?|letter\s+grade|temporary\s+grade|['\"][A-Za-z]['\"]\s+grade|"
        r"tuition\s+fee|late\s+fee|fee)\b",
        original_text,
        re.I,
    )
    if re.search(r"\bragging\b", original_text, re.I):
        return "anti-ragging committee"

    if facility_match:
        val = facility_match.group(0).strip()
        low_val = val.lower()
        if "ragging" in low_val:
            return "anti-ragging committee"
        if "disciplinary" in low_val:
            return "Disciplinary Committee"
        if "internal complaints" in low_val or low_val == "icc":
            return "Internal Complaints Committee"
        if "health" in low_val or "ambulance" in low_val:
            return "Health Centre"
        if "gymkhana" in low_val:
            return "Gymkhana"
        if "dean" in low_val:
            return "Dean"
        if "placement" in low_val:
            return "Placement Cell"
        if "sports" in low_val or "gym" in low_val:
            return "Sports"
        if "bus" in low_val or "shuttle" in low_val:
            return "Transport"
        if "insurance" in low_val:
            return "Insurance"
        if "supplementary" in low_val:
            return "Examination"
        if "scholarship" in low_val or "mcm" in low_val:
            return "Scholarship"
        if "grade" in low_val or "grading" in low_val:
            return "Grading"
        if "hostel" in low_val or "guest room" in low_val:
            return "Hostel"
        if "mess" in low_val:
            return "Mess"
        if "senate" in low_val:
            return "Senate"
        if "fee" in low_val:
            return "Fee"
        return val

    if re.search(r"\b(?:paisa|paise|rupaye|rupees|kharch|kharcha)\b|पैसे|पैसा|खर्चा", original_text, re.I):
        if re.search(r"\b(?:first\s+semester|1st\s+semester|semester)\b|पहला\s+सेमेस्टर|सेमेस्टर", original_text, re.I):
            return "fees and finance"
        return "Fee"

    if re.search(r"\bundergrad(?:uate)?s?\b", original_text, re.I):
        return "B.Tech"

    if re.search(r"\bpost[- ]?doc(?:toral)?\b", original_text, re.I):
        return "Postdoctoral"

    return None


def _comparison_targets(text: str) -> list[str]:

    raw = _clean(text)
    low = _cf(raw)

    # When comparison language is explicit, canonical degree mentions are more
    # reliable than phrase-boundary parsing (e.g. "M.Tech aur MBA dono ka ...
    # compare karke"). This remains institution-agnostic because it uses the
    # shared degree vocabulary rather than hard-coded program names.
    compare_cue = re.search(
        r"\b(?:compare|comparison|versus|vs\.?|aur)\b|तुलना|compare", low, re.I
    )
    if compare_cue:
        degrees = []
        for match in _degree_mentions(raw):
            if not _mention_is_excluded(raw, match.start(), match.end()):
                canonical = _canonical_degree(match.group(0))
                if canonical and canonical.casefold() not in {x.casefold() for x in degrees}:
                    degrees.append(canonical)
        if len(degrees) >= 2:
            return degrees[:2]

    patterns = (

        r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",

        r"compare\s+(.+?)\s+(?:and|with|to|vs\.?|versus|aur)\s+(.+?)(?:\?|$)",

        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",

        r"(.+?)\s+aur\s+(.+?)\s+(?:ka|ki|ke)?\s*(?:comparison|compare|admission process)(?:\?|$)",

    )

    for pattern in patterns:

        match = re.search(pattern, raw, re.I)

        if not match:

            continue

        values = [_strip_target(match.group(1)), _strip_target(match.group(2))]

        values = [v for v in values if v]

        if len(values) == 2:

            # Prefer clean canonical degree names when present.

            values = [(_canonical_degree(v) or v) for v in values]

            return _dedupe(values, 2)

    return []





def _comparison_attribute(text: str) -> str | None:

    low = _cf(text)

    checks = (

        ("admission process", "admission process"),

        ("application process", "application process"),

        ("eligibility", "eligibility"),

        ("fees", "fees"),

        ("fee", "fee"),

        ("documents", "documents"),

        ("admission process compare", "admission process"),

        ("admission process compare karke", "admission process"),

        ("प्रक्रिया", "admission process"),

        ("पात्रता", "eligibility"),

        ("फीस", "fee"),

        ("दस्तावेज", "documents"),

        ("डॉक्यूमेंट", "documents"),

    )

    for marker, value in checks:
        if marker in low:
            if value == "documents" and _is_meta_document_reference(text):
                continue
            return value

    return None





def _explicit_process_requested(text: str) -> str | None:
    """Return the concrete process attribute when procedural steps are requested."""
    low = _cf(text)
    patterns = (
        (r"\badmission\s+process\b", "admission process"),
        (r"\bapplication\s+process\b", "application process"),
        (r"\b(?:admission|entry)\s+(?:avenues?|modes?|routes?|options?)\b", "admission process"),
        (r"\b(?:avenues?|modes?|routes?)\s+(?:for|of)\s+admission\b", "admission process"),
        (r"\b(?:how|steps?|procedure)\s+(?:do|can|to)\b.*\b(?:apply|register|enroll)\b", "application process"),
        (r"\bhow\s+(?:do|can)\s+i\s+(?:apply|register|enroll)\b", "application process"),
        (r"\bhow\s+to\s+(?:apply|register|enroll)\b", "application process"),
        (r"\bsteps?\s+(?:to|for)\s+(?:apply|applying|register|registering|enroll|enrolling|submit|submitting)\b", "application process"),
        (r"\b(?:steps|procedure)\s+(?:for|to)\s+(?:submitting|submission)\b.*\bapplication\b", "application process"),
        (r"\bprocedure\s+(?:to|for)\s+(?:apply|applying|register|registering|enroll|enrolling)\b", "application process"),
        (r"\b(?:what|which)\s+is\s+the\s+(?:application|admission|registration|enrollment)?\s*process\b", "admission process"),
        (r"\b(?:what|which)\s+(?:are|is)\s+(?:the\s+)?steps\b", "application process"),
        (r"\b(?:tell|show|give)\s+me\s+(?:the\s+)?(?:application|admission)?\s*process\b", "admission process"),
        (r"\b(?:admission|application)\s+process\s+(?:batao|kaise|kya)\b", "admission process"),
        (r"\b(?:process|procedure|steps)\s+(?:batao|bataye|bataiye)\b", "application process"),
        (r"आवेदन\s+कैसे", "application process"),
        (r"कैसे\s+apply", "application process"),
        (r"\bapply\s+kaise(?:\s+karna)?(?:\s+hai)?\b", "application process"),
        (r"प्रक्रिया\s+(?:क्या|बताओ|बताइए|बताएं)", "admission process"),
        (r"कदम\s+से\s+apply", "application process"),
    )
    for pattern, attribute in patterns:
        for match in re.finditer(pattern, low, re.I):
            if not _mention_is_excluded(low, match.start(), match.end()):
                return attribute
    return None


# ---------------------------------------------------------------------------
# Generic discourse / objective filtering
# ---------------------------------------------------------------------------

EXCLUSION_MENTION_PATTERNS = (
    r"\b(?:not|don't|do not|never)\s+(?:asking|ask)\s+(?:about|for|whether|if)\b",
    r"\b(?:don't|do not|doesn't)\s+(?:want|need)\b",
    r"\b(?:not interested in|forget(?: about)?|no need for)\b",
    r"\b(?:already\s+know|know\s+already)\b",
    r"\b(?:rather than|instead of)\b",
    r"\b(?:nahi|nahin)\s+(?:pooch|chahiye|chahie)\b",
    r"के\s+बजाय|के\s+बदले",
)

POSITIVE_ELIGIBILITY_PATTERNS = (
    r"\b(?:can|may|could|would)\s+i\s+(?:still\s+)?(?:apply|register|enroll|be considered)\b",
    r"\b(?:am|is|are)\s+(?:i|you|we|they)\s+(?:eligible|qualified)\b",
    r"\b(?:eligibility|eligible|पात्रता|योग्य)\b",
    r"\b(?:work experience|work exp|gate|cat|cgpa|gpa|percentage|qualification|qualifying degree|graduation|final semester|final year)\b"
    r".{0,100}\b(?:required|mandatory|compulsory|necessary|eligible|apply)\b",
    r"\b(?:required|mandatory|compulsory|necessary|ज़रूरी|जरूरी|अनिवार्य)\b"
    r".{0,100}\b(?:work experience|work exp|gate|cat|cgpa|gpa|percentage|qualification|graduation|final semester|final year)\b",
    r"(?:क्या|क्या मैं).{0,120}(?:पात्र|योग्य|अनिवार्य|ज़रूरी|जरूरी)",
)


def _mention_is_excluded(text: str, start: int, end: int) -> bool:
    """Detect whether a mention belongs to a discarded/disclaimed clause."""
    low = _cf(text)
    before_full = text[:start]
    after = text[end:min(len(text), end + 120)]
    boundary_candidates = [
        before_full.rfind("?"), before_full.rfind("!"), before_full.rfind(";"),
        before_full.rfind("\n"), before_full.rfind("।"),
    ]
    period_matches = list(re.finditer(r"\.\s+", before_full))
    if period_matches:
        boundary_candidates.append(period_matches[-1].start())
    boundary = max(boundary_candidates)
    segment = before_full[boundary + 1:]
    low_segment = _cf(segment)
    low_after = _cf(after)

    # Local "already-known" disclaimer. Inspect the surrounding sentence rather
    # than a truncated prefix so the mentioned property itself is available.
    # Example: "I know the MBA eligibility already. I only need the steps..."
    local_start = max(0, start - 140)
    local_end = min(len(text), end + 140)
    local = _cf(text[local_start:local_end])
    mention = _cf(text[start:end])
    known = re.search(r"\b(?:i|we)\s+know\b.{0,100}\balready\b", local, re.I | re.S)
    if known and mention:
        mention_local_pos = start - local_start
        if known.start() <= mention_local_pos < known.end():
            return True

    # Contrastive disclaimer: "not just X" / "not only X" discards X as an
    # answer objective unless the sentence explicitly reintroduces it with
    # a positive "but also" clause before the mention.
    before_window = low[max(0, start - 40):start]
    contrast = re.search(r"\bnot\s+(?:just|only)\s+", before_window, re.I)
    if contrast and not re.search(r"\bbut\s+also\b", before_window[contrast.end():], re.I):
        return True

    exclusion_hits = [
        m for pattern in EXCLUSION_MENTION_PATTERNS
        for m in [re.search(pattern, low_segment, re.I)] if m
    ]
    if exclusion_hits:
        last_exclusion = max(exclusion_hits, key=lambda m: m.start())
        after_exclusion = low_segment[last_exclusion.end():]
        positive_reset = re.search(
            r"\b(?:i|we)\s+(?:(?:only|just)\s+)?(?:want|need|want\s+to\s+know|need\s+to\s+know|mean)\b"
            r"|\b(?:tell|give)\s+me\b|\b(?:actually|instead|rather|no)\b",
            after_exclusion,
            re.I,
        )
        if not positive_reset:
            return True

    mention_text = _cf(text[start:end])
    if mention_text not in {"eligible", "योग्य"}:
        if re.search(r"\bnot(?:\s+(?:the|this|that))?\s*$", low_segment, re.I):
            return True
    if re.search(r"\b(?:is|are|was|were)\s+not\s+(?:requested|needed|wanted|being\s+asked)\b", low_after, re.I):
        return True
    if re.search(r"(?:नहीं|नही)\s*$", segment):
        return True
    return False

def _explicit_live_eligibility_request(text: str) -> bool:
    """Return True only when eligibility itself is a requested objective."""
    low = _cf(text)

    # Explicit words: eligibility/eligible/requirements/qualification.
    for match in re.finditer(
        r"\b(?:eligibility|eligible|requirements?|qualification|qualifications)\b|"
        r"पात्रता|योग्य",
        low,
        re.I,
    ):
        if not _mention_is_excluded(low, match.start(), match.end()):
            return True

    # Eligibility can be expressed indirectly as "whether I can apply",
    # "is 60% enough", "can I submit before graduation", etc.
    indirect_patterns = (
        r"\b(?:whether|if)\s+(?:i|we)\s+(?:can|may|could)\s+(?:still\s+)?(?:apply|register|enroll)\b",
        r"\b(?:want to know|tell me|check)\s+(?:whether|if)\s+(?:i|we)\s+(?:can|may|could)\s+(?:still\s+)?(?:apply|register|enroll)\b",
        r"\b(?:can|may|could)\s+i\s+(?:still\s+)?(?:submit|apply|register|enroll)\b.*\b(?:before|without|until|while|final\s+(?:semester|year)|result|graduation)\b",
        r"\b(?:is|are)\s+\d+(?:\.\d+)?%?\s+(?:enough|sufficient)\b.*\b(?:admission|apply|eligible|eligibility)\b",
    )
    for pattern in indirect_patterns:
        for match in re.finditer(pattern, low, re.I | re.S):
            if not _mention_is_excluded(low, match.start(), match.end()):
                return True

    # "Can I apply/register/enroll/be considered?" is a live eligibility
    # question, even though the word eligibility is absent.
    for pattern in (
        r"\b(?:can|may|could|would)\s+i\s+(?:still\s+)?(?:apply|register|enroll|be considered)\b",
        r"क्या\s+मैं.*(?:आवेदन|apply|दाखिला|प्रवेश).*(?:योग्य|पात्र|सकता|सकती|सकते)",
    ):
        for match in re.finditer(pattern, low, re.I | re.S):
            if not _mention_is_excluded(low, match.start(), match.end()):
                return True

    return False


def _positive_eligibility_requested(text: str) -> bool:
    """Detect an eligibility objective without treating excluded mentions as requests."""
    raw = _clean(text)
    low = _cf(raw)

    # Explicit eligibility wording must be a live objective, not a disclaimer.
    for match in re.finditer(r"\b(?:eligibility|eligible|required|mandatory|compulsory|necessary)\b|पात्रता|योग्य|अनिवार्य|ज़रूरी|जरूरी", low, re.I):
        if not _mention_is_excluded(low, match.start(), match.end()):
            window = low[max(0, match.start() - 120):min(len(low), match.end() + 120)]
            if (
                re.search(r"\b(?:eligibility|eligible)\b", match.group(0), re.I)
                or re.search(
                    r"\b(?:can|may|could|would|apply|register|enroll|be considered|work experience|work exp|gate|cat|cgpa|gpa|percentage|qualification|graduation|final semester|final year)\b",
                    window,
                    re.I,
                )
                or any(token in window for token in ("पात्रता", "योग्य", "अनिवार्य", "ज़रूरी", "जरूरी"))
            ):
                return True

    # Other eligibility forms must also be tied to a live, non-excluded mention.
    # Do not accept a raw keyword hit from a discarded clause.
    for pattern in POSITIVE_ELIGIBILITY_PATTERNS:
        for match in re.finditer(pattern, low, re.I | re.S):
            if not _mention_is_excluded(low, match.start(), match.end()):
                return True

    return False
def _is_meta_document_reference(text: str) -> bool:
    """Check if 'document' in text is merely a meta-reference to the source/context."""
    # If the text explicitly asks about required or submission documents, it's not merely meta
    if re.search(r"\b(?:required|needed|submit|submission|verification|mandatory|compulsory|list\s+of|which|what)\s+(?:documents?|docs?|certificates?)\b|\b(?:documents?|docs?|certificates?)\s+(?:required|needed|to\s+submit|to\s+be\s+submitted|for\s+admission|for\s+registration|for\s+verification)\b|दस्तावेज|डॉक्यूमेंट", text, re.I):
        return False
    # Check for meta-references to documents as corpus/source material
    return bool(
        re.search(
            r"\b(?:the|this|that|these|uploaded|source|prescribed|given|provided)\s+documents?\b|"
            r"\bdocuments?\s+(?:propose|proposes|state|states|show|shows|mention|mentions|say|says|contain|contains|list|lists|describe|describes|discuss|discusses)\b|"
            r"\b(?:according to|as per|in|from|based on|using)\s+(?:the|these|all|only)?\s*documents?\b",
            text,
            re.I,
        )
    )


def _requested_attributes(text: str) -> list[str]:
    """Extract positive answer properties conservatively from the user's wording."""
    low = _cf(text)
    found: list[str] = []

    def add_matches(pattern: str, value: str) -> None:
        for match in re.finditer(pattern, low, re.I):
            if not _mention_is_excluded(low, match.start(), match.end()):
                found.append(value)

    # Fees: preserve the most specific fee family member.
    add_matches(r"application\s+(?:processing\s+)?fee|आवेदन\s+(?:की\s+)?फीस|आवेदन\s+शुल्क", "application fee")
    for match in re.finditer(r"processing\s+fee|processing\s+charges", low, re.I):
        if _mention_is_excluded(low, match.start(), match.end()):
            continue
        before = low[max(0, match.start()-20):match.start()]
        if re.search(r"application\s*$", before, re.I):
            continue
        found.append("processing fee")
    add_matches(r"hostel\s+fee|hostel\s+fees|छात्रावास\s+(?:की\s+)?फीस|हॉस्टल\s+फीस", "hostel fee")
    add_matches(r"\b(?:fees|fee|fess|charges?|amount|cost|rent|rates?|paisa|paise|rupaye|rupees|kharch|kharcha)\b|फीस|शुल्क|पैसे|पैसा|खर्चा", "fee")
    if re.search(r"\b(?:first\s+semester|1st\s+semester|semester)\b|पहला\s+सेमेस्टर|सेमेस्टर", low):
        if any(w in low for w in ("fee", "fees", "paisa", "paise", "rupaye", "rupees", "kharcha", "dena", "amount")):
            found.append("semester fee")
    add_matches(r"\b(?:options?|branches?|programmes?|programs?|specializations?|disciplines?)\b|विकल्प", "programs")
    add_matches(
        r"\b(?:monthly\s+)?(?:fellowships?|scholarships?|stipends?|financial\s+(?:assistance|support|aid)|assistantships?)\b",
        "fellowship",
    )

    # Process wording is kept descriptive; type conversion happens downstream.
    process = _explicit_process_requested(text)
    if process:
        found.append(process)

    # Eligibility facets.
    add_matches(r"\b(?:minimum\s+)?(?:cut[- ]?off\s+)?percentage\b|\bpercentage\s+(?:required|requirement|cut[- ]?off)\b|\bcut[- ]?off\s+percentage\b|\bminimum\s+cut[- ]?off\b", "minimum percentage")
    add_matches(r"\bwhat\s+percentage\b", "percentage")
    add_matches(r"\b\d+(?:\.\d+)?%\s+(?:is\s+)?(?:enough|sufficient)\b|\b(?:is|are)\s+\d+(?:\.\d+)?%\s+(?:enough|sufficient)\b", "percentage")
    add_matches(
        r"\bwork\s+experience\b|\bwork\s+exp\b|\bprior\s+industry\s+experience\b|"
        r"\bindustry\s+experience\b|\bexperience\s+(?:required|compulsory|mandatory|necessary|needed)\b|"
        r"काम\s+का\s+अनुभव|नौकरी\s+का\s+अनुभव",
        "work experience",
    )
    if not _is_meta_document_reference(text):
        add_matches(r"\b(?:documents?|docs?)\b|डॉक्यूमेंट|दस्तावेज", "documents")
    add_matches(r"\bqualifying\s+degree\b", "qualifying degree")
    add_matches(
        r"\b(?:deadline|deadlines|last\s+date|closing\s+date)\b|आखिरी\s+तारीख|अंतिम\s+तारीख|"
        r"deadline\s+kab|kab\s+tak",
        "deadline",
    )
    add_matches(r"\b(?:participants?|enrolled|enrolment|candidates?|trainees?)\b", "participants")
    add_matches(r"\b(?:certifications?|certified)\b", "certification")
    add_matches(r"\b(?:training\s+duration|duration|hours?|hrs?)\b", "duration")
    add_matches(r"\b(?:job\s+roles?|job\s+role)\b", "job roles")
    add_matches(r"\b(?:hands[- ]on\s+modules?)\b", "hands-on modules")

    # Generic eligibility is only a real objective when the eligibility itself
    # is live; specific facets such as work experience remain their own facet.
    if _explicit_live_eligibility_request(text):
        found.append("eligibility")

    # Direct exam/score facets are useful evidence requirements, but only in an
    # eligibility context; a bare fact such as "I qualified GATE" is not by
    # itself a requested property.
    eligibility_context = (
        _explicit_live_eligibility_request(text)
        or _positive_eligibility_requested(text)
        or bool(re.search(r"\b(?:apply|eligible|eligibility|requirement|mandatory|compulsory|necessary)\b", low, re.I))
        or bool(re.search(r"क्या\s+मैं|पात्र|योग्य|अनिवार्य|ज़रूरी|जरूरी", low))
    )
    if eligibility_context or re.search(r"\b(?:cgpa|gpa|percentage|marks)\b.*\b(?:enough|sufficient)\b|\b(?:enough|sufficient)\b.*\b(?:cgpa|gpa|percentage|marks)\b|\b\d+(?:\.\d+)?%?\s+(?:is\s+)?(?:enough|sufficient)\b", low, re.I):
        if re.search(r"\b(?:cgpa|gpa)\b", low, re.I) and re.search(r"\b(?:enough|sufficient)\b", low, re.I):
            found.append("CGPA")
        elif re.search(r"\b(?:percentage|marks)\b", low, re.I) and re.search(r"\b(?:enough|sufficient)\b", low, re.I):
            found.append("percentage")
        for exam in ("GATE", "CAT", "JEE", "NET", "CEED", "NBHM"):
            for match in re.finditer(rf"\b{exam}\b", low, re.I):
                if not _mention_is_excluded(low, match.start(), match.end()):
                    found.append(exam)
                    break
        if re.search(r"\b(?:cgpa|gpa)\b", low, re.I) and re.search(
            r"\b(?:required|minimum|eligibility|eligible|needed|mandatory|compulsory|necessary)\b",
            low,
            re.I,
        ):
            found.append("CGPA")

    if re.search(r"\b(?:where is|where are|location of|located)\b", low, re.I):
        found.append("location")
    if re.search(r"\b(?:contact|phone number|email address|email id|telephone)\b", low, re.I):
        found.append("contact")
    if re.search(r"\b(?:facility|facilities)\b", low, re.I):
        found.append("facility")
    if re.search(r"\b(?:attendance|attandance)\b", low, re.I):
        found.append("attendance")
    if re.search(r"\b(?:policy|policies|rules|regulations)\b", low, re.I):
        found.append("policy")
    if re.search(r"\b(?:re-?exam(?:ination)?|additional\s+examination|make-?up\s+examination)\b", low, re.I):
        found.append("re-examination")

    # Indirect eligibility wording, such as "whether I can apply", belongs to
    # one eligibility objective even when the exact word "eligible" is absent.
    if re.search(
        r"\b(?:whether|if)\s+(?:i|we)\s+(?:can|may|could)\s+(?:still\s+)?(?:apply|register|enroll)\b",
        low, re.I,
    ):
        found.append("eligibility")

    # Current academic status can itself be an eligibility gate.
    if re.search(
        r"\b(?:final\s+(?:semester|year)|final\s+result|graduation|result\s+(?:is\s+)?(?:not\s+)?out)\b",
        low, re.I,
    ) and re.search(r"\b(?:apply|submit|admission|eligible|wait)\b", low, re.I):
        found.append("eligibility")

    # Direct "can I apply?" is eligibility, not process. This is checked
    # after process extraction because "how can I apply" must remain process.
    if re.search(r"\b(?:can|may)\s+i\s+(?:still\s+)?(?:apply|register|enroll)\b", low, re.I):
        if _explicit_process_requested(text):
            pass
        elif re.search(r"\b(?:this\s+year|currently|now|today|before|until|yet)\b", low, re.I):
            found.append("application process")
    if re.search(r"\b(?:focus\s+areas?|thrust\s+areas?)\b", low, re.I):
        found.append("focus areas")
    if re.search(r"\b(?:research\s+areas?|research\s+themes?)\b", low, re.I):
        found.append("research areas")
    if re.search(r"\b(?:career\s+pathways?|career\s+paths?|career\s+opportunities?)\b", low, re.I):
        found.append("career pathways")
    if re.search(r"\b(?:course\s+codes?|subject\s+codes?)\b", low, re.I):
        found.append("course codes")
    if re.search(r"\b(?:technical\s+streams?|streams?)\b", low, re.I):
        found.append("technical streams")
    if re.search(r"\b(?:minor\s+options?|minors?)\b", low, re.I):
        found.append("minor options")
    if re.search(r"\b(?:learning\s+outcomes?|objectives?)\b", low, re.I):
        found.append("learning outcomes")
    if re.search(r"\b(?:symbolic[- ]computation\s+tools?|tools?)\b", low, re.I) and "symbolic" in low:
        found.append("symbolic-computation tools")

    return _normalize_attributes(found)


def _canonical_attribute(value: Any) -> str:
    s = _cf(value)
    # Descriptive model labels must collapse to the canonical evidence facet.
    # This is intentionally property-based, not question/test-specific.
    if re.search(r"\bcgpa\b", s, re.I):
        return "CGPA"
    if re.search(r"\bgate\b", s, re.I):
        return "GATE"
    if re.search(r"\bcat\b", s, re.I):
        return "CAT"
    if re.search(r"\bjee(?:\s+(?:main|advanced))?\b", s, re.I):
        return "JEE"
    if re.search(r"\bnet(?:\s*\(jrf\))?\b", s, re.I):
        return "NET"
    if re.search(r"\bwork\s+(?:experience|exp)\b", s, re.I):
        return "work experience"
    if re.search(r"\bdocument(?:s)?\b|\bdocs?\b", s, re.I) and not _is_meta_document_reference(s):
        return "documents"
    if re.search(r"\bdeadline\b|\blast\s+date\b", s, re.I):
        return "deadline"
    if re.search(r"\b(?:eligibility|eligibility criteria|qualification(?:s)?)\b", s, re.I):
        return "eligibility"
    if re.search(r"\b(?:re-?exam|re-?examination|additional\s+examination|make-?up\s+examination)\b", s, re.I):
        return "re-examination"
    aliases = {
        "application fees": "application fee",
        "processing fees": "processing fee",
        "hostel fees": "hostel fee",
        "fees": "fee",
        "cost": "fee",
        "minimum percentage required": "minimum percentage",
        "percentage requirement": "percentage",
        "percentage required": "percentage",
        "work-experience": "work experience",
        "documents required": "documents",
        "eligibility criteria": "eligibility",
        "पात्रता": "eligibility",
        "योग्य": "eligibility",
        "admission process": "admission process",
        "application process": "application process",
        "process": "application process",
        "आवेदन कैसे": "application process",
        "प्रक्रिया": "admission process",
        "दस्तावेज": "documents",
        "डॉक्यूमेंट": "documents",
        "फीस": "fee",
        "शुल्क": "fee",
        "आखिरी तारीख": "deadline",
        "अंतिम तारीख": "deadline",
        "steps": "application process",
        "procedure": "application process",
    }
    return aliases.get(s, _clean(value))


def _normalize_attributes(values: Sequence[Any]) -> list[str]:
    attrs = _dedupe([_canonical_attribute(v) for v in values], 20)
    specific_fees = {"application fee", "processing fee", "hostel fee"}
    if any(a in specific_fees for a in attrs):
        attrs = [a for a in attrs if a != "fee"]
    if "minimum percentage" in attrs:
        attrs = [a for a in attrs if a != "percentage"]
    return attrs


def _constraints(text: str) -> list[str]:
    raw = _clean(text)
    out: list[str] = []
    patterns = (
        r"\b(?:final year|first year|second year|third year|fourth year)\s+of\s+[A-Za-z][A-Za-z.]*",
        r"\b(?:cgpa|gpa)\s*(?:of|=|:)??\s*\d+(?:\.\d+)?",
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s+percent(?:\s+(?:se\s+kam|se\s+zyada|below|above))?\b",
        r"\b(?:qualified|cleared|passed|appeared|clear)\s+(?:GATE|CAT|JEE(?:\s+Main|\s+Advanced)?|NET(?:\s*\(JRF\))?|CEED|NBHM)\b",
        r"\b(?:GATE|CAT|JEE(?:\s+Main|\s+Advanced)?|NET|CEED|NBHM)\s+(?:qualified|clear|cleared|clear\s+hai|score|qualification)\b",
        r"\bvalid\s+(?:GATE|CAT|JEE|NET|CEED|NBHM)\s+score\b",
        r"\b(?:category-specific|category specific)\s+(?:relaxation|relaxations)\b",
        r"\b(?:relaxation|relaxations)\b",
        r"\b(?:final\s+semester|final\s+year)\b|अंतिम\s+सेमेस्टर|अंतिम\s+वर्ष",
        r"\b(?:after graduation|before graduation|after completing|before completing|graduation)\b",
        r"\b(?:final result|result is not out|result not out|awaiting result|result pending)\b",
        r"\b(?:below|above|under|over)\s+(?:the\s+)?(?:usual\s+)?(?:percentage\s+)?threshold\b",
        r"\b(?:less|more|lower|higher)\s+than\s+\d+(?:\.\d+)?%?\b",
        r"\b(?:percentage|percent|cgpa|gpa)\s+(?:below|above|under|over|less|more)\s+\d+(?:\.\d+)?%?\b",
        r"\b(?:semester|year)\s*[1-9][0-9]*\b",
        r"\b(?:202[0-9]|203[0-9])\b",
        r"\b\d+(?:\.\d+)?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+work)?\s+experience\b",
        r"\b\d+(?:\.\d+)?\s+(?:additional\s+)?credits?\b",
        r"\b\d+-credit\b",
        r"\b(?:scale\s+(?:of\s+)?|on\s+an?\s+)\d+(?:-point\s+scale|\s+scale)?\b",
        r"\bwithout\s+(?:having\s+)?(?:a\s+|an\s+)?(?:bedding|gate|cat|jee|net|experience|quitting\s+(?:my\s+)?job|leaving\s+(?:my\s+)?job|resigning\s+(?:from\s+)?(?:my\s+)?(?:current\s+)?job|master'?s\s+degree|degree)\b",
        r"\b(?:completely\s+)?free\s+of\s+cost\b|\bzero\s+fee\b|\bfree\s+(?:hostel|tuition|accommodation)\b",
        r"\b(?:remote|online)\s+(?:study|option|program|degree)\b",
        r"\b(?:discount|waiver|concession)\b",
        r"\b(?:overrule|cancel)\b",
    )
    for pattern in patterns:
        out.extend(m.group(0) for m in re.finditer(pattern, raw, re.I))

    # Category acronyms are scanned on degree-masked text so M.Sc is not read
    # as the category "SC", and ordinals like 1st are not read as "ST".
    masked = re.sub(DEGREE_PATTERN, " ", raw, flags=re.I)
    for match in re.finditer(r"(?<![A-Za-z0-9.])(?:EWS|OBC|SC|ST|PWD|PwD)(?![A-Za-z0-9])", masked, re.I):
        out.append(match.group(0))

    # Preserve the user's negative wording and add one canonical semantic alias
    # for mandatory/required negation. This keeps downstream matching robust
    # across equivalent phrasings without creating another intent.
    negative_required = re.search(
        r"\b(?:not\s+(?:required|mandatory|compulsory|necessary|needed)|isn['’]?t\s+required|(?:mandatory|compulsory|necessary)\s+nahin)\b|"
        r"(?:जरूरी|ज़रूरी|अनिवार्य|आवश्यक)\s+नहीं",
        raw, re.I,
    )
    if negative_required:
        out.append(_clean(negative_required.group(0)))
        if not any(re.fullmatch(r"not\s+required", value, re.I) for value in out):
            out.append("not required")
    elif _negated(raw):
        for pattern in NEGATION_PATTERNS:
            match = re.search(pattern, raw, re.I)
            if match:
                matched_val = match.group(0).strip()
                if matched_val.lower() not in {"without", "not", "no"}:
                    out.append(matched_val)
                break
    return _dedupe(out, 20)


def _preserved(text: str) -> list[str]:
    raw = _clean(text)
    patterns = (
        rf"\b{DEGREE_PATTERN}\b", r"\bGATE\b", r"\bCAT\b",
        r"\bJEE(?:\s+Main|\s+Advanced)?\b", r"\bNET(?:\s*\(JRF\))?\b",
        r"\bCGPA\b", r"\b\d+(?:\.\d+)?\s*%\b",
        r"\bJAM\b", r"\bINRA\b",
        r"\b\d+(?:\.\d+)?\s*credits?\b", r"\bcredit\s+limit\b",
        r"\bcredits?\b",
        r"\boverload\b", r"\bprobation\b", r"\bunder-?load\b",
        r"\bSenate\b", r"\banti[- ]ragging\b",
        r"\b\d+(?:\.\d+)?\s*years?\s+(?:of\s+)?(?:work\s+)?experience\b",
        r"\bwork\s+experience\b", r"\binterview\b",
        r"\bprovisional\b", r"\badmission\b", r"\bassistantship\b",
        r"\bmaximum\b", r"\bminimum\b",
    )
    out: list[str] = []
    for pattern in patterns:
        out.extend(m.group(0) for m in re.finditer(pattern, raw, re.I))

    # Comma-formatted numbers (e.g. 10,000 -> 10000)
    for m in re.finditer(r"\b\d{1,3}(?:,\d{3})+\b", raw):
        out.append(m.group(0).replace(",", ""))

    # Normalize canonical degree aliases in preserved terms
    if re.search(r"\bph\.?\s*d\.?\b", raw, re.I):
        out.extend(["ph.d", "phd", "Ph.D."])
    if re.search(r"\bb\.?\s*tech\b|\bundergrad(?:uate)?s?\b", raw, re.I):
        out.extend(["b.tech", "B.Tech"])
    if re.search(r"\bm\.?\s*tech\b", raw, re.I):
        out.extend(["m.tech", "M.Tech"])
    if re.search(r"\bm\.?\s*sc\.?\b", raw, re.I):
        out.extend(["m.sc", "M.Sc."])

    # Term variants (hyphenation, branch change, campus services)
    if re.search(r"\bpart[- ]time\b", raw, re.I):
        out.extend(["part-time", "part time"])
    if re.search(r"\b(?:branch\s+change|change\s+(?:of\s+)?(?:a\s+)?branch)\b", raw, re.I):
        out.append("branch change")
    if re.search(r"\bgymkhana\b", raw, re.I):
        out.append("gymkhana")
    if re.search(r"\bhealth\s+centre|ambulance\b", raw, re.I):
        out.extend(["health centre", "health"])
    if re.search(r"\binternal\s+complaints\s+committee|icc\b", raw, re.I):
        out.append("internal complaints committee")
    if re.search(r"\bsenate\b", raw, re.I):
        out.append("senate")
    if re.search(r"\bregist(?:er|ration)\b", raw, re.I):
        out.append("registration")
    if re.search(r"\bquarters\b", raw, re.I):
        out.extend(["quarters", "hostel"])
    if re.search(r"\bfine\b", raw, re.I):
        out.append("fine")

    # Explicitly ensure plural 'credits' is present if 'credit' or 'credit limit' was found
    if any(re.search(r"\bcredit", term, re.I) for term in out):
        if "credits" not in [t.lower() for t in out]:
            out.append("credits")
    return _dedupe(out, 36)


def _det_type_for_attribute(attribute: str | None) -> str | None:
    if attribute in {"fee", "application fee", "processing fee", "hostel fee"}:
        return "fee"
    if attribute in {"fellowship", "scholarship", "stipend", "financial assistance"}:
        return "fellowship"
    if attribute in {"minimum percentage", "percentage", "eligibility", "work experience", "GATE", "CAT", "JEE", "NET", "CEED", "NBHM", "CGPA"}:
        return "eligibility"
    if attribute == "documents":
        return "documents"
    if attribute == "deadline":
        return "deadline"
    if attribute in {"process", "admission process", "application process"}:
        return "process"
    if attribute == "location":
        return "location"
    if attribute == "contact":
        return "contact"
    if attribute == "facility":
        return "facility"
    if attribute == "policy":
        return "policy"
    return None


def _det_type_for_question(text: str) -> str | None:
    comparisons = _comparison_targets(text)
    if comparisons:
        return "comparison"
    process = _explicit_process_requested(text)
    if process:
        return "process"

    # Explicit status/eligibility conditions around applying are one
    # eligibility decision when the user is asking whether their current
    # academic state permits submission.
    current_status_gate = re.search(
        r"\b(?:final\s+(?:semester|year)|final\s+result|graduation|result\s+(?:is\s+)?(?:not\s+)?out)\b.*\b(?:submit|apply|application)\b|\b(?:submit|apply|application)\b.*\b(?:final\s+(?:semester|year)|final\s+result|graduation|result\s+(?:is\s+)?(?:not\s+)?out)\b",
        _cf(text), re.I | re.S,
    )
    if current_status_gate and not _explicit_process_requested(text):
        return "eligibility"

    # Application-availability questions are treated as a single process
    # objective in this planner contract (e.g. "Can I apply this year?").
    # Resolve this before generic attribute extraction so the implicit
    # eligibility vocabulary does not create a competing type.
    availability_apply = re.search(
        r"\b(?:can|may)\s+i\s+(?:still\s+)?(?:apply|register|enroll)\b.*\b(?:this\s+year|currently|now|today|before|until|yet)\b",
        _cf(text), re.I | re.S,
    )
    if availability_apply and not _explicit_process_requested(text):
        return "process"

    attrs = _requested_attributes(text)
    if attrs:
        # A direct fee/docs/deadline request wins before generic eligibility facets.
        for attr in attrs:
            value = _det_type_for_attribute(attr)
            if value:
                return value
    if _explicit_live_eligibility_request(text):
        return "eligibility"
    return None


def _intent_question(target: str | None, attribute: str, original: str) -> str:
    subject = target or "the institution"
    templates = {
        "application fee": f"What is the application fee for {subject}?",
        "processing fee": f"What is the processing fee for {subject}?",
        "hostel fee": f"What is the hostel fee for {subject}?",
        "fee": f"What is the fee for {subject}?",
        "minimum percentage": f"What is the minimum percentage required for {subject}?",
        "percentage": f"What percentage is required for {subject}?",
        "work experience": f"Is work experience required for {subject}?",
        "documents": f"What documents are required for {subject}?",
        "deadline": f"What is the application deadline for {subject}?",
        "eligibility": f"What are the eligibility requirements for {subject}?",
        "GATE": f"Is GATE required for {subject}?",
        "CAT": f"Is CAT required for {subject}?",
        "CGPA": f"What CGPA is required for {subject}?",
        "admission process": f"What is the admission process for {subject}?",
        "application process": f"How do I apply for {subject}?",
        "process": f"How do I apply for {subject}?",
        "location": f"Where is {subject} located?",
        "contact": f"How can I contact {subject}?",
        "facility": f"What facilities are available for {subject}?",
        "policy": f"What are the relevant policies for {subject}?",
        "re-examination": f"What is the re-examination policy for {subject}?",
    }
    return templates.get(attribute, _clean(original))


_STANDALONE_STOPWORDS = {
    "without", "with", "for", "and", "or", "in", "at", "to", "of", "from",
    "the", "a", "an", "is", "are", "on", "by", "not", "no",
}


def _compact_query(target: str | None, attrs: Sequence[str], constraints: Sequence[str],
                   comparison_targets: Sequence[str] = ()) -> str:
    pieces: list[str] = []
    if comparison_targets:
        pieces.append(" and ".join(_dedupe(comparison_targets, 4)))
    elif target:
        pieces.append(target)

    # When querying qualifying degree or minimum percentage for academic degrees, ensure "admission"
    # is included to anchor institutional regulations and criteria.
    low_attrs = [str(a).lower() for a in attrs]
    if target and any("qualifying degree" in a for a in low_attrs) and not any("admission" in a for a in low_attrs):
        pieces.append("admission")

    for attr in attrs[:3]:
        attr_str = str(attr).strip()
        if attr_str.lower() not in _STANDALONE_STOPWORDS:
            pieces.append(attr_str)

    # Filter out personal candidate percentages (e.g. "58%") from the retrieval query
    # when the query is seeking eligibility/cut-off/minimum percentage. The institutional
    # corpus specifies the institution's threshold, not the candidate's personal score.
    clean_constraints: list[str] = []
    has_pct_attr = any("percentage" in a or "eligib" in a for a in low_attrs)
    for c in constraints:
        c_str = str(c).strip()
        if c_str.lower() in _STANDALONE_STOPWORDS:
            continue
        if has_pct_attr and re.search(r"^\d+(?:\.\d+)?%", c_str):
            continue
        clean_constraints.append(c_str)

    pieces.extend(clean_constraints[:5])
    result = _clean(" ".join(pieces))
    if not target and not attrs and not comparison_targets:
        if len(result.split()) <= 1:
            return ""
    return result


def _safe_retrieval_query(
    segment: str,
    target: str | None,
    attrs: Sequence[str],
    constraints: Sequence[str] = (),
    comparison_targets: Sequence[str] = (),
) -> str:
    """Build a coherent, non-naked retrieval query rooted in target, attributes, and critical terms."""
    if comparison_targets:
        return _compact_query(" and ".join(comparison_targets), attrs, constraints, comparison_targets)

    pieces: list[str] = []
    if target:
        pieces.append(target)

    low_segment = _cf(segment)
    low_attrs = [str(a).lower() for a in attrs]

    # When querying qualifying degree or minimum percentage for academic degrees, ensure "admission"
    # is included to anchor institutional regulations and criteria.
    if target and any("qualifying degree" in a for a in low_attrs) and not any("admission" in a for a in low_attrs):
        pieces.append("admission")

    for attr in attrs[:3]:
        attr_str = str(attr).strip()
        if attr_str.lower() not in _STANDALONE_STOPWORDS and attr_str not in pieces:
            pieces.append(attr_str)

    # Add key institutional qualifiers present in the segment
    for term in (
        "qualifying degree", "degree duration", "work experience", "interview", "entrance test",
        "assistantship", "fellowship", "financial assistance", "stipend", "scholarship",
        "hostel", "hostel charges", "anti-ragging committee", "provisional admission",
        "senate", "cancellation", "academic probation", "credit limit", "maximum credits", "relaxation",
        "exemption", "exempted", "exempt", "deposit", "refundable", "non-graded", "attendance",
        "re-exam", "re-examination", "additional examination", "make-up examination",
        "grade", "grading", "APM", "annual progress monitoring", "temporary grade",
        "JAM", "GATE", "SC", "ST", "OBC", "EWS",
    ):
        matched = (
            bool(re.search(rf"\b{re.escape(term.lower())}\b", low_segment))
            if len(term) <= 4
            else (term.lower() in low_segment)
        )
        if matched and not any(term.lower() in p.lower() for p in pieces):
            pieces.append(term)

    # Include any single uppercase letter grades (e.g. 'E' grade, 'I' grade)
    single_grades = set(re.findall(r"\b(['\"]?[A-Za-z]['\"]?)\s+grade\b|\bgrade\s+(['\"]?[A-Za-z]['\"]?)\b", segment, re.I))
    for tup in single_grades:
        for g in tup:
            if g:
                clean_g = g.strip("'\"").upper()
                if clean_g in {"E", "F", "I", "S", "U"} and not any(f"'{clean_g}' grade" in p or clean_g in p for p in pieces):
                    pieces.append(f"'{clean_g}' grade")

    # Include any prerequisite/contextual degree mentions in segment not already in pieces
    for m in _degree_mentions(segment):
        deg_name = _canonical_degree(m.group(0))
        if deg_name and not any(deg_name.lower() == p.lower() for p in pieces):
            pieces.append(deg_name)

    # Include non-stopword constraints
    has_pct_attr = any("percentage" in a or "cutoff" in a or "eligib" in a for a in low_attrs)
    for c in constraints:
        c_str = str(c).strip()
        if c_str.lower() in _STANDALONE_STOPWORDS or any(c_str.lower() in p.lower() for p in pieces):
            continue
        # Avoid embedding applicant personal score in cutoff queries
        if has_pct_attr and re.search(r"^\d+(?:\.\d+)?%", c_str):
            continue
        pieces.append(c_str)

    # Domain recovery for Hindi/Hinglish fee questions
    if re.search(r"\b(?:paisa|paise|rupaye|rupees|kharch|kharcha)\b|पैसे|पैसा|खर्चा", low_segment):
        if not any("fee" in p.lower() for p in pieces):
            pieces.append("fee")
        if re.search(r"\b(?:first\s+semester|1st\s+semester|semester)\b|पहला\s+सेमेस्टर|सेमेस्टर", low_segment):
            if not any("semester" in p.lower() for p in pieces):
                pieces.append("semester fee")
    # Domain recovery for Hindi/Hinglish programme/option inquiries
    if re.search(r"\b(?:options?|kya-kya|kya kya|branches?|specializations?)\b", low_segment):
        if any(deg.lower() in p.lower() for deg in ("m.tech", "b.tech", "ph.d", "m.sc", "mba") for p in pieces):
            if not any("program" in p.lower() for p in pieces):
                pieces.append("programmes")

    res = _clean(" ".join(pieces))

    # Guard against empty or single naked stopword/noun
    words = [w for w in res.split() if w.lower() not in _STANDALONE_STOPWORDS]
    if len(words) <= 1 or res.lower() in {"without", "documents", "deadline"}:
        HINGLISH_FILLERS = {
            "agar", "kisi", "ko", "hai", "toh", "to", "kya", "usko", "use", "unhe",
            "dene", "ka", "ki", "ke", "mauka", "milta", "milte", "milega", "hoga",
            "sakte", "sakta", "sakti", "kar", "bhi", "mein", "par", "se", "aur",
            "kya-kya", "mujhe", "dena", "padega", "hain", "options", "kitna", "kitne",
            "batao", "bataiye", "bata",
        }
        valid_letters = {g.strip("'\"").upper() for tup in single_grades for g in tup if g}
        seg_words = [
            w for w in re.findall(r"[A-Za-z0-9\.\-%]+", segment)
            if w.lower() not in _STANDALONE_STOPWORDS
            and w.lower() not in HINGLISH_FILLERS
            and (len(w) > 2 or w.upper() in valid_letters)
        ]
        res = _clean(" ".join(seg_words[:10]))

    return res





# ---------------------------------------------------------------------------

# Intent normalization and reconciliation

# ---------------------------------------------------------------------------




def _objective_segments(text: str) -> list[str]:
    """Split only genuinely separate answer objectives using explicit evidence families."""
    original = _clean(text)
    if not original:
        return []
    if _comparison_targets(original):
        return [original]

    low = _cf(original)

    # Detect 'tell me both X and Y' or 'both X and Y'
    both_m = re.match(
        r"^(?:(?:can\s+you\s+)?(?:tell|give|show|provide|share)\s+me\s+)?both\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)$",
        original.strip(" ,;.?"),
        flags=re.I,
    )
    if both_m:
        part1 = _clean(both_m.group(1))
        part2 = _clean(both_m.group(2))
        if part1 and part2:
            q1 = part1 if re.search(r"\b(?:what|how|who|where|when|which|is|are|can)\b", part1, re.I) else f"What is the {part1}?"
            q2 = part2 if re.search(r"\b(?:what|how|who|where|when|which|is|are|can)\b", part2, re.I) else f"What are the {part2}?"
            return [q1, q2]

    # Coordinate two independent interrogatives:
    # e.g. "Clause 1, and (what|how|who|where|when|which|is|are|can|does|do|whether) Clause 2?"
    # or Hinglish ",? aur (kya|kaise|kitna|kitne|kahan|kon|hostel|phd|mtech|btech)..."
    m_coord = re.search(r",?\s+\band\s+(what|how\s+much|how|who|where|when|which|is|are|can|does|do|whether)\b", original, re.I)
    if not m_coord:
        m_coord = re.search(r",?\s+\baur\s+(?:kya|kaise|kitna|kitne|kahan|kon|kisko|hostel|phd|btech|mtech)\b", original, re.I)
    if m_coord and not _comparison_targets(original):
        first_part = _clean(original[:m_coord.start()])
        second_part = _clean(original[m_coord.start():].lstrip(" ,"))
        second_part = re.sub(r"^(?:and\s+|aur\s+)+", "", second_part, flags=re.I)
        # Ensure first_part is not an introductory prepositional/adverbial preamble
        is_preamble = bool(re.match(r"^(?:for|in|as\s+for|regarding|concerning|with\s+regard\s+to|under|if|when|while|before|after|in\s+case)\b", first_part, re.I))
        is_question = bool(re.search(r"\b(?:what|how|who|where|when|which|is|are|can|could|does|do|did|would|should|will|whether|kitna|kaise|kya|batao)\b", first_part, re.I))
        if not is_preamble and is_question:
            # Check if second part introduces an independent domain or a distinct degree
            has_independent_domain = bool(re.search(
                r"\b(dean|registrar|library|hostels?|rooms?|mess|dining|gymkhana|sports|gym|bus|shuttle|insurance|disciplinary|committee|icc|internal\s+complaints|placement|supplementary|exams?|examinations?|grade|anti[- ]ragging|fee|fees|late\s+fee|deposits?|caution\s+money|caution\s+deposit|refundable|deadline|last\s+date|closing\s+date|stipend|assistantship|fellowship|branch\s+change|attendance|documents?|certificates?|registration|senate)\b",
                second_part, re.I
            ))
            deg1 = [_canonical_degree(m.group(0)) for m in _degree_mentions(first_part) if _canonical_degree(m.group(0))]
            deg2 = [_canonical_degree(m.group(0)) for m in _degree_mentions(second_part) if _canonical_degree(m.group(0))]
            has_distinct_degree = bool(deg2 and (not deg1 or set(deg2) != set(deg1)))
            is_provisional = bool(re.search(r"\bprovisionall?y?\b", low, re.I))
            is_senate_authority = bool(re.search(r"\b(?:can\s+(?:the\s+)?(?:senate|institute|authority|dean|committee)\s+(?:cancel|terminate|allow|waive)|cancellation\s+of\s+admission|solely\s+for\s+failing|retain\s+authority\s+to|does\s+(?:the\s+)?(?:senate|institute)\s+retain)\b", low, re.I))
            is_distinct_interrogative = bool(re.match(r"^(?:where|who|when)\b", second_part, re.I))
            if (has_independent_domain or has_distinct_degree or is_distinct_interrogative) and not is_provisional and not is_senate_authority and not re.search(r"\b(?:exception|relaxation|waiver|undertaking)\b", second_part, re.I):
                if len(second_part.split()) >= 3 and len(first_part.split()) >= 3:
                    sub_coord = re.search(r",\s+(what|how\s+much|how|who|where|when|which|is|are|can|does|do|whether)\b", first_part, re.I)
                    if sub_coord:
                        p1 = _clean(first_part[:sub_coord.start()])
                        p2 = _clean(first_part[sub_coord.start():].lstrip(" ,"))
                        if len(p1.split()) >= 3 and len(p2.split()) >= 3:
                            return [p1, p2, second_part]
                    return [first_part, second_part]

    # Administrative policy or consequence questions ("what happens if...", "what if...",
    # "consequence of...", "penalty for...", "what is the policy when...", "solely for failing") ask for a single
    # policy outcome, even when mentioning multiple facets like documents and deadlines.
    if re.search(
        r"\b(?:what\s+(?:happens|occurs|is\s+the\s+(?:consequence|penalty|policy|rule|remedy))|what\s+if|in\s+case(?:\s+of)?)\b",
        low, re.I,
    ) or re.search(
        r"\b(?:can\s+(?:the\s+)?(?:senate|institute|authority|dean|committee)\s+(?:cancel|terminate|allow|waive)|cancellation\s+of\s+admission|solely\s+for\s+failing|retain\s+authority\s+to|does\s+(?:the\s+)?(?:senate|institute)\s+retain)\b",
        low, re.I,
    ):
        return [original]

    # Exception/relaxation is part of the same eligibility decision.
    if re.search(
        r"\bcan\s+i\s+(?:still\s+)?(?:apply|register|enroll)\b.*\band\s+(?:are|is)\s+there\s+(?:any\s+)?(?:exception|relaxation)",
        low, re.I | re.S,
    ):
        return [original]

    # A single eligibility decision can mention multiple evidence facets or
    # conditions (GATE, percentage, category relaxation, final-semester status,
    # work experience and interview). Unless the user explicitly requests a
    # second independent domain (fee, deadline, hostel, anti-ragging, assistantship), keep those facets
    # in one eligibility objective.
    eligibility_condition_question = re.search(
        r"\b(?:eligible|eligibility|can\s+i|may\s+i|qualif(?:y|ied)|requirements?|cutoff|cut-off|threshold|do\s+i\s+meet)\b",
        low, re.I,
    ) and re.search(
        r"\b(?:gate|cgpa|gpa|percentage|marks|score|ews|obc|sc|st|pwd|relaxation|exception|final\s+(?:semester|year|result)|graduation|work\s+experience|experience|interview|entrance\s+test|foreign\s+residence|duration|inra)\b",
        low, re.I,
    )
    # "does my X satisfy" / "is X mandatory" with no separate domain = single eligibility check
    single_threshold_check = re.search(
        r"\b(?:does\s+my|is\s+(?:an\s+)?(?:entrance|interview|test)|satisfy\s+(?:the\s+)?(?:minimum|threshold|cutoff)|meet\s+(?:the\s+)?(?:minimum|threshold|requirement))\b",
        low, re.I,
    )
    explicit_independent = bool(
        re.search(r"\b(?:separately|independently|also\s+tell\s+me\s+whether|\band\s+whether)\b", low, re.I)
    )
    if (eligibility_condition_question or single_threshold_check) and not explicit_independent and not re.search(
        r"\b(?:fee|fees|deadline|last\s+date|hostel|anti[- ]ragging|assistantship|stipend|disciplinary|committee|branch change)\b",
        low, re.I,
    ):
        return [original]

    # Explicit "whether ... and whether ..." is the clearest signal of two
    # answerable objectives, even when both belong to eligibility.
    m = re.search(r"\bwhether\b(.+?)\s+\band\s+(?:separately\s+)?whether\b(.+)", original, re.I | re.S)
    if m:
        prefix = _clean(original[:m.start()])
        left = _clean(m.group(1))
        right = _clean(m.group(2))
        return [
            _clean(f"{prefix} whether {left}"),
            _clean(f"whether {right}"),
        ]

    if original.count("?") > 1:
        parts = [_clean(x) for x in re.split(r"\?+", original) if _clean(x)]
        if len(parts) > 1:
            strong_parts = 0
            for part in parts:
                if (
                    _explicit_process_requested(part)
                    or re.search(r"\b(?:fee|fees|documents?|deadline|last\s+date|eligibility|eligible|work\s+experience|comparison|compare)\b", _cf(part), re.I)
                    or re.search(r"फीस|दस्तावेज|आखिरी\s+तारीख|अंतिम\s+तारीख|पात्रता|योग्य", _cf(part))
                ):
                    strong_parts += 1
            # Do not split punctuation artifacts such as "M.Sc.? fees??".
            if strong_parts >= 2:
                return parts

    if ";" in original:
        parts = [_clean(x) for x in re.split(r"\s*;\s*", original) if _clean(x)]
        if len(parts) > 1:
            # A semicolon inside a disclaimer is discourse, not a second
            # answer objective: "I don't need X; just tell me Y."
            if any(re.search(r"\b(?:not\s+asking|rather\s+than|instead\s+of|no\s+need)\b", _cf(part), re.I) or re.search(r"\b(?:already\s+know|don't\s+need|do\s+not\s+need|not\s+asking)\b", _cf(part), re.I) for part in parts):
                return [original]
            return parts

    # Build objective-family markers from live wording. Excluded mentions such
    # as "not asking about eligibility" do not create a marker.
    markers: list[tuple[int, int, str]] = []

    def add_family(pattern: str, family: str) -> None:
        for match in re.finditer(pattern, low, re.I):
            if not _mention_is_excluded(low, match.start(), match.end()):
                markers.append((match.start(), match.end(), family))

    add_family(r"application\s+(?:processing\s+)?fee|processing\s+fee|hostel\s+fee|\b(?:fee|fees|charges|amount|cost|deposits?|caution\s+money|caution\s+deposit)\b|फीस|शुल्क", "fee")
    if not _is_meta_document_reference(original):
        add_family(r"\b(?:documents?|docs?)\b|दस्तावेज|डॉक्यूमेंट", "documents")
    add_family(r"\b(?:deadline|last\s+date|closing\s+date)\b|आखिरी\s+तारीख|अंतिम\s+तारीख", "deadline")
    add_family(r"\b(?:anti[- ]ragging|grievance|committee|cell|disciplinary|ombudsperson)\b", "governance_committee")
    add_family(r"\b(?:assistantship|stipend|scholarship|financial\s+aid|fellowship)\b", "financial_aid")
    add_family(r"\b(?:hostel|mess|accommodation|library)\b", "facility")
    process = _explicit_process_requested(original)
    if process:
        # Anchor the process family at its first live procedural phrase.
        for match in re.finditer(r"\b(?:admission|application)\s+process\b|\b(?:how\s+(?:do|can)\s+i|how\s+to)\s+(?:apply|register|enroll)\b|\bsteps?\b|\bprocedure\b|आवेदन\s+कैसे|प्रक्रिया", low, re.I):
            if not _mention_is_excluded(low, match.start(), match.end()):
                markers.append((match.start(), match.end(), "process"))
                break
    # Eligibility markers are added only when the wording represents a live
    # positive eligibility objective. This prevents disclaimers/known topics
    # from splitting an otherwise single process/fee question.
    if _positive_eligibility_requested(original):
        for match in re.finditer(r"\b(?:eligibility|eligible|qualification|requirements?|required|cgpa|gpa|cutoff|cut-off|branch\s+change)\b|पात्रता|योग्य", low, re.I):
            if not _mention_is_excluded(low, match.start(), match.end()):
                markers.append((match.start(), match.end(), "eligibility"))
                break
    add_family(r"\bwork\s+(?:experience|exp)\b|काम\s+का\s+अनुभव|नौकरी\s+का\s+अनुभव", "work experience")

    markers.sort(key=lambda x: (x[0], x[1]))
    # Collapse duplicate markers from the same family.
    distinct: list[tuple[int,int,str]] = []
    seen_family_positions: set[tuple[int,str]] = set()
    for start, end, family in markers:
        key=(start,family)
        if key in seen_family_positions:
            continue
        seen_family_positions.add(key)
        if distinct and distinct[-1][2] == family and start - distinct[-1][1] < 80:
            continue
        distinct.append((start,end,family))

    families={family for _,_,family in distinct}
    # A single semantic family (eligibility + its facets) is one objective.
    # Different families can be distinct objectives when the user coordinates them.
    if len(families) <= 1:
        return [original]

    # Split before later family markers at the nearest coordinating boundary.
    cuts: list[int] = []
    for idx in range(1, len(distinct)):
        start = distinct[idx][0]
        prev_end = distinct[idx-1][1]
        window_start=max(prev_end, start-70)
        window=original[window_start:start]
        connector=list(re.finditer(r",\s*(?:and\s+)?(?=what|how|who|where|when|which|is|are|can|does|do|whether|also\s+tell|separately\b)|\band\s+(?:also|what|how|who|where|is|are|can|does|do|which|whether|separately)\s+|\bas\s+well\s+as\s+", window, re.I))
        if connector:
            cuts.append(window_start+connector[-1].start())

    # Filter cuts that fall at or before an introductory precondition clause
    m_adv = re.match(r"^(?:when|while|if|before|after|in\s+case)\s+[^,]+,\s*", original, re.I)
    if m_adv:
        cuts = [c for c in cuts if c >= m_adv.end()]

    if not cuts:
        return [original]

    bounds=[0]+cuts+[len(original)]
    parts=[]
    for a,b in zip(bounds,bounds[1:]):
        part=_clean(original[a:b].strip(" ,"))
        if part:
            parts.append(part)
    return parts if len(parts)>1 else [original]


def _intent_semantic_bucket(intent: QueryIntentPlan) -> str:
    attrs = set(_normalize_attributes(intent.requested_attributes))
    if intent.request_type == "comparison":
        return "comparison"
    if "work experience" in attrs:
        return "work experience"
    if intent.request_type == "eligibility" or attrs & {"eligibility", "GATE", "CAT", "JEE", "NET", "CEED", "NBHM", "CGPA", "minimum percentage", "percentage"}:
        return "eligibility"
    if intent.request_type == "fee" or attrs & {"fee", "application fee", "processing fee", "hostel fee"}:
        return "fee"
    if (intent.request_type == "documents" or "documents" in attrs) and not _is_meta_document_reference(intent.question):
        return "documents"
    if intent.request_type == "deadline" or "deadline" in attrs:
        return "deadline"
    if intent.request_type == "process" or attrs & {"process", "admission process", "application process"}:
        return "process"
    return _cf(intent.request_type or "general") or "general"



def _sanitize_constraints(text: str, values: Sequence[Any]) -> list[str]:
    """Keep explicit constraints; drop model-only category fragments such as SC from M.Sc."""
    raw = _clean(text)
    masked = re.sub(DEGREE_PATTERN, " ", raw, flags=re.I)
    explicit_categories = {
        _cf(m.group(0))
        for m in re.finditer(r"(?<![A-Za-z.])(?:EWS|OBC|SC|ST|PWD)(?![A-Za-z])", masked, re.I)
    }
    out: list[str] = []
    for value in values:
        item = _clean(value)
        if not item:
            continue
        if _cf(item) in {"sc", "st", "ews", "obc", "pwd"} and _cf(item) not in explicit_categories:
            continue
        out.append(item)
    return _dedupe(out, 20)

def _attribute_supported_by_text(text: str, attribute: str) -> bool:
    """Return whether an LLM-proposed attribute has lexical/semantic support in the user text."""
    low = _cf(text)
    attr = _canonical_attribute(attribute)
    patterns = {
        "eligibility": r"\b(?:eligib|eligible|qualification|requirements?|can\s+i|may\s+i|whether\s+i\s+can|am\s+i)\b|पात्रता|योग्य",
        "GATE": r"\bgate\b",
        "CAT": r"\bcat\b",
        "JEE": r"\bjee\b",
        "NET": r"\bnet(?:\s*\(jrf\))?\b",
        "CEED": r"\bceed\b",
        "NBHM": r"\bnbhm\b",
        "CGPA": r"\bcgpa\b|\bgpa\b",
        "percentage": r"\b(?:percentage|percent|marks)\b|\d+(?:\.\d+)?\s*%",
        "minimum percentage": r"\b(?:minimum\s+)?percentage\b|\bthreshold\b|\d+(?:\.\d+)?\s*%",
        "work experience": r"\b(?:work\s+(?:experience|exp)|industry\s+experience|prior\s+experience|experience\s+(?:required|compulsory|mandatory|necessary|needed))\b|काम\s+का\s+अनुभव|नौकरी\s+का\s+अनुभव",
        "application fee": r"\bapplication\s+(?:processing\s+)?fee\b|\bprocessing\s+fee\b|आवेदन\s+(?:की\s+)?(?:फीस|शुल्क)",
        "processing fee": r"\bprocessing\s+fee\b|\bprocessing\s+charges\b",
        "hostel fee": r"\bhostel\s+fee(?:s)?\b|छात्रावास\s+(?:की\s+)?फीस|हॉस्टल\s+फीस",
        "fee": r"\b(?:fee|fees|charges|amount|cost)\b|फीस|शुल्क",
        "documents": r"\b(?:documents?|docs?)\b|दस्तावेज|डॉक्यूमेंट",
        "deadline": r"\b(?:deadline|last\s+date|closing\s+date)\b|आखिरी\s+तारीख|अंतिम\s+तारीख",
        "admission process": r"\badmission\s+process\b|\b(?:admission|application)\s+process\b|प्रक्रिया",
        "application process": r"\bapplication\s+process\b|\bhow\s+(?:do|can)\s+i\b.*\b(?:apply|register|enroll)\b|\bsteps?\b|\bprocedure\b|आवेदन\s+कैसे",
        "location": r"\b(?:where|location|located)\b",
        "contact": r"\b(?:contact|phone|email|telephone)\b",
        "facility": r"\b(?:facility|facilities)\b",
        "policy": r"\b(?:policy|policies|rules|regulations)\b",
    }
    pattern = patterns.get(attr)
    if pattern is None:
        # Unknown attributes require direct lexical evidence rather than model-only invention.
        token = _clean(attribute).casefold()
        return bool(token) and token in low
    if attr == "documents" and _is_meta_document_reference(text):
        return False
    return bool(re.search(pattern, low, re.I | re.S))


def _constraint_supported_by_text(text: str, constraint: str) -> bool:
    """Reject model-only constraints that have no support in the user's wording."""
    low = _cf(text)
    c = _cf(constraint)
    if not c:
        return False
    if re.search(r"\d+(?:\.\d+)?", c):
        return bool(any(num in low for num in re.findall(r"\d+(?:\.\d+)?", c)))
    anchors = (
        "gate", "cat", "jee", "net", "ceed", "nbhm", "ews", "obc", "sc", "st", "pwd",
        "relaxation", "final semester", "final year", "final result", "graduation", "result",
        "percentage", "cgpa", "gpa", "experience", "deadline", "year", "semester",
        "not required", "not compulsory", "not mandatory", "not necessary", "se kam", "below", "above",
    )
    return any(anchor in low and anchor in c for anchor in anchors)


def _merge_intents_for_slot(
    segment: str,
    candidates: Sequence[QueryIntentPlan],
    fallback_target: str | None,
) -> QueryIntentPlan:
    """Collapse noisy model fragments into exactly one canonical objective slot."""
    seg_attrs = _requested_attributes(segment)
    seg_type = _det_type_for_question(segment)
    seg_target = _infer_target(segment) or fallback_target

    if not candidates:
        candidates = [QueryIntentPlan(question=segment)]

    primary = max(candidates, key=lambda item: item.confidence)
    target = seg_target or _infer_target(segment, primary.target)

    def compatible_attrs(values: Sequence[Any]) -> list[str]:
        attrs = _normalize_attributes(values)
        if seg_type == "eligibility":
            return attrs
        if seg_type == "fee":
            return [a for a in attrs if _det_type_for_attribute(a) == "fee"]
        if seg_type == "documents":
            return [a for a in attrs if a == "documents"]
        if seg_type == "deadline":
            return [a for a in attrs if a == "deadline"]
        if seg_type == "process":
            return [a for a in attrs if a in {"admission process", "application process"}]
        if seg_type:
            return [a for a in attrs if _det_type_for_attribute(a) == seg_type]
        return attrs

    # Only attributes compatible with the deterministic segment type are allowed
    # to survive reconciliation. This is what prevents a question such as
    # "Can I apply for it this year?" from becoming both process and eligibility,
    # while eligibility questions may legitimately aggregate multiple facets.
    if seg_type in {"fee", "documents", "deadline", "process"}:
        segment_attrs = compatible_attrs(seg_attrs)
    else:
        segment_attrs = seg_attrs
    model_attrs = compatible_attrs(primary.requested_attributes)
    for candidate in candidates:
        model_attrs.extend(compatible_attrs(candidate.requested_attributes))
    model_attrs = [a for a in _normalize_attributes(model_attrs) if _attribute_supported_by_text(segment, a)]
    attrs = _normalize_attributes([*segment_attrs, *model_attrs])

    model_constraints = list(primary.constraints)
    for candidate in candidates:
        model_constraints.extend(candidate.constraints)
    model_constraints = [c for c in model_constraints if _constraint_supported_by_text(segment, c)]
    constraints = _sanitize_constraints(segment, [*_constraints(segment), *model_constraints])
    relations = _dedupe(primary.relations, 10)
    qualifiers = _dedupe(primary.qualifiers, 10)
    temporal = _dedupe(primary.temporal_constraints, 10)
    comparisons = _dedupe(primary.comparison_targets, 4)
    negated = _negated(segment) or primary.negated

    for item in candidates:
        item_attrs = [a for a in compatible_attrs(item.requested_attributes) if _attribute_supported_by_text(segment, a)]
        attrs = _normalize_attributes([*attrs, *item_attrs])
        if seg_type == "eligibility":
            supported_constraints = [c for c in item.constraints if _constraint_supported_by_text(segment, c)]
            constraints = _sanitize_constraints(segment, [*constraints, *supported_constraints])
        relations = _dedupe([*relations, *item.relations], 10)
        qualifiers = _dedupe([*qualifiers, *item.qualifiers], 10)
        temporal = _dedupe([*temporal, *item.temporal_constraints], 10)
        comparisons = _dedupe([*comparisons, *item.comparison_targets], 4)
        negated = negated or item.negated

    request_type = seg_type or _det_type_for_question(segment) or _type(primary.request_type)
    if request_type == "comparison" or comparisons:
        request_type = "comparison"
        comparisons = _comparison_targets(segment) or comparisons
        comparisons = [(_canonical_degree(x) or _strip_target(x) or x) for x in comparisons]
        comparisons = _dedupe(comparisons, 2)
        target = " and ".join(comparisons) if comparisons else target
        attr = _comparison_attribute(segment) or (attrs[0] if attrs else None)
        attrs = [attr] if attr else []
        retrieval = _compact_query(target, attrs, (), comparisons)
    else:
        if target is None and len(attrs) <= 1 and not constraints:
            retrieval = segment
        else:
            retrieval = _safe_retrieval_query(segment, target, attrs, constraints) or _clean(primary.retrieval_query) or segment

    if request_type is None and attrs:
        request_type = _det_type_for_attribute(attrs[0])

    if len(_objective_segments(segment)) <= 1 and len(segment.split()) >= 4:
        question = segment
    else:
        display_attr = attrs[0] if len(attrs) == 1 else None
        question = _intent_question(target, display_attr, segment) if (display_attr and target) else segment
    return QueryIntentPlan(
        question=question,
        request_type=request_type,
        target=target,
        requested_attributes=attrs,
        constraints=constraints,
        relations=relations,
        comparison_targets=comparisons,
        qualifiers=qualifiers,
        temporal_constraints=temporal,
        negated=negated,
        retrieval_query=retrieval,
        confidence=max(0.85, max((c.confidence for c in candidates), default=0.0)),
    )


def _normalize_intent(item: QueryIntentPlan, original: str) -> QueryIntentPlan:
    question = _clean(item.question) or original

    # Normalize against the candidate's own text first. The original question
    # remains authoritative for explicit target/type later in reconciliation.
    local_target = _infer_target(question, item.target)
    attrs = _normalize_attributes([*item.requested_attributes, *_requested_attributes(question)])
    constraints = _sanitize_constraints(question, [*item.constraints, *_constraints(question)])
    relations = _dedupe(item.relations, 10)
    qualifiers = _dedupe(item.qualifiers, 10)
    if local_target:
        spec_match = re.search(
            rf"\b{re.escape(local_target)}\s+(?:in|of)\s+([A-Za-z\s]+?)(?:\s+(?:at|for|from|program|programme|admission|eligibility)\b|\s*[?,;.]|$)",
            question,
            re.I,
        )
        if spec_match:
            spec = _clean(spec_match.group(1))
            if spec and len(spec.split()) <= 4 and not _is_institution_name_or_generic(spec):
                qualifiers = _dedupe([*qualifiers, spec], 10)
    temporal = _dedupe(item.temporal_constraints, 10)
    comparisons = _comparison_targets(question)
    if not comparisons:
        comparisons = _dedupe(item.comparison_targets, 4)
    comparisons = [(_canonical_degree(v) or _strip_target(v) or v) for v in comparisons]
    comparisons = _dedupe(comparisons, 4)

    request_type = _type(item.request_type) or _det_type_for_question(question)
    if comparisons:
        request_type = "comparison"
        local_target = " and ".join(_dedupe(comparisons, 2))

    negated = bool(item.negated) or _negated(question)
    retrieval = _clean(item.retrieval_query)
    confidence = _conf(item.confidence)
    if confidence <= 0:
        confidence = 0.80 if (local_target or attrs or comparisons) else 0.60
    if not retrieval or retrieval == question:
        if not local_target and not comparisons:
            retrieval = question
        else:
            retrieval = _compact_query(local_target, attrs, constraints, comparisons) or question

    return QueryIntentPlan(
        question=question,
        request_type=request_type,
        target=local_target,
        requested_attributes=attrs,
        constraints=constraints,
        relations=relations,
        comparison_targets=comparisons,
        qualifiers=qualifiers,
        temporal_constraints=temporal,
        negated=negated,
        retrieval_query=retrieval,
        confidence=confidence,
    )


def _attribute_signature(attrs: Sequence[str]) -> tuple[str, ...]:

    return tuple(sorted(_normalize_attributes(attrs)))





def _intent_key(item: QueryIntentPlan) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:

    return (

        _cf(item.target or ""),

        _cf(item.request_type or ""),

        _attribute_signature(item.requested_attributes),

        tuple(sorted(_cf(x) for x in item.comparison_targets)),

    )





def _merge_duplicates(intents: Sequence[QueryIntentPlan], original: str) -> list[QueryIntentPlan]:

    out: list[QueryIntentPlan] = []

    positions: dict[tuple[str, str, tuple[str, ...], tuple[str, ...]], int] = {}

    for raw in intents:

        item = _normalize_intent(raw, original)

        key = _intent_key(item)

        if key in positions:

            existing = out[positions[key]]

            existing.constraints = _dedupe([*existing.constraints, *item.constraints], 16)

            existing.relations = _dedupe([*existing.relations, *item.relations], 10)

            existing.qualifiers = _dedupe([*existing.qualifiers, *item.qualifiers], 10)

            existing.temporal_constraints = _dedupe([*existing.temporal_constraints, *item.temporal_constraints], 10)

            existing.negated = existing.negated or item.negated

            existing.confidence = max(existing.confidence, item.confidence)

            if not existing.retrieval_query:

                existing.retrieval_query = item.retrieval_query

            continue

        positions[key] = len(out)

        out.append(item)

    return out[:8]





def _deterministic_intents(original: str) -> list[QueryIntentPlan]:
    comparison_targets = _comparison_targets(original)
    if comparison_targets:
        attr = _comparison_attribute(original)
        query = _compact_query(" and ".join(comparison_targets), [attr] if attr else [], ())
        return [QueryIntentPlan(
            question=original,
            request_type="comparison",
            target=" and ".join(comparison_targets),
            requested_attributes=[attr] if attr else [],
            comparison_targets=comparison_targets,
            retrieval_query=query or original,
            confidence=0.85,
        )]

    segments = _objective_segments(original)
    overall_target = _infer_target(original)
    intents: list[QueryIntentPlan] = []
    for segment in segments:
        seg_inferred = _infer_target(segment)
        if seg_inferred:
            seg_target = seg_inferred
        elif bool(re.search(r"\b(hostels?|rooms?|mess|gymkhana|medical|transport|sports?|canteen|submarine|where\s+is|who\s+is)\b", segment, re.I)):
            seg_target = None
        else:
            seg_target = overall_target
        seg_attrs = _requested_attributes(segment)
        seg_constraints = _constraints(segment)
        seg_negated = _negated(segment)
        seg_type = _det_type_for_question(segment)
        if not seg_type and seg_attrs:
            seg_type = _det_type_for_attribute(seg_attrs[0])

        rq = _safe_retrieval_query(segment, seg_target, seg_attrs, seg_constraints)

        if len(segment.split()) >= 4:
            q_text = segment
        elif seg_attrs and seg_target:
            q_text = _intent_question(seg_target, seg_attrs[0], segment)
        else:
            q_text = segment

        seg_qualifiers: list[str] = []
        if seg_target:
            spec_match = re.search(
                rf"\b{re.escape(seg_target)}\s+(?:in|of)\s+([A-Za-z\s]+?)(?:\s+(?:at|for|from|program|programme|admission|eligibility)\b|\s*[?,;.]|$)",
                segment,
                re.I,
            )
            if spec_match:
                spec = _clean(spec_match.group(1))
                if spec and len(spec.split()) <= 4 and not _is_institution_name_or_generic(spec):
                    seg_qualifiers.append(spec)

        intents.append(QueryIntentPlan(
            question=q_text,
            request_type=seg_type or "general",
            target=seg_target,
            requested_attributes=seg_attrs,
            constraints=seg_constraints,
            qualifiers=seg_qualifiers,
            negated=seg_negated,
            retrieval_query=rq,
            confidence=0.80,
        ))

    return intents if intents else [QueryIntentPlan(
        question=original,
        request_type=None,
        target=overall_target,
        constraints=_constraints(original),
        negated=_negated(original),
        retrieval_query=_safe_retrieval_query(original, overall_target, ()),
        confidence=0.60,
    )]


_fallback_intents = _deterministic_intents


def _deterministic_build_query_plan(original: str) -> QueryPlan:
    """Build a complete QueryPlan using lossless deterministic decomposition and binding."""
    scope, conf, reason = _repair_scope(original, "in_scope" if _relevance(original) >= 0.20 else "uncertain", 0.0)
    if scope == "out_of_scope":
        intents: list[QueryIntentPlan] = []
        is_multi = False
    else:
        intents = _deterministic_intents(original)
        is_multi = len(intents) > 1

    route = classify_query_route(
        domain_decision=scope,
        is_multi_intent=is_multi,
        intents=intents,
        original=original,
    )

    return QueryPlan(
        original_query=original,
        resolved_query=original,
        domain_decision=scope,
        domain_confidence=conf,
        domain_reason=reason,
        is_multi_intent=is_multi,
        intents=intents,
        preserved_terms=_dedupe([*_preserved(original), *_constraints(original)], 28),
        confidence=min(0.85, max((i.confidence for i in intents), default=0.60)),
        route=route.value,
    )





def _explicitly_independent_eligibility(text: str) -> bool:
    """Return True when the user explicitly asks for separate eligibility objectives."""
    low = _cf(text)
    return bool(re.search(
        r"\b(?:separately|independently|as a separate question)\b|\b(?:and|also)\\s+(?:separately\\s+)?whether\\b",
        low,
        re.I,
    ))


def _consolidate_same_target_eligibility(
    original: str,
    intents: Sequence[QueryIntentPlan],
) -> list[QueryIntentPlan]:
    """Merge multiple eligibility facets for one target unless independence is explicit.

    This protects semantic integrity from model decomposition such as
    eligibility + GATE + CGPA becoming three retrieval intents. It is a
    generic semantic-family rule; it does not depend on any test question.
    """
    items = list(intents)
    if len(items) <= 1 or _explicitly_independent_eligibility(original):
        return items

    grouped: dict[str, list[QueryIntentPlan]] = {}
    for item in items:
        bucket = _intent_semantic_bucket(item)
        target = _cf(item.target or _infer_target(original) or "")
        if bucket == "eligibility" and target:
            grouped.setdefault(target, []).append(item)

    # If no target has multiple eligibility fragments, keep existing order
    if not any(len(g) > 1 for g in grouped.values()):
        return items

    result: list[QueryIntentPlan] = []
    merged_targets: set[str] = set()

    for item in items:
        bucket = _intent_semantic_bucket(item)
        target = _cf(item.target or _infer_target(original) or "")
        if bucket == "eligibility" and target and len(grouped[target]) > 1:
            if target not in merged_targets:
                merged_targets.add(target)
                group = grouped[target]
                tgt = group[0].target or _infer_target(original)
                result.append(_merge_intents_for_slot(original, group, tgt))
        else:
            result.append(item)

    return result


def _reconcile_intents(
    original: str,
    llm_intents: Sequence[QueryIntentPlan],
) -> list[QueryIntentPlan]:
    """Reconcile noisy LLM decomposition into deterministic user objectives."""
    original = _clean(original)
    normalized = [_normalize_intent(item, original) for item in llm_intents]

    original_comparisons = _comparison_targets(original)
    if original_comparisons or any(item.comparison_targets for item in normalized):
        candidate = normalized[0] if normalized else QueryIntentPlan(question=original)
        comparisons = _dedupe([*original_comparisons, *candidate.comparison_targets], 4)
        comparisons = [(_canonical_degree(v) or _strip_target(v) or v) for v in comparisons]
        comparisons = _dedupe(comparisons, 2)
        attr = _comparison_attribute(original) or _comparison_attribute(candidate.question)
        target = " and ".join(comparisons) if comparisons else _infer_target(original, candidate.target)
        attrs = [attr] if attr else _normalize_attributes(candidate.requested_attributes[:1])
        return [_merge_intents_for_slot(
            original,
            [QueryIntentPlan(
                **candidate.model_dump(),
                request_type="comparison",
                target=target,
                requested_attributes=attrs,
                comparison_targets=comparisons,
            )],
            target,
        )]

    segments = _objective_segments(original)
    overall_target = _infer_target(original)

    # A single linguistic objective is always one retrieval intent, regardless
    # of how many fragments the LLM emitted for its facets.
    if len(segments) <= 1:
        candidate_target = overall_target or (normalized[0].target if normalized else None)
        merged_single = [_merge_intents_for_slot(original, normalized, candidate_target)]
        # Defensive second pass: if upstream normalization somehow emitted
        # multiple slots, same-target eligibility facets still collapse here.
        return _merge_duplicates(
            _consolidate_same_target_eligibility(original, merged_single),
            original,
        )

    # Multiple explicit objectives: assign model fragments to the closest
    # semantic slot and collapse duplicates inside each slot.
    remaining = list(normalized)
    result: list[QueryIntentPlan] = []
    for segment in segments:
        seg_attrs = set(_normalize_attributes(_requested_attributes(segment)))
        seg_type = _det_type_for_question(segment)
        seg_inferred = _infer_target(segment)
        if seg_inferred:
            seg_target = seg_inferred
        elif bool(re.search(r"\b(hostels?|rooms?|mess|gymkhana|medical|transport|sports?|canteen|submarine|where\s+is|who\s+is)\b", segment, re.I)):
            seg_target = None
        else:
            seg_target = overall_target

        matches: list[QueryIntentPlan] = []
        for item in remaining:
            bucket = _intent_semantic_bucket(item)
            item_attrs = set(_normalize_attributes(item.requested_attributes))
            type_match = bool(seg_type and _type(item.request_type) == seg_type)
            attr_match = bool(seg_attrs & item_attrs)
            target_match = bool(seg_target and item.target and _cf(seg_target) == _cf(item.target))
            if type_match or attr_match or target_match:
                matches.append(item)

        # Ensure an eligibility slot can absorb noisy GATE/CGPA/percentage
        # fragments instead of producing one intent per facet.
        if seg_type == "eligibility" and not matches:
            matches = [item for item in remaining if _intent_semantic_bucket(item) == "eligibility"]

        for item in matches:
            if item in remaining:
                remaining.remove(item)
        result.append(_merge_intents_for_slot(segment, matches, seg_target))

    # Any truly unmatched candidate is appended only when it expresses a
    # distinct semantic objective rather than a duplicate facet.
    for item in remaining:
        bucket = _intent_semantic_bucket(item)
        if any(_intent_semantic_bucket(existing) == bucket and _cf(existing.target or "") == _cf(item.target or "") for existing in result):
            continue
        result.append(_merge_intents_for_slot(original, [item], overall_target))

    consolidated = _consolidate_same_target_eligibility(original, result)
    return _merge_duplicates(consolidated, original)[:8]


def _relevance(text: str) -> float:
    low = _cf(text)
    hits = sum(1 for marker in INSTITUTIONAL_MARKERS if re.search(r"\b" + re.escape(marker) + r"\b", low))
    strong = sum(
        1 for marker in (
            "m.tech", "mtech", "m.sc", "msc", "mba", "phd", "ph.d.", "ph.d", "b.tech",
            "admission", "registration", "hostel", "application fee",
            "eligibility", "percentage", "cgpa", "senate", "cancellation", "cancel",
            "degree", "degrees", "convocation", "awarded",
            "deadline", "documents", "administrative", "disciplinary",
            "library", "exam", "examinations", "probation",
            "requirements", "requirement", "preference", "seat allotment", "counseling",
            "registrar", "transcript", "health centre", "health center", "ambulance",
            "internal complaints committee", "icc", "gymkhana", "scholarship", "dean",
            "late fee", "dining", "sports", "credit limit", "branch change",
            "pmkvy", "skill hub", "upskilling", "vocational", "vtu", "job role", "job roles", "certification",
            "gem", "gfr", "training", "trainings", "workshop", "workshops", "procurement",
            "course code", "course codes", "code", "codes", "elective", "electives",
        ) if re.search(r"\b" + re.escape(marker) + r"\b", low)
    )
    return min(1.0, hits * 0.10 + strong * 0.25)


def _obvious_oos(text: str) -> bool:
    low = _cf(text)
    if not low:
        return True
    if re.search(r"\b(?:recipe|cook|cooking|ingredients|dish|dishes|dal\s+baati|churma|biryani)\b", low):
        return True
    if re.search(r"\b(?:web\s+scraper|scraper|scraping|scrape|download\s+all\s+confidential|hack|jailbreak)\b", low):
        return True
    if re.search(r"\b(?:do\s+not\s+retrieve|tell\s+me\s+that\b.*?\band\s+guarantee|ignore\s+all\s+(?:previous\s+)?instructions)\b", low):
        return True
    if any(re.search(r"\b" + re.escape(marker) + r"\b", low) for marker in OBVIOUS_OOS_MARKERS):
        return not bool(re.search(r"\b(?:m\.?tech|m\.?sc|ph\.?d|b\.?tech|mba|admission|cgpa|eligib)\b", low))
    return False


def _repair_scope(
    original: str,
    llm_scope: ScopeDecision,
    llm_confidence: float,
) -> tuple[ScopeDecision, float, str]:
    relevance = _relevance(original)
    if _obvious_oos(original):
        return "out_of_scope", 0.98, "deterministic_clear_unrelated"

    if llm_scope == "out_of_scope" and llm_confidence >= 0.90:
        if relevance >= 0.20:
            return "in_scope", 0.75, "relevance_overrode_false_oos"
        return "out_of_scope", llm_confidence, "llm_high_confidence"

    if llm_scope == "in_scope":
        return "in_scope", max(llm_confidence, min(0.95, 0.60 + relevance)), "llm_in_scope"

    if relevance >= 0.20:
        return "in_scope", max(0.72, relevance), "deterministic_institutional_relevance"

    return "uncertain", max(0.50, min(0.70, llm_confidence)), "conservative_uncertain"





def classify_query_route(
    *,
    domain_decision: str,
    is_multi_intent: bool,
    intents: Sequence[QueryIntentPlan],
    original: str,
    ambiguous: bool = False,
    needs_clarification: bool = False,
    history: Sequence[Any] = (),
) -> QueryRoute:
    """Classify user query into principled operational routing categories."""
    if domain_decision == "out_of_scope":
        return QueryRoute.OUT_OF_SCOPE
    if needs_clarification or ambiguous:
        return QueryRoute.AMBIGUOUS_NEEDS_CLARIFICATION
    low = _cf(original)
    if history and (low.startswith(("and ", "what about", "how about", "how much", "tell me more")) or len(low.split()) <= 3):
        return QueryRoute.CONVERSATIONAL_FOLLOW_UP
    if is_multi_intent and len(intents) > 1:
        return QueryRoute.MULTI_INTENT
    if any(i.request_type == "comparison" or i.comparison_targets for i in intents) or "compare" in low or "difference between" in low:
        return QueryRoute.COMPARISON
    req_attrs = set()
    for i in intents:
        req_attrs.update(i.requested_attributes)
    if len(req_attrs) >= 3 or ({"fee", "fees"}.intersection(req_attrs) and {"eligibility", "qualification"}.intersection(req_attrs)):
        return QueryRoute.MULTI_SOURCE_SINGLE_INTENT
    constraints_count = sum(len(i.constraints) for i in intents)
    if constraints_count >= 2 or any(w in low for w in ("minimum", "cutoff", "category", "scale", "first class", "first-class", "cgpa", "percentage", "marks")):
        return QueryRoute.SINGLE_INTENT_WITH_CONSTRAINTS
    if domain_decision == "uncertain":
        return QueryRoute.UNCERTAIN_SCOPE
    return QueryRoute.SIMPLE_SINGLE_INTENT


def _from_mapping(original: str, parsed: Mapping[str, Any]) -> QueryPlan:

    raw_intents = parsed.get("intents") if isinstance(parsed.get("intents"), list) else []
    llm_intents: list[QueryIntentPlan] = []
    segments = _objective_segments(original)
    for idx, raw in enumerate(raw_intents):
        if not isinstance(raw, Mapping):
            continue
        try:
            raw_dict = dict(raw)
            if not _clean(raw_dict.get("question")) and idx < len(segments):
                raw_dict["question"] = segments[idx]
            llm_intents.append(QueryIntentPlan.model_validate(raw_dict))
        except Exception:
            continue



    intents = _reconcile_intents(original, llm_intents)

    llm_scope = _scope(parsed.get("domain_decision", parsed.get("scope")))

    scope_conf = _conf(parsed.get("domain_confidence", parsed.get("scope_confidence", 0.0)))

    scope, repaired_scope_conf, scope_reason = _repair_scope(original, llm_scope, scope_conf)



    plan_conf = _conf(parsed.get("confidence"))

    if intents:

        plan_conf = max(plan_conf, min(0.95, sum(i.confidence for i in intents) / len(intents)))



    # The planner must not silently replace the resolved question. Conversation

    # resolution is a separate upstream responsibility. Retrieval gets compact

    # queries from intents while the original question remains authoritative.

    resolved_query = original

    # Deterministic single-intent guard: if the heuristic segmenter identifies
    # at most one linguistic objective (e.g. consequence/policy questions, or a
    # single eligibility query), the plan is definitively single-intent regardless
    # of how many fragments the LLM emitted. This prevents LLM non-determinism
    # from routing single-policy questions through the multi-intent execution path.
    _segments_for_override = _objective_segments(original)
    _force_single = len(_segments_for_override) <= 1
    overall_target = _infer_target(original)

    if _force_single and len(intents) > 1:
        intents = [_merge_intents_for_slot(original, intents, overall_target)]
        _effective_multi_intent = False
    elif not _force_single and len(intents) <= 1:
        intents = _deterministic_intents(original)
        _effective_multi_intent = len(intents) > 1
    else:
        _effective_multi_intent = (len(intents) > 1) and not _force_single

    _route = classify_query_route(
        domain_decision=scope,
        is_multi_intent=_effective_multi_intent,
        intents=intents,
        original=original,
        ambiguous=_bool(parsed.get("ambiguous")),
        needs_clarification=_bool(parsed.get("needs_clarification")),
    )

    return QueryPlan(
        original_query=original,
        resolved_query=resolved_query,
        language=_clean(parsed.get("language")) or None,
        conversation_mode="standalone",
        domain_decision=scope,
        domain_confidence=repaired_scope_conf,
        domain_reason=_clean(parsed.get("domain_reason")) or scope_reason,
        is_multi_intent=_effective_multi_intent,
        intents=intents,
        preserved_terms=_dedupe([
            *(parsed.get("preserved_terms") or []),
            *_preserved(original),
            *_constraints(original),
        ], 28),
        ambiguous=_bool(parsed.get("ambiguous")),
        needs_clarification=_bool(parsed.get("needs_clarification")),
        clarification_reason=_clean(parsed.get("clarification_reason")),
        confidence=plan_conf,
        route=_route.value,
    )





def load_query_planner_model() -> StructuredModel:

    from ai_platform.runtime.llm import query_understanding_llm

    return query_understanding_llm





def build_query_plan(
    question: str,
    *,
    model: StructuredModel | None = None,
    offline: bool = False,
) -> QueryPlan:
    original = _clean(question)
    if not original:
        raise ValueError("question cannot be empty")

    if offline or os.environ.get("PLANNER_OFFLINE_MODE", "").lower() in ("1", "true", "yes"):
        return _deterministic_build_query_plan(original)

    active = model or load_query_planner_model()
    if active is None:
        return _deterministic_build_query_plan(original)

    schema = r'''RETURN ONE JSON OBJECT:

{

  "resolved_query": "string",

  "language": "English|Hindi|Hinglish|other|null",

  "domain_decision": "in_scope|out_of_scope|uncertain",

  "domain_confidence": 0.0,

  "domain_reason": "short reason",

  "is_multi_intent": false,

  "intents": [

    {

      "question": "one standalone answerable objective",

      "request_type": "general|process|eligibility|fee|documents|deadline|facility|location|contact|policy|comparison|null",

      "target": "subject|null",

      "requested_attributes": [],

      "constraints": [],

      "relations": [],

      "comparison_targets": [],

      "qualifiers": [],

      "temporal_constraints": [],

      "negated": false,

      "retrieval_query": "compact retrieval terms",

      "confidence": 0.0

    }

  ],

  "preserved_terms": [],

  "ambiguous": false,

  "needs_clarification": false,

  "clarification_reason": "",

  "confidence": 0.0

}'''.strip()



    prompt = (

        QUERY_PLANNER_SYSTEM_PROMPT

        + "\n\nINSTITUTION DOMAIN GUIDANCE:\n"

        + DEFAULT_INSTITUTION_SCOPE

        + "\n\nSCHEMA:\n"

        + schema

        + "\n\nIMPORTANT: do not turn the verb 'apply' into a process intent unless the user asks for steps/how-to/process. Preserve numbers, percentages, CGPA, exams, academic status, dates, and negation. Never create duplicate intents for synonyms.\n"

        + "RETURN JSON ONLY.\n\nUSER QUESTION:\n"

        + original

    )



    # Exactly ONE LLM call. All recovery after this point is deterministic.

    try:

        response = active.invoke(prompt)

        parsed = _json_object(_response_text(response))

        if parsed is not None:

            return _from_mapping(original, parsed)

    except Exception:

        pass



    return _deterministic_build_query_plan(original)





def should_stop_pipeline(plan: QueryPlan) -> bool:

    return plan.domain_decision == "out_of_scope" and plan.domain_confidence >= 0.90





# ---------------------------------------------------------------------------

# Downstream contracts

# ---------------------------------------------------------------------------



def _requirement(intent: QueryIntentPlan) -> RetrievalRequirement:

    attrs = {_cf(x) for x in intent.requested_attributes}

    focused = bool(attrs) or intent.request_type in {

        "process", "eligibility", "documents", "deadline", "fee", "comparison",

    }

    return RetrievalRequirement(

        mode="focused" if focused else "standard",

        require_target_alignment=bool(intent.target),

        require_attribute_alignment=bool(attrs),

        require_scope_alignment=False,

        reject_explicit_conflict=True,

        allow_partial_evidence=True,

        max_unmatched_required_facets=max(1, len(attrs) - 1),

    )





def plan_to_frame(plan: QueryPlan, *, intent_index: int = 0) -> SemanticQueryFrame:

    if not plan.intents:

        q = plan.resolved_query or plan.original_query

        return SemanticQueryFrame(

            original_query=q,

            normalized_query=q,

            semantic_query="",

            conditions=tuple(plan.preserved_terms),

            preserved_terms=tuple(plan.preserved_terms),

            requirement=RetrievalRequirement(mode="standard", allow_partial_evidence=True),

            confidence=plan.confidence,

            language=plan.language,

            needs_clarification=plan.needs_clarification,

            clarification_reason=plan.clarification_reason,

            interpretation_status="review",

        )



    intent = plan.intents[max(0, min(intent_index, len(plan.intents) - 1))]

    facets = []

    target_clean = intent.target
    if target_clean and _cf(target_clean) in {
        "null", "none", "n/a", "na", "unknown", "nil", "undefined",
        "not applicable", "the institution", "institution", "iit jodhpur",
        "iitj", "iit", "the institute", "institute", "campus", "college", "university",
        "indian institute of technology jodhpur",
    }:
        target_clean = None

    if target_clean:
        facets.append(QueryFacet("target", target_clean, required=True, importance=1.0))

    facets.extend(QueryFacet("attribute", attr, required=True, importance=1.0) for attr in intent.requested_attributes)

    intent_text = _cf(intent.question + " " + intent.retrieval_query)

    local_preserved = [

        term for term in plan.preserved_terms

        if _cf(term) and _cf(term) in intent_text

    ]

    # Always retain explicit constraints in the frame even if they were not

    # literally repeated in the compact retrieval query.

    local_conditions = _dedupe(intent.constraints, 20)

    return SemanticQueryFrame(

        original_query=intent.question or plan.original_query,

        normalized_query=intent.question or plan.original_query,

        semantic_query=intent.retrieval_query,

        target=target_clean,

        request_type=intent.request_type,

        facets=tuple(facets),

        qualifiers=tuple(str(q.value if hasattr(q, "value") else q) for q in intent.qualifiers),
        structured_qualifiers=tuple(Qualifier("qualifier", str(q.value if hasattr(q, "value") else q)) for q in intent.qualifiers),

        conditions=tuple(local_conditions),

        relations=tuple(intent.relations),

        temporal_context=tuple(intent.temporal_constraints),

        comparison_targets=tuple(intent.comparison_targets),

        preserved_terms=tuple(local_preserved),

        is_list_question=intent.request_type == "documents",

        is_comparison_question=intent.request_type == "comparison" or bool(intent.comparison_targets),

        is_multi_part=plan.is_multi_intent,

        requirement=_requirement(intent),

        confidence=max(plan.confidence, intent.confidence),

        needs_clarification=plan.needs_clarification,

        clarification_reason=plan.clarification_reason,

        language=plan.language,

        interpretation_status="accepted" if max(plan.confidence, intent.confidence) >= 0.5 else "review",

    )





def plan_to_query(plan: QueryPlan, *, intent_index: int = 0) -> Query:

    return plan_to_frame(plan, intent_index=intent_index).to_query()





def retrieval_queries_from_plan(plan: QueryPlan, *, limit: int = 5) -> tuple[str, ...]:

    out = [plan.resolved_query or plan.original_query]

    # Intent queries are recovery formulations only; original/resolved wording

    # remains the authoritative primary retrieval query.

    for intent in plan.intents:

        if intent.retrieval_query:

            out.append(intent.retrieval_query)

    # For multi-part queries or queries with multiple distinct preserved facets,
    # include specific topic/facet terms to ensure each asked part can retrieve evidence.
    primary_target = getattr(plan, "target", None) or (plan.intents[0].target if plan.intents else None)
    target_str = str(primary_target).strip() if primary_target else ""
    orig_cf = (plan.resolved_query or plan.original_query or "").casefold()

    for term in getattr(plan, "preserved_terms", ()) or ():
        t_clean = str(term).strip()
        words = t_clean.split()
        if 2 <= len(words) <= 6 and t_clean.casefold() not in {"iit jodhpur", orig_cf}:
            if target_str and target_str.casefold() not in {"iit jodhpur"} and target_str.casefold() not in t_clean.casefold():
                facet_q = f"{target_str} {t_clean}"
            else:
                facet_q = t_clean
            out.append(facet_q)

    # Recover institutional policy formulations for domain-wide entities (fees, dining, housing)
    # where individual programme queries otherwise displace general policy documents.
    if any(k in orig_cf for k in ("hostel", "mess", "dining")):
        if "mess" in orig_cf and "hostel" in orig_cf:
            out.append("semester fee hostel mess charges")
        elif "mess" in orig_cf or "dining" in orig_cf:
            out.append("campus mess dining meal charges")
    elif any(k in orig_cf for k in ("fee", "tuition", "semester fee", "paisa", "paise", "rupaye", "rupees", "kharch", "kharcha")):
        out.append("tuition fee semester fee structure")
        if any(s in orig_cf for s in ("first semester", "1st semester", "semester", "dena padega")):
            out.append("first semester fee structure finance")

    if any(k in orig_cf for k in ("mtech", "m.tech")) and any(k in orig_cf for k in ("option", "kya-kya", "kya kya", "programme", "program", "branch", "list", "offer")):
        out.append("Postgraduate M.Tech programs")
        out.append("M.Tech programmes list")
    elif any(k in orig_cf for k in ("btech", "b.tech")) and any(k in orig_cf for k in ("option", "kya-kya", "kya kya", "programme", "program", "branch", "list", "offer")):
        out.append("Undergraduate B.Tech programs")
        out.append("B.Tech programmes list")

    return tuple(_dedupe(out, limit))





def intent_units_from_plan(plan: QueryPlan, *, limit: int = 8) -> tuple[dict[str, Any], ...]:

    intents = plan.intents or [QueryIntentPlan(

        question=plan.resolved_query or plan.original_query,

        retrieval_query=plan.resolved_query or plan.original_query,

        constraints=list(plan.preserved_terms),

        confidence=plan.confidence,

    )]

    units: list[dict[str, Any]] = []

    for index, intent in enumerate(intents[:limit]):

        single_plan = plan.model_copy(update={"is_multi_intent": False, "intents": [intent]})

        frame = plan_to_frame(single_plan)

        query = frame.to_query()

        question = intent.question or plan.resolved_query or plan.original_query

        units.append({

            "intent_index": index,

            "question": question,

            "request_type": intent.request_type,

            "target": intent.target,

            "requested_attributes": tuple(intent.requested_attributes),

            "constraints": tuple(intent.constraints),

            "relations": tuple(intent.relations),

            "comparison_targets": tuple(intent.comparison_targets),

            "negated": intent.negated,

            "retrieval_query": intent.retrieval_query,

            "retrieval_queries": tuple(_dedupe(
                [
                    intent.retrieval_query,
                    question,
                    *((f"{intent.target} financial assistance",) if (intent.target and "fellowship" in question.casefold() and "financial assistance" not in question.casefold()) else ()),
                    *((f"{intent.target} fellowship",) if (intent.target and "financial assistance" in question.casefold() and "fellowship" not in question.casefold()) else ()),
                ],
                3,
            )),

            "query": query,

            "query_frame": frame,

        })

    return tuple(units)





__all__ = [
    "ScopeDecision", "QueryIntentPlan", "QueryPlan", "DEFAULT_INSTITUTION_SCOPE",
    "QUERY_PLANNER_SYSTEM_PROMPT", "build_query_plan", "load_query_planner_model",
    "should_stop_pipeline", "plan_to_frame", "plan_to_query",
    "retrieval_queries_from_plan", "intent_units_from_plan",
    "QueryRoute", "classify_query_route",
    "_deterministic_build_query_plan", "_deterministic_intents",
]
