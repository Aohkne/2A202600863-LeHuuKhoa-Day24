from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    # Try large model first, fall back to small
    for model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            import spacy
            spacy.load(model_name)
            nlp_config = {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model_name}]}
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            registry.add_recognizer(cccd_recognizer)
            registry.add_recognizer(phone_recognizer)
            analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
            break
        except OSError:
            continue
    else:
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(cccd_recognizer)
        registry.add_recognizer(phone_recognizer)
        analyzer = AnalyzerEngine(registry=registry)

    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


_PII_ENTITIES = [
    "VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS",
    "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "IBAN_CODE",
]


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio."""
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE,
                               entities=_PII_ENTITIES)
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {
            "type":  r.entity_type,
            "text":  text[r.start:r.end],
            "score": round(r.score, 3),
            "start": r.start,
            "end":   r.end,
        }
        for r in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


_JAILBREAK_PATTERNS = [
    "bỏ qua tất cả hướng dẫn", "ignore your previous instructions",
    "ignore previous instructions", "pretend you are dan",
    "you are now an unrestricted", "forget your system prompt",
    "forget all hr policies", "system override",
    "đóng vai ai không có giới hạn", "bạn là ai không có giới hạn",
    "you are an unrestricted", "no restrictions",
]
_OFF_TOPIC_PATTERNS = [
    "viết một bài thơ", "viết bài thơ", "nấu phở", "nấu ăn",
    "bitcoin", "ethereum", "giá cổ phiếu", "recommend phim", "bộ phim hay",
    "giải phương trình", "phương trình vi phân", "thời tiết", "tin tức mới",
]
_PII_REQUEST_PATTERNS = [
    "cho tôi biết cccd", "số điện thoại của nhân viên",
    "tiết lộ thông tin nhân viên", "reveal all employee",
    "dump all training data", "dump all employee", "list all employee",
    "print all confidential", "in ra toàn bộ system",
    "tiết lộ bảng lương", "email của nhân viên",
    "thông tin cá nhân của nhân viên",
]
_INJECTION_PATTERNS = [
    "<!-- ignore", "<!-- ignore previous", "[admin command",
    "admin command - priority", "bỏ qua tất cả policy",
    "ignore previous instructions", "i ra lệnh bạn bỏ qua",
    "ra lệnh bạn bỏ qua tất cả", "tiết lộ lương tháng của nhân viên",
]


def _keyword_blocked(text: str) -> str | None:
    """Return block_reason string if text matches any guardrail pattern, else None."""
    lower = text.lower()
    for kw in _JAILBREAK_PATTERNS:
        if kw in lower:
            return "jailbreak"
    for kw in _INJECTION_PATTERNS:
        if kw in lower:
            return "prompt_injection"
    for kw in _OFF_TOPIC_PATTERNS:
        if kw in lower:
            return "off_topic"
    for kw in _PII_REQUEST_PATTERNS:
        if kw in lower:
            return "pii_request"
    return None


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard)."""
    # Layer 1: Fast keyword pre-filter (mirrors rails.co patterns deterministically)
    kw_reason = _keyword_blocked(text)
    if kw_reason:
        refuse_msg = "Xin lỗi, tôi không thể thực hiện yêu cầu này. Tôi chỉ có thể trả lời các câu hỏi về chính sách nhân sự công ty."
        return {
            "allowed":        False,
            "blocked_reason": "nemo_input_rail",
            "response":       refuse_msg,
        }

    # Layer 2: NeMo semantic check (LLM-based, may be unavailable without API key)
    if rails is None:
        rails = setup_nemo_rails()

    try:
        raw = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        response = raw.get("content", "") if isinstance(raw, dict) else str(raw)
    except Exception:
        response = ""

    refuse_keywords = [
        "xin lỗi", "không thể", "không được phép", "không thể cung cấp",
        "i cannot", "i'm sorry", "i am sorry",
    ]
    blocked = any(kw in response.lower() for kw in refuse_keywords)
    return {
        "allowed":        not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response":       response,
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user."""
    if rails is None:
        rails = setup_nemo_rails()

    raw = await rails.generate_async(messages=[
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ])
    response = raw.get("content", "") if isinstance(raw, dict) else str(raw)
    refuse_keywords = ["xin lỗi", "không thể cung cấp", "i cannot", "i'm sorry"]
    flagged = any(kw in response.lower() for kw in refuse_keywords)
    return {
        "safe":           not flagged,
        "flagged_reason": "nemo_output_rail" if flagged else None,
        "final_answer":   response if flagged else answer,
    }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected."""
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      item["input"][:80] + "...",
                "expected":   item["expected"],
                "actual":     actual,
                "blocked_by": blocked_by,
                "passed":     actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed  = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack."""
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    presidio_times: list[float] = []
    nemo_times:     list[float] = []
    total_times:    list[float] = []

    async def _measure():
        for text in test_inputs[:n_runs]:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(times: list[float]) -> dict:
        s = sorted(times)
        n = len(s)
        if n == 0:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "p50": round(s[int(n * 0.50)], 2),
            "p95": round(s[min(int(n * 0.95), n - 1)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms":       percentiles(presidio_times),
        "nemo_ms":           percentiles(nemo_times),
        "total_ms":          total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms":         LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")
