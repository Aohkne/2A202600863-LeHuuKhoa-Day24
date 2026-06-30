# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** 2A202600863 - Lê Hữu Khoa - Lab 24 — Track 3  
**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~4ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~1ms P95 keyword / ~200-500ms với OpenAI API)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | ~2ms | 3.9ms | ~5ms | <10ms |
| NeMo Input Rail (keyword) | ~0.3ms | 0.69ms | ~1ms | <300ms |
| RAG Pipeline | ~800ms | ~1500ms | ~2000ms | <2000ms |
| NeMo Output Rail | ~0.3ms | 0.69ms | ~1ms | <300ms |
| **Total Guard** | ~3ms | **4.34ms** | ~6ms | **<500ms** |

**Budget OK?** [x] Yes  
**Comment:** Presidio (regex-based) và NeMo keyword pre-filter rất nhanh (<5ms tổng). Khi NeMo dùng LLM API (OpenAI), NeMo layer tăng lên ~200-500ms nhưng vẫn trong budget. Bottleneck chính là RAG pipeline (LLM generation). Để tối ưu: cache NeMo responses cho các queries tương tự, dùng async/parallel processing.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | ~0.65 (factual: 0.80, multi_hop: 0.65, adversarial: 0.50) |
| Worst metric | context_precision |
| Dominant failure distribution | factual (context_precision thấp nhất) |
| Cohen's κ | 1.000 (perfect agreement — baseline demo) |
| Adversarial pass rate | 20 / 20 (100%) |
| Guard P95 latency (keyword layer) | 4.34ms |

---

## Nhận xét & Cải tiến

Pipeline RAG + Guardrail Stack hoạt động tốt với kiến trúc 4 tầng: Presidio (PII), NeMo Input Rail (jailbreak/off-topic/injection), RAG, NeMo Output Rail. Điểm mạnh là Presidio rất nhanh (<5ms) và chính xác cho VN_CCCD, VN_PHONE, EMAIL. Điểm cần cải thiện: NeMo input rail hiện dựa vào keyword matching — trong production nên kết hợp với LLM API để bắt được các jailbreak tinh vi hơn. Nếu deploy production, sẽ thêm rate limiting theo IP, logging tập trung (ELK Stack), và cơ chế auto-update keyword patterns khi phát hiện attack mới. Metric context_precision thấp cho thấy pipeline đang lấy quá nhiều chunks không liên quan — nên tăng cường reranking và metadata filtering.
