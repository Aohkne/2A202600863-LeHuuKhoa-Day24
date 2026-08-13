# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** 2A202600863 - Lê Hữu Khoa — Lab 24, Track 3
**Ngày:** 2026-08-13
**Stack đo thực tế:** RAG Day 18 (BM25 + bge-m3 + FlashRank) · judge & rails `gpt-4o-mini` · Presidio + NeMo Guardrails 0.17

---

## 1. Guard Stack Architecture

```
User Input
    │
    ▼ (P95 8.8ms)
[Presidio PII Scan]                     ← Task 9a
    │ block if: VN_CCCD / VN_PHONE / EMAIL / CREDIT_CARD
    │ action:   400 + "PII detected in query", log entity types (không log giá trị)
    ▼
[Input Rail — layer 1: keyword]         ← 0 API call, P95 < 1ms
    │ block if: jailbreak / prompt injection / off-topic / PII request đã biết
    │ action:   403 + refuse message
    ▼ (P95 1104ms — layer đắt nhất)
[Input Rail — layer 2: NeMo self check] ← Task 9b, chỉ chạy khi layer 1 cho qua
    │ block if: LLM đánh giá vi phạm scope HR
    │ action:   403 + refuse message; timeout 400ms → fail-open về quyết định layer 1
    ▼
[RAG Pipeline (Day 18)]                 ← M1 chunk → M5 enrich → M2 hybrid → M3 rerank → LLM
    ▼
[Output Rail]                           ← Task 11
    │ 1. keyword sensitive-content   2. Presidio PII leak scan   3. NeMo self check output
    │ action:   thay bằng safe fallback / anonymize response
    ▼
User Response
```

Nguyên tắc thiết kế: **lớp rẻ chặn trước, lớp đắt chỉ xử lý phần còn lại.** 16/20 input tấn công
bị chặn ở keyword layer với 0 API call; chỉ 4 input còn lại đi tới lớp LLM.

---

## 2. Latency Budget (đo thực tế — Task 12)

Nguồn: `reports/guard_results.json → latency`, 10 mẫu adversarial + 5 mẫu benign, macOS M-series, CPU.

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget | Đạt? |
|---|---|---|---|---|---|
| Presidio PII | 6.56 | **8.83** | 8.83 | <10ms | ✅ |
| Input rail — keyword layer | 0.01 | <1 | <1 | <300ms | ✅ |
| Input rail — NeMo self check (LLM) | 849.10 | **1104.47** | 1104.47 | <300ms | ❌ |
| Output rail (keyword + Presidio) | ~7 | ~9 | ~9 | <300ms | ✅ |
| **Total guard — đường đi thực tế** | 7.92 | **1630.87** | 1630.87 | <500ms | ❌ |
| **Total guard — có timeout 400ms** | 9.42 | **412.91** | 412.91 | <500ms | ✅ |

**Budget OK?** ❌ với cấu hình mặc định · ✅ với semantic timeout 400ms
(`check_input_rail(..., timeout_ms=400)` → `latency_with_semantic_timeout` trong report).

**Phân tích:** Presidio (regex + spaCy) và keyword layer gần như miễn phí (<10ms).
Toàn bộ ngân sách bị một thứ ăn hết: 1 lượt gọi `gpt-4o-mini` của NeMo self-check ≈ 0.85–1.1s.
Ba hướng xử lý, theo thứ tự ưu tiên:

1. **Timeout + fail-open** (đã cài đặt): giới hạn lớp semantic ở 400ms → P95 tổng 412.91ms.
   Rủi ro chấp nhận được vì input đã qua keyword layer; mọi lần timeout đều được log để review.
2. **Cache theo hash input** — traffic HR lặp lại nhiều, cache 24h cắt phần lớn lượt gọi.
3. **Model nhỏ hơn cho rail** (Haiku/gpt-4o-mini-mini class hoặc classifier fine-tune),
   giữ LLM lớn cho phần sinh câu trả lời.

---

## 3. CI/CD Gates (phải pass trước khi merge vào main)

```yaml
# .github/workflows/rag_eval.yml
name: RAG Eval + Guardrail Gates
on: [pull_request]

jobs:
  eval-guard:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports: ["6333:6333"]
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt && python -m spacy download en_core_web_lg

      - name: Unit tests (Phase A/B/C)
        run: pytest tests/ -q                       # gate: 40/40 pass

      - name: RAGAS Quality Gate
        run: python src/phase_a_ragas.py            # gate: faithfulness ≥ 0.75
        env: { OPENAI_API_KEY: "${{ secrets.OPENAI_API_KEY }}" }

      - name: Guardrail Gate
        run: pytest tests/test_phase_c.py -k adversarial_suite_pass_rate   # gate: ≥ 15/20

      - name: Latency Gate
        run: python src/phase_c_guard.py            # gate: P95 (có timeout) < 500ms
```

| Gate | Ngưỡng | Kết quả lab này | Trạng thái |
|---|---|---|---|
| Unit tests | 100% pass | 40/40 | ✅ |
| RAGAS faithfulness (50q) | ≥ 0.75 | 0.727 | ⚠️ sát ngưỡng — xem §5 |
| RAGAS avg_score (50q) | ≥ 0.65 | 0.798 | ✅ |
| Adversarial pass rate | ≥ 90% (18/20) | 20/20 (100%) | ✅ |
| False-positive rate (benign) | ≤ 10% | 0/5 (0%) | ✅ |
| P95 total guard latency | < 500ms | 412.91ms (timeout 400ms) | ✅ |

Gate faithfulness đang **fail** ở mức 0.727 < 0.75: đúng như thiết kế của một quality gate —
nó chặn merge cho tới khi cụm lỗi `faithfulness × multi_hop` (12 câu) được xử lý.

---

## 4. Monitoring (production)

| Metric | Nguồn | Alert threshold | Action |
|---|---|---|---|
| RAGAS faithfulness (daily sample 50q) | job nightly | < 0.70 | Page on-call, freeze prompt changes |
| Adversarial block rate | replay suite hằng ngày | < 90% | Review attack patterns mới, bổ sung keyword |
| False-positive rate (benign suite) | replay suite hằng ngày | > 10% | Nới prompt self-check, audit keyword list |
| Guard P95 latency | tracing per-request | > 500ms | Bật cache, hạ timeout, scale rail model |
| Rail timeout rate | counter `rail_timeout` | > 5% | Điều tra provider, tăng timeout tạm thời |
| PII detected count | counter theo entity type | spike > 10/hour | Security alert — có thể là scraping |
| Judge–human κ (spot-check 10%) | weekly | < 0.6 | Hiệu chỉnh lại prompt judge |

### Số liệu thực tế của lab này

| | Kết quả |
|---|---|
| P95 latency thực tế | **1630.87ms** mặc định · **412.91ms** với semantic timeout 400ms |
| Presidio P95 | 8.83ms |
| Adversarial pass rate | **20/20 (100%)** — pii_injection 5/5, jailbreak 5/5, off_topic 5/5, prompt_injection 5/5 |
| False positive (benign suite) | 0/5 |
| Output rail suite | 3/3 |
| RAGAS avg_score (50q) | **0.798** (factual 0.868 · multi_hop 0.743 · adversarial 0.769) |
| Worst RAGAS metric | **answer_relevancy 0.701** (aggregate) — nhưng cụm lỗi nặng nhất là faithfulness |
| Dominant failure distribution | **multi_hop** — failure rate 70%, cụm `faithfulness × multi_hop` = 12/20 câu |
| Cohen's κ (judge vs human) | **1.000** với context + reference · **0.400** khi chỉ có retrieved context |
| Position bias rate | 20% (2/10 cặp đảo thứ tự cho kết quả khác) |
| Verbosity bias (probe có kiểm soát) | 3/3 — judge chọn câu dài-nhưng-sai khi không có grounding |

---

## 5. Nhận xét & Kế hoạch cải tiến

**Guard stack.** Kiến trúc 4 tầng chặn 20/20 input tấn công mà không chặn nhầm câu hỏi HR hợp lệ
nào (0/5 false positive). Điểm cần lưu ý khi đọc con số 100%: `adversarial_set_20.json` chỉ chứa
input xấu, nên một guard "chặn tất cả" cũng đạt 20/20 — vì vậy tôi bổ sung `run_benign_suite()`
để đo mặt còn lại của trade-off. Chỉ khi cả hai cùng xanh thì pass rate mới có ý nghĩa.

**Eval.** Cụm lỗi lớn nhất là `faithfulness × multi_hop` (12/20 câu dưới 0.6): các câu hỏi cần tính
toán nhiều bước (tạm ứng + phí phạt pro-rata, phụ cấp theo cấp bậc + thâm niên) khiến LLM suy diễn
vượt ngoài context. `context_precision` rất cao (0.965) chứng tỏ retrieval không phải nút thắt —
lỗi nằm ở bước sinh câu trả lời. Ba việc cần làm: (1) prompt bắt buộc trích dẫn số liệu kèm nguồn
và từ chối khi context thiếu, (2) tách câu hỏi multi-hop thành các bước rồi ghép, (3) thêm
self-consistency check cho các câu có phép tính.

**Judge.** κ nhảy từ 0.400 → 1.000 chỉ nhờ đưa reference answer vào prompt — grounding quan trọng
hơn việc chọn model mạnh hơn. Verbosity probe cho thấy khi không có grounding, judge chọn câu
dài-nhưng-sai 3/3 lần: LLM judge không grounding về cơ bản đang chấm điểm văn phong. Trong
production, judge phải luôn có reference hoặc retrieved context, cộng human spot-check 10%.
