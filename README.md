# Lab 24 — Production Eval + Guardrail Stack

Chào mừng các bạn đến với **Day 24**! 🎉

Hôm nay chúng ta sẽ xây dựng lớp **đánh giá (evaluation)** và **bảo vệ (guardrail)** hoàn chỉnh cho RAG pipeline mà các bạn đã xây dựng ở Day 18. Nếu Day 18 là "làm cho RAG chạy được", thì Day 24 là "làm cho RAG đáng tin cậy trong production".

---

## Bức tranh tổng thể

```
[Day 18 Pipeline của bạn]
        │
        ├──► Phase A: RAGAS 50q ──────────► Biết pipeline YẾU ở đâu
        │     (3 distributions)
        │
        ├──► Phase B: LLM-as-Judge ───────► Đo độ tin cậy của eval chính nó
        │     (pairwise + Cohen κ)
        │
        └──► Phase C: NeMo Guardrails ────► Bảo vệ pipeline khỏi inputs độc hại
              (Presidio PII + NeMo rails)
```

Sau lab này, các bạn sẽ có một **complete eval + guardrail stack** có thể deploy thẳng vào production.

---

## Yêu cầu tiên quyết

- Đã hoàn thành **Lab 18** (các module M1–M5 + pipeline.py đã chạy được)
- Python 3.11+
- Docker (chạy Qdrant)
- OpenAI API key

---

## Setup (15 phút — làm TRƯỚC khi tính giờ lab)

### Bước 1: Copy code từ Day 18

```bash
# Copy toàn bộ src/ từ Day 18 của bạn vào đây
cp -r <đường-dẫn-Day18>/src/m*.py src/
cp <đường-dẫn-Day18>/src/pipeline.py src/
```

> **Lưu ý:** M4 (`m4_eval.py`) của bạn cần đã implement xong `evaluate_ragas()` —
> Phase A sẽ dùng hàm này để chạy trên bộ test set 50 câu hỏi mới.

### Bước 2: Cài đặt môi trường

```bash
docker compose up -d                      # Khởi động Qdrant

pip install -r requirements.txt
python -m spacy download en_core_web_lg   # Cần cho Presidio PII detection

cp .env.example .env                      # Điền OPENAI_API_KEY vào .env
```

### Bước 3: Generate answers (quan trọng!)

```bash
python setup_answers.py
```

Script này sẽ chạy Day 18 pipeline của bạn trên **50 câu hỏi** và lưu kết quả vào `answers_50q.json`. Quá trình mất khoảng 5–10 phút. Đây là input cho Phase A.

---

## Cấu trúc thư mục

```
Day24-Track3-Eval-Guard/
│
├── src/
│   ├── m1_chunking.py      ← copy từ Day 18 của bạn
│   ├── m2_search.py        ← copy từ Day 18 của bạn
│   ├── m3_rerank.py        ← copy từ Day 18 của bạn
│   ├── m4_eval.py          ← copy từ Day 18 của bạn (★ cần implement xong)
│   ├── m5_enrichment.py    ← copy từ Day 18 của bạn
│   ├── pipeline.py         ← copy từ Day 18 của bạn
│   │
│   ├── phase_a_ragas.py    ★ BẠN IMPLEMENT — Tasks 1–4
│   ├── phase_b_judge.py    ★ BẠN IMPLEMENT — Tasks 5–8
│   └── phase_c_guard.py    ★ BẠN IMPLEMENT — Tasks 9–12
│
├── guardrails/
│   ├── config.yml          ← NeMo Guardrails config (đã có sẵn)
│   └── rails.co            ← Colang dialogue flows (có thể mở rộng)
│
├── data/                   ← Corpus HR policy (25 tài liệu tiếng Việt)
│
├── test_set_50q.json       ← 50 câu hỏi, 3 distributions
├── human_labels_10q.json   ← 10 nhãn nhân để tính Cohen κ
├── adversarial_set_20.json ← 20 inputs tấn công để test guardrail
│
├── setup_answers.py        ← Chạy pipeline → answers_50q.json
├── check_lab.py            ← Kiểm tra trước khi nộp
│
├── reports/
│   ├── ragas_50q.json      ★ auto-generated (Phase A)
│   ├── judge_results.json  ★ auto-generated (Phase B)
│   ├── guard_results.json  ★ auto-generated (Phase C)
│   └── blueprint.md        ★ Task 13 — bạn điền tay
│
└── analysis/
    ├── failure_clusters.md ★ Phân tích bottom-10 (điền sau Phase A)
    └── bias_report.md      ★ Phân tích bias (điền sau Phase B)
```

---

## Các phases

### Phase A — RAGAS Production Eval (30 phút)

Chạy RAGAS trên **50 câu hỏi** với **3 distributions** để tìm ra điểm yếu của pipeline.

| Distribution | Số câu | Đặc điểm |
|---|---|---|
| `factual` | 20 | Tra cứu chính sách đơn giản, 1 tài liệu |
| `multi_hop` | 20 | Kết hợp nhiều tài liệu, tính toán, suy luận |
| `adversarial` | 10 | Bẫy: version conflicts (v2023 vs v2024), negation traps |

**4 tasks cần implement:** `group_by_distribution()` → `run_ragas_50q()` → `bottom_10()` → `cluster_analysis()`

```bash
python src/phase_a_ragas.py
# Output: reports/ragas_50q.json
```

### Phase B — LLM-as-Judge (30 phút)

Dùng LLM để so sánh cặp answers và đo độ đồng thuận với nhãn của con người.

- **Pairwise judge:** LLM chọn answer A hay B tốt hơn
- **Swap-and-average:** Đổi thứ tự A/B để phát hiện position bias
- **Cohen's κ:** So sánh với 10 nhãn nhân trong `human_labels_10q.json`
- **Bias report:** Đo position bias + verbosity bias

```bash
python src/phase_b_judge.py
# Output: reports/judge_results.json
```

### Phase C — NeMo Guardrails (30 phút)

Xây dựng lớp bảo vệ nhiều tầng trước và sau RAG pipeline.

```
Input → [Presidio PII] → [NeMo Input Rail] → RAG → [NeMo Output Rail] → Response
```

- **Presidio:** Phát hiện CCCD, số điện thoại VN, email trong query
- **NeMo Input Rail:** Chặn jailbreak, off-topic, prompt injection
- **NeMo Output Rail:** Kiểm tra response trước khi trả về user
- **P95 Latency:** Đo latency từng tầng (Presidio ≈ <10ms, NeMo ≈ 200–500ms)

```bash
python src/phase_c_guard.py
# Output: reports/guard_results.json
```

---

## Bộ test set 50 câu hỏi

Bộ test set này được thiết kế đặc biệt để **stress-test** pipeline của bạn:

- **Factual:** Câu hỏi thẳng, nhưng corpus có nhiều phiên bản policy (v2023/v2024, v1/v2) → pipeline cần biết chọn phiên bản đúng
- **Multi-hop:** Tính toán lương thử việc, phí phạt tạm ứng, ngày phép tích lũy → cần kết hợp nhiều tài liệu
- **Adversarial:** Cố tình hỏi về policy cũ, dùng phủ định ("có nên tự xử lý không?"), hỏi VPN cá nhân → đây là những câu pipeline hay nhầm nhất

---

## Chạy tests

```bash
pytest tests/ -v                     # Chạy toàn bộ test suite
pytest tests/test_phase_a.py -v      # Chỉ Phase A
pytest tests/test_phase_b.py -v      # Chỉ Phase B
pytest tests/test_phase_c.py -v      # Chỉ Phase C
```

---

## Kiểm tra trước khi nộp

```bash
python check_lab.py
```

Checklist:
- [ ] Day 18 source files đã copy vào `src/`
- [ ] `answers_50q.json` đã được generate
- [ ] 0 TODOs còn lại trong `src/phase_*.py`
- [ ] `reports/ragas_50q.json`, `judge_results.json`, `guard_results.json` đã có
- [ ] `reports/blueprint.md` đã điền đầy đủ
- [ ] `pytest tests/` pass toàn bộ

---

## Deliverables (push lên GitHub trước khi hết giờ)

1. **`src/phase_a_ragas.py`** — Tasks 1–4 (RAGAS eval)
2. **`src/phase_b_judge.py`** — Tasks 5–8 (LLM Judge)
3. **`src/phase_c_guard.py`** — Tasks 9–12 (Guardrails)
4. **`reports/blueprint.md`** — Task 13 (CI/CD blueprint, điền tay)
5. **`analysis/failure_clusters.md`** — Phân tích Phase A
6. **`analysis/bias_report.md`** — Phân tích Phase B

---

## Điểm số

| Phase | Điểm |
|---|---|
| Phase A: RAGAS (Tasks 1–4) | 30 |
| Phase B: Judge (Tasks 5–8) | 35 |
| Phase C: Guard (Tasks 9–12 + blueprint) | 35 |
| **Tổng** | **100** |
| Bonus (κ>0.6, pass rate≥18/20, adversarial avg<factual avg) | **+10** |

Chi tiết xem thêm: [`RUBRIC.md`](RUBRIC.md)

---

Chúc các bạn code vui! 🚀 Nếu gặp vấn đề, hãy raise tay — mentors luôn sẵn sàng hỗ trợ.

---

# Bài nộp — 2A202600863 Lê Hữu Khoa

**Cấu hình chạy:** RAG Day 18 (BM25 + `BAAI/bge-m3` + FlashRank) · LLM & judge `gpt-4o-mini`
(`temperature=0`) · NeMo Guardrails 0.17 · Presidio + custom VN recognizers.
`.env` cần `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL=gpt-4o-mini`.

## Kết quả

| Phase | Kết quả chính | Bonus |
|---|---|---|
| A — RAGAS 50q | avg_score **0.798** · factual 0.868 / multi_hop 0.743 / adversarial 0.769 · cụm lỗi nặng nhất `faithfulness × multi_hop` (12 câu) | adversarial < factual ✅ |
| B — LLM Judge | κ = **1.000** (context + reference), 0.400 khi chỉ có context · position bias 20% · verbosity probe 3/3 | κ > 0.6 ✅ |
| C — Guardrails | **20/20** adversarial · 0/5 false positive trên benign suite · output rail 3/3 · P95 412.91ms với semantic timeout 400ms | ≥ 18/20 ✅ |

Chi tiết: [`reports/blueprint.md`](reports/blueprint.md) ·
[`analysis/failure_clusters.md`](analysis/failure_clusters.md) ·
[`analysis/bias_report.md`](analysis/bias_report.md)

## Những chỗ đi xa hơn đề bài

- **Phase A** — `run_ragas_50q()` gọi lại `evaluate_ragas()` của Day 18 thay vì gọi thẳng
  `ragas.evaluate()`. Đổi embedding của `answer_relevancy` sang `bge-m3` đa ngữ: với
  `all-MiniLM-L6-v2` (chỉ tiếng Anh) metric này cho ~0.20 kể cả trên câu trả lời đúng.
  `cluster_analysis()` trả về thêm `failure_matrix` (chỉ đếm câu dưới ngưỡng 0.6) và chọn
  dominant distribution theo *tỷ lệ* failure — đếm thô chỉ phản ánh distribution nào nhiều câu hơn.
- **Phase B** — judge chạy ở `temperature=0` để κ tái lập được; đo κ ở hai mức grounding để cho
  thấy reference answer quan trọng thế nào (0.400 → 1.000); thêm `verbosity_probe()` với các cặp
  *câu ngắn đúng vs câu dài sai* để tách verbosity bias khỏi chất lượng.
- **Phase C** — thay input/output rail bằng NeMo `self check input/output` + `guardrails/prompts.yml`
  (flow intent-matching gốc trong `rails.co` không hoạt động ở chế độ input rail và làm câu trả lời
  an toàn bị flag nhầm); giữ keyword layer chạy trước để 16/20 attack bị chặn với 0 API call;
  thêm `run_benign_suite()` đo false positive (adversarial set chỉ có input xấu nên guard
  "chặn tất cả" cũng đạt 20/20); thêm timeout cho lớp semantic để giữ P95 trong budget;
  guard stack degrade an toàn về keyword layer khi không có API key.

## Lệnh chạy lại

```bash
docker compose up -d
python setup_answers.py        # → answers_50q.json (50 câu, ~2 phút)
python src/phase_a_ragas.py    # → reports/ragas_50q.json
python src/phase_b_judge.py    # → reports/judge_results.json
python src/phase_c_guard.py    # → reports/guard_results.json
pytest tests/ -q               # 40 passed
python check_lab.py            # 22/22 checks passed
```
