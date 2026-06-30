# Failure Cluster Analysis — Phase A

**Sinh viên:** 2A202600863 - Lê Hữu Khoa - Lab 24 — Track 3  
**Ngày:** 2026-06-30

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | ~0.80 | ~0.65 | ~0.50 |
| answer_relevancy | ~0.85 | ~0.70 | ~0.55 |
| context_precision | ~0.75 | ~0.60 | ~0.45 |
| context_recall | ~0.80 | ~0.65 | ~0.50 |
| **avg_score** | **~0.80** | **~0.65** | **~0.50** |

---

## 2. Bottom 10 Questions

Các questions adversarial xuất hiện nhiều nhất trong bottom-10, tiếp theo là multi_hop:

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | adversarial | Nghỉ phép bao nhiêu ngày (v2023)? | ~0.30 | faithfulness |
| 2 | adversarial | Có nên tự xử lý sự cố bảo mật không? | ~0.35 | context_recall |
| 3 | multi_hop | Tính phí phạt tạm ứng quá hạn | ~0.38 | context_precision |
| ... | ... | ... | ... | ... |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 3 | 5 | 5 | 13 |
| answer_relevancy | 4 | 4 | 2 | 10 |
| context_precision | 8 | 6 | 2 | 16 |
| context_recall | 5 | 5 | 1 | 11 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual (vì có 20 questions nên chiếm tổng failure count cao nhất)  
**Dominant metric:** context_precision (16 questions)

**Lý do phân tích:**

Context_precision thấp nghĩa là pipeline đang retrieve quá nhiều chunks không liên quan đến câu hỏi. Với corpus HR policy, nhiều tài liệu có nội dung tương tự (ví dụ: nghi_phep_nam_v2023.md và nghi_phep_nam_v2024.md), khiến BM25 retrieval mang cả hai versions vào context. Adversarial questions bị ảnh hưởng nặng nhất bởi faithfulness vì LLM nhận được context có conflicting information (v2023 vs v2024) và không biết chọn phiên bản nào.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating khi context conflict | Tighten system prompt: "luôn dùng phiên bản mới nhất", lower temperature |
| context_recall | Missing relevant chunks | Tăng top_k, thêm BM25 với Vietnamese tokenizer |
| context_precision | Too many irrelevant chunks | Add metadata filter (sort by date), cải thiện reranking |
| answer_relevancy | Answer không match question intent | Improve prompt template, thêm chain-of-thought |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial distribution có avg_score thấp nhất (~0.50) so với factual (~0.80) và multi_hop (~0.65). Pipeline bị "nhầm" rõ ràng bởi version conflicts — khi hỏi về policy cũ (v2023), pipeline trả lời theo v2023 thay vì v2024 mới nhất. Các câu với negation traps ("có nên tự xử lý không?") cũng gây confusing vì pipeline không nhận ra context phủ định và trả lời sai hướng. Kết luận: cần thêm explicit version metadata trong retrieval và cải thiện handling của negation patterns trong system prompt.
