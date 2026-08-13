# Failure Cluster Analysis — Phase A

**Sinh viên:** 2A202600863 - Lê Hữu Khoa — Lab 24, Track 3
**Ngày:** 2026-08-13
**Nguồn số liệu:** `reports/ragas_50q.json` (RAGAS 4 metrics × 50 câu, judge `gpt-4o-mini`, embeddings `BAAI/bge-m3`)

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) | Tổng (50) |
|---|---|---|---|---|
| faithfulness | 0.933 | **0.460** | 0.850 | 0.727 |
| answer_relevancy | 0.780 | 0.670 | 0.608 | 0.701 |
| context_precision | 0.958 | 0.971 | 0.967 | 0.965 |
| context_recall | 0.800 | 0.871 | **0.650** | 0.798 |
| **avg_score** | **0.868** | **0.743** | **0.769** | **0.798** |

**Ghi chú về embedding model:** lần chạy đầu dùng `all-MiniLM-L6-v2` (chỉ train tiếng Anh) cho
`answer_relevancy` và nhận ~0.20 trên cả những câu trả lời đúng — đó là artifact của phép đo, không
phải lỗi pipeline. Đổi sang `BAAI/bge-m3` (đa ngữ, cùng model với retrieval) thì
`answer_relevancy` lên 0.701. Bài học: metric dựa trên embedding chỉ có nghĩa khi embedding
hiểu được ngôn ngữ của corpus.

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | Manager thâm niên 12 năm: tổng phụ cấp + ngày phép? | 0.375 | faithfulness |
| 2 | adversarial | Manager dùng VPN cá nhân (NordVPN) khi WFH? | 0.417 | faithfulness |
| 3 | factual | Mua thiết bị 55 triệu cần ai phê duyệt? | 0.483 | context_recall |
| 4 | factual | Nam nhân viên nghỉ bao nhiêu ngày khi vợ sinh con? | 0.500 | faithfulness |
| 5 | multi_hop | So sánh yêu cầu mật khẩu v1.0 vs v2.0 | 0.500 | faithfulness |
| 6 | multi_hop | Tạm ứng 8 triệu quá hạn 30 ngày: ai duyệt + phí phạt? | 0.500 | faithfulness |
| 7 | multi_hop | Công tác 2 ngày, khách sạn 1.5 triệu/đêm: hoàn bao nhiêu? | 0.615 | faithfulness |
| 8 | adversarial | Thử việc có được bảo hiểm sức khỏe PVI không? | 0.667 | answer_relevancy |
| 9 | adversarial | Phát hiện malware — có nên tự xử lý không? | 0.667 | answer_relevancy |
| 10 | multi_hop | Tạm ứng 4 triệu và 7 triệu: ai phê duyệt từng khoản? | 0.670 | faithfulness |

7/10 câu tệ nhất có `worst_metric = faithfulness`, và 6/10 là multi_hop.

---

## 3. Failure Cluster Matrix

### 3.1 Ma trận đầy đủ — mọi câu hỏi, đếm theo `worst_metric`

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 2 | **14** | 2 | 18 |
| answer_relevancy | 12 | 4 | 2 | 18 |
| context_recall | 6 | 1 | 6 | 13 |
| context_precision | 0 | 1 | 0 | 1 |

### 3.2 Ma trận failure thực sự — chỉ câu có điểm worst_metric < 0.6

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 1 | **12** | 2 | 15 |
| answer_relevancy | 0 | 0 | 2 | 2 |
| context_recall | 6 | 1 | 1 | 8 |
| context_precision | 0 | 1 | 0 | 1 |

**Vì sao cần hai ma trận:** mỗi câu hỏi luôn đóng góp đúng một ô vào ma trận 3.1, nên tổng theo cột
chỉ phản ánh số câu hỏi trong distribution đó (20/20/10) chứ không nói lên pipeline yếu ở đâu.
Ma trận 3.2 chỉ đếm câu thật sự trượt ngưỡng, và tỷ lệ mới là thứ so sánh được.

| Distribution | Số câu | Câu failure | Failure rate |
|---|---|---|---|
| factual | 20 | 7 | 35% |
| multi_hop | 20 | 14 | **70%** |
| adversarial | 10 | 5 | 50% |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** `multi_hop` — failure rate 70%, avg_score 0.743
**Dominant metric:** `faithfulness` — 15/26 failure
**Cụm nặng nhất:** `faithfulness × multi_hop` = **12 câu** dưới ngưỡng 0.6

**Diễn giải.** `context_precision` đạt 0.965 trên cả ba distribution — retrieval đưa về đúng tài
liệu và gần như không có nhiễu. Vậy lỗi không nằm ở khâu tìm kiếm mà ở khâu sinh câu trả lời:
với câu hỏi nhiều bước (tạm ứng 8 triệu quá hạn → ai duyệt + phí phạt pro-rata; Manager 12 năm →
phụ cấp theo cấp bậc + ngày phép theo thâm niên), LLM phải ghép nhiều mảnh chính sách và tự làm
phép tính, và nó lấp chỗ trống bằng suy diễn — đúng định nghĩa của faithfulness thấp.

Nhóm lỗi thứ hai là `context_recall × factual` (6 câu): các câu hỏi ngưỡng phê duyệt như
"mua thiết bị 55 triệu cần ai duyệt" cần đúng dòng ngưỡng trong `mua_sam.md`; hybrid search kéo về
đúng tài liệu nhưng chunk chứa bảng ngưỡng bị cắt mất — recall ở mức chunk chứ không phải mức tài liệu.

Adversarial có `context_recall` thấp nhất (0.650): các câu version-conflict (v2023 vs v2024) cần
**cả hai** phiên bản trong context để trả lời được "phiên bản nào đang hiệu lực", nhưng retrieval
thường chỉ lấy một.

---

## 5. Suggested Fixes

| Cụm lỗi | Root cause | Fix đề xuất | Ưu tiên |
|---|---|---|---|
| faithfulness × multi_hop (12) | LLM tự suy diễn khi phải ghép nhiều mảnh + tính toán | Prompt bắt buộc trích dẫn số liệu kèm nguồn, từ chối khi context thiếu; tách câu hỏi thành các bước rồi ghép; self-consistency cho câu có phép tính | Cao |
| context_recall × factual (6) | Chunk cắt mất bảng ngưỡng phê duyệt | Chunk theo cấu trúc bảng (giữ nguyên bảng trong 1 chunk), tăng `RERANK_TOP_K` 3 → 5 | Cao |
| context_recall × adversarial (1) + version conflict | Chỉ lấy 1 trong 2 phiên bản policy | Thêm metadata `version`/`effective_date`, luôn kéo về mọi phiên bản của cùng chủ đề rồi để LLM chọn bản hiệu lực | Trung bình |
| answer_relevancy × adversarial (2) | Câu hỏi phủ định ("có nên tự xử lý không?") bị trả lời lệch hướng | Prompt xử lý câu phủ định: nêu rõ nên/không nên trước, giải thích sau | Trung bình |
| context_precision (1) | Gần như không có lỗi | Không cần can thiệp | Thấp |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial có avg_score **0.769**, thấp hơn factual **0.868** — test set phân biệt được độ khó
đúng như kỳ vọng (`adversarial_harder_than_factual: true` trong report).

Điều đáng chú ý là adversarial **không** phải distribution tệ nhất: multi_hop mới là nhóm yếu nhất
(0.743, failure rate 70%). Adversarial giữ được faithfulness 0.850 vì phần lớn là câu hỏi
single-hop có bẫy phiên bản — pipeline trích đúng tài liệu và không bịa; nó chỉ trượt ở
`context_recall` (0.650) khi bẫy yêu cầu đối chiếu hai phiên bản chính sách cùng lúc.

Kết luận cho vòng cải tiến tiếp theo: ưu tiên khâu **generation cho câu hỏi nhiều bước**, không
phải khâu retrieval. Nếu chỉ nhìn avg_score theo distribution thì sẽ đi sai hướng — phải nhìn
ma trận failure theo cặp (metric × distribution) mới thấy đúng chỗ cần sửa.
