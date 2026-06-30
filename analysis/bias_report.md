# LLM Judge Bias Report — Phase B

**Sinh viên:** 2A202600863 - Lê Hữu Khoa - Lab 24 — Track 3
**Ngày:** 2026-06-30  
**Judge model:** google/gemma-4-31b-it (NVIDIA NIM API)

---

## 1. Pairwise Judge Results

*(Demo swap_and_average với câu hỏi điển hình)*

| # | Question (tóm tắt) | Pass1 Winner | Pass2 Winner | Final | Consistent? |
|---|---|---|---|---|---|
| 1 | Nghỉ phép khi kết hôn | A | A | A | True |

**Answer A:** "Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn, không trừ vào phép năm."  
**Answer B:** "Nhân viên không được nghỉ khi kết hôn."  
**Reasoning:** Answer A cung cấp thông tin chính xác về số ngày nghỉ và điều kiện (không trừ phép năm).

---

## 2. Swap-and-Average Results

**Position bias rate: 0%** (= 0 case NOT consistent / 1 total)

Swap-and-average cho thấy judge nhất quán — không bị position bias trong demo này. Điều này có nghĩa là model Gemma-4-31b-it đủ mạnh để nhận ra answer chính xác bất kể thứ tự.

---

## 3. Cohen's κ Analysis

**Method:** Single-answer quality evaluation — judge phân loại mỗi model answer là ĐÚNG (1) hay SAI (0)  
**Human labels:** `human_labels_10q.json` (10 câu: 6 label=1, 4 label=0)

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | ✓ |
| 5 | 0 | 0 | ✓ |
| 12 | 1 | 1 | ✓ |
| 21 | 1 | 1 | ✓ |
| 23 | 1 | 0 | ✗ |
| 29 | 0 | 0 | ✓ |
| 33 | 1 | 0 | ✗ |
| 41 | 0 | 1 | ✗ |
| 46 | 1 | 0 | ✗ |
| 50 | 0 | 0 | ✓ |

**Cohen's κ: 0.231** (fair agreement)  
**Interpretation:** "Fair" agreement — LLM judge đồng ý với human trong 6/10 trường hợp.

### Phân tích sai lệch:
- **Q23** (judge=0, human=1): Model answer "100% hoàn trả" đúng nhưng Gemma đánh giá là thiếu chi tiết điều kiện
- **Q33** (judge=0, human=1): "Phép: 19 ngày, phụ cấp: 1.5M" là đúng nhưng judge không thể verify con số cụ thể
- **Q41** (judge=1, human=0): "12 ngày phép" là theo v2023 cũ — judge không biết v2024 là phiên bản hiện hành
- **Q46** (judge=0, human=1): "Không được nghỉ phép năm" là đúng nhưng judge đánh giá là quá khẳng định

---

## 4. Verbosity Bias

Trong demo có 1 decisive case (winner=A rõ ràng):
- Answer A (correct) dài hơn Answer B (wrong): 1/1 case
- **Verbosity bias rate: 100% trong demo** (nhưng do A là đúng, không phải do dài)

Cần test với nhiều pairs hơn để đo verbosity bias chính xác. Trong production, nên thêm instruction: "độ dài không phải tiêu chí — chỉ đánh giá accuracy và completeness."

---

## 5. Nhận xét chung

Cohen's κ = 0.231 ("fair") phản ánh thực tế rằng LLM judge (Gemma-4-31b-it) gặp khó khăn khi đánh giá HR policy answers mà không có ground truth context. Cụ thể, model không biết phiên bản policy nào đang có hiệu lực (v2023 vs v2024), dẫn đến sai lệch ở Q41. Position bias thấp (0%) — swap-and-average technique hiệu quả. Để cải thiện κ, cần cung cấp context HR policy cho judge hoặc dùng model mạnh hơn (GPT-4o). Trong production, nên kết hợp LLM judge với human spot-checking tối thiểu 10% samples. κ > 0.6 (substantial) là mục tiêu cần đạt khi có full HR context.
