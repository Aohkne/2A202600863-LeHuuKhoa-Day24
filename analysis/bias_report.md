# LLM Judge Bias Report — Phase B

**Sinh viên:** 2A202600863 - Lê Hữu Khoa — Lab 24, Track 3
**Ngày:** 2026-08-13
**Judge model:** `gpt-4o-mini` (OpenAI API), `temperature=0`, `response_format=json_object`
**Nguồn số liệu:** `reports/judge_results.json` — 10 câu có human label, 20 lượt pairwise (swap-and-average), 3 cặp verbosity probe

---

## 1. Thiết kế thí nghiệm

| Thành phần | Cấu hình |
|---|---|
| Pairwise (Task 5) | A = câu trả lời của model, B = ground truth từ `test_set_50q.json` |
| Swap-and-average (Task 6) | 2 lượt, lượt 2 đảo vị trí rồi map winner về không gian gốc; bất đồng → `tie` |
| Quality eval cho κ (Task 7) | Chạy 2 cấu hình: **(a)** chỉ retrieved context, **(b)** context + reference answer |
| Verbosity (Task 8) | Đo trên cặp thật + một probe có kiểm soát độ dài |

`temperature=0` cho mọi lượt judge: nếu không, κ dao động 0.44 → 0.62 → 0.80 giữa các lần chạy
trên cùng dữ liệu. Judge không deterministic thì không dùng làm CI gate được.

---

## 2. Position Bias (swap-and-average)

**Position bias rate: 20%** — 2/10 cặp cho kết quả khác nhau khi đảo thứ tự.

| # | Question | Pass 1 | Pass 2 (đã map về gốc) | Final | Consistent |
|---|---|---|---|---|---|
| q1 | Nghỉ khi kết hôn | B | B | B | ✅ |
| q5 | Mua thiết bị 55 triệu | B | B | B | ✅ |
| q12 | Thưởng Tết tối thiểu | B | B | B | ✅ |
| **q21** | **Senior 9 năm: phép + lương** | **B** | **tie** | **tie** | ❌ |
| q23 | Hoàn chi đào tạo 25 triệu | B | B | B | ✅ |
| q29 | Tạm ứng 8 triệu quá hạn | B | B | B | ✅ |
| q33 | Manager 12 năm: phụ cấp + phép | B | B | B | ✅ |
| q41 | Ngày phép năm (bẫy v2023) | B | B | B | ✅ |
| **q46** | **Thử việc có phép năm không** | **B** | **A** | **tie** | ❌ |
| q50 | VPN cá nhân khi WFH | B | B | B | ✅ |

Hai ca bất đồng đều là ca hai câu trả lời **cùng đúng**, chỉ khác độ chi tiết — và lý do judge đưa
ra tự tố cáo chính nó:

- **q21** — pass 1: *"Answer B cung cấp thông tin chi tiết hơn"*; pass 2 (sau khi đảo):
  *"Cả hai đều chính xác, nhưng A có phần giải thích chi tiết hơn"*. Cùng một cặp nội dung,
  "chi tiết hơn" được gán cho bất kỳ câu nào đang nằm ở vị trí sau.
- **q46** — pass 1 chọn B, pass 2 chọn A; cả hai lần đều lập luận rằng câu được chọn "đầy đủ hơn".

Swap-and-average bắt được đúng cả hai ca này và hạ chúng xuống `tie` thay vì ghi nhận một winner
giả. Nếu chỉ chạy một lượt, 2/10 kết quả trong report sẽ là nhiễu vị trí.

---

## 3. Cohen's κ — Judge vs Human

| Cấu hình judge | κ | Landis–Koch | Agreement |
|---|---|---|---|
| Chỉ retrieved context | **0.400** | fair | 7/10 |
| Context + reference answer | **1.000** | almost perfect | 10/10 |

| Question ID | Human | Judge (ctx only) | Judge (ctx + ref) | Ghi chú |
|---|---|---|---|---|
| 1 | 1 | 1 ✓ | 1 ✓ | |
| 5 | 0 | 0 ✓ | 0 ✓ | Sai ngưỡng: 55 triệu phải CEO duyệt |
| 12 | 1 | 1 ✓ | 1 ✓ | |
| 21 | 1 | 1 ✓ | 1 ✓ | |
| **23** | 1 | **0 ✗** | 1 ✓ | Context không chứa điều khoản hoàn 100% → judge cho là thiếu căn cứ |
| **29** | 0 | **1 ✗** | 0 ✓ | Judge bỏ sót phần thiếu: cần thêm Kế toán trưởng + phạt pro-rata |
| **33** | 1 | **0 ✗** | 1 ✓ | Không kiểm chứng được phép tính phụ cấp + thâm niên |
| 41 | 0 | 0 ✓ | 0 ✓ | Bẫy v2023 — cả hai đều bắt được |
| 46 | 1 | 1 ✓ | 1 ✓ | |
| 50 | 0 | 0 ✓ | 0 ✓ | VPN cá nhân bị cấm theo v1.3 |

**Nhận xét.** Cả ba ca sai của cấu hình chỉ-có-context đều là ca judge **không kiểm chứng được**:
q23 và q33 bị đánh SAI vì context không đủ để xác nhận con số (dù câu trả lời đúng), q29 bị đánh
ĐÚNG vì judge không biết còn thiếu cấp phê duyệt thứ hai. Nói cách khác, judge không grounding
đang đo *độ tự tin của chính nó*, không phải chất lượng câu trả lời.

Thêm reference answer đưa κ lên 1.000 — grounding tạo khác biệt lớn hơn nhiều so với việc đổi sang
model mạnh hơn. Lưu ý trung thực về con số này: n = 10 nên κ = 1.000 có sai số rất rộng
(một ca sai sẽ kéo κ xuống ~0.79); nó chứng minh judge dùng được, không chứng minh judge hoàn hảo.

---

## 4. Verbosity Bias

### 4.1 Đo trên cặp thật (Task 8)

`verbosity_bias = 1.000` — 8/8 ca có winner rõ ràng đều là B thắng **và** B dài hơn A.

Con số này **không** đọc được như bằng chứng bias: trong thiết kế cặp của lab, B luôn là ground
truth, vốn vừa đúng hơn vừa dài hơn (A trung bình ~48 ký tự, B ~152 ký tự). Chất lượng và độ dài
bị trộn lẫn, nên metric này chỉ nói "câu thắng thường dài hơn", không nói "judge thích câu dài".

### 4.2 Probe có kiểm soát độ dài

Để tách hai yếu tố, tôi dựng 3 cặp mà **câu ngắn là câu đúng, câu dài là câu sai** (dài gấp
~7–13 lần, viết theo văn phong chính sách trôi chảy), rồi chạy swap-and-average để khử position bias:

| Question | Câu ngắn (đúng) | Câu dài (sai) | Judge chọn |
|---|---|---|---|
| Ngày phép năm | "15 ngày (v2024)" | 12 ngày + nhiều chi tiết quy trình | **câu dài (sai)** |
| Mua thiết bị 55 triệu | "CEO duyệt, vượt ngưỡng 50 triệu" | Giám đốc phòng ban + mô tả quy trình dài | **câu dài (sai)** |
| Thử việc có phép năm | "Không, xin nghỉ không lương" | Có, theo tỷ lệ + mô tả HRM | **câu dài (sai)** |

**Verbosity bias thực đo: 3/3 = 100%.**

Đây là kết quả đáng lo nhất của Phase B: khi không có reference, judge chọn câu **sai** cả 3 lần
chỉ vì nó dài và trôi chảy hơn. Kết hợp với §3, kết luận nhất quán — LLM judge không grounding
chấm điểm văn phong chứ không chấm sự thật.

---

## 5. Kết luận & Khuyến nghị Production

| Vấn đề | Bằng chứng | Biện pháp |
|---|---|---|
| Position bias | 2/10 cặp đảo kết quả khi swap | Luôn chạy swap-and-average; bất đồng → `tie`, không lấy winner của lượt 1 |
| Judge không grounding | κ 0.400 vs 1.000; probe 3/3 chọn câu dài-sai | Bắt buộc đưa reference answer hoặc retrieved context vào prompt judge |
| Verbosity bias | Probe 3/3 | Thêm chỉ dẫn "độ dài không phải tiêu chí"; cân nhắc cắt cả hai câu về cùng độ dài trước khi chấm |
| Judge không tái lập | κ dao động 0.44–0.80 khi `temperature` mặc định | `temperature=0` cho mọi lượt judge trong CI |
| n nhỏ | n = 10 → κ có sai số rộng | Mở rộng human label lên ≥ 50 câu; spot-check 10% traffic hằng tuần, alert khi κ < 0.6 |
